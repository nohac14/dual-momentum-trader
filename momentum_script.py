#!/usr/bin/env python3
"""
Dual Momentum — Tuned (Month-End, Ensemble, Smoothed Vol Targeting)
+ Macro-aware defensive bond routing (Deflation Crash vs Inflation Downturn)

This version is updated to match the advanced logic of the console script:
- Month-end signals; ensemble across relative lookbacks & absolute gates
- Chooses strongest performer from multiple US & International equity options
- 63-day vol targeting with floor/cap; residual to CASH in Risk-ON
- Macro classifier (at month-end) picks defensive bond among TLT/BND/IEF
- Sends a clean HTML email notification with the target allocation
"""

import os
import sys
import datetime as dt
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import pytz
import yfinance as yf
import smtplib
from email.message import EmailMessage
from pandas.tseries.offsets import MonthEnd

# --------- CONFIG ---------
US_EQ_SPY = "SPMO"
US_EQ_QQQ = "QQQM"
US_EQ_OPTIONS = [US_EQ_SPY, US_EQ_QQQ]  # Strategy picks the stronger of these two
INTL_EQ_OPTIONS = ["IQLT", "DIVI"]
GOLD      = "IAUM"  # Falls back to IAU for history
CASH      = "BIL"   # Risk-ON residual

# Defensive bond candidates chosen by the macro classifier:
BOND_DEF_NEUTRAL   = "IEF"   # neutral (7–10Y UST)
BOND_DEF_INFLATION = "BND"   # inflation downturn (Agg)
BOND_DEF_DEFLATION = "TLT"   # deflation crash (20+Y UST)

# Macro classifier proxies:
CLASSIFIER_TICKERS = ["TLT", "BND", "TIP", "IEF", "DBC"]

# Ensemble of relative momentum lookbacks (months):
REL_LOOKBACKS_ENSEMBLE = [[6, 12], [9, 12], [6, 9, 12]]

# Absolute gates to ensemble over:
ABS_GATES = ["abs12", "ma10"]  # 12m TR > 0 OR price > ~10-mo SMA (200d)

# Defensive split when Risk-OFF: (BONDS weight, GOLD weight)
DEFENSIVE_SPLIT = (0.50, 0.50)

# Vol targeting parameters
TARGET_VOL   = 0.15  # 15% annualized target for the equity sleeve
VOL_WINDOW   = 63    # ~3 months of daily returns
EQUITY_FLOOR = 0.30  # floor on equity sleeve during Risk-ON
EQUITY_CAP   = 1.00  # cap on equity sleeve during Risk-ON

# Macro classifier parameters
MACRO_LB_SHORT = 6
MACRO_LB_LONG  = 12
COMMO_STRONG_THRESHOLD = 0.05  # +5% on DBC (6m) indicates strong inflation pressure

# Data history window for robustness
HISTORY_MONTHS = 240  # 20 years

# Timezone
NY_TZ = pytz.timezone("America/New_York")
# ----------------------------------------------------------

# ---------- Helpers ----------
def _next_business_day(d: dt.date) -> dt.date:
    wd = d.weekday()  # Mon=0
    if wd >= 4:  # Fri/Sat -> next Monday
        return d + dt.timedelta(days=7 - wd)
    return d + dt.timedelta(days=1)

def _last_completed_month_end(asof_date: dt.date) -> dt.date:
    ts = pd.Timestamp(asof_date)
    return (ts - MonthEnd(1)).normalize().date()

def _safe_adj_close(tickers: List[str], start, end) -> pd.DataFrame:
    # de-dup while preserving order
    dl = list(dict.fromkeys(tickers))
    # ensure IAU for GOLD fallback if using IAUM
    if GOLD == "IAUM" and "IAU" not in dl:
        dl.append("IAU")

    data = yf.download(dl, start=start, end=end, auto_adjust=False, progress=False)
    if data.empty or "Adj Close" not in data:
        raise RuntimeError("Failed to download market data (Adj Close).")
    adj = data["Adj Close"].copy()

    # GOLD fallback IAUM <- IAU
    if GOLD == "IAUM" and "IAU" in adj.columns:
        if "IAUM" in adj.columns:
            adj["IAUM"] = adj["IAUM"].combine_first(adj["IAU"])
        else:
            adj["IAUM"] = adj["IAU"]
    return adj

def _roll_months_back(date: pd.Timestamp, months: int) -> pd.Timestamp:
    return pd.Timestamp(date) - pd.DateOffset(months=months)

def _monthly_signal_date(adj_close: pd.DataFrame, asof: dt.date) -> pd.Timestamp:
    eom = pd.Timestamp(_last_completed_month_end(asof))
    idx = adj_close.index[adj_close.index <= eom]
    if len(idx) == 0:
        raise RuntimeError("No price data up to last completed month-end.")
    return idx[-1]

def _get_rel_momentum_score(prices: pd.Series, asof_ts: pd.Timestamp, lookbacks: List[int]) -> Tuple[float, List[Tuple]]:
    s = prices.dropna()
    if s.empty:
        return 0.0, []
    last_px = float(s.loc[:asof_ts].iloc[-1])
    comps = []
    for m in lookbacks:
        target = _roll_months_back(asof_ts, m)
        idx = s.index.searchsorted(target)
        idx = min(idx, s.index.get_loc(asof_ts))
        base_px = float(s.iloc[idx])
        comps.append((m, (last_px / base_px) - 1.0 if base_px > 0 else np.nan,
                          s.index[idx], base_px, last_px))
    returns = [r for _, r, *_ in comps if pd.notna(r)]
    score = float(np.mean(returns)) if returns else 0.0
    return score, comps

def _abs_gate(prices: pd.Series, asof_ts: pd.Timestamp, gate: str) -> Dict[str, Any]:
    s = prices.dropna()
    out = {"passed": False, "detail": {}}
    if s.empty or asof_ts not in s.index:
        return out
    px = float(s.loc[asof_ts])
    if gate == "abs12":
        base_idx = s.index.searchsorted(_roll_months_back(asof_ts, 12))
        base_idx = min(base_idx, s.index.get_loc(asof_ts))
        base_px = float(s.iloc[base_idx])
        abs12 = (px / base_px) - 1.0 if base_px > 0 else -1.0
        out["passed"] = abs12 > 0.0
        out["detail"] = {"px": px, "base_px": base_px, "abs12": abs12}
    elif gate == "ma10":
        sma = s.loc[:asof_ts].rolling(window=200).mean().iloc[-1]
        out["passed"] = px > float(sma) if pd.notna(sma) else False
        out["detail"] = {"px": px, "sma200": float(sma) if pd.notna(sma) else np.nan}
    else:
        raise ValueError(f"Unknown abs gate: {gate}")
    return out

def _vol_target(prices: pd.Series, asof_ts: pd.Timestamp) -> Tuple[float, float]:
    s = prices.dropna().loc[:asof_ts]
    rets = s.pct_change()
    if len(rets) < 5:
        return 0.0, EQUITY_CAP
    recent = rets.iloc[-VOL_WINDOW:].std() * np.sqrt(252) if len(rets) >= VOL_WINDOW else rets.std() * np.sqrt(252)
    if pd.isna(recent) or recent <= 0:
        return 0.0, EQUITY_CAP
    raw = TARGET_VOL / float(recent)
    w = float(np.clip(raw, EQUITY_FLOOR, EQUITY_CAP))
    return float(recent), w

# ---------- Macro Classifier ----------
def _total_return(s: pd.Series, asof_ts: pd.Timestamp, months: int) -> float:
    s = s.dropna()
    if s.empty or asof_ts not in s.index:
        return np.nan
    end_px = float(s.loc[asof_ts])
    base_idx = s.index.searchsorted(_roll_months_back(asof_ts, months))
    base_idx = min(base_idx, s.index.get_loc(asof_ts))
    start_px = float(s.iloc[base_idx])
    return (end_px / start_px) - 1.0 if start_px > 0 else np.nan

def _classify_macro(adj_close: pd.DataFrame, asof_ts: pd.Timestamp) -> dict:
    need = ["TLT","BND","TIP","IEF","DBC"]
    for t in need:
        if t not in adj_close.columns:
            raise RuntimeError(f"Missing {t} in classifier data.")

    def tr(tkr, m): return _total_return(adj_close[tkr], asof_ts, m)

    tlt6, tlt12 = tr("TLT", MACRO_LB_SHORT), tr("TLT", MACRO_LB_LONG)
    bnd6, bnd12 = tr("BND", MACRO_LB_SHORT), tr("BND", MACRO_LB_LONG)
    tip6, tip12 = tr("TIP", MACRO_LB_SHORT), tr("TIP", MACRO_LB_LONG)
    ief6, ief12 = tr("IEF", MACRO_LB_SHORT), tr("IEF", MACRO_LB_LONG)
    dbc6, dbc12 = tr("DBC", MACRO_LB_SHORT), tr("DBC", MACRO_LB_LONG)

    duration_pref  = ((tlt6 + tlt12)/2) - ((bnd6 + bnd12)/2)  # >0 → favor long duration (deflationary vibe)
    breakeven_drop = ((tip6 + tip12)/2) - ((ief6 + ief12)/2)  # <0 → falling breakevens
    commodities_6m = dbc6

    deflation_votes = 0
    inflation_votes = 0

    if not np.isnan(duration_pref) and duration_pref > 0:
        deflation_votes += 1
    else:
        inflation_votes += 1

    if not np.isnan(breakeven_drop) and breakeven_drop < 0:
        deflation_votes += 1
    else:
        inflation_votes += 1

    if not np.isnan(commodities_6m):
        if commodities_6m <= 0.0:
            deflation_votes += 1
        elif commodities_6m >= COMMO_STRONG_THRESHOLD:
            inflation_votes += 1
        # small positive → no vote

    if deflation_votes >= 2:
        regime = "deflation"
        chosen = BOND_DEF_DEFLATION
    elif inflation_votes >= 2:
        regime = "inflation"
        chosen = BOND_DEF_INFLATION
    else:
        regime = "neutral"
        chosen = BOND_DEF_NEUTRAL

    return {
        "regime": regime,
        "chosen_bond": chosen,
        "signals": {
            "tlt6": tlt6, "tlt12": tlt12, "bnd6": bnd6, "bnd12": bnd12,
            "tip6": tip6, "tip12": tip12, "ief6": ief6, "ief12": ief12,
            "dbc6": dbc6, "dbc12": dbc12, "duration_pref": duration_pref,
            "breakeven_drop": breakeven_drop, "commodities_trend_6m": commodities_6m,
            "votes": {"deflation": deflation_votes, "inflation": inflation_votes}
        }
    }

# ---------- Core Logic ----------
def decide_allocation(asof: dt.date | None = None) -> Dict[str, Any]:
    """Compute month-end signals using an ensemble and return final allocation."""
    if asof is None:
        asof = dt.datetime.now(NY_TZ).date()

    start_date = (pd.Timestamp(asof) - pd.DateOffset(months=HISTORY_MONTHS)).date()
    end_date   = asof + dt.timedelta(days=2)

    # Tickers needed for momentum + classifier
    tickers = [
        *US_EQ_OPTIONS, *INTL_EQ_OPTIONS, GOLD, CASH,
        BOND_DEF_NEUTRAL, BOND_DEF_INFLATION, BOND_DEF_DEFLATION,
        *CLASSIFIER_TICKERS
    ]
    adj_close = _safe_adj_close(tickers, start=start_date, end=end_date)

    signal_ts = _monthly_signal_date(adj_close, asof)
    trade_date = _next_business_day(signal_ts.date())

    # --- Macro classifier decides which bond ETF is our Risk-OFF bond today ---
    macro = _classify_macro(adj_close, signal_ts)
    selected_bond = macro["chosen_bond"]

    # Winner selection + gates + vol targeting for each ensemble member
    per_model_results = []
    for lbs in REL_LOOKBACKS_ENSEMBLE:
        # Calculate scores for all U.S. equity options and pick a winner
        us_scores = {
            t: _get_rel_momentum_score(adj_close[t], signal_ts, lbs)
            for t in US_EQ_OPTIONS
        }
        us_winner_ticker = max(us_scores, key=lambda t: us_scores[t][0])
        us_winner_score, us_winner_comps = us_scores[us_winner_ticker]

        # Calculate scores for all international options and pick a winner
        intl_scores = {
            t: _get_rel_momentum_score(adj_close[t], signal_ts, lbs)
            for t in INTL_EQ_OPTIONS
        }
        intl_winner_ticker = max(intl_scores, key=lambda t: intl_scores[t][0])
        intl_winner_score, intl_winner_comps = intl_scores[intl_winner_ticker]

        # Compare U.S. winner vs International winner for final risk-on asset
        if us_winner_score >= intl_winner_score:
            winner, winner_score, winner_comps = us_winner_ticker, us_winner_score, us_winner_comps
        else:
            winner, winner_score, winner_comps = intl_winner_ticker, intl_winner_score, intl_winner_comps

        for gate in ABS_GATES:
            abs_check = _abs_gate(adj_close[winner], signal_ts, gate)
            recent_vol, eq_w = _vol_target(adj_close[winner], signal_ts)

            # Initialize allocation dict with all possible equity assets
            alloc = {t: 0.0 for t in US_EQ_OPTIONS}
            for t in INTL_EQ_OPTIONS:
                alloc[t] = 0.0

            if abs_check["passed"]:
                regime = f"RISK-ON → {winner}"
                alloc.update({selected_bond: 0.0, GOLD: 0.0, CASH: 1.0 - eq_w})
                alloc[winner] = eq_w
            else:
                regime = "RISK-OFF → Defensive"
                alloc.update({selected_bond: DEFENSIVE_SPLIT[0], GOLD: DEFENSIVE_SPLIT[1], CASH: 0.0})

            per_model_results.append({
                "lbs": lbs, "gate": gate, "winner": winner, "winner_score": winner_score,
                "winner_comps": winner_comps, "abs_check": abs_check, "recent_vol": recent_vol,
                "eq_weight": eq_w, "regime": regime, "alloc": alloc
            })

    # Average allocations across ensemble
    keys = [*US_EQ_OPTIONS, *INTL_EQ_OPTIONS, selected_bond, GOLD, CASH]
    final_alloc = {k: 0.0 for k in keys}
    for r in per_model_results:
        for k, v in r["alloc"].items():
            final_alloc[k] = final_alloc.get(k, 0.0) + v
    n = len(per_model_results)
    for k in final_alloc:
        final_alloc[k] /= n

    # Determine final regime text (majority vote)
    risk_on_votes = sum(1 for r in per_model_results if r["regime"].startswith("RISK-ON"))
    regime = "RISK-ON (ensemble majority)" if risk_on_votes >= (n/2) else "RISK-OFF (ensemble majority)"

    return {
        "as_of_date": str(asof), "signal_date": str(signal_ts.date()),
        "trade_date": str(trade_date), "regime": regime, "allocation": final_alloc,
        "selected_bond": selected_bond, "macro_regime": macro["regime"],
        "macro_signals": macro["signals"], "ensemble_details": per_model_results
    }

# ---------- Email Formatting ----------
def format_results_for_email(out: dict) -> tuple[str, str]:
    is_risk_on = out['regime'].startswith("RISK-ON")
    status_color = "#28a745" if is_risk_on else "#fd7e14"
    subject = f"Dual Momentum Signal ({out['signal_date']}): {out['regime']}"

    # Define order and labels for all possible assets
    tick_order = [*US_EQ_OPTIONS, *INTL_EQ_OPTIONS, out["selected_bond"], GOLD, CASH]
    labels = {
        US_EQ_SPY: f"US S&P 500 ({US_EQ_SPY})",
        US_EQ_QQQ: f"US Nasdaq 100 ({US_EQ_QQQ})",
        "IQLT": f"Intl Quality ({'IQLT'})",
        "DIVI": f"Intl Dividends ({'DIVI'})",
        out["selected_bond"]: f"Bonds ({out['selected_bond']})",
        GOLD: f"Gold ({GOLD})",
        CASH: f"Cash/T-Bills ({CASH})",
    }

    alloc_rows = ""
    for t in tick_order:
        w = out["allocation"].get(t, 0.0)
        # Skip rows that are guaranteed to be zero to keep the email clean
        if w < 0.0001:
            continue
        
        is_equity_winner = (t in US_EQ_OPTIONS or t in INTL_EQ_OPTIONS) and w > 0
        strong = ' style="font-weight:bold;"' if is_equity_winner else ""
        alloc_rows += (
            f'<tr{strong}>'
            f'<td><b>{labels.get(t, t)}</b></td>'
            f'<td style="text-align:right;">{w:.2%}</td>'
            f'</tr>'
        )

    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
    <body style="margin:0; padding:0; font-family:-apple-system, BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background-color:#f7f7f8;">
      <table role="presentation" style="width:100%; max-width:600px; margin:20px auto; background:#fff; border-collapse:collapse; border-radius:10px; box-shadow:0 4px 10px rgba(0,0,0,.06);">
        <tr>
          <td style="padding:20px; text-align:center; background:#1f2937; color:#fff; border-radius:10px 10px 0 0;">
            <h2 style="margin:0;">Dual Momentum Signal</h2>
            <p style="margin:6px 0 0; color:#ffffff;">Signal Date: <b>{out['signal_date']}</b></p>
            <p style="margin:4px 0 0; color:#ffffff;">Trade on: <b>{out['trade_date']}</b></p>
          </td>
        </tr>
        <tr>
          <td style="padding:22px;">
            <div style="padding:14px 16px; background:{status_color}; color:#fff; text-align:center; border-radius:8px;">
              <strong style="font-size:18px;">{out['regime']}</strong>
            </div>
            <h3 style="border-bottom:2px solid #e5e7eb; padding-bottom:6px; margin-top:26px; margin-bottom:12px;">Target Allocation</h3>
            <table role="presentation" style="width:100%; border-collapse:separate; border-spacing:0 6px;">
              {alloc_rows}
            </table>
          </td>
        </tr>
        <tr>
          <td style="padding:14px 20px 20px; text-align:center; color:#6b7280; font-size:12px;">
            This is an automated monthly notification. Macro regime: {out['macro_regime']}.
          </td>
        </tr>
      </table>
    </body>
    </html>
    """
    return subject, html_body

def send_email(subject: str, html_body: str):
    sender = os.environ.get('GMAIL_ADDRESS')
    password = os.environ.get('GMAIL_APP_PASSWORD')
    recipient = os.environ.get('EMAIL_RECIPIENT')

    if not all([sender, password, recipient]):
        print("ERROR: One or more secrets (GMAIL_ADDRESS, GMAIL_APP_PASSWORD, EMAIL_RECIPIENT) not found.", file=sys.stderr)
        return

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = recipient
    msg.set_content("Please enable HTML to view this message.")
    msg.add_alternative(html_body, subtype='html')

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender, password)
            smtp.send_message(msg)
        print("Email notification sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}", file=sys.stderr)

def main():
    try:
        out = decide_allocation()

        print("=== Dual Momentum (Tuned, Month-End, Ensemble) — Email Version ===")
        print(f"As of (NY): {out['as_of_date']}")
        print(f"Signal date (EOM): {out['signal_date']} | Trade next business day: {out['trade_date']}")
        print(f"Macro regime: {out['macro_regime']} | Defensive bond: {out['selected_bond']}")
        print(f"Ensemble regime: {out['regime']}\n")

        print("Final Target allocation:")
        for k, v in sorted(out["allocation"].items()):
            if v > 1e-6:
                print(f"  {k}: {v*100:.2f}%")

        # Email
        email_subject, email_body = format_results_for_email(out)
        send_email(email_subject, email_body)

    except Exception as e:
        print(f"FATAL ERROR: {e}", file=sys.stderr)
        try:
            send_email("Dual Momentum Script FAILED", f"<h3>The script failed with a fatal error:</h3><p>{e}</p>")
        except Exception as email_err:
            print(f"Could not send failure notification email: {email_err}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
