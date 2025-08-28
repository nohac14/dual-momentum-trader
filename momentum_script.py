"""
Dual Momentum — Tuned (Month-End, Ensemble, Smoothed Vol Targeting)
+ Macro-aware defensive bond routing (Deflation Crash vs Inflation Downturn)

Adds a macro classifier at the last completed month-end using ETF proxies:
- Duration tilt:   TLT vs BND
- Breakevens:      TIP vs IEF
- Commodities:     DBC
Routes Risk-OFF bond sleeve to one of: TLT (deflation), BND (inflation), IEF (neutral).
"""

import os
import sys
import datetime as dt
import numpy as np
import pandas as pd
import pytz
import yfinance as yf
import smtplib
from email.message import EmailMessage
from pandas.tseries.offsets import MonthEnd

# --------- CONFIG ---------
US_EQ   = "VOO"
INTL_EQ = "VXUS"
# Base defensive components (GOLD fixed; BOND is decided dynamically by classifier)
BOND_DEF_NEUTRAL   = "IEF"   # 7-10Y Treasuries
BOND_DEF_INFLATION = "BND"   # Agg (inflation/downturn default)
BOND_DEF_DEFLATION = "TLT"   # 20+Y Treasuries (crash hedge)
GOLD    = "IAUM"             # Falls back to IAU automatically
CASH    = "BIL"              # T-Bills (risk-on sink)

# Proxies required by the classifier
CLASSIFIER_TICKERS = ["TLT", "BND", "TIP", "IEF", "DBC"]

# Ensemble of relative momentum lookback sets (months)
REL_LOOKBACKS_ENSEMBLE = [[6, 12], [9, 12], [6, 9, 12]]

# Absolute gate variants to ensemble over: "abs12" (12m TR > 0) and "ma10" (~10-month SMA via 200d)
ABS_GATES = ["abs12", "ma10"]

# Defensive split when Risk-OFF: (BONDS weight, GOLD weight)
DEFENSIVE_SPLIT = (0.50, 0.50)

# Vol targeting parameters
TARGET_VOL  = 0.15   # 15% annualized target for the equity sleeve
VOL_WINDOW  = 63     # ~3 months of daily returns
EQUITY_FLOOR = 0.30  # floor on equity sleeve during Risk-ON
EQUITY_CAP   = 1.00  # cap on equity sleeve during Risk-ON

# Macro classifier parameters
MACRO_LB_SHORT = 6     # months for short lookback
MACRO_LB_LONG  = 12    # months for long lookback
COMMO_STRONG_THRESHOLD = 0.05   # +5% 6m on DBC counts as “strongly positive” inflation pressure

# Data history back to this many months for robustness
HISTORY_MONTHS = 240  # 20 years
# --------------------------

NY_TZ = pytz.timezone("America/New_York")

# ---------- Helpers ----------
def _next_business_day(d: dt.date) -> dt.date:
    wd = d.weekday()
    if wd >= 4:  # Fri/Sat -> next Monday
        return d + dt.timedelta(days=7 - wd)
    return d + dt.timedelta(days=1)

def _last_completed_month_end(asof_date: dt.date) -> dt.date:
    ts = pd.Timestamp(asof_date)
    return (ts - MonthEnd(1)).normalize().date()

def _safe_adj_close(all_tickers, start, end):
    """Download Adj Close for tickers; handle IAUM fallback to IAU history."""
    dl_tickers = list(dict.fromkeys(all_tickers))  # de-dup while preserving order
    if GOLD == "IAUM" and "IAU" not in dl_tickers:
        dl_tickers.append("IAU")

    data = yf.download(dl_tickers, start=start, end=end, auto_adjust=False, progress=False)
    if data.empty:
        raise RuntimeError("Failed to download market data.")
    if "Adj Close" not in data:
        raise RuntimeError("Unexpected data format from yfinance.")
    adj = data["Adj Close"].copy()

    # GOLD fallback: IAUM <- IAU where IAUM is NaN/short
    if GOLD == "IAUM" and "IAU" in adj.columns:
        if "IAUM" in adj.columns:
            adj["IAUM"] = adj["IAUM"].combine_first(adj["IAU"])
        else:
            adj["IAUM"] = adj["IAU"]
    return adj

def _roll_months_back(date: pd.Timestamp, months: int) -> pd.Timestamp:
    return pd.Timestamp(date) - pd.DateOffset(months=months)

def _monthly_signal_date(adj_close: pd.DataFrame, asof: dt.date) -> pd.Timestamp:
    last_completed_eom = pd.Timestamp(_last_completed_month_end(asof))
    valid_idx = adj_close.index[adj_close.index <= last_completed_eom]
    if len(valid_idx) == 0:
        raise RuntimeError("No price data up to the last completed month-end.")
    return valid_idx[-1]

def _get_rel_momentum_score(prices: pd.Series, asof_ts: pd.Timestamp, lookbacks: list[int]) -> tuple[float, list]:
    comps = []
    s = prices.dropna()
    if s.empty:
        return 0.0, []
    last_px = float(s.loc[:asof_ts].iloc[-1])
    for m in lookbacks:
        base_ts = s.index[s.index.searchsorted(_roll_months_back(asof_ts, m))]
        if base_ts > asof_ts:
            base_ts = s.index[max(0, s.index.get_loc(asof_ts) - 1)]
        base_px = float(s.loc[:asof_ts].iloc[s.index.get_loc(base_ts)])
        if base_px > 0:
            comps.append((m, (last_px / base_px) - 1.0, base_ts, base_px, last_px))
    score = np.mean([r for _, r, *_ in comps]) if comps else 0.0
    return score, comps

def _abs_gate(prices: pd.Series, asof_ts: pd.Timestamp, gate: str) -> dict:
    s = prices.dropna()
    out = {"passed": False, "detail": {}}
    if len(s) == 0 or asof_ts not in s.index:
        return out

    px = float(s.loc[asof_ts])

    if gate == "abs12":
        base_ts = s.index[s.index.searchsorted(_roll_months_back(asof_ts, 12))]
        if base_ts > asof_ts:
            base_ts = s.index[max(0, s.index.get_loc(asof_ts) - 1)]
        base_px = float(s.loc[:asof_ts].iloc[s.index.get_loc(base_ts)])
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

def _vol_target(prices: pd.Series, asof_ts: pd.Timestamp) -> tuple[float, float]:
    s = prices.dropna().loc[:asof_ts]
    rets = s.pct_change()
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
    """
    Returns:
      {
        'regime': 'deflation' | 'inflation' | 'neutral',
        'chosen_bond': ticker,
        'signals': { ... detailed metrics ... }
      }
    """
    need = ["TLT","BND","TIP","IEF","DBC"]
    for t in need:
        if t not in adj_close.columns:
            raise RuntimeError(f"Missing {t} in classifier data.")

    # 6m and 12m total returns
    def tr(tkr, m): return _total_return(adj_close[tkr], asof_ts, m)

    tlt6,  tlt12  = tr("TLT", MACRO_LB_SHORT),  tr("TLT", MACRO_LB_LONG)
    bnd6,  bnd12  = tr("BND", MACRO_LB_SHORT),  tr("BND", MACRO_LB_LONG)
    tip6,  tip12  = tr("TIP", MACRO_LB_SHORT),  tr("TIP", MACRO_LB_LONG)
    ief6,  ief12  = tr("IEF", MACRO_LB_SHORT),  tr("IEF", MACRO_LB_LONG)
    dbc6,  dbc12  = tr("DBC", MACRO_LB_SHORT),  tr("DBC", MACRO_LB_LONG)

    # Signals
    duration_pref = ((tlt6 + tlt12)/2) - ((bnd6 + bnd12)/2)   # >0 means prefer long duration
    breakeven_drop = ((tip6 + tip12)/2) - ((ief6 + ief12)/2)  # <0 means falling breakevens (disinflation/deflation)
    commodities_trend = dbc6  # short window for “pressure”; also keep 12m for context

    # Voting logic
    deflation_votes = 0
    inflation_votes = 0

    if duration_pref is not np.nan and duration_pref > 0:
        deflation_votes += 1
    else:
        inflation_votes += 1  # long duration lagging favors inflation/ rising rates

    if breakeven_drop is not np.nan and breakeven_drop < 0:
        deflation_votes += 1
    else:
        inflation_votes += 1

    if not np.isnan(commodities_trend):
        if commodities_trend <= 0.0:
            deflation_votes += 1
        elif commodities_trend >= COMMO_STRONG_THRESHOLD:
            inflation_votes += 1
        # small positive -> no vote (ties handled by others)

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
            "tlt6": tlt6, "tlt12": tlt12,
            "bnd6": bnd6, "bnd12": bnd12,
            "tip6": tip6, "tip12": tip12,
            "ief6": ief6, "ief12": ief12,
            "dbc6": dbc6, "dbc12": dbc12,
            "duration_pref": duration_pref,
            "breakeven_drop": breakeven_drop,
            "commodities_trend_6m": commodities_trend,
            "votes": {"deflation": deflation_votes, "inflation": inflation_votes}
        }
    }

# ---------- Core Logic ----------
def decide_allocation(asof: dt.date | None = None) -> dict:
    """Compute month-end signals using an ensemble and return final allocation."""
    if asof is None:
        asof = dt.datetime.now(NY_TZ).date()

    # Data window
    start_date = (pd.Timestamp(asof) - pd.DateOffset(months=HISTORY_MONTHS)).date()
    end_date   = asof + dt.timedelta(days=2)  # small pad

    # Tickers needed for momentum + classifier
    base_tickers = [US_EQ, INTL_EQ, GOLD, CASH,
                    BOND_DEF_NEUTRAL, BOND_DEF_INFLATION, BOND_DEF_DEFLATION] + CLASSIFIER_TICKERS
    adj_close = _safe_adj_close(base_tickers, start=start_date, end=end_date)

    # Month-end signal date & next trade date
    signal_ts = _monthly_signal_date(adj_close, asof)
    trade_date = _next_business_day(signal_ts.date())

    # --- Macro classifier decides which bond ETF is our Risk-OFF bond today ---
    macro = _classify_macro(adj_close, signal_ts)
    selected_bond = macro["chosen_bond"]

    # Winner selection + gates + vol targeting for each ensemble member
    per_model_results = []
    for lbs in REL_LOOKBACKS_ENSEMBLE:
        us_score, us_comps     = _get_rel_momentum_score(adj_close[US_EQ], signal_ts, lbs)
        intl_score, intl_comps = _get_rel_momentum_score(adj_close[INTL_EQ], signal_ts, lbs)

        if us_score >= intl_score:
            winner, winner_score, winner_comps = US_EQ, us_score, us_comps
        else:
            winner, winner_score, winner_comps = INTL_EQ, intl_score, intl_comps

        for gate in ABS_GATES:
            abs_check = _abs_gate(adj_close[winner], signal_ts, gate)
            recent_vol, eq_w = _vol_target(adj_close[winner], signal_ts)

            if abs_check["passed"]:
                regime = f"RISK-ON → {winner}"
                alloc = {US_EQ: 0.0, INTL_EQ: 0.0, selected_bond: 0.0, GOLD: 0.0, CASH: 1.0 - eq_w}
                alloc[winner] = eq_w
            else:
                regime = "RISK-OFF → Defensive"
                alloc = {US_EQ: 0.0, INTL_EQ: 0.0, selected_bond: DEFENSIVE_SPLIT[0], GOLD: DEFENSIVE_SPLIT[1], CASH: 0.0}

            per_model_results.append({
                "lbs": lbs,
                "gate": gate,
                "winner": winner,
                "winner_score": winner_score,
                "winner_comps": winner_comps,
                "abs_check": abs_check,
                "recent_vol": recent_vol,
                "eq_weight": eq_w,
                "regime": regime,
                "alloc": alloc
            })

    # Average allocations across ensemble
    # Ensure keys cover all display assets (even if zero)
    all_keys = {US_EQ, INTL_EQ, GOLD, CASH, selected_bond}
    final_alloc = {k: 0.0 for k in all_keys}
    for r in per_model_results:
        for k, v in r["alloc"].items():
            final_alloc[k] = final_alloc.get(k, 0.0) + v
    n = len(per_model_results)
    for k in final_alloc:
        final_alloc[k] /= n

    # Determine final regime text (majority vote on regimes)
    risk_on_votes = sum(1 for r in per_model_results if r["regime"].startswith("RISK-ON"))
    regime = "RISK-ON (ensemble majority)" if risk_on_votes >= (n/2) else "RISK-OFF (ensemble majority)"

    return {
        "as_of_date": str(asof),
        "signal_date": str(signal_ts.date()),
        "trade_date": str(trade_date),
        "regime": regime,
        "allocation": final_alloc,
        "selected_bond": selected_bond,
        "macro_regime": macro["regime"],
        "macro_signals": macro["signals"],
        "ensemble_details": per_model_results,
        "config": {
            "rel_lookbacks": REL_LOOKBACKS_ENSEMBLE,
            "abs_gates": ABS_GATES,
            "target_vol": TARGET_VOL,
            "vol_window_days": VOL_WINDOW,
            "equity_floor": EQUITY_FLOOR,
            "equity_cap": EQUITY_CAP,
            "defensive_split": DEFENSIVE_SPLIT,
            "classifier_lookbacks": {"short_m": MACRO_LB_SHORT, "long_m": MACRO_LB_LONG},
            "commodities_strong_thresh": COMMO_STRONG_THRESHOLD,
            "bond_map": {
                "deflation": BOND_DEF_DEFLATION,
                "inflation": BOND_DEF_INFLATION,
                "neutral":   BOND_DEF_NEUTRAL,
            }
        }
    }

# ---------- Email (updated to show bond choice) ----------
def format_results_for_email(out: dict) -> tuple[str, str]:
    """Minimal email: headline + trade date + clean allocation table."""
    is_risk_on = out['regime'].startswith("RISK-ON")
    status_color = "#28a745" if is_risk_on else "#fd7e14"
    subject = f"Dual Momentum Signal ({out['signal_date']}): {out['regime']}"

    # Build labels WITH tickers (so don't add ({t}) again in the row)
    tick_order = [US_EQ, INTL_EQ, out["selected_bond"], GOLD, CASH]
    labels = {
        US_EQ: f"US ({US_EQ})",
        INTL_EQ: f"Intl ({INTL_EQ})",
        out["selected_bond"]: f"Bonds ({out['selected_bond']})",
        GOLD: f"Gold ({GOLD})",
        CASH: f"Cash/T-Bills ({CASH})",
    }

    # Allocation table (no duplicate ticker printing)
    alloc_rows = ""
    for t in tick_order:
        w = out["allocation"].get(t, 0.0)
        strong = ' style="font-weight:bold;"' if (w > 0 and t in [US_EQ, INTL_EQ]) else ""
        alloc_rows += (
            f'<tr{strong}>'
            f'<td><b>{labels[t]}</b></td>'
            f'<td style="text-align:right;">{w:.2%}</td>'
            f'</tr>'
        )

    # Minimal HTML
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
    <body style="margin:0; padding:0; font-family:-apple-system, BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background-color:#f7f7f8;">
      <table role="presentation" style="width:100%; max-width:600px; margin:20px auto; background:#fff; border-collapse:collapse; border-radius:10px; box-shadow:0 4px 10px rgba(0,0,0,.06);">
        <tr>
          <td style="padding:20px; text-align:center; background:#1f2937; color:#fff; border-radius:10px 10px 0 0;">
            <h2 style="margin:0;">Dual Momentum</h2>
            <p style="margin:6px 0 0; color:#ffffff;">Signal Date: {out['signal_date']} &nbsp;•&nbsp; Trade on: <b>{out['trade_date']}</b></p>
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
            This is an automated monthly notification.
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

        print("=== Dual Momentum (Tuned, Month-End, Ensemble) ===")
        print(f"As of (NY): {out['as_of_date']}")
        print(f"Signal date (EOM): {out['signal_date']} | Trade next business day: {out['trade_date']}")
        print(f"Macro regime: {out['macro_regime']} | Defensive bond: {out['selected_bond']}")
        ms = out["macro_signals"]
        print(f"  duration_pref (TLT−BND avg 6/12m): {ms['duration_pref']:+.2%}")
        print(f"  breakeven_drop (TIP−IEF avg 6/12m): {ms['breakeven_drop']:+.2%}")
        print(f"  commodities 6m (DBC): {ms['commodities_trend_6m']:+.2%}")
        print(f"Regime: {out['regime']}\n")

        print("Target allocation:")
        for k, v in out["allocation"].items():
            print(f"  {k}: {v*100:.2f}%")

        # Email
        email_subject, email_body = format_results_for_email(out)
        send_email(email_subject, email_body)

    except Exception as e:
        print(f"Error computing allocation: {e}", file=sys.stderr)
        try:
            send_email("Dual Momentum Script FAILED", f"<p>The script failed with error: {e}</p>")
        except Exception:
            pass
        sys.exit(1)

if __name__ == "__main__":
    main()




