#!/usr/bin/env python3
"""
Dual Momentum (Tuned) — Console Edition + Macro-Aware Defensive Bond
- Month-end signals; ensemble across relative lookbacks & absolute gates
- 63-day vol targeting with floor/cap; residual to CASH in Risk-ON
- Macro classifier (at month-end) picks defensive bond among TLT/BND/IEF
- Rich console output (ANSI colors), CSV export, deep-dive inspection

Usage:
  python dual_momentum_console_macro.py
  python dual_momentum_console_macro.py --asof 2025-08-28 --details 12 --deep 3 --export-csv ensemble.csv
  python dual_momentum_console_macro.py --no-color
"""

import argparse
import datetime as dt
import sys
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import pytz
import yfinance as yf
from pandas.tseries.offsets import MonthEnd

# --------- CONFIG ---------
US_EQ   = "VOO"
INTL_EQ = "VXUS"
GOLD    = "IAUM"   # Falls back to IAU for history
CASH    = "BIL"    # Risk-ON residual

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

# ---------- ANSI Colors ----------
class C:
    def __init__(self, enable=True):
        self.enable = enable
        self.reset   = "" if not enable else "\x1b[0m"
        self.dim     = "" if not enable else "\x1b[2m"
        self.bold    = "" if not enable else "\x1b[1m"
        self.green   = "" if not enable else "\x1b[32m"
        self.red     = "" if not enable else "\x1b[31m"
        self.yellow  = "" if not enable else "\x1b[33m"
        self.cyan    = "" if not enable else "\x1b[36m"
        self.blue    = "" if not enable else "\x1b[34m"
        self.magenta = "" if not enable else "\x1b[35m"

# ---------- Date helpers ----------
def next_business_day(d: dt.date) -> dt.date:
    wd = d.weekday()  # Mon=0
    if wd >= 4:  # Fri/Sat -> next Monday
        return d + dt.timedelta(days=7 - wd)
    return d + dt.timedelta(days=1)

def last_completed_month_end(asof_date: dt.date) -> dt.date:
    ts = pd.Timestamp(asof_date)
    return (ts - MonthEnd(1)).normalize().date()

# ---------- Data ----------
def safe_adj_close(tickers: List[str], start, end) -> pd.DataFrame:
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

def monthly_signal_date(adj_close: pd.DataFrame, asof: dt.date) -> pd.Timestamp:
    eom = pd.Timestamp(last_completed_month_end(asof))
    idx = adj_close.index[adj_close.index <= eom]
    if len(idx) == 0:
        raise RuntimeError("No price data up to last completed month-end.")
    return idx[-1]

# ---------- Momentum & gates ----------
def roll_months_back(date: pd.Timestamp, months: int) -> pd.Timestamp:
    return pd.Timestamp(date) - pd.DateOffset(months=months)

def relative_momentum_score(prices: pd.Series, asof_ts: pd.Timestamp, lookbacks: List[int]) -> Tuple[float, List[Tuple]]:
    s = prices.dropna()
    if s.empty:
        return 0.0, []
    last_px = float(s.loc[:asof_ts].iloc[-1])
    comps = []
    for m in lookbacks:
        target = roll_months_back(asof_ts, m)
        idx = s.index.searchsorted(target)
        idx = min(idx, s.index.get_loc(asof_ts))
        base_px = float(s.iloc[idx])
        comps.append((m, (last_px / base_px) - 1.0 if base_px > 0 else np.nan,
                      s.index[idx], base_px, last_px))
    returns = [r for _, r, *_ in comps if pd.notna(r)]
    score = float(np.mean(returns)) if returns else 0.0
    return score, comps

def absolute_gate(prices: pd.Series, asof_ts: pd.Timestamp, gate: str) -> Dict[str, Any]:
    s = prices.dropna()
    out = {"passed": False, "detail": {}}
    if s.empty or asof_ts not in s.index:
        return out
    px = float(s.loc[asof_ts])
    if gate == "abs12":
        base_idx = s.index.searchsorted(roll_months_back(asof_ts, 12))
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

def vol_target(prices: pd.Series, asof_ts: pd.Timestamp) -> Tuple[float, float]:
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
def total_return(s: pd.Series, asof_ts: pd.Timestamp, months: int) -> float:
    s = s.dropna()
    if s.empty or asof_ts not in s.index:
        return np.nan
    end_px = float(s.loc[asof_ts])
    base_idx = s.index.searchsorted(roll_months_back(asof_ts, months))
    base_idx = min(base_idx, s.index.get_loc(asof_ts))
    start_px = float(s.iloc[base_idx])
    return (end_px / start_px) - 1.0 if start_px > 0 else np.nan

def classify_macro(adj_close: pd.DataFrame, asof_ts: pd.Timestamp) -> dict:
    need = ["TLT","BND","TIP","IEF","DBC"]
    for t in need:
        if t not in adj_close.columns:
            raise RuntimeError(f"Missing {t} in classifier data.")

    def tr(tkr, m): return total_return(adj_close[tkr], asof_ts, m)

    tlt6, tlt12 = tr("TLT", MACRO_LB_SHORT), tr("TLT", MACRO_LB_LONG)
    bnd6, bnd12 = tr("BND", MACRO_LB_SHORT), tr("BND", MACRO_LB_LONG)
    tip6, tip12 = tr("TIP", MACRO_LB_SHORT), tr("TIP", MACRO_LB_LONG)
    ief6, ief12 = tr("IEF", MACRO_LB_SHORT), tr("IEF", MACRO_LB_LONG)
    dbc6, dbc12 = tr("DBC", MACRO_LB_SHORT), tr("DBC", MACRO_LB_LONG)

    duration_pref   = ((tlt6 + tlt12)/2) - ((bnd6 + bnd12)/2)   # >0 → favor long duration (deflationary vibe)
    breakeven_drop  = ((tip6 + tip12)/2) - ((ief6 + ief12)/2)   # <0 → falling breakevens
    commodities_6m  = dbc6

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
            "tlt6": tlt6, "tlt12": tlt12,
            "bnd6": bnd6, "bnd12": bnd12,
            "tip6": tip6, "tip12": tip12,
            "ief6": ief6, "ief12": ief12,
            "dbc6": dbc6, "dbc12": dbc12,
            "duration_pref": duration_pref,
            "breakeven_drop": breakeven_drop,
            "commodities_trend_6m": commodities_6m,
            "votes": {"deflation": deflation_votes, "inflation": inflation_votes}
        }
    }

# ---------- Core allocation ----------
def decide_allocation(asof: dt.date | None = None) -> Dict[str, Any]:
    if asof is None:
        asof = dt.datetime.now(NY_TZ).date()

    start_date = (pd.Timestamp(asof) - pd.DateOffset(months=HISTORY_MONTHS)).date()
    end_date   = asof + dt.timedelta(days=2)

    # Tickers needed for momentum + classifier (dedup handled in safe_adj_close)
    tickers = [
        US_EQ, INTL_EQ, GOLD, CASH,
        BOND_DEF_NEUTRAL, BOND_DEF_INFLATION, BOND_DEF_DEFLATION,
        *CLASSIFIER_TICKERS
    ]
    adj_close = safe_adj_close(tickers, start=start_date, end=end_date)

    signal_ts = monthly_signal_date(adj_close, asof)
    trade_date = next_business_day(signal_ts.date())

    # Macro decides which bond ETF is used in Risk-OFF
    macro = classify_macro(adj_close, signal_ts)
    selected_bond = macro["chosen_bond"]

    per_model = []
    for lbs in REL_LOOKBACKS_ENSEMBLE:
        us_score, us_comps = relative_momentum_score(adj_close[US_EQ], signal_ts, lbs)
        intl_score, intl_comps = relative_momentum_score(adj_close[INTL_EQ], signal_ts, lbs)

        if us_score >= intl_score:
            winner, winner_score, winner_comps = US_EQ, us_score, us_comps
        else:
            winner, winner_score, winner_comps = INTL_EQ, intl_score, intl_comps

        for gate in ABS_GATES:
            abs_check = absolute_gate(adj_close[winner], signal_ts, gate)
            recent_vol, eq_w = vol_target(adj_close[winner], signal_ts)

            if abs_check["passed"]:
                regime = f"RISK-ON → {winner}"
                alloc = {US_EQ: 0.0, INTL_EQ: 0.0, selected_bond: 0.0, GOLD: 0.0, CASH: 1.0 - eq_w}
                alloc[winner] = eq_w
            else:
                regime = "RISK-OFF → Defensive"
                alloc = {US_EQ: 0.0, INTL_EQ: 0.0, selected_bond: DEFENSIVE_SPLIT[0], GOLD: DEFENSIVE_SPLIT[1], CASH: 0.0}

            per_model.append({
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

    # Average allocations across ensemble; ensure we print all relevant keys
    keys = [US_EQ, INTL_EQ, selected_bond, GOLD, CASH]
    final_alloc = {k: 0.0 for k in keys}
    for r in per_model:
        for k, v in r["alloc"].items():
            final_alloc[k] = final_alloc.get(k, 0.0) + v
    n = len(per_model)
    for k in final_alloc:
        final_alloc[k] /= n

    risk_on_votes = sum(1 for r in per_model if r["regime"].startswith("RISK-ON"))
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
        "ensemble_details": per_model,
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

# ---------- Pretty printing ----------
def pct(x: float) -> str:
    return f"{x*100:6.2f}%"

def print_header(c: C, title: str):
    print(f"{c.bold}{title}{c.reset}")

def print_config(c: C, cfg: Dict[str, Any]):
    print(f"{c.dim}Config: lookbacks={cfg['rel_lookbacks']}, gates={cfg['abs_gates']}, "
          f"target vol={cfg['target_vol']:.0%}, vol window={cfg['vol_window_days']}d, "
          f"floor/cap=({int(cfg['equity_floor']*100)}%–{int(cfg['equity_cap']*100)}%), "
          f"defensive={int(DEFENSIVE_SPLIT[0]*100)}% Bonds / {int(DEFENSIVE_SPLIT[1]*100)}% Gold{c.reset}")

def print_allocation(c: C, alloc: Dict[str, float], selected_bond: str, title="Final Averaged Allocation"):
    print_header(c, f"\n{title}")
    order = [US_EQ, INTL_EQ, selected_bond, GOLD, CASH]
    labels = {US_EQ:"US", INTL_EQ:"Intl", GOLD:"Gold", CASH:"Cash/T-Bills", selected_bond: f"Bonds ({selected_bond})"}
    for t in order:
        w = alloc.get(t, 0.0)
        color = c.green if (t in [US_EQ, INTL_EQ] and w > 0) else c.cyan if t == CASH else c.yellow
        print(f"  {labels[t]:16s} ({t}): {color}{pct(w)}{c.reset}")

def print_macro_block(c: C, out: Dict[str, Any]):
    ms = out["macro_signals"]
    print_header(c, "\nMacro Classifier (month-end)")
    print(f"  Macro regime: {c.bold}{out['macro_regime']}{c.reset}   Defensive bond: {c.bold}{out['selected_bond']}{c.reset}")
    print(f"  duration_pref (TLT−BND avg 6/12m): {ms['duration_pref']:+.2%}  "
          f"breakeven_drop (TIP−IEF avg 6/12m): {ms['breakeven_drop']:+.2%}  "
          f"commodities 6m (DBC): {ms['commodities_trend_6m']:+.2%}")

def print_ensemble_table(c: C, rows: List[Dict[str, Any]], limit: int):
    print_header(c, "\nEnsemble Members")
    print("  #  Lookbacks      Gate    Winner  RelScore   AbsGate   AbsMetric              RecentVol  EqW    Regime")
    print("  -- -------------- ------- ------- ---------- --------- --------------------- ---------- ------ ---------------------------")
    for i, r in enumerate(rows[:max(1, limit)]):
        lbs = f"{r['lbs']}"
        gate = "12m>0" if r["gate"] == "abs12" else "MA~10"
        rel = f"{r['winner_score']:+.4f}"
        # Abs metric
        if r["gate"] == "abs12":
            am = r["abs_check"]["detail"].get("abs12", np.nan)
            abs_metric = f"12m={am:+.2%}"
            passed = r["abs_check"]["passed"]
        else:
            sma = r["abs_check"]["detail"].get("sma200", np.nan)
            px  = r["abs_check"]["detail"].get("px", np.nan)
            abs_metric = f"px/sma={px:.2f}/{sma:.2f}"
            passed = r["abs_check"]["passed"]
        pcol = c.green if passed else c.red
        rv  = r["recent_vol"]
        eqw = r["eq_weight"]
        regime = r["regime"]
        print(f"  {i+1:>2} {lbs:<14s} {gate:<7s} {r['winner']:<7s} {rel:>10s} "
              f"{pcol}{'PASS' if passed else 'FAIL':<9s}{c.reset} {abs_metric:<21s} {rv:>9.2%} {eqw:>6.2%} {regime}")

def print_momentum_components(c: C, r: Dict[str, Any], idx: int):
    print_header(c, f"\nDetails for ensemble member #{idx+1}: LB={r['lbs']}, gate={'12m>0' if r['gate']=='abs12' else 'MA~10'}")
    print(f"  Winner: {c.bold}{r['winner']}{c.reset}  | RelScore: {r['winner_score']:+.4f}  | Regime: {r['regime']}")
    print("  Relative momentum components:")
    print("    m   return     base_ts      base_px   last_px")
    for m, ret, base_ts, base_px, last_px in r["winner_comps"]:
        print(f"   {m:>2}  {ret:>7.2%}  {str(pd.Timestamp(base_ts).date()):>10s}   {base_px:>8.2f}  {last_px:>8.2f}")
    if r["gate"] == "abs12":
        d = r["abs_check"]["detail"]
        print(f"  Absolute gate (12m>0): 12m={d.get('abs12', float('nan')):>.2%}  px={d.get('px', float('nan')):.2f}  base_px={d.get('base_px', float('nan')):.2f}")
    else:
        d = r["abs_check"]["detail"]
        print(f"  Absolute gate (MA~10 via 200d): px={d.get('px', float('nan')):.2f}  sma200={d.get('sma200', float('nan')):.2f}  -> {'PASS' if r['abs_check']['passed'] else 'FAIL'}")
    print(f"  Vol targeting: recent_vol={r['recent_vol']:.2%}  eq_weight={r['eq_weight']:.2%}")
    print("  Allocation:")
    for k, v in r["alloc"].items():
        print(f"    {k:6s}: {pct(v)}")

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Dual Momentum (Tuned) — Console + Macro")
    p.add_argument("--asof", type=str, default=None, help="As-of date (YYYY-MM-DD). Defaults to today (NY).")
    p.add_argument("--details", type=int, default=6, help="How many ensemble rows to print in the summary table.")
    p.add_argument("--deep", type=int, default=0, help="Deep-dive this ensemble row index (1-based). 0=off.")
    p.add_argument("--export-csv", type=str, default=None, help="Optional path to export ensemble rows as CSV.")
    p.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
    # NOTE: parse_known_args allows running in notebooks (ignores injected -f kernel args)
    args, _unknown = p.parse_known_args()
    return args

def to_csv_rows(per_model: List[Dict[str, Any]], selected_bond: str) -> pd.DataFrame:
    recs = []
    for r in per_model:
        if r["gate"] == "abs12":
            abs_val = r["abs_check"]["detail"].get("abs12", np.nan)
        else:
            d = r["abs_check"]["detail"]
            abs_val = (d.get("px", np.nan) / d.get("sma200", np.nan)) if d.get("sma200", np.nan) not in [0, np.nan] else np.nan
        row = {
            "lookbacks": str(r["lbs"]),
            "gate": r["gate"],
            "winner": r["winner"],
            "rel_score": r["winner_score"],
            "abs_pass": r["abs_check"]["passed"],
            "abs_metric": abs_val,
            "recent_vol": r["recent_vol"],
            "eq_weight": r["eq_weight"],
            "regime": r["regime"],
            "selected_bond": selected_bond,
            "alloc_US": r["alloc"].get(US_EQ, 0.0),
            "alloc_INTL": r["alloc"].get(INTL_EQ, 0.0),
            "alloc_BOND": r["alloc"].get(selected_bond, 0.0),
            "alloc_GOLD": r["alloc"].get(GOLD, 0.0),
            "alloc_CASH": r["alloc"].get(CASH, 0.0),
        }
        recs.append(row)
    return pd.DataFrame.from_records(recs)

def main():
    args = parse_args()
    c = C(enable=not args.no_color)

    # Resolve as-of date
    if args.asof:
        try:
            asof = dt.date.fromisoformat(args.asof)
        except Exception:
            print("ERROR: --asof must be YYYY-MM-DD", file=sys.stderr)
            sys.exit(2)
    else:
        asof = dt.datetime.now(NY_TZ).date()

    try:
        out = decide_allocation(asof)
    except Exception as e:
        print(f"ERROR computing allocation: {e}", file=sys.stderr)
        sys.exit(1)

    # Header + macro
    print_header(c, "=== Dual Momentum (Tuned, Month-End, Ensemble) — Console + Macro ===")
    print(f"As of (NY):        {out['as_of_date']}")
    print(f"Signal date (EOM): {out['signal_date']}   Trade next business day: {out['trade_date']}")
    print(f"Ensemble regime:   {c.bold}{out['regime']}{c.reset}")
    print_macro_block(c, out)
    print_config(c, out["config"])

    # Allocation
    print_allocation(c, out["allocation"], out["selected_bond"])

    # Ensemble summary
    per_model = out["ensemble_details"]
    print_ensemble_table(c, per_model, limit=args.details)

    # Optional deep dive
    if args.deep and 1 <= args.deep <= len(per_model):
        print_momentum_components(c, per_model[args.deep - 1], args.deep - 1)

    # Optional CSV export
    if args.export_csv:
        df = to_csv_rows(per_model, out["selected_bond"])
        try:
            df.to_csv(args.export_csv, index=False)
            print(f"\n{c.dim}Exported ensemble rows to {args.export_csv}{c.reset}")
        except Exception as e:
            print(f"\nERROR exporting CSV: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
