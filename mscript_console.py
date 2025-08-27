"""
Enhanced Dual Momentum (Monthly Rules) - Console Report Version

Strategy Rules:
1. Relative Momentum: Pick winner between US & Intl. equities based on the average of 6, 9, and 12-month returns.
2. Crash Protection: A two-factor system using a slow trigger (blended momentum) and a fast trigger (100d SMA + 6m return).
3. Risk Management: Volatility targeting adjusts the allocation size during RISK-ON periods.
"""

import sys
import datetime as dt
import numpy as np
import pandas as pd
import pytz
import yfinance as yf

# --------- CONFIG ---------
US_EQ = "VOO"       # U.S. equities FXAIX
INTL_EQ = "VXUS"    # International ex-US FTIHX
BONDS = "FNBGX"     # FNBGX (Deflation Crash) or FXNAX (Inflation Downturn)
GOLD = "IAUM"
LOOKBACKS = [6, 9, 12]
DEFENSIVE_SPLIT = (0.50, 0.50)
# --------------------------

NY_TZ = pytz.timezone("America/New_York")

def get_momentum_score(price_series: pd.Series, last_ts: pd.Timestamp):
    """Calculates the average momentum score and its components."""
    comps = []
    price_series = price_series.dropna()
    if price_series.empty:
        return 0.0, []

    last_px = price_series.iloc[-1]
    for months in LOOKBACKS:
        target_date = pd.Timestamp(last_ts.tz_localize(None)) - pd.DateOffset(months=months)
        idx = price_series.index.searchsorted(target_date)
        idx = min(idx, len(price_series) - 1)
        
        base_px = price_series.iloc[idx]
        if pd.notna(base_px) and base_px > 0:
            comps.append((months, (last_px / base_px) - 1.0,
                          price_series.index[idx], float(base_px), float(last_px)))
                          
    score = np.mean([r for _, r, *_ in comps]) if comps else 0.0
    return score, comps

def decide_allocation(asof: dt.date | None = None) -> dict:
    """Apply professional-grade rules with Volatility Targeting."""
    if asof is None: asof = dt.datetime.now(NY_TZ).date()

    all_tickers = [US_EQ, INTL_EQ, BONDS, GOLD]
    start_date = (pd.Timestamp(asof) - pd.DateOffset(months=max(LOOKBACKS)+12)).date()
    all_data = yf.download(all_tickers, start=start_date, end=asof + dt.timedelta(days=1), auto_adjust=False, progress=False)
    
    if all_data.empty: raise RuntimeError("Failed to download market data.")
        
    adj_close = all_data["Adj Close"]
    last_ts = adj_close.index[-2] if pd.isna(adj_close.iloc[-1]).any() else adj_close.index[-1]

    us_score, us_comps = get_momentum_score(adj_close[US_EQ], last_ts)
    intl_score, intl_comps = get_momentum_score(adj_close[INTL_EQ], last_ts)

    if us_score >= intl_score:
        winner, winner_score, winner_comps = US_EQ, us_score, us_comps
    else:
        winner, winner_score, winner_comps = INTL_EQ, intl_score, intl_comps

    winner_prices = adj_close[winner].dropna()
    if len(winner_prices) < 200:
      raise RuntimeError(f"Not enough data for {winner}")

    slow_trigger_passed = winner_score > 0.0
    winner_price = winner_prices.iloc[-1]
    winner_sma100 = winner_prices.rolling(window=100).mean().iloc[-1]
    is_above_sma100 = winner_price > winner_sma100
    six_month_return = next((r for m, r, *_ in winner_comps if m == 6), None)
    is_6m_positive = six_month_return > 0.0 if six_month_return is not None else False
    fast_trigger_passed = is_above_sma100 and is_6m_positive
    
    target_vol = 0.15
    daily_returns = winner_prices.pct_change()
    recent_vol = daily_returns.iloc[-21:].std() * np.sqrt(252)
    vol_based_alloc = min(1.0, target_vol / recent_vol) if recent_vol > 0 else 1.0

    if slow_trigger_passed and fast_trigger_passed:
        regime = f"RISK-ON → {winner}"
        final_winner_alloc = vol_based_alloc
        alloc = {US_EQ: 0.0, INTL_EQ: 0.0, BONDS: 1.0 - final_winner_alloc, GOLD: 0.0}
        alloc[winner] = final_winner_alloc
    else:
        regime = f"RISK-OFF → Defensive"
        alloc = {US_EQ: 0.0, INTL_EQ: 0.0,
                 BONDS: DEFENSIVE_SPLIT[0], GOLD: DEFENSIVE_SPLIT[1]}

    return {
        "as_of_date": str(asof), "winner": winner, "regime": regime, "allocation": alloc,
        "us_score": us_score, "us_comps": us_comps,
        "intl_score": intl_score, "intl_comps": intl_comps,
        "protection_checks": {
            "slow_trigger": {"score": winner_score, "passed": slow_trigger_passed},
            "fast_trigger": {"price": winner_price, "sma100": winner_sma100, "is_above_sma100": is_above_sma100,
                             "6m_return": six_month_return, "is_6m_positive": is_6m_positive, "passed": fast_trigger_passed}
        },
        "volatility_targeting": {
            "target": target_vol, "recent": recent_vol, "allocation_pct": vol_based_alloc
        }
    }

def main():
    """
    Calculates the allocation and prints a detailed report to the console.
    """
    try:
        out = decide_allocation()
        
        # --- Helper for PASS/FAIL emojis ---
        def status_emoji(passed):
            return "✅ PASS" if passed else "❌ FAIL"

        # --- Header ---
        print("="*50)
        print(" ".join(f"DUAL MOMENTUM MONTHLY SIGNAL REPORT".split()))
        print("="*50)
        print(f"As of (NY): {out['as_of_date']}\n")

        # --- Momentum Breakdown ---
        print("-" * 20 + " MOMENTUM ANALYSIS " + "-" * 19)
        def show_momentum_details(name, score, comps):
            print(f"  {name} Blended Score: {score:.2%}")
            for m, r, base_ts, base_px, last_px in comps:
                print(f"    - {m:>2}m Return: {r:7.2%} (from {base_ts.date()} @ {base_px:.2f})")
        
        show_momentum_details(US_EQ, out['us_score'], out['us_comps'])
        print()
        show_momentum_details(INTL_EQ, out['intl_score'], out['intl_comps'])
        print(f"\n  WINNER: {out['winner']}\n")

        # --- Crash Protection Status ---
        print("-" * 18 + " CRASH PROTECTION STATUS " + "-" * 15)
        checks = out['protection_checks']
        slow_check = checks['slow_trigger']
        fast_check = checks['fast_trigger']
        
        print(f"  Slow Trigger (Blended Score > 0%): {status_emoji(slow_check['passed'])}")
        print(f"    - Score: {slow_check['score']:.2%}")
        
        print(f"\n  Fast Trigger (Complex): {status_emoji(fast_check['passed'])}")
        print(f"    - Price > 100d SMA: {status_emoji(fast_check['is_above_sma100'])}")
        print(f"      (Price: {fast_check['price']:.2f}, SMA: {fast_check['sma100']:.2f})")
        print(f"    - 6-Month Return > 0%: {status_emoji(fast_check['is_6m_positive'])}")
        print(f"      (Return: {fast_check.get('6m_return', 0):.2%})\n")

        # --- Final Regime ---
        print("-" * 24 + " REGIME " + "-" * 24)
        print(f"  ==> {out['regime']}\n")

        # --- Volatility Targeting ---
        if "RISK-ON" in out['regime']:
            print("-" * 18 + " VOLATILITY TARGETING " + "-" * 16)
            vol_info = out['volatility_targeting']
            print(f"  Recent Volatility (Annualized): {vol_info['recent']:.2%}")
            print(f"  Target Volatility: {vol_info['target']:.2%}")
            print(f"  Calculated Allocation Size: {vol_info['allocation_pct']:.2%}\n")
            
        # --- Target Allocation ---
        print("-" * 19 + " FINAL ALLOCATION " + "-" * 19)
        for ticker in [US_EQ, INTL_EQ, BONDS, GOLD]:
            weight = out["allocation"].get(ticker, 0.0)
            print(f"  {ticker:<5}: {weight:7.2%}")
        print("="*50)

    except Exception as e:
        print(f"\nERROR COMPUTING ALLOCATION: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
