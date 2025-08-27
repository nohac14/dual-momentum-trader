"""
Enhanced Dual Momentum (Monthly Rules) with Email Notifications for GitHub Actions

Strategy Rules:
1. Relative Momentum: Pick winner between US & Intl. equities based on the average of 6, 9, and 12-month returns.
2. Absolute Momentum (Crash Protection): Only invest in the winner if its score is positive AND its price is above its 200-day moving average.
3. Defensive Allocation: If absolute momentum is negative, switch to defensive assets.
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
    # More history is needed for volatility calculation
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

    # (Crash protection logic remains the same)
    slow_trigger_passed = winner_score > 0.0
    winner_price = winner_prices.iloc[-1]
    winner_sma100 = winner_prices.rolling(window=100).mean().iloc[-1]
    is_above_sma100 = winner_price > winner_sma100
    six_month_return = next((r for m, r, *_ in winner_comps if m == 6), None)
    is_6m_positive = six_month_return > 0.0 if six_month_return is not None else False
    fast_trigger_passed = is_above_sma100 and is_6m_positive
    
    # --- NEW: Volatility Targeting Logic ---
    target_vol = 0.15  # Target 15% annualized volatility
    # Calculate daily returns for the last month (~21 trading days)
    daily_returns = winner_prices.pct_change()
    # Calculate annualized volatility
    recent_vol = daily_returns.iloc[-21:].std() * np.sqrt(252)
    
    # Determine allocation based on volatility (capped at 100%)
    vol_based_alloc = min(1.0, target_vol / recent_vol) if recent_vol > 0 else 1.0

    if slow_trigger_passed and fast_trigger_passed:
        regime = f"RISK-ON → {winner}"
        # Allocation is now dynamic
        final_winner_alloc = vol_based_alloc
        alloc = {US_EQ: 0.0, INTL_EQ: 0.0, BONDS: 1.0 - final_winner_alloc, GOLD: 0.0}
        alloc[winner] = final_winner_alloc
    else:
        regime = f"RISK-OFF → Defensive"
        alloc = {US_EQ: 0.0, INTL_EQ: 0.0,
                 BONDS: DEFENSIVE_SPLIT[0], GOLD: DEFENSIVE_SPLIT[1]}

    return {
        "as_of_date": str(asof), "winner": winner, "regime": regime, "allocation": alloc,
        "protection_checks": {
            "slow_trigger": {"score": winner_score, "passed": slow_trigger_passed},
            "fast_trigger": {"price": winner_price, "sma100": winner_sma100, "is_above_sma100": is_above_sma100,
                             "6m_return": six_month_return, "is_6m_positive": is_6m_positive, "passed": fast_trigger_passed}
        },
        "volatility_targeting": {
            "target": target_vol, "recent": recent_vol, "allocation_pct": vol_based_alloc
        }
    }

def format_results_for_email(out: dict) -> tuple[str, str]:
    """Formats the allocation results into a detailed, mobile-friendly HTML email."""
    
    is_risk_on = "RISK-ON" in out['regime']
    status_color = "#28a745" if is_risk_on else "#fd7e14"
    subject = f"Dual Momentum Signal: {out['regime']}"
    
    # (Protection checks section remains the same)
    checks = out['protection_checks']
    slow_check, fast_check = checks['slow_trigger'], checks['fast_trigger']
    def status_tag(passed):
        color, text = ("#28a745", "PASS") if passed else ("#dc3545", "FAIL")
        return f'<b style="color: {color};">{text}</b>'

    protection_rows = f"""
    <tr><td><b>Slow Trigger</b> (Blended Score > 0%)</td><td style="text-align:right;">{status_tag(slow_check['passed'])}</td></tr>
    <tr><td><b>Fast Trigger</b> (Price > 100d SMA AND 6m Ret > 0%)</td><td style="text-align:right;">{status_tag(fast_check['passed'])}</td></tr>
    """

    # --- NEW: Build HTML for Volatility Targeting ---
    vol_info = out['volatility_targeting']
    volatility_rows = ""
    if is_risk_on:
        volatility_rows = f"""
        <tr><td>Recent Volatility (Annualized)</td><td style="text-align:right;">{vol_info['recent']:.2%}</td></tr>
        <tr><td>Target Volatility</td><td style="text-align:right;">{vol_info['target']:.2%}</td></tr>
        <tr><td><b>Calculated Allocation</b></td><td style="text-align:right;"><b>{vol_info['allocation_pct']:.2%}</b></td></tr>
        """
    
    alloc_rows = ""
    for ticker in [US_EQ, INTL_EQ, BONDS, GOLD]:
        weight = out['allocation'].get(ticker, 0.0)
        style = ' style="font-weight: bold;"' if weight > 0 and ticker in [US_EQ, INTL_EQ] else ''
        alloc_rows += f"<tr{style}><td><b>{ticker}</b></td><td>{weight:.2%}</td></tr>"

    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
    <body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f4f4f4;">
      <table role="presentation" style="width: 100%; max-width: 600px; margin: 20px auto; background-color: #ffffff; border-collapse: collapse; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
        <tr><td style="padding: 20px; text-align: center; background-color: #4a5568; color: #ffffff; border-radius: 8px 8px 0 0;"><h2 style="margin: 0;">Dual Momentum Signal</h2><p style="margin: 5px 0 0;">for {out['as_of_date']}</p></td></tr>
        <tr><td style="padding: 25px;">
            <table role="presentation" style="width: 100%;"><tr><td style="padding: 15px; background-color: {status_color}; color: #ffffff; text-align: center; border-radius: 5px;"><h3 style="margin: 0; font-size: 20px;">{out['regime']}</h3></td></tr></table>
            <h3 style="border-bottom: 2px solid #e2e8f0; padding-bottom: 5px; margin-top: 30px;">Crash Protection Status</h3>
            <table role="presentation" style="width: 100%;">{protection_rows}</table>
            
            {"<h3 style='border-bottom: 2px solid #e2e8f0; padding-bottom: 5px; margin-top: 30px;'>Volatility Targeting</h3><table role='presentation' style='width: 100%;'>" + volatility_rows + "</table>" if is_risk_on else ""}

            <h3 style="border-bottom: 2px solid #e2e8f0; padding-bottom: 5px; margin-top: 30px;">Target Allocation</h3>
            <table role="presentation" style="width: 100%;">{alloc_rows}</table>
        </td></tr>
        <tr><td style="padding: 20px; text-align: center; color: #718096; font-size: 0.8em;"><p style="margin: 0;">This is an automated notification.</p></td></tr>
      </table>
    </body>
    </html>
    """
    return subject, html_body

def send_email(subject: str, html_body: str):
    """Sends an email using credentials stored in GitHub Secrets."""
    sender = os.environ.get('GMAIL_ADDRESS')
    password = os.environ.get('GMAIL_APP_PASSWORD')
    # ADD THIS LINE
    recipient = os.environ.get('EMAIL_RECIPIENT') 

    # UPDATE THE CHECK
    if not all([sender, password, recipient]):
        print("ERROR: One or more secrets (GMAIL_ADDRESS, GMAIL_APP_PASSWORD, EMAIL_RECIPIENT) not found.", file=sys.stderr)
        return

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = sender
    # UPDATE THIS LINE
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
        
        print("=== Enhanced Dual Momentum (SMA Filter) ===")
        print(f"As of (NY): {out['as_of_date']} | Lookbacks: {LOOKBACKS} months\n")
        print(f"Regime: {out['regime']}\n")
        print("Target allocation:")
        for k in [US_EQ, INTL_EQ, BONDS, GOLD]:
            w = out["allocation"].get(k, 0.0)
            print(f"  {k}: {w*100:.2f}%")
        
        print("\n---")
        email_subject, email_body = format_results_for_email(out)
        send_email(email_subject, email_body)

    except Exception as e:
        print(f"Error computing allocation: {e}", file=sys.stderr)
        send_email("Dual Momentum Script FAILED", f"<p>The script failed with error: {e}</p>")
        sys.exit(1)

if __name__ == "__main__":

    main()


