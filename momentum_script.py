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
    """Apply the enhanced rules and produce target allocations."""
    if asof is None: asof = dt.datetime.now(NY_TZ).date()

    all_tickers = [US_EQ, INTL_EQ, BONDS, GOLD]
    start_date = (pd.Timestamp(asof) - pd.DateOffset(months=max(LOOKBACKS)+9)).date()
    all_data = yf.download(all_tickers, start=start_date, end=asof + dt.timedelta(days=1), auto_adjust=False, progress=False)
    
    if all_data.empty: raise RuntimeError("Failed to download market data.")
        
    adj_close = all_data["Adj Close"]
    last_ts = adj_close.index[-2] if pd.isna(adj_close.iloc[-1]).any() else adj_close.index[-1]

    us_score, us_comps = get_momentum_score(adj_close[US_EQ], last_ts)
    intl_score, intl_comps = get_momentum_score(adj_close[INTL_EQ], last_ts)

    if us_score >= intl_score:
        winner, winner_score = US_EQ, us_score
    else:
        winner, winner_score = INTL_EQ, intl_score

    winner_prices = adj_close[winner].dropna()
    if len(winner_prices) < 200:
      raise RuntimeError(f"Not enough data to compute 200-day SMA for {winner}")

    winner_price = winner_prices.iloc[-1]
    winner_sma200 = winner_prices.rolling(window=200).mean().iloc[-1]
    is_above_sma = winner_price > winner_sma200
    
    if winner_score > 0.0 and is_above_sma:
        alloc = {US_EQ: 1.0, INTL_EQ: 0.0, BONDS: 0.0, GOLD: 0.0} if winner == US_EQ else \
                {US_EQ: 0.0, INTL_EQ: 1.0, BONDS: 0.0, GOLD: 0.0}
        regime = f"RISK-ON → {winner}"
    else:
        alloc = {US_EQ: 0.0, INTL_EQ: 0.0,
                 BONDS: DEFENSIVE_SPLIT[0], GOLD: DEFENSIVE_SPLIT[1]}
        regime = f"RISK-OFF → {DEFENSIVE_SPLIT[0]*100:.0f}% {BONDS} + {DEFENSIVE_SPLIT[1]*100:.0f}% {GOLD}"

    return {
        "as_of_date": str(asof), "us_momentum_score": us_score, "us_comps": us_comps,
        "intl_momentum_score": intl_score, "intl_comps": intl_comps, "winner": winner,
        "winner_score": winner_score, "regime": regime, "allocation": alloc,
        "sma_check": {"price": winner_price, "sma200": winner_sma200, "is_above": is_above_sma}
    }

def format_results_for_email(out: dict) -> tuple[str, str]:
    """Formats the allocation results into a detailed HTML string for the email body."""
    subject = f"Dual Momentum Signal: {out['regime']}"
    sma_info = out['sma_check']
    sma_status = "PASS" if sma_info['is_above'] else "FAIL"
    
    # Helper function to build the detailed momentum rows for one asset
    def _build_momentum_rows(name, score, comps):
        rows = f'<tr><td colspan="2"><b>{name}</b></td><td style="text-align:right;"><b>{score:.2%}</b></td></tr>'
        for m, r, base_ts, base_px, last_px in comps:
            rows += (f'<tr><td style="padding-left: 20px;"><em>{m}-month</em></td>'
                     f'<td style="font-size: 0.8em; color: #555;"><em>({base_ts.date()})</em></td>'
                     f'<td style="text-align:right;"><em>{r:.2%}</em></td></tr>')
        return rows

    # Build the momentum breakdown table
    us_momentum_rows = _build_momentum_rows(US_EQ, out['us_momentum_score'], out['us_comps'])
    intl_momentum_rows = _build_momentum_rows(INTL_EQ, out['intl_momentum_score'], out['intl_comps'])
    
    # Build the allocation table
    alloc_rows = ""
    for ticker in [US_EQ, INTL_EQ, BONDS, GOLD]:
        weight = out['allocation'].get(ticker, 0.0)
        alloc_rows += f"<tr><td><b>{ticker}</b></td><td>{weight:.2%}</td></tr>"

    html_body = f"""
    <html>
      <head>
        <style>
          body {{ font-family: sans-serif; color: #333; }}
          table {{ border-collapse: collapse; width: 100%; max-width: 400px; }}
          th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        </style>
      </head>
      <body>
        <h2>Dual Momentum Signal for {out['as_of_date']}</h2>
        <p><b>Regime: {out['regime']}</b></p>
        <p>
          Winner: {out['winner']} (Score: {out['winner_score']:.2%})<br>
          SMA(200) Check: Price ({sma_info['price']:.2f}) vs SMA ({sma_info['sma200']:.2f}) → <b>{sma_status}</b>
        </p>

        <h3>Momentum Score Breakdown:</h3>
        <table>
          {us_momentum_rows}
          {intl_momentum_rows}
        </table>

        <h3>Target Allocation:</h3>
        <table>
          {alloc_rows}
        </table>
        <p><small>This is an automated notification.</small></p>
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
