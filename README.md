# Dual Momentum Strategy — Automated (Roth IRA Friendly)

This project implements a **tuned dual momentum strategy** that runs automatically each month, sends you an **email allocation signal**, and can also be inspected manually via a **console version**.  
The system includes:

- 📈 **Strategy logic**: Dual momentum with ensemble lookbacks, absolute momentum gates, volatility targeting, and a macro-aware defensive bond sleeve (TLT vs BND vs IEF).
- 📧 **Email version** (`momentum_script.py`): Runs via GitHub Actions on the 1st of each month and emails the new allocation.
- 💻 **Console version** (`mscript_console.py`): For manual inspection, prints a detailed report with ensemble rows, macro classifier diagnostics, and optional CSV export.
- ⚙️ **Workflow** (`run_momentum.yml`): Automates running the email script monthly with retries and artifact uploads.
- 📦 **Requirements** (`requirements.txt`): Python dependencies.

---

## Strategy Overview

**Enhanced Dual Momentum — Tuned**

- **Relative Momentum (ensemble):**  
  Compare U.S. (`VOO`) vs. International (`VXUS`) equities using lookbacks `[6, 12]`, `[9, 12]`, `[6, 9, 12]`.

- **Absolute Momentum (two gates):**  
  1. 12-month total return > 0  
  2. Price above ~10-month SMA (200 trading days)  

- **Volatility Targeting:**  
  Scale equity exposure to target **15% annualized vol**, using 63 trading days of history, with a **30% floor / 100% cap**.

- **Risk-ON:**  
  Allocate to winner (VOO or VXUS), residual to cash (`BIL`).

- **Risk-OFF (macro-aware):**  
  Defensive 50/50 split between:  
  - Gold (`IAUM`, fallback `IAU`)  
  - Bonds, chosen dynamically via macro classifier:  
    - **Deflation Crash:** `TLT` (long Treasuries)  
    - **Inflation Downturn:** `BND` (Agg)  
    - **Neutral:** `IEF` (7–10Y Treasuries)

- **Signal date:** last completed month-end trading day.  
- **Trade date:** next U.S. business day after the signal.

---

## 📧 Email Version (`momentum_script.py`)

- Generates a **mobile-friendly HTML email** with:
  - Regime (RISK-ON or RISK-OFF)  
  - Target allocation (VOO, VXUS, Bonds, Gold, Cash)  
  - Defensive bond choice + macro classifier signals  
  - Ensemble sample rows  

### Setup

1. Add Gmail secrets in your repo settings:
   - `GMAIL_ADDRESS`  
   - `GMAIL_APP_PASSWORD` (generate an app password in Gmail)  
   - `EMAIL_RECIPIENT`

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

3. Run locally:

   ```bash
   python momentum_script.py
   ```

---

## 💻 Console Version (`mscript_console.py`)

* Prints a **rich console report** with ANSI colors:

  * Signal date, trade date, final allocation
  * Macro classifier diagnostics (deflation vs inflation vs neutral)
  * Full ensemble table
  * Optional deep-dive into any ensemble member
  * CSV export of ensemble rows

### Usage

```bash
python mscript_console.py --asof 2025-08-28 --details 12 --deep 3 --export-csv ensemble.csv
```

Options:

* `--asof YYYY-MM-DD` : override as-of date (default = today)
* `--details N` : number of ensemble rows to show (default 6)
* `--deep K` : deep dive into ensemble row K (1-based)
* `--export-csv file.csv` : export ensemble rows
* `--no-color` : disable ANSI colors

---

## ⚙️ Workflow (`run_momentum.yml`)

```yaml
on:
  schedule:
    - cron: '10 21 1 * *'   # 1st of every month, 21:10 UTC (~5:10 PM New York)
  workflow_dispatch:
```

* Installs Python & dependencies
* Runs `momentum_script.py`
* Retries once on failure
* Uploads logs/artifacts (including HTML email body if you save it to `/tmp/email.html`)

---

## 🕒 Trading Instructions

* **Email arrives:** evening of the 1st of each month (\~5:10 PM New York time).
* **Signal date:** last completed month-end (e.g. Aug 29 if Aug 31 was a Sunday).
* **Trade date:** the next U.S. business day (shown clearly in the email).
* **Action:** place your orders on that trade date (typically market open or shortly after).
  Since this is a **monthly strategy**, a few hours one way or the other doesn’t matter much.

---

## 📂 Project Structure

```
momentum_script.py        # Email version (automated run via workflow)
mscript_console.py        # Console version (manual runs, CSV, deep dive)
run_momentum.yml          # GitHub Actions workflow
requirements.txt          # Python dependencies
README.md                 # This file
```

---

## 🔒 Notes

* Designed with Roth IRA / other tax-sheltered accounts in mind (since monthly rebalancing generates no tax drag). The same strategy works in regular taxable accounts, but frequent reallocations may trigger short-term capital gains.
* Uses **ETFs instead of mutual funds** for automation & intraday data (VOO, VXUS, TLT, BND, IEF, IAUM, BIL).
* All allocations + macro bond decisions are **reported clearly** in the email and console.

