# Dual Momentum Strategy

This project implements a **tuned dual momentum strategy** that runs automatically each month, sends you an **email allocation signal**, and can also be inspected manually via a **console version**.
The system includes:

  - 📈 **Strategy logic**: Dual momentum with ensemble lookbacks, absolute momentum gates, volatility targeting, and a macro-aware defensive bond sleeve (`TLT` vs `BND` vs `IEF`).
  - 📧 **Email version** (`momentum_script.py`): Runs via GitHub Actions on the 1st of each month and emails the new allocation.
  - 💻 **Console version** (`mscript_console.py`): For manual inspection, prints a detailed report with ensemble rows, macro classifier diagnostics, and optional CSV export.
  - ⚙️ **Workflow** (`run_momentum.yml`): Automates running the email script monthly with retries and artifact uploads.
  - 📦 **Requirements** (`requirements.txt`): Python dependencies.

-----

## Strategy Overview

### Enhanced Dual Momentum — Tuned

  - **Relative Momentum (ensemble):**
    Selects the strongest performer from a list of U.S. and International equity options using lookbacks `[6, 12]`, `[9, 12]`, and `[6, 9, 12]`.

      - **U.S. Options:** **`SPMO`** (S\&P 500 Momentum) vs. **`QQQM`** (Nasdaq 100)
      - **Int'l Options:** **`IQLT`** (Int'l Quality) vs. **`DIVI`** (Int'l Dividends)

  - **Absolute Momentum (two gates):**

    1.  12-month total return \> 0
    2.  Price above \~10-month SMA (200 trading days)

  - **Volatility Targeting:**
    Scale equity exposure to target **15% annualized vol**, using 63 trading days of history, with a **30% floor / 100% cap**.

  - **Risk-ON:**
    Allocate to the **winning equity ETF**, with the residual to cash (`BIL`).

  - **Risk-OFF (macro-aware):**
    Defensive 50/50 split between:

      - Gold (`IAUM`, fallback `IAU`)
      - Bonds, chosen dynamically via macro classifier:
          - **Deflation Crash:** `TLT` (long Treasuries)
          - **Inflation Downturn:** `BND` (Agg)
          - **Neutral:** `IEF` (7–10Y Treasuries)

  - **Signal date:** last completed month-end trading day.

  - **Trade date:** next U.S. business day after the signal.

-----

## 📧 Email Version (`momentum_script.py`)

  - Generates a **mobile-friendly HTML email** with:
      - Regime (RISK-ON or RISK-OFF)
      - Target allocation showing weights for all potential assets (**`SPMO`**, **`QQQM`**, **`IQLT`**, **`DIVI`**, Bonds, Gold, Cash)
      - Defensive bond chosen by the macro classifier.

### Setup

1.  Add Gmail secrets in your repo's `Settings` \> `Secrets and variables` \> `Actions`:

      - `GMAIL_ADDRESS`
      - `GMAIL_APP_PASSWORD` (generate an app password in your Google account)
      - `EMAIL_RECIPIENT`

2.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3.  Run locally:

    ```bash
    python momentum_script.py
    ```

-----

## 💻 Console Version (`mscript_console.py`)

  - Prints a **rich console report** with ANSI colors:
      - Signal date, trade date, and final averaged allocation
      - Macro classifier diagnostics (deflation vs inflation vs neutral)
      - Full table of all ensemble member decisions
      - Optional deep-dive into any single ensemble member's logic
      - CSV export of ensemble rows

### Usage

```bash
# Get signal for a specific date with full details and CSV export
python mscript_console.py --asof 2025-08-28 --details 12 --deep 3 --export-csv ensemble.csv
```

**Options:**

  * `--asof YYYY-MM-DD`: override as-of date (default = today)
  * `--details N`: number of ensemble rows to show (default 6)
  * `--deep K`: deep dive into ensemble row K (1-based)
  * `--export-csv file.csv`: export ensemble rows
  * `--no-color`: disable ANSI colors

-----

## ⚙️ Workflow (`run_momentum.yml`)

```yaml
on:
  schedule:
    - cron: '10 21 1 * *'   # 1st of every month, 21:10 UTC (~5:10 PM New York)
  workflow_dispatch:
```

  - Installs Python & dependencies
  - Runs `momentum_script.py`
  - Retries once on failure
  - Uploads logs as workflow artifacts for debugging

-----

## 🕒 Trading Instructions

  - **Email arrives:** evening of the 1st of each month (\~5:10 PM New York time).
  - **Signal date:** last completed month-end (e.g., Sept 30 for the October signal).
  - **Trade date:** the next U.S. business day (shown clearly in the email).
  - **Action:** Place your orders on that trade date. Since this is a **monthly strategy**, precise timing during the day is not critical.

-----

## 📂 Project Structure

```
momentum_script.py      # Email version (automated run via workflow)
mscript_console.py      # Console version (manual runs, CSV, deep dive)
run_momentum.yml        # GitHub Actions workflow
requirements.txt        # Python dependencies
README.md               # This file
```

-----

## 🔒 Notes

  - Designed with Roth IRA / other tax-sheltered accounts in mind, since monthly rebalancing generates no tax drag.
  - Uses **ETFs instead of mutual funds** for automation & intraday data (**`SPMO`**, **`QQQM`**, **`IQLT`**, **`DIVI`**, `TLT`, `BND`, `IEF`, `IAUM`, `BIL`).
  - All allocation decisions are **reported clearly** in the email and console output.
