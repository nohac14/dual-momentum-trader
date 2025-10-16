# Dual Momentum (Tuned) + Macro-Aware Defensive Bond — Plain-English Overview

**Goal:** A simple, rules-based portfolio you update **once a month**. It tries to own the **strongest equity index when trend is healthy (Risk-ON)** and shifts to a **defensive mix of bonds + gold when trend weakens (Risk-OFF)**. Sizing adapts to volatility, and the *type* of defensive bond is chosen by a small macro “classifier.”

---

## What it holds (the building blocks)

* **US Equities (choose one):**
  **SPMO** (S&P 500 momentum tilt) **or** **QQQM** (Nasdaq-100).
  → The strategy picks whichever has **stronger momentum**.

* **International Equities (choose one):**
  **IQLT** (international **quality**) **or** **DIVI** (international **dividend/low-vol**).
  → Compare the US winner vs. the International winner; **own the stronger** of those two.

* **Defensives:**

  * **Gold** (IAUM).
  * **Bonds** (one of): **VGLT** (long Treasuries), **BND** (US Aggregate), **VGIT** (intermediate Treasuries).
  * **Cash/T-Bills** (SGOV) as a **residual** when Risk-ON sizing is less than 100%.

* **Cadence:** **Signals at month-end only**; trade **next business day**.

---

## How it decides (the core logic)

1. **Relative momentum:** rank candidates by simple 6/9/12-month total returns (an **ensemble** of lookbacks).
2. **Absolute trend “gate”:** proceed to Risk-ON only if **12-month return > 0%** or **price > ~200-day MA**; otherwise Risk-OFF.
3. **Volatility targeting (equity sleeve):** target ~**15% annualized vol** over a ~63-day lookback with **floor/cap (30%–100%)**. Unused weight sits in **T-Bills**.
4. **Risk-OFF mix:** **50/50** between **bonds + gold**.

   * A **macro classifier** picks the bond type:

     * **Deflation/crash vibe → VGLT** (long duration)
     * **Inflation-tilted slowdown → BND** (broad Aggregate)
     * **Neutral → VGIT** (intermediate duration)
   * Votes come from 6/12-mo trends in long-duration vs aggregate, TIPS vs intermediate Treasuries (breakevens), and commodities.
5. **Ensemble averaging:** average allocations across lookback/gate variants to **reduce model luck**.

---

## Why these tickers (over the popular alternatives)

### US: **Why SPMO over SPY?**

* **Purpose:** In Risk-ON, we want the **strongest horse**, not the average of the herd.
* **SPMO** is an **S&P 500 momentum** sleeve: it **tilts toward recent winners** among large caps, which often **outperform the broad index (SPY)** during rising or trending markets.
* **Trade-off:** SPMO can **rebalance into new leaders** (higher turnover, sector tilts), which may **underperform** in sharp regime flips. The strategy tempers this with:

  * The **absolute trend gate** (to avoid big drawdowns),
  * **Vol targeting** (to right-size exposure),
  * A **direct head-to-head** with **QQQM** (if mega-cap tech leadership is dominant, QQQM often wins).

> TL;DR: In a momentum-driven Risk-ON sleeve, **SPMO** gives more **beta to leadership** than **SPY**, which is designed to be **average, not strongest**.

### US growth proxy: **Why QQQM (vs QQQ)?**

* **Same index** exposure (Nasdaq-100), but **QQQM** has a **lower expense ratio** and is designed for **buy-and-hold** allocations.
* Liquidity in QQQ is deeper **in options**, but for monthly rebalances **QQQM’s lower fee** is a cleaner fit.

### International: **Why IQLT/DIVI over VXUS?**

* **VXUS** is a **cap-weighted “own everything”** fund. That’s perfect for beta, but our Risk-ON sleeve seeks **relative strength + resiliency**.
* **IQLT** (quality tilt) tends to favor firms with **strong profitability, stable earnings, and lower leverage**—traits that **hold up better** in choppy regimes and **compound** more cleanly.
* **DIVI** (dividend/low-vol tilt) tends to emphasize **defensive, cash-generative** names and can **cushion drawdowns** vs a broad cap-weight.
* The strategy **lets them compete**: whichever of **IQLT** or **DIVI** has stronger momentum faces the US winner (SPMO/QQQM). If international leadership is real, it **earns the slot**; if not, the US winner holds it.

> TL;DR: **IQLT/DIVI** are **factor-enhanced** sleeves that often **behave better** than VXUS in the exact moments when a momentum strategy cares (trend, resilience, and follow-through). VXUS is excellent core beta; here we want **edge**.

### Defensives: **Why VGLT / BND / VGIT, not “one-size-fits-all” bonds?**

* **Crashes with falling inflation expectations:** **VGLT** (long duration) typically **hedges best** and **rallies most** when real yields drop hard.
* **Inflation-tinged slowdowns:** **BND** (broad Agg) is **more resilient** than pure long duration when inflation/breakevens are sticky.
* **Muddle-through / uncertain:** **VGIT** (intermediate) balances **rate sensitivity** and **drawdown control**.
* The **macro classifier** is deliberately simple (trend-based) so we **don’t overfit** to narratives.

### Gold: **Why IAUM (vs GLD)?**

* **IAUM** is a **low-cost physical gold trust**; for monthly rebalances and modest trade sizes, the **fee edge compounds**.
* If you routinely place **very large orders or need options liquidity**, **GLD** may be preferable. The strategy itself is **agnostic**; IAUM is used for **net-of-fees efficiency**.

---

## What a month looks like (workflow)

1. **Month-end**: pull prices; compute 6/9/12-mo returns.
2. Pick **US winner** (SPMO vs QQQM) and **Intl winner** (IQLT vs DIVI); then **final equity winner** is the stronger of those two.
3. Check **absolute trend gate** (12-mo > 0% *or* above ~200-DMA).
4. If **Risk-ON**:

   * **Vol-target** equity to keep risk near **15%** (floor **30%**, cap **100%**);
   * **Residual to T-Bills**.
5. If **Risk-OFF**:

   * Run **macro classifier** → pick **VGLT / BND / VGIT**;
   * Allocate **50% bonds / 50% gold**.
6. **Average** across ensemble members; **trade next business day**.

---

## Strengths & trade-offs

**Strengths**

* **Own leadership** (SPMO/QQQM/IQLT/DIVI) **when trend confirms**.
* **Defend smartly** (bond type chosen by macro context) + **gold** ballast.
* **Vol targeting** smooths the ride; **ensemble** reduces lookback luck.
* **Monthly**—low maintenance, low noise.

**Trade-offs**

* Momentum/trend can **whipsaw** in range-bound markets.
* Factor tilts (SPMO, IQLT, DIVI) can **lag** broad beta **when leadership rotates abruptly**.
* The macro classifier is **intentionally simple**—it won’t catch every nuance.

---

## Quick glossary

* **Relative momentum:** own what’s been winning vs peers.
* **Absolute momentum (trend):** own risk assets only if their **own** trend is positive.
* **Vol targeting:** size positions to a **risk budget**, not a gut feeling.
* **Duration:** interest-rate sensitivity of bonds (more duration → bigger moves when yields change).

---

### Bottom line

> **Pick the strongest equity sleeve when the trend is on; otherwise own quality defense chosen for the regime.**
> Keep risk steady with **vol targeting**. Recheck **once a month**.
