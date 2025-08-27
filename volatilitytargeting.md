Volatility targeting makes your **"RISK-ON"** allocation dynamic. Instead of always being 100% in stocks, your stock percentage now scales down automatically as market risk increases, with the remaining portion of your portfolio shifting into the safety of bonds.

---
## The "Dimmer Switch" for Your Allocation

Think of it as a dimmer switch instead of a simple ON/OFF switch. The system continuously adjusts your exposure based on current market conditions.

* **Calm Markets (Low Volatility)**
    When the market's recent volatility is **less than or equal to** our 15% target, the system calculates that it's safe to be fully invested. The allocation will be **100% in the winning stock** (`VOO` or `VXUS`).

* **Chaotic Markets (High Volatility)**
    When the market's recent volatility **rises above** the 15% target, the system automatically de-risks. The stock allocation will be **less than 100%**, and the unallocated portion is moved to your bond fund (`FNBGX`) for stability.

---
## Practical Examples

The allocation is calculated using the formula: `Allocation % = (Target Volatility / Recent Volatility)`

* **Scenario 1: Stable Bull Market**
    * Recent Volatility is **10%**.
    * Calculation: `15% / 10% = 1.5` (Capped at 1.0)
    * **Resulting Allocation: 100% Stocks, 0% Bonds**

* **Scenario 2: A Crisis Begins**
    * Recent Volatility spikes to **30%**.
    * Calculation: `15% / 30% = 0.5`
    * **Resulting Allocation: 50% Stocks, 50% Bonds**

* **Scenario 3: Extreme Market Panic**
    * Recent Volatility hits **60%**.
    * Calculation: `15% / 60% = 0.25`
    * **Resulting Allocation: 25% Stocks, 75% Bonds**

---
## What About "RISK-OFF"?

It's important to remember that volatility targeting **only applies when your main crash protection signals say it's "RISK-ON."**

If the Slow Trigger (blended score) or Fast Trigger (100d SMA + 6m return) fails, the system overrides the volatility calculation and moves to the fixed **50% Bonds / 50% Gold** defensive allocation.
