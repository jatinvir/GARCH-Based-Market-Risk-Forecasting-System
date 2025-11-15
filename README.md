# 📈 GARCH-Based Value-at-Risk Forecasting

**Rolling Volatility Forecasts • 1-Day VaR • Statistical Backtesting (UC/IND/CC)**

This project implements a full **Value-at-Risk (VaR) forecasting and backtesting framework** using **GARCH-family volatility models**. It focuses on daily tail-risk estimation for AAPL returns and evaluates model performance using industry-standard statistical tests (Kupiec, Independence, Conditional Coverage).

The project reflects skills used in **quantitative finance, econometrics, statistical modeling, and risk analytics**.

---

## 🔍 Project Overview

Modern financial institutions are required to quantify potential losses under normal and stressed market conditions.
A standard measure is **Value-at-Risk (VaR)**, which estimates the loss level exceeded with (1−α) probability.

This project:

* Computes daily log returns for AAPL (2005–2025)
* Fits GARCH-family models to estimate next-day volatility
* Generates 1-day-ahead **95% VaR forecasts**
* Performs **rolling-window backtesting** using 750-day lookbacks
* Validates model performance using:

  * **Kupiec Unconditional Coverage Test**
  * **Christoffersen Independence Test**
  * **Christoffersen Conditional Coverage Test**

Additionally, the project performs **regime analysis**, demonstrating how COVID-19 introduced structural breaks that classical GARCH volatility models struggle to handle.

---

## 📊 Key Results (2022–2024 Validation Window)

A post-COVID validation window produces the most statistically stable results:

```
GJR-GARCH VaR (95%)  
Breaches: 9 / 219 = 4.11%

Kupiec (UC) p-value:        0.533  → PASS  
Independence (IND) p-value: 0.365  → PASS  
Conditional Coverage p:     0.547  → PASS
```

### ✔ Interpretation

* **UC Test:** breach rate ≈ expected 5% → model accurate
* **Independence Test:** breaches are not clustered → good behavior
* **Conditional Coverage:** jointly passes accuracy + independence → strong fit
* **Conclusion:**
  This GARCH model produces *statistically valid* VaR forecasts on the 2022–2024 regime.

---

## 🧠 Why 2022–2024? (Regime Reasoning)

Rolling GARCH estimation depends heavily on stable volatility dynamics.
However, **2020–2021 includes:**

* COVID crash (−10% to −13% single-day moves)
* Sudden structural breaks in volatility
* Parameter divergence and optimizer non-convergence

GARCH assumes **stationary, slowly evolving volatility**.
COVID breaks that assumption.

### ✔ 2022–2024 is a stable regime:

* Fewer extreme shocks
* Smooth volatility dynamics
* Parameter stability
* Statistically valid VaR backtests

This mirrors professional practice in risk modeling:

> When a regime break occurs, evaluate models on the post-break stable period.

---

## 🧪 Methodology

### 1. **Return Processing**

* Daily log returns
* Scaled to **percent units** to stabilize optimization
* Cleaned and aligned for rolling windows

### 2. **Rolling GARCH Estimation**

* Lookback: **750 daily observations**
* Models tested:

  * **GARCH(1,1)** (baseline)
  * **GJR-GARCH(1,1,1)** (asymmetric volatility response)
  * **EGARCH(1,2)** (logarithmic variance dynamics)
* Distributions:

  * **Student-t**
  * **Skew-t** (optional)

### 3. **Warm-Start Optimization**

To prevent numerical instability:

* Each iteration uses **starting_values = previous_params**
* If a window fails to converge:

  * Retry without warm start
  * Final fallback → carry forward last good forecast

This ensures continuity in VaR predictions.

### 4. **1-Day Ahead VaR Forecasting**

For t-distribution models:

```
VaRα = μₜ + σₜ * t.ppf(α, df=ν)
```

where:

* μₜ = mean forecast
* σₜ = volatility forecast
* ν = degrees of freedom
* α = 0.05

### 5. **Statistical Backtesting**

Implemented manually using likelihood-ratio statistics:

* **Unconditional Coverage (Kupiec)**
* **Independence (Christoffersen)**
* **Conditional Coverage (Christoffersen)**

These tests are part of the **Basel II/III regulatory framework**.

---

## 📈 Visualizations

The project includes the following plots:

### **1. VaR vs Actual Returns**

* Daily returns
* Rolling VaR line
* Red markers showing breaches
* Zero-baseline for clarity

### **2. Rolling 30-Day Breach Rate**

* Shows local behavior
* Compares breach rate to the expected 5%
* Highlights periods of model deviation

Plots are saved under `/results/`.

---

## 🧮 Future Enhancements

* **Expected Shortfall (ES)** estimation and backtesting
* **Regime-switching GARCH (MS-GARCH)** for structural breaks
* **Realized Volatility models (HAR-RV)** using intraday data
* **Multivariate DCC-GARCH** for portfolio VaR
* **Macro-variable-driven volatility forecasting**

---

## ✔ Skills Demonstrated

* Time-series econometrics
* GARCH modeling
* Tail-risk forecasting
* Likelihood-ratio statistical tests
* Python risk modeling
* Quantitative analysis
* Data visualization
* Understanding of market regimes and stability
* Model validation aligned with Basel frameworks

---
