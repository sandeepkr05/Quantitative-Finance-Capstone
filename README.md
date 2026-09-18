# When Not to Trade: Liquidity-Aware Abstention for Intraday Nifty 50 ETF Trading

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-orange.svg)](https://xgboost.readthedocs.io/)
[![Quant Research](https://img.shields.io/badge/Domain-Quantitative_Finance-success.svg)]()

> **A quantitative research project investigating whether a trading system can benefit from explicitly deciding when *not* to trade.**

---

## 📌 Overview

Many machine-learning trading systems assume that every prediction should result in an executed trade. However, a model can have reasonable classification performance while still failing to generate profitable signals after transaction costs, slippage, and changing market conditions.

This project investigates a different question:

> **"Even if a machine-learning model produces a signal, does the system know when it should abstain?"**

To investigate this, the project combines an **XGBoost directional classifier** with a programmatic **4-action Decision Controller**:

* `BUY`
* `SELL`
* `WAIT`
* `REDUCE SIZE`

The controller uses:

1. **Prediction confidence** to determine whether a signal should be acted upon.
2. **Trading volume** as a proxy for market liquidity.
3. **Position sizing** to reduce exposure during lower-volume conditions.

The system is evaluated using an **out-of-sample chronological backtest** on hourly NIFTYBEES data.

---

## 🎯 Research Question

The primary research question is:

> **Can confidence-based abstention and volume-aware position sizing reduce drawdown and capital losses when the underlying machine-learning trading model has weak directional performance?**

A secondary question is:

> **Does the proposed decision controller provide value beyond simply reducing market exposure?**

The second question is evaluated using an **exposure-matched Random Abstention baseline**.

---

## 🔬 Research Hypothesis

### H₁ — Alternative Hypothesis

A decision controller that abstains from low-confidence predictions and reduces exposure under low-volume conditions can reduce drawdown and capital losses relative to unrestricted execution, even when the underlying predictive model has weak directional performance.

### H₀ — Null Hypothesis

The decision controller does not provide a meaningful improvement over unrestricted execution or equivalent random exposure reduction.

---

# ⚙️ System Architecture

```text
                  Historical OHLCV Data
                          │
                          ▼
                 Feature Engineering
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
       XGBoost Classifier       Volume Analysis
              │                       │
              ▼                       ▼
       Prediction Probability    MA20 Volume
              │                       │
              └───────────┬───────────┘
                          ▼
                 Decision Controller
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
        WAIT          REDUCE SIZE      BUY / SELL
     Position = 0    Position = 0.5    Position = 1.0
```

---

# 🤖 1. Predictive Model

### Asset

**NIFTYBEES.NS — Nifty 50 ETF**

### Timeframe

**60-minute / hourly bars**

### Dataset

Approximately **two years of historical intraday data**, with a chronological 80/20 train-test split.

The unseen out-of-sample test period covers:

**March 30, 2026 – August 21, 2026**

### Features

The model uses:

* Open
* High
* Low
* Close
* Volume
* Average True Range (ATR)
* Relative Strength Index (RSI)
* Relative Volume Proxy

### Algorithm

**XGBoost Binary Classifier**

The model predicts whether the next hourly closing price is higher than the current close.

The target is:

$$
y_t =
\begin{cases}
1 & \text{if } Close_{t+1} > Close_t \\
0 & \text{otherwise}
\end{cases}
$$

---

# 🧠 2. Prediction Confidence Filter

The model's class probabilities are used to determine whether a prediction has sufficient confidence for execution.

A static threshold of:

**0.55**

was selected before evaluating the out-of-sample test set.

The threshold was therefore not tuned using test-set performance.

### Decision

```text
Confidence <= 0.55
        ↓
      WAIT
Position Size = 0
```

---

# 💧 3. Volume-Based Liquidity Proxy

Actual order-book liquidity data was not used.

Instead, the project uses trading volume as a **proxy for market liquidity**.

A 20-period moving average of volume is calculated:

$$
MA_{20}(Volume)
$$

Current volume is then compared against this moving average.

```text
Volume > MA20
     ↓
Higher-volume condition
     ↓
Full position size
```

If volume is below the moving average, exposure is reduced.

---

# 🎛️ 4. Four-Action Decision Controller

The controller transforms the binary XGBoost prediction into one of four actions.

```python
IF confidence <= 0.55:
    WAIT
    Position Size = 0

ELSE IF Volume_t > MA20(Volume):
    BUY / SELL
    Position Size = 1.0

ELSE:
    REDUCE SIZE
    Position Size = 0.5
```

### Actions

| Condition                     | Action        | Position Size |
| ----------------------------- | ------------- | ------------: |
| Low confidence                | `WAIT`        |            0% |
| High confidence + high volume | `BUY / SELL`  |          100% |
| High confidence + low volume  | `REDUCE SIZE` |           50% |

---

# ⏱️ Execution & Leakage Prevention

To prevent look-ahead bias:

1. Features at time \(t\) use only information available at or before the candle close.
2. The model generates a signal at the close of candle \(t\).
3. The resulting position is executed at the **\(t+1\) open**.
4. A simplified **0.05% round-trip transaction cost** is applied.

The Buy & Hold benchmark uses the same start and end execution timestamps as the algorithmic strategies.

---

# 📊 Results

## Machine Learning Performance

The chronological out-of-sample test showed that the underlying XGBoost classifier did **not** establish a meaningful directional edge.

| Metric                  |     Result |
| ----------------------- | ---------: |
| Accuracy                | **49.13%** |
| ROC-AUC                 | **0.5066** |
| Majority-Class Baseline | **51.46%** |

The model therefore performed below the naive majority-class baseline on the unseen test period.

---

## Trading Performance

| Strategy                 | Total Return | Max Drawdown |     Sharpe | Trades |   Exposure | Active-Bar Win Rate |
| ------------------------ | -----------: | -----------: | ---------: | -----: | ---------: | ------------------: |
| Buy & Hold               |  **+21.62%** |  **-15.97%** |   **1.19** |    N/A |       100% |                 N/A |
| XGBoost — Always Trade   |  **-28.02%** |  **-28.99%** | **-11.75** |    311 |       100% |              48.29% |
| Full Decision Controller |  **-14.58%** |  **-15.10%** |  **-9.93** |    442 | **39.70%** |              31.29% |

### Interpretation

Relative to unrestricted XGBoost execution, the Decision Controller reduced:

* **Total loss by approximately 48%**
* **Maximum drawdown by approximately 48%**

However, the controller remained unprofitable.

This demonstrates that exposure control reduced the damage caused by the underlying directional model, but does not demonstrate that the controller generated profitable alpha.

---

# 🎲 Exposure-Matched Random Abstention Baseline

An important question is whether the controller's improvement came from **intelligent filtering** or simply from **being in the market less often**.

To investigate this, an exposure-matched Random Abstention baseline was constructed with the same:

**39.70% market exposure**

as the Decision Controller.

### Result

| Strategy            | Market Exposure | Total Return |
| ------------------- | --------------: | -----------: |
| Decision Controller |          39.70% |  **-14.58%** |
| Random Abstention   |          39.70% |   **-3.12%** |

In this simulation, the random exposure-reduction strategy preserved more capital than the proposed controller.

### Key Finding

> **The controller reduced risk relative to unrestricted execution, but the current experiment does not demonstrate that its confidence and volume signals identify unfavorable trades better than equivalent random exposure reduction.**

This suggests that **reduced market exposure**, rather than demonstrated predictive interception, was the dominant source of risk reduction in the current implementation.

---

# 📈 Key Findings

### 1. The underlying ML model did not produce directional alpha

The XGBoost classifier achieved:

**49.13% accuracy and 0.5066 ROC-AUC**

on the unseen test data.

### 2. The Decision Controller reduced losses

The controller reduced the unrestricted strategy's:

* Total loss from **28.02% → 14.58%**
* Maximum drawdown from **28.99% → 15.10%**

### 3. Exposure reduction was important

The controller reduced market exposure to:

**39.70%**

compared with 100% exposure for the unrestricted strategy.

### 4. Intelligent abstention was not demonstrated

The exposure-matched random baseline achieved **-3.12%** in the tested simulation, compared with **-14.58%** for the controller.

Therefore, the current evidence does not establish that the controller's confidence mechanism is superior to simply reducing exposure.

### 5. The negative result is itself informative

The experiment highlights an important distinction between:

> **Risk reduction through lower exposure**

and

> **Risk reduction through accurately identifying unfavorable predictions.**

The current implementation demonstrates the former, but not yet the latter.

---

# ⚠️ Limitations

This project has several important limitations:

* The underlying XGBoost model did not generate predictive alpha.
* The experiment uses a single ETF and hourly timeframe.
* Volume is used as a proxy for liquidity rather than direct order-book measurements.
* The transaction-cost model is simplified.
* The 0.55 confidence threshold is static.
* The initial Random Abstention comparison is based on a single simulation.
* The current experiment does not establish generalization across different instruments, market regimes, or transaction-cost environments.
* The controller's ability to identify unfavorable trades has not yet been demonstrated beyond exposure reduction.

---

# 🔭 Future Work

## 1. Repeated Random-Abstention Analysis

Run **1,000+ exposure-matched random simulations** and compare:

* Mean return
* Median return
* Standard deviation
* 5th/95th percentile returns
* Maximum drawdown distribution
* Sharpe ratio distribution
* Controller percentile

This will determine whether the controller performs differently from the distribution expected under random exposure reduction.

---

## 2. Probability Calibration

Apply calibration techniques such as:

* Platt Scaling
* Isotonic Regression

The goal is to determine whether calibrated probabilities improve abstention performance relative to the current controller and random baseline.

---

## 3. Ablation Studies

Test the individual components separately:

```text
XGBoost
   │
   ├── Confidence Only
   │
   ├── Liquidity Only
   │
   └── Confidence + Liquidity
```

This will help identify which component, if any, contributes incremental value.

---

## 4. Transaction-Cost Sensitivity

Evaluate the strategy under different transaction-cost assumptions:

* 0.00%
* 0.025%
* 0.05%
* 0.10%

This will help determine how sensitive the observed results are to execution friction.

---

# 🛠️ Tech Stack

* **Python**
* **Pandas**
* **NumPy**
* **XGBoost**
* **Scikit-learn**
* **Matplotlib**
* **Technical indicators**
* **Quantitative backtesting**

---

---

# 📚 References

1. Krauss, C., Do, X. A., & Huck, N. (2017). *Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500*. European Journal of Operational Research, 259(2), 689–702.

2. Novy-Marx, R., & Velikov, M. (2016). *A taxonomy of anomalies and their trading costs*. The Review of Financial Studies, 29(1), 104–147.

3. López de Prado, M. (2018). *Advances in Financial Machine Learning*. John Wiley & Sons.

4. Chordia, T., Roll, R., & Subrahmanyam, A. (2001). *Market liquidity and trading activity*. The Journal of Finance, 56(2), 501–530.

---

# 👤 Author

**Sandeep Kumar**

YEL FinanceMeta Summer Cohort '26

---

## ⚠️ Disclaimer

This repository is intended for **educational and research purposes only**.

The strategies, models, backtests, and results presented here do not constitute financial advice or a recommendation to buy or sell any security.

Backtested performance does not guarantee future results. Actual trading performance may differ substantially due to market conditions, execution quality, liquidity, slippage, transaction costs, and other factors.
