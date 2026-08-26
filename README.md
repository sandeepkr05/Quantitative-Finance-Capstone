# Quantitative-Finance-Capstone
# When Not to Trade: Liquidity-Aware Uncertainty Abstention 

**Author:** Sandeep Kumar  
**Program:** YEL FinanceMeta Summer Cohort '26  

## Project Overview
This repository contains the codebase for my quantitative finance capstone project. It investigates whether a machine learning trading system can improve risk-adjusted performance by explicitly determining when a prediction is not economically viable to execute. 

While traditional quantitative models focus solely on prediction accuracy (outputting strict `BUY` or `SELL` signals), this study implements a **4-action decision controller (`BUY`, `SELL`, `WAIT`, `REDUCE SIZE`)** that integrates predictive uncertainty and liquidity proxies. The goal is to prove that filtering trades based on model confidence and market friction can protect portfolio capital during unfavorable regimes.

## System Architecture
The framework is built in Python and operates in four sequential stages:
1. **Primary Predictive Model:** An XGBoost classifier trained on OHLC data, Average True Range (ATR), Relative Strength Index (RSI), Volume, and a Relative Volume proxy.
2. **Uncertainty Estimator:** A strict confidence threshold set at 55%.
3. **Liquidity Filter:** A moving 20-period average of trading volume to ensure current market conditions can support execution without severe slippage.
4. **Decision Controller:** An algorithmic routing system that overrides the raw ML model:
   - **BUY / SELL:** High confidence + High liquidity
   - **REDUCE SIZE:** High confidence + Low liquidity (Limits slippage impact)
   - **WAIT:** Low confidence (Model stays out of the market)

## Data & Methodology
- **Asset:** Nippon India Nifty 50 ETF (`NIFTYBEES.NS`)
- **Timeframe:** 60-minute intraday (hourly) bars spanning a 2-year period.
- **Evaluation:** Chronological 80/20 train-test split to prevent look-ahead bias, evaluated using a walk-forward backtest on 5 months of unseen data (March 2026 – August 2026).
- **Friction:** A 0.05% transaction cost was hardcoded to simulate real-world execution drag.

## Key Results
The chronological holdout test revealed a critical distinction between raw ML accuracy and structural risk management. 

| Strategy Configuration | Total Return (%) | Max Drawdown (%) | Sharpe Ratio |
| :--- | :--- | :--- | :--- |
| **Buy & Hold (Passive Benchmark)** | 8.10 | -5.46 | 2.95 |
| **XGBoost (Always Trade)** | -28.02 | -28.99 | -11.75 |
| **ML Full Decision Controller** | -14.58 | -15.10 | -9.93 |

**Conclusion:** The baseline XGBoost classifier failed to find a directional edge (achieving only 49.13% accuracy), causing the unrestricted "Always Trade" strategy to suffer a massive -28.99% drawdown. However, the introduction of the 4-action Decision Controller successfully intercepted hundreds of flawed trades. By systematically abstaining from the market during low-conviction setups, the controller successfully cut the unrestricted model's total losses and maximum drawdown nearly in half. 

## Tech Stack
* **Language:** Python 3
* **Data Handling:** `pandas`, `numpy`
* **Machine Learning:** `xgboost`, `scikit-learn`
* **Data Sourcing:** `yfinance`
* **Visualization:** `matplotlib`

## How to Run
1. Clone the repository.
2. Install the required dependencies: `pip install pandas numpy yfinance xgboost scikit-learn matplotlib`
3. Run the Jupyter Notebook or Python script to fetch the latest ETF data, train the model, and output the backtest evaluation metrics and equity curves.