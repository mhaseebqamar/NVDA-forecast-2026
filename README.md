# NVIDIA (NVDA) Stock Price Forecasting
### 5-Day Ahead Price Prediction Using Multi-Model Ensemble

> **Academic Project** — Machine Learning in Python, SGH Warsaw School of Economics, June 2026  
> **Team QuantX** — Muhammad Haseeb Qamar

---

## 📌 Project Overview

This project was built as a graded academic assignment with one goal: forecast NVIDIA's closing stock price for **June 8–12, 2026** as accurately as possible using machine learning.

Rather than copying a single tutorial model, we designed a full professional-grade pipeline — combining four distinct model architectures, financial NLP sentiment analysis, and a stationary return-based forecasting framework. The project was developed iteratively, with each design decision made deliberately and understood conceptually.

**This is student work. It was built to learn, not to trade.**

---

## 🎯 Final Forecasts (Ensemble)

| Date | Day | Predicted Close (USD) |
|------|-----|-----------------------|
| 2026-06-08 | Monday | **$205.03** |
| 2026-06-09 | Tuesday | **$204.94** |
| 2026-06-10 | Wednesday | **$204.85** |
| 2026-06-11 | Thursday | **$204.75** |
| 2026-06-12 | Friday | **$204.66** |

**Starting anchor:** $205.10 (June 5, 2026 close)  
**Implied 5-day change:** −0.2%  
**5-day average forecast:** $204.85

---

## 🏗️ Architecture

### The Core Design Decision — Stationary Return Framework

Most student forecasting projects predict price directly. We predict **daily percentage returns** instead, then reconstruct prices:

```
Predicted_Close = Prior_Close × (1 + Predicted_Return)
```

This is the correct professional approach. Predicting prices directly causes models to learn a lazy pattern — "tomorrow will be similar to today." Predicting returns forces the model to find real signals in price movement patterns.

### Four Models, One Ensemble

```
┌─────────────────────────────────────────────────────────┐
│                   INPUT: OHLCV + Features               │
└──────────┬──────────────┬──────────────┬────────────────┘
           │              │              │              │
     ┌─────▼──────┐ ┌─────▼──────┐ ┌───▼──────┐ ┌────▼────┐
     │ LightGBM   │ │  XGBoost   │ │  LSTM +  │ │   TFT   │
     │  (30%)     │ │  (20%)     │ │Attention │ │  (30%)  │
     │            │ │            │ │  (20%)   │ │         │
     └─────┬──────┘ └─────┬──────┘ └───┬──────┘ └────┬────┘
           └──────────────┴─────────────┴──────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Weighted Ensemble   │
                    │  Recursive 5-Step     │
                    │  Price Reconstruction │
                    └───────────────────────┘
```

---

## 📊 Model Details

### Model 1 — LightGBM (Weight: 30%)

Gradient boosted decision trees — the gold standard for tabular financial data. Hyperparameter tuned with Optuna (50 trials) and 5-fold walk-forward time-series cross-validation.

- Trains on **280 features** (60 base technical indicators × 4 structural lag copies)
- Structural lag matrix gives the model a full trading week of memory per row
- Walk-forward CV prevents any data leakage

**Validation:** MAE = $3.7962 | RMSE = $4.9563 | R² = 0.9145

### Model 2 — XGBoost (Weight: 20%)

Independent gradient boosting model with separate Optuna tuning. Same feature set, different tree-growth strategy (level-wise vs leaf-wise). Used for ensemble diversity — two models making independent errors average to something more robust than either alone.

**Validation:** MAE = $3.8351 | RMSE = $5.0252 | R² = 0.9121

### Model 3 — LSTM with Bahdanau Attention (Weight: 20%)

2-layer Long Short-Term Memory network with a custom attention mechanism. Input: 30-day sliding windows of 14 normalized features.

```
Input (30 × 14) → LSTM (128 units, 2 layers) → Attention → FC(128→64→1) → Return
```

The attention layer learns which of the 30 past days are most relevant for each prediction — rather than weighting all days equally. Trained with HuberLoss, early stopping, and ReduceLROnPlateau scheduling. Early stopped at epoch 51, best val loss: 0.002496.

**Validation:** MAE = $3.7810 | RMSE = $4.9695 | R² = 0.9140

### Model 4 — Temporal Fusion Transformer / TFT-Lite (Weight: 30%)

Custom TFT-inspired architecture with two components not found in standard tutorials:

**UpgradedVariableSelectionNetwork:** Learns per-timestep dynamic feature weights. On each trading day it generates a separate importance score for all 14 input features — RSI might get 0.4 weight on a day near overbought territory and 0.1 on a normal day.

**TransformerEncoder:** 4-head self-attention across the 30-day sequence. Unlike LSTM which processes days sequentially, the Transformer computes relationships between every pair of days simultaneously.

**Validation:** MAE = $3.8111 | RMSE = $5.0027 | R² = 0.9128

### NLP Signal — FinBERT Sentiment (Feature for all models)

`ProsusAI/finbert` applied to recent NVIDIA headlines from Yahoo Finance. Each headline scored as positive (+confidence), negative (−confidence), or neutral (0.0). Mean sentiment at submission time: **+0.0609 (Neutral)**.

---

## 📊 Validation Performance Summary

| Model | MAE ($) | RMSE ($) | R² | Weight |
|-------|---------|----------|----|--------|
| LightGBM | 3.7962 | 4.9563 | 0.9145 | 30% |
| XGBoost | 3.8351 | 5.0252 | 0.9121 | 20% |
| LSTM + Attention | 3.7810 | 4.9695 | 0.9140 | 20% |
| TFT | 3.8111 | 5.0027 | 0.9128 | 30% |

---

## 📐 Feature Engineering

**60 base features** computed from OHLCV data using the `ta` library:

| Category | Features |
|---|---|
| Trend | EMA(9/21/50), SMA(20/50), MACD + Signal + Histogram, ADX |
| Momentum | RSI(7/14), Stochastic(K/D), ROC(5/10) |
| Volatility | Bollinger Bands (upper/lower/width/%B), ATR(14) |
| Volume | OBV, VWAP, Volume ratio vs 20-day average |
| Price-derived | Daily return, log return, High-Low%, Open-Close% |
| Rolling stats | Mean/Std/Max/Min over 5/10/20 day windows |
| Calendar | Day of week, Month, Quarter |
| Ratio | Price vs EMA50, Price vs SMA20 |
| NLP | FinBERT news sentiment score |

For tree models, each base feature is additionally lagged 1/2/3/4 steps → **280 total features**.

---

## 🔁 Recursive Multi-Step Forecasting

```
Real data (Jan 2024 → Jun 5) → Predict Jun 8 return → Reconstruct Jun 8 price
Append Jun 8 prediction      → Predict Jun 9 return → Reconstruct Jun 9 price
Append Jun 9 prediction      → Predict Jun 10 ...
```

Error accumulates across steps — day 5 is inherently less reliable than day 1. This is honest multi-step forecasting, not cheating with future data.

---

## ✅ Validation Strategy

**Walk-forward TimeSeriesSplit (5 folds)** — the only correct approach for time-series data. Random shuffling would allow the model to train on future data and evaluate on the past, producing falsely optimistic metrics.

---

## 📈 Data

- **Source:** Yahoo Finance via `yfinance`
- **Period:** January 2, 2024 → June 5, 2026
- **Rows:** 609 trading days (560 after feature engineering warmup)
- **Train/Test split:** 85% / 15% chronological
- **Forecast anchor:** June 5, 2026 close = $205.10

---

## 🧰 Tech Stack

```
Python 3.10+
├── Data:          yfinance, pandas, numpy
├── Features:      ta (Technical Analysis library)
├── Tree models:   lightgbm, xgboost
├── Tuning:        optuna
├── Deep learning: torch, torch.nn
├── NLP:           transformers (ProsusAI/finbert)
├── Evaluation:    scikit-learn
└── Visualization: matplotlib, seaborn
```

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/nvda-forecast-2026.git
cd nvda-forecast-2026

# 2. Install dependencies
pip install yfinance lightgbm xgboost optuna torch transformers ta scikit-learn pandas numpy matplotlib seaborn

# 3. Open notebook
jupyter notebook ML_Project.ipynb

# 4. Run all cells top to bottom
# Final predictions appear in Cell 36
```

**Note:** `DOWNLOAD_LIVE = False` in Cell 5 uses frozen historical data for reproducibility. Set to `True` to download fresh data from Yahoo Finance.

**Runtime:** ~20-25 minutes on CPU. Use GPU for faster results.

---

## ⚠️ Academic Honesty Statement

This project was built as a **conceptual learning exercise** for an MSc-level Machine Learning course. It was developed iteratively and guided, with the author understanding the pipeline architecture and the reasoning behind each design decision.

**What was understood and learned:**
- Why stationary return prediction outperforms direct price prediction
- How walk-forward cross-validation prevents data leakage
- What gradient boosting does and why Optuna finds better hyperparameters than grid search
- How LSTM memory cells work and what the attention mechanism adds
- How the TFT Variable Selection Network creates dynamic per-timestep feature weights
- Why ensembling independent models produces more robust predictions than any single model

**What this project is not:**
- A trading system — do not use this to make financial decisions
- A claim that short-term stock prices are reliably predictable — they largely are not
- Independent research — it is a graded university assignment

---

## 📁 Repository Structure

```
nvda-forecast-2026/
│
├── ML_Project.ipynb     # Main notebook — 36 cells
└── README.md            # This file
```

---

## 👤 Author

**Muhammad Haseeb Qamar**  
MSc Advanced Analytics & Big Data — SGH Warsaw School of Economics  
[LinkedIn](https://linkedin.com/in/YOUR_PROFILE) · [GitHub](https://github.com/YOUR_USERNAME)

---

*Built June 2026 as part of Machine Learning in Python coursework at SGH Warsaw.*
