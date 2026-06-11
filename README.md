# NVIDIA (NVDA) Stock Price Forecasting
### 5-Day Ahead Price Prediction Using Multi-Model Ensemble

> **Academic Project** — Machine Learning in Python, SGH Warsaw School of Economics, June 2026  
> **Team QuantX** — Muhammad Haseeb Qamar · Dariia Snisarenko

---

## 📌 Project Overview

This project was built as a graded academic assignment with one goal: forecast NVIDIA's closing stock price for **June 8–12, 2026** as accurately as possible using machine learning.

Rather than copying a single tutorial model, we designed a full professional-grade pipeline from scratch — combining four distinct model architectures, financial NLP sentiment analysis, and a stationary return-based forecasting framework. The project was developed iteratively over several days, with each design decision made deliberately and understood conceptually by both team members.

**This is student work. It was built to learn, not to trade.**

---

## 🎯 Final Forecasts

| Date | Day | LightGBM | XGBoost | LSTM+Attn | TFT | **Ensemble** |
|------|-----|----------|---------|-----------|-----|-------------|
| 2026-06-08 | Monday | $205.06 | $205.36 | $204.72 | $204.96 | **$205.02** |
| 2026-06-09 | Tuesday | $205.15 | $205.45 | $204.32 | $204.81 | **$204.92** |
| 2026-06-10 | Wednesday | $205.23 | $205.53 | $203.91 | $204.64 | **$204.81** |
| 2026-06-11 | Thursday | $205.32 | $205.61 | $203.47 | $204.47 | **$204.69** |
| 2026-06-12 | Friday | $205.40 | $205.70 | $202.97 | $204.30 | **$204.54** |

**Starting anchor:** $205.10 (June 5, 2026 close)  
**Implied 5-day change:** −0.3% (conservative, reflects bearish recent momentum)

---

## 🏗️ Architecture

### The Core Design Decision — Stationary Return Framework

Most student forecasting projects predict price directly. We predict **daily percentage returns** instead, then reconstruct prices:

```
Predicted_Close = Prior_Close × (1 + Predicted_Return)
```

This is the correct professional approach. Predicting prices directly causes models to learn a lazy pattern — "tomorrow will be similar to today" — because that is technically accurate most of the time. Predicting returns forces the model to find real signals in price movement patterns.

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
           │              │             │              │
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

Gradient boosted decision trees are the gold standard for tabular financial data. We used Microsoft's LightGBM with Optuna hyperparameter tuning (50 trials) and 5-fold walk-forward time-series cross-validation.

Key design choices:
- Trains on **280 features** (60 base technical indicators × 4 structural lag copies)
- Structural lag matrix gives the model a "memory" of the past 4 trading days in a single row
- Walk-forward CV prevents any data leakage — future data is never used to evaluate past predictions

**Validation performance:** MAE = $3.80 | RMSE = $4.96 | R² = 0.9145

### Model 2 — XGBoost (Weight: 20%)

Independent gradient boosting model from a different codebase (University of Washington origin vs Microsoft). Identical feature set but different tree-growth strategy (level-wise vs leaf-wise). Used primarily for ensemble diversity — two models making independent errors average to something more robust than either alone.

**Validation performance:** MAE = $3.84 | RMSE = $5.03 | R² = 0.9121

### Model 3 — LSTM with Bahdanau Attention (Weight: 20%)

2-layer Long Short-Term Memory network with a custom attention mechanism. Input: 30-day sliding windows of 14 normalized features.

Architecture:
```
Input (30 × 14) → LSTM (128 units, 2 layers) → Attention → FC(128→64→1) → Return
```

The attention layer learns which of the 30 past days are most relevant for each prediction step — rather than weighting all days equally. Trained with HuberLoss (robust to outlier trading days), early stopping, and ReduceLROnPlateau scheduling.

Early stopped at epoch 51. Best validation loss: 0.002496.

**Validation performance:** MAE = $3.78 | RMSE = $4.97 | R² = 0.9140

### Model 4 — Temporal Fusion Transformer / TFT-Lite (Weight: 30%)

Custom TFT-inspired architecture with two components not found in standard tutorials:

**UpgradedVariableSelectionNetwork:** Learns per-timestep dynamic feature weights. On each trading day, it generates a separate importance score for all 14 input features — RSI might get 0.4 weight on a day near overbought territory and 0.1 on a normal day. This is dynamic, not fixed.

**TransformerEncoder:** 4-head self-attention across the 30-day sequence. Unlike LSTM which processes days sequentially, the Transformer computes relationships between every pair of days simultaneously — day 1 can directly interact with day 28 without passing through 27 intermediate steps.

Trained with AdamW optimizer, HuberLoss, and gradient clipping.

**Validation performance:** MAE = $3.79 | RMSE = $4.99 | R² = 0.9133

### NLP Signal — FinBERT Sentiment (Feature for all models)

`ProsusAI/finbert` — a BERT model fine-tuned specifically on financial news — was applied to recent NVIDIA headlines from Yahoo Finance. Each headline is scored as positive (+confidence), negative (−confidence), or neutral (0.0). The mean daily sentiment is added as a feature to all models.

At submission time, mean sentiment was **−0.1824** (mildly bearish), consistent with the recent NVDA price decline from $235 peak.

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

For tree models, each base feature is additionally lagged 1/2/3/4 steps → **280 total features** with full structural memory.

---

## 🔁 Recursive Multi-Step Forecasting

We have no real data for June 8–12. So each prediction feeds back as input for the next:

```
Real data (Jan 2024 → Jun 5) → Predict Jun 8 return → Reconstruct Jun 8 price
Append Jun 8 prediction      → Predict Jun 9 return → Reconstruct Jun 9 price
Append Jun 9 prediction      → Predict Jun 10 ...
```

Error accumulates across steps — day 5 is inherently less reliable than day 1. This is honest multi-step forecasting, not cheating with future data.

---

## ✅ Validation Strategy

**Walk-forward TimeSeriesSplit (5 folds)** — the correct approach for time-series data.

```
Fold 1: Train [Jan 2024 → Jul 2024]  |  Validate [Aug 2024 → Oct 2024]
Fold 2: Train [Jan 2024 → Oct 2024]  |  Validate [Nov 2024 → Jan 2025]
Fold 3: Train [Jan 2024 → Jan 2025]  |  Validate [Feb 2025 → Apr 2025]
...
```

Random shuffling (what most beginners do) would allow the model to train on future data and evaluate on the past — that is data leakage and produces falsely optimistic metrics. Walk-forward validation mimics real deployment conditions.

---

## 📈 Data

- **Source:** Yahoo Finance via `yfinance`
- **Period:** January 2, 2024 → June 5, 2026
- **Rows:** 609 trading days (560 after feature engineering warmup)
- **Train/Test split:** 85% / 15% chronological (no shuffling)
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

**To forecast a different date range:**
- Change `DATA_END_DATE` in Cell 2
- Change `end` to `DATA_END_DATE + 1 day`
- Run all cells — forecasts auto-generate for the next 5 business days

**Runtime:** ~20-25 minutes on CPU (LSTM + TFT training). Use GPU for faster results.

---

## ⚠️ Academic Honesty Statement

This project was built as a **conceptual learning exercise** for an MSc-level Machine Learning course. Both team members understood the pipeline architecture and the reasoning behind each design decision, even where implementation was iterative and guided.

**What we built and understand:**
- Why stationary return prediction outperforms direct price prediction
- How walk-forward cross-validation prevents data leakage
- What gradient boosting does and why Optuna finds better hyperparameters than grid search
- How LSTM memory cells work and what the attention mechanism adds
- How the TFT Variable Selection Network creates dynamic per-timestep feature weights
- Why ensembling independent models produces more robust predictions than any single model

**What this project is not:**
- A trading system (do not use this to make financial decisions)
- A claim that stock prices are predictable in the short term (they largely are not)
- Independent research — it is a graded university assignment

The core honest truth about 5-day stock forecasting is that no model can do it reliably. The market is fundamentally noisy at short horizons. This project was evaluated on methodology quality, not prediction accuracy — and that is the right way to assess it.

---

## 📁 Repository Structure

```
nvda-forecast-2026/
│
├── ML_Project.ipynb          # Main notebook — all 36 cells
├── README.md                 # This file
└── images/                   # Charts generated by the notebook
    ├── nvda_price_chart.png
    ├── correlation_heatmap.png
    ├── lgb_feature_importance.png
    ├── lstm_training_curve.png
    ├── model_comparison.png
    └── nvda_final_forecast.png
```

---

## 👤 Author

**Muhammad Haseeb Qamar**  
MSc Advanced Analytics & Big Data — SGH Warsaw School of Economics  
[LinkedIn](https://linkedin.com/in/YOUR_PROFILE) · [GitHub](https://github.com/YOUR_USERNAME)

---

*Built June 2026 as part of Machine Learning in Python coursework at SGH Warsaw.*
