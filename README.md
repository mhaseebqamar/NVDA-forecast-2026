# NVIDIA (NVDA) 5-Day Close Price Forecast with a Multi-Model Ensemble

An individual Machine Learning in Python project (SGH Warsaw School of Economics, MSc Advanced Analytics & Big Data, June 2026). It forecasts NVDA's daily closing price for 8–12 June 2026 using an ensemble of LightGBM, XGBoost, an LSTM with attention and a TFT-inspired Transformer, with FinBERT news sentiment as an extra feature. The project was submitted under the team alias "QuantX"; I was the only member.

## Project context

This was a graded forecasting assignment: each student submitted a 5-day price forecast made before the forecast week. It was developed iteratively and with guidance, and the focus was on understanding why each design choice was made (stationary return targets, walk-forward cross-validation, recurrent vs attention-based sequence models, ensembling), not on building a trading system. Short-term stock prices are largely unpredictable, and the results below should be read with that in mind.

## What this project does

* Uses 609 trading days of NVDA daily OHLCV data (2 Jan 2024 – 5 Jun 2026, Yahoo Finance), stored in the notebook as a frozen dataset for reproducibility
* Engineers technical indicators with the `ta` library (trend, momentum, volatility, volume, price-derived, rolling statistics, calendar and ratio features)
* Predicts the **next-day percentage return** rather than the price, then rebuilds the price as `Predicted_Close = Prior_Close × (1 + Predicted_Return)`
* Trains four models on a chronological 85/15 split (471 training days, 84 hold-out days):

| Model | Setup | Ensemble weight |
|---|---|---|
| LightGBM | 280 features (56 base features plus 4 daily lags of each); Optuna, 50 trials, 5-fold walk-forward `TimeSeriesSplit` | 30% |
| XGBoost | Same features; separate Optuna study (50 trials, walk-forward CV) | 20% |
| LSTM + attention (PyTorch) | 30-day windows of 14 scaled features; 2-layer LSTM (128 units) + Bahdanau-style attention; Huber loss, ReduceLROnPlateau, early stopping; 214,274 parameters | 20% |
| TFT-inspired Transformer (PyTorch) | Same windows; per-timestep variable selection network + 2-layer Transformer encoder (4 heads, d_model 128); Huber loss, AdamW; 671,375 parameters | 30% |

* Adds a FinBERT (`ProsusAI/finbert`) sentiment score from recent NVIDIA-related Yahoo Finance headlines as a feature
* Produces the 5-day forecast recursively: each predicted day is fed back in as input for the next day

## Final forecast (as submitted)

| Date | Day | Ensemble forecast (USD) |
|---|---|---|
| 2026-06-08 | Monday | 205.03 |
| 2026-06-09 | Tuesday | 204.94 |
| 2026-06-10 | Wednesday | 204.85 |
| 2026-06-11 | Thursday | 204.75 |
| 2026-06-12 | Friday | 204.66 |

Starting point: $205.10 (close on 5 June 2026). Implied 5-day change: −0.2%. Individual model forecasts for the week ranged from $203.04 to $205.70.

## Results on the 84-day hold-out period

Errors are measured on the rebuilt next-day price in USD.

| Model | MAE ($) | RMSE ($) | R² |
|---|---|---|---|
| LSTM + attention | 3.78 | 4.97 | 0.914 |
| LightGBM | 3.80 | 4.96 | 0.915 |
| TFT-inspired Transformer | 3.81 | 5.00 | 0.913 |
| XGBoost | 3.84 | 5.03 | 0.912 |

**How to read these numbers**

* **All four models are almost identical.** MAE ranges from $3.78 to $3.84, and every model forecast the week as roughly flat. In practice the models learned to predict next-day returns close to zero.
* **The high R² comes mostly from the previous day's close, not from the models.** The predicted price is built on top of the prior close, which already explains most of the variation in the next day's price. A naive "tomorrow equals today" forecast would score a similar R² on the same window, and the notebook does not include that baseline.
* **In return terms**, the tuned tree models' cross-validated error was about 2.2% per day (LightGBM 2.19%, XGBoost 2.18%), which is close to the typical size of a daily NVDA move in this period.
* **The FinBERT signal had very little effect.** Only 10 recent headlines were available. Several were not about NVIDIA, and only one of the eight shown got a non-neutral score (mean +0.06, neutral). Because news only covered the last few days, the sentiment feature is zero for almost all historical rows.

## Tech stack

* Python
* Data: yfinance, pandas, NumPy
* Features: `ta` (technical analysis)
* Tree models: LightGBM, XGBoost
* Tuning: Optuna (TPE sampler), scikit-learn `TimeSeriesSplit`
* Deep learning: PyTorch
* NLP: Hugging Face `transformers` (ProsusAI/finbert)
* Evaluation: scikit-learn
* Visualisation: matplotlib, seaborn

## Project structure

```
NVDA-forecast-2026/
├── ML_Project.ipynb     # Full notebook: data → features → FinBERT → 4 models → ensemble forecast
└── README.md
```

## How to run

```bash
git clone https://github.com/mhaseebqamar/NVDA-forecast-2026.git
cd NVDA-forecast-2026
pip install yfinance lightgbm xgboost optuna torch transformers ta scikit-learn pandas numpy matplotlib seaborn
jupyter notebook ML_Project.ipynb
```

Run all cells from top to bottom. With `DOWNLOAD_LIVE = False`, the price data comes from the frozen dataset in the notebook. The FinBERT cell still downloads the current Yahoo Finance headlines, so the sentiment value, and therefore the forecasts, can differ slightly from the submitted run.

## Honest limitations

* **The hold-out period was not fully independent.** It was also used for early stopping (LightGBM, XGBoost, LSTM) and, for the LSTM and Transformer, the feature scaler was fitted on the full dataset. The hold-out metrics are therefore somewhat optimistic.
* **No naive baseline and no directional accuracy.** Price-level MAE and R² alone do not show whether the models beat a simple random-walk forecast.
* **The ensemble itself was not evaluated.** Weights were set by hand (30/20/20/30), and only the individual models were scored on the hold-out period.
* **The Transformer is TFT-inspired, not a full Temporal Fusion Transformer.** It keeps the variable selection idea and self-attention, but leaves out gating layers, static covariates and quantile outputs. It was trained for a fixed 50 epochs without validation-based stopping.
* **Small dataset.** About 470 training days is very little for deep sequence models with hundreds of thousands of parameters.
* **Recursive forecasting compounds error**, so day 5 is less reliable than day 1.
* This is a university assignment, not a trading system, and should not be used for investment decisions.
