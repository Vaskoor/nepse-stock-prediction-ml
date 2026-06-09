# Stock Market Analysis Using Ensemble Machine Learning

## NBL MAX PROFIT SYSTEM v5.2 — NEPSE Stock Direction Prediction

A machine learning-based stock market analysis system that predicts the direction of NEPSE (Nepal Stock Exchange) stocks using an ensemble of supervised learning algorithms. Built as a college minor project, the system ingests raw OHLCV (Open, High, Low, Close, Volume) data, engineers 50+ technical indicators, applies a Triple-Barrier target labeling methodology, and produces actionable BUY/SELL/WAIT signals with comprehensive evaluation metrics and visualizations.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Output Files](#output-files)
- [Model Details](#model-details)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
- [Configuration & Tuning](#configuration--tuning)
- [Methodology Summary](#methodology-summary)
- [Limitations](#limitations)
- [References](#references)

---

## Project Overview

The Nepal Stock Exchange (NEPSE) is the sole secondary market for trading securities in Nepal. Unlike developed markets, NEPSE lacks institutional-grade analysis tools accessible to retail investors. This project bridges that gap by building a machine learning pipeline that:

1. Reads raw OHLCV stock data from a CSV file (e.g., SANIMA Bank)
2. Engineers 50+ technical features (RSI, MACD, Bollinger Bands, ADX, Stochastic Oscillator, ATR, volume-based indicators, etc.)
3. Labels rows using a **Triple-Barrier / Minimum-Move** target methodology (not naive next-day up/down)
4. Trains a **Soft Voting Ensemble** of Random Forest, Gradient Boosting, XGBoost, and Logistic Regression
5. Evaluates with time-series cross-validation, walk-forward validation, and out-of-sample backtesting
6. Produces 10+ publication-quality visualizations and a full classification report
7. Outputs a final **BUY / SELL / WAIT** prediction for the latest trading day

---

## Features

| Feature | Description |
|---|---|
| **Triple-Barrier Labeling** | Labels target using multi-day forward return with a minimum-move threshold, filtering out ambiguous near-flat rows |
| **50+ Technical Indicators** | RSI, MACD, Bollinger Bands, ADX, Stochastic %K/%D, ATR, SMA/EMA crossovers, volume force, institutional proxy, and more |
| **Ensemble ML Model** | Soft-voting ensemble of RF + GBM + XGBoost + Logistic Regression with tuned weights |
| **Leakage-Free Pipeline** | Scaler fit only on training data; target computed from raw close prices; no future information leaks |
| **Walk-Forward Validation** | Expanding-window validation simulating real-world sequential deployment |
| **Realistic Backtesting** | ATR-based trailing stop-loss, brokerage fees on entry/exit, equity curve vs. buy-and-hold benchmark |
| **10+ Visualizations** | Target distribution, box plots, correlation heatmap, confusion matrices, learning curves, ROC-AUC, feature importance, equity curve |
| **Model Persistence** | Save/load trained models via pickle for reuse without retraining |
| **Daily Prediction** | Actionable BUY/SELL/WAIT signal with confidence score for the latest trading day |

---

## System Architecture

```
SANIMA_cleanedlatest.csv (Raw OHLCV)
        │
        ▼
┌────────────────────────┐
│  Data Acquisition       │  Read CSV, parse dates, sort chronologically
└────────┬───────────────┘
         ▼
┌────────────────────────┐
│  Feature Engineering    │  50+ indicators from OHLCV (backward-looking only)
└────────┬───────────────┘
         ▼
┌────────────────────────┐
│  Target Labeling        │  Triple-Barrier: UP / DOWN / Dropped (ambiguous)
│  (make_target)          │  Horizon = 7 days, Threshold = 1.0%
└────────┬───────────────┘
         ▼
┌────────────────────────┐
│  Train/Val/Test Split   │  70% / 15% / 15% (chronological, no shuffle)
└────────┬───────────────┘
         ▼
┌────────────────────────┐
│  Scaling (RobustScaler) │  Fit on training set ONLY
└────────┬───────────────┘
         ▼
┌────────────────────────┐
│  Model Training         │  VotingClassifier: RF + GB + XGB + LR (soft vote)
│  + Cross-Validation     │  5-Fold TimeSeriesSplit CV on training data
└────────┬───────────────┘
         ▼
┌────────────────────────┐
│  Evaluation & Viz       │  Confusion matrix, ROC-AUC, learning curves, etc.
└────────┬───────────────┘
         ▼
┌────────────────────────┐
│  Backtesting            │  Equity curve, Sharpe ratio, drawdown analysis
└────────┬───────────────┘
         ▼
┌────────────────────────┐
│  Prediction (Today)     │  BUY / SELL / WAIT with confidence percentage
└────────────────────────┘
```

---

## Technologies Used

### Programming Language

| Technology | Version | Purpose |
|---|---|---|
| **Python** | 3.8+ | Core programming language |

### Libraries & Frameworks

| Library | Purpose |
|---|---|
| **NumPy** | Numerical computation, array operations |
| **Pandas** | Data manipulation, CSV I/O, time-series handling |
| **Matplotlib** | Core plotting library for all visualizations |
| **Seaborn** | Statistical heatmaps and enhanced plot styling |
| **scikit-learn** | ML algorithms (RF, GB, LR), preprocessing (RobustScaler), metrics, cross-validation, learning curves |
| **XGBoost** | Extreme Gradient Boosting classifier for the ensemble |
| **pickle** | Model serialization and persistence |
| **JSON** | Metadata storage for model configuration |
| **logging** | Structured console output and status messages |
| **pathlib** | Cross-platform file path handling |

### Machine Learning Algorithms

| Algorithm | Role in Ensemble | Weight |
|---|---|---|
| **Random Forest** | Tree-based ensemble (bagging) — captures non-linear patterns | 2 |
| **Gradient Boosting (GBM)** | Sequential boosting — corrects errors iteratively | 2 |
| **XGBoost** | Optimized gradient boosting with regularization | 2 |
| **Logistic Regression** | Linear baseline — provides stability and interpretability | 1 |

### Preprocessing & Validation

| Technique | Purpose |
|---|---|
| **RobustScaler** | Scales features using median and IQR — robust to outliers in financial data |
| **TimeSeriesSplit** | K-fold cross-validation that respects temporal ordering |
| **Walk-Forward Validation** | Expanding-window simulation of real sequential predictions |

---

## Dataset

- **Source**: Nepal Stock Exchange (NEPSE) — SANIMA Bank historical data
- **File**: `SANIMA_cleanedlatest.csv`
- **Required Columns**: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`
- **Format**: Standard OHLCV CSV with chronological rows

> All technical indicators are computed inside the script. Do NOT pre-compute or pre-bake any indicators in the CSV.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Steps

```bash
# 1. Clone or download the project
cd Stock-Market-Analysis

# 2. Install required packages
pip install numpy pandas matplotlib seaborn scikit-learn xgboost

# 3. Place your dataset in the project directory
# Ensure SANIMA_cleanedlatest.csv is in the same folder as train_test11.py
```

### Verify Installation

```bash
python -c "import numpy, pandas, matplotlib, seaborn, sklearn, xgboost; print('All dependencies installed.')"
```

---

## Usage

### Training + Prediction (Full Pipeline)

```bash
python train_test11.py
```

This single command will:
1. Load and validate the CSV dataset
2. Display class balance statistics
3. Engineer 50+ features from raw OHLCV data
4. Train the ensemble model with cross-validation
5. Run walk-forward validation
6. Execute out-of-sample backtesting
7. Generate all visualizations in `nbl_v5_output/`
8. Save the trained model to `nbl_v5_output/model/`
9. Print today's BUY/SELL/WAIT prediction

### Reusing a Saved Model (No Retraining)

```python
from train_test11 import NBLMaxProfitV5

system = NBLMaxProfitV5()
system.load()                                    # Load saved model
system.predict_today(data_path="SANIMA_cleanedlatest.csv")  # Predict
```

---

## Output Files

All outputs are saved to the `nbl_v5_output/` directory:

### Visualizations

| File | Description |
|---|---|
| `target_distribution.png` | Bar chart + pie chart showing UP/DOWN/Dropped row counts after triple-barrier filtering |
| `feature_boxplots.png` | Box-plot distributions for all engineered features (EDA) |
| `correlation_heatmap.png` | Pearson correlation heatmap of feature pairs |
| `confusion_matrix_validation.png` | Confusion matrix (counts + percentages) on validation set |
| `confusion_matrix_test.png` | Confusion matrix (counts + percentages) on test set |
| `learning_curves.png` | Train vs. Validation F1-score across increasing training sizes with overfit diagnosis |
| `feature_importance.png` | Top-20 feature importances averaged across RF + GBM |
| `roc_curve_validation.png` | ROC-AUC curve with Youden's J optimal threshold (validation) |
| `roc_curve_test.png` | ROC-AUC curve with Youden's J optimal threshold (test) |
| `equity_curve.png` | Strategy equity vs. Buy & Hold benchmark with drawdown subplot |

### Reports

| File | Description |
|---|---|
| `metrics_report.txt` | Full classification report with Accuracy, F1, MCC, Kappa, ROC-AUC for train/val/test |
| `backtest_summary.txt` | Backtest results: return %, win rate, Sharpe ratio, max drawdown |

### Saved Model

| File | Description |
|---|---|
| `model/model.pkl` | Serialized ensemble VotingClassifier |
| `model/scaler.pkl` | Serialized RobustScaler (fit on training data) |
| `model/last_data.pkl` | Last 50 rows of processed data for quick prediction |
| `model/meta.json` | Feature list, thresholds, fee rate, target parameters |

---

## Model Details

### Ensemble Architecture

The system uses a **Soft Voting Classifier** that combines probability estimates from four base models:

```
VotingClassifier (soft voting)
├── Random Forest       (n=300, depth=3, leaf=20)  weight=2
├── Gradient Boosting   (n=100, depth=3, lr=0.02)  weight=2
├── XGBoost             (n=200, depth=2, lr=0.05)  weight=2
└── Logistic Regression (C=0.3, balanced)           weight=1
```

### Target Labeling (Triple-Barrier)

Instead of predicting naive next-day up/down (which is noisy), the system uses:

- **Horizon**: 7 trading days forward
- **Threshold**: 1.0% minimum absolute move
- **UP (1)**: Future 7-day return > +1.0%
- **DOWN (0)**: Future 7-day return < -1.0%
- **Dropped**: |Future return| <= 1.0% (ambiguous, removed from training)

### Curated Feature Set

The model uses 13 hand-selected features optimized for NEPSE bank stocks:

| Feature | Category |
|---|---|
| `return` | Price return |
| `rsi_14` | Momentum (RSI) |
| `macd` | Trend (MACD line) |
| `bb_width` | Volatility (Bollinger Band width) |
| `ATR_14` | Volatility (Average True Range) |
| `ADX_14` | Trend strength |
| `volatility_5` | Short-term volatility |
| `momentum_10` | 10-day momentum |
| `trend_strength` | SMA20/SMA50 ratio |
| `range_pct` | Daily range as % of close |
| `vol_force` | Volume force indicator |
| `inst_proxy` | Institutional volume proxy |
| `price_squat` | Price squat indicator |

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Accuracy** | Overall correctness of predictions |
| **Precision** | Of all predicted UP, how many were actually UP |
| **Recall** | Of all actual UP, how many were correctly predicted |
| **F1-Score** | Harmonic mean of Precision and Recall (macro + weighted) |
| **ROC-AUC** | Area under the Receiver Operating Characteristic curve |
| **Matthews Correlation Coefficient (MCC)** | Balanced metric even for imbalanced classes (-1 to +1) |
| **Cohen's Kappa** | Agreement beyond random chance |
| **Confusion Matrix** | TP/TN/FP/FN breakdown |
| **5-Fold TimeSeriesSplit CV** | Cross-validation respecting temporal order |
| **Walk-Forward Validation** | Expanding-window sequential prediction test |
| **Sharpe Ratio** | Risk-adjusted return in backtesting |
| **Max Drawdown** | Largest peak-to-trough portfolio decline |
| **Win Rate** | Percentage of profitable trades |

---

## Project Structure

```
Stock-Market-Analysis/
├── train_test11.py              # Main script (training, evaluation, prediction)
├── SANIMA_cleanedlatest.csv     # Input dataset (raw OHLCV — user-provided)
├── README.md                    # This file
├── research_paper.md            # Detailed research paper / project report
└── nbl_v5_output/               # Generated outputs (created on first run)
    ├── target_distribution.png
    ├── feature_boxplots.png
    ├── correlation_heatmap.png
    ├── confusion_matrix_validation.png
    ├── confusion_matrix_test.png
    ├── learning_curves.png
    ├── feature_importance.png
    ├── roc_curve_validation.png
    ├── roc_curve_test.png
    ├── equity_curve.png
    ├── metrics_report.txt
    ├── backtest_summary.txt
    └── model/
        ├── model.pkl
        ├── scaler.pkl
        ├── last_data.pkl
        └── meta.json
```

---

## Configuration & Tuning

Global parameters are defined at the top of `train_test11.py`:

```python
# Target labeling
TARGET_HORIZON   = 7      # Trading days to look forward
TARGET_THRESHOLD = 0.01   # 1.0% minimum absolute move

# Prediction thresholds
BUY_THRESHOLD    = 0.50   # Probability above this → BUY
SELL_THRESHOLD   = 0.45   # Probability below this → SELL

# Backtesting
fee_percent      = 0.003  # 0.3% brokerage fee per trade (entry + exit)
initial_capital  = 100000 # Starting capital in NPR
```

### Tuning Guidelines

| Parameter | Effect of Increasing |
|---|---|
| `TARGET_HORIZON` | Predicts longer-term moves; fewer but stronger signals |
| `TARGET_THRESHOLD` | Drops more rows; cleaner signal but smaller dataset |
| `BUY_THRESHOLD` | More conservative; fewer but higher-confidence BUY signals |
| `SELL_THRESHOLD` | More conservative; faster exit from positions |

---

## Methodology Summary

1. **Data Acquisition**: Historical OHLCV data from NEPSE for SANIMA Bank loaded from CSV
2. **Feature Engineering**: 50+ features computed from raw price/volume data using backward-looking windows only
3. **Target Labeling**: Triple-Barrier method with configurable horizon and threshold
4. **Data Splitting**: 70/15/15 chronological split (no shuffling to prevent future leakage)
5. **Scaling**: RobustScaler fit exclusively on training data
6. **Model Training**: Soft-voting ensemble with 4 diverse classifiers
7. **Cross-Validation**: 5-fold TimeSeriesSplit on training data
8. **Walk-Forward Validation**: Expanding-window test on train+val period
9. **Backtesting**: Realistic simulation with fees, ATR-based stop-loss, minimum hold period
10. **Prediction**: Probability-based BUY/SELL/WAIT signal for the latest data point

---

## Limitations

- Trained and tested on a single NEPSE stock (SANIMA Bank); generalization to other stocks requires retraining
- Does not account for market-wide factors (index movement, macroeconomic events, policy changes)
- NEPSE's low liquidity and circuit breakers may cause real-world slippage not captured in backtesting
- The model predicts direction only — it does not predict magnitude of price movement
- Historical performance does not guarantee future results

---

## References

1. Patel, J., Shah, S., Thakkar, P., & Kotecha, K. (2015). Predicting stock and stock price index movement using Trend Deterministic Data Preparation and machine learning techniques. *Expert Systems with Applications*, 42(1), 259-268.
2. Bustos, O., & Pomares-Quimbaya, A. (2020). Stock market movement forecast: A systematic review. *Expert Systems with Applications*, 156, 113464.
3. Khaidem, L., Saha, S., & Dey, S. R. (2016). Predicting the direction of stock market prices using random forest. *Applied Mathematical Finance*, 1-20.
4. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD*, 785-794.
5. de Prado, M. L. (2018). *Advances in Financial Machine Learning*. Wiley. (Triple-Barrier Method)
6. Nepal Stock Exchange (NEPSE). Official website: https://nepalstock.com.np/

---

## License

This project is developed as a college minor project for academic purposes.
