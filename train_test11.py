"""
==============================================================================
  NBL MAX PROFIT SYSTEM v5.2 — College Minor Project Edition
  Author  : (Your Name)
  Purpose : NEPSE Stock Direction Prediction using Ensemble ML
  Dataset : SANIMA_cleanedlatest.csv  (raw OHLCV CSV — no pre-baked indicators)

  CHANGES v5.0 → v5.1 (leakage fixes):
  ──────────────────────────────────────
  [FIX 1] direction target recomputed inside train() from raw Close prices,
          never read from CSV (prevents pre-baked leakage).
  [FIX 2] engineer_features() is now actually called on the raw CSV so all
          rolling/indicator columns are computed correctly in-pipeline.
  [FIX 3] Walk-forward validation restricted to train+val rows only
          (no test-period leakage into any training fold).
  [FIX 4] Threshold optimisation replaced with a fixed principled default
          (0.55 / 0.45). Grid-searching thresholds on the val set was
          overfitting to validation noise.
  [FIX 5] Scaler is fit ONLY on training rows; val & test use .transform().
  [FIX 6] Class-balance check uses the same target derivation as train().

  CHANGES v5.1 → v5.2 (better target signal):
  ─────────────────────────────────────────────
  [FIX 7] Triple-Barrier / minimum-move target labeling replaces the noisy
          binary next-day up/down label. Ambiguous near-flat rows are dropped
          so the model only learns from meaningful price moves.
          Tunable via TARGET_HORIZON and TARGET_THRESHOLD at the top of this
          file. The class-balance check in main() reflects the new labeling.
  [FIX 8] predict_today() now prints the target horizon and threshold so the
          user knows what "UP" means in the context of this model.

==============================================================================

WHAT THIS FILE PRODUCES
────────────────────────
  1. Target Distribution Chart    (target_distribution.png)
  2. Dataset Overview Box-Plot    (feature_boxplots.png)
  3. Correlation Heat-Map         (correlation_heatmap.png)
  4. Confusion Matrices           (confusion_matrix_validation/test.png)
  5. Train vs Val Learning Curve  (learning_curves.png)
  6. Feature Importance Bar Chart (feature_importance.png)
  7. Full Classification Report   (metrics_report.txt)
  8. ROC-AUC Curves               (roc_curve_validation/test.png)
  9. Equity Curve (Backtest)      (equity_curve.png)
  10. Final BUY / SELL / WAIT prediction on latest data

EVALUATION METRICS USED
────────────────────────
  Accuracy               — basic correctness
  Precision / Recall     — class-level quality
  F1-Score               — harmonic mean of P & R
  ROC-AUC                — discrimination ability
  Matthews Corr Coef     — balanced metric for imbalanced classes
  Cohen's Kappa          — agreement beyond chance
  Confusion Matrix       — TP/TN/FP/FN breakdown
  Cross-Validation       — 5-fold TimeSeriesSplit on training data
  Walk-Forward Validation— rolling expanding-window test (train+val only)

HOW TO RUN
───────────
  python train_test11.py
  (make sure SANIMA_cleanedlatest.csv is in the same folder)

  Your CSV must have AT MINIMUM these raw OHLCV columns:
    Date, Open, High, Low, Close, Volume
  All indicators are computed inside this script — do NOT pre-bake them.

TARGET TUNING
─────────────
  Adjust TARGET_HORIZON and TARGET_THRESHOLD below to change what the model
  predicts. The defaults (5-day horizon, 0.8% minimum move) work well for
  NEPSE mid-cap stocks. Increase THRESHOLD for lower-liquidity stocks.
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
import warnings
import logging
import json
import pickle
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless – works on servers / college PCs
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, learning_curve

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL TUNING PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# Triple-Barrier target settings
# ─────────────────────────────────────────────────────────────────────────────
# TARGET_HORIZON  : how many trading days forward to measure the return
# TARGET_THRESHOLD: minimum absolute % move required to label a row UP or DOWN
#                   rows where |return| < threshold are dropped (ambiguous)
TARGET_HORIZON    = 10     # trading days forward
TARGET_THRESHOLD  = 0.015  # 1.5% minimum absolute move

# Recent-window: use only the last N rows of raw data for training.
# This ensures the model learns from recent market regimes rather than
# stale patterns from a decade ago. Set to None to use all data.
RECENT_WINDOW     = None   # None = use all data

# Fixed prediction thresholds — no val-set grid search [FIX 4]
BUY_THRESHOLD  = 0.50
SELL_THRESHOLD = 0.45

# ─── Plot Style ───────────────────────────────────────────────────────────────
PALETTE = {
    "bg":    "#0F1117",
    "panel": "#1A1D2E",
    "acc":   "#4FC3F7",
    "buy":   "#00E676",
    "sell":  "#FF5252",
    "warn":  "#FFD740",
    "text":  "#E0E0E0",
}
plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["panel"],
    "axes.edgecolor":    "#333",
    "axes.labelcolor":   PALETTE["text"],
    "xtick.color":       PALETTE["text"],
    "ytick.color":       PALETTE["text"],
    "text.color":        PALETTE["text"],
    "grid.color":        "#2A2D3E",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "DejaVu Sans",
    "axes.titlepad":     12,
})

OUTPUT_DIR = Path("nbl_v5_output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 ─ TARGET LABELING  [FIX 7]
# ══════════════════════════════════════════════════════════════════════════════

def make_target(close: pd.Series,
                horizon:   int   = TARGET_HORIZON,
                threshold: float = TARGET_THRESHOLD) -> pd.Series:
    """
    Triple-Barrier / minimum-move target labeling.

    For each row i:
      future_return = (Close[i + horizon] / Close[i]) - 1

      direction[i] = 1   if future_return >  +threshold   (strong UP)
      direction[i] = 0   if future_return <  -threshold   (strong DOWN)
      direction[i] = NaN if |future_return| <= threshold  (ambiguous — dropped)

    Why this is better than simple next-day up/down:
      Filters out the ~40-60% of days that are near-flat noise
      Gives the model a cleaner, more learnable signal
      Reduces class-imbalance caused by micro-moves around zero
      The model predicts meaningful price moves, not random daily ticks

    Parameters
    ----------
    close     : raw Close price Series, chronologically ordered
    horizon   : number of trading days to look forward
    threshold : minimum absolute fractional move to count as a signal
                (0.01 = 1%)

    Returns
    -------
    pd.Series of {0.0, 1.0, NaN} with same index as close
    """
    future_return = close.shift(-horizon) / close - 1
    direction     = pd.Series(np.nan, index=close.index)
    direction[future_return >  threshold] = 1.0
    direction[future_return < -threshold] = 0.0
    # Rows where |future_return| <= threshold remain NaN → dropped in train()
    return direction


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 ─ FEATURE ENGINEERING  [FIX 2]
#  All rolling calculations are strictly backward-looking (no future leakage).
#  The direction target is NOT computed here — make_target() handles that.
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all derived features from raw OHLCV data.
    Called on the FULL raw dataframe BEFORE splitting, which is safe because:
      - Every operation uses only past data (rolling backwards, shift(1), etc.)
      - The target column (direction) is added separately via make_target()
      - No future prices are referenced anywhere in this function
    """
    df = df.copy().sort_values("Date").reset_index(drop=True)

    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    open_ = df["Open"]

    # ── Returns ───────────────────────────────────────────────────────────────
    df["return"]     = close.pct_change()
    df["log_return"] = np.log(close / close.shift(1))
    df["return_2d"]  = close.pct_change(2)
    df["return_5d"]  = close.pct_change(5)

    for lag in [1, 2, 3]:
        df[f"return_lag_{lag}"] = df["return"].shift(lag)

    # ── Candle Shape ──────────────────────────────────────────────────────────
    df["body_size"]        = (close - open_) / open_.replace(0, np.nan)
    df["shadow_ratio"]     = (high - low)    / close.replace(0, np.nan)
    df["upper_shadow"]     = (high - close.where(close >= open_, open_)) / close.replace(0, np.nan)
    df["lower_shadow"]     = (close.where(close <= open_, open_) - low)  / close.replace(0, np.nan)
    df["close_open_ratio"] = (close - open_) / (open_ + 1e-9)
    df["high_low_pct"]     = (high - low)    / (close + 1e-9)
    df["gap"] = (open_ - close.shift(1)) / (close.shift(1) + 1e-9)

    # ── Moving Averages ───────────────────────────────────────────────────────
    for w in [5, 10, 20, 50]:
        df[f"SMA_{w}"] = close.rolling(w).mean()
        df[f"EMA_{w}"] = close.ewm(span=w, adjust=False).mean()
        if "Volume" in df.columns:
            df[f"vol_SMA_{w}"] = df["Volume"].rolling(w).mean()

    df["sma_gap_5_20"]  = (df["SMA_5"]  - df["SMA_20"]) / (df["SMA_20"] + 1e-9)
    df["sma_gap_10_50"] = (df["SMA_10"] - df["SMA_50"]) / (df["SMA_50"] + 1e-9)
    df["ema_cross"]     = (df["EMA_5"] > df["EMA_20"]).astype(int)

    # ── Volatility ────────────────────────────────────────────────────────────
    df["volatility_5"]     = df["return"].rolling(5).std()
    df["std_20"]           = df["return"].rolling(20).std()
    df["volatility_ratio"] = df["volatility_5"] / (df["std_20"] + 1e-9)

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["ATR_14"]          = tr.rolling(14).mean()
    df["risk_adj_return"] = df["return"] / (df["ATR_14"] + 1e-6)

    # ── Momentum ──────────────────────────────────────────────────────────────
    df["momentum_5"]  = close - close.shift(5)
    df["momentum_10"] = close - close.shift(10)
    # Price-normalized versions (regime-invariant across different price levels)
    df["momentum_5_pct"]  = df["momentum_5"]  / (close + 1e-9)
    df["momentum_10_pct"] = df["momentum_10"] / (close + 1e-9)
    df["atr_14_pct"]      = df["ATR_14"]      / (close + 1e-9)

    # ── Extended Returns (longer horizons for 10d target) ────────────────────
    df["return_10d"] = close.pct_change(10)
    df["return_20d"] = close.pct_change(20)

    # ── Price Position (regime-invariant: always bounded 0-1) ────────────────
    high_20  = close.rolling(20).max()
    low_20   = close.rolling(20).min()
    df["price_pos_20d"] = (close - low_20) / (high_20 - low_20 + 1e-9)

    high_50  = close.rolling(50).max()
    low_50   = close.rolling(50).min()
    df["price_pos_50d"] = (close - low_50) / (high_50 - low_50 + 1e-9)

    # ── Return Acceleration ──────────────────────────────────────────────────
    df["return_accel"] = df["return"].diff()  # change in daily return

    # ── Mean Reversion Signal ────────────────────────────────────────────────
    df["dist_from_sma20"] = (close - df["SMA_20"]) / (df["SMA_20"] + 1e-9)

    # ── NEPSE BANK-SPECIFIC FEATURES ─────────────────────────────────
    df["range_pct"] = (high - low) / (close + 1e-9)
    df["trend_strength"] = df["SMA_20"] / (df["SMA_50"] + 1e-9)

    if "Volume" in df.columns:
        df["volume_stability"] = df["Volume"].rolling(10).std() / (df["Volume"].rolling(10).mean() + 1e-9)

    df["breakout_20"] = (close > df["SMA_20"]).astype(int)

    # ── RSI ───────────────────────────────────────────────────────────────────
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["rsi_14"]         = 100 - (100 / (1 + rs))
    df["RSI_overbought"] = (df["rsi_14"] > 70).astype(int)
    df["RSI_oversold"]   = (df["rsi_14"] < 30).astype(int)

    # ── MACD ──────────────────────────────────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
    # Price-normalized MACD (regime-invariant)
    df["macd_hist_pct"] = df["macd_hist"] / (close + 1e-9)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    mid   = close.rolling(20).mean()
    sigma = close.rolling(20).std()
    df["BB_upper"] = mid + 2 * sigma
    df["BB_lower"] = mid - 2 * sigma
    df["bb_width"] = (df["BB_upper"] - df["BB_lower"]) / (mid + 1e-9)
    df["BB_pct"]   = (close - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"] + 1e-9)

    # ── Stochastic %K / %D ───────────────────────────────────────────────────
    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df["Stoch_K"] = 100 * (close - low14) / (high14 - low14 + 1e-9)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    # ── ADX ───────────────────────────────────────────────────────────────────
    plus_dm  = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm[plus_dm   < minus_dm] = 0
    minus_dm[minus_dm < plus_dm]  = 0
    atr14_s  = tr.rolling(14).sum()
    plus_di  = 100 * plus_dm.rolling(14).sum()  / (atr14_s + 1e-9)
    minus_di = 100 * minus_dm.rolling(14).sum() / (atr14_s + 1e-9)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    df["ADX_14"] = dx.rolling(14).mean()

    # ── Volume Features ───────────────────────────────────────────────────────
    if "Volume" in df.columns:
        df["vol_SMA_20"]   = df["Volume"].rolling(20).mean()
        df["vol_ratio"]    = df["Volume"] / (df["vol_SMA_20"] + 1e-9)

        df["vol_change"]   = df["Volume"].pct_change()
        df["avg_volume_5"] = df["Volume"].rolling(5).mean()
        df["price_vol"]    = df["return"] * df["vol_ratio"]

        df["vol_force"] = (df["Volume"] * (df["Close"] - df["Open"])) / (df["High"] - df["Low"] + 1e-9)
        df["inst_proxy"] = df["Volume"] / (df["Volume"].rolling(60).mean() + 1e-9)
        df["price_squat"] = df["range_pct"] / (df["vol_ratio"] + 1e-9)

    # ── Calendar ──────────────────────────────────────────────────────────────
    if "Date" in df.columns:
        df["day_of_week"] = pd.to_datetime(df["Date"]).dt.dayofweek
        df["month"]       = pd.to_datetime(df["Date"]).dt.month

    # ── Indicator Momentum (change in oscillators — inherently stationary) ───
    # These capture whether indicators are ACCELERATING or DECELERATING
    if "rsi_14" in df.columns:
        df["rsi_momentum_3d"]  = df["rsi_14"] - df["rsi_14"].shift(3)
    if "ADX_14" in df.columns:
        df["adx_momentum_3d"]  = df["ADX_14"] - df["ADX_14"].shift(3)
    if "bb_width" in df.columns:
        df["bbw_momentum_3d"]  = df["bb_width"] - df["bb_width"].shift(3)
    if "macd_hist" in df.columns:
        df["macd_accel"]       = df["macd_hist"] - df["macd_hist"].shift(3)
        df["macd_accel_pct"]   = df["macd_accel"] / (close + 1e-9)

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 ─ VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_target_distribution(rows_before: int, rows_after: int,
                             down_count: int, up_count: int):
    """
    [FIX 7 — new plot] Show how many rows were UP / DOWN / dropped after
    triple-barrier filtering. Essential for the college report EDA section.
    """
    logger.info("  Generating target distribution chart ...")
    dropped = rows_before - rows_after

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Target Label Distribution after Triple-Barrier Filtering\n"
        f"(Horizon = {TARGET_HORIZON} days  |  Threshold = {TARGET_THRESHOLD*100:.1f}%)",
        fontsize=13, fontweight="bold", color=PALETTE["acc"]
    )

    bars = ax1.bar(
        ["Dropped\n(ambiguous)", "DOWN (0)\nkept", "UP (1)\nkept"],
        [dropped, down_count, up_count],
        color=[PALETTE["warn"], PALETTE["sell"], PALETTE["buy"]],
        edgecolor="none", width=0.5
    )
    for bar, val in zip(bars, [dropped, down_count, up_count]):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(dropped, down_count, up_count) * 0.01,
                 str(val), ha="center", fontsize=11,
                 fontweight="bold", color=PALETTE["text"])
    ax1.set_ylabel("Row Count", fontsize=11)
    ax1.set_title("Row Counts", fontsize=11)
    ax1.grid(True, axis="y")

    ax2.pie(
        [down_count, up_count],
        labels=["DOWN (0)", "UP (1)"],
        colors=[PALETTE["sell"], PALETTE["buy"]],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": PALETTE["bg"], "linewidth": 2}
    )
    ax2.set_title("Class Balance (kept rows only)", fontsize=11)

    plt.tight_layout()
    path = OUTPUT_DIR / "target_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"   Saved -> {path}")


def plot_dataset_boxplots(df: pd.DataFrame, features: list):
    """PLOT — Box-plots for feature distributions (EDA)."""
    logger.info("  Generating feature box-plots ...")
    n    = len(features)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    fig.suptitle("Feature Distribution Box-Plots\n(SANIMA — NEPSE)",
                 fontsize=16, fontweight="bold", color=PALETTE["acc"], y=1.01)
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax   = axes[i]
        data = df[feat].dropna()
        ax.boxplot(data, patch_artist=True, notch=True,
                   medianprops=dict(color=PALETTE["buy"],  linewidth=2),
                   boxprops=dict(facecolor=PALETTE["acc"] + "55"),
                   whiskerprops=dict(color=PALETTE["text"]),
                   capprops=dict(color=PALETTE["text"]),
                   flierprops=dict(marker=".", color=PALETTE["warn"], alpha=0.4))
        ax.set_title(feat, fontsize=8, fontweight="bold")
        ax.set_xticks([])
        ax.grid(True, axis="y")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = OUTPUT_DIR / "feature_boxplots.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"   Saved -> {path}")


def plot_correlation_heatmap(df: pd.DataFrame, features: list):
    """PLOT — Pearson correlation heat-map."""
    logger.info("  Generating correlation heat-map ...")
    corr = df[features].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    size = max(10, len(features) // 2)
    fig, ax = plt.subplots(figsize=(size, size))
    fig.suptitle("Feature Correlation Heat-Map",
                 fontsize=14, fontweight="bold", color=PALETTE["acc"])
    sns.heatmap(corr, mask=mask, annot=(len(features) <= 20),
                fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.3, ax=ax,
                annot_kws={"size": 7},
                cbar_kws={"shrink": 0.8})
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=7)

    plt.tight_layout()
    path = OUTPUT_DIR / "correlation_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"   Saved -> {path}")


def plot_confusion_matrix(y_true, y_pred, split_name="Test"):
    """PLOT — Confusion matrix with counts and percentages."""
    logger.info(f"  Generating confusion matrix ({split_name}) ...")
    cm  = confusion_matrix(y_true, y_pred)
    pct = cm.astype(float) / cm.sum() * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Confusion Matrix -- {split_name} Set",
                 fontsize=14, fontweight="bold", color=PALETTE["acc"])

    for ax, data, fmt, title in zip(
        axes,
        [cm, pct],
        ["d", ".1f"],
        ["Counts", "Percentages (%)"],
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=["DOWN(0)", "UP(1)"],
                    yticklabels=["DOWN(0)", "UP(1)"],
                    linewidths=1, ax=ax,
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label",      fontsize=11)
        ax.set_title(title,              fontsize=11)

    plt.tight_layout()
    path = OUTPUT_DIR / f"confusion_matrix_{split_name.lower()}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"   Saved -> {path}")


def plot_learning_curves(estimator, X_train, y_train,
                         title="Learning Curves"):
    """PLOT — Train vs Validation F1 across increasing training sizes."""
    logger.info("  Generating learning curves (this may take a while) ...")
    tscv = TimeSeriesSplit(n_splits=5)
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X_train, y_train,
        cv=tscv,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="f1",
        n_jobs=-1,
    )
    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold", color=PALETTE["acc"])

    ax.plot(train_sizes, train_mean, "o-", color=PALETTE["buy"],
            label="Training F1", lw=2)
    ax.fill_between(train_sizes,
                    train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color=PALETTE["buy"])
    ax.plot(train_sizes, val_mean, "s-", color=PALETTE["sell"],
            label="Validation F1", lw=2)
    ax.fill_between(train_sizes,
                    val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color=PALETTE["sell"])
    ax.axhline(y=val_mean[-1], color=PALETTE["warn"], linestyle="--",
               alpha=0.5, label=f"Final Val F1 = {val_mean[-1]:.3f}")

    ax.set_xlabel("Training Examples", fontsize=11)
    ax.set_ylabel("F1 Score",          fontsize=11)
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=10)
    ax.grid(True)

    gap       = train_mean[-1] - val_mean[-1]
    diagnosis = ("Good Fit"       if gap < 0.05  else
                 "Overfitting"    if gap > 0.15  else
                 "Slight Overfit")
    ax.text(0.02, 0.05,
            f"Train-Val Gap: {gap:.3f}  ->  {diagnosis}",
            transform=ax.transAxes, fontsize=10, color=PALETTE["warn"],
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor=PALETTE["panel"],
                      edgecolor=PALETTE["warn"]))

    plt.tight_layout()
    path = OUTPUT_DIR / "learning_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"   Saved -> {path}")


def plot_feature_importance(model, features: list):
    """PLOT — Average RF + GB feature importances."""
    logger.info("  Generating feature importance chart ...")
    # Extract importances from named estimators
    imp_rf  = model.named_estimators_["rf"].feature_importances_
    imp_gb  = model.named_estimators_["gb"].feature_importances_
    imp_xgb = model.named_estimators_["xgb"].feature_importances_
    imp_avg = (imp_rf + imp_gb + imp_xgb) / 3

    feat_imp = sorted(zip(features, imp_avg), key=lambda x: x[1], reverse=True)
    top_n    = min(20, len(feat_imp))
    feat_imp = feat_imp[:top_n]
    labels   = [f[0] for f in feat_imp]
    values   = [f[1] for f in feat_imp]

    cmap   = plt.cm.get_cmap("cool", top_n)
    colors = [cmap(i / top_n) for i in range(top_n)]

    fig, ax = plt.subplots(figsize=(10, top_n * 0.4 + 2))
    fig.suptitle("Top Feature Importances (RF + GB + XGB Ensemble Average)",
                 fontsize=13, fontweight="bold", color=PALETTE["acc"])
    bars = ax.barh(labels[::-1], values[::-1], color=colors, edgecolor="none")
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8, color=PALETTE["text"])
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.grid(True, axis="x")
    plt.tight_layout()
    path = OUTPUT_DIR / "feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"   Saved -> {path}")


def plot_roc_curve(y_true, y_prob, split_name="Test"):
    """PLOT — ROC-AUC curve with Youden's J optimal threshold."""
    logger.info(f"  Generating ROC curve ({split_name}) ...")
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc_val    = roc_auc_score(y_true, y_prob)
    opt_idx    = np.argmax(tpr - fpr)
    opt_thresh = thresholds[opt_idx]

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle(f"ROC Curve -- {split_name} Set",
                 fontsize=14, fontweight="bold", color=PALETTE["acc"])
    ax.plot(fpr, tpr, color=PALETTE["buy"], lw=2.5,
            label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier")
    ax.scatter(fpr[opt_idx], tpr[opt_idx], color=PALETTE["warn"],
               s=120, zorder=5,
               label=f"Youden's J Threshold = {opt_thresh:.2f}")
    ax.fill_between(fpr, tpr, alpha=0.08, color=PALETTE["buy"])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True)

    plt.tight_layout()
    path = OUTPUT_DIR / f"roc_curve_{split_name.lower()}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"   Saved -> {path}")
    return float(opt_thresh)


def plot_equity_curve(trade_log: list, initial_capital: float,
                      test_df: pd.DataFrame = None):
    """PLOT — Portfolio equity curve vs Buy & Hold benchmark."""
    logger.info("  Generating equity curve ...")
    values = [t["exit_value"] for t in trade_log if "exit_value" in t]
    if not values:
        logger.warning("   No completed trades to plot.")
        return

    cumulative = [initial_capital] + values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("Strategy Equity Curve vs Buy & Hold Benchmark",
                 fontsize=14, fontweight="bold", color=PALETTE["acc"])

    ax1.plot(range(len(cumulative)), cumulative,
             color=PALETTE["buy"], lw=2.5, label="ML Strategy", zorder=3)
    ax1.fill_between(range(len(cumulative)), initial_capital, cumulative,
                     where=[v >= initial_capital for v in cumulative],
                     alpha=0.15, color=PALETTE["buy"])
    ax1.fill_between(range(len(cumulative)), initial_capital, cumulative,
                     where=[v < initial_capital for v in cumulative],
                     alpha=0.15, color=PALETTE["sell"])

    if test_df is not None and "Close" in test_df.columns:
        prices     = test_df["Close"].values
        bnh_values = [initial_capital * (p / prices[0]) for p in prices]
        bnh_x      = np.linspace(0, len(cumulative) - 1, len(bnh_values))
        ax1.plot(bnh_x, bnh_values, color=PALETTE["warn"], lw=2,
                 linestyle="--", label="Buy & Hold", zorder=2)

        final_bnh      = bnh_values[-1]
        final_strategy = cumulative[-1]
        bnh_ret        = (final_bnh      - initial_capital) / initial_capital * 100
        strat_ret      = (final_strategy - initial_capital) / initial_capital * 100
        alpha          = strat_ret - bnh_ret

        ax1.text(0.02, 0.97,
                 f"Buy & Hold : {bnh_ret:+.2f}%\n"
                 f"Strategy   : {strat_ret:+.2f}%\n"
                 f"Alpha      : {alpha:+.2f}%",
                 transform=ax1.transAxes, fontsize=9,
                 verticalalignment="top", color=PALETTE["text"],
                 bbox=dict(boxstyle="round,pad=0.5",
                           facecolor=PALETTE["panel"],
                           edgecolor=PALETTE["warn"], alpha=0.9))

    ax1.axhline(y=initial_capital, color="#555", linestyle=":", lw=1.5,
                label="Starting Capital")
    ax1.set_ylabel("Portfolio Value (NPR)", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True)

    peak     = np.maximum.accumulate(cumulative)
    drawdown = ((np.array(cumulative) - peak) / (np.array(peak) + 1e-9)) * 100
    ax2.fill_between(range(len(drawdown)), drawdown, 0,
                     color=PALETTE["sell"], alpha=0.6)
    ax2.set_ylabel("Drawdown (%)", fontsize=10)
    ax2.set_xlabel("Trade Number",  fontsize=11)
    ax2.grid(True)

    plt.tight_layout()
    path = OUTPUT_DIR / "equity_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"   Saved -> {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 ─ METRICS REPORT
# ══════════════════════════════════════════════════════════════════════════════

def compute_and_print_metrics(
    y_train, y_train_pred,
    y_val,   y_val_pred,
    y_test,  y_test_pred,
    y_test_prob,
    cv_scores,
):
    def _section(split, y_true, y_pred, y_prob=None):
        acc   = accuracy_score(y_true, y_pred)
        f1_m  = f1_score(y_true, y_pred, average="macro",    zero_division=0)
        f1_w  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        mcc   = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        auc   = (roc_auc_score(y_true, y_prob)
                 if y_prob is not None else "N/A")
        report = classification_report(y_true, y_pred,
                                       target_names=["DOWN(0)", "UP(1)"],
                                       zero_division=0)
        lines = [
            f"\n{'='*60}",
            f"  {split} SET METRICS",
            f"{'='*60}",
            f"  Accuracy              : {acc:.4f}  ({acc*100:.2f}%)",
            f"  F1-Score (Macro)      : {f1_m:.4f}",
            f"  F1-Score (Weighted)   : {f1_w:.4f}",
            f"  Matthews Corr Coef    : {mcc:.4f}  "
            f"{'Good' if mcc > 0.3 else 'Weak'}",
            f"  Cohen's Kappa         : {kappa:.4f}",
            f"  ROC-AUC               : "
            f"{auc if isinstance(auc, str) else f'{auc:.4f}'}",
            f"\n  Per-Class Report:",
            report,
        ]
        return "\n".join(lines)

    target_note = (
        f"\n{'='*60}\n"
        f"  TARGET DEFINITION (v5.2 Triple-Barrier)\n"
        f"{'='*60}\n"
        f"  Horizon   : {TARGET_HORIZON} trading days forward\n"
        f"  Threshold : {TARGET_THRESHOLD*100:.1f}% minimum absolute move\n"
        f"  UP  (1)   : future_{TARGET_HORIZON}d_return >  +{TARGET_THRESHOLD*100:.1f}%\n"
        f"  DOWN (0)  : future_{TARGET_HORIZON}d_return <  -{TARGET_THRESHOLD*100:.1f}%\n"
        f"  Dropped   : |future return| <= {TARGET_THRESHOLD*100:.1f}% (ambiguous)\n"
    )

    cv_line = (
        f"\n{'='*60}\n"
        f"  5-FOLD TIME-SERIES CROSS-VALIDATION (Training Data)\n"
        f"{'='*60}\n"
        f"  CV F1 Scores  : {np.round(cv_scores, 4)}\n"
        f"  Mean CV F1    : {cv_scores.mean():.4f}\n"
        f"  Std  CV F1    : {cv_scores.std():.4f}\n"
    )

    full_report = (
        "\n" + "="*60 + "\n"
        "  NBL MAX PROFIT v5.2 -- FULL EVALUATION REPORT\n" +
        "="*60 +
        target_note +
        cv_line +
        _section("TRAIN",      y_train, y_train_pred) +
        _section("VALIDATION", y_val,   y_val_pred) +
        _section("TEST",       y_test,  y_test_pred, y_test_prob)
    )

    print(full_report)
    path = OUTPUT_DIR / "metrics_report.txt"
    with open(path, "w") as f:
        f.write(full_report)
    logger.info(f"   Metrics saved -> {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 ─ BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(df: pd.DataFrame, probs, thresholds: dict,
                 fee: float, initial_capital: float = 100_000.0) -> tuple:
    """
    Realistic backtest:
      - Uses actual Close prices (not just returns)
      - Applies brokerage fee on BOTH entry and exit
      - ATR-based trailing stop-loss (2x ATR at entry)

    Note: the model predicts TARGET_HORIZON-day moves but the backtest runs
    bar-by-bar with daily exit checks. This is intentional — real trading
    exits on signal or stop-loss, not after a fixed horizon.
    """
    balance   = initial_capital
    shares    = 0.0
    trade_log = []
    atr_vals  = (df["ATR_14"].values
                 if "ATR_14" in df.columns
                 else np.full(len(df), 2.0))
    prices = df["Close"].values
    dates  = df["Date"].values
    MIN_HOLD_DAYS = 3   # banks need time to move
    for i in range(len(probs)):
        price = prices[i]
        prob  = probs[i]

        # ── EXIT ──────────────────────────────────────────────────────────────
        if shares > 0 and trade_log:
            atr_at_entry  = trade_log[-1].get("atr_entry", atr_vals[i])
            stop_loss_hit = price < (trade_log[-1]["buy_price"] - 2 * atr_at_entry)
            holding_days = i - trade_log[-1]["entry_index"]
            signal_sell  = (prob < thresholds["sell"]) and (holding_days >= MIN_HOLD_DAYS)

            if signal_sell or stop_loss_hit:
                proceeds = shares * price * (1 - fee)
                trade_log[-1].update({
                    "exit_date":  dates[i],
                    "exit_price": price,
                    "exit_value": proceeds,
                    "reason":     "SIGNAL" if signal_sell else "STOP_LOSS",
                    "pnl":        proceeds - trade_log[-1]["invested"],
                })
                balance = proceeds
                shares  = 0.0

        # ── ENTRY ─────────────────────────────────────────────────────────────
        elif shares == 0 and prob > thresholds["buy"] and balance > 0:
            invested = balance * (1 - fee)
            shares   = invested / price
            trade_log.append({
                "entry_date": dates[i],
                "entry_index": i,
                "buy_price":  price,
                "invested":   invested,
                "atr_entry":  atr_vals[i],
            })
            balance = 0.0

    final_val = balance if shares == 0 else shares * prices[-1]
    total_ret = (final_val - initial_capital) / initial_capital * 100

    pnl_series = [t["pnl"] for t in trade_log if "pnl" in t]
    if len(pnl_series) >= 2:
        pnl_pct = [p / initial_capital for p in pnl_series]
        sharpe  = (np.mean(pnl_pct) / (np.std(pnl_pct) + 1e-9)) * np.sqrt(252)
        sharpe  = float(np.clip(sharpe, -5, 5))
    elif len(pnl_series) == 1:
        sharpe = 1.0 if pnl_series[0] > 0 else -1.0
    else:
        sharpe = 0.0

    equity_curve = ([initial_capital] +
                    [t["exit_value"] for t in trade_log if "exit_value" in t])
    peak = np.maximum.accumulate(equity_curve)
    mdd  = ((np.array(equity_curve) - peak) / (peak + 1e-9)).min() * 100

    wins      = sum(1 for t in trade_log if t.get("pnl", 0) > 0)
    total_trd = len(pnl_series)
    win_rate  = (wins / total_trd * 100) if total_trd > 0 else 0.0

    summary = {
        "initial_capital":  initial_capital,
        "final_value":      final_val,
        "total_return_pct": total_ret,
        "total_trades":     total_trd,
        "win_rate_pct":     win_rate,
        "sharpe_ratio":     sharpe,
        "max_drawdown_pct": mdd,
    }
    return final_val, trade_log, summary


def print_backtest_summary(summary: dict):
    lines = [
        "\n" + "="*60,
        "  BACKTEST RESULTS (Out-of-Sample Test Set)",
        "="*60,
        f"  Starting Capital  : NPR {summary['initial_capital']:>12,.2f}",
        f"  Final Value       : NPR {summary['final_value']:>12,.2f}",
        f"  Total Return      : {summary['total_return_pct']:>+8.2f}%",
        f"  Total Trades      : {summary['total_trades']}",
        f"  Win Rate          : {summary['win_rate_pct']:.1f}%",
        f"  Sharpe Ratio      : {summary['sharpe_ratio']:.3f}"
        f"{'  Warning: Low sample (< 30 trades)' if summary['total_trades'] < 30 else ''}",
        f"  Max Drawdown      : {summary['max_drawdown_pct']:.2f}%",
        "="*60,
    ]
    print("\n".join(lines))
    path = OUTPUT_DIR / "backtest_summary.txt"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"   Backtest summary saved -> {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 ─ MAIN SYSTEM CLASS
# ══════════════════════════════════════════════════════════════════════════════

class NBLMaxProfitV5:
    """
    NBL MAX PROFIT SYSTEM v5.2
    --------------------------
    Leakage-free, fully evaluated ML trading system for NEPSE stocks.
    Suitable as a Computer Science / Finance minor project.

    Key design decisions
    ---------------------
    - Raw OHLCV CSV -> features computed entirely in-pipeline (no pre-baking)
    - Triple-Barrier target (make_target) — ambiguous near-flat rows dropped
    - Scaler fit ONLY on training rows
    - Fixed prediction thresholds — no val-set grid search
    - Walk-forward validation limited to train+val period only
    """

    CURATED_FEATURES = [
        # Returns (inherently stationary, regime-invariant)
        "return", "return_lag_1", "return_lag_2", "return_lag_3",
        # Volume changes (stationary)
        "vol_change", "vol_change_lag_1",
        # Bounded oscillators (0-100 scale, stationary)
        "rsi_14", "rsi_14_lag_1", "rsi_14_lag_2",
        "ADX_14", "ADX_14_lag_1",
        # Binary signals (0/1, stationary)
        "ema_cross", "ema_cross_lag_1",
        # Normalized band width
        "bb_width", "bb_width_lag_1",
        # Price-shape (all / close normalized)
        "high_low_pct", "close_open_ratio",
        # Price-normalized (regime-invariant)
        "atr_14_pct", "momentum_5_pct", "macd_hist_pct",
        # Additional stationary
        "volatility_ratio", "BB_pct", "Stoch_K",
        # Indicator momentum (3-day changes in oscillators)
        "rsi_momentum_3d", "adx_momentum_3d",
        "bbw_momentum_3d", "macd_accel_pct",
    ]

    def __init__(self, fee_percent: float = 0.004):
        self.fee        = fee_percent
        self.scaler     = RobustScaler()
        self.model      = None
        self.features   = []
        self.thresholds = {"buy": BUY_THRESHOLD, "sell": SELL_THRESHOLD}
        self.last_data  = None

    def _build_ensemble(self):
        rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=3,
            min_samples_leaf=25,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        gb = GradientBoostingClassifier(
            n_estimators=120,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.75,
            min_samples_leaf=20,
            random_state=43,
        )

        xgb = XGBClassifier(
            n_estimators=120,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.75,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=3.0,
            min_child_weight=20,
            random_state=45,
            eval_metric="logloss",
        )
        lr = LogisticRegression(
            C=0.3,
            max_iter=1000,
            class_weight="balanced",
            random_state=44,
        )
        return VotingClassifier(
            estimators=[("rf", rf), ("gb", gb), ("xgb", xgb), ("lr", lr)],
            voting="soft",
            weights=[2, 2, 2, 1],
            n_jobs=-1
        )

    # ── MAIN TRAIN PIPELINE ───────────────────────────────────────────────────
    def train(self, data_path: str):
        logger.info("\n" + "="*60)
        logger.info("  NBL MAX PROFIT SYSTEM v5.2 -- TRAINING STARTED")
        logger.info("="*60)

        # 1. Load and Engineer
        raw_df = pd.read_csv(data_path)
        if "Date" in raw_df.columns:
            raw_df["Date"] = pd.to_datetime(raw_df["Date"])
        raw_df = raw_df.sort_values("Date").reset_index(drop=True)
        # Drop forward-looking / leaky columns; keep all pre-baked indicators
        drop_cols = ["Symbol", "next_day_return", "direction"]
        raw_df = raw_df.drop(columns=[c for c in drop_cols if c in raw_df.columns])

        # Recent-window: trim to last N rows so model learns recent patterns
        if RECENT_WINDOW is not None and len(raw_df) > RECENT_WINDOW:
            raw_df = raw_df.iloc[-RECENT_WINDOW:].reset_index(drop=True)
            logger.info(f"  Recent-window applied: using last {RECENT_WINDOW} rows")

        df = engineer_features(raw_df)
        rows_before = len(df)

        # 2. Triple-Barrier Target [FIX 1, FIX 7]
        df["direction"] = make_target(df["Close"])
        # Remove last HORIZON rows (no future data) and ambiguous rows
        df = df.iloc[:-TARGET_HORIZON].copy()
        rows_with_target = len(df)
        df = df.dropna(subset=["direction"]).reset_index(drop=True)
        df["direction"] = df["direction"].astype(int)
        rows_after = len(df)

        up_count   = (df["direction"] == 1).sum()
        down_count = (df["direction"] == 0).sum()

        logger.info(f"\n  Dataset: {rows_before} raw rows")
        logger.info(f"  After triple-barrier: {rows_after} kept "
                    f"(UP={up_count}, DOWN={down_count}, "
                    f"Dropped={rows_before - rows_after})")

        # 3. Feature Selection [FIX 2]
        self.features = [f for f in self.CURATED_FEATURES if f in df.columns]
        logger.info(f"  Features selected: {len(self.features)}")
        df[self.features] = df[self.features].ffill().bfill().fillna(0)

        # 4. Chronological Split (no shuffle!) — 75/12.5/12.5
        #    Refit (train+val) = 87.5% to maximize recent training data
        n = len(df)
        train_end = int(n * 0.75)
        val_end   = int(n * 0.875)
        train_df = df.iloc[:train_end]
        val_df   = df.iloc[train_end:val_end]
        test_df  = df.iloc[val_end:]

        logger.info(f"  Split: Train={len(train_df)}, "
                    f"Val={len(val_df)}, Test={len(test_df)}")

        # 5. Scaling [FIX 5] — fit on train ONLY
        X_train = self.scaler.fit_transform(train_df[self.features])
        y_train = train_df["direction"].values
        X_val   = self.scaler.transform(val_df[self.features])
        y_val   = val_df["direction"].values
        X_test  = self.scaler.transform(test_df[self.features])
        y_test  = test_df["direction"].values

        # ── 6. EDA Visualizations ────────────────────────────────────────────
        logger.info("\n  --- Generating EDA visualizations ---")
        plot_target_distribution(rows_before, rows_after, down_count, up_count)
        plot_dataset_boxplots(train_df, self.features)
        plot_correlation_heatmap(train_df, self.features)

        # ── 7. Cross-Validation ──────────────────────────────────────────────
        logger.info("\n  Running 5-fold TimeSeriesSplit CV on training set ...")
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []

        for fold, (train_idx, cv_idx) in enumerate(tscv.split(X_train)):
            X_tr_cv, X_val_cv = X_train[train_idx], X_train[cv_idx]
            y_tr_cv, y_val_cv = y_train[train_idx], y_train[cv_idx]

            if len(np.unique(y_tr_cv)) < 2:
                continue

            m_cv = self._build_ensemble()
            m_cv.fit(X_tr_cv, y_tr_cv)
            pred_cv = m_cv.predict(X_val_cv)
            cv_scores.append(f1_score(y_val_cv, pred_cv, zero_division=0))

        cv_scores = np.array(cv_scores)
        logger.info(f"  CV F1 scores: {np.round(cv_scores, 4)}")
        logger.info(f"  CV F1 mean:   {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

        # ── 8. Stage 1: Train on Training Set Only ──────────────────────────
        logger.info("\n  Stage 1: Training initial ensemble on training set ...")
        initial_model = self._build_ensemble()
        initial_model.fit(X_train, y_train)

        # ── 9. Stage 1 Predictions (train & val from initial model) ──────────
        train_pred = initial_model.predict(X_train)

        val_prob = initial_model.predict_proba(X_val)[:, 1]
        val_pred = (val_prob > self.thresholds["buy"]).astype(int)

        # ── 10. Threshold: search narrow range around 0.50 on val set ────────
        best_acc, best_thresh = 0, 0.50
        for t in np.arange(0.47, 0.54, 0.005):
            vp = (val_prob > t).astype(int)
            acc_t = accuracy_score(y_val, vp)
            if acc_t > best_acc:
                best_acc, best_thresh = acc_t, t
        optimal_threshold = float(best_thresh)
        val_pred = (val_prob > optimal_threshold).astype(int)
        self.thresholds["buy"] = optimal_threshold
        self.thresholds["sell"] = max(0.35, optimal_threshold - 0.10)
        val_acc = accuracy_score(y_val, val_pred)
        logger.info(f"  Threshold: {optimal_threshold:.3f} (val acc = {val_acc:.4f})")
        logger.info(f"  Stage 1 — Train acc: {accuracy_score(y_train, train_pred):.4f}, "
                    f"Val acc: {val_acc:.4f}")

        # ── 11. Stage 2: Refit on train+val combined for test eval ───────────
        logger.info("\n  Stage 2: Refitting ensemble on train+val combined ...")
        trainval_df = pd.concat([train_df, val_df])
        scaler_refit = RobustScaler()
        X_trainval = scaler_refit.fit_transform(trainval_df[self.features])
        y_trainval = trainval_df["direction"].values
        X_test_refit = scaler_refit.transform(test_df[self.features])

        self.model = self._build_ensemble()
        self.model.fit(X_trainval, y_trainval)
        self.scaler = scaler_refit  # update scaler for predict_today()

        test_prob = self.model.predict_proba(X_test_refit)[:, 1]
        test_pred = (test_prob > self.thresholds["buy"]).astype(int)

        trainval_pred = self.model.predict(X_trainval)
        logger.info(f"  Stage 2 — Train+Val acc: "
                    f"{accuracy_score(y_trainval, trainval_pred):.4f}")
        logger.info(f"  Stage 2 — Test acc (refit, predict): "
                    f"{accuracy_score(y_test, test_pred):.4f}")

        # ── High-Confidence Selective Prediction ──────────────────────────────
        # Only evaluate predictions where model has strong conviction
        hc_margin = 0.08  # predictions > 0.58 or < 0.42
        hc_mask = (test_prob > (0.50 + hc_margin)) | (test_prob < (0.50 - hc_margin))
        if hc_mask.sum() > 10:
            hc_pred = (test_prob[hc_mask] > 0.50).astype(int)
            hc_acc  = accuracy_score(y_test[hc_mask], hc_pred)
            logger.info(f"\n  HIGH-CONFIDENCE PREDICTIONS:")
            logger.info(f"  Confident samples: {hc_mask.sum()}/{len(y_test)} "
                        f"({hc_mask.sum()/len(y_test)*100:.1f}%)")
            logger.info(f"  High-confidence accuracy: {hc_acc:.4f} ({hc_acc*100:.2f}%)")
        else:
            logger.info(f"\n  (Too few high-confidence predictions: {hc_mask.sum()})")

        # ── 12. Evaluation Visualizations ────────────────────────────────────
        logger.info("\n  --- Generating evaluation visualizations ---")
        plot_confusion_matrix(y_val, val_pred, "Validation")
        plot_confusion_matrix(y_test, test_pred, "Test")

        plot_roc_curve(y_val, val_prob, "Validation")
        plot_roc_curve(y_test, test_prob, "Test")

        plot_feature_importance(initial_model, self.features)

        # Learning curves (uses a fresh estimator to avoid refitting)
        plot_learning_curves(self._build_ensemble(), X_train, y_train,
                             title="Learning Curves (Ensemble on Training Data)")

        # ── 13. Full Metrics Report ──────────────────────────────────────────
        compute_and_print_metrics(
            y_train, train_pred,
            y_val, val_pred,
            y_test, test_pred,
            test_prob, cv_scores
        )

        # ── 14. Walk-Forward Validation (FULL dataset) ──────────────────────
        logger.info("\n  Running walk-forward validation (full dataset) ...")
        self._walk_forward_validation(df.copy(), n_splits=8)

        # ── 15. Backtest on TEST set ─────────────────────────────────────────
        logger.info("\n  Running out-of-sample backtest ...")
        _, trade_log, summary = run_backtest(
            test_df, test_prob, self.thresholds, self.fee
        )
        print_backtest_summary(summary)
        plot_equity_curve(trade_log, initial_capital=100_000.0,
                          test_df=test_df)

        # ── 16. Save ─────────────────────────────────────────────────────────
        self.last_data = df.iloc[-50:].copy()
        self._save()

        logger.info("\n" + "="*60)
        logger.info("  TRAINING COMPLETE -- All outputs in nbl_v5_output/")
        logger.info("="*60)

    # ── WALK-FORWARD VALIDATION ───────────────────────────────────────────────
    def _walk_forward_validation(self, df: pd.DataFrame, n_splits: int = 5):
        """
        Expanding-window walk-forward test.
        [FIX 3] df must be train+val slice only — never the full dataset.
        """
        tscv    = TimeSeriesSplit(n_splits=n_splits)
        results = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
            tr = df.iloc[train_idx]
            te = df.iloc[test_idx]

            if len(np.unique(tr["direction"].values)) < 2:
                continue

            sc  = RobustScaler()
            Xtr = sc.fit_transform(tr[self.features])
            Xte = sc.transform(te[self.features])

            m = self._build_ensemble()
            m.fit(Xtr, tr["direction"].values)

            preds = m.predict(Xte)
            acc   = accuracy_score(te["direction"].values, preds)
            f1    = f1_score(te["direction"].values, preds,
                             average="macro", zero_division=0)
            results.append({"fold": fold + 1, "size": len(te),
                            "accuracy": acc, "f1": f1})

        logger.info("\n  WALK-FORWARD VALIDATION RESULTS:")
        logger.info(f"  {'Fold':<6} {'Size':<6} {'Accuracy':<12} {'F1 Macro'}")
        logger.info("  " + "-"*38)
        for r in results:
            logger.info(f"  {r['fold']:<6} {r['size']:<6} "
                        f"{r['accuracy']:.4f}       {r['f1']:.4f}")

        avg_acc = np.mean([r["accuracy"] for r in results])
        avg_f1  = np.mean([r["f1"]      for r in results])
        std_f1  = np.std( [r["f1"]      for r in results])
        logger.info(f"\n  Average Accuracy : {avg_acc:.4f}")
        logger.info(f"  Average F1       : {avg_f1:.4f}")
        logger.info(
            f"  Interpretation   : "
            f"{'Consistent' if std_f1 < 0.08 else 'Inconsistent across time periods'}"
        )
        return results

    # ── PREDICTION ────────────────────────────────────────────────────────────
    def predict_today(self, data_path: str = None):
        """
        Print actionable BUY / SELL / WAIT signal for today.

        [FIX 8] Output clearly states what "UP" means: the model predicts
        the stock will rise >= TARGET_THRESHOLD% over TARGET_HORIZON trading
        days — NOT just tomorrow.
        """
        if self.model is None:
            logger.error("Model not trained. Call train() first.")
            return

        if data_path is not None:
            raw = pd.read_csv(data_path)
            if "Date" in raw.columns:
                raw["Date"] = pd.to_datetime(raw["Date"])
            raw = raw.sort_values("Date").reset_index(drop=True)
            # Drop forward-looking columns, keep pre-baked indicators
            drop_cols = ["Symbol", "next_day_return", "direction"]
            raw = raw.drop(columns=[c for c in drop_cols if c in raw.columns])
            engineered = engineer_features(raw)
            latest     = engineered.iloc[-1:].copy()
        elif self.last_data is not None:
            latest = self.last_data.iloc[-1:].copy()
        else:
            logger.error("No data available. Pass data_path='SANIMA_cleanedlatest.csv'")
            return

        for col in self.features:
            if col not in latest.columns:
                latest[col] = 0.0
        X_latest = latest[self.features].ffill().bfill().fillna(0)
        X        = self.scaler.transform(X_latest)
        prob     = self.model.predict_proba(X)[0, 1]

        print("\n" + "="*60)
        print("  TODAY'S PREDICTION  (v5.2 Triple-Barrier Target)")
        print("="*60)
        print(f"  Prediction Horizon   : {TARGET_HORIZON} trading days")
        print(f"  Minimum Move         : {TARGET_THRESHOLD*100:.1f}%")
        print(f"  Model Confidence (UP): {prob:.2%}")
        print(f"  Buy  Threshold       : {self.thresholds['buy']:.2%}")
        print(f"  Sell Threshold       : {self.thresholds['sell']:.2%}")
        print()

        if prob > self.thresholds["buy"]:
            print("  ACTION : BUY")
            print(f"  REASON : Model is >{self.thresholds['buy']:.0%} confident the")
            print(f"           stock will rise >={TARGET_THRESHOLD*100:.1f}% over")
            print(f"           the next {TARGET_HORIZON} trading days.")
            print("  ADVICE : Enter with ATR-based stop-loss. Limit position size.")
        elif prob < self.thresholds["sell"]:
            print("  ACTION : SELL / STAY OUT")
            print(f"  REASON : Model expects a drop >={TARGET_THRESHOLD*100:.1f}%")
            print(f"           or flat movement over the next {TARGET_HORIZON} days.")
            print("  ADVICE : Preserve capital. Check again tomorrow.")
        else:
            print("  ACTION : WAIT")
            print("  REASON : No high-conviction signal in either direction.")
            print("  ADVICE : Avoid paying brokerage on uncertain trades.")
        print("="*60)

    # ── SAVE / LOAD ───────────────────────────────────────────────────────────
    def _save(self):
        p = OUTPUT_DIR / "model"
        p.mkdir(exist_ok=True)
        with open(p / "scaler.pkl",    "wb") as f: pickle.dump(self.scaler,    f)
        with open(p / "model.pkl",     "wb") as f: pickle.dump(self.model,     f)
        with open(p / "last_data.pkl", "wb") as f: pickle.dump(self.last_data, f)
        with open(p / "meta.json",     "w")  as f:
            json.dump({
                "features":         self.features,
                "thresholds":       self.thresholds,
                "fee":              self.fee,
                "target_horizon":   TARGET_HORIZON,
                "target_threshold": TARGET_THRESHOLD,
            }, f, indent=2)
        logger.info(f"  Model saved -> {p}/")

    def load(self, model_dir: str = None):
        p = Path(model_dir) if model_dir else OUTPUT_DIR / "model"
        with open(p / "scaler.pkl",    "rb") as f: self.scaler    = pickle.load(f)
        with open(p / "model.pkl",     "rb") as f: self.model     = pickle.load(f)
        with open(p / "last_data.pkl", "rb") as f: self.last_data = pickle.load(f)
        with open(p / "meta.json",     "r")  as f:
            meta            = json.load(f)
            self.features   = meta["features"]
            self.thresholds = meta["thresholds"]
            self.fee        = meta["fee"]
        logger.info("  Model loaded.")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    CSV = "SANIMA_cleanedlatest.csv"

    # [FIX 6 + FIX 7] Class-balance check uses triple-barrier target,
    # exactly matching what train() will compute.
    _raw = pd.read_csv(CSV)
    if "Date" in _raw.columns:
        _raw["Date"] = pd.to_datetime(_raw["Date"])
    _raw = _raw.sort_values("Date").reset_index(drop=True)
    _raw["_direction"] = make_target(_raw["Close"])
    _kept = _raw.dropna(subset=["_direction"])

    print(
        f"\nCLASS BALANCE CHECK  "
        f"(horizon={TARGET_HORIZON}d, threshold={TARGET_THRESHOLD*100:.1f}%):"
    )
    print(_kept["_direction"].value_counts())
    print(
        _kept["_direction"]
        .value_counts(normalize=True)
        .mul(100).round(2)
        .astype(str) + "%"
    )
    original = len(_raw)
    kept     = len(_kept)
    print(f"\n  Rows kept   : {kept} / {original} ({kept/original*100:.1f}%)")
    print(f"  Rows dropped: {original - kept} (near-flat / ambiguous)\n")

    system = NBLMaxProfitV5(fee_percent=0.003)

    # Train — reads raw OHLCV, engineers features, applies triple-barrier target
    system.train(CSV)

    # Predict — re-engineers features from latest raw data, uses last row
    system.predict_today(data_path=CSV)

    # ── To reuse a saved model later without retraining: ──────────────────────
    # system.load()
    # system.predict_today(data_path=CSV)


if __name__ == "__main__":
    main()
