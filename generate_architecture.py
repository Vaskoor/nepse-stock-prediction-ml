"""
Generate Professional System Architecture Diagram for NBL MAX PROFIT System v5.2
Step-by-step numbered flow with clear interconnections.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  SETUP
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 20))
ax.set_xlim(0, 12)
ax.set_ylim(0, 22)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── Color scheme ───────────────────────────────────────────────────────────
COLORS = {
    "data":     ("#DBEAFE", "#1E40AF"),   # blue
    "feat":     ("#D1FAE5", "#065F46"),   # green
    "split":    ("#EDE9FE", "#5B21B6"),   # violet
    "model":    ("#FEF3C7", "#92400E"),   # amber
    "eval":     ("#FCE7F3", "#9D174D"),   # pink
    "output":   ("#F0FDF4", "#166534"),   # green-dark
    "web":      ("#ECFDF5", "#047857"),   # teal
    "step_bg":  "#1E3A5F",               # step number circle
    "arrow":    "#475569",
    "conn":     "#94A3B8",               # lighter connectors
}


def step_circle(x, y, num, color="#1E3A5F"):
    """Draw a numbered step circle."""
    circle = plt.Circle((x, y), 0.28, facecolor=color, edgecolor="white",
                         linewidth=2, zorder=5, transform=ax.transData)
    ax.add_patch(circle)
    ax.text(x, y, str(num), ha="center", va="center", fontsize=10,
            fontweight="bold", color="white", zorder=6)


def box(x, y, w, h, title, desc, colors, title_size=10.5, desc_size=7.5):
    """Draw a styled box with title and description."""
    fc, ec = colors
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                       facecolor=fc, edgecolor=ec, linewidth=1.6,
                       zorder=2, transform=ax.transData)
    ax.add_patch(b)
    if desc:
        ax.text(x + w/2, y + h/2 + 0.15, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color="#1E293B", zorder=3)
        ax.text(x + w/2, y + h/2 - 0.18, desc, ha="center", va="center",
                fontsize=desc_size, color="#475569", zorder=3)
    else:
        ax.text(x + w/2, y + h/2, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color="#1E293B", zorder=3)


def arrow_down(x, y1, y2, color=COLORS["arrow"]):
    """Vertical arrow downward."""
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8),
                zorder=1)


def arrow_right(x1, y, x2, color=COLORS["arrow"]):
    """Horizontal arrow."""
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5),
                zorder=1)


def arrow_angled(x1, y1, x2, y2, color=COLORS["arrow"], lw=1.5):
    """Angled arrow."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw),
                zorder=1)


def bracket_line(x1, y, x2, color=COLORS["conn"], lw=1.2):
    """Horizontal bracket line."""
    ax.plot([x1, x2], [y, y], color=color, lw=lw, zorder=1)


# ══════════════════════════════════════════════════════════════════════════════
#  TITLE
# ══════════════════════════════════════════════════════════════════════════════
ax.text(6, 21.5, "System Architecture", ha="center", va="center",
        fontsize=18, fontweight="bold", color="#0F172A",
        fontfamily="sans-serif")
ax.text(6, 21.1, "NBL MAX PROFIT System v5.2 --- NEPSE Stock Direction Prediction",
        ha="center", va="center", fontsize=9, color="#64748B",
        fontfamily="sans-serif")
# Separator line
ax.plot([1, 11], [20.85, 20.85], color="#CBD5E1", lw=1.5, zorder=1)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1: DATA INPUT
# ══════════════════════════════════════════════════════════════════════════════
y = 19.8
step_circle(0.6, y + 0.35, 1)
ax.text(1.2, y + 0.35, "Data Input", fontsize=11, fontweight="bold",
        color="#1E293B", va="center")

box(1.5, y - 0.45, 4.0, 0.7, "Raw OHLCV CSV File",
    "Date | Open | High | Low | Close | Volume", COLORS["data"])
box(6.5, y - 0.45, 4.5, 0.7, "Flask Web Interface",
    "User uploads CSV + sets parameters via sidebar", COLORS["web"])

# Arrow from web to pipeline
ax.annotate("", xy=(5.5, y - 0.1), xytext=(6.5, y - 0.1),
            arrowprops=dict(arrowstyle="-|>", color="#047857", lw=1.3,
                            linestyle="--"), zorder=1)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2: DATA PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
y = 18.2
step_circle(0.6, y + 0.35, 2)
ax.text(1.2, y + 0.35, "Data Preprocessing", fontsize=11,
        fontweight="bold", color="#1E293B", va="center")

box(1.5, y - 0.45, 9.5, 0.7, "Clean & Prepare Data",
    "Chronological sort | Drop leaky columns (Symbol, direction) | Forward/Back-fill NaN",
    COLORS["data"])

arrow_down(6, 19.35, y + 0.25)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3: FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
y = 16.6
step_circle(0.6, y + 0.55, 3)
ax.text(1.2, y + 0.55, "Feature Engineering & Target Labeling", fontsize=11,
        fontweight="bold", color="#1E293B", va="center")

box(1.5, y - 0.55, 4.5, 1.0,
    "engineer_features()",
    "50+ indicators computed\n27 curated: Returns, RSI, ADX,\nBollinger, MACD, ATR (%-normalized)",
    COLORS["feat"], title_size=9.5, desc_size=6.5)

box(6.5, y - 0.55, 4.5, 1.0,
    "make_target()",
    "Triple-Barrier Labeling\nHorizon = 10 days, Threshold = 1.5%\nUP(1) / DOWN(0) / Drop ambiguous",
    COLORS["data"], title_size=9.5, desc_size=6.5)

arrow_down(3.75, 17.75, y + 0.45)
arrow_down(8.75, 17.75, y + 0.45)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4: DATA SPLITTING
# ══════════════════════════════════════════════════════════════════════════════
y = 14.5
step_circle(0.6, y + 0.55, 4)
ax.text(1.2, y + 0.55, "Chronological Data Split", fontsize=11,
        fontweight="bold", color="#1E293B", va="center")

# Three split boxes
box(1.5, y - 0.35, 3.0, 0.7, "Train (75%)",
    "1,656 rows | Model fitting", COLORS["split"], title_size=9)
box(4.7, y - 0.35, 3.0, 0.7, "Val (12.5%)",
    "276 rows | Threshold tuning", COLORS["split"], title_size=9)
box(7.9, y - 0.35, 3.0, 0.7, "Test (12.5%)",
    "276 rows | Final evaluation", COLORS["split"], title_size=9)

# Timeline arrow underneath
ax.annotate("", xy=(10.9, y - 0.55), xytext=(1.5, y - 0.55),
            arrowprops=dict(arrowstyle="-|>", color="#5B21B6", lw=1.0),
            zorder=1)
ax.text(6.2, y - 0.72, "Time (2012 ------> 2026)",
        ha="center", fontsize=6.5, color="#5B21B6", style="italic")

arrow_angled(3.75, 16.05, 3.0, y + 0.35)
arrow_angled(8.75, 16.05, 9.4, y + 0.35)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5: SCALING
# ══════════════════════════════════════════════════════════════════════════════
y = 12.8
step_circle(0.6, y + 0.35, 5)
ax.text(1.2, y + 0.35, "Feature Scaling", fontsize=11,
        fontweight="bold", color="#1E293B", va="center")

box(1.5, y - 0.45, 9.5, 0.7, "RobustScaler (Median + IQR)",
    "fit_transform(Train) --> transform(Val) --> transform(Test) | No data leakage",
    COLORS["feat"])

arrow_down(6, 14.15, y + 0.25)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6: ENSEMBLE MODEL
# ══════════════════════════════════════════════════════════════════════════════
y = 10.6
step_circle(0.6, y + 0.95, 6)
ax.text(1.2, y + 0.95, "Ensemble Model Training", fontsize=11,
        fontweight="bold", color="#1E293B", va="center")

# Outer ensemble box
ens = FancyBboxPatch((1.5, y - 0.45), 9.5, 1.6,
                     boxstyle="round,pad=0.15",
                     facecolor=COLORS["model"][0],
                     edgecolor=COLORS["model"][1],
                     linewidth=2.0, zorder=2)
ax.add_patch(ens)
ax.text(6.25, y + 0.95, "Soft Voting Ensemble (Weighted Average of Probabilities)",
        ha="center", fontsize=9, fontweight="bold", color="#78350F", zorder=3)

# Four model sub-boxes
mdls = [
    ("Random Forest",  "150 trees\ndepth=3, w=2", 1.7),
    ("Gradient Boost",  "120 trees\nlr=0.03, w=2", 4.1),
    ("XGBoost",         "120 trees\nL1+L2 reg, w=2", 6.5),
    ("Logistic Reg",    "C=0.3\nbalanced, w=1", 8.9),
]
for name, desc, mx in mdls:
    sb = FancyBboxPatch((mx, y - 0.30), 2.2, 0.85,
                        boxstyle="round,pad=0.08",
                        facecolor="#FFFBEB", edgecolor="#D97706",
                        linewidth=1.0, zorder=3)
    ax.add_patch(sb)
    ax.text(mx + 1.1, y + 0.28, name, ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="#92400E", zorder=4)
    ax.text(mx + 1.1, y - 0.02, desc, ha="center", va="center",
            fontsize=5.8, color="#78716C", zorder=4)

arrow_down(6.25, 12.35, y + 1.15)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7: TWO-STAGE TRAINING
# ══════════════════════════════════════════════════════════════════════════════
y = 8.6
step_circle(0.6, y + 0.55, 7)
ax.text(1.2, y + 0.55, "Two-Stage Training Strategy", fontsize=11,
        fontweight="bold", color="#1E293B", va="center")

box(1.5, y - 0.55, 4.5, 1.0,
    "Stage 1: Train Set Only",
    "5-fold TimeSeriesSplit CV (F1)\nOptimize threshold on Val set\nNarrow search: 0.47 -- 0.54",
    COLORS["eval"], title_size=9, desc_size=6.5)

box(6.5, y - 0.55, 4.5, 1.0,
    "Stage 2: Refit (Train+Val)",
    "87.5% total data for training\nFinal model for test evaluation\nMaximize recent market context",
    COLORS["eval"], title_size=9, desc_size=6.5)

# Stage arrow
arrow_right(6.0, y, 6.5, color="#9D174D")

arrow_down(3.75, 10.15, y + 0.45)
arrow_down(8.75, 10.15, y + 0.45)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 8: EVALUATION & BACKTESTING
# ══════════════════════════════════════════════════════════════════════════════
y = 6.4
step_circle(0.6, y + 0.65, 8)
ax.text(1.2, y + 0.65, "Evaluation & Backtesting", fontsize=11,
        fontweight="bold", color="#1E293B", va="center")

box(1.5, y - 0.55, 3.0, 1.0,
    "Classification\nMetrics",
    "Accuracy, F1, ROC-AUC\nMCC, Kappa, CM",
    COLORS["eval"], title_size=8.5, desc_size=6.5)

box(4.75, y - 0.55, 3.0, 1.0,
    "Walk-Forward\nValidation",
    "8-fold expanding window\nTrain+Val period only",
    COLORS["eval"], title_size=8.5, desc_size=6.5)

box(8.0, y - 0.55, 3.0, 1.0,
    "Realistic\nBacktesting",
    "ATR stop-loss, 0.3% fees\nEquity curve, Sharpe ratio",
    COLORS["eval"], title_size=8.5, desc_size=6.5)

arrow_angled(3.75, 8.05, 3.0, y + 0.45)
arrow_angled(8.75, 8.05, 9.5, y + 0.45)
arrow_down(6.25, 8.05, y + 0.45)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 9: VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
y = 4.6
step_circle(0.6, y + 0.35, 9)
ax.text(1.2, y + 0.35, "Visualization Suite", fontsize=11,
        fontweight="bold", color="#1E293B", va="center")

box(1.5, y - 0.45, 9.5, 0.7,
    "10+ Publication-Quality Charts",
    "Target Distribution | Box-Plots | Correlation Heatmap | Confusion Matrices | ROC-AUC | Learning Curves | Feature Importance | Equity Curve",
    ("#F1F5F9", "#334155"), title_size=9, desc_size=6)

arrow_down(6.25, 5.85, y + 0.25)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 10: PREDICTION OUTPUT
# ══════════════════════════════════════════════════════════════════════════════
y = 2.7
step_circle(0.6, y + 0.55, 10)
ax.text(1.2, y + 0.55, "Prediction & Web Output", fontsize=11,
        fontweight="bold", color="#1E293B", va="center")

# Three signal boxes
box(1.5, y - 0.55, 2.8, 0.9, "BUY",
    "P(UP) > 51%\nEnter position",
    ("#DCFCE7", "#166534"), title_size=14, desc_size=6.5)

box(4.6, y - 0.55, 2.8, 0.9, "SELL",
    "P(UP) < 41%\nStay out / Exit",
    ("#FEE2E2", "#991B1B"), title_size=14, desc_size=6.5)

box(7.7, y - 0.55, 2.8, 0.9, "WAIT",
    "41% < P < 51%\nNo conviction",
    ("#FEF9C3", "#854D0E"), title_size=14, desc_size=6.5)

arrow_down(6.25, 4.15, y + 0.35)

# ══════════════════════════════════════════════════════════════════════════════
#  BOTTOM: FLASK WEB
# ══════════════════════════════════════════════════════════════════════════════
y = 1.0
box(1.5, y - 0.35, 9.5, 0.7,
    "Flask Web Dashboard (Tailwind CSS)",
    "CSV Upload | Sidebar Configuration | Signal Cards | Prediction Table | Confidence Scores",
    COLORS["web"], title_size=9.5, desc_size=6.5)

arrow_angled(2.9, 2.15, 4.0, y + 0.35)
arrow_down(6.0, 2.15, y + 0.35)
arrow_angled(9.1, 2.15, 8.0, y + 0.35)

# ── Side annotation: Model Persistence ─────────────────────────────────────
ax.annotate("", xy=(11.4, y + 0.35), xytext=(11.4, 11.75),
            arrowprops=dict(arrowstyle="<->", color="#94A3B8", lw=1.0,
                            connectionstyle="arc3,rad=0", linestyle=":"),
            zorder=1)
ax.text(11.7, 6.5, "Model\nSave /\nLoad\n(pickle +\nJSON)",
        ha="center", va="center", fontsize=5.5, color="#94A3B8",
        fontweight="bold", linespacing=1.3)

# ══════════════════════════════════════════════════════════════════════════════
#  SAVE
# ══════════════════════════════════════════════════════════════════════════════
out = "nbl_v5_output/system_architecture.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white",
            edgecolor="none", pad_inches=0.3)
plt.close(fig)
print(f"[OK] System architecture diagram saved -> {out}")
