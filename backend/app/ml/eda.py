"""
Exploratory Data Analysis (EDA) Module.
Generates various plots as base64-encoded PNG images for the frontend.
Includes histograms, box plots, correlation heatmap, scatter plots,
confusion matrix, ROC curves, actual vs predicted, and residual plots.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from io import BytesIO
import base64
import warnings
warnings.filterwarnings("ignore")

# ── Theme ────────────────────────────────────────────────────
BG_COLOR = "#0f172a"
CARD_COLOR = "#1e293b"
TEXT_COLOR = "#e2e8f0"
ACCENT_COLOR = "#818cf8"
GRID_COLOR = "#334155"


def _setup_style():
    """Apply dark theme globally."""
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": CARD_COLOR,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "grid.color": GRID_COLOR,
        "font.size": 10,
    })


def _fig_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=BG_COLOR)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def generate_histograms(df, target_column, max_cols=8):
    """Generate histograms for numeric columns."""
    _setup_style()
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    numeric_cols = numeric_cols[:max_cols]

    if not numeric_cols:
        return None

    n_cols = min(4, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col].dropna(), bins=30, color=ACCENT_COLOR, alpha=0.8, edgecolor="#6366f1")
        axes[i].set_title(col, fontsize=10, fontweight="bold")
        axes[i].set_xlabel("")

    for j in range(len(numeric_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _fig_to_base64(fig)


def generate_box_plots(df, target_column, max_cols=8):
    """Generate box plots for numeric columns."""
    _setup_style()
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    numeric_cols = numeric_cols[:max_cols]

    if not numeric_cols:
        return None

    n_cols = min(4, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        bp = axes[i].boxplot(df[col].dropna(), patch_artist=True, widths=0.6)
        bp["boxes"][0].set_facecolor(ACCENT_COLOR)
        bp["boxes"][0].set_alpha(0.7)
        for element in ["whiskers", "caps", "medians"]:
            for line in bp[element]:
                line.set_color(TEXT_COLOR)
        axes[i].set_title(col, fontsize=10, fontweight="bold")

    for j in range(len(numeric_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Box Plots", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _fig_to_base64(fig)


def generate_correlation_heatmap(df, target_column):
    """Generate a correlation heatmap for numeric columns."""
    _setup_style()
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    if numeric_df.shape[1] < 2:
        return None

    # Limit to 15 columns for readability
    if numeric_df.shape[1] > 15:
        numeric_df = numeric_df.iloc[:, :15]

    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))

    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=cmap,
                center=0, ax=ax, linewidths=0.5, linecolor=GRID_COLOR,
                cbar_kws={"shrink": 0.8})

    ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _fig_to_base64(fig)


def generate_target_distribution(df, target_column, problem_type):
    """Generate target column distribution plot."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    if problem_type in ("binary", "multiclass"):
        counts = df[target_column].value_counts()
        colors = sns.color_palette("viridis", len(counts))
        ax.bar(counts.index.astype(str), counts.values, color=colors, edgecolor="#1e293b")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_title(f"Target Distribution: {target_column}", fontsize=14, fontweight="bold")
    else:
        ax.hist(df[target_column].dropna(), bins=40, color=ACCENT_COLOR, alpha=0.8, edgecolor="#6366f1")
        ax.set_xlabel(target_column)
        ax.set_ylabel("Frequency")
        ax.set_title(f"Target Distribution: {target_column}", fontsize=14, fontweight="bold")

    fig.tight_layout()
    return _fig_to_base64(fig)


def generate_scatter_plots(df, target_column, max_pairs=4):
    """Generate scatter plots for top correlated numeric feature pairs."""
    _setup_style()
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)

    if len(numeric_cols) < 2:
        return None

    # Find top correlated pairs
    corr = df[numeric_cols].corr().abs()
    pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            pairs.append((numeric_cols[i], numeric_cols[j], corr.iloc[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = pairs[:max_pairs]

    n_cols = min(2, len(top_pairs))
    n_rows = (len(top_pairs) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (col1, col2, corr_val) in enumerate(top_pairs):
        axes[i].scatter(df[col1], df[col2], alpha=0.4, c=ACCENT_COLOR, s=15, edgecolors="none")
        axes[i].set_xlabel(col1)
        axes[i].set_ylabel(col2)
        axes[i].set_title(f"{col1} vs {col2} (r={corr_val:.2f})", fontsize=10, fontweight="bold")

    for j in range(len(top_pairs), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Scatter Plots (Top Correlations)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _fig_to_base64(fig)


def generate_confusion_matrix_plot(confusion_mat, class_labels=None):
    """Generate a confusion matrix heatmap."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    cm = np.array(confusion_mat)
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", ax=ax,
                linewidths=1, linecolor=GRID_COLOR,
                xticklabels=class_labels, yticklabels=class_labels)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

    fig.tight_layout()
    return _fig_to_base64(fig)


def generate_roc_curve(model, X_test, y_test, n_classes):
    """Generate ROC curve plot."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    if not hasattr(model, "predict_proba"):
        return None

    if n_classes == 2:
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=ACCENT_COLOR, lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.5)
        ax.fill_between(fpr, tpr, alpha=0.15, color=ACCENT_COLOR)
    else:
        y_bin = label_binarize(y_test, classes=sorted(y_test.unique()))
        y_proba = model.predict_proba(X_test)
        colors = sns.color_palette("viridis", n_classes)

        for i in range(min(n_classes, y_bin.shape[1])):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i], lw=2,
                    label=f"Class {i} (AUC = {roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.5)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    return _fig_to_base64(fig)


def generate_actual_vs_predicted(y_test, y_pred):
    """Generate actual vs predicted scatter plot for regression."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(y_test, y_pred, alpha=0.4, c=ACCENT_COLOR, s=20, edgecolors="none")

    # Perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], "w--", lw=2, alpha=0.7, label="Perfect Prediction")

    ax.set_xlabel("Actual Values", fontsize=12)
    ax.set_ylabel("Predicted Values", fontsize=12)
    ax.set_title("Actual vs Predicted", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    return _fig_to_base64(fig)


def generate_residual_plot(y_test, y_pred):
    """Generate residual plot for regression."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    residuals = np.array(y_test) - np.array(y_pred)
    ax.scatter(y_pred, residuals, alpha=0.4, c=ACCENT_COLOR, s=20, edgecolors="none")
    ax.axhline(y=0, color="#ef4444", linestyle="--", lw=2, alpha=0.7)

    ax.set_xlabel("Predicted Values", fontsize=12)
    ax.set_ylabel("Residuals", fontsize=12)
    ax.set_title("Residual Plot", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    return _fig_to_base64(fig)


def generate_all_eda(df, target_column, problem_type):
    """
    Generate all EDA visualizations for the dataset.
    Returns a dict of base64-encoded plot images.
    """
    plots = {}

    plots["histograms"] = generate_histograms(df, target_column)
    plots["box_plots"] = generate_box_plots(df, target_column)
    plots["correlation_heatmap"] = generate_correlation_heatmap(df, target_column)
    plots["target_distribution"] = generate_target_distribution(df, target_column, problem_type)
    plots["scatter_plots"] = generate_scatter_plots(df, target_column)

    # Remove None entries
    plots = {k: v for k, v in plots.items() if v is not None}
    return plots
