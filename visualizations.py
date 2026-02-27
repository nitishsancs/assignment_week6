"""
Visualization functions for Week 6 Dashboard.
Generates confusion matrices, specificity vs sensitivity plots,
distribution charts, feature importance plots, and metrics tables.
Returns matplotlib figure objects for use with Streamlit's st.pyplot(fig).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

from muller_loop import ModelResult, CLASS_ORDER


# Color palette for algorithms
ALGO_COLORS = {
    "XGBoost": "#2ecc71",
    "MLP": "#e74c3c",
    "Random Forest": "#3498db",
    "SVM": "#9b59b6",
}

# Color palette for severity classes
SEVERITY_COLORS = {
    "Critical": "#e74c3c",
    "High": "#e67e22",
    "Medium": "#f1c40f",
    "Low": "#2ecc71",
}


def plot_confusion_matrix(
    result: ModelResult,
    normalize: bool = True,
    title: Optional[str] = None,
):
    """Plot a confusion matrix heatmap for a single model result."""
    classes = [c for c in CLASS_ORDER if c in set(result.y_true) | set(result.y_pred)]
    cm = result.confusion_mat

    fig, ax = plt.subplots(figsize=(7, 6))

    if normalize and cm.sum() > 0:
        cm_display = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_display = cm_display / row_sums
        fmt = ".2f"
        vmax = 1.0
    else:
        cm_display = cm
        fmt = "d"
        vmax = cm.max() if cm.max() > 0 else 1

    sns.heatmap(
        cm_display, annot=True, fmt=fmt, cmap="YlOrRd",
        xticklabels=classes, yticklabels=classes, ax=ax,
        vmin=0, vmax=vmax, linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )

    title = title or f"Confusion Matrix — {result.algorithm}"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)

    if normalize:
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j + 0.5, i + 0.78, f"(n={cm[i, j]})",
                        ha="center", va="center", fontsize=7, color="gray")

    fig.tight_layout()
    return fig


def plot_all_confusion_matrices(results: Dict[str, ModelResult], normalize: bool = True):
    """Plot confusion matrices for all algorithms in a 2x2 grid."""
    n = len(results)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No results", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        return fig

    cols = 2
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx]
        classes = [c for c in CLASS_ORDER if c in set(result.y_true) | set(result.y_pred)]
        cm = result.confusion_mat
        if normalize and cm.sum() > 0:
            cm_d = cm.astype(float)
            rs = cm.sum(axis=1, keepdims=True); rs[rs == 0] = 1
            cm_d = cm_d / rs; fmt = ".2f"
        else:
            cm_d = cm; fmt = "d"

        sns.heatmap(cm_d, annot=True, fmt=fmt, cmap="YlOrRd",
                    xticklabels=classes, yticklabels=classes, ax=ax,
                    vmin=0, vmax=1.0 if normalize else (cm.max() or 1),
                    linewidths=0.5, linecolor="white")
        color = ALGO_COLORS.get(name, "#333")
        ax.set_title(f"{name} (F1={result.f1_macro:.3f})", fontsize=12, fontweight="bold", color=color)
        ax.set_ylabel("True"); ax.set_xlabel("Predicted")

    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Confusion Matrices — All Algorithms", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_specificity_vs_sensitivity(results: Dict[str, ModelResult]):
    """Plot specificity vs sensitivity for each class and algorithm."""
    fig, ax = plt.subplots(figsize=(9, 7))

    for name, result in results.items():
        color = ALGO_COLORS.get(name, "#333")
        classes = list(result.per_class_recall.keys())
        sens = [result.per_class_recall.get(c, 0) for c in classes]
        spec = [result.per_class_specificity.get(c, 0) for c in classes]

        ax.scatter(spec, sens, label=name, color=color, s=120,
                   edgecolors="white", linewidth=1.5, zorder=3)
        for i, cls in enumerate(classes):
            ax.annotate(cls, (spec[i], sens[i]), textcoords="offset points",
                        xytext=(8, 8), fontsize=8, color=color, alpha=0.8)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.3)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.2)
    ax.set_xlabel("Specificity (TNR)", fontsize=12)
    ax.set_ylabel("Sensitivity (Recall / TPR)", fontsize=12)
    ax.set_title("Specificity vs Sensitivity by Class & Algorithm", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def plot_metrics_comparison(baseline_results, current_results):
    """Plot side-by-side bar chart comparing baseline vs current F1 scores."""
    algos = list(set(baseline_results.keys()) | set(current_results.keys()))
    algos = sorted(algos, key=lambda x: list(ALGO_COLORS.keys()).index(x) if x in ALGO_COLORS else 99)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(algos))
    width = 0.35
    b_f1 = [baseline_results[a].f1_macro if a in baseline_results else 0 for a in algos]
    c_f1 = [current_results[a].f1_macro if a in current_results else 0 for a in algos]

    bars1 = ax.bar(x - width/2, b_f1, width, label="Baseline", color="#95a5a6", edgecolor="white")
    bars2 = ax.bar(x + width/2, c_f1, width, label="Current", color="#3498db", edgecolor="white")

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("F1 Score (Macro)"); ax.set_title("Baseline vs Current — F1 Comparison", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(algos, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(max(b_f1), max(c_f1)) * 1.15 + 0.05)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def plot_distribution_comparison(y_original: pd.Series, y_modified: pd.Series):
    """Plot before/after class distribution bar chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    orig_counts = y_original.value_counts().reindex(CLASS_ORDER).fillna(0)
    colors_o = [SEVERITY_COLORS.get(c, "#999") for c in orig_counts.index]
    ax1.bar(orig_counts.index, orig_counts.values, color=colors_o, edgecolor="white")
    ax1.set_title("Original Distribution", fontweight="bold")
    ax1.set_ylabel("Count")
    for i, v in enumerate(orig_counts.values):
        ax1.text(i, v + max(orig_counts.values)*0.02, str(int(v)), ha="center", fontweight="bold")

    mod_counts = y_modified.value_counts().reindex(CLASS_ORDER).fillna(0)
    colors_m = [SEVERITY_COLORS.get(c, "#999") for c in mod_counts.index]
    ax2.bar(mod_counts.index, mod_counts.values, color=colors_m, edgecolor="white")
    ax2.set_title("Modified Distribution", fontweight="bold")
    ax2.set_ylabel("Count")
    for i, v in enumerate(mod_counts.values):
        ax2.text(i, v + max(mod_counts.values)*0.02, str(int(v)), ha="center", fontweight="bold")

    max_y = max(orig_counts.max(), mod_counts.max()) * 1.15
    ax1.set_ylim(0, max_y); ax2.set_ylim(0, max_y)
    fig.suptitle("Class Distribution: Before vs After", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, method_name: str = "Gini", top_n: int = 15):
    """Plot horizontal bar chart of feature importances."""
    df = importance_df.head(top_n).copy().iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.4)))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df)))
    ax.barh(df["feature"], df["importance"], color=colors, edgecolor="white")
    ax.set_xlabel("Importance Score"); ax.set_title(f"Feature Importance — {method_name}", fontweight="bold")
    ax.tick_params(axis="y", labelsize=9); ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return fig


def plot_feature_distribution_histogram(df_orig, df_mod, feature_name, feature_type):
    """Plot histogram/bar of a specific feature before/after modification."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    if feature_type in ("binary", "categorical", "target"):
        oc = df_orig[feature_name].value_counts().sort_index()
        mc = df_mod[feature_name].value_counts().sort_index()
        ax1.bar(oc.index.astype(str), oc.values, color="#3498db", edgecolor="white")
        ax1.set_title(f"Original — {feature_name}", fontweight="bold"); ax1.tick_params(axis="x", rotation=45)
        ax2.bar(mc.index.astype(str), mc.values, color="#e74c3c", edgecolor="white")
        ax2.set_title(f"Modified — {feature_name}", fontweight="bold"); ax2.tick_params(axis="x", rotation=45)
    else:
        bins = min(50, max(10, len(df_orig) // 100))
        ax1.hist(df_orig[feature_name].dropna(), bins=bins, color="#3498db", edgecolor="white", alpha=0.8)
        ax1.set_title(f"Original — {feature_name}", fontweight="bold")
        ax2.hist(df_mod[feature_name].dropna(), bins=bins, color="#e74c3c", edgecolor="white", alpha=0.8)
        ax2.set_title(f"Modified — {feature_name}", fontweight="bold")

    ax1.set_ylabel("Count"); ax2.set_ylabel("Count")
    fig.suptitle(f"Feature Distribution: {feature_name}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_radar_chart(results: Dict[str, ModelResult]):
    """Plot radar/spider chart comparing algorithms across multiple metrics."""
    metrics = ["F1 Macro", "Accuracy", "Precision", "Recall", "ROC AUC"]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    for name, result in results.items():
        values = [result.f1_macro, result.accuracy, result.precision_macro,
                  result.recall_macro, result.roc_auc if result.roc_auc else 0]
        values += values[:1]
        color = ALGO_COLORS.get(name, "#333")
        ax.plot(angles, values, "o-", linewidth=2, label=name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Algorithm Comparison — Radar Chart", fontsize=14, fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True)
    fig.tight_layout()
    return fig
