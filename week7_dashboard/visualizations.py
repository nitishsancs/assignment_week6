"""
Visualization functions for Week 7 Dashboard.
Includes all Week 6 plots plus:
- Optimization trajectory plots
- Learning curves
- Overfitting/underfitting indicators
- Before/after confusion matrix comparison
- Train vs Validation score plots
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


ALGO_COLORS = {
    "XGBoost": "#2ecc71",
    "MLP": "#e74c3c",
    "Random Forest": "#3498db",
    "SVM": "#9b59b6",
}

SEVERITY_COLORS = {
    "Critical": "#e74c3c",
    "High": "#e67e22",
    "Medium": "#f1c40f",
    "Low": "#2ecc71",
}

FIT_STATUS_COLORS = {
    "overfit": "#e74c3c",
    "slight_overfit": "#e67e22",
    "underfit": "#3498db",
    "optimal": "#2ecc71",
    "acceptable": "#95a5a6",
}


# ── Week 6 plots (carried over) ──────────────────────────

def plot_confusion_matrix(result: ModelResult, normalize: bool = True, title: Optional[str] = None):
    """Plot a confusion matrix heatmap for a single model result."""
    classes = [c for c in CLASS_ORDER if c in set(result.y_true) | set(result.y_pred)]
    cm = result.confusion_mat
    fig, ax = plt.subplots(figsize=(7, 6))
    if normalize and cm.sum() > 0:
        cm_display = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_display = cm_display / row_sums
        fmt = ".2f"; vmax = 1.0
    else:
        cm_display = cm; fmt = "d"; vmax = cm.max() if cm.max() > 0 else 1
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap="YlOrRd",
                xticklabels=classes, yticklabels=classes, ax=ax,
                vmin=0, vmax=vmax, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Proportion" if normalize else "Count"})
    title = title or f"Confusion Matrix — {result.algorithm}"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
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
    cols = 2; rows = (n + 1) // 2
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
    ax.set_xlabel("Specificity (TNR)", fontsize=12)
    ax.set_ylabel("Sensitivity (Recall / TPR)", fontsize=12)
    ax.set_title("Specificity vs Sensitivity", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def plot_metrics_comparison(baseline_results, current_results):
    """Plot side-by-side bar chart comparing baseline vs current F1 scores."""
    algos = sorted(set(baseline_results.keys()) | set(current_results.keys()),
                   key=lambda x: list(ALGO_COLORS.keys()).index(x) if x in ALGO_COLORS else 99)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(algos)); width = 0.35
    b_f1 = [baseline_results[a].f1_macro if a in baseline_results else 0 for a in algos]
    c_f1 = [current_results[a].f1_macro if a in current_results else 0 for a in algos]
    bars1 = ax.bar(x - width/2, b_f1, width, label="Baseline", color="#95a5a6", edgecolor="white")
    bars2 = ax.bar(x + width/2, c_f1, width, label="Optimal", color="#3498db", edgecolor="white")
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("F1 Score (Macro)"); ax.set_title("Baseline vs Optimal — F1 Comparison", fontweight="bold")
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
    ax1.set_title("Original Distribution", fontweight="bold"); ax1.set_ylabel("Count")
    for i, v in enumerate(orig_counts.values):
        ax1.text(i, v + max(orig_counts.values)*0.02, str(int(v)), ha="center", fontweight="bold")
    mod_counts = y_modified.value_counts().reindex(CLASS_ORDER).fillna(0)
    colors_m = [SEVERITY_COLORS.get(c, "#999") for c in mod_counts.index]
    ax2.bar(mod_counts.index, mod_counts.values, color=colors_m, edgecolor="white")
    ax2.set_title("Optimized Distribution", fontweight="bold"); ax2.set_ylabel("Count")
    for i, v in enumerate(mod_counts.values):
        ax2.text(i, v + max(mod_counts.values)*0.02, str(int(v)), ha="center", fontweight="bold")
    max_y = max(orig_counts.max(), mod_counts.max()) * 1.15
    ax1.set_ylim(0, max_y); ax2.set_ylim(0, max_y)
    fig.suptitle("Class Distribution: Before vs After Optimization", fontsize=15, fontweight="bold")
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
        ax2.set_title(f"Optimized — {feature_name}", fontweight="bold"); ax2.tick_params(axis="x", rotation=45)
    else:
        bins = min(50, max(10, len(df_orig) // 100))
        ax1.hist(df_orig[feature_name].dropna(), bins=bins, color="#3498db", edgecolor="white", alpha=0.8)
        ax1.set_title(f"Original — {feature_name}", fontweight="bold")
        ax2.hist(df_mod[feature_name].dropna(), bins=bins, color="#e74c3c", edgecolor="white", alpha=0.8)
        ax2.set_title(f"Optimized — {feature_name}", fontweight="bold")
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


# ── Week 7 NEW plots ─────────────────────────────────────

def plot_optimization_trajectory(opt_result, algos_to_show: Optional[List[str]] = None):
    """
    Plot F1 scores across slider values for each algorithm.
    Shows the optimization trajectory with optimal point marked.
    """
    from automl_optimizer import optimization_result_to_df

    df = optimization_result_to_df(opt_result)
    if df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No optimization data", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=(12, 6))

    f1_cols = [c for c in df.columns if c.endswith("_f1")]
    for col in f1_cols:
        algo_name = col.replace("_f1", "")
        if algos_to_show and algo_name not in algos_to_show:
            continue
        color = ALGO_COLORS.get(algo_name, "#333")
        ax.plot(df["slider_value"], df[col], "o-", label=algo_name,
                color=color, linewidth=2, markersize=4, alpha=0.8)

    # Mark optimal point
    ax.axvline(x=opt_result.optimal_value, color="#e74c3c", linestyle="--",
               linewidth=2, alpha=0.7, label=f"Optimal ({opt_result.optimal_value:.2f})")

    # Mark baseline (slider=0)
    ax.axvline(x=0, color="#95a5a6", linestyle=":", linewidth=1.5, alpha=0.5, label="Baseline (0.0)")

    ax.set_xlabel("Slider Value (← Downsample | Upsample →)", fontsize=12)
    ax.set_ylabel("F1 Score (Macro)", fontsize=12)
    ax.set_title(f"Optimization Trajectory — {opt_result.feature_name}\n"
                 f"Exit: {opt_result.exit_reason} | Best: {opt_result.best_f1:.4f} "
                 f"({opt_result.best_algorithm})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def plot_optimization_delta(opt_result):
    """Plot the derivative (delta) of F1 score at each optimization step."""
    from automl_optimizer import optimization_result_to_df

    df = optimization_result_to_df(opt_result)
    if df.empty or len(df) < 2:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "Not enough data", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["#2ecc71" if d >= 0 else "#e74c3c" for d in df["delta"]]
    ax.bar(df["slider_value"], df["delta"], width=0.04, color=colors, edgecolor="white", alpha=0.8)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.axvline(x=opt_result.optimal_value, color="#e74c3c", linestyle="--", linewidth=2, alpha=0.7)
    ax.set_xlabel("Slider Value", fontsize=12)
    ax.set_ylabel("Δ F1 (step-to-step)", fontsize=12)
    ax.set_title(f"Performance Derivative — {opt_result.feature_name}", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def plot_learning_curves(result: ModelResult):
    """Plot learning curves (training score vs validation score vs training size)."""
    if (result.learning_curve_train_sizes is None or
        result.learning_curve_train_scores is None or
        result.learning_curve_val_scores is None):
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.text(0.5, 0.5, f"No learning curve data for {result.algorithm}",
                ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        return fig

    train_sizes = result.learning_curve_train_sizes
    train_mean = result.learning_curve_train_scores.mean(axis=1)
    train_std = result.learning_curve_train_scores.std(axis=1)
    val_mean = result.learning_curve_val_scores.mean(axis=1)
    val_std = result.learning_curve_val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="#2ecc71")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="#e74c3c")
    ax.plot(train_sizes, train_mean, "o-", color="#2ecc71", label="Training Score", linewidth=2)
    ax.plot(train_sizes, val_mean, "o-", color="#e74c3c", label="Validation Score", linewidth=2)

    # Annotate the gap
    gap = train_mean[-1] - val_mean[-1]
    ax.annotate(f"Gap: {gap:.4f}",
                xy=(train_sizes[-1], (train_mean[-1] + val_mean[-1]) / 2),
                fontsize=11, fontweight="bold",
                color="#e67e22", ha="right")

    ax.set_xlabel("Training Set Size", fontsize=12)
    ax.set_ylabel("F1 Score (Macro)", fontsize=12)
    ax.set_title(f"Learning Curve — {result.algorithm}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def plot_train_vs_val(results: Dict[str, ModelResult]):
    """Plot train vs validation F1 for all algorithms (overfitting indicator)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    algos = list(results.keys())
    x = np.arange(len(algos))
    width = 0.35

    train_scores = [results[a].train_f1 for a in algos]
    val_scores = [results[a].val_f1 for a in algos]

    bars1 = ax.bar(x - width/2, train_scores, width, label="Train F1", color="#2ecc71", edgecolor="white", alpha=0.8)
    bars2 = ax.bar(x + width/2, val_scores, width, label="Validation F1", color="#e74c3c", edgecolor="white", alpha=0.8)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    # Annotate gap
    for i, algo in enumerate(algos):
        gap = train_scores[i] - val_scores[i]
        color = "#e74c3c" if gap > 0.05 else "#2ecc71" if gap < 0.03 else "#e67e22"
        ax.annotate(f"Δ={gap:.3f}", xy=(i, max(train_scores[i], val_scores[i]) + 0.02),
                    ha="center", fontsize=9, fontweight="bold", color=color)

    ax.set_ylabel("F1 Score (Macro)", fontsize=12)
    ax.set_title("Train vs Validation F1 — Overfitting Check", fontsize=14, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(algos, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(max(train_scores), max(val_scores)) * 1.15 + 0.05)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def plot_cv_scores(results: Dict[str, ModelResult]):
    """Plot cross-validation score distributions as box plots."""
    fig, ax = plt.subplots(figsize=(10, 6))

    data = []
    labels = []
    colors_list = []
    for name, result in results.items():
        if result.cv_scores is not None and len(result.cv_scores) > 0:
            data.append(result.cv_scores)
            labels.append(name)
            colors_list.append(ALGO_COLORS.get(name, "#333"))

    if not data:
        ax.text(0.5, 0.5, "No CV data available", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Add mean markers
    for i, (d, name) in enumerate(zip(data, labels)):
        ax.scatter(i + 1, np.mean(d), marker="D", color="white", edgecolors="black",
                   s=60, zorder=5, label=f"Mean" if i == 0 else "")
        ax.text(i + 1.15, np.mean(d), f"{np.mean(d):.4f}", fontsize=9, va="center")

    ax.set_ylabel("F1 Score (Macro)", fontsize=12)
    ax.set_title("Cross-Validation Score Distributions", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def plot_fit_diagnosis_summary(diagnoses: Dict[str, dict]):
    """Plot a summary of overfitting/underfitting diagnoses for all algorithms."""
    if not diagnoses:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No diagnosis data", ha="center", va="center")
        return fig

    fig, axes = plt.subplots(1, len(diagnoses), figsize=(4 * len(diagnoses), 5))
    if len(diagnoses) == 1:
        axes = [axes]

    for ax, (algo, diag) in zip(axes, diagnoses.items()):
        status = diag["status"]
        color = FIT_STATUS_COLORS.get(status, "#95a5a6")

        # Gauge-style visualization
        categories = ["Train F1", "Val F1", "CV Mean"]
        values = [diag["train_f1"], diag["val_f1"], diag["cv_mean"]]
        bars = ax.barh(categories, values, color=[color]*3, edgecolor="white", alpha=0.8)

        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{val:.4f}", va="center", fontsize=10)

        status_label = status.replace("_", " ").title()
        ax.set_title(f"{algo}\n({status_label})", fontsize=12, fontweight="bold", color=color)
        ax.set_xlim(0, 1.15)
        ax.grid(axis="x", alpha=0.2)

    fig.suptitle("Fit Diagnosis Summary", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_gap_widening(gap_data: list):
    """
    Plot how the train-val F1 gap changes across slider values.
    gap_data is a list of dicts: [{"slider": float, "algo": str, "train_f1": float, "val_f1": float, "gap": float}, ...]
    Shows whether upsampling causes the gap to widen (overfitting signal).
    """
    if not gap_data:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, "No gap data available", ha="center", va="center")
        return fig

    df = pd.DataFrame(gap_data)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    # Top: Train & Val F1 lines per algorithm
    for algo in df["algo"].unique():
        adf = df[df["algo"] == algo].sort_values("slider")
        color = ALGO_COLORS.get(algo, "#333")
        ax1.plot(adf["slider"], adf["train_f1"], "o-", color=color, linewidth=2,
                 markersize=4, alpha=0.8, label=f"{algo} Train")
        ax1.plot(adf["slider"], adf["val_f1"], "s--", color=color, linewidth=2,
                 markersize=4, alpha=0.5, label=f"{algo} Val")

    ax1.set_ylabel("F1 Score (Macro)", fontsize=12)
    ax1.set_title("Train vs Validation F1 Across Distribution Shifts\n"
                   "(Gap widening = overfitting signal)", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=8, ncol=4, loc="lower left")
    ax1.grid(True, alpha=0.2)

    # Bottom: Gap magnitude per algorithm
    for algo in df["algo"].unique():
        adf = df[df["algo"] == algo].sort_values("slider")
        color = ALGO_COLORS.get(algo, "#333")
        ax2.plot(adf["slider"], adf["gap"], "o-", color=color, linewidth=2,
                 markersize=4, label=algo)

    ax2.axhline(y=0.05, color="#e74c3c", linestyle="--", alpha=0.5, label="Overfit threshold (0.05)")
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.fill_between(df["slider"].unique(), 0.05, df["gap"].max() * 1.2 if df["gap"].max() > 0.05 else 0.1,
                     alpha=0.05, color="#e74c3c")
    ax2.set_xlabel("Slider Value (← Downsample | Upsample →)", fontsize=12)
    ax2.set_ylabel("Gap (Train - Val F1)", fontsize=12)
    ax2.set_title("Train-Val Gap Across Distribution Shifts", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    return fig


def plot_multi_feature_optimization_summary(opt_results: Dict):
    """Plot a summary comparing optimal values across multiple features."""
    if not opt_results:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No optimization results", ha="center", va="center")
        return fig

    features = list(opt_results.keys())
    optimal_vals = [opt_results[f].optimal_value for f in features]
    best_f1s = [opt_results[f].best_f1 for f in features]
    baseline_f1s = [max(opt_results[f].baseline_scores.values()) if opt_results[f].baseline_scores else 0
                    for f in features]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Optimal slider values
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in optimal_vals]
    ax1.barh(features, optimal_vals, color=colors, edgecolor="white", alpha=0.8)
    ax1.axvline(x=0, color="black", linewidth=1)
    ax1.set_xlabel("Optimal Slider Value")
    ax1.set_title("Optimal Distribution Shift per Feature", fontweight="bold")
    for i, v in enumerate(optimal_vals):
        ax1.text(v + 0.02 if v >= 0 else v - 0.02, i, f"{v:.2f}",
                 va="center", ha="left" if v >= 0 else "right", fontweight="bold")
    ax1.grid(axis="x", alpha=0.2)

    # F1 improvement
    x = np.arange(len(features)); width = 0.35
    ax2.barh(x - width/2, baseline_f1s, width, label="Baseline", color="#95a5a6", edgecolor="white")
    ax2.barh(x + width/2, best_f1s, width, label="Optimal", color="#2ecc71", edgecolor="white")
    ax2.set_yticks(x); ax2.set_yticklabels(features)
    ax2.set_xlabel("Best F1 Score (Macro)")
    ax2.set_title("Baseline vs Optimal F1", fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(axis="x", alpha=0.2)

    fig.suptitle("Multi-Feature Optimization Summary", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig
