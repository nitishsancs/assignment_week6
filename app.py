"""
Week 6 Dashboard — Streamlit Application
Interactive dashboard for exploring how data distribution changes
affect ML model metrics on CVE severity classification.
Uses real NVD CVE data from local JSON files.
"""

import sys
import gc
from pathlib import Path

# Ensure the dashboard package is importable
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from data_loader import load_data, CACHE_PATH
from feature_engineering import prepare_features, get_manipulable_features
from feature_importance import compute_all_importances
from muller_loop import run_muller_loop, results_to_summary_df, CLASS_ORDER
from distribution import modify_distribution_raw, modify_distribution_smote, get_distribution_stats
from visualizations import (
    plot_all_confusion_matrices,
    plot_specificity_vs_sensitivity,
    plot_metrics_comparison,
    plot_distribution_comparison,
    plot_feature_importance,
    plot_feature_distribution_histogram,
    plot_radar_chart,
    ALGO_COLORS,
    SEVERITY_COLORS,
)

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="CVE Distribution Analysis — Week 6",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────
# Caching helpers
# ──────────────────────────────────────────────
@st.cache_data(show_spinner="Loading NVD CVE data...")
def cached_load_data():
    return load_data()


@st.cache_data(show_spinner="Engineering features...")
def cached_prepare_features(_df):
    return prepare_features(_df, return_feature_names=True)


@st.cache_data(show_spinner="Computing feature importances (this may take a minute)...")
def cached_importances(_X, _y):
    result = compute_all_importances(_X, _y)
    gc.collect()
    return result


@st.cache_data(show_spinner="Running baseline Muller loop (4 algorithms)...")
def cached_baseline_muller(_X, _y):
    result = run_muller_loop(_X, _y)
    gc.collect()
    return result


# ──────────────────────────────────────────────
# Initialize data (all cached to survive reruns)
# ──────────────────────────────────────────────
raw_df = cached_load_data()

X_base, y_base, feature_names = cached_prepare_features(raw_df)
gc.collect()

# Subsample for training (SVM is O(n^2), so we cap training data)
MAX_TRAIN_SAMPLES = 5000

def _subsample(X, y, max_n=MAX_TRAIN_SAMPLES):
    """Stratified subsample if data exceeds max_n."""
    if len(X) <= max_n:
        return X, y
    from sklearn.model_selection import train_test_split
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=max_n, random_state=42, stratify=y
    )
    return X_sub, y_sub

X_train_base, y_train_base = _subsample(X_base, y_base)

importances = cached_importances(X_train_base, y_train_base)

baseline_results = cached_baseline_muller(X_train_base, y_train_base)

if "current_results" not in st.session_state:
    st.session_state["current_results"] = dict(baseline_results)

current_results = st.session_state["current_results"]

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.title("🔐 CVE Data Distribution Analysis Dashboard")
st.markdown("""
**Week 6 — Changing Data Distribution & Impact on Model Metrics**

Modify the distribution of a selected feature (or target class balance),
re-run the **Muller loop** (XGBoost, MLP, Random Forest, SVM),
and visualize how metrics change dynamically.
Real NVD CVE data loaded from local JSON files.
""")

# Dataset summary
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Total CVEs", f"{len(raw_df):,}")
col_b.metric("Features", f"{X_base.shape[1]}")
col_c.metric("Severity Classes", f"{y_base.nunique()}")
col_d.metric("Year Range", f"{raw_df['year_published'].min()}-{raw_df['year_published'].max()}")

st.divider()

# ──────────────────────────────────────────────
# Sidebar — Controls
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Distribution Controls")

    manip_features = get_manipulable_features()
    feature_options = {f["display"]: f for f in manip_features}

    selected_display = st.selectbox(
        "Feature to Manipulate",
        options=list(feature_options.keys()),
        help="Select which feature's distribution to modify",
    )
    selected_feat = feature_options[selected_display]
    feat_name = selected_feat["name"]
    feat_type = selected_feat["type"]

    st.caption(f"*{selected_feat['description']}*")

    slider_value = st.slider(
        "Distribution Change",
        min_value=-1.0, max_value=1.0, value=0.0, step=0.05,
        help="← Downsample | No change | Upsample →",
    )

    max_samples = st.number_input(
        "Max Training Samples",
        min_value=1000, max_value=len(raw_df), value=MAX_TRAIN_SAMPLES, step=1000,
        help="Cap training data size (SVM is slow on large datasets)",
    )

    use_smote = st.checkbox(
        "Use SMOTE (target class only)",
        value=False,
        help="Use SMOTE synthetic oversampling instead of random oversampling",
    )

    st.divider()
    st.subheader("Algorithm Selection")
    algo_options = ["XGBoost", "MLP", "Random Forest", "SVM"]
    selected_algos = st.multiselect(
        "Algorithms to run",
        options=algo_options,
        default=algo_options,
    )

    st.divider()
    run_clicked = st.button(
        "▶ Re-run Muller Loop",
        type="primary",
        use_container_width=True,
    )

    st.divider()
    st.subheader("Feature Importance")
    imp_method = st.selectbox(
        "Importance Method",
        options=list(importances.keys()),
        index=0,
    )

    normalize_cm = st.checkbox("Normalize Confusion Matrix", value=True)

    st.divider()
    if st.button("🗑️ Clear cache & reload data"):
        if CACHE_PATH.exists():
            CACHE_PATH.unlink()
        st.cache_data.clear()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ──────────────────────────────────────────────
# Run Muller loop on button click
# ──────────────────────────────────────────────
y_current = y_base  # default for distribution plot

if run_clicked:
    if not selected_algos:
        st.error("Select at least one algorithm.")
    else:
        with st.spinner("Modifying distribution and retraining models..."):
            try:
                if feat_type == "target" and use_smote:
                    X_mod, y_mod = prepare_features(raw_df)[:2]
                    X_mod, y_mod = _subsample(X_mod, y_mod, max_samples)
                    X_mod, y_mod = modify_distribution_smote(X_mod, y_mod, slider_value)
                else:
                    df_mod = modify_distribution_raw(raw_df, feat_name, feat_type, slider_value)
                    X_mod, y_mod, _ = prepare_features(df_mod)
                    X_mod, y_mod = _subsample(X_mod, y_mod, max_samples)

                new_results = run_muller_loop(X_mod, y_mod, selected_algos=selected_algos)
                st.session_state["current_results"] = new_results
                st.session_state["y_current"] = y_mod
                current_results = new_results
                y_current = y_mod

                st.success(
                    f"Done! Trained on **{len(X_mod):,}** samples "
                    f"(slider={slider_value:+.2f}, feature={feat_name})"
                )
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                traceback.print_exc()
else:
    y_current = st.session_state.get("y_current", y_base)
    current_results = st.session_state.get("current_results", baseline_results)


# ──────────────────────────────────────────────
# Main content — Tabs
# ──────────────────────────────────────────────
tab_metrics, tab_cm, tab_spec, tab_dist, tab_imp, tab_radar = st.tabs([
    "📊 Metrics",
    "🔲 Confusion Matrices",
    "📈 Specificity vs Sensitivity",
    "📉 Distribution",
    "⭐ Feature Importance",
    "🎯 Radar Chart",
])


# --- Tab 1: Metrics ---
with tab_metrics:
    st.subheader("Model Performance Metrics")

    summary_df = results_to_summary_df(current_results)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    if current_results != baseline_results:
        st.subheader("Baseline vs Current Comparison")
        cols = st.columns(len(current_results))
        for i, (name, result) in enumerate(current_results.items()):
            with cols[i]:
                if name in baseline_results:
                    delta = result.f1_macro - baseline_results[name].f1_macro
                    st.metric(
                        f"{name} F1",
                        f"{result.f1_macro:.4f}",
                        delta=f"{delta:+.4f}",
                    )
                else:
                    st.metric(f"{name} F1", f"{result.f1_macro:.4f}")

        fig_comp = plot_metrics_comparison(baseline_results, current_results)
        st.pyplot(fig_comp)
        plt.close(fig_comp)

    st.subheader("Per-Class Metrics")
    for name, result in current_results.items():
        with st.expander(f"{name} — Classification Report"):
            st.text(result.classification_rep)

            pcols = st.columns(4)
            for j, cls in enumerate(CLASS_ORDER):
                if cls in result.per_class_f1:
                    with pcols[j]:
                        st.metric(
                            f"{cls}",
                            f"F1: {result.per_class_f1[cls]:.3f}",
                            delta=f"Spec: {result.per_class_specificity.get(cls, 0):.3f}",
                            delta_color="off",
                        )


# --- Tab 2: Confusion Matrices ---
with tab_cm:
    st.subheader("Confusion Matrices — All Algorithms")
    fig_cm = plot_all_confusion_matrices(current_results, normalize=normalize_cm)
    st.pyplot(fig_cm)
    plt.close(fig_cm)

    st.subheader("Individual Confusion Matrices")
    from visualizations import plot_confusion_matrix
    cm_cols = st.columns(2)
    for i, (name, result) in enumerate(current_results.items()):
        with cm_cols[i % 2]:
            fig_single = plot_confusion_matrix(result, normalize=normalize_cm)
            st.pyplot(fig_single)
            plt.close(fig_single)


# --- Tab 3: Specificity vs Sensitivity ---
with tab_spec:
    st.subheader("Specificity vs Sensitivity by Class & Algorithm")
    fig_ss = plot_specificity_vs_sensitivity(current_results)
    st.pyplot(fig_ss)
    plt.close(fig_ss)

    st.subheader("Specificity & Sensitivity Table")
    spec_rows = []
    for name, result in current_results.items():
        for cls in CLASS_ORDER:
            if cls in result.per_class_recall:
                spec_rows.append({
                    "Algorithm": name,
                    "Class": cls,
                    "Sensitivity (Recall)": round(result.per_class_recall[cls], 4),
                    "Specificity (TNR)": round(result.per_class_specificity.get(cls, 0), 4),
                    "Precision": round(result.per_class_precision.get(cls, 0), 4),
                    "F1": round(result.per_class_f1.get(cls, 0), 4),
                })
    if spec_rows:
        st.dataframe(pd.DataFrame(spec_rows), use_container_width=True, hide_index=True)


# --- Tab 4: Distribution ---
with tab_dist:
    st.subheader("Class Distribution: Original vs Modified")
    fig_dist = plot_distribution_comparison(y_base, y_current)
    st.pyplot(fig_dist)
    plt.close(fig_dist)

    stats_df = get_distribution_stats(y_base, y_current)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    if feat_name in raw_df.columns and feat_name != "severity" and slider_value != 0.0:
        st.subheader(f"Feature Distribution: {feat_name}")
        df_mod_hist = modify_distribution_raw(raw_df, feat_name, feat_type, slider_value)
        fig_fdist = plot_feature_distribution_histogram(raw_df, df_mod_hist, feat_name, feat_type)
        st.pyplot(fig_fdist)
        plt.close(fig_fdist)


# --- Tab 5: Feature Importance ---
with tab_imp:
    st.subheader(f"Feature Importance — {imp_method}")
    if imp_method in importances:
        fig_imp = plot_feature_importance(importances[imp_method], imp_method, top_n=20)
        st.pyplot(fig_imp)
        plt.close(fig_imp)

        st.subheader("Top Features Table")
        st.dataframe(importances[imp_method].head(20), use_container_width=True, hide_index=True)
    else:
        st.warning(f"No importance data for method: {imp_method}")


# --- Tab 6: Radar ---
with tab_radar:
    st.subheader("Algorithm Comparison — Radar Chart")
    fig_radar = plot_radar_chart(current_results)
    st.pyplot(fig_radar)
    plt.close(fig_radar)

    st.subheader("Algorithm Summary")
    summary_df2 = results_to_summary_df(current_results)
    st.dataframe(summary_df2, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.divider()
st.caption(
    "CVE Distribution Analysis Dashboard | Week 6 Assignment | "
    f"Data: {len(raw_df):,} CVEs from NVD | "
    "Sexy Securities Project"
)
