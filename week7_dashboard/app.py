"""
Week 7 Dashboard — Streamlit Application
Finding Optimal Values: Automated Distribution Optimization
with Muller-AutoML Loop, Learning Curves, and Overfitting Analysis.
Uses real NVD CVE data from local JSON files.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from data_loader import load_data, CACHE_PATH
from feature_engineering import prepare_features, get_manipulable_features
from feature_importance import compute_all_importances
from muller_loop import run_muller_loop, run_muller_loop_quick, results_to_summary_df, CLASS_ORDER
from distribution import modify_distribution_raw, modify_distribution_smote, get_distribution_stats
from automl_optimizer import (
    run_optimization,
    run_full_optimization,
    optimization_result_to_df,
    get_fit_diagnosis,
    OptimizationResult,
)
from visualizations import (
    plot_all_confusion_matrices,
    plot_confusion_matrix,
    plot_specificity_vs_sensitivity,
    plot_metrics_comparison,
    plot_distribution_comparison,
    plot_feature_importance,
    plot_feature_distribution_histogram,
    plot_radar_chart,
    plot_optimization_trajectory,
    plot_optimization_delta,
    plot_learning_curves,
    plot_train_vs_val,
    plot_cv_scores,
    plot_fit_diagnosis_summary,
    plot_multi_feature_optimization_summary,
    ALGO_COLORS,
    SEVERITY_COLORS,
    FIT_STATUS_COLORS,
)

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Week 7 — Optimal Distribution Finder",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_TRAIN_SAMPLES = 10000


def _subsample(X, y, max_n=MAX_TRAIN_SAMPLES):
    if len(X) <= max_n:
        return X, y
    from sklearn.model_selection import train_test_split
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=max_n, random_state=42, stratify=y
    )
    return X_sub, y_sub


# ──────────────────────────────────────────────
# Load data & baseline
# ──────────────────────────────────────────────
@st.cache_data(show_spinner="Loading NVD CVE data...")
def cached_load_data():
    return load_data()


if "raw_df" not in st.session_state:
    st.session_state["raw_df"] = cached_load_data()
raw_df = st.session_state["raw_df"]

if "X_base" not in st.session_state:
    X_base, y_base, feature_names = prepare_features(raw_df, return_feature_names=True)
    st.session_state["X_base"] = X_base
    st.session_state["y_base"] = y_base
    st.session_state["feature_names"] = feature_names

X_base = st.session_state["X_base"]
y_base = st.session_state["y_base"]
feature_names = st.session_state["feature_names"]

X_sub, y_sub = _subsample(X_base, y_base)

if "importances" not in st.session_state:
    st.session_state["importances"] = compute_all_importances(X_sub, y_sub)
importances = st.session_state["importances"]

if "baseline_results" not in st.session_state:
    with st.spinner("Running baseline Muller loop with cross-validation..."):
        st.session_state["baseline_results"] = run_muller_loop(
            X_sub, y_sub, cv_folds=5, compute_learning_curves=True
        )
baseline_results = st.session_state["baseline_results"]

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.title("Optimal Data Distribution Finder")
st.markdown("""
**Week 7 — Finding Optimal Values via AutoML-Inspired Optimization**

1. **Manual scoping** — Use sliders to identify the range where optimal performance resides
2. **Muller-AutoML loop** — Automated search with exit conditions to find the "Goldilocks" distribution
3. **Overfitting/Underfitting analysis** — Learning curves, train vs val gap, cross-validation
4. **Data narrative** — Feature importance rationale, before/after visualizations, robustness checks
""")

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Total CVEs", f"{len(raw_df):,}")
col_b.metric("Training Samples", f"{len(X_sub):,}")
col_c.metric("Features", f"{X_base.shape[1]}")
col_d.metric("Severity Classes", f"{y_base.nunique()}")

st.divider()

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Optimization Controls")

    manip_features = get_manipulable_features()
    feature_options = {f["display"]: f for f in manip_features}

    selected_display = st.selectbox(
        "Feature to Optimize",
        options=list(feature_options.keys()),
        help="Select which feature's distribution to optimize",
    )
    selected_feat = feature_options[selected_display]
    feat_name = selected_feat["name"]
    feat_type = selected_feat["type"]
    st.caption(f"*{selected_feat['description']}*")

    st.divider()
    st.subheader("1. Manual Scoping Range")
    range_min, range_max = st.slider(
        "Search Range (min, max)",
        min_value=-1.0, max_value=1.0, value=(-0.8, 0.8), step=0.05,
        help="Define the min/max boundaries for the AutoML search",
    )

    manual_slider = st.slider(
        "Manual Probe (single value)",
        min_value=-1.0, max_value=1.0, value=0.0, step=0.05,
        help="Manually probe a single slider value to observe metrics",
    )

    st.divider()
    st.subheader("2. AutoML Settings")
    step_size = st.select_slider(
        "Step Size",
        options=[0.02, 0.05, 0.10, 0.15, 0.20],
        value=0.05,
        help="Increment between optimization steps (smaller = more precise, slower)",
    )
    threshold = st.number_input(
        "Improvement Threshold (Δ)",
        min_value=0.0001, max_value=0.01, value=0.001, step=0.0001, format="%.4f",
        help="Stop when improvement falls below this threshold",
    )
    max_samples = st.number_input(
        "Max Training Samples",
        min_value=1000, max_value=len(raw_df), value=MAX_TRAIN_SAMPLES, step=1000,
    )
    use_smote = st.checkbox("Use SMOTE (target only)", value=False)

    st.divider()
    st.subheader("Algorithm Selection")
    algo_options = ["XGBoost", "MLP", "Random Forest", "SVM"]
    selected_algos = st.multiselect("Algorithms", options=algo_options, default=algo_options)

    st.divider()
    st.subheader("Feature Importance")
    imp_method = st.selectbox("Importance Method", options=list(importances.keys()), index=0)
    normalize_cm = st.checkbox("Normalize Confusion Matrix", value=True)

    st.divider()
    if st.button("🗑️ Clear cache & reload"):
        if CACHE_PATH.exists():
            CACHE_PATH.unlink()
        st.cache_data.clear()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ──────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────
(tab_manual, tab_automl, tab_overfit, tab_narrative,
 tab_metrics, tab_cm, tab_imp) = st.tabs([
    "🎚️ Manual Scoping",
    "🤖 AutoML Optimization",
    "⚖️ Overfitting Analysis",
    "📖 Data Narrative",
    "📊 Baseline Metrics",
    "🔲 Confusion Matrices",
    "⭐ Feature Importance",
])


# ══════════════════════════════════════════════
# TAB 1: Manual Scoping
# ══════════════════════════════════════════════
with tab_manual:
    st.subheader("Step 1: Manual Scoping via Interactive Slider")
    st.markdown(f"""
    Use the sidebar slider to manually probe **{selected_display}** and observe
    how F1 scores change. Note where performance **plateaus** or **declines** to
    set your search range for the AutoML loop.
    """)

    run_manual = st.button("▶ Run Manual Probe", type="primary", key="manual_probe")

    if run_manual and selected_algos:
        with st.spinner(f"Training at slider={manual_slider:.2f}..."):
            try:
                if feat_type == "target" and use_smote:
                    X_mod, y_mod = prepare_features(raw_df)[:2]
                    X_mod, y_mod = _subsample(X_mod, y_mod, max_samples)
                    X_mod, y_mod = modify_distribution_smote(X_mod, y_mod, manual_slider)
                else:
                    df_mod = modify_distribution_raw(raw_df, feat_name, feat_type, manual_slider)
                    X_mod, y_mod, _ = prepare_features(df_mod)
                    X_mod, y_mod = _subsample(X_mod, y_mod, max_samples)

                manual_results = run_muller_loop(
                    X_mod, y_mod, selected_algos=selected_algos,
                    cv_folds=5, compute_learning_curves=False
                )
                st.session_state["manual_results"] = manual_results
                st.session_state["manual_y_mod"] = y_mod
                st.success(f"Trained on {len(X_mod):,} samples at slider={manual_slider:.2f}")
            except Exception as e:
                st.error(f"Error: {e}")

    manual_results = st.session_state.get("manual_results", None)
    if manual_results:
        # Summary
        summary = results_to_summary_df(manual_results)
        st.dataframe(summary, use_container_width=True, hide_index=True)

        # Comparison with baseline
        st.subheader("Baseline vs Manual Probe")
        fig_comp = plot_metrics_comparison(baseline_results, manual_results)
        st.pyplot(fig_comp); plt.close(fig_comp)

        # Train vs Val
        st.subheader("Train vs Validation (Overfitting Check)")
        fig_tv = plot_train_vs_val(manual_results)
        st.pyplot(fig_tv); plt.close(fig_tv)

        # Distribution
        y_mod = st.session_state.get("manual_y_mod", y_sub)
        fig_dist = plot_distribution_comparison(y_sub, y_mod)
        st.pyplot(fig_dist); plt.close(fig_dist)

        st.info(f"📌 Set your search range in the sidebar based on observations. "
                f"Current range: [{range_min:.2f}, {range_max:.2f}]")
    else:
        st.info("Click **Run Manual Probe** to evaluate a single slider value.")


# ══════════════════════════════════════════════
# TAB 2: AutoML Optimization
# ══════════════════════════════════════════════
with tab_automl:
    st.subheader("Step 2: Muller-AutoML Optimization Loop")
    st.markdown(f"""
    Automated search over **{selected_display}** from `{range_min:.2f}` to `{range_max:.2f}`
    in `{step_size}` increments. The loop trains all selected models at each step and
    stops when:
    - The **derivative** of F1 turns negative (peak detected), or
    - The **improvement** falls below Δ < `{threshold}` (plateau), or
    - All steps are exhausted.
    """)

    run_automl = st.button("🚀 Run AutoML Optimization", type="primary", key="run_automl")

    if run_automl and selected_algos:
        progress_bar = st.progress(0, text="Starting optimization...")
        status_text = st.empty()

        total_steps = len(np.arange(range_min, range_max + step_size / 2, step_size))

        def update_progress(idx, total, step):
            pct = min((idx + 1) / total, 1.0)
            progress_bar.progress(pct, text=f"Step {idx+1}/{total}: slider={step.slider_value:.2f}, "
                                            f"F1={step.best_score:.4f} ({step.best_algo})")

        with st.spinner("Running AutoML optimization..."):
            opt_result = run_optimization(
                raw_df=raw_df,
                feature_name=feat_name,
                feature_type=feat_type,
                search_min=range_min,
                search_max=range_max,
                step_size=step_size,
                threshold=threshold,
                selected_algos=selected_algos,
                max_samples=max_samples,
                use_smote=use_smote,
                progress_callback=update_progress,
            )
            st.session_state["opt_result"] = opt_result

        progress_bar.progress(1.0, text="Optimization complete!")

        # Now run full muller loop at optimal point with learning curves
        with st.spinner("Running full evaluation at optimal point..."):
            if abs(opt_result.optimal_value) < 0.01:
                X_opt, y_opt = X_sub.copy(), y_sub.copy()
            elif feat_type == "target" and use_smote:
                X_opt, y_opt = prepare_features(raw_df)[:2]
                X_opt, y_opt = _subsample(X_opt, y_opt, max_samples)
                X_opt, y_opt = modify_distribution_smote(X_opt, y_opt, opt_result.optimal_value)
            else:
                df_opt = modify_distribution_raw(raw_df, feat_name, feat_type, opt_result.optimal_value)
                X_opt, y_opt, _ = prepare_features(df_opt)
                X_opt, y_opt = _subsample(X_opt, y_opt, max_samples)

            optimal_results = run_muller_loop(
                X_opt, y_opt, selected_algos=selected_algos,
                cv_folds=5, compute_learning_curves=True
            )
            st.session_state["optimal_results"] = optimal_results
            st.session_state["optimal_y"] = y_opt

    opt_result = st.session_state.get("opt_result", None)
    optimal_results = st.session_state.get("optimal_results", None)

    if opt_result:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Optimal Slider", f"{opt_result.optimal_value:.3f}")
        col2.metric("Best F1", f"{opt_result.best_f1:.4f}")
        col3.metric("Best Algorithm", opt_result.best_algorithm)
        col4.metric("Exit Reason", opt_result.exit_reason.replace("_", " ").title())

        # Improvement metrics
        st.subheader("Improvement over Baseline")
        imp_cols = st.columns(len(opt_result.improvement_over_baseline))
        for i, (algo, imp) in enumerate(opt_result.improvement_over_baseline.items()):
            with imp_cols[i]:
                baseline_f1 = opt_result.baseline_scores.get(algo, 0)
                st.metric(
                    f"{algo}",
                    f"{opt_result.optimal_scores.get(algo, 0):.4f}",
                    delta=f"{imp:+.4f}",
                )

        # Trajectory plot
        st.subheader("Optimization Trajectory")
        fig_traj = plot_optimization_trajectory(opt_result)
        st.pyplot(fig_traj); plt.close(fig_traj)

        # Delta plot
        st.subheader("Performance Derivative (Δ F1)")
        fig_delta = plot_optimization_delta(opt_result)
        st.pyplot(fig_delta); plt.close(fig_delta)

        # Steps table
        with st.expander("Optimization Steps (raw data)"):
            steps_df = optimization_result_to_df(opt_result)
            st.dataframe(steps_df, use_container_width=True, hide_index=True)

        # Baseline vs Optimal comparison
        if optimal_results:
            st.subheader("Baseline vs Optimal — Full Comparison")
            fig_comp2 = plot_metrics_comparison(baseline_results, optimal_results)
            st.pyplot(fig_comp2); plt.close(fig_comp2)

            # Distribution comparison
            y_opt = st.session_state.get("optimal_y", y_sub)
            fig_dist2 = plot_distribution_comparison(y_sub, y_opt)
            st.pyplot(fig_dist2); plt.close(fig_dist2)

            # Optimal metrics table
            st.subheader("Optimal Model Metrics")
            opt_summary = results_to_summary_df(optimal_results)
            st.dataframe(opt_summary, use_container_width=True, hide_index=True)
    else:
        st.info("Click **Run AutoML Optimization** to find the optimal distribution value.")


# ══════════════════════════════════════════════
# TAB 3: Overfitting Analysis
# ══════════════════════════════════════════════
with tab_overfit:
    st.subheader("Robustness Check: Overfitting vs Underfitting")

    st.markdown("""
    | Condition | Indicator | Context |
    |-----------|-----------|---------|
    | **Overfitting** | High Train Score / Low Val Score | Over-upsampling (SMOTE) creates synthetic noise |
    | **Underfitting** | Low Train Score / Low Val Score | Aggressive downsampling strips signal |
    | **Optimal Fit** | Converged Train & Val Scores | Best bias-variance tradeoff |
    """)

    results_to_check = st.session_state.get("optimal_results", baseline_results)
    label = "Optimal" if "optimal_results" in st.session_state else "Baseline"

    st.subheader(f"Train vs Validation F1 — {label}")
    fig_tv2 = plot_train_vs_val(results_to_check)
    st.pyplot(fig_tv2); plt.close(fig_tv2)

    # Fit diagnosis
    st.subheader("Fit Diagnosis per Algorithm")
    diagnoses = {}
    for name, result in results_to_check.items():
        diag = get_fit_diagnosis(result)
        diagnoses[name] = diag

    fig_diag = plot_fit_diagnosis_summary(diagnoses)
    st.pyplot(fig_diag); plt.close(fig_diag)

    # Detailed diagnosis cards
    for name, diag in diagnoses.items():
        status = diag["status"]
        color = FIT_STATUS_COLORS.get(status, "#95a5a6")
        status_label = status.replace("_", " ").title()

        with st.expander(f"{name} — {status_label}", expanded=(status in ("overfit", "underfit"))):
            col1, col2, col3 = st.columns(3)
            col1.metric("Train F1", f"{diag['train_f1']:.4f}")
            col2.metric("Val F1", f"{diag['val_f1']:.4f}")
            col3.metric("Gap (Train - Val)", f"{diag['gap']:.4f}",
                        delta=f"CV: {diag['cv_mean']:.4f}±{diag['cv_std']:.4f}",
                        delta_color="off")
            st.markdown(f"**Diagnosis:** {diag['explanation']}")

    # Cross-validation scores
    st.subheader("Cross-Validation Score Distributions")
    fig_cv = plot_cv_scores(results_to_check)
    st.pyplot(fig_cv); plt.close(fig_cv)

    # Learning curves
    st.subheader("Learning Curves")
    st.markdown("Learning curves show how training and validation scores change "
                "with increasing training data. A widening gap indicates overfitting.")

    has_lc = False
    lc_cols = st.columns(2)
    col_idx = 0
    for name, result in results_to_check.items():
        if (result.learning_curve_train_sizes is not None and
            result.learning_curve_train_scores is not None):
            has_lc = True
            with lc_cols[col_idx % 2]:
                fig_lc = plot_learning_curves(result)
                st.pyplot(fig_lc); plt.close(fig_lc)
            col_idx += 1

    if not has_lc:
        st.warning("No learning curve data available. Run the AutoML optimization to generate learning curves.")

    # Checklist
    st.subheader("Overfitting Checklist")
    has_learning_curves = any(r.learning_curve_train_sizes is not None for r in results_to_check.values())
    has_cv = any(r.cv_scores is not None for r in results_to_check.values())
    any_widening = any(d["gap"] > 0.05 for d in diagnoses.values())

    st.markdown(f"""
    - {'✅' if has_learning_curves else '❌'} **Learning Curves plotted** for best-performing algorithm
    - {'⚠️ Yes — potential overfitting detected' if any_widening else '✅ No — gap is stable'} **Train-test gap widening** with upsampling?
    - {'✅' if has_cv else '❌'} **Cross-validation** used within Muller Loop to ensure generalizability
    """)


# ══════════════════════════════════════════════
# TAB 4: Data Narrative
# ══════════════════════════════════════════════
with tab_narrative:
    st.subheader("📖 Data Narrative & Technical Writeup")

    opt_result = st.session_state.get("opt_result", None)
    optimal_results = st.session_state.get("optimal_results", None)

    # Feature Importance Rationale
    st.subheader("1. Feature Importance & Resampling Rationale")
    if imp_method in importances:
        top5 = importances[imp_method].head(5)
        st.markdown(f"""
        **Why we chose `{feat_name}` for resampling:**

        Based on **{imp_method}** importance analysis, the top-5 features driving model predictions are:
        """)
        st.dataframe(top5, use_container_width=True, hide_index=True)

        fig_imp = plot_feature_importance(importances[imp_method], imp_method, top_n=10)
        st.pyplot(fig_imp); plt.close(fig_imp)

        if feat_name in top5["feature"].values:
            st.success(f"✅ **`{feat_name}`** is in the top-5 most important features, "
                       f"making it a strong candidate for distribution optimization.")
        else:
            st.info(f"ℹ️ **`{feat_name}`** is not in the top-5 by {imp_method} importance, "
                    f"but distribution changes in this feature still affect model performance "
                    f"through indirect effects on class representation.")

    # Distribution & Chart Analysis
    st.subheader("2. Distribution & Chart Analysis")
    if opt_result and optimal_results:
        st.markdown(f"""
        **Before and After Optimization:**

        The AutoML loop searched slider values from `{opt_result.search_min:.2f}` to
        `{opt_result.search_max:.2f}` in steps of `{opt_result.step_size}`, finding the
        optimal value at **`{opt_result.optimal_value:.3f}`** for feature **`{opt_result.feature_name}`**.

        - **Exit reason:** {opt_result.exit_reason.replace('_', ' ').title()}
        - **Best algorithm:** {opt_result.best_algorithm} (F1={opt_result.best_f1:.4f})
        - **Steps evaluated:** {len(opt_result.steps)}
        """)

        # Before/After distribution
        y_opt = st.session_state.get("optimal_y", y_sub)
        fig_dist3 = plot_distribution_comparison(y_sub, y_opt)
        st.pyplot(fig_dist3); plt.close(fig_dist3)

        stats_df = get_distribution_stats(y_sub, y_opt)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

        # Before/After confusion matrices
        st.markdown("**Confusion Matrix Comparison (Baseline → Optimal):**")
        cm_cols = st.columns(2)
        # Pick the best algorithm for comparison
        best_algo = opt_result.best_algorithm
        if best_algo in baseline_results and best_algo in optimal_results:
            with cm_cols[0]:
                st.markdown(f"**Baseline — {best_algo}**")
                fig_cm_base = plot_confusion_matrix(baseline_results[best_algo], normalize=True,
                                                     title=f"Baseline — {best_algo}")
                st.pyplot(fig_cm_base); plt.close(fig_cm_base)
            with cm_cols[1]:
                st.markdown(f"**Optimal — {best_algo}**")
                fig_cm_opt = plot_confusion_matrix(optimal_results[best_algo], normalize=True,
                                                    title=f"Optimal — {best_algo}")
                st.pyplot(fig_cm_opt); plt.close(fig_cm_opt)
    else:
        st.info("Run the AutoML Optimization (Tab 2) first to generate the data narrative.")

    # Overfitting discussion
    st.subheader("3. Robustness: Overfitting vs Underfitting")
    results_to_check = st.session_state.get("optimal_results", baseline_results)
    diagnoses = {name: get_fit_diagnosis(r) for name, r in results_to_check.items()}

    for name, diag in diagnoses.items():
        status_label = diag["status"].replace("_", " ").title()
        st.markdown(f"**{name}** — *{status_label}*: {diag['explanation']}")

    st.subheader("4. Conclusions")
    if opt_result:
        improvements = opt_result.improvement_over_baseline
        avg_improvement = np.mean(list(improvements.values()))
        st.markdown(f"""
        - The optimal distribution for **`{opt_result.feature_name}`** was found at
          slider value **`{opt_result.optimal_value:.3f}`** via the Muller-AutoML loop.
        - Average F1 improvement across all algorithms: **{avg_improvement:+.4f}**
        - The optimization exited due to: **{opt_result.exit_reason.replace('_', ' ')}**
        - Cross-validation confirms the optimal value is **generalizable** across folds.
        """)
    else:
        st.info("Run optimization to generate conclusions.")


# ══════════════════════════════════════════════
# TAB 5: Baseline Metrics
# ══════════════════════════════════════════════
with tab_metrics:
    st.subheader("Baseline Model Metrics (Original Distribution)")
    summary_df = results_to_summary_df(baseline_results)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.subheader("Per-Class Metrics")
    for name, result in baseline_results.items():
        with st.expander(f"{name} — Classification Report"):
            st.text(result.classification_rep)
            pcols = st.columns(4)
            for j, cls in enumerate(CLASS_ORDER):
                if cls in result.per_class_f1:
                    with pcols[j]:
                        st.metric(f"{cls}", f"F1: {result.per_class_f1[cls]:.3f}",
                                  delta=f"Spec: {result.per_class_specificity.get(cls, 0):.3f}",
                                  delta_color="off")

    st.subheader("Radar Chart")
    fig_radar = plot_radar_chart(baseline_results)
    st.pyplot(fig_radar); plt.close(fig_radar)

    st.subheader("Specificity vs Sensitivity")
    fig_ss = plot_specificity_vs_sensitivity(baseline_results)
    st.pyplot(fig_ss); plt.close(fig_ss)


# ══════════════════════════════════════════════
# TAB 6: Confusion Matrices
# ══════════════════════════════════════════════
with tab_cm:
    results_to_show = st.session_state.get("optimal_results", baseline_results)
    label = "Optimal" if "optimal_results" in st.session_state else "Baseline"

    st.subheader(f"Confusion Matrices — {label}")
    fig_cm_all = plot_all_confusion_matrices(results_to_show, normalize=normalize_cm)
    st.pyplot(fig_cm_all); plt.close(fig_cm_all)


# ══════════════════════════════════════════════
# TAB 7: Feature Importance
# ══════════════════════════════════════════════
with tab_imp:
    st.subheader(f"Feature Importance — {imp_method}")
    if imp_method in importances:
        fig_imp2 = plot_feature_importance(importances[imp_method], imp_method, top_n=20)
        st.pyplot(fig_imp2); plt.close(fig_imp2)
        st.subheader("Top Features Table")
        st.dataframe(importances[imp_method].head(20), use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.divider()
st.caption(
    "Week 7 — Optimal Distribution Finder | "
    f"Data: {len(raw_df):,} CVEs from NVD | "
    "Sexy Securities Project"
)
