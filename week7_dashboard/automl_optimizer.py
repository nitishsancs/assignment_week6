"""
AutoML Optimizer for Week 7 Dashboard.
Implements the "Muller-AutoML" optimization loop that searches through
distribution ranges to find the optimal data distribution value for
each feature-algorithm pair.

Exit conditions:
1. Derivative of performance metric turns negative (peak detected)
2. Improvement falls below threshold (plateau detected)
3. Maximum iterations reached
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from copy import deepcopy

from feature_engineering import prepare_features
from distribution import modify_distribution_raw, modify_distribution_smote
from muller_loop import run_muller_loop_quick, run_muller_loop


@dataclass
class OptimizationStep:
    """Single step in the optimization loop."""
    slider_value: float
    scores: Dict[str, float]  # algo_name -> f1_macro
    best_algo: str
    best_score: float
    n_samples: int
    delta: float = 0.0  # improvement from previous step


@dataclass
class OptimizationResult:
    """Complete result of an optimization run for one feature."""
    feature_name: str
    feature_type: str
    search_min: float
    search_max: float
    step_size: float
    steps: List[OptimizationStep]
    optimal_value: float
    optimal_scores: Dict[str, float]
    exit_reason: str  # "peak_detected", "plateau", "max_iterations", "complete"
    best_algorithm: str
    best_f1: float
    baseline_scores: Dict[str, float]
    improvement_over_baseline: Dict[str, float]


def _subsample(X, y, max_n=10000):
    """Stratified subsample if data exceeds max_n."""
    if len(X) <= max_n:
        return X, y
    from sklearn.model_selection import train_test_split
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=max_n, random_state=42, stratify=y
    )
    return X_sub, y_sub


def run_optimization(
    raw_df: pd.DataFrame,
    feature_name: str,
    feature_type: str,
    search_min: float = -1.0,
    search_max: float = 1.0,
    step_size: float = 0.05,
    threshold: float = 0.001,
    selected_algos: Optional[List[str]] = None,
    max_samples: int = 10000,
    metric: str = "f1_macro",
    use_smote: bool = False,
    progress_callback: Optional[Callable] = None,
) -> OptimizationResult:
    """
    Run the AutoML optimization loop over a distribution range.

    Iterates through slider values from search_min to search_max in increments
    of step_size. At each step, modifies the distribution, trains all selected
    models, and records the F1 score. Stops early if:
      - The derivative turns negative (peak detected)
      - Improvement falls below threshold (plateau)

    Args:
        raw_df: Raw CVE DataFrame.
        feature_name: Feature to manipulate.
        feature_type: Type of feature (target/binary/continuous/categorical).
        search_min: Start of search range.
        search_max: End of search range.
        step_size: Increment between steps.
        threshold: Minimum improvement to continue.
        selected_algos: Algorithms to evaluate.
        max_samples: Max training samples.
        metric: Metric to optimize (f1_macro).
        use_smote: Use SMOTE for target feature.
        progress_callback: Optional callable(step_idx, total_steps, step_result).

    Returns:
        OptimizationResult with complete trajectory and optimal value.
    """
    # Generate slider values
    slider_values = np.arange(search_min, search_max + step_size / 2, step_size)
    slider_values = np.round(slider_values, 3)
    total_steps = len(slider_values)

    # Compute baseline (slider=0)
    X_base, y_base, _ = prepare_features(raw_df)
    X_base, y_base = _subsample(X_base, y_base, max_samples)
    baseline_scores = run_muller_loop_quick(X_base, y_base, selected_algos=selected_algos)

    steps = []
    best_avg_score = -1.0
    best_slider = 0.0
    best_scores = dict(baseline_scores)
    prev_avg_score = None
    consecutive_declines = 0
    exit_reason = "complete"

    for idx, sv in enumerate(slider_values):
        try:
            # Modify distribution
            if abs(sv) < 0.01:
                X_mod, y_mod = X_base.copy(), y_base.copy()
            elif feature_type == "target" and use_smote:
                X_mod, y_mod = prepare_features(raw_df)[:2]
                X_mod, y_mod = _subsample(X_mod, y_mod, max_samples)
                X_mod, y_mod = modify_distribution_smote(X_mod, y_mod, sv)
            else:
                df_mod = modify_distribution_raw(raw_df, feature_name, feature_type, sv)
                X_mod, y_mod, _ = prepare_features(df_mod)
                X_mod, y_mod = _subsample(X_mod, y_mod, max_samples)

            # Train and evaluate
            scores = run_muller_loop_quick(X_mod, y_mod, selected_algos=selected_algos)
            n_samples = len(X_mod)

            # Compute average score across algorithms
            avg_score = np.mean(list(scores.values())) if scores else 0.0
            best_algo = max(scores, key=scores.get) if scores else "N/A"
            best_score_this_step = max(scores.values()) if scores else 0.0

            # Delta from previous step
            delta = (avg_score - prev_avg_score) if prev_avg_score is not None else 0.0

            step = OptimizationStep(
                slider_value=float(sv),
                scores=scores,
                best_algo=best_algo,
                best_score=best_score_this_step,
                n_samples=n_samples,
                delta=delta,
            )
            steps.append(step)

            # Track global best
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                best_slider = float(sv)
                best_scores = dict(scores)
                consecutive_declines = 0
            else:
                consecutive_declines += 1

            # Progress callback
            if progress_callback:
                progress_callback(idx, total_steps, step)

            # Exit conditions
            if prev_avg_score is not None:
                # 1. Derivative turned negative after a peak
                if delta < -threshold and consecutive_declines >= 2:
                    exit_reason = "peak_detected"
                    break

                # 2. Plateau: improvement too small for multiple steps
                if abs(delta) < threshold and consecutive_declines >= 3:
                    exit_reason = "plateau"
                    break

            prev_avg_score = avg_score

        except Exception as e:
            print(f"  Step {sv:.2f} failed: {e}")
            continue

    # Find best algorithm at optimal point
    best_algorithm = max(best_scores, key=best_scores.get) if best_scores else "N/A"
    best_f1 = max(best_scores.values()) if best_scores else 0.0

    # Improvement over baseline
    improvement = {}
    for algo in best_scores:
        if algo in baseline_scores and baseline_scores[algo] > 0:
            improvement[algo] = best_scores[algo] - baseline_scores[algo]
        else:
            improvement[algo] = 0.0

    return OptimizationResult(
        feature_name=feature_name,
        feature_type=feature_type,
        search_min=search_min,
        search_max=search_max,
        step_size=step_size,
        steps=steps,
        optimal_value=best_slider,
        optimal_scores=best_scores,
        exit_reason=exit_reason,
        best_algorithm=best_algorithm,
        best_f1=best_f1,
        baseline_scores=baseline_scores,
        improvement_over_baseline=improvement,
    )


def run_full_optimization(
    raw_df: pd.DataFrame,
    features: List[dict],
    search_min: float = -1.0,
    search_max: float = 1.0,
    step_size: float = 0.05,
    threshold: float = 0.001,
    selected_algos: Optional[List[str]] = None,
    max_samples: int = 10000,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, OptimizationResult]:
    """
    Run optimization for multiple features.

    Args:
        raw_df: Raw CVE DataFrame.
        features: List of feature dicts from get_manipulable_features().
        Other args: Same as run_optimization.

    Returns:
        Dict mapping feature_name -> OptimizationResult.
    """
    results = {}
    for feat in features:
        print(f"\n=== Optimizing: {feat['display']} ===")
        result = run_optimization(
            raw_df=raw_df,
            feature_name=feat["name"],
            feature_type=feat["type"],
            search_min=search_min,
            search_max=search_max,
            step_size=step_size,
            threshold=threshold,
            selected_algos=selected_algos,
            max_samples=max_samples,
            progress_callback=progress_callback,
        )
        results[feat["name"]] = result
        print(f"  Optimal: slider={result.optimal_value:.3f}, "
              f"best_f1={result.best_f1:.4f} ({result.best_algorithm}), "
              f"exit={result.exit_reason}")

    return results


def optimization_result_to_df(result: OptimizationResult) -> pd.DataFrame:
    """Convert optimization steps to a DataFrame for plotting."""
    rows = []
    for step in result.steps:
        row = {
            "slider_value": step.slider_value,
            "n_samples": step.n_samples,
            "delta": step.delta,
            "best_algo": step.best_algo,
            "best_score": step.best_score,
        }
        for algo, score in step.scores.items():
            row[f"{algo}_f1"] = score
        rows.append(row)
    return pd.DataFrame(rows)


def run_gap_analysis(
    raw_df: pd.DataFrame,
    feature_name: str,
    feature_type: str,
    slider_values: list,
    selected_algos=None,
    max_samples: int = 10000,
    use_smote: bool = False,
    progress_callback=None,
) -> list:
    """
    Collect train-val F1 gaps across a set of slider values.
    Returns list of dicts with keys: slider, algo, train_f1, val_f1, gap.
    Used to detect gap widening (overfitting signal) as upsampling increases.
    """
    from muller_loop import run_muller_loop

    gap_data = []
    total = len(slider_values)
    for idx, sv in enumerate(slider_values):
        try:
            if abs(sv) < 0.01:
                X_mod, y_mod, _ = prepare_features(raw_df)
                X_mod, y_mod = _subsample(X_mod, y_mod, max_samples)
            elif feature_type == "target" and use_smote:
                X_mod, y_mod = prepare_features(raw_df)[:2]
                X_mod, y_mod = _subsample(X_mod, y_mod, max_samples)
                X_mod, y_mod = modify_distribution_smote(X_mod, y_mod, sv)
            else:
                df_mod = modify_distribution_raw(raw_df, feature_name, feature_type, sv)
                X_mod, y_mod, _ = prepare_features(df_mod)
                X_mod, y_mod = _subsample(X_mod, y_mod, max_samples)

            results = run_muller_loop(
                X_mod, y_mod, selected_algos=selected_algos,
                cv_folds=3, compute_learning_curves=False,
            )
            for algo_name, result in results.items():
                gap_data.append({
                    "slider": float(sv),
                    "algo": algo_name,
                    "train_f1": result.train_f1,
                    "val_f1": result.val_f1,
                    "gap": result.train_f1 - result.val_f1,
                })

            if progress_callback:
                progress_callback(idx, total)

        except Exception as e:
            print(f"  Gap analysis step {sv:.2f} failed: {e}")
            continue

    return gap_data


def get_fit_diagnosis(result: "ModelResult") -> dict:
    """
    Diagnose overfitting/underfitting for a model result.

    Returns dict with:
        status: "overfit", "underfit", "optimal"
        train_f1, val_f1, gap, cv_mean, cv_std
        explanation: human-readable diagnosis
    """
    train_f1 = result.train_f1
    val_f1 = result.val_f1
    gap = train_f1 - val_f1
    cv_mean = result.cv_mean
    cv_std = result.cv_std

    if train_f1 > 0.95 and gap > 0.05:
        status = "overfit"
        explanation = (
            f"High training F1 ({train_f1:.4f}) but lower validation F1 ({val_f1:.4f}). "
            f"Gap of {gap:.4f} suggests the model is memorizing training noise. "
            f"This may occur from over-upsampling (SMOTE) creating synthetic points "
            f"too similar to existing noise."
        )
    elif train_f1 < 0.7 and val_f1 < 0.7:
        status = "underfit"
        explanation = (
            f"Both training F1 ({train_f1:.4f}) and validation F1 ({val_f1:.4f}) are low. "
            f"The model lacks capacity to learn the signal. "
            f"This may occur from aggressive downsampling that stripped essential data."
        )
    elif gap < 0.03 and val_f1 > 0.7:
        status = "optimal"
        explanation = (
            f"Training F1 ({train_f1:.4f}) and validation F1 ({val_f1:.4f}) have converged "
            f"(gap={gap:.4f}). CV mean={cv_mean:.4f}±{cv_std:.4f}. "
            f"This represents a good bias-variance tradeoff."
        )
    elif gap > 0.03:
        status = "slight_overfit"
        explanation = (
            f"Training F1 ({train_f1:.4f}) is somewhat higher than validation F1 ({val_f1:.4f}). "
            f"Gap of {gap:.4f} indicates mild overfitting. "
            f"Consider reducing model complexity or increasing regularization."
        )
    else:
        status = "acceptable"
        explanation = (
            f"Training F1 ({train_f1:.4f}) and validation F1 ({val_f1:.4f}) are reasonable. "
            f"Gap={gap:.4f}, CV={cv_mean:.4f}±{cv_std:.4f}."
        )

    return {
        "status": status,
        "train_f1": train_f1,
        "val_f1": val_f1,
        "gap": gap,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "explanation": explanation,
    }
