"""
Feature importance analysis for Week 6 Dashboard.
Computes Gini importance, permutation importance, and SHAP values.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Optional


def compute_gini_importance(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute Gini (MDI) feature importance using RandomForest.

    Returns:
        DataFrame with columns ['feature', 'importance'] sorted descending.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=1,
        max_depth=10,
    )
    rf.fit(X, y)

    importances = rf.feature_importances_
    result = pd.DataFrame({
        "feature": X.columns,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return result


def compute_permutation_importance(
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
    test_size: float = 0.2,
) -> pd.DataFrame:
    """
    Compute permutation importance (model-agnostic).

    Returns:
        DataFrame with columns ['feature', 'importance', 'std'] sorted descending.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=1,
        max_depth=10,
    )
    rf.fit(X_train, y_train)

    perm_result = permutation_importance(
        rf, X_test, y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
        scoring="f1_macro",
    )

    result = pd.DataFrame({
        "feature": X.columns,
        "importance": perm_result.importances_mean,
        "std": perm_result.importances_std,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return result


def compute_shap_importance(
    X: pd.DataFrame,
    y: pd.Series,
    max_samples: int = 200,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Optional[object]]:
    """
    Compute SHAP feature importance using TreeExplainer on RandomForest.

    Returns:
        (importance_df, shap_values) - DataFrame + raw SHAP values for plotting.
    """
    try:
        import shap
    except ImportError:
        print("WARNING: shap not installed. Skipping SHAP analysis.")
        return pd.DataFrame(columns=["feature", "importance"]), None

    rf = RandomForestClassifier(
        n_estimators=50,
        random_state=random_state,
        n_jobs=1,
        max_depth=8,
    )
    rf.fit(X, y)

    # Subsample for speed and memory
    if len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=random_state)
    else:
        X_sample = X

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_sample, check_additivity=False)

    # Mean absolute SHAP value across all classes
    if isinstance(shap_values, list):
        # Multi-class: list of 2D arrays (one per class)
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    elif shap_values.ndim == 3:
        # Multi-class: 3D array (samples, features, classes)
        mean_abs = np.abs(shap_values).mean(axis=(0, 2))
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)

    result = pd.DataFrame({
        "feature": X.columns,
        "importance": mean_abs,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return result, shap_values


def compute_all_importances(
    X: pd.DataFrame,
    y: pd.Series,
) -> Dict[str, pd.DataFrame]:
    """
    Compute all three importance methods.

    Returns:
        Dict with keys 'gini', 'permutation', 'shap' mapping to DataFrames.
    """
    print("Computing Gini importance...")
    gini = compute_gini_importance(X, y)

    print("Computing permutation importance...")
    perm = compute_permutation_importance(X, y)

    print("Computing SHAP importance...")
    shap_df, _ = compute_shap_importance(X, y)

    return {
        "gini": gini,
        "permutation": perm,
        "shap": shap_df,
    }


def get_top_features(importances: Dict[str, pd.DataFrame], top_n: int = 10) -> pd.DataFrame:
    """
    Aggregate top features across all methods.

    Returns:
        DataFrame with feature, gini_rank, perm_rank, shap_rank, avg_rank.
    """
    rankings = {}

    for method, df in importances.items():
        if df.empty:
            continue
        for rank, row in enumerate(df.itertuples(), start=1):
            feat = row.feature
            if feat not in rankings:
                rankings[feat] = {}
            rankings[feat][f"{method}_rank"] = rank

    if not rankings:
        return pd.DataFrame()

    methods = [m for m in importances if not importances[m].empty]
    n_features = max(len(df) for df in importances.values() if not df.empty)

    rows = []
    for feat, ranks in rankings.items():
        row = {"feature": feat}
        total = 0
        for method in methods:
            r = ranks.get(f"{method}_rank", n_features)
            row[f"{method}_rank"] = r
            total += r
        row["avg_rank"] = total / len(methods)
        rows.append(row)

    result = pd.DataFrame(rows).sort_values("avg_rank").reset_index(drop=True)
    return result.head(top_n)


if __name__ == "__main__":
    from data_loader import load_data
    from feature_engineering import prepare_features

    df = load_data()
    X, y, feat_names = prepare_features(df, return_feature_names=True)

    importances = compute_all_importances(X, y)

    print("\n=== Top 10 Features (Gini) ===")
    print(importances["gini"].head(10).to_string(index=False))

    print("\n=== Top 10 Features (Permutation) ===")
    print(importances["permutation"].head(10).to_string(index=False))

    print("\n=== Top 10 Features (SHAP) ===")
    print(importances["shap"].head(10).to_string(index=False))

    print("\n=== Aggregated Top Features ===")
    top = get_top_features(importances)
    print(top.to_string(index=False))
