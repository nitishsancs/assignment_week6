"""
Distribution manipulation for Week 6 Dashboard.
Handles upsampling (SMOTE) and downsampling (RandomUnderSampler)
based on slider values for selected features.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from collections import Counter


def _get_binary_split(
    df: pd.DataFrame, feature: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into two groups by binary feature."""
    group_1 = df[df[feature] == 1]
    group_0 = df[df[feature] == 0]
    return group_0, group_1


def _get_continuous_split(
    df: pd.DataFrame, feature: str, threshold: Optional[float] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into low/high groups by continuous feature median."""
    if threshold is None:
        threshold = df[feature].median()
    group_low = df[df[feature] <= threshold]
    group_high = df[df[feature] > threshold]
    return group_low, group_high


def _get_categorical_split(
    df: pd.DataFrame, feature: str
) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    """Split by most common vs rest for a categorical feature."""
    counts = df[feature].value_counts()
    majority_val = counts.index[0]
    majority = df[df[feature] == majority_val]
    minority = df[df[feature] != majority_val]
    return majority_val, majority, minority


def modify_distribution_raw(
    df: pd.DataFrame,
    feature_name: str,
    feature_type: str,
    slider_value: float,
) -> pd.DataFrame:
    """
    Modify the distribution of data in the raw DataFrame BEFORE feature encoding.

    Args:
        df: Raw CVE DataFrame (from data_loader).
        feature_name: Column name to manipulate.
        feature_type: One of 'binary', 'continuous', 'categorical', 'target'.
        slider_value: Float from -1.0 (full downsample) to +1.0 (full upsample).
                      0.0 = no change.

    Returns:
        Modified DataFrame with altered distribution.
    """
    if abs(slider_value) < 0.01:
        return df.copy()

    work = df.copy()

    if feature_type == "target":
        return _modify_target_distribution(work, feature_name, slider_value)
    elif feature_type == "binary":
        return _modify_binary_distribution(work, feature_name, slider_value)
    elif feature_type == "continuous":
        return _modify_continuous_distribution(work, feature_name, slider_value)
    elif feature_type == "categorical":
        return _modify_categorical_distribution(work, feature_name, slider_value)
    else:
        return work


def _modify_target_distribution(
    df: pd.DataFrame, feature: str, slider_value: float
) -> pd.DataFrame:
    """
    Directly modify the target class balance.
    Positive slider = upsample minority classes.
    Negative slider = downsample majority classes.
    """
    counts = df[feature].value_counts()
    max_count = counts.max()
    min_count = counts.min()

    if slider_value > 0:
        # Upsample minority classes toward majority
        target_count = int(min_count + (max_count - min_count) * slider_value)
        frames = []
        for cls in counts.index:
            cls_df = df[df[feature] == cls]
            if len(cls_df) < target_count:
                # Oversample with replacement
                upsampled = cls_df.sample(n=target_count, replace=True, random_state=42)
                frames.append(upsampled)
            else:
                frames.append(cls_df)
        return pd.concat(frames, ignore_index=True)
    else:
        # Downsample majority classes toward minority
        ratio = abs(slider_value)
        target_count = int(max_count - (max_count - min_count) * ratio)
        target_count = max(target_count, min_count)
        frames = []
        for cls in counts.index:
            cls_df = df[df[feature] == cls]
            if len(cls_df) > target_count:
                downsampled = cls_df.sample(n=target_count, random_state=42)
                frames.append(downsampled)
            else:
                frames.append(cls_df)
        return pd.concat(frames, ignore_index=True)


def _modify_binary_distribution(
    df: pd.DataFrame, feature: str, slider_value: float
) -> pd.DataFrame:
    """
    Modify binary feature distribution.
    Positive = upsample the 1-class (minority, e.g., has_kev=1).
    Negative = downsample the 1-class.
    """
    group_0, group_1 = _get_binary_split(df, feature)

    if len(group_1) == 0 or len(group_0) == 0:
        return df.copy()

    if slider_value > 0:
        # Upsample group_1 (minority)
        current = len(group_1)
        target = int(current + (len(group_0) - current) * slider_value)
        target = max(target, current)
        upsampled = group_1.sample(n=target, replace=True, random_state=42)
        return pd.concat([group_0, upsampled], ignore_index=True)
    else:
        # Downsample group_1
        ratio = abs(slider_value)
        target = int(len(group_1) * (1 - ratio))
        target = max(target, 1)
        downsampled = group_1.sample(n=target, random_state=42)
        return pd.concat([group_0, downsampled], ignore_index=True)


def _modify_continuous_distribution(
    df: pd.DataFrame, feature: str, slider_value: float
) -> pd.DataFrame:
    """
    Modify continuous feature distribution by up/downsampling high-value records.
    Positive = upsample records with high feature values.
    Negative = downsample records with high feature values.
    """
    median = df[feature].median()
    group_low, group_high = _get_continuous_split(df, feature, median)

    if len(group_high) == 0 or len(group_low) == 0:
        return df.copy()

    if slider_value > 0:
        # Upsample the high group
        current = len(group_high)
        target = int(current * (1 + slider_value * 2))
        target = max(target, current)
        upsampled = group_high.sample(n=target, replace=True, random_state=42)
        return pd.concat([group_low, upsampled], ignore_index=True)
    else:
        # Downsample the high group
        ratio = abs(slider_value)
        target = int(len(group_high) * (1 - ratio * 0.8))
        target = max(target, 1)
        downsampled = group_high.sample(n=target, random_state=42)
        return pd.concat([group_low, downsampled], ignore_index=True)


def _modify_categorical_distribution(
    df: pd.DataFrame, feature: str, slider_value: float
) -> pd.DataFrame:
    """
    Modify categorical feature distribution.
    Positive = upsample minority categories.
    Negative = downsample majority category.
    """
    majority_val, majority, minority = _get_categorical_split(df, feature)

    if len(majority) == 0 or len(minority) == 0:
        return df.copy()

    if slider_value > 0:
        # Upsample minority categories
        current = len(minority)
        target = int(current + (len(majority) - current) * slider_value)
        target = max(target, current)
        upsampled = minority.sample(n=target, replace=True, random_state=42)
        return pd.concat([majority, upsampled], ignore_index=True)
    else:
        # Downsample majority category
        ratio = abs(slider_value)
        target = int(len(majority) * (1 - ratio * 0.8))
        target = max(target, len(minority))
        downsampled = majority.sample(n=target, random_state=42)
        return pd.concat([downsampled, minority], ignore_index=True)


def modify_distribution_smote(
    X: pd.DataFrame,
    y: pd.Series,
    slider_value: float,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE-based oversampling / random undersampling on the TARGET variable.
    This modifies class balance directly using imblearn.

    Args:
        X: Feature matrix.
        y: Target labels.
        slider_value: -1.0 to +1.0. Positive=SMOTE oversample minority,
                      Negative=undersample majority.

    Returns:
        (X_resampled, y_resampled)
    """
    if abs(slider_value) < 0.01:
        return X.copy(), y.copy()

    counts = Counter(y)
    max_count = max(counts.values())
    min_count = min(counts.values())

    if slider_value > 0:
        # SMOTE oversample minority classes
        try:
            from imblearn.over_sampling import SMOTE

            # Target ratio: move minority toward majority
            target_count = int(min_count + (max_count - min_count) * slider_value)
            sampling_strategy = {}
            for cls, cnt in counts.items():
                if cnt < target_count:
                    sampling_strategy[cls] = target_count

            if sampling_strategy:
                k_neighbors = min(5, min_count - 1) if min_count > 1 else 1
                k_neighbors = max(k_neighbors, 1)
                smote = SMOTE(
                    sampling_strategy=sampling_strategy,
                    k_neighbors=k_neighbors,
                    random_state=42,
                )
                X_res, y_res = smote.fit_resample(X, y)
                return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res)
        except ImportError:
            print("WARNING: imbalanced-learn not installed, using random oversampling")
        except Exception as e:
            print(f"SMOTE failed ({e}), using random oversampling")

        # Fallback: random oversampling
        return _random_oversample(X, y, slider_value)

    else:
        # Random undersample majority classes
        try:
            from imblearn.under_sampling import RandomUnderSampler

            ratio = abs(slider_value)
            target_count = int(max_count - (max_count - min_count) * ratio)
            target_count = max(target_count, min_count)
            sampling_strategy = {}
            for cls, cnt in counts.items():
                if cnt > target_count:
                    sampling_strategy[cls] = target_count

            if sampling_strategy:
                rus = RandomUnderSampler(
                    sampling_strategy=sampling_strategy,
                    random_state=42,
                )
                X_res, y_res = rus.fit_resample(X, y)
                return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res)
        except ImportError:
            print("WARNING: imbalanced-learn not installed, using random undersampling")
        except Exception as e:
            print(f"RandomUnderSampler failed ({e}), using manual undersampling")

        # Fallback
        return _random_undersample(X, y, slider_value)

    return X.copy(), y.copy()


def _random_oversample(
    X: pd.DataFrame, y: pd.Series, slider_value: float
) -> Tuple[pd.DataFrame, pd.Series]:
    """Simple random oversampling fallback."""
    counts = Counter(y)
    max_count = max(counts.values())
    min_count = min(counts.values())
    target = int(min_count + (max_count - min_count) * slider_value)

    X_list, y_list = [X], [y]
    for cls in counts:
        if counts[cls] < target:
            mask = y == cls
            n_needed = target - counts[cls]
            idx = X[mask].sample(n=n_needed, replace=True, random_state=42).index
            X_list.append(X.loc[idx])
            y_list.append(y.loc[idx])

    return pd.concat(X_list, ignore_index=True), pd.concat(y_list, ignore_index=True)


def _random_undersample(
    X: pd.DataFrame, y: pd.Series, slider_value: float
) -> Tuple[pd.DataFrame, pd.Series]:
    """Simple random undersampling fallback."""
    counts = Counter(y)
    max_count = max(counts.values())
    min_count = min(counts.values())
    ratio = abs(slider_value)
    target = int(max_count - (max_count - min_count) * ratio)
    target = max(target, min_count)

    X_list, y_list = [], []
    for cls in counts:
        mask = y == cls
        cls_X = X[mask]
        cls_y = y[mask]
        if len(cls_X) > target:
            sampled = cls_X.sample(n=target, random_state=42)
            X_list.append(sampled)
            y_list.append(cls_y.loc[sampled.index])
        else:
            X_list.append(cls_X)
            y_list.append(cls_y)

    return pd.concat(X_list, ignore_index=True), pd.concat(y_list, ignore_index=True)


def get_distribution_stats(
    y_original: pd.Series, y_modified: pd.Series
) -> pd.DataFrame:
    """Compare class distributions before and after modification."""
    orig_counts = y_original.value_counts().sort_index()
    mod_counts = y_modified.value_counts().sort_index()

    all_classes = sorted(set(orig_counts.index) | set(mod_counts.index))

    rows = []
    for cls in all_classes:
        orig = orig_counts.get(cls, 0)
        mod = mod_counts.get(cls, 0)
        diff = mod - orig
        pct_change = ((mod - orig) / orig * 100) if orig > 0 else 0
        rows.append({
            "Class": cls,
            "Original": orig,
            "Modified": mod,
            "Difference": diff,
            "% Change": round(pct_change, 1),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from data_loader import load_data
    from feature_engineering import prepare_features

    df = load_data()

    print("=== Raw Distribution Modification ===")
    print(f"Original severity dist:\n{df['severity'].value_counts()}\n")

    # Test upsample
    df_up = modify_distribution_raw(df, "severity", "target", 0.5)
    print(f"After upsample (0.5):\n{df_up['severity'].value_counts()}\n")

    # Test downsample
    df_down = modify_distribution_raw(df, "severity", "target", -0.5)
    print(f"After downsample (-0.5):\n{df_down['severity'].value_counts()}\n")

    print("=== SMOTE-based Modification ===")
    X, y, _ = prepare_features(df)
    print(f"Original: {Counter(y)}")

    X_up, y_up = modify_distribution_smote(X, y, 0.5)
    print(f"SMOTE upsample (0.5): {Counter(y_up)}")

    X_down, y_down = modify_distribution_smote(X, y, -0.5)
    print(f"Undersample (-0.5): {Counter(y_down)}")
