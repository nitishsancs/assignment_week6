"""
Feature engineering for Week 6 Dashboard.
Transforms raw NVD CVE DataFrame into ML-ready feature matrix + labels.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional


# Top CWEs to keep as individual categories; rest become "OTHER"
TOP_CWE_COUNT = 20

# Categorical columns to one-hot encode
CATEGORICAL_COLS = [
    "attack_vector",
    "attack_complexity",
    "privileges_required",
    "user_interaction",
    "scope",
    "confidentiality",
    "integrity",
    "availability",
    "primary_cwe_group",
]

# Numeric columns to scale
NUMERIC_COLS = [
    "cvss2_score",
    "exploitability_score",
    "impact_score",
    "num_vendors",
    "num_weaknesses",
    "num_references",
    "desc_length",
    "year_published",
]

# Binary columns (already 0/1)
BINARY_COLS = [
    "has_exploit",
]

# Target column
TARGET_COL = "severity"

# Valid severity classes
VALID_SEVERITIES = ["Critical", "High", "Medium", "Low"]


def _group_cwe(df: pd.DataFrame) -> pd.Series:
    """Group rare CWEs into 'OTHER'."""
    cwe_counts = df["primary_cwe"].value_counts()
    top_cwes = set(cwe_counts.head(TOP_CWE_COUNT).index)
    return df["primary_cwe"].apply(lambda x: x if x in top_cwes else "OTHER")


def prepare_features(
    df: pd.DataFrame,
    scale_numeric: bool = True,
    return_feature_names: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, Optional[List[str]]]:
    """
    Transform raw CVE DataFrame into ML-ready features and labels.

    Args:
        df: Raw DataFrame from data_loader.
        scale_numeric: Whether to StandardScale numeric features.
        return_feature_names: If True, also return list of feature names.

    Returns:
        X: Feature DataFrame (numeric, ready for sklearn)
        y: Target Series (severity labels)
        feature_names: (optional) list of column names in X
    """
    work = df.copy()

    # Filter to valid severities only
    work = work[work[TARGET_COL].isin(VALID_SEVERITIES)].reset_index(drop=True)

    if len(work) == 0:
        raise ValueError("No valid severity labels found in data")

    # Group rare CWEs
    work["primary_cwe_group"] = _group_cwe(work)

    # Fill missing categoricals
    for col in CATEGORICAL_COLS:
        work[col] = work[col].fillna("UNKNOWN").astype(str)

    # Fill missing numerics
    for col in NUMERIC_COLS:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)

    # Fill missing binaries
    for col in BINARY_COLS:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0).astype(int)

    # One-hot encode categoricals
    encoded = pd.get_dummies(work[CATEGORICAL_COLS], prefix_sep="_", dtype=int)

    # Assemble feature matrix
    X = pd.concat([
        work[NUMERIC_COLS].reset_index(drop=True),
        work[BINARY_COLS].reset_index(drop=True),
        encoded.reset_index(drop=True),
    ], axis=1)

    # Scale numeric features
    if scale_numeric:
        scaler = StandardScaler()
        X[NUMERIC_COLS] = scaler.fit_transform(X[NUMERIC_COLS])

    # Target
    y = work[TARGET_COL].reset_index(drop=True)

    feature_names = list(X.columns) if return_feature_names else None

    if return_feature_names:
        return X, y, feature_names
    return X, y, None


def get_manipulable_features() -> List[dict]:
    """
    Return features suitable for distribution manipulation via slider.
    These are features in the raw DataFrame (before encoding).
    """
    return [
        {
            "name": "severity",
            "display": "Severity Class Balance (Target)",
            "type": "target",
            "description": "Directly upsample/downsample severity classes (Critical/High/Medium/Low)",
        },
        {
            "name": "has_exploit",
            "display": "Has Public Exploit",
            "type": "binary",
            "description": "Whether the CVE has a publicly available exploit (from NVD references)",
        },
        {
            "name": "attack_vector",
            "display": "Attack Vector (Network vs Other)",
            "type": "categorical",
            "description": "CVSS Attack Vector: NETWORK, ADJACENT_NETWORK, LOCAL, PHYSICAL",
        },
        {
            "name": "primary_cwe",
            "display": "Primary CWE (Weakness Type)",
            "type": "categorical",
            "description": "The primary CWE weakness type of the vulnerability",
        },
        {
            "name": "num_vendors",
            "display": "Number of Affected Vendors",
            "type": "continuous",
            "description": "How many vendors are affected by this CVE (from CPE configurations)",
        },
        {
            "name": "exploitability_score",
            "display": "CVSS Exploitability Sub-Score",
            "type": "continuous",
            "description": "CVSS v3.1 exploitability sub-score (0-3.9)",
        },
    ]


if __name__ == "__main__":
    from data_loader import load_data

    df = load_data()
    X, y, feat_names = prepare_features(df, return_feature_names=True)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"\nFeature names ({len(feat_names)}):")
    for name in feat_names:
        print(f"  {name}")
