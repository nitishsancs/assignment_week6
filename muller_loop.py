"""
Muller Loop for Week 6 Dashboard.
Trains multiple classification algorithms and collects metrics.
Algorithms: XGBoost, MLP, Random Forest, SVM.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


# Valid severity classes in order
CLASS_ORDER = ["Low", "Medium", "High", "Critical"]


@dataclass
class ModelResult:
    """Stores results for a single trained model."""
    algorithm: str
    f1_macro: float
    f1_weighted: float
    accuracy: float
    precision_macro: float
    recall_macro: float  # sensitivity
    confusion_mat: np.ndarray
    classification_rep: str
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: Optional[np.ndarray] = None
    roc_auc: Optional[float] = None
    # Per-class metrics
    per_class_precision: Dict[str, float] = field(default_factory=dict)
    per_class_recall: Dict[str, float] = field(default_factory=dict)
    per_class_specificity: Dict[str, float] = field(default_factory=dict)
    per_class_f1: Dict[str, float] = field(default_factory=dict)
    train_time: float = 0.0


def _compute_per_class_specificity(
    y_true: np.ndarray, y_pred: np.ndarray, classes: List[str]
) -> Dict[str, float]:
    """Compute specificity (true negative rate) for each class."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    specificities = {}
    for i, cls in enumerate(classes):
        # True negatives: sum of all elements except row i and column i
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        # False positives: sum of column i minus true positive
        fp = cm[:, i].sum() - cm[i, i]
        if (tn + fp) > 0:
            specificities[cls] = tn / (tn + fp)
        else:
            specificities[cls] = 0.0
    return specificities


def _get_algorithms() -> Dict[str, object]:
    """Initialize all 4 algorithms with reasonable defaults."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    except ImportError:
        print("WARNING: xgboost not installed, using GradientBoosting fallback")
        from sklearn.ensemble import GradientBoostingClassifier
        xgb = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
        )

    algorithms = {
        "XGBoost": xgb,
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        ),
        "SVM": SVC(
            kernel="rbf",
            probability=True,
            random_state=42,
            max_iter=5000,
        ),
    }
    return algorithms


def train_single_model(
    name: str,
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    classes: List[str],
) -> ModelResult:
    """Train a single model and compute all metrics."""
    import time

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)

    # Probabilities for ROC AUC
    y_proba = None
    roc_auc = None
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            # Compute ROC AUC (one-vs-rest)
            y_test_bin = label_binarize(y_test, classes=classes)
            if y_test_bin.shape[1] > 1:
                roc_auc = roc_auc_score(
                    y_test_bin, y_proba, multi_class="ovr", average="macro"
                )
    except Exception:
        pass

    # Core metrics
    f1_mac = f1_score(y_test, y_pred, average="macro", labels=classes, zero_division=0)
    f1_wt = f1_score(y_test, y_pred, average="weighted", labels=classes, zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    prec_mac = precision_score(y_test, y_pred, average="macro", labels=classes, zero_division=0)
    rec_mac = recall_score(y_test, y_pred, average="macro", labels=classes, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    # Classification report
    report = classification_report(y_test, y_pred, labels=classes, zero_division=0)

    # Per-class metrics
    prec_per = precision_score(y_test, y_pred, average=None, labels=classes, zero_division=0)
    rec_per = recall_score(y_test, y_pred, average=None, labels=classes, zero_division=0)
    f1_per = f1_score(y_test, y_pred, average=None, labels=classes, zero_division=0)
    spec_per = _compute_per_class_specificity(y_test, y_pred, classes)

    per_class_precision = {cls: float(prec_per[i]) for i, cls in enumerate(classes)}
    per_class_recall = {cls: float(rec_per[i]) for i, cls in enumerate(classes)}
    per_class_f1 = {cls: float(f1_per[i]) for i, cls in enumerate(classes)}

    return ModelResult(
        algorithm=name,
        f1_macro=f1_mac,
        f1_weighted=f1_wt,
        accuracy=acc,
        precision_macro=prec_mac,
        recall_macro=rec_mac,
        confusion_mat=cm,
        classification_rep=report,
        y_true=np.array(y_test),
        y_pred=np.array(y_pred),
        y_proba=y_proba,
        roc_auc=roc_auc,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        per_class_specificity=spec_per,
        per_class_f1=per_class_f1,
        train_time=train_time,
    )


def run_muller_loop(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    algorithms: Optional[Dict[str, object]] = None,
    selected_algos: Optional[List[str]] = None,
) -> Dict[str, ModelResult]:
    """
    Run the Muller loop: train all algorithms and return results.

    Args:
        X: Feature matrix.
        y: Target labels.
        test_size: Fraction for test split.
        random_state: Random seed.
        algorithms: Optional dict of name->model. Uses defaults if None.
        selected_algos: Optional list of algorithm names to run. Runs all if None.

    Returns:
        Dict mapping algorithm name to ModelResult.
    """
    from sklearn.preprocessing import LabelEncoder

    if algorithms is None:
        algorithms = _get_algorithms()

    if selected_algos:
        algorithms = {k: v for k, v in algorithms.items() if k in selected_algos}

    # Determine classes present in data
    classes = [c for c in CLASS_ORDER if c in y.unique()]

    # Encode string labels to integers (required for XGBoost)
    le = LabelEncoder()
    le.fit(classes)
    y_encoded = pd.Series(le.transform(y), index=y.index)

    # Stratified split
    X_train, X_test, y_train_enc, y_test_enc = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    # Decode back for metrics computation
    y_train_str = pd.Series(le.inverse_transform(y_train_enc), index=y_train_enc.index)
    y_test_str = pd.Series(le.inverse_transform(y_test_enc), index=y_test_enc.index)

    results = {}
    for name, model in algorithms.items():
        print(f"  Training {name}...", end=" ", flush=True)
        try:
            # Train with encoded labels
            import time
            start = time.time()
            model.fit(X_train, y_train_enc)
            train_time = time.time() - start

            y_pred_enc = model.predict(X_test)
            y_pred_str = le.inverse_transform(y_pred_enc)

            # Probabilities for ROC AUC
            y_proba = None
            roc_auc = None
            try:
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test)
                    y_test_bin = label_binarize(y_test_str, classes=classes)
                    if y_test_bin.shape[1] > 1:
                        roc_auc = roc_auc_score(
                            y_test_bin, y_proba, multi_class="ovr", average="macro"
                        )
            except Exception:
                pass

            # Metrics with string labels
            f1_mac = f1_score(y_test_str, y_pred_str, average="macro", labels=classes, zero_division=0)
            f1_wt = f1_score(y_test_str, y_pred_str, average="weighted", labels=classes, zero_division=0)
            acc = accuracy_score(y_test_str, y_pred_str)
            prec_mac = precision_score(y_test_str, y_pred_str, average="macro", labels=classes, zero_division=0)
            rec_mac = recall_score(y_test_str, y_pred_str, average="macro", labels=classes, zero_division=0)
            cm = confusion_matrix(y_test_str, y_pred_str, labels=classes)
            report = classification_report(y_test_str, y_pred_str, labels=classes, zero_division=0)

            prec_per = precision_score(y_test_str, y_pred_str, average=None, labels=classes, zero_division=0)
            rec_per = recall_score(y_test_str, y_pred_str, average=None, labels=classes, zero_division=0)
            f1_per = f1_score(y_test_str, y_pred_str, average=None, labels=classes, zero_division=0)
            spec_per = _compute_per_class_specificity(y_test_str, y_pred_str, classes)

            per_class_precision = {cls: float(prec_per[i]) for i, cls in enumerate(classes)}
            per_class_recall = {cls: float(rec_per[i]) for i, cls in enumerate(classes)}
            per_class_f1 = {cls: float(f1_per[i]) for i, cls in enumerate(classes)}

            result = ModelResult(
                algorithm=name,
                f1_macro=f1_mac,
                f1_weighted=f1_wt,
                accuracy=acc,
                precision_macro=prec_mac,
                recall_macro=rec_mac,
                confusion_mat=cm,
                classification_rep=report,
                y_true=np.array(y_test_str),
                y_pred=np.array(y_pred_str),
                y_proba=y_proba,
                roc_auc=roc_auc,
                per_class_precision=per_class_precision,
                per_class_recall=per_class_recall,
                per_class_specificity=spec_per,
                per_class_f1=per_class_f1,
                train_time=train_time,
            )
            results[name] = result
            print(f"F1={result.f1_macro:.3f}, Acc={result.accuracy:.3f} ({result.train_time:.1f}s)")
        except Exception as e:
            print(f"FAILED: {e}")

    return results


def results_to_summary_df(results: Dict[str, ModelResult]) -> pd.DataFrame:
    """Convert results dict to a summary DataFrame."""
    rows = []
    for name, r in results.items():
        rows.append({
            "Algorithm": name,
            "F1 (Macro)": round(r.f1_macro, 4),
            "F1 (Weighted)": round(r.f1_weighted, 4),
            "Accuracy": round(r.accuracy, 4),
            "Precision": round(r.precision_macro, 4),
            "Recall (Sensitivity)": round(r.recall_macro, 4),
            "ROC AUC": round(r.roc_auc, 4) if r.roc_auc else "N/A",
            "Train Time (s)": round(r.train_time, 2),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from data_loader import load_data
    from feature_engineering import prepare_features

    print("=== Muller Loop Baseline ===\n")
    df = load_data()
    X, y, _ = prepare_features(df)

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution:\n{y.value_counts()}\n")

    results = run_muller_loop(X, y)

    print("\n=== Summary ===")
    summary = results_to_summary_df(results)
    print(summary.to_string(index=False))

    print("\n=== Detailed Reports ===")
    for name, r in results.items():
        print(f"\n--- {name} ---")
        print(r.classification_rep)
