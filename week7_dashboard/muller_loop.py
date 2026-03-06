"""
Muller Loop for Week 7 Dashboard.
Enhanced with cross-validation, train/val scoring for overfitting detection,
and learning curve support.
Algorithms: XGBoost, MLP, Random Forest, SVM.
"""

import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from copy import deepcopy

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    learning_curve,
)
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize, LabelEncoder


CLASS_ORDER = ["Low", "Medium", "High", "Critical"]


@dataclass
class ModelResult:
    """Stores results for a single trained model."""
    algorithm: str
    f1_macro: float
    f1_weighted: float
    accuracy: float
    precision_macro: float
    recall_macro: float
    confusion_mat: np.ndarray
    classification_rep: str
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: Optional[np.ndarray] = None
    roc_auc: Optional[float] = None
    per_class_precision: Dict[str, float] = field(default_factory=dict)
    per_class_recall: Dict[str, float] = field(default_factory=dict)
    per_class_specificity: Dict[str, float] = field(default_factory=dict)
    per_class_f1: Dict[str, float] = field(default_factory=dict)
    train_time: float = 0.0
    # Week 7 additions
    train_f1: float = 0.0
    val_f1: float = 0.0
    cv_scores: Optional[np.ndarray] = None
    cv_mean: float = 0.0
    cv_std: float = 0.0
    learning_curve_train_sizes: Optional[np.ndarray] = None
    learning_curve_train_scores: Optional[np.ndarray] = None
    learning_curve_val_scores: Optional[np.ndarray] = None


def _compute_per_class_specificity(
    y_true: np.ndarray, y_pred: np.ndarray, classes: List[str]
) -> Dict[str, float]:
    """Compute specificity (true negative rate) for each class."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    specificities = {}
    for i, cls in enumerate(classes):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
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


def run_muller_loop(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    algorithms: Optional[Dict[str, object]] = None,
    selected_algos: Optional[List[str]] = None,
    cv_folds: int = 5,
    compute_learning_curves: bool = False,
) -> Dict[str, ModelResult]:
    """
    Run the Muller loop with cross-validation and train/val scoring.

    Args:
        X: Feature matrix.
        y: Target labels.
        test_size: Fraction for test split.
        random_state: Random seed.
        algorithms: Optional dict of name->model.
        selected_algos: Optional list of algorithm names to run.
        cv_folds: Number of cross-validation folds.
        compute_learning_curves: Whether to compute learning curves (slower).

    Returns:
        Dict mapping algorithm name to ModelResult.
    """
    if algorithms is None:
        algorithms = _get_algorithms()

    if selected_algos:
        algorithms = {k: v for k, v in algorithms.items() if k in selected_algos}

    classes = [c for c in CLASS_ORDER if c in y.unique()]

    le = LabelEncoder()
    le.fit(classes)
    y_encoded = pd.Series(le.transform(y), index=y.index)

    X_train, X_test, y_train_enc, y_test_enc = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    y_train_str = pd.Series(le.inverse_transform(y_train_enc), index=y_train_enc.index)
    y_test_str = pd.Series(le.inverse_transform(y_test_enc), index=y_test_enc.index)

    results = {}
    for name, model in algorithms.items():
        print(f"  Training {name}...", end=" ", flush=True)
        try:
            model_copy = deepcopy(model)

            start = time.time()
            model_copy.fit(X_train, y_train_enc)
            train_time = time.time() - start

            # Predictions
            y_pred_enc = model_copy.predict(X_test)
            y_pred_str = le.inverse_transform(y_pred_enc)
            y_train_pred_enc = model_copy.predict(X_train)
            y_train_pred_str = le.inverse_transform(y_train_pred_enc)

            # Train vs Val F1 (overfitting detection)
            train_f1 = f1_score(y_train_str, y_train_pred_str, average="macro",
                                labels=classes, zero_division=0)
            val_f1 = f1_score(y_test_str, y_pred_str, average="macro",
                              labels=classes, zero_division=0)

            # Cross-validation
            cv_scores = None
            cv_mean = 0.0
            cv_std = 0.0
            try:
                skf = StratifiedKFold(n_splits=min(cv_folds, min(pd.Series(y_train_enc).value_counts())),
                                      shuffle=True, random_state=random_state)
                cv_model = deepcopy(model)
                cv_scores = cross_val_score(
                    cv_model, X_train, y_train_enc,
                    cv=skf, scoring="f1_macro", n_jobs=-1
                )
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            except Exception as cv_err:
                print(f"(CV failed: {cv_err})", end=" ")

            # Learning curves (optional, expensive)
            lc_train_sizes = None
            lc_train_scores = None
            lc_val_scores = None
            if compute_learning_curves:
                try:
                    lc_model = deepcopy(model)
                    lc_train_sizes, lc_train_scores, lc_val_scores = learning_curve(
                        lc_model, X_train, y_train_enc,
                        cv=min(3, min(pd.Series(y_train_enc).value_counts())),
                        scoring="f1_macro",
                        train_sizes=np.linspace(0.1, 1.0, 8),
                        n_jobs=-1,
                        random_state=random_state,
                    )
                except Exception as lc_err:
                    print(f"(LC failed: {lc_err})", end=" ")

            # Probabilities for ROC AUC
            y_proba = None
            roc_auc = None
            try:
                if hasattr(model_copy, "predict_proba"):
                    y_proba = model_copy.predict_proba(X_test)
                    y_test_bin = label_binarize(y_test_str, classes=classes)
                    if y_test_bin.shape[1] > 1:
                        roc_auc = roc_auc_score(
                            y_test_bin, y_proba, multi_class="ovr", average="macro"
                        )
            except Exception:
                pass

            # Core metrics
            f1_mac = val_f1
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
                train_f1=train_f1,
                val_f1=val_f1,
                cv_scores=cv_scores,
                cv_mean=cv_mean,
                cv_std=cv_std,
                learning_curve_train_sizes=lc_train_sizes,
                learning_curve_train_scores=lc_train_scores,
                learning_curve_val_scores=lc_val_scores,
            )
            results[name] = result
            print(f"F1={val_f1:.3f}, TrainF1={train_f1:.3f}, CV={cv_mean:.3f}±{cv_std:.3f} ({train_time:.1f}s)")
        except Exception as e:
            print(f"FAILED: {e}")

    return results


def run_muller_loop_quick(
    X: pd.DataFrame,
    y: pd.Series,
    selected_algos: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Lightweight Muller loop that returns only F1 macro scores.
    Used by the AutoML optimizer for fast iteration.
    """
    algorithms = _get_algorithms()
    if selected_algos:
        algorithms = {k: v for k, v in algorithms.items() if k in selected_algos}

    classes = [c for c in CLASS_ORDER if c in y.unique()]
    le = LabelEncoder()
    le.fit(classes)
    y_encoded = pd.Series(le.transform(y), index=y.index)

    X_train, X_test, y_train_enc, y_test_enc = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    y_test_str = pd.Series(le.inverse_transform(y_test_enc), index=y_test_enc.index)

    scores = {}
    for name, model in algorithms.items():
        try:
            model_copy = deepcopy(model)
            model_copy.fit(X_train, y_train_enc)
            y_pred_enc = model_copy.predict(X_test)
            y_pred_str = le.inverse_transform(y_pred_enc)
            f1 = f1_score(y_test_str, y_pred_str, average="macro", labels=classes, zero_division=0)
            scores[name] = f1
        except Exception:
            scores[name] = 0.0
    return scores


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
            "Recall": round(r.recall_macro, 4),
            "ROC AUC": round(r.roc_auc, 4) if r.roc_auc else "N/A",
            "Train F1": round(r.train_f1, 4),
            "Val F1": round(r.val_f1, 4),
            "CV Mean": round(r.cv_mean, 4),
            "CV Std": round(r.cv_std, 4),
            "Train Time (s)": round(r.train_time, 2),
        })
    return pd.DataFrame(rows)
