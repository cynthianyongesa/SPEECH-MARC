"""
Evaluation utilities for SPEECH-MARC

This module provides:
- Cross-validated evaluation for binary classification (AUC, accuracy, F1, sensitivity, specificity, precision)
- A compact set of model presets aligned with the study (L1-Logistic, linear SVM)
- ROC plotting helpers
- A comparison helper to evaluate multiple feature sets (e.g., linguistic, acoustic, combined)

Intended usage:
1) Build X and y (e.g., residualized + selected features).
2) Call `evaluate_cv(X, y)` for a small model panel, or
   `compare_feature_sets({"linguistic": X_ling, "acoustic": X_acou, "combined": X_all}, y, model_name="Logistic_L1")`.
"""

from __future__ import annotations
from typing import Dict, Optional, Mapping, Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    RocCurveDisplay,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# ---------------------------------------------------------------------------
# Model presets
# ---------------------------------------------------------------------------

def model_presets() -> Dict[str, object]:
    """
    Return a dictionary of classifier presets used in the study.
    - Logistic_L1: L1-regularized logistic regression (liblinear)
    - SVM_Linear:  Linear SVM with probability estimates
    """
    return {
        "Logistic_L1": LogisticRegression(
            penalty="l1", solver="liblinear", max_iter=2000, C=1.0, random_state=42
        ),
        "SVM_Linear": SVC(kernel="linear", probability=True, random_state=42),
    }


def make_tabular_pipeline(estimator) -> Pipeline:
    """
    Standard tabular pipeline: impute missing values → standardize → classifier.
    """
    return Pipeline(
        steps=[
            ("impute", SimpleImputer()),
            ("scale", StandardScaler()),
            ("clf", estimator),
        ]
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


def _cv_probabilities(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> np.ndarray:
    """
    Obtain cross-validated P(y=1). Falls back to decision_function → sigmoid if needed.
    """
    # Try predict_proba
    try:
        proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")
        if proba.ndim == 2:  # shape (n_samples, 2)
            return proba[:, 1]
    except Exception:
        pass

    # Fall back to decision_function
    scores = cross_val_predict(pipe, X, y, cv=cv, method="decision_function")
    return _sigmoid(scores)


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute standard binary metrics. Specificity derived from confusion matrix.
    """
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)  # sensitivity
    f1 = f1_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    auc = roc_auc_score(y_true, y_prob)

    return {
        "auc": float(auc),
        "accuracy": float(acc),
        "f1": float(f1),
        "sensitivity": float(rec),
        "specificity": float(spec),
        "precision": float(pre),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_cv(
    X: pd.DataFrame,
    y: pd.Series,
    models: Optional[Mapping[str, object]] = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Perform stratified K-fold cross-validation for one feature matrix across a small model panel.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (rows: samples, columns: features).
    y : pd.Series
        Binary labels (0/1).
    models : Mapping[str, object], optional
        Dict of {model_name: estimator}. Defaults to `model_presets()`.
    n_splits : int
        Number of CV folds (default: 5).
    random_state : int
        CV shuffling seed.

    Returns
    -------
    pd.DataFrame
        Metrics per model sorted by AUC (descending), columns:
        ['model', 'auc', 'accuracy', 'f1', 'sensitivity', 'specificity', 'precision']
    """
    models = dict(models) if models is not None else model_presets()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rows = []
    for name, estimator in models.items():
        pipe = make_tabular_pipeline(estimator)
        y_prob = _cv_probabilities(pipe, X, y, cv=cv)
        metrics = _binary_metrics(y.values, y_prob)
        metrics["model"] = name
        rows.append(metrics)

    df = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    return df


def compare_feature_sets(
    feature_sets: Mapping[str, pd.DataFrame],
    y: pd.Series,
    model_name: str = "Logistic_L1",
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Evaluate multiple feature sets (e.g., linguistic, acoustic, combined) with a single model.

    Parameters
    ----------
    feature_sets : Mapping[str, DataFrame]
        Dict of {set_name: X} matrices, all aligned to the same y.
    y : pd.Series
        Binary labels (0/1).
    model_name : str
        Which preset model to use ('Logistic_L1' or 'SVM_Linear').
    n_splits : int
        Number of CV folds (default: 5).
    random_state : int
        CV shuffling seed.

    Returns
    -------
    pd.DataFrame
        Metrics per feature set sorted by AUC (descending).
    """
    presets = model_presets()
    if model_name not in presets:
        raise ValueError(f"Unknown model_name='{model_name}'. Options: {list(presets)}")
    estimator = presets[model_name]

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []
    for set_name, X in feature_sets.items():
        pipe = make_tabular_pipeline(estimator)
        y_prob = _cv_probabilities(pipe, X, y, cv=cv)
        metrics = _binary_metrics(y.values, y_prob)
        metrics["feature_set"] = set_name
        rows.append(metrics)

    df = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    return df


def plot_roc_cv(
    X: pd.DataFrame,
    y: pd.Series,
    estimator,
    n_splits: int = 5,
    random_state: int = 42,
    title: str = "ROC (cross-validated)",
):
    """
    Plot a ROC curve using cross-validated probabilities for a single estimator.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pipe = make_tabular_pipeline(estimator)
    # Use predict_proba when available; else use decision_function→sigmoid
    try:
        y_prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    except Exception:
        scores = cross_val_predict(pipe, X, y, cv=cv, method="decision_function")
        y_prob = _sigmoid(scores)

    RocCurveDisplay.from_predictions(y, y_prob)
    plt.title(title)
    plt.show()
