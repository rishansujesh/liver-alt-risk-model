"""Evaluation utilities for training scripts."""

from __future__ import annotations

import os
from typing import Dict, List, Sequence

import numpy as np
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def evaluate(pipe, X_test, y_test) -> Dict[str, float]:
    """Evaluate a fitted pipeline on the held-out test data."""
    y_pred = pipe.predict(X_test)

    if hasattr(pipe, "predict_proba"):
        y_prob = pipe.predict_proba(X_test)[:, 1]
    elif hasattr(pipe, "decision_function"):
        scores = pipe.decision_function(X_test)
        y_prob = 1 / (1 + np.exp(-scores))
    else:
        # fall back to binary predictions
        y_prob = y_pred

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auroc": roc_auc_score(y_test, y_prob),
    }
    return metrics


def plot_confusion_matrix(y_true, y_pred, outpath: str) -> None:
    """Plot and save a confusion matrix."""
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(outpath, dpi=144, bbox_inches="tight")
    plt.close()


def plot_roc(y_true, y_prob, outpath: str) -> None:
    """Plot and save ROC curve."""
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=144, bbox_inches="tight")
    plt.close()


def plot_pr(y_true, y_prob, outpath: str) -> None:
    """Plot and save Precision-Recall curve."""
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label=f"PR curve (AUC = {pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(outpath, dpi=144, bbox_inches="tight")
    plt.close()


def plot_feature_importance(
    model,
    feature_names: Sequence[str],
    outpath: str,
    top_k: int = 15,
) -> None:
    """Plot feature importances for tree-based models."""
    importance = getattr(model, "feature_importances_", None)
    if importance is None or len(importance) == 0:
        return

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    indices = np.argsort(importance)[::-1][:top_k]
    top_features = np.array(feature_names)[indices]
    top_importances = importance[indices]

    plt.figure(figsize=(8, max(4, top_k * 0.3)))
    plt.barh(top_features[::-1], top_importances[::-1])
    plt.xlabel("Importance")
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.savefig(outpath, dpi=144, bbox_inches="tight")
    plt.close()


def get_feature_names(column_transformer: ColumnTransformer) -> List[str]:
    """Extract transformed feature names from a fitted ColumnTransformer."""
    if not hasattr(column_transformer, "transformers_"):
        raise ValueError("ColumnTransformer is not fitted yet.")

    feature_names: List[str] = []

    for name, transformer, cols in column_transformer.transformers_:
        if transformer == "drop" or len(cols) == 0:
            continue
        if name == "remainder":
            if isinstance(transformer, str) and transformer == "drop":
                continue
            if transformer == "passthrough":
                if isinstance(cols, slice):
                    raise ValueError("Slice columns not supported for passthrough.")
                feature_names.extend(cols)
            continue

        extracted = _extract_feature_names(transformer, cols)
        feature_names.extend(extracted)

    return feature_names


def _extract_feature_names(transformer, input_features: Sequence[str]) -> List[str]:
    """Helper to pull feature names from different transformer types."""
    if hasattr(transformer, "get_feature_names_out"):
        names = transformer.get_feature_names_out(input_features)
        return list(names)

    if isinstance(transformer, Pipeline):
        last_step = transformer.steps[-1][1]
        if hasattr(last_step, "get_feature_names_out"):
            return list(last_step.get_feature_names_out(input_features))
        if isinstance(last_step, OneHotEncoder):
            return list(last_step.get_feature_names_out(input_features))
        # fall back to penultimate steps if available
        for _, step in reversed(transformer.steps):
            if hasattr(step, "get_feature_names_out"):
                return list(step.get_feature_names_out(input_features))

    # Default: return original feature names
    if isinstance(input_features, (list, tuple)):
        return list(input_features)
    return [input_features]


__all__ = [
    "evaluate",
    "plot_confusion_matrix",
    "plot_roc",
    "plot_pr",
    "plot_feature_importance",
    "get_feature_names",
]
