"""Model construction utilities."""

from __future__ import annotations

import warnings
from typing import Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier  # type: ignore
except ImportError:  # pragma: no cover - depends on environment
    XGBClassifier = None


def build_models(require_xgb: bool = True) -> Dict[str, object]:
    """Instantiate the baseline models used in the project."""
    models: Dict[str, object] = {
        "logreg": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        ),
        "rf": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
    }

    if XGBClassifier is None:
        if require_xgb:
            raise ImportError(
                "xgboost is required for the XGBClassifier model. "
                "Install it with `pip install xgboost` and retry."
            )
        warnings.warn(
            "xgboost not installed; skipping XGBClassifier model.",
            RuntimeWarning,
        )
    else:
        models["xgb"] = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
            n_jobs=-1,
        )

    return models


def small_param_grids() -> Dict[str, Dict[str, list]]:
    """Provide lightweight parameter grids for optional tuning."""
    grids: Dict[str, Dict[str, list]] = {
        "rf": {
            "model__n_estimators": [200, 300, 400],
            "model__max_depth": [None, 8, 12],
        },
    }
    if XGBClassifier is not None:
        grids["xgb"] = {
            "model__max_depth": [3, 4, 5],
            "model__learning_rate": [0.05, 0.1],
            "model__subsample": [0.85, 0.9, 1.0],
        }

    return grids


__all__ = ["build_models", "small_param_grids"]
