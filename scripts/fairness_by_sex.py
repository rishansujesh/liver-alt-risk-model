#!/usr/bin/env python
"""Evaluate fairness metrics for the best NHANES liver model."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_prep import TARGET_COLUMN

RESULTS_DIR = Path("results")
BEST_MODEL_PATH = RESULTS_DIR / "best_model.joblib"
BEST_MODEL_CONFIG_PATH = RESULTS_DIR / "best_model_config.json"
FAIRNESS_OUTPUT_PATH = RESULTS_DIR / "fairness_by_sex.csv"
DATA_PATH = Path("nhanes_cleaned.csv")


def load_best_model():
    if not BEST_MODEL_PATH.exists() or not BEST_MODEL_CONFIG_PATH.exists():
        raise FileNotFoundError("Best model artifacts not found. Run scripts/train_all.py first.")

    pipeline = joblib.load(BEST_MODEL_PATH)
    metadata = json.loads(BEST_MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    return pipeline, metadata


def prepare_data(metadata: Dict) -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df


def select_test_split(
    df: pd.DataFrame,
    feature_columns: List[str],
    numeric_columns: List[str],
    test_indices: List[int] | None,
):
    X = df[feature_columns].copy()
    for col in numeric_columns:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    y = df[TARGET_COLUMN]

    if test_indices is None:
        # Fallback to reproducible split if indices missing
        from sklearn.model_selection import train_test_split

        _, X_test, _, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
    else:
        X_test = X.loc[test_indices]
        y_test = y.loc[test_indices]

    return X_test, y_test


def compute_metrics(pipeline, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    if len(X) == 0:
        return {"accuracy": np.nan, "f1": np.nan, "auroc": np.nan}

    y_pred = pipeline.predict(X)
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X)[:, 1]
    elif hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(X)
        y_prob = 1 / (1 + np.exp(-scores))
    else:
        y_prob = y_pred

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "f1": f1_score(y, y_pred, zero_division=0),
    }

    try:
        metrics["auroc"] = roc_auc_score(y, y_prob)
    except ValueError:
        metrics["auroc"] = np.nan

    return metrics


def main() -> None:
    pipeline, metadata = load_best_model()
    df = prepare_data(metadata)

    if "RIAGENDR" not in df.columns:
        raise KeyError("RIAGENDR column missing; cannot compute fairness by sex.")

    feature_columns = metadata.get("feature_columns", [])
    numeric_columns = metadata.get("numeric_columns", [])
    test_indices = metadata.get("test_indices")

    X_test, y_test = select_test_split(df, feature_columns, numeric_columns, test_indices)
    sex_series = df.loc[X_test.index, "RIAGENDR"]

    subgroup_metrics = []
    for sex_value in [1, 2]:
        mask = sex_series == sex_value
        X_slice = X_test.loc[mask]
        y_slice = y_test.loc[mask]
        metrics = compute_metrics(pipeline, X_slice, y_slice)
        subgroup_metrics.append(
            {
                "subgroup": f"RIAGENDR=={sex_value}",
                "n": int(mask.sum()),
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "auroc": metrics["auroc"],
            }
        )

    results_df = pd.DataFrame(subgroup_metrics)
    results_df.to_csv(FAIRNESS_OUTPUT_PATH, index=False)

    aurocs = results_df.set_index("subgroup")["auroc"]
    counts = results_df.set_index("subgroup")["n"]

    if aurocs.notna().all():
        diff = abs(aurocs["RIAGENDR==1"] - aurocs["RIAGENDR==2"])
        print(
            f"RIAGENDR AUROC gap: {diff:.3f} "
            f"(n1={int(counts['RIAGENDR==1'])}, n2={int(counts['RIAGENDR==2'])})."
        )
        print("Review results/fairness_by_sex.csv for detailed subgroup metrics.")
    else:
        print("Insufficient class variation for one subgroup; AUROC marked as NaN.")
        print("Review results/fairness_by_sex.csv for detailed subgroup metrics.")


if __name__ == "__main__":
    main()
