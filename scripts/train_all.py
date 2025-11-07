#!/usr/bin/env python
"""Train baseline models for NHANES liver classification project."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import (
    evaluate,
    get_feature_names,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pr,
    plot_roc,
)
from src.feature_prep import TARGET_COLUMN, build_preprocessor, infer_column_groups
from src.models import build_models

RESULTS_DIR = Path("results")
IMAGES_DIR = Path("images")
BEST_MODEL_PATH = RESULTS_DIR / "best_model.joblib"
BEST_MODEL_CONFIG_PATH = RESULTS_DIR / "best_model_config.json"
METRICS_TABLE_PATH = RESULTS_DIR / "metrics_all.csv"
SUMMARY_PATH = RESULTS_DIR / "summary.md"
DATA_PATH = Path("nhanes_cleaned.csv")

ABLATIONS = {
    "alcohol": "Alcohol-only features",
    "alcohol_demo": "Alcohol + Demographics",
    "all": "Alcohol + Demo + Labs",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train models for NHANES liver project.")
    parser.add_argument(
        "--ablation",
        choices=list(ABLATIONS.keys()),
        help="Run only one ablation setting.",
    )
    parser.add_argument(
        "--model",
        choices=["logreg", "rf", "xgb"],
        help="Run only one model.",
    )
    return parser.parse_args()


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_directories() -> None:
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    IMAGES_DIR.mkdir(exist_ok=True, parents=True)


def build_ablation_features(groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
    alcohol = groups["alcohol"]
    demo = groups["demo"]
    all_features = groups["all"]

    ablations = {
        "alcohol": sorted(set(alcohol)),
        "alcohol_demo": sorted(set(alcohol) | set(demo)),
        "all": sorted(set(all_features)),
    }
    return ablations


def prepare_feature_sets(
    df: pd.DataFrame,
    feature_cols: List[str],
    categorical_cols: List[str],
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    X = df[feature_cols].copy()

    categorical = [col for col in categorical_cols if col in feature_cols]
    numeric = [col for col in feature_cols if col not in categorical]

    for col in numeric:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    y = df[TARGET_COLUMN]

    return X, y, categorical, numeric


def save_metrics_artifacts(
    ablation: str,
    model_name: str,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> None:
    stem = f"{ablation}_{model_name}"
    cm_path = IMAGES_DIR / f"{stem}_confusion_matrix.png"
    roc_path = IMAGES_DIR / f"{stem}_roc.png"
    pr_path = IMAGES_DIR / f"{stem}_pr.png"

    plot_confusion_matrix(y_test, y_pred, str(cm_path))
    plot_roc(y_test, y_prob, str(roc_path))
    plot_pr(y_test, y_prob, str(pr_path))


def write_individual_metrics(
    ablation: str,
    model_name: str,
    record: Dict[str, float],
) -> None:
    stem = f"{ablation}_{model_name}"
    json_path = RESULTS_DIR / f"{stem}_metrics.json"
    csv_path = RESULTS_DIR / f"{stem}_metrics.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    pd.DataFrame([record]).to_csv(csv_path, index=False)


def record_feature_importance(
    pipeline: Pipeline,
    feature_names: List[str],
    ablation: str,
    model_name: str,
) -> None:
    estimator = pipeline.named_steps.get("model")
    if estimator is None:
        return
    plot_path = IMAGES_DIR / f"{ablation}_{model_name}_feature_importance.png"
    plot_feature_importance(estimator, feature_names, str(plot_path))


def summarize_to_markdown(df: pd.DataFrame, class_balance: Dict[str, float]) -> None:
    best_overall = df.sort_values("auroc", ascending=False).iloc[0]
    best_by_ablation = (
        df.sort_values("auroc", ascending=False)
        .groupby("ablation", as_index=False)
        .first()
    )

    lines = [
        "# Model Summary",
        "",
        "## Best Model Per Ablation",
    ]
    for _, row in best_by_ablation.iterrows():
        lines.append(
            f"- **{ABLATIONS[row['ablation']]}** → {row['model']} "
            f"(AUROC {row['auroc']:.3f}, F1 {row['f1']:.3f})"
        )

    lines.extend(
        [
            "",
            "## Overall Best Model",
            f"- {best_overall['model']} on {ABLATIONS[best_overall['ablation']]} "
            f"(AUROC {best_overall['auroc']:.3f}, Accuracy {best_overall['accuracy']:.3f}, "
            f"F1 {best_overall['f1']:.3f})",
            "",
            "## Class Balance",
            f"- Train positive rate: {class_balance['train_pos_rate']:.3f}",
            f"- Test positive rate: {class_balance['test_pos_rate']:.3f}",
            "",
            "## Notes",
            "- Median/mode imputation applied for missing values; heavy missingness may impact calibration.",
            "- Feature importances only available for tree-based models.",
        ]
    )

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seeds(42)
    ensure_directories()

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    print("Inferring column groups...")
    column_groups = infer_column_groups(df)
    ablation_features = build_ablation_features(column_groups)

    if args.ablation:
        ablation_features = {args.ablation: ablation_features[args.ablation]}

    require_xgb = args.model in (None, "xgb")
    models = build_models(require_xgb=require_xgb)
    if args.model:
        if args.model not in models:
            raise ValueError(f"Requested model '{args.model}' is not available.")
        models = {args.model: models[args.model]}

    results_records = []
    best_result = None
    best_pipeline: Pipeline | None = None
    best_metadata = {}

    for ablation_key, feature_cols in ablation_features.items():
        print(f"\n=== Ablation: {ABLATIONS[ablation_key]} ({len(feature_cols)} features) ===")
        if not feature_cols:
            print("No features available; skipping.")
            continue

        X, y, categorical_cols, numeric_cols = prepare_feature_sets(
            df, feature_cols, column_groups["categorical"]
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        for model_name, model in models.items():
            print(f"Training model: {model_name}")
            preprocessor = build_preprocessor(categorical_cols, numeric_cols)
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", clone(model)),
                ]
            )

            pipeline.fit(X_train, y_train)

            metrics = evaluate(pipeline, X_test, y_test)
            y_pred = pipeline.predict(X_test)
            if hasattr(pipeline, "predict_proba"):
                y_prob = pipeline.predict_proba(X_test)[:, 1]
            elif hasattr(pipeline, "decision_function"):
                scores = pipeline.decision_function(X_test)
                y_prob = 1 / (1 + np.exp(-scores))
            else:
                y_prob = y_pred

            record = {
                "ablation": ablation_key,
                "model": model_name,
                "accuracy": float(metrics["accuracy"]),
                "f1": float(metrics["f1"]),
                "auroc": float(metrics["auroc"]),
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
                "pos_rate_train": float(y_train.mean()),
                "pos_rate_test": float(y_test.mean()),
            }
            results_records.append(record)

            write_individual_metrics(ablation_key, model_name, record)
            save_metrics_artifacts(ablation_key, model_name, y_test, y_pred, y_prob)

            if model_name in {"rf", "xgb"}:
                feature_names = get_feature_names(
                    pipeline.named_steps["preprocessor"]
                )
                record_feature_importance(pipeline, feature_names, ablation_key, model_name)

            if best_result is None or record["auroc"] > best_result["auroc"]:
                best_result = record
                best_pipeline = pipeline
                best_metadata = {
                    "ablation": ablation_key,
                    "model": model_name,
                    "feature_columns": list(feature_cols),
                    "categorical_columns": list(categorical_cols),
                    "numeric_columns": list(numeric_cols),
                    "metrics": metrics,
                    "test_indices": X_test.index.tolist(),
                }

    if not results_records:
        raise RuntimeError("No models were trained.")

    comparison_df = pd.DataFrame(results_records).sort_values("auroc", ascending=False)
    comparison_df.to_csv(METRICS_TABLE_PATH, index=False)

    class_balance = {
        "train_pos_rate": comparison_df.iloc[0]["pos_rate_train"],
        "test_pos_rate": comparison_df.iloc[0]["pos_rate_test"],
    }
    summarize_to_markdown(comparison_df, class_balance)

    print("\n=== Model Comparison (sorted by AUROC) ===")
    print(
        comparison_df[
            ["ablation", "model", "accuracy", "f1", "auroc", "n_train", "n_test"]
        ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    if best_pipeline is not None:
        joblib.dump(best_pipeline, BEST_MODEL_PATH)
        with BEST_MODEL_CONFIG_PATH.open("w", encoding="utf-8") as f:
            json.dump(best_metadata, f, indent=2)
        print(
            f"\nBest model: {best_result['model']} on {ABLATIONS[best_result['ablation']]} "
            f"(AUROC {best_result['auroc']:.3f})"
        )
        print(f"Saved best pipeline to {BEST_MODEL_PATH}")
        print(f"Saved configuration to {BEST_MODEL_CONFIG_PATH}")


if __name__ == "__main__":
    main()
