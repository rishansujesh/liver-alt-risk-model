/src/: Source modules for feature grouping, preprocessing, model builders, and evaluation helpers used by the training scripts.
/src/feature_prep.py: Infers column groups (categorical, numeric, alcohol, demographics, labs) and builds the ColumnTransformer used in all experiments.
/src/models.py: Defines the sklearn estimators currently used (Logistic Regression baseline plus tree-based backup) with shared hyperparameters.
/src/evaluate.py: Provides metric computation along with confusion matrix, ROC, and PR plotting utilities and transformed feature-name extraction.

/scripts/: Command-line entry points for training and fairness analysis.
/scripts/train_all.py: Pipeline that loads `nhanes_cleaned.csv`, runs the configured ablation, trains the selected model (currently verified with Logistic Regression), saves plots/metrics to `images/` and `results/`, and persists the best model artifacts.
/scripts/fairness_by_sex.py: Reloads the saved best pipeline, evaluates RIAGENDR subgroups on the held-out test split, and writes `results/fairness_by_sex.csv` with a short console interpretation.

/results/: Auto-generated artifacts such as per-run metrics, consolidated tables, summaries, fairness outputs, and serialized models (created by the two scripts).
/results/metrics_all.csv: Aggregated comparison of accuracy/F1/AUROC with dataset stats for each ablation/model that has been executed so far.
/results/summary.md: Markdown report capturing the best run and class-balance notes from the latest training invocation.
/results/fairness_by_sex.csv: Subgroup metrics (Accuracy/F1/AUROC) for RIAGENDR slices computed from the saved best model.
/results/best_model.joblib: Joblib-serialized sklearn Pipeline for the top-performing configuration, reused for fairness evaluation.
/results/best_model_config.json: Metadata describing the winning run (ablation, columns, metrics, and test indices) to keep evaluations reproducible.

/images/: Visualization outputs (confusion matrices, ROC curves, and PR curves) produced per ablation/model run at 144 dpi.
/images/*_confusion_matrix.png: Confusion matrix heatmaps labeled by ablation and model name for quick error inspection.
/images/*_roc.png: ROC curves with AUROC annotations for each evaluated model/ablation pair.
/images/*_pr.png: Precision–Recall curves summarizing performance under class imbalance.

/nhanes_cleaned.csv: Canonical dataset created by the preprocessing notebook; serves as the single source of truth for all experiments.
/00_clean_dataset.py: Script/notebook used to clean raw NHANES tables and generate `nhanes_cleaned.csv`.
/01_baseline_logreg.ipynb: Exploratory notebook for baseline Logistic Regression experiments that informed the scripted pipeline.
/nhanes_merge.py: Utility for merging the NHANES XPT files into an intermediate CSV before cleaning; useful for data provenance.

/P_ALQ.xpt: Raw NHANES Alcohol Use module (XPT format) used when regenerating the cleaned dataset.
/P_BIOPRO.xpt: Raw NHANES laboratory biomarkers file required by the merge/cleaning scripts.
/P_DEMO.xpt: Raw NHANES demographics module consumed by the merge/cleaning scripts.
