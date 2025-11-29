# README.md

# Detailed Report: <a href="finalreport.md" target="_blank">Final Report</a>
---

`/src/`: Source modules for feature grouping, preprocessing, model builders, and evaluation helpers used by the training scripts.  
`/src/feature_prep.py`: Infers column groups (categorical, numeric, alcohol, demographics, labs) and builds the ColumnTransformer used in all experiments.  
`/src/models.py`: Defines the sklearn estimators currently used (Logistic Regression baseline plus tree-based backup) with shared hyperparameters.  
`/src/evaluate.py`: Provides metric computation along with confusion matrix, ROC, and PR plotting utilities and transformed feature-name extraction.  

---

`/scripts/`: Command-line entry points for training and fairness analysis.  
`/scripts/train_all.py`: Pipeline that loads `nhanes_cleaned.csv`, runs the configured ablation, trains the selected model (currently verified with Logistic Regression), saves plots/metrics to `images/` and `results/`, and persists the best model artifacts.  
`/scripts/fairness_by_sex.py`: Reloads the saved best pipeline, evaluates RIAGENDR subgroups on the held-out test split, and writes `results/fairness_by_sex.csv` with a short console interpretation.  

---

`/results/`: Auto-generated artifacts such as per-run metrics, consolidated tables, summaries, fairness outputs, and serialized models (created by the two scripts).  
`/results/metrics_all.csv`: Aggregated comparison of accuracy, F1, and AUROC with dataset stats for each ablation/model that has been executed so far.  
`/results/model_comparison.csv`: Comparison table from the multi-model evaluation (Logistic Regression, Random Forest, XGBoost).  
`/results/summary.md`: Markdown report capturing the best run and class-balance notes from the latest training invocation.  
`/results/fairness_by_sex.csv`: Subgroup metrics (Accuracy, F1, AUROC) for RIAGENDR slices computed from the saved best model.  
`/results/best_model.joblib`: Joblib-serialized sklearn Pipeline for the top-performing configuration, reused for fairness evaluation.  
`/results/best_model_config.json`: Metadata describing the winning run (ablation, columns, metrics, and test indices) to keep evaluations reproducible.  

---

`/images/`: Visualization outputs (confusion matrices, ROC curves, and PR curves) produced per ablation/model run at 144 dpi.  
`/images/*_confusion_matrix.png`: Confusion matrix heatmaps labeled by ablation and model name for quick error inspection.  
`/images/*_roc.png`: ROC curves with AUROC annotations for each evaluated model/ablation pair.  
`/images/*_pr.png`: Precision–Recall curves summarizing performance under class imbalance.  


# Newly added final-report images:
`/images/logistic_regression_confusion_matrix_tuned.png`: Tuned-threshold Logistic Regression confusion matrix.  
`/images/random_forest_confusion_matrix_tuned.png`: Tuned-threshold Random Forest confusion matrix.  
`/images/xgboost_confusion_matrix_tuned.png`: Tuned-threshold XGBoost confusion matrix.  

`/images/logistic regression_confusion_matrix.png`: Baseline Logistic Regression confusion matrix.  
`/images/logistic regression_roc.png`: Logistic Regression ROC curve.  
`/images/logistic regression_pr.png`: Logistic Regression PR curve.  

`/images/random forest_confusion_matrix.png`: Baseline Random Forest confusion matrix.  
`/images/random forest_roc.png`: Random Forest ROC curve.  
`/images/random forest_pr.png`: Random Forest PR curve.  

`/images/xgboost_confusion_matrix.png`: Baseline XGBoost confusion matrix.  
`/images/xgboost_roc.png`: XGBoost ROC curve.  
`/images/xgboost_pr.png`: XGBoost PR curve.  


---

`/nhanes_cleaned.csv`: Canonical dataset created by the preprocessing notebook; serves as the single source of truth for all experiments.  
`/00_clean_dataset.py`: Script/notebook used to clean raw NHANES tables and generate `nhanes_cleaned.csv`.  

`/01_baseline_logreg.ipynb`:  
Exploratory notebook that performs the full modeling workflow:  
- Loads `nhanes_cleaned.csv` and constructs the preprocessing pipeline  
- Trains the baseline Logistic Regression model  
- Generates baseline confusion matrix, ROC curve, and PR curve  
- Trains Random Forest and XGBoost models for comparison  
- Produces all model visualizations (CM/ROC/PR)  
- Computes tuned thresholds for each model and generates tuned confusion matrices  
This notebook contains the entire analysis used in both the midterm and final reports.

`/nhanes_merge.py`: Utility for merging the NHANES XPT files into an intermediate CSV before cleaning; useful for data provenance.  

---

`/P_ALQ.xpt`: Raw NHANES Alcohol Use module (XPT format) used when regenerating the cleaned dataset.  
`/P_BIOPRO.xpt`: Raw NHANES laboratory biomarkers file required by the merge/cleaning scripts.  
`/P_DEMO.xpt`: Raw NHANES demographics module consumed by the merge/cleaning scripts.  

---
