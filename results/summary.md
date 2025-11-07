# Model Summary

## Best Model Per Ablation
- **Alcohol-only features** → logreg (AUROC 0.583, F1 0.173)

## Overall Best Model
- logreg on Alcohol-only features (AUROC 0.583, Accuracy 0.738, F1 0.173)

## Class Balance
- Train positive rate: 0.078
- Test positive rate: 0.078

## Notes
- Median/mode imputation applied for missing values; heavy missingness may impact calibration.
- Feature importances only available for tree-based models.