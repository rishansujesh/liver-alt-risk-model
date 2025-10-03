# Predicting Elevated Liver Enzyme Levels from Alcohol Consumption and Lifestyle Factors in NHANES

---

## 1. Introduction / Background

Alcohol consumption is a leading cause of chronic liver disease worldwide. Elevated liver enzymes — such as alanine aminotransferase (ALT), aspartate aminotransferase (AST), gamma-glutamyl transferase (GGT), bilirubin, and alkaline phosphatase — are widely used biomarkers of liver stress.

### ✅ Literature Review  
- Whitfield demonstrated the utility of GGT as an alcohol biomarker [1].  
- Salaspuro reviewed enzyme-based diagnosis of alcohol-related organ damage [2].  
- The CDC’s NHANES survey provides nationally representative population data on alcohol, demographics, and lab tests [3].  

### ✅ Dataset Description  
We use the 2017–March 2020 Pre-Pandemic NHANES cycle, merging:  
- **Demographics (P_DEMO):** age, sex, race, education, income  
- **Alcohol Use (P_ALQ):** drinking frequency, drinks/day, binge episodes, years drinking  
- **Biochemistry (P_BIOPRO / Lab modules):** ALT, AST, GGT, bilirubin, albumin, alkaline phosphatase  
Roughly ~8,000–10,000 participants, ~50-100 features after merging and cleaning.  

### ✅ Dataset Link  
[CDC NHANES Continuous Datasets](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx) :contentReference[oaicite:0]{index=0}  
https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_DEMO.htm
https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_BIOPRO.htm
https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_ALQ.htm

---

## 2. Problem Definition

### ✅ Problem  
Predict whether an individual has **elevated ALT levels** (above a clinically meaningful threshold) from alcohol use, demographics, and lifestyle features.

### ✅ Motivation  
ALT elevation is a marker of liver stress and early damage. A predictive model could highlight the relative contributions of different alcohol and lifestyle variables. This may inform public health strategies or targeted interventions.

---

## 3. Methods

### ✅ 3+ Data Preprocessing Methods Identified  
We plan to use standard scikit-learn tools:  
- `SimpleImputer(strategy="median")` for missing numeric values, and `strategy="most_frequent"` for categoricals  
- `OneHotEncoder(handle_unknown="ignore")` for categorical encoding  
- `StandardScaler()` for feature scaling  
- Use `Pipeline` and `ColumnTransformer` to combine preprocessing steps  

We may also include:  
- Feature selection (e.g. `sklearn.feature_selection.SelectKBest`)  
- Principal component analysis (PCA) or dimensionality reduction if needed  

### ✅ 3+ ML Algorithms/Models Identified  
- Logistic Regression: `sklearn.linear_model.LogisticRegression()` (baseline, interpretable)  
- Random Forest: `sklearn.ensemble.RandomForestClassifier()` (nonlinear, good default)  
- XGBoost: `xgboost.XGBClassifier()` (strong performance on tabular)  
- (Stretch) Neural Network: `sklearn.neural_network.MLPClassifier()`  


---

## 4. (Potential) Results and Discussion

### ✅ 3+ Quantitative Metrics  
We will evaluate using:  
- `accuracy_score`  
- `f1_score`  
- `roc_auc_score`  
- (Optional) As a regression fallback: `mean_absolute_error`, `mean_squared_error`  

### ✅ Project Goals  
- Build models that are **interpretable** and highlight key predictors.  
- Evaluate **fairness** across subgroups (e.g. sex, race).  
- Address **ethical concerns**: explicit disclaimers that this is not medical diagnosis.  
- Ensure **sustainability / efficiency**: models should be relatively lightweight and scalable.  

### ✅ Expected Results  
- Logistic Regression AUC ~ 0.65–0.70  
- Random Forest / XGBoost AUC ~ 0.75–0.80  
- Top predictors likely: drinks/week, years drinking, binge episodes, age, sex  

---

## 5. References

[1] E. S. Whitfield, “Gamma glutamyl transferase,” *Crit. Rev. Clin. Lab. Sci.*, vol. 38, no. 4, pp. 263–355, 2001.  
[2] C. M. Salaspuro, “Use of enzymes for the diagnosis of alcohol-related organ damage,” *Enzyme*, vol. 41, no. 1, pp. 17–28, 1991.  
[3] Centers for Disease Control and Prevention, “NHANES Continuous Datasets,” CDC/NCHS. :contentReference[oaicite:1]{index=1}  

---

## 6. Gantt Chart  
https://docs.google.com/spreadsheets/d/1DGG8UT4Gw_CVyhd8yabUYZYYpycG1OVeVBzA1P4FZMg/edit?usp=sharing

## 7. Contribution Table

| Name      | Proposal Contributions                                 |
|-----------|----------------------------------------------------------|
| Rishan    | Dataset merging, literature review, baseline logistic regression |
| Avaneesh  | Preprocessing, feature engineering, fairness analyses    |
| Ashfiq    | Random Forest and XGBoost implementation and tuning      |
| Sai       | Neural net stretch goal, hyperparameter tuning, comparisons |
|           | Slides, GitHub Pages setup, video recording & editing    |
