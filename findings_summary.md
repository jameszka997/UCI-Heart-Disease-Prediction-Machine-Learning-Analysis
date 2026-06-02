# Heart Disease Prediction — Findings Summary

*UCI Cleveland Dataset · Binary Classification · 297 Patients*

---

## Introduction

This report summarises the findings from a machine learning analysis aimed at predicting the presence of heart disease from routine clinical measurements. The dataset used is the Cleveland subset of the UCI Heart Disease dataset, containing 13 clinical and demographic features for 303 patients, reduced to 297 after preprocessing.

The central question: can standard, routinely-collected health indicators — cholesterol, resting blood pressure, ECG results, exercise response — reliably predict whether a patient has heart disease?

---

## Data Preprocessing

The dataset was imported directly from the UCI repository with no column headers, requiring manual assignment of feature names based on the dataset documentation.

Two issues were identified and resolved during preprocessing:

**Target variable redefinition.** The original target column encodes disease severity on a 0–4 scale. For binary classification, this was recoded as 0 (no disease) vs. 1 (any disease presence), following standard practice for this dataset.

**Invalid entries in `ca` and `thal`.** Six rows (approximately 2% of the dataset) contained `?` placeholder values in the `ca` (number of major vessels) and `thal` (thallium stress test) columns. Because both are categorical features, mean or median imputation would be clinically meaningless, so the affected rows were dropped. The small proportion justifies this approach without materially affecting the analysis.

After cleaning, the dataset contains 297 patients with a near-balanced target: 138 with no heart disease (46.5%) and 159 with heart disease (53.5%).

---

## Exploratory Data Analysis

Several patterns emerged from the EDA that are consistent with established clinical knowledge.

**Sex and disease prevalence.** Male patients account for the large majority of heart disease cases in this dataset, despite a more even distribution across no-disease patients. This aligns with the well-documented higher cardiovascular risk in male populations, particularly in the age range represented here.

**Chest pain type is counterintuitive.** Patients with asymptomatic chest pain (type 4) show a higher rate of confirmed heart disease than those reporting typical angina (type 1). This reflects a known clinical paradox: the absence of classic symptoms does not reduce risk, and may be associated with more advanced disease.

**Age and maximum heart rate.** Older patients tend to achieve lower maximum heart rates during stress testing (`thalach`), and this group shows a higher concentration of heart disease cases. The scatter of age vs. thalach shows reasonable separation between disease-positive and disease-negative patients, suggesting thalach is an informative predictor.

**Correlation structure.** The correlation heatmap reveals moderate positive correlations between `cp`, `thalach`, and the target (indicating disease), and moderate negative correlations between `ca`, `oldpeak`, `exang`, and the target. Most feature pairs show low inter-correlation, which is favourable for model training.

---

## Classification I — Benchmark Model Comparison

Two baseline classifiers were trained and evaluated using stratified 10-fold cross-validation: a Logistic Regression model and a Decision Tree (max depth 5). Three metrics were computed across all folds: accuracy, sensitivity (recall), and specificity.

### Results

| Model | Accuracy | Sensitivity | Specificity |
|---|---|---|---|
| Logistic Regression | 82.9% ± 4.2% | 78.3% | 86.9% |
| Decision Tree | 73.8% ± 8.0% | 71.8% | 75.6% |

### Interpretation

**Logistic Regression outperforms the Decision Tree across all metrics by 7–10 percentage points.** The gap is consistent whether measuring overall accuracy, the ability to correctly identify true disease cases (sensitivity), or the ability to correctly identify true healthy cases (specificity).

Equally important is the variance. The Decision Tree shows a standard deviation of ±8.0% in accuracy across folds — nearly double that of the Logistic Regression (±4.2%). This indicates the Decision Tree is more sensitive to which patients end up in which fold, suggesting some overfitting despite the max-depth constraint.

**Sensitivity vs. specificity trade-off.** For heart disease screening, sensitivity is the higher-stakes metric: a false negative (missing a patient who has the disease) carries greater clinical cost than a false positive. The Logistic Regression achieves 78.3% sensitivity, meaning roughly 1 in 5 true disease patients are misclassified as healthy. While this is acceptable for a baseline model trained on 297 patients, it underscores the importance of clinical judgement alongside any algorithmic prediction.

**Model recommendation.** Logistic Regression is the preferred choice from this baseline comparison. It offers higher and more consistent performance, is more interpretable (coefficients can be directly examined), and generalises more reliably on this dataset size. Decision trees are prone to instability on small datasets, which the fold variance reflects.

---

## Classification II — XGBoost Hyperparameter Optimisation

### Setup

An XGBoost classifier was tuned using `RandomizedSearchCV` with 5-fold cross-validation, optimising for F1 score. Twenty parameter configurations were sampled from the following distributions:

| Hyperparameter | Search Range |
|---|---|
| `n_estimators` | 50–250 |
| `max_depth` | 3–10 |
| `learning_rate` | 0.01–0.26 |
| `subsample` | 0.6–1.0 |
| `colsample_bytree` | 0.6–1.0 |

### Optimal Configuration

The search identified the following best parameters with a cross-validation F1 score of **0.8015**:

| Parameter | Optimal Value |
|---|---|
| `n_estimators` | 120 |
| `max_depth` | 3 |
| `learning_rate` | 0.051 |
| `subsample` | 0.769 |
| `colsample_bytree` | 0.719 |

The default XGBoost model achieved a mean F1 score of approximately 0.79, representing a modest improvement of roughly 1 percentage point from optimisation.

### Hyperparameter Sensitivity Analysis

Scatter plots of F1 score against each hyperparameter reveal a broadly flat landscape:

- **Max depth:** No clear trend across values 3–10. Shallow trees (depth 3) perform on par with deeper ones, suggesting the dataset does not support highly complex tree structures.
- **Learning rate:** Lower rates (0.01–0.10) produce more consistently high F1 scores. Higher rates (>0.20) introduce more variance, consistent with the principle that smaller steps lead to more stable convergence.
- **N estimators and sampling parameters:** F1 scores remain stable across the full search range for these parameters, confirming the model converges quickly and is not sensitive to ensemble size or sampling ratios on this dataset.

The flat performance surface across most hyperparameters is expected given the small dataset size. XGBoost's default configuration was already near-optimal for 297 patients with 13 features. A larger, more complex dataset would likely show steeper sensitivity to these choices and greater gains from tuning.

---

## Classification III — Model Interpretability with SHAP

SHAP (SHapley Additive exPlanations) values were computed using `TreeExplainer` on the optimised XGBoost model. The summary plot shows each feature's contribution to individual predictions, with colour indicating feature value (red = high, blue = low) and horizontal position indicating the direction and magnitude of impact on the predicted probability of heart disease.

### Key Findings

**Strongest predictors:**

`ca` — Number of major vessels coloured by fluoroscopy is the most impactful feature. Higher values (more obstructed vessels) are strongly associated with increased heart disease probability, which is directly consistent with the clinical interpretation of coronary artery disease severity.

`cp` — Chest pain type shows clear separation between high and low values. As noted in EDA, asymptomatic presentation (high cp value) increases predicted disease probability — a well-documented paradox in cardiology where atypical or absent symptoms may indicate more advanced disease.

`thal` — Thallium stress test result shows the sharpest binary separation in SHAP values of any feature. High values (reversible defect) strongly increase disease probability; low values (normal result) strongly decrease it. This is consistent with `thal` being one of the primary clinical tools used in cardiac diagnostics.

**Moderate predictors:**

`oldpeak` — ST depression induced by exercise is positively associated with disease probability. Higher ST depression reflects reduced myocardial perfusion under stress, a standard diagnostic criterion.

`sex` — Male sex offers a clear directional signal, with male patients showing increased predicted risk. This is the most interpretable binary predictor in the model and aligns with epidemiological evidence.

`thalach` — Lower maximum heart rate achieved is associated with higher disease probability. Reduced exercise capacity is a clinically recognised indicator of cardiovascular compromise.

`age` — Older age generally increases disease probability, though the relationship is noisier than the top three features. A spread of both high and low SHAP values across age values suggests age alone is not a reliable discriminator in this dataset.

**Weakest predictors:**

`trestbps` (resting blood pressure), `restecg` (resting ECG), and `fbs` (fasting blood sugar) all show SHAP values tightly clustered near zero, contributing minimally to model predictions. This is somewhat surprising for resting blood pressure and ECG, which are commonly used clinical indicators, but may reflect the limited sample size and the stronger signal carried by stress-test-based features (`thal`, `oldpeak`, `thalach`).

### Clinical Relevance

The SHAP results tell a coherent clinical story: the most predictive features in this dataset are those that capture the heart's behaviour under stress (`thal`, `oldpeak`, `thalach`, `exang`) rather than resting-state measurements. This is consistent with the clinical rationale for exercise stress testing — disease that is not apparent at rest becomes visible under physiological demand. The model is, in effect, learning the same heuristic that clinicians apply when ordering stress tests.

---

## Ideas for Future Work

Several directions would extend and strengthen this analysis:

**Larger and more diverse datasets.** The Cleveland subset contains 297 patients from a single centre. Applying these models to the full multi-centre UCI dataset (Cleveland, Hungary, Switzerland, and VA Long Beach) would test geographic and demographic generalisability, and allow examination of whether the feature importance hierarchy holds across populations with different dietary patterns and environmental risk factors.

**Feature selection.** Given that `trestbps`, `restecg`, and `fbs` show near-zero SHAP values, it would be worthwhile to retrain models excluding these features and measure whether predictive performance is preserved or improved. Reducing feature count also has practical clinical value — fewer required measurements lowers the cost and burden of data collection.

**Threshold optimisation.** Default classification thresholds (0.5) were used throughout. Given the clinical asymmetry between false negatives and false positives in disease screening, an ROC analysis with explicit threshold tuning for sensitivity targets would better reflect real-world deployment requirements.

**External validation.** All evaluations here use cross-validation on the same dataset. Validating on a fully external, held-out dataset from a different source would provide a more honest estimate of real-world performance.
