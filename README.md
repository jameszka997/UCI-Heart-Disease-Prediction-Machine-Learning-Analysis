# UCI Heart Disease Prediction — Machine Learning Analysis

> Predicting heart disease presence from routine clinical indicators using interpretable machine learning — from exploratory data analysis through to SHAP-based feature attribution.

---

## Overview

Can doctors reliably predict the likelihood of heart disease from routinely collected health indicators alone — cholesterol levels, resting blood pressure, exercise response, and ECG results?

This project applies a full machine learning pipeline to the [UCI Heart Disease dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease) (Cleveland subset, 303 patients) to answer that question. The analysis covers data preprocessing, exploratory data analysis, benchmark model comparison, hyperparameter optimisation with XGBoost, and model interpretability using SHAP values.

The project was completed as a take-home data science exercise for a clinical research position and received strongly positive feedback.

---

## Dataset

| Property | Detail |
|---|---|
| Source | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data) |
| Patients | 297 (after removing 6 rows with invalid entries) |
| Features | 13 clinical and demographic predictors |
| Target | Binary — presence (1) vs. absence (0) of heart disease |

**Feature summary:**

| Feature | Description | Type |
|---|---|---|
| `age` | Patient age (years) | Numerical |
| `sex` | Sex (1 = Male, 0 = Female) | Categorical |
| `cp` | Chest pain type (1–4) | Categorical |
| `trestbps` | Resting blood pressure on admission (mm Hg) | Numerical |
| `chol` | Serum cholesterol (mg/dl) | Numerical |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = yes) | Binary |
| `restecg` | Resting ECG results (0–2) | Categorical |
| `thalach` | Maximum heart rate achieved (BPM) | Numerical |
| `exang` | Exercise-induced angina (1 = yes) | Binary |
| `oldpeak` | ST depression induced by exercise | Numerical |
| `slope` | Slope of peak exercise ST segment (1–3) | Categorical |
| `ca` | Number of major vessels by fluoroscopy (0–3) | Categorical |
| `thal` | Thallium stress test result | Categorical |

---

## Pipeline Summary

```
Data Import → Preprocessing → EDA → Benchmark Models → XGBoost + Hyperparameter Tuning → SHAP Interpretability
```

1. **Data preprocessing** — Identified and removed 6 rows with invalid `?` entries in `ca` and `thal` columns (~2% of data). Redefined the multi-class target into a binary outcome.
2. **Exploratory data analysis** — Correlation heatmap, distribution plots, and grouped visualisations examining relationships between clinical features and heart disease presence.
3. **Classification I — Benchmark models** — Logistic Regression and Decision Tree trained and evaluated using stratified 10-fold cross-validation across accuracy, sensitivity, and specificity.
4. **Classification II — XGBoost + RandomizedSearchCV** — Hyperparameter optimisation over 5 parameters using 5-fold CV, scored on F1. Compared optimised vs. default model performance.
5. **Classification III — SHAP interpretability** — TreeExplainer applied to the optimised XGBoost model to identify the most influential clinical predictors and their directional effects.

---

## Key Results

### Model Comparison (10-fold Cross-Validation)

| Model | Accuracy | Sensitivity | Specificity |
|---|---|---|---|
| Logistic Regression | **82.9% ± 4.2%** | **78.3%** | **86.9%** |
| Decision Tree | 73.8% ± 8.0% | 71.8% | 75.6% |

Logistic Regression outperforms the Decision Tree across all three metrics by 7–10 percentage points and shows notably lower variance across folds, indicating more consistent and reliable predictions on this dataset.

### XGBoost Optimisation

| Configuration | F1 Score |
|---|---|
| Default XGBoost | ~0.79 |
| Optimised (RandomizedSearchCV) | **0.8015** |

Optimal hyperparameters: `n_estimators=120`, `max_depth=3`, `learning_rate=0.051`, `subsample=0.769`, `colsample_bytree=0.719`. The shallow, regularised configuration reflects the limited dataset size.

### Top Predictors (SHAP Analysis)

The three strongest predictors of heart disease identified by SHAP:

1. **`ca`** — Number of major vessels by fluoroscopy
2. **`cp`** — Chest pain type (notably, asymptomatic presentation paradoxically increases predicted risk)
3. **`thal`** — Thallium stress test result (strongest binary separation, consistent with its primary clinical role in cardiac diagnostics)

Weakest contributors: `trestbps`, `restecg`, and `fbs`, all showing SHAP values clustered near zero.

---

## Reproducing the Analysis

### Requirements

```bash
pip install -r requirements.txt
```

### Running the notebook

```bash
jupyter notebook notebook/heart_disease_analysis.ipynb
```

The dataset is loaded directly from the UCI repository URL — no manual download required.

> **Note:** SHAP summary plots are generated interactively within the notebook. Run all cells to reproduce the full set of figures.

---

## Repository Structure

```
UCI_heart_disease_prediction_ml/
├── README.md
├── LICENSE
├── requirements.txt
├── notebook/
│   └── heart_disease_analysis.ipynb   # Full analysis notebook
├── report/
│   └── findings_summary.md            # Standalone written findings report
└── figures/                           # Exported key visualisations
    ├── eda_collage_1.png
    ├── model_comparison_boxplot.png
    ├── xgb_hyperparameter_scatter.png
    └── shap_summary_plot.png
```

---

## Dependencies

See `requirements.txt`. Core libraries:

- `pandas`, `numpy` — data handling
- `scikit-learn` — modelling, cross-validation, hyperparameter search
- `xgboost` — gradient boosting classifier
- `shap` — model interpretability
- `matplotlib`, `seaborn` — visualisation

---

## References & Acknowledgements

- UCI Heart Disease Dataset: [archive.ics.uci.edu](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- XGBoost implementation reference: [najeebuddinm98/xgboost_heartdisease_pred](https://github.com/najeebuddinm98/xgboost_heartdisease_pred/blob/main/prototyping.ipynb)
- Additional implementation references: [marianadeem755](https://github.com/marianadeem755/ModelTrainingOnHeartDisease-UCI-dataset) · [clarkszw](https://clarkszw.github.io/Python/Heart_Disease/logistic_regression.html) · [harinda0](https://github.com/harinda0/Heart-Disease-Prediction-ML/blob/main)

All code was written independently; the above resources were consulted for methodological orientation and visual implementation.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
