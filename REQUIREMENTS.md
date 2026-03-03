# Requirements

This document describes the Python dependencies required to run the scripts in the `python/` folder.

## Python version

Python **3.12** (the version used by the project virtualenv).

## Installation

```bash
# Activate the project virtualenv first
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

---

## Dependencies

| Package | Min version | Purpose |
|---|---|---|
| `pandas` | 2.0 | DataFrame I/O, data filtering and manipulation |
| `numpy` | 1.26 | Numerical arrays and math operations |
| `matplotlib` | 3.8 | Plotting (scatter plots, histograms, colorbars) |
| `seaborn` | 0.13 | Color palettes and styled statistical plots |
| `scikit-learn` | 1.4 | `StandardScaler`, `OneClassSVM`, `IsolationForest`, `LocalOutlierFactor`, `DecisionTreeClassifier`, `GridSearchCV`, cross-validation, metrics (F1, AUC-ROC, Jaccard, recall, precision) |
| `statsmodels` | 0.14 | Variance Inflation Factor (VIF) for multicollinearity detection; OLS linear regression for normality model validation |
| `imbalanced-learn` | 0.12 | `RandomOverSampler` / `RandomUnderSampler` for handling class imbalance in outlier/anomaly classifiers |
| `pyod` | 2.0 | Unsupervised outlier detection models: `IForest`, `KNN`, `LOF`, `OCSVM`, `PCA`, `KPCA`, `GMM`, `KDE`, `CBLOF`, `COF`, `HBOS`, `SOD`, `COPOD`, `ECOD`, `LODA`, `DeepSVDD` |
| `xgboost` | 2.0 | Gradient-boosted regression (`XGBRegressor`) and classification (`XGBClassifier`) for normality and outlier models |
| `lightgbm` | 4.3 | Alternative gradient-boosted regression (`LGBMRegressor`) |
| `shap` | 0.45 | SHAP explainability (`TreeExplainer`, summary plots) |
| `graphviz` | 0.20 | Exporting and rendering decision-tree graphs to PNG |
| `safeaipackage` | latest | S.A.F.E. AI risk metrics: `rga` (accuracy), `compute_rge_values` (explainability), `compute_rgr_values` (robustness), `compute_rga_parity` (fairness) |

---

## Standard-library modules (no install needed)

| Module | Usage |
|---|---|
| `sys` | Adding `attribute_name_files/` to the Python path |
| `time` | Measuring model training elapsed time |
| `warnings` | Suppressing non-critical warnings during model fitting |

---

## Local module

| Module | Location | Usage |
|---|---|---|
| `nokia_data_attributes` | `python/attribute_name_files/nokia_data_attributes.py` | Provides the `attributes_throughput` list used to select relevant columns from Nokia datasets |

---

## Scripts overview

| Script | Description |
|---|---|
| `python/collect_raw_data.py` | Loads raw Nokia CSV datasets, filters by test type and qualifier, and saves cleaned numerical/categorical CSVs to `data/raw_data/` |
| `python/outlier_detection_Subset1A.py` | Full outlier-detection pipeline: loads clean data, fits a normality model (XGB or LGB), detects KPI outliers and feature anomalies, compares 12 unsupervised detectors, and exports decision-tree explanations |
| `python/outlier_detection_comparative.py` | Runs the same outlier detection comparison across multiple dataset subsets (1A–2B) and produces radar charts for sensitivity, specificity, F1, Jaccard, and AUC-ROC |

## Notebooks overview

| Notebook | Description |
|---|---|
| `python/01-cleaning-small-tabular-data.ipynb` | Preprocessing pipeline: loads a CSV dataset, selects relevant features, removes constant/duplicate columns, handles missing values, removes multicollinear features via VIF, and detects/removes features correlated with the KPI using GMM-based clustering |
| `python/02-outlier-and-anomaly-detection-benchmark.ipynb` | Benchmark of outlier detectors on the KPI and feature space; builds a normality XGBoost model, detects anomalies in the residuals, and compares multiple PyOD unsupervised detectors |
| `python/03-outlier-and-anomaly-detection.ipynb` | Extended detection pipeline with grid-search optimisation for `IsolationForest`, `LOF`, and `OneClassSVM`; trains XGBoost outlier and anomaly classifiers with oversampling (`imbalanced-learn`) and explains them with SHAP |
