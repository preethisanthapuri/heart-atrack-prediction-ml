# Prediction of Heart Attacks Using Machine Learning

**Project identifier:** `heart-atrack-prediction-ml`

A research-oriented project using the user-provided `heart.csv` dataset. Work proceeds in verified phases; results and claims are limited to experiments actually run. This is a research project, not a clinical diagnostic tool.

## Current progress

- **Phase 1 — Dataset audit:** Complete. The supplied CSV had 303 rows and 14 numeric columns, no parsed missing values, and one exact duplicate. `output` is the target candidate; its semantics remain unconfirmed from dataset provenance.
- **Phase 2 — Data cleaning:** Complete. Removed one exact duplicate, yielding 302 rows; no missing or invalid values were found and IQR-flagged measurements were retained.
- **Phase 3 — EDA and statistical analysis:** Complete. Generated five figures and four tables from the cleaned data. In exploratory univariate tests with Benjamini-Hochberg correction, 12 of 13 predictors were associated with the observed `output` class at q < 0.05; `fbs` was not (q=0.641). These associations do not imply causality, predictive performance, or clinical validity.
- **Phase 4 — Preprocessing:** Complete. A stratified 80/20 split produced 241 training and 61 test observations. Preprocessing was fitted on training rows only and transformed 13 inputs into 30 features.
- **Phase 5 — Baseline models:** Complete. Seven scikit-learn baselines were screened on the saved holdout split using accuracy only. Logistic Regression had the highest observed accuracy (**0.852; 52/61**) on this one split; this preliminary result does not establish a final model.
- **Phase 6 and later:** Not started.

## Dataset

Place the provided CSV at `data/raw/heart_disease.csv`. The raw and processed patient-level data are intentionally excluded from GitHub. Keep the original source file unchanged.

Phase 2 creates `data/processed/cleaned_dataset.csv`. Its observed binary target candidate is `output` (classes 0 and 1). The label meaning and dataset provenance have not been independently verified.

## Setup and run

Use Python 3.10 or newer. From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_audit.py
python run_cleaning.py
python -m unittest tests.test_data_cleaning -v
python run_eda.py
python run_preprocessing.py
python -m unittest tests.test_preprocessing -v
python run_baselines.py
python -m unittest tests.test_models -v
```

`run_eda.py` reads `data/processed/cleaned_dataset.csv` and writes figures to `results/figures/eda/` and tables/reports to `results/tables/`.

`run_preprocessing.py` recreates the stratified split and fits transformations on training rows only. It writes `models/preprocessing_pipeline.joblib` and `results/metrics/preprocessing_split.json`.

`run_baselines.py` loads those Phase 4 artifacts, fits the baseline classifiers on the training partition, saves fitted classifiers under `models/baselines/`, and writes the aggregate holdout accuracy screen to `results/tables/baseline_screening.csv`. Phase 6 will provide the full metric suite, cross-validation, and diagnostic plots. The current accuracy ranking is preliminary and is not used to select a final model.

The CSV dataset, fitted model/pipeline artifacts, split metadata, and generated EDA results are excluded from GitHub.

## Phase 3 methods

The EDA includes target counts, feature distributions, numeric box plots by target, categorical count plots by target, and a Spearman correlation heatmap. Encoded category values are shown as ordinal in that heatmap for visualization; categorical association tests treat categories as labels.

Continuous features are compared between observed target classes using two-sided Mann-Whitney U tests with rank-biserial effect sizes. Encoded categorical features use Pearson chi-square, Fisher exact for sparse 2×2 tables, or a label-permutation Pearson test for sparse larger tables. Benjamini-Hochberg correction is applied across all 13 predictor tests. All tests are univariate and exploratory.

## Phase 4 preprocessing design

The numerical measurements (`age`, `trtbps`, `chol`, `thalachh`, `oldpeak`) use median imputation followed by standard scaling. Encoded categorical fields (`sex`, `cp`, `fbs`, `restecg`, `exng`, `slp`, `caa`, `thall`) use most-frequent imputation followed by one-hot encoding with unknown categories ignored. A scikit-learn `Pipeline` and `ColumnTransformer` keep preprocessing reusable. The held-out split is stratified, and no fitted preprocessing statistics use test rows.

## Phase 5 baseline models

The screen compares Logistic Regression, K-Nearest Neighbors, Decision Tree, RBF Support Vector Classifier, Random Forest, Extra Trees, and Gradient Boosting. All models use the same training-fitted preprocessing and the same holdout rows. XGBoost was not added because the current small dataset and seven included scikit-learn baselines do not justify an additional package at this stage.

## Key files

- `src/data_loader.py` — read-only CSV loading
- `src/data_cleaning.py` — audit and cleaning utilities
- `src/eda.py` — descriptive tables and figures
- `src/statistical_analysis.py` — univariate tests and interpretation report
- `src/preprocessing.py` — split and reusable leakage-safe preprocessing pipeline
- `src/models.py` — baseline classifier definitions
- `src/evaluation.py` — holdout baseline screening
- `run_audit.py`, `run_cleaning.py`, `run_eda.py`, `run_preprocessing.py`, `run_baselines.py` — phase runners
- `tests/` — Phase 2, 4, and 5 tests

## Limitations

The dataset is small and from a single source. Feature definitions, measurement timing, label semantics, and provenance require confirmation. Statistical association and one holdout accuracy screen do not establish a causal relationship, robust generalization, or clinical validity. No external validation or clinical validation has been performed.
