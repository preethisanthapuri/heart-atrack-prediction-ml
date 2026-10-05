# Prediction of Heart Attacks Using Machine Learning

**Project identifier:** `heart-atrack-prediction-ml`

A research-oriented project using the user-provided `heart.csv` dataset. Work proceeds in verified phases; results and claims are limited to experiments actually run. This is a research project, not a clinical diagnostic tool.

## Current progress

- **Phase 1 — Dataset audit:** Complete. The supplied CSV had 303 rows and 14 numeric columns, no parsed missing values, and one exact duplicate. `output` is the target candidate; its semantics remain unconfirmed from dataset provenance.
- **Phase 2 — Data cleaning:** Complete. Removed one exact duplicate, yielding 302 rows; no missing or invalid values were found and IQR-flagged measurements were retained.
- **Phase 3 — EDA and statistical analysis:** Complete. Generated five figures and four tables from the cleaned dataset. In exploratory univariate tests with Benjamini-Hochberg correction, 12 of 13 predictors were associated with the observed `output` class at q < 0.05; `fbs` was not (q=0.641). These associations do not imply causality, predictive performance, or clinical validity.
- **Phase 4 and later:** Not started.

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
```

`run_eda.py` reads only `data/processed/cleaned_dataset.csv`. It writes figures to `results/figures/eda/` and tables and the statistical interpretation to `results/tables/`.

## Phase 3 methods

The EDA includes target counts, feature distributions, numeric box plots by target, categorical count plots by target, and a Spearman correlation heatmap. Encoded category values are shown as ordinal in that heatmap for visualization; the categorical association tests instead use contingency tables and treat categories as labels.

Continuous features are compared between the two observed target classes using two-sided Mann-Whitney U tests with rank-biserial effect sizes. Encoded categorical features use Pearson chi-square, Fisher exact for sparse 2×2 tables, or a label-permutation Pearson test for sparse larger tables. Benjamini-Hochberg correction is applied across all 13 predictor tests. All tests are univariate and exploratory.

## Key files

- `src/data_loader.py` — read-only CSV loading
- `src/data_cleaning.py` — audit and cleaning utilities
- `src/eda.py` — descriptive tables and figures
- `src/statistical_analysis.py` — univariate tests and interpretation report
- `run_audit.py`, `run_cleaning.py`, `run_eda.py` — phase runners
- `tests/test_data_cleaning.py` — Phase 2 unit tests

## Limitations

The dataset is small and from a single source. Feature definitions, measurement timing, label semantics, and provenance require confirmation. Statistical association is not a causal claim and does not establish a useful prediction model. No external validation or clinical validation has been performed.
