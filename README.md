# Prediction of Heart Attacks Using Machine Learning

**Project identifier:** `heart-atrack-prediction-ml`

A research-oriented project using the user-provided `heart.csv` dataset. The work is being completed in phases; results and claims will be limited to experiments actually run. This project is not a clinical diagnostic tool.

## Current progress

- **Phase 1 — Dataset audit:** Complete. The supplied CSV contained 303 rows, 14 numeric columns, no parsed missing values, and one exact duplicate. `output` is the target candidate; its semantics remain to be confirmed from dataset provenance.
- **Phase 2 — Data cleaning:** Complete. The cleaning pipeline wrote a 302-row, 14-column dataset, removed one exact duplicate, found no rows with invalid values, and retained all IQR-flagged values for review.
- **Phase 3 and later:** Not started.

## Dataset files

- Original input: your local copy of `heart.csv` (not committed to GitHub)
- Place the raw copy at `data/raw/heart_disease.csv` before running the audit or cleaning pipeline (data files are excluded from Git)
- Cleaned data: `data/processed/cleaned_dataset.csv`

## Run Phase 2

From this project directory, run:

```powershell
python run_cleaning.py
python -m unittest tests.test_data_cleaning -v
```

The run writes `results/metrics/cleaning_report.json` and `results/tables/cleaning_report.md` alongside the cleaned CSV.

## Cleaning policy

The pipeline checks the discovered column schema, converts values to numeric, removes rows with missing labels, imputes missing predictors (mode for encoded code columns and median for continuous predictors), checks observed encoded domains and basic measurement constraints, removes exact duplicate records, and reports constant features. It investigates continuous-variable outliers with the 1.5×IQR rule but does not remove outliers on that basis alone.

The dataset's clinical provenance, label definition, and measurement timing have not yet been independently verified. No predictive performance or clinical validity is claimed.

