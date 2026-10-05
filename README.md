# Prediction of Heart Attacks Using Machine Learning

**Project identifier:** `heart-atrack-prediction-ml`

A research-oriented project using the user-provided `heart.csv` dataset. Work proceeds in verified phases; results and claims are limited to experiments actually run. This is a research project, not a clinical diagnostic tool.

## Current progress

- **Phase 1 — Dataset audit:** Complete. The supplied CSV had 303 rows and 14 numeric columns, no parsed missing values, and one exact duplicate. `output` is the target candidate; its semantics remain unconfirmed from dataset provenance.
- **Phase 2 — Data cleaning:** Complete. Removed one exact duplicate, yielding 302 rows; no missing or invalid values were found and IQR-flagged measurements were retained.
- **Phase 3 — EDA and statistical analysis:** Complete. Generated five figures and four tables from the cleaned data. In exploratory univariate tests with Benjamini-Hochberg correction, 12 of 13 predictors were associated with the observed `output` class at q < 0.05; `fbs` was not (q=0.641). These associations do not imply causality, predictive performance, or clinical validity.
- **Phase 4 — Preprocessing:** Complete. A stratified 80/20 split produced 241 training and 61 test observations. Preprocessing was fitted on training rows only and transformed 13 inputs into 30 features.
- **Phase 5 — Baseline models:** Complete. Seven scikit-learn baselines were screened on the saved holdout split using accuracy only. Logistic Regression had the highest observed accuracy (**0.852; 52/61**) on this one split; this preliminary result does not establish a final model.
- **Phase 6 — Model evaluation and cross-validation:** Complete. Seven baseline classifiers were evaluated with accuracy, precision, class-1 recall (sensitivity), class-0 specificity, F1, ROC-AUC, and average precision (PR-AUC). Five-fold stratified cross-validation ran on the 241-row training partition, with preprocessing refit inside every fold. Logistic Regression had CV ROC-AUC 0.908 ± 0.039 and accuracy 0.859 ± 0.045; KNN had the highest mean CV accuracy (0.867 ± 0.064), while Logistic Regression had the highest mean ROC-AUC. The holdout set produced descriptive results only because it had already been screened for accuracy during Phase 5.
- **Phase 7 — Feature engineering and feature selection:** Complete. Compared all 30 one-hot/preprocessed features with four fold-local selection methods, each retaining 15 encoded features, for Logistic Regression and KNN. Mutual-information selection with Logistic Regression gave mean CV ROC-AUC 0.914 ± 0.040 and PR-AUC 0.926 ± 0.022, versus 0.908 ± 0.039 and 0.919 ± 0.027 using all features; its accuracy was lower (0.847 ± 0.060 vs 0.859 ± 0.045). The small metric differences do not establish a meaningful improvement. No manually derived clinical interaction or ratio features were added because feature definitions, units, and provenance are not verified.
- **Phase 8 — Hyperparameter optimization:** Complete. GridSearchCV tuned Logistic Regression (10 configurations) and RBF SVC (40 configurations); RandomizedSearchCV evaluated 12 KNN configurations. Searches used five-fold stratified CV with preprocessing fitted inside each fold and mean ROC-AUC as the predeclared refit criterion. The best search configurations had CV ROC-AUC estimates of 0.911 (Logistic Regression), 0.909 (SVC), and 0.912 (KNN). These are hyperparameter-selection scores and are likely optimistic; they are not independent performance estimates.
- **Phase 9 — Ensemble learning:** Complete. Evaluated soft voting (Logistic Regression, KNN, Random Forest) and stacking (Logistic Regression, KNN, RBF SVC) against individual baselines on identical training-only stratified folds. Soft voting had mean ROC-AUC 0.913 ± 0.045, compared with 0.908 ± 0.039 for Logistic Regression; its mean accuracy was 0.851, below KNN at 0.867. Stacking had ROC-AUC 0.909 ± 0.040 and accuracy 0.847. Evidence does not establish a broad or reliable ensemble improvement; soft voting remains an experimental comparator only.
- **Phase 10 — Probability calibration:** Complete. Compared uncalibrated probabilities with sigmoid and isotonic `CalibratedClassifierCV` using outer five-fold out-of-fold predictions and three-fold calibration internal to each outer training fold. Brier score and ROC-AUC were recorded, with calibration curves. Calibration did not help consistently: raw soft voting had the lowest Brier score (0.113), while sigmoid reduced Random Forest Brier from 0.123 to 0.120 and isotonic reduced Logistic Regression Brier from 0.117 to 0.116. The other model-method pairs were unchanged or worse. No single calibration method is retained as universally preferable.
- **Phase 11 — Uncertainty-aware prediction:** Complete. Evaluated split-conformal class sets from soft-voting probabilities with 90% and 80% nominal coverage, plus confidence-based risk-coverage. At 90% nominal coverage, observed out-of-fold coverage was 0.888, mean set size 1.108, and 10.8% of predictions had both labels in their set; singleton accuracy was 0.875 versus 0.851 argmax accuracy overall. At 80%, coverage was 0.817 and 4.5% of prediction sets were empty. Confidence-ranked error was 0.074 among the most confident 50% versus 0.149 over all rows, with non-monotonic intermediate points. These estimates are exploratory and depend on exchangeability; no clinical threshold is used.
- **Phase 12 — SHAP explainability:** Complete. `shap.LinearExplainer` explained the saved Phase 6 Logistic Regression model on its 241 training rows using a deterministic 100-row training background and the saved training-fitted preprocessor. Global mean absolute SHAP values ranked `cp_0`, `caa_0`, `thall_2`, `chol`, and `oldpeak` highest for this fitted model (log-odds scale). The selected training row (dataset index 110) had class-1 probability 0.649; its largest absolute contributions were `chol` (-0.880 log-odds), `trtbps` (-0.729), and `cp_0` (-0.684). Unit tests confirmed SHAP additivity to the model decision function. These are model-output associations, not causal or clinical effects; explanations were generated on training rows and do not validate generalization.
- **Phase 13 — External validation:** Complete as an exploratory site-level test on the UCI Hungarian cohort (294 rows), using the frozen Phase 6 Logistic Regression model and Phase 4 preprocessing. On project class labels, accuracy was 0.833, ROC-AUC 0.883, average precision 0.909, and Brier score 0.130 at the fixed 0.5 cutoff. The cohort has extensive missingness (slope 190/294, ca 291/294, thal 266/294), with primary-compatible missing sentinels and the saved training-fitted imputer applied. Code inspection and a Cleveland-format feature comparison indicate `output=1` aligns with UCI `num=0` (absence of angiographic disease), not disease presence; the class counts differ by one record and the original supplied-file provenance remains unconfirmed. These metrics are therefore reported only for dataset codes and do not establish heart-attack/risk prediction or clinical validity.
- **Phase 14 — Ablation and subgroup analysis:** Complete. Compared six staged configurations using five-fold stratified out-of-fold predictions on the 241-row training partition; the saved 61-row holdout was excluded. Logistic Regression with 15 fold-local mutual-information features had the highest pooled ROC-AUC (0.916) and average precision (0.922). The integrated tuned/selected/calibrated soft vote had lower accuracy (0.834), ROC-AUC (0.885), and higher Brier score (0.126) than simpler candidates. Subgroup metrics were descriptive; sparse groups and unresolved label/codebook prevent fairness conclusions. See `paper/ablation_fairness.md`.
- **Phase 15 — Final model:** Complete. Selected Stage B (mutual-information SelectKBest, k=15, plus Logistic Regression) based on its best Phase 14 pooled training-only ROC-AUC (0.916), average precision (0.922), and Brier score (0.112), while documenting its lower accuracy/F1 than the all-feature baseline. The raw-input pipeline was refit on all 241 saved training rows; the 61-row holdout was not used. No post-hoc calibration or Stage-B uncertainty method is claimed. See `paper/final_model_selection.md`.
- **Phase 16 — Web application:** Complete. Added a Flask research interface backed by the saved Stage B raw-input pipeline. It validates all 13 fields and category codes, warns when numeric inputs are outside the observed training range, displays dataset-class probabilities and local Logistic Regression contributions, and explicitly reports that Stage-B uncertainty is not estimated. This is not a medical diagnostic tool. See `app/`.
- **Phase 17 — Testing and final results:** Complete. The complete suite passed (38 tests); the Flask API was smoke-tested against the saved model. The final-results runner reconciles recorded experiment tables with final-model metadata and writes a consolidated evidence CSV, table catalog, publication-ready baseline/ablation figures, machine-readable summary, and `paper/final_results.md`. No model fitting was repeated for this synthesis. All reported evidence is qualified by split, class-code, and provenance limitations.
- **Phase 18 — Literature review and research gap:** Complete. Added a focused, explicitly non-systematic scholarly review and a bounded gap/claim assessment in paper/literature_review.md and paper/research_gap.md. The review documents why the supplied 0/1 target cannot yet be asserted to mean heart disease or heart attack, distinguishes this diagnostic-style benchmark from prospective risk prediction, and finds no support for algorithmic novelty or clinical validity. The defensible contribution is a reproducible, leakage-aware empirical comparison with transparent tradeoffs and limitations. These files do not resolve dataset provenance or constitute a systematic review.

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
python run_model_evaluation.py
python run_feature_selection.py
python run_tuning.py
python run_ensemble.py
python run_calibration.py
python run_uncertainty.py
python -m unittest tests.test_phase14 -v
python run_ablation.py
python -m unittest tests.test_final_model -v
python run_final_model.py
python -m unittest tests.test_app -v
python run_tests.py
python run_final_results.py
python app/app.py
```

`run_eda.py` reads `data/processed/cleaned_dataset.csv` and writes figures to `results/figures/eda/` and tables/reports to `results/tables/`.

`run_preprocessing.py` recreates the stratified split and fits transformations on training rows only. It writes `models/preprocessing_pipeline.joblib` and `results/metrics/preprocessing_split.json`.

`run_baselines.py` loads those Phase 4 artifacts, fits the baseline classifiers on the training partition, saves fitted classifiers under `models/baselines/`, and writes the aggregate holdout accuracy screen to `results/tables/baseline_screening.csv`. `run_model_evaluation.py` writes `results/tables/model_comparison.csv`, `results/tables/cross_validation.csv`, and nine confusion-matrix, ROC, and precision-recall figures. Cross-validation uses only the training partition and fits preprocessing within each fold. Holdout figures and metrics are descriptive because Phase 5 already screened this same split for accuracy; no test-set tuning was performed and no final model is selected in Phase 6.

`run_feature_selection.py` compares all one-hot-expanded inputs with 15-feature selections using mutual information, ANOVA SelectKBest, Logistic Regression RFE, and an Extra Trees model-based selector. It evaluates both Logistic Regression and KNN using the Phase 6 stratified folds on training data only. Preprocessing and feature selection are refitted inside every fold. It saves CV metrics to `results/tables/feature_selection_comparison.csv` and selector rank/fold-stability information to `results/tables/feature_selection.csv`. The holdout set is not used by this experiment.

`run_tuning.py` searches training rows only. Grid search covers Logistic Regression and RBF SVC; randomized search samples 12 KNN configurations. The pipelines fit preprocessing within each CV training fold. Mean ROC-AUC is the predeclared refit criterion; accuracy, F1, and average precision are also recorded. It saves each search candidate to `results/tables/tuning_results.csv` and best settings to `results/tables/tuning_summary.csv`. The reported best CV scores are selected from multiple candidates on these folds and must not be interpreted as unbiased tuned-model performance. The holdout is not loaded or scored.

`run_ensemble.py` compares soft voting (Logistic Regression, KNN, Random Forest) and stacking (Logistic Regression, KNN, RBF SVC) with the constituent individual models and Random Forest. It evaluates all candidates on the same training-only stratified folds, fits preprocessing within each fold, and uses stacking's internal CV only inside the outer training fold. It saves summary and per-fold metric tables to `results/tables/ensemble_comparison.csv` and `results/tables/ensemble_folds.csv`. No holdout records are used.

`run_calibration.py` compares native uncalibrated probabilities against sigmoid and isotonic `CalibratedClassifierCV` for Logistic Regression, KNN, Random Forest, and soft voting. It obtains out-of-fold predictions using five outer stratified folds, with three-fold calibration contained in each outer training fold. It writes Brier score and ROC-AUC summaries to `results/tables/calibration.csv`, binned calibration-curve points to `results/tables/calibration_curve.csv`, and `results/figures/calibration/calibration_curves.png`. It does not use the holdout or choose a threshold.

`run_uncertainty.py` uses the Phase 9 soft-voting model within five outer training folds. Each outer training fold is divided into model-fit and conformal-calibration subsets; the outer validation fold remains untouched. It creates split-conformal sets using the nonconformity score `1 - p(true class)` at alpha 0.10 and 0.20, summarizes empirical coverage, set size, singleton/ambiguous/empty set rates, and selective accuracy, and measures a confidence-ranked risk-coverage curve. It writes summaries and derived out-of-fold scores under `results/tables/uncertainty*.csv`; patient-level inputs are not copied into these tables. These statistical coverage estimates assume exchangeability and are not clinical guarantees.

`run_explainability.py` uses the saved Phase 6 Logistic Regression baseline and Phase 4 preprocessor to compute SHAP LinearExplainer values in log-odds space. It explains training partition rows only, with a reproducible background sample of up to 100 training rows, and saves global/local contribution tables, a local prediction metadata JSON, a beeswarm summary, a global bar plot, and a local waterfall under `results/`. SHAP contributions explain the fitted model's outputs; they are neither causal effects nor evidence of clinical validity.

`run_external_validation.py` applies the frozen Phase 6 model to the local UCI Hungarian site file in `data/external/uci_heart_disease_hungarian.data`. It does not refit, recalibrate, or tune the model. The source code crosswalk maps the UCI feature and target encodings into the supplied dataset's apparent Cleveland-derived coding; `paper/external_validation.md` documents the mappings and the one-record target-count discrepancy. It saves aggregate metrics and a source-data SHA-256, not row-level predictions. The external cohort has severe missingness in slope, ca, and thal; missing ca/thal sentinels match the supplied file, while remaining missing values use the saved primary-training imputer. This is historical same-family site testing with uncertain primary provenance, not clinical validation.

`run_ablation.py` compares the Logistic Regression baseline, mutual-information selection, inner-fold tuning, Phase 9 soft voting, calibrated voting, and an integrated tuned/selected/calibrated vote. Five outer folds generate OOF predictions using only saved training indices. Preprocessing, feature selection, searches, and calibration are fitted within outer training folds; Stage F reserves a disjoint calibration subset. Tables are written to `results/tables/ablation*.csv` and `results/tables/fairness*.csv`. Stage F did not improve overall performance, so this experiment does not support the full combination as superior. Subgroup output uses dataset class codes and descriptive age bins, not a clinical fairness claim. See `paper/ablation_fairness.md`.

`run_final_model.py` selects Stage B using the recorded Phase 14 training-only OOF comparison and refits the complete preprocessing, mutual-information selector, and Logistic Regression pipeline on the saved 241-row training partition. It does not read or score the saved 61-row holdout. The raw-schema prediction pipeline is saved as `models/final_model.joblib` and its fitted selector as `models/feature_selector.joblib`; evidence and selected feature names are recorded in `results/metrics/final_model.json`. The final model returns dataset-class probabilities without selecting a medical threshold. Calibration and uncertainty claims from other model configurations are not transferred to this selected model. See `paper/final_model_selection.md`.

`python app/app.py` starts the local Flask app at `http://127.0.0.1:5000/`. Generate the final model first with `python run_final_model.py`. The form requires all 13 fields, validates values against the saved training schema, and flags numeric values beyond the observed training range. The API returns probabilities for numeric dataset classes and model-local Logistic Regression contributions in log-odds units; these contributions are not causal. Uncertainty is reported as unavailable for this final model. The app stores no submitted values and binds to localhost by default. It is for research and educational decision-support purposes, not diagnosis.

`run_tests.py` executes every `unittest` test under `tests/` and records case IDs, pass/fail counts, Python version, and run time in `results/metrics/test_results.json`. `run_final_results.py` checks that required Phase 1–16 results exist and that final-model metadata agrees with the Phase 14 ablation; it requires a clean saved test summary. It writes `paper/final_results.md`, a tidy caveat-annotated metric catalog, consolidated Stage A–F results, table/figure catalogs, `results/metrics/final_results.json`, and final-results charts in PNG/PDF formats. It only assembles previously recorded experiment outputs and does not retrain models.

The CSV dataset, fitted model/pipeline artifacts, split metadata, and generated EDA results are excluded from GitHub.

## Phase 3 methods

The EDA includes target counts, feature distributions, numeric box plots by target, categorical count plots by target, and a Spearman correlation heatmap. Encoded category values are shown as ordinal in that heatmap for visualization; categorical association tests treat categories as labels.

Continuous features are compared between observed target classes using two-sided Mann-Whitney U tests with rank-biserial effect sizes. Encoded categorical features use Pearson chi-square, Fisher exact for sparse 2×2 tables, or a label-permutation Pearson test for sparse larger tables. Benjamini-Hochberg correction is applied across all 13 predictor tests. All tests are univariate and exploratory.

## Phase 4 preprocessing design

The numerical measurements (`age`, `trtbps`, `chol`, `thalachh`, `oldpeak`) use median imputation followed by standard scaling. Encoded categorical fields (`sex`, `cp`, `fbs`, `restecg`, `exng`, `slp`, `caa`, `thall`) use most-frequent imputation followed by one-hot encoding with unknown categories ignored. A scikit-learn `Pipeline` and `ColumnTransformer` keep preprocessing reusable. The held-out split is stratified, and no fitted preprocessing statistics use test rows.

## Phase 5 baseline models

The screen compares Logistic Regression, K-Nearest Neighbors, Decision Tree, RBF Support Vector Classifier, Random Forest, Extra Trees, and Gradient Boosting. All models use the same training-fitted preprocessing and the same holdout rows. XGBoost was not added because the current small dataset and seven included scikit-learn baselines do not justify an additional package at this stage.

## Phase 6 evaluation design and observed results

Metrics treat the numeric target value `1` as the positive class and `0` as the negative class; the clinical meaning of those values is not verified. Specificity is the true-negative rate for class 0. PR-AUC is reported as average precision. Five-fold CV reports the unweighted mean and sample standard deviation across folds. On this run, Logistic Regression had CV ROC-AUC 0.908 ± 0.039; KNN had mean CV accuracy 0.867 ± 0.064. On the previously screened holdout, Logistic Regression had accuracy 0.852, recall 0.879, specificity 0.821, and ROC-AUC 0.897. These small-sample estimates are exploratory and do not establish clinical performance or model superiority.

## Phase 7 feature selection design and observed results

The input schema expands to 30 features after one-hot encoding. A fixed budget of 15 encoded features was applied to mutual-information SelectKBest, ANOVA SelectKBest, Logistic Regression RFE, and Extra Trees SelectFromModel. Each selector was fit only on its current training fold; the same five stratified folds compare the full feature set and selected sets. The strongest selected Logistic Regression ROC-AUC was 0.914 ± 0.040 (mutual information), versus 0.908 ± 0.039 using all 30 features; selected-set accuracy was 0.847 ± 0.060 versus 0.859 ± 0.045. For KNN, using all features had the highest mean CV accuracy (0.867 ± 0.064). These are exploratory estimates from 241 training rows; no selector is established as superior. Feature rankings and fold-selection rates are not evidence of causality or clinical importance.

## Phase 8 tuning design and observed search results

Model search was limited to Logistic Regression and RBF SVC (among the strongest CV ROC-AUC results in Phase 6) and KNN (highest Phase 6 mean CV accuracy). Searches use five-fold stratification and a pipeline that fits preprocessing within each fold. GridSearchCV evaluated 10 Logistic Regression and 40 SVC configurations; RandomizedSearchCV sampled 12 KNN configurations. Best mean CV ROC-AUC settings were Logistic Regression: 0.911 (C=0.1, class_weight=balanced); SVC: 0.909 (C=100, gamma=0.001, no class weighting); KNN: 0.912 (n_neighbors=9, Manhattan distance, distance weighting). Since each score is the maximum selected from a search over candidates, these values are optimization criteria with selection optimism, not independent performance estimates. No final estimator is designated in this phase.

## Phase 9 ensemble design and observed results

Using the same five stratified training folds as Phase 6, soft voting (Logistic Regression, KNN, and Random Forest) produced accuracy 0.851 ± 0.053, ROC-AUC 0.913 ± 0.045, and average precision 0.919 ± 0.038. Logistic Regression alone produced 0.859 ± 0.045, 0.908 ± 0.039, and 0.919 ± 0.027, respectively. Stacking (Logistic Regression, KNN, and RBF SVC) produced accuracy 0.847 ± 0.060 and ROC-AUC 0.909 ± 0.040. KNN had the highest mean accuracy at 0.867 ± 0.064. Thus soft voting's small mean ROC-AUC increase comes with lower accuracy and overlapping fold variability; stacking did not show an overall advantage. These results do not establish statistical significance or generalization. No ensemble is selected as the final model.

## Phase 10 calibration design and observed results

The calibration experiment generated outer-fold predictions on the 241 training rows; sigmoid/isotonic calibrators were fit only within each outer training fold using three-fold CV. Brier score is lower-is-better; ROC-AUC tracks ranking and is included to monitor discrimination changes. Mean fold Brier / ROC-AUC results were: Logistic Regression uncalibrated 0.117 / 0.908, sigmoid 0.117 / 0.908, isotonic 0.116 / 0.906; KNN uncalibrated 0.122 / 0.897, sigmoid 0.126 / 0.899, isotonic 0.127 / 0.894; Random Forest uncalibrated 0.123 / 0.902, sigmoid 0.120 / 0.899, isotonic 0.122 / 0.894; soft voting uncalibrated 0.113 / 0.913, sigmoid 0.115 / 0.908, isotonic 0.114 / 0.904. The small sample and fold variability do not establish a reliable method winner. Calibration curves use fixed-width probability bins; sparse bins make their shape noisy. No clinical threshold is applied.

## Phase 11 uncertainty design and observed results

The soft-voting candidate was fit on each outer fold's model-fit subset, while a disjoint calibration subset determined the finite-sample split-conformal quantile; the outer validation fold was untouched until scoring. At alpha 0.10 (nominal coverage 0.90), pooled out-of-fold empirical coverage was 0.888, mean set size 1.108, singleton rate 0.892, ambiguous two-label rate 0.108, empty-set rate 0, and singleton accuracy 0.875. At alpha 0.20 (nominal 0.80), pooled coverage was 0.817, mean set size 0.955, singleton rate 0.955, empty-set rate 0.045, and singleton accuracy 0.855. Argmax accuracy overall was 0.851. Confidence-based selective error was 0.074 among the top-confidence 50% of rows and 0.149 when retaining all rows; intermediate coverage points were not strictly monotonic. These are small-sample estimates from one dataset and depend on exchangeability; coverage is not a clinical guarantee. Non-singleton/empty sets indicate that the model abstains from a single-label prediction, not a medical referral decision.

## Key files

- `src/data_loader.py` — read-only CSV loading
- `src/data_cleaning.py` — audit and cleaning utilities
- `src/eda.py` — descriptive tables and figures
- `src/statistical_analysis.py` — univariate tests and interpretation report
- `src/preprocessing.py` — split and reusable leakage-safe preprocessing pipeline
- `src/models.py` — baseline classifier definitions
- `src/evaluation.py` — baseline screening, holdout metrics, cross-validation, and plots
- `src/feature_selection.py` — fold-local feature-selection experiment and stability summaries
- `src/tuning.py` — training-only grid and randomized hyperparameter searches
- `src/ensemble.py` — fold-local soft voting and stacking comparisons
- `src/calibration.py` — nested out-of-fold probability calibration and reliability-curve evaluation
- `src/uncertainty.py` — split-conformal prediction sets and confidence-based risk-coverage
- `src/explainability.py` — training-partition SHAP global/local explanations for Logistic Regression
- `src/external_validation.py` — external-source schema/category mapping and frozen-model scoring
- `src/ablation.py`, `src/fairness.py` — staged training-fold ablation and descriptive subgroup metrics
- `src/final_model.py` — evidence-based candidate selection and training-only final refit
- `src/final_results.py` — result integrity checks, aggregate tables, plots, and final report
- `run_tests.py`, `run_final_results.py` — complete project test run and final-results assembly
- `paper/final_results.md` — consolidated actual results and evaluation limitations
- `app/app.py`, `app/templates/index.html`, `app/static/` — local Flask form, validation API, and interface assets
- `tests/test_app.py` — Flask page, prediction, and invalid-input checks
- `paper/ablation_fairness.md` — Phase 14 design, observed findings, and limitations
- `paper/final_model_selection.md` — Phase 15 selection evidence, fit boundary, and limitations
- `run_audit.py`, `run_cleaning.py`, `run_eda.py`, `run_preprocessing.py`, `run_baselines.py`, `run_model_evaluation.py`, `run_feature_selection.py`, `run_tuning.py`, `run_ensemble.py`, `run_calibration.py`, `run_uncertainty.py`, `run_explainability.py`, `run_external_validation.py` — phase runners
- `paper/external_validation.md` — Phase 13 cohort, crosswalk, results, and limitations
- `tests/` — Phase 2, 4–14 tests

## Limitations

The dataset is small and from a single source. Feature definitions, measurement timing, label semantics, and provenance require confirmation. Statistical association, internal cross-validation, a holdout accuracy screen, and an exploratory historical external-site score do not establish a causal relationship, robust generalization, or clinical validity. No prospective or clinical validation has been performed.
