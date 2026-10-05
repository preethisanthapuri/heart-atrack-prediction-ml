# Phase 17 — Testing and final results

## Verification

The project test suite completed with 38 passing tests out of 38 discovered, with 0 failures and 0 errors. Detailed test case IDs and the run environment are in `results/metrics/test_results.json`. The Flask API was smoke-tested against the saved Stage B model and returned HTTP 200 with both class probabilities and local contributions. Python modules compiled successfully. Final-results tables and figures were assembled from previously executed experiment artifacts; this consolidation did not refit models or recalculate patient-level predictions.

## Dataset and evaluation boundaries

The provided primary CSV contained 303 raw rows and 14 columns. Cleaning removed 1 exact duplicate, leaving 302 rows. The target is `output` with numeric dataset classes 0 and 1; label semantics and provenance remain unverified. The saved stratified split has 241 training rows and 61 holdout rows.

The model comparison below reports training-only five-fold cross-validation mean ± sample SD. The holdout was previously screened for accuracy in Phase 5, so its metrics are only descriptive and are not an untouched final estimate. Phases 7–16 selection, ablation, and final model fitting used training indices only. The historical Hungary result is separate from internal CV and has substantial feature missingness and unresolved primary-target orientation.

## Baseline cross-validation

| Model | Accuracy, mean ± SD | ROC-AUC, mean ± SD | Average precision, mean ± SD |
|---|---:|---:|---:|
| Logistic Regression | 0.859 ± 0.045 | 0.908 ± 0.039 | 0.919 ± 0.027 |
| Support Vector Classifier (RBF) | 0.834 ± 0.062 | 0.907 ± 0.044 | 0.908 ± 0.031 |
| Random Forest | 0.822 ± 0.080 | 0.902 ± 0.064 | 0.909 ± 0.056 |
| K-Nearest Neighbors | 0.867 ± 0.064 | 0.897 ± 0.062 | 0.870 ± 0.076 |
| Extra Trees | 0.838 ± 0.060 | 0.895 ± 0.058 | 0.898 ± 0.058 |
| Gradient Boosting | 0.817 ± 0.107 | 0.880 ± 0.064 | 0.888 ± 0.062 |
| Decision Tree | 0.751 ± 0.148 | 0.751 ± 0.141 | 0.731 ± 0.111 |

## Staged ablation

Pooled out-of-fold metrics for stages A–F are shown below. Outer-fold mean and standard deviation are shown in `results/tables/final_ablation_summary.csv`; fold SD is descriptive, not a confidence interval.

| Stage | Accuracy | Sensitivity | Specificity | F1 | ROC-AUC | Average precision | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0.859 | 0.885 | 0.827 | 0.872 | 0.910 | 0.915 | 0.118 |
| B (selected) | 0.846 | 0.878 | 0.809 | 0.861 | 0.916 | 0.922 | 0.112 |
| C | 0.842 | 0.878 | 0.800 | 0.858 | 0.905 | 0.907 | 0.118 |
| D | 0.851 | 0.878 | 0.818 | 0.865 | 0.913 | 0.913 | 0.113 |
| E | 0.855 | 0.893 | 0.809 | 0.870 | 0.911 | 0.909 | 0.114 |
| F | 0.834 | 0.901 | 0.755 | 0.855 | 0.885 | 0.871 | 0.126 |

Stage B (mutual-information selection plus Logistic Regression) was selected as the probability-oriented research candidate. It has the strongest pooled ROC-AUC (0.916), average precision (0.922), and Brier score (0.112) among the staged candidates. Stage A had higher accuracy (0.859) and F1 (0.872). Stage F did not improve the overall result. This is a metric tradeoff from a small internal sample, not proof of superiority.

## Calibration and uncertainty scope

| Soft voting probability method | Brier, mean ± SD | ROC-AUC, mean ± SD |
|---|---:|---:|
| uncalibrated | 0.113 ± 0.034 | 0.913 ± 0.045 |
| sigmoid | 0.115 ± 0.033 | 0.908 ± 0.050 |
| isotonic | 0.114 ± 0.035 | 0.904 ± 0.054 |

The calibration experiment concerned its listed baseline/voting candidates; a calibrated Stage B model was not directly evaluated. No post-hoc calibrator is attached to the final artifact.

| Nominal coverage | Empirical OOF coverage | Mean set size | Ambiguous rate | Empty rate |
|---:|---:|---:|---:|---:|
| 0.90 | 0.888 | 1.108 | 0.108 | 0.000 |
| 0.80 | 0.817 | 0.955 | 0.000 | 0.045 |

These Phase 11 split-conformal results are for soft voting, not Stage B, and are not a guarantee of future or clinical coverage.

## External site and subgroup results

The frozen Phase 6 baseline scored 294 historical UCI Hungary records: accuracy 0.833, ROC-AUC 0.883, average precision 0.909, and Brier 0.130. Project class 1 appears aligned to UCI angiographic absence, with a one-record count discrepancy and unconfirmed source mapping. Reported values are dataset-code metrics, not disease-positive performance.

The following Stage F subgroup estimates are descriptive and have no uncertainty intervals or fairness-inference interpretation:

| Grouping | Code/bin | n | Class-1 recall | Class-0 specificity | Accuracy |
|---|---|---:|---:|---:|---:|
| sex_code | 0 | 76 | 0.964 | 0.714 | 0.895 |
| sex_code | 1 | 165 | 0.855 | 0.764 | 0.806 |
| age_group | <50 | 71 | 0.959 | 0.636 | 0.859 |
| age_group | 50-59 | 98 | 0.863 | 0.766 | 0.816 |
| age_group | 60-69 | 63 | 0.840 | 0.816 | 0.825 |
| age_group | 70+ | 9 | 1.000 | 0.667 | 0.889 |

Small group sizes include n=9 for the 70+ bin. Target direction, subgroup codebook, and provenance are not verified; these results do not establish fairness or discriminatory impact.

## Final artifacts

- `results/tables/final_ablation_summary.csv` — staged metrics and fold summaries.
- `results/tables/final_results_evidence.csv` — tidy multi-phase metrics annotated with evaluation scope and caveats.
- `results/tables/final_results_catalog.csv` — final catalog of available phase tables.
- `results/tables/final_figure_catalog.csv` — inventory of generated phase figures (formats and file sizes).
- `results/figures/final_results/ablation_tradeoffs.png` and `.pdf` — accuracy, ROC-AUC, Brier comparison with outer-fold variability.
- `results/figures/final_results/baseline_cv_roc_auc.png` and `.pdf` — baseline training CV ROC-AUC with fold variability.

The figure catalog contains 22 generated figures across EDA, model evaluation, calibration, SHAP, and final results.

The full tuning, feature-selection, calibration, uncertainty, external, ablation, and subgroup tables remain under `results/tables/`. Patient-level inputs and fitted model binaries are local-only/ignored by Git.

## Limitations

The project dataset is small. Target meaning and source provenance remain uncertain, the holdout was screened during baseline work, and the external cohort is historical with severe missingness. Calibration and uncertainty evidence are not specific to the selected Stage B model. Model explanations describe associations in model outputs and do not imply causality. No prospective clinical validation or clinical utility study has been performed. The application remains for research and education, not diagnosis.
