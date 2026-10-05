# Phase 13 — External validation

## Dataset and design

The external cohort is the Hungarian site file from the UCI Heart Disease collection. UCI describes four distinct collections (Cleveland, Hungary, Switzerland, and VA Long Beach) and 14 commonly used attributes. The Cleveland processed file is the source most consistent with the supplied dataset's 303-row schema; however, the supplied file's original provenance is not independently documented. UCI defines `num=0` as absence of angiographic disease and `num=1..4` as disease presence. This is a historical diagnostic endpoint, not acute myocardial infarction or prospective risk.

The frozen Phase 6 Logistic Regression baseline was trained on the existing primary training split, and the frozen Phase 4 preprocessing pipeline was fit only on that split. Neither was refit, recalibrated, nor tuned on Hungary. Predictions use the fixed 0.5 probability cutoff solely for descriptive class metrics. All 294 external rows with target labels were scored; no record-level predictions are retained.

## Mapping and provenance qualification

The UCI source stores `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, num`. The supplied CSV uses renamed fields and re-coded categories. Mapping of the categorical features was checked against the processed Cleveland-format cohort using the supplied dataset's overlapping row characteristics. The applied mappings are:

| UCI source | Project field | Mapping |
|---|---|---|
| `cp` | `cp` | 1→3, 2→1, 3→2, 4→0 |
| `restecg` | `restecg` | 0→1, 1→2, 2→0 |
| `slope` | `slp` | 1→2, 2→1, 3→0 |
| `ca` | `caa` | 0–3 unchanged; missing→4 source sentinel |
| `thal` | `thall` | 3→2, 6→1, 7→3; missing→0 source sentinel |
| `num` | `output` | 0→1, 1–4→0, inferred from matched Cleveland-format rows |

The label orientation is especially consequential: project class 1 aligns with UCI's angiographic *absence* label in matched rows. There is a one-row difference between the mapped full Cleveland target counts and the supplied CSV counts, so this alignment is a strong empirical mapping, not confirmed documentation of the supplied target's meaning. Metrics below are therefore class-coded and must not be described as heart-disease sensitivity without resolving target provenance.

## Results

The external file has 294 rows: UCI `num` values 0: 188 and 1–4 combined: 106. Source feature missingness is highly uneven: `slope` 190/294, `ca` 291/294, and `thal` 266/294 are missing; cholesterol is missing for 23/294. Missing `ca` and `thal` values were mapped to the supplied Cleveland-derived file's observed sentinels (4 and 0 respectively); remaining missing fields were passed through the saved primary-training imputer. The model consequently sees many imputed/sentinel values, and transport results are conditional on this mismatch in measurement availability.

| Metric | Observed value |
|---|---:|
| Accuracy at fixed 0.5 | 0.833 |
| Precision, project class 1 | 0.872 |
| Recall, project class 1 | 0.867 |
| Specificity, project class 0 | 0.774 |
| F1, project class 1 | 0.869 |
| ROC-AUC | 0.883 |
| Average precision | 0.909 |
| Brier score | 0.130 |
| Log loss | 0.416 |
| Confusion matrix (TN, FP, FN, TP; class 1 is the positive metric class) | 82, 24, 25, 163 |

These are one-site, unadjusted external estimates. No confidence interval, threshold selection, refitting, recalibration, or subgroup analysis was performed. The confusion-matrix class names are statistical labels; because project class 1 aligns to `num=0`, conventional disease-positive interpretation is inverted.

## Interpretation and limits

This run demonstrates that an independent UCI site file can be mapped and scored by the frozen pipeline. It does not establish that the model predicts heart attacks or prospective cardiovascular risk. The target is angiographic disease status. The site is historical, from the same UCI collection family, and its missingness is severe for three model fields. The original supplied CSV has undocumented source provenance and its output orientation requires confirmation from its provider. No clinical validity is established.

## Sources

- Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease*. UCI Machine Learning Repository. DOI: 10.24432/C52P4X. [Dataset page](https://archive.ics.uci.edu/dataset/45/heart%2Bdisease).
- UCI's historical [Heart Disease variable codebook](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart-disease.names) and [processed Hungarian site file](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data).
