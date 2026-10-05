# Viva guide

## Project in 30 seconds

This project evaluates common tabular classifiers on a user-supplied dataset whose source and target meaning are not verified. After removing one exact duplicate, 302 rows remained. The study uses leakage-aware training-fold preprocessing and cross-validation to compare baselines and staged extensions. Logistic Regression with fold-local mutual-information selection had the strongest pooled ROC-AUC, average precision, and Brier score in the staged ablation, while the all-feature Logistic Regression baseline had higher accuracy and F1. The dataset classes cannot currently be interpreted as heart attacks, confirmed disease, or future cardiovascular risk. The work is an exploratory benchmark, not a clinical model.

## Core evidence to remember

| Item | Recorded evidence | Qualification |
|---|---|---|
| Primary data | 303 supplied rows, 14 columns; one exact duplicate removed; 302 rows retained | File provenance and endpoint codebook remain unconfirmed |
| Target | `output`, binary codes 0 and 1; 138 and 164 cleaned observations | The class direction and medical meaning are not established |
| Split | 241 development rows and 61 holdout rows | Holdout accuracy was screened during baseline work, so it is not untouched |
| Baseline CV | Logistic Regression ROC-AUC 0.908 ± 0.039; accuracy 0.859 ± 0.045 | Five-fold stratified CV on the training partition; SD is descriptive |
| Selected ablation stage | Stage B: mutual-information SelectKBest, k=15, plus Logistic Regression; pooled OOF ROC-AUC 0.916, average precision 0.922, Brier 0.112 | Accuracy 0.846 and F1 0.861 were below Stage A: 0.859 and 0.872 |
| Integrated Stage F | Accuracy 0.834, ROC-AUC 0.885, Brier 0.126 | Did not improve overall evidence |
| External data | Frozen Phase 6 baseline on 294 historical UCI Hungary records: accuracy 0.833, ROC-AUC 0.883 | Severe missingness, likely reversed class orientation, one-record count discrepancy; not a Stage B validation |
| Explanation and uncertainty | SHAP was run for the Phase 6 Logistic Regression baseline; conformal analysis used soft voting | Neither is a Stage B explanation or uncertainty estimate |
| Testing | 38 tests passed in the recorded Phase 17 run | Evidence is recorded in `results/metrics/test_results.json`; this documentation task does not rerun tests |

## Likely questions and defensible answers

### 1. What is the research question?

Whether fold-local feature selection and staged model extensions improve internal out-of-fold metrics over an all-feature Logistic Regression baseline on this supplied dataset. The question is dataset-specific and does not test prospective heart-attack prediction.

### 2. Why does the title refer to heart attacks if the target is unresolved?

The requested project title is retained as a repository label. The scientific interpretation is deliberately narrower: `output` is treated as a dataset class. No claim is made that it represents a heart attack, incident event, or future risk. A publication title should be revised after provenance and endpoint verification.

### 3. What is the main contribution?

A reproducible, leakage-aware comparison and staged ablation with explicit reporting of metric tradeoffs and evidence boundaries. The contribution is methodological transparency for this supplied file, not a new algorithm.

### 4. Why select Stage B?

Stage B led the six-stage comparison on pooled ROC-AUC (0.916), average precision (0.922), and Brier score (0.112). This does not make it best on every objective: Stage A had higher accuracy and F1. Stage B is the probability-oriented research candidate chosen from those observed tradeoffs, not a clinically optimal model.

### 5. What does ROC-AUC measure here?

It measures ranking discrimination between the recorded numeric classes across score thresholds. Since the target direction is unverified, “positive” means computational class 1, not disease-positive.

### 6. Why report average precision and Brier score too?

Average precision summarizes the precision-recall tradeoff under the observed class mix. Brier score measures squared error of probabilistic predictions. They capture aspects ROC-AUC does not. Here, the Stage B Brier score is a training-only OOF estimate and does not establish calibrated clinical risk.

### 7. How did the project limit leakage?

Preprocessing and feature selection were fitted inside cross-validation training folds. Tuning and calibration steps were nested within outer training partitions in the corresponding experiments. Final Stage B selection and fitting used saved training indices only; the holdout was not used for final selection or fitting.

### 8. Is the holdout an independent final test set?

No. Phase 5 screened the same holdout for accuracy. Later phases did not use it for selection or fitting, but the earlier screening means it cannot be presented as an untouched final estimate.

### 9. Why is the external evaluation not strong validation?

It evaluated a frozen Phase 6 baseline, not final Stage B. The UCI Hungary data had extensive missingness, a crosswalk suggesting the project class orientation may be opposite to disease presence, and a one-record target-count discrepancy. It is historical site testing with unresolved mapping, not independent validation of a heart-attack-risk model.

### 10. Why not claim the ensemble improved the model?

The soft-voting model had some competitive metrics, but no consistent gain across measures. The integrated Stage F configuration had lower accuracy and ROC-AUC and a higher Brier score than Stage B. The experiment does not support a general ensemble-improvement claim.

### 11. What is calibration, and what did this study establish?

Calibration concerns agreement between predicted probabilities and observed outcome frequencies. It was compared for selected baseline/voting configurations. Those results do not transfer to Stage B, which was not specifically calibrated or evaluated for calibration.

### 12. What does the uncertainty analysis say?

Exploratory soft-voting conformal sets had 0.888 observed OOF coverage at nominal 90% and mean set size 1.108. Those results depend on exchangeability and concern soft voting only. They are not guarantees and do not characterize Stage B.

### 13. Are the SHAP features causal risk factors?

No. SHAP values describe how the fitted Phase 6 Logistic Regression model allocated its output among encoded features for the analyzed rows. They are not causal effects, clinical evidence, or explanations of Stage B.

### 14. Did the project establish fairness?

No. Subgroup metrics were descriptive, with sparse groups and unverified demographic codebooks and target orientation. No fairness certification or evidence of equitable clinical performance is claimed.

### 15. Why use conventional tabular models rather than deep learning?

The supplied data are small and tabular, with no image, ECG waveform, sensor stream, or time series. The evaluated conventional classifiers match the observed data structure. A deep architecture would not be justified by the supplied inputs.

### 16. What is the largest limitation?

The unresolved source provenance and target semantics prevent a defensible biomedical endpoint claim. Small sample size, a previously screened holdout, and lack of Stage B independent external validation further limit generalization.

### 17. Can the web app be used for patient care?

No. It is a research and educational demonstration. The app explicitly says it is not a medical diagnostic tool and reports probabilities for dataset classes, not validated cardiovascular risk.

### 18. What must happen before a clinical-risk study?

Define the intended population, predictor timing, incident endpoint and prediction horizon; verify source and label coding; collect an adequately sized representative cohort; lock a development protocol; validate independently across time or sites; evaluate calibration, subgroup performance, and clinical utility; and obtain applicable governance and ethics review.

## Author preparation checklist

- Replace the presentation author placeholder with the real author names and affiliations.
- Verify every reported number against the tracked experiment tables before the defense.
- Be prepared to explain class-code language, holdout screening, and why results from different model variants cannot be combined.
- Do not describe this as validated heart-attack prediction or clinical decision support.
- State the selected metric tradeoff plainly and accept that accuracy/F1 favored Stage A.
