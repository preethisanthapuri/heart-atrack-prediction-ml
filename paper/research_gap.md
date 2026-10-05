# Phase 18 — Research gap and defensible contribution

## Research question represented by the completed experiments

Within the 302-row cleaned user-provided dataset, do fold-local mutual-information feature selection and/or the staged extensions tested in Phases 7–14 improve training-only out-of-fold discrimination and probability-error metrics over an all-feature Logistic Regression baseline?

This question is dataset-bound. It is not a clinical question about future heart attack events [1, 2, 4].

## What the literature establishes

1. The 14-column, 303-record structure resembles the widely used UCI Cleveland Heart Disease benchmark, but resemblance does not identify the supplied file. UCI documents a multicentre source collection and a num outcome coded from absence (0) through disease codes (1–4). The supplied file uses output with classes 0/1, and its provenance and mapping are unconfirmed. Existing project inspection found a possible opposite class orientation relative to the Hungarian source crosswalk. Until lineage and label semantics are verified, outcome naming and disease-positive sensitivity are not secure.
2. Cleveland-style classification benchmarks have been used extensively. Re-running common classifiers, adding an ensemble, or adding SHAP is not novel by itself. Results from unlike splits and protocols are not a valid leaderboard.
3. Prediction-model methodology emphasizes sample size, overfitting control, complete evaluation (including calibration), transparent reporting, and external validation [8–12]. Systematic reviews find recurring bias and generalizability problems in ML prediction. A small internal benchmark can investigate methods but cannot close those broader gaps.
4. Discrimination, calibration, uncertainty, explanations, and subgroup metrics answer different questions. Evidence for one fitted model or task does not automatically apply to another.

## Empirical answer from this project

The actual Phase 14 five-fold training-only OOF ablation found:

- Stage A (all-feature Logistic Regression) had accuracy 0.859 and F1 0.872.
- Stage B (fold-local mutual-information selection of 15 encoded features plus Logistic Regression) had ROC-AUC 0.916, average precision 0.922, and Brier score 0.112, leading those staged candidates on these pooled metrics; its accuracy was 0.846 and F1 0.861, below Stage A.
- Stage F, the integrated selected/tuned/calibrated soft vote, had lower accuracy (0.834), ROC-AUC (0.885), and worse Brier score (0.126) than simpler Stage B.
- The experiments show a metric tradeoff and no evidence that combining components makes the model broadly superior. Fold variability and the limited sample do not support a claim of statistically established superiority [8, 9].

The nominal external evaluation used a historical UCI Hungary cohort and a frozen Phase 6 baseline, not final Stage B [1]. Extensive missingness and uncertain primary target orientation/provenance prevent it from functioning as definitive independent validation of the final model.

## Defensible gap statement

A narrow evidence gap remains for this exact supplied file: a leakage-aware, reproducible evaluation had not yet been documented in this project that compares an all-feature baseline with fold-local feature selection and staged extensions while preserving metric tradeoffs and disclosing unresolved label provenance. The implemented study supplies that bounded analysis and traceable artifacts.

This is a project-specific empirical gap, not evidence that no prior publication conducted the same analysis. The literature review was focused rather than exhaustive, and the common Cleveland benchmark has many previous ML studies. Do not claim a novel algorithm, a new clinical risk score, improved heart-attack prediction, or clinical utility [3–7, 10, 11].

## Contribution claims supported by the evidence

- The project provides an executable, leakage-aware comparison using the supplied dataset as the primary data source.
- It compares seven baseline classifiers and records a staged feature-selection/tuning/ensemble/calibration ablation with pooled OOF evidence.
- It makes selection of Stage B explicit as a probability/ranking-oriented tradeoff, preserves Stage A's higher accuracy/F1, and reports that Stage F did not improve the combined result.
- It documents unresolved target/provenance issues, limits calibration and uncertainty claims to the models actually evaluated, and avoids causal SHAP interpretations [12–14].
- It provides a research/educational Flask interface with explicit non-diagnostic limitations.

These are reproducibility and evidence-reporting contributions, not claims of prospective, clinical, or population-level validity.

## Claims not supported

- Predicts heart attacks or future myocardial infarction risk. The target is not verified as a prospective event with a stated horizon.
- The meaning or positive-class orientation of output.
- Superiority over clinical risk scores, clinicians, or published values from different data/splits.
- A clinically useful operating threshold, calibration of selected Stage B, fairness, causal feature effects, or guaranteed uncertainty coverage in new populations.
- External validation of Stage B. The available exploratory Hungary analysis evaluated a different frozen baseline and has mapping limitations.
- Novelty based solely on mutual information, Logistic Regression, ensembles, calibration, conformal prediction, explainability, or the web application.

## Work needed before a clinical prediction claim

1. Recover the exact source URL/version and data dictionary for the supplied CSV; verify feature mapping, target derivation, and positive-class orientation against the source without publishing patient-level data.
2. Define intended population, prediction moment, outcome, horizon, and use. A future-attack-risk goal requires temporally defined incident events and predictors available at prediction time.
3. Prespecify predictors, model-selection protocol, metrics, and thresholds; plan sample size based on outcome prevalence and model complexity.
4. Evaluate on an untouched external cohort from a distinct time/site/population, assessing discrimination, calibration-in-the-large/slope/curve, clinical utility, and subgroup uncertainty.
5. Follow TRIPOD+AI reporting and assess development/evaluation quality and applicability with PROBAST+AI. Obtain appropriate ethics/privacy review for any clinical or patient-facing study.

## Recommended manuscript framing

Use a title such as **“Leakage-Aware Benchmarking and Ablation of Tabular Classifiers on a Supplied Heart-Disease Dataset”** unless source provenance and target meaning can be established [1, 2]. Describe classes as “dataset class 0/1,” call the model a research classifier, and identify the analysis as exploratory. Do not title this work as heart-attack prediction on current evidence.
