# Leakage-Aware Benchmarking and Ablation of Tabular Classifiers on a Supplied Heart-Disease Dataset

**Author and affiliation information:** To be completed by the project authors.

## Abstract

Small tabular heart-disease datasets are frequently used to compare machine-learning classifiers, but benchmark scores alone do not establish clinical validity or future cardiovascular-event risk. We report a leakage-aware analysis of a user-provided dataset containing 303 rows and 14 columns. Its source provenance and binary target semantics could not be independently confirmed. Removing one exact duplicate left 302 observations. A stratified split allocated 241 rows to development and 61 to a holdout; the holdout had been screened for accuracy during baseline work and is not treated as an untouched final evaluation. On training-only five-fold stratified cross-validation, Logistic Regression achieved ROC-AUC 0.908 ± 0.039 and accuracy 0.859 ± 0.045. A staged out-of-fold ablation selected Logistic Regression with fold-local mutual-information selection of 15 encoded features for pooled ROC-AUC (0.916), average precision (0.922), and Brier score (0.112). The all-feature baseline had higher accuracy (0.859 versus 0.846) and F1 (0.872 versus 0.861); an integrated tuned, selected, calibrated soft-voting configuration did not improve overall performance. A historical UCI Hungary cohort was scored with a frozen baseline, but missingness and uncertain target mapping limit interpretation. The results support a reproducible, dataset-specific benchmark comparison with explicit metric tradeoffs. They do not establish heart-attack prediction, clinical utility, or population-level validity.

**Keywords—** Cardiovascular machine learning, heart-disease dataset, leakage-aware evaluation, feature selection, calibration, model validation.

## I. Introduction

Cardiovascular prediction studies require evaluation that separates model development from performance assessment and distinguishes discrimination from probability quality. Small tabular benchmarks are particularly sensitive to sampling, preprocessing, feature selection, and outcome definition. Methodological reviews document recurring concerns involving sample size, overfitting, missing data, calibration reporting, and external validation [3], [4], [8], [9].

This project asks a bounded empirical question: within the supplied dataset, do fold-local feature selection and staged extensions improve training-only out-of-fold metrics over an all-feature Logistic Regression baseline? The study is not designed as a prospective risk model. The source file's provenance and target coding are unverified, and no future-event horizon is documented.

The contribution is a reproducible comparison of seven baselines, a staged training-only ablation, and explicit documentation of label, holdout, external-data, calibration, uncertainty, and subgroup limitations. No novel algorithm or clinical validation is claimed.

## II. Related Work

The UCI Heart Disease repository describes four source sites and documents the commonly used 14-column subset. Its target num is angiographic disease coded 0–4, with many Cleveland studies dichotomizing absence (0) versus presence (1–4) [1]. Detrano et al. studied a probability algorithm for coronary artery disease diagnosis [2]. This context does not prove that the supplied CSV came from UCI or define the meaning of its output column.

A cardiovascular ML meta-analysis covered 103 cohorts and over 3.3 million people across several endpoints; the authors reported heterogeneous algorithm-specific results and highlighted clinical interpretation and evaluation challenges [3]. A review comparing machine learning with traditional primary-prevention ASCVD scores found only three of 16 studies externally validated models, and 11 were judged at high risk of bias [4]. These future-risk settings are not directly comparable to this dataset's unverified class labels. Cleveland-based classifier comparisons remain common. One published implementation study reported single-split Random Forest accuracy of 88.5% and ROC-AUC of 0.92 on 303 Cleveland records [5]; protocol and target differences preclude direct comparison. Recent reviews continue to identify small benchmarks, transportability, transparency, and independent validation as concerns [6], [7].

Clinical prediction guidance emphasizes sample-size planning, transparent reporting, and risk-of-bias appraisal [8]–[11]. Calibration is distinct from discrimination and necessary for interpreting probabilities [12]. SHAP attributes model outputs but does not identify causal effects [13]. Conformal prediction coverage depends on assumptions such as exchangeability [14].

## III. Research Gap and Scope

This focused narrative review is not a registered systematic review. The commonly used benchmark has extensive prior classifier literature; familiar algorithms, an ensemble, or SHAP do not establish novelty. The narrow project-specific gap is a documented comparison for the supplied file that uses fold-local preprocessing and selection, staged OOF evidence, and explicit disclosure of metric tradeoffs and unresolved label provenance.

The supplied file's dimensions resemble the UCI benchmark but do not establish lineage or binary outcome orientation. Results therefore use the neutral term **dataset class**. We do not assert that class 1 means disease, that the endpoint is myocardial infarction, or that the model estimates future risk.

## IV. Methodology

### A. Dataset and cleaning

The provided CSV contained 303 rows and 14 columns. It had 13 integer and one floating-point column at initial audit. The target candidate was output, with observed classes 0 and 1. Cleaning removed one exact duplicate, leaving 302 observations; no missing or invalid values were found in the primary cleaned dataset. IQR-flagged observations were retained. The target counts were 138 class 0 and 164 class 1. The 13 predictors were age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, and thall. Eight were handled as categorical codes; their clinical codebook remains unverified.

### B. Preprocessing and split

A stratified 80/20 split (random state 42) produced 241 training rows and 61 holdout rows. Training contained 110 class-0 and 131 class-1 observations; holdout contained 28 and 33. Numeric features used median imputation and standard scaling; categorical codes used most-frequent imputation and one-hot encoding with unknown categories ignored. The pipeline transformed 13 raw inputs into 30 columns. Preprocessing was refit within CV training folds.

The holdout was screened for accuracy in Phase 5. Its later metrics are descriptive, not an untouched final estimate. Phases 7–16 selection, ablation, and final fitting used training indices only.

### C. Models and metrics

Seven baselines were evaluated: Logistic Regression, K-Nearest Neighbors, Decision Tree, RBF Support Vector Classifier, Random Forest, Extra Trees, and Gradient Boosting. Five-fold stratified CV was run on training data only. Metrics included accuracy, precision, class-1 recall, class-0 specificity, F1, ROC-AUC, and average precision. Class 1 is a computational label, not a confirmed disease-positive class.

### D. Feature selection and tuning

Experiments compared all 30 transformed features with fold-local mutual information, ANOVA SelectKBest, Logistic Regression RFE, and Extra Trees model-based selection, retaining 15 encoded features for selected variants. Logistic Regression and KNN were compared. GridSearchCV tuned Logistic Regression and RBF SVC; RandomizedSearchCV sampled KNN settings. Mean ROC-AUC was the refit criterion. Tuning used training data only; selected CV scores are not unbiased estimates of tuned-model performance.

### E. Ensemble, calibration, and uncertainty

Soft voting combined Logistic Regression, KNN, and Random Forest. Stacking combined Logistic Regression, KNN, and RBF SVC. Both were evaluated on the same training-only outer folds. Calibration compared uncalibrated, sigmoid, and isotonic CalibratedClassifierCV probabilities for baseline/voting models using five outer folds and three internal calibration folds. It did not directly evaluate calibrated Stage B.

Split-conformal class sets and confidence-ranked risk-coverage were evaluated for soft voting only. Empirical coverage is exploratory and conditional on exchangeability; it is not transferred to Stage B.

### F. Explainability

SHAP LinearExplainer was applied to the Phase 6 Logistic Regression baseline using its saved preprocessing pipeline and training data. Attributions are model-output explanations, not causal effects. SHAP was not run for selected Stage B.

### G. External and subgroup analysis

A frozen Phase 6 Logistic Regression baseline and training-fitted preprocessor were applied to the UCI Hungary site without refitting. The feature/target crosswalk is uncertain and the site has substantial missingness. Descriptive subgroup metrics were calculated from Stage F OOF predictions by sex code and age group; no inferential fairness analysis or threshold adjustment was performed.

## V. Experimental Setup

Model development used the 241-row training partition. Five-fold stratified CV generated baseline and Phase 14 outer-fold predictions. Fold standard deviations are descriptive, not confidence intervals. Candidate tuning remained within training folds in the ablation. The 61-row holdout was excluded from Phases 7–16 final selection and fitting but had been screened in Phase 5; no untouched holdout claim is made.

The selected Stage B pipeline consists of preprocessing, mutual-information SelectKBest (k=15), and Logistic Regression (C=1.0, max_iter=2000). It was refit on the 241 training rows after selection. No medical threshold was selected. Probabilities refer to dataset codes 0 and 1.

## VI. Results

### A. Baseline cross-validation

**TABLE I — FIVE-FOLD TRAINING CV, MEAN ± SAMPLE SD**

| Model | Accuracy | Sensitivity, class 1 | Specificity, class 0 | F1 | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.859 ± 0.045 | 0.885 ± 0.061 | 0.827 ± 0.081 | 0.872 ± 0.040 | 0.908 ± 0.039 | 0.919 ± 0.027 |
| K-Nearest Neighbors | 0.867 ± 0.064 | 0.901 ± 0.064 | 0.827 ± 0.087 | 0.881 ± 0.057 | 0.897 ± 0.062 | 0.870 ± 0.076 |
| Decision Tree | 0.751 ± 0.148 | 0.747 ± 0.240 | 0.755 ± 0.083 | 0.750 ± 0.186 | 0.751 ± 0.141 | 0.731 ± 0.111 |
| SVC (RBF) | 0.834 ± 0.062 | 0.855 ± 0.074 | 0.809 ± 0.093 | 0.848 ± 0.057 | 0.907 ± 0.044 | 0.908 ± 0.031 |
| Random Forest | 0.822 ± 0.080 | 0.832 ± 0.085 | 0.809 ± 0.109 | 0.835 ± 0.073 | 0.902 ± 0.064 | 0.909 ± 0.056 |
| Extra Trees | 0.838 ± 0.060 | 0.877 ± 0.069 | 0.791 ± 0.094 | 0.855 ± 0.053 | 0.895 ± 0.058 | 0.898 ± 0.058 |
| Gradient Boosting | 0.817 ± 0.107 | 0.831 ± 0.135 | 0.800 ± 0.119 | 0.830 ± 0.107 | 0.880 ± 0.064 | 0.888 ± 0.062 |

KNN had the highest mean baseline accuracy and F1; Logistic Regression had the highest mean ROC-AUC and average precision. These are small-sample internal estimates.

### B. Feature selection, tuning, and ensembles

Mutual-information selection with Logistic Regression had mean CV ROC-AUC 0.914 ± 0.040 and average precision 0.926 ± 0.022, versus 0.908 ± 0.039 and 0.919 ± 0.027 for all features. Accuracy was lower for the selected model (0.847 ± 0.060 versus 0.859 ± 0.045). The differences do not establish meaningful improvement. Best hyperparameter-search CV ROC-AUC values were 0.911 for Logistic Regression, 0.909 for SVC, and 0.912 for KNN; these are candidate-selection scores. Soft voting achieved mean ROC-AUC 0.913 ± 0.045 and accuracy 0.851 ± 0.053; stacking achieved 0.909 ± 0.040 and 0.847 ± 0.060. Ensembles did not improve all metrics.

### C. Ablation

**TABLE II — POOLED TRAINING OOF METRICS**

| Stage | Configuration | Accuracy | Sensitivity | Specificity | F1 | ROC-AUC | Avg. precision | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | Logistic Regression, all features | 0.859 | 0.885 | 0.827 | 0.872 | 0.910 | 0.915 | 0.118 |
| B | MI selection k=15 + Logistic Regression | 0.846 | 0.878 | 0.809 | 0.861 | 0.916 | 0.922 | 0.112 |
| C | B + inner-fold tuning | 0.842 | 0.878 | 0.800 | 0.858 | 0.905 | 0.907 | 0.118 |
| D | Soft voting | 0.851 | 0.878 | 0.818 | 0.865 | 0.913 | 0.913 | 0.113 |
| E | D + sigmoid calibration | 0.855 | 0.893 | 0.809 | 0.870 | 0.911 | 0.909 | 0.114 |
| F | Integrated tuned/selected/calibrated vote | 0.834 | 0.901 | 0.755 | 0.855 | 0.885 | 0.871 | 0.126 |

Stage B led on pooled ROC-AUC, average precision, and Brier score, while Stage A had higher accuracy and F1. Stage F did not improve the overall metrics. This is a metric tradeoff, not proof of superiority.

### D. Calibration and uncertainty

For soft voting, uncalibrated probabilities had Brier 0.113 ± 0.034 and ROC-AUC 0.913 ± 0.045. Sigmoid calibration yielded 0.115 ± 0.033 and 0.908 ± 0.050; isotonic yielded 0.114 ± 0.035 and 0.904 ± 0.054. Calibration was not consistently beneficial and was not evaluated specifically for Stage B.

For soft-voting conformal sets, nominal 90% coverage yielded OOF coverage 0.888, mean set size 1.108, and ambiguous-set rate 0.108. At nominal 80%, observed coverage was 0.817 and empty-set rate 0.045. These exploratory results depend on exchangeability and apply to soft voting, not Stage B.

### E. External cohort and subgroup results

The frozen Phase 6 baseline scored 294 historical UCI Hungary records with accuracy 0.833, ROC-AUC 0.883, average precision 0.909, and Brier 0.130. Project class 1 appears aligned with UCI num=0 (angiographic absence), with a one-record count discrepancy and unconfirmed mapping. Missingness included slope in 190/294, ca in 291/294, and thal in 266/294 records. These are dataset-code metrics, not disease-positive performance or validation of final Stage B.

Stage F subgroup estimates were descriptive. For sex_code 0 (n=76) and 1 (n=165), class-1 recall was 0.964 and 0.855, specificity 0.714 and 0.764, and accuracy 0.895 and 0.806. For age groups <50, 50–59, 60–69, and 70+ (n=71, 98, 63, 9), recall was 0.959, 0.863, 0.840, and 1.000; specificity was 0.636, 0.766, 0.816, and 0.667. Small groups and unverified codebooks do not support fairness conclusions.

## VII. Discussion

Model rankings depended on the metric. Feature selection yielded the best pooled ranking and Brier values in the staged ablation, while all-feature Logistic Regression had higher accuracy and F1. The integrated ensemble/calibration stage did not dominate simpler alternatives. Reporting the full tradeoff is more faithful to the evidence than claiming a general ML improvement.

The results are internal estimates from one small dataset. Cross-validation reduces dependence on one random split but cannot create population diversity or remove all model-selection optimism. The holdout had prior accuracy screening, so it is not an untouched estimate. The external study scored a frozen Phase 6 baseline, not final Stage B; severe missingness and uncertain label mapping limit transportability claims. Calibration and uncertainty findings apply to other model configurations. SHAP was run on the Phase 6 baseline, and subgroup differences are descriptive only.

## VIII. Limitations

Limitations include the small sample and unverified source; ambiguous target meaning and orientation; prior holdout screening; absence of a documented incident-event horizon; severe external-cohort missingness; no Stage-B-specific external validation, calibration, uncertainty, or SHAP analysis; sparse subgroups; and no clinical utility or prospective validation. This is exploratory benchmark research.

## IX. Ethical Considerations

The target and codebook require verification before biomedical interpretation. Dataset-class probabilities are not validated heart-attack risks. The Flask interface is for research and education and states that it is not a medical diagnostic tool. Feature attributions are not causal evidence, and subgroup results are not fairness certification. A future clinical study would require a defined intended use, appropriate governance and ethics review, and independent validation.

## X. Conclusion and Future Work

This study compared common classifiers and staged extensions on a supplied small heart-disease dataset. Mutual-information selection with Logistic Regression led on pooled ROC-AUC, average precision, and Brier score, while the all-feature baseline achieved higher accuracy and F1. The integrated ensemble did not improve overall performance. These results are a bounded empirical comparison, not evidence of heart-attack prediction or clinical utility.

Future work should first verify the supplied file's provenance and target mapping. Future-event prediction requires documented incident outcomes, prediction timing and horizon, adequate sample size, and independent temporal/site validation. A subsequent study should evaluate the selected model's calibration, uncertainty, subgroup performance, and clinical utility on an appropriate external population.

## References

[1] A. Janosi, W. Steinbrunn, M. Pfisterer, and R. Detrano, “Heart Disease,” UCI Machine Learning Repository, doi: 10.24432/C52P4X. [Online]. Available: https://archive.ics.uci.edu/dataset/45/heart+disease

[2] R. Detrano et al., “International application of a new probability algorithm for the diagnosis of coronary artery disease,” *The American Journal of Cardiology*, vol. 64, no. 5, pp. 304–310, 1989, doi: 10.1016/0002-9149(89)90524-9.

[3] C. Krittanawong et al., “Machine learning prediction in cardiovascular diseases: a meta-analysis,” *Scientific Reports*, vol. 10, Art. no. 16057, 2020, doi: 10.1038/s41598-020-72685-1.

[4] “Machine-learning versus traditional approaches for atherosclerotic cardiovascular risk prognostication in primary prevention cohorts: a systematic review and meta-analysis,” *European Heart Journal—Quality of Care and Clinical Outcomes*, vol. 9, no. 4, pp. 310–322, 2023, doi: 10.1093/ehjqcco/qcad017.

[5] K. Karthick, S. K. Aruna, R. Samikannu, R. Kuppusamy, Y. Teekaraman, and A. R. Thelkar, “Implementation of a heart disease risk prediction model using machine learning,” *Computational and Mathematical Methods in Medicine*, vol. 2022, Art. no. 6517716, 2022, doi: 10.1155/2022/6517716.

[6] T. Banerjee and İ. Paçal, “A systematic review of machine learning in heart disease prediction,” *Turkish Journal of Biology*, vol. 49, no. 5, pp. 600–634, 2025, doi: 10.55730/1300-0152.2766.

[7] Y. Cai, Y.-Q. Cai, L.-Y. Tang, et al., “Artificial intelligence in the risk prediction models of cardiovascular disease and development of an independent validation screening tool: a systematic review,” *BMC Medicine*, vol. 22, no. 1, Art. no. 56, 2024, doi: 10.1186/s12916-024-03273-7.

[8] R. D. Riley et al., “Calculating the sample size required for developing a clinical prediction model,” *BMJ*, vol. 368, Art. no. m441, 2020, doi: 10.1136/bmj.m441.

[9] C. L. Andaur Navarro et al., “Risk of bias in studies on prediction models developed using supervised machine learning techniques: systematic review,” *BMJ*, vol. 375, Art. no. n2281, 2021, doi: 10.1136/bmj.n2281.

[10] G. S. Collins et al., “TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods,” *BMJ*, vol. 385, Art. no. q902, 2024, doi: 10.1136/bmj-2023-078378.

[11] K. G. M. Moons et al., “PROBAST+AI: an updated quality, risk of bias, and applicability assessment tool for prediction models using regression or artificial intelligence methods,” *BMJ*, vol. 388, Art. no. e082505, 2025, doi: 10.1136/bmj-2024-082505.

[12] B. Van Calster, D. J. McLernon, M. van Smeden, L. Wynants, and E. W. Steyerberg, “Calibration: the Achilles heel of predictive analytics,” *BMC Medicine*, vol. 17, Art. no. 230, 2019, doi: 10.1186/s12916-019-1466-7.

[13] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in *Advances in Neural Information Processing Systems 30*, 2017.

[14] V. Vovk, A. Gammerman, and G. Shafer, *Algorithmic Learning in a Random World*. New York, NY, USA: Springer, 2005, doi: 10.1007/b106715.
