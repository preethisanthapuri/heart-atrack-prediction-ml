# Phase 18 — Focused literature review

**Project:** heart-atrack-prediction-ml  
**Review type:** Focused narrative review; not a registered or exhaustive systematic review.  
**Search date:** 2026-10-06  
**Approach:** Targeted scholarly searches for the UCI benchmark and outcome, cardiovascular ML reviews/comparisons, sample size and bias, calibration, reporting/appraisal, and model explanations. Selection was relevance-based; no PRISMA screening counts are claimed.

## Dataset and outcome context

The UCI Heart Disease repository describes four source sites (Cleveland, Hungary, Switzerland, and VA Long Beach), 76 original attributes, and the conventionally used 14-column subset. It documents num as angiographic heart-disease diagnosis coded 0–4, with many Cleveland experiments dichotomizing 0 (absence) against 1–4 (presence). The repository reports 303 Cleveland records and identifies Detrano et al. (1989) as the introductory paper. The supplied project file has 303 rows and 14 numeric columns and a target called output, but its provenance and target recoding are not independently established. Similar dimensions do not prove that this file is the canonical UCI Cleveland data or that its 0/1 orientation follows an assumed convention. Accordingly, results here describe dataset classes 0 and 1, not a verified clinical endpoint. [1, 2]

The original clinical context also matters: the Cleveland records arose from patients referred for coronary angiography, rather than a representative prospective primary-prevention population. The documented endpoint is heart-disease diagnosis; this is not a documented acute myocardial infarction prediction target or a future-event risk horizon. Calling the current system “heart-attack prediction” therefore exceeds the documented endpoint. [1, 2]

## Prior cardiovascular and heart-disease prediction research

Krittanawong et al. synthesized 103 cohorts comprising 3,377,318 individuals across coronary artery disease, heart failure, stroke, and arrhythmia prediction. Their hierarchical summary ROC analyses found algorithm-specific pooled discrimination estimates, including a pooled AUC of 0.88 (95% CI 0.84–0.91) for boosting in coronary artery disease prediction. The authors emphasized heterogeneity, limited clinical interpretability, and a need to evaluate calibration, thresholds, and external reproduction. This broad evidence supports evaluating more than accuracy, but pooled results across endpoints, populations, and studies are not a direct benchmark for this 302-row dataset. [3]

A systematic review comparing ML with traditional risk scores in primary-prevention ASCVD cohorts included 16 retrospective studies and more than 3.3 million people. Only three studies externally validated models; 11 reported calibration and 11 were judged high risk of bias. The pooled discrimination difference favored the best ML models by a small amount. This review concerns future ASCVD risk in primary prevention and comparator risk equations, scientifically different from classifying a historical angiography-associated dataset. It cautions against reading a small discrimination advantage as implementation evidence. [4]

Heart-disease classification papers repeatedly use the 303-row Cleveland benchmark. For example, an implementation study reports a Random Forest test accuracy of 88.5% and ROC-AUC of 0.92 on selected Cleveland features. These single-split results are not directly comparable with this project's training-fold cross-validation: split, preprocessing, target construction, and selection can materially change estimates. A high benchmark score alone establishes neither improvement nor transportability. [5]

Recent reviews continue to identify repeated use of small Cleveland/Statlog-style benchmark datasets and lack of independent, multi-site or prospective evaluation as generalization limitations. A 2025 systematic review of heart-disease ML studies likewise identifies small benchmark cohorts, interpretability, and external validation as unresolved issues. Reviews vary in inclusion criteria and quality; they map recurring limitations rather than give a precise census of the field. [6, 7]

## Methodological guidance

Riley et al. explain that prediction-model sample size must be considered relative to outcome frequency, candidate parameters, and anticipated performance; small datasets can yield overfitting and unstable selected features. They recommend using all available data for development with resampling for internal validation when no separate external dataset is available, and provide context-specific sample-size methods. With only 302 cleaned rows, 13 raw inputs, and a 241-row development partition, this project is exploratory. Five-fold CV reduces dependence on one split but cannot create missing population diversity or precise external-performance evidence. [8]

A systematic review of 152 supervised-ML prediction studies found high overall risk of bias in 133/152 (88%) model developments and 15/19 (79%) external validations; analysis weaknesses included sample size, overfitting, missing data, and incomplete calibration/performance reporting. This motivates explicit leakage boundaries and experiment reporting, but following individual safeguards does not remove this project's sample-size or outcome-definition limitations. [9]

TRIPOD+AI is a 27-item reporting guideline for prediction-model studies using regression or ML. It addresses transparent reporting, including open-science information; it is not a prescription for model development or a risk-of-bias score. PROBAST+AI separately structures assessment of development quality, evaluation risk of bias, and applicability, with attention to participant/data sources, predictors, outcomes, analysis, and fairness. These are appropriate manuscript-preparation checklists, not evidence that this project has clinical validity. [10, 11]

## Probability quality, explainability, and uncertainty

Discrimination ranks cases above non-cases; calibration asks whether predicted probabilities agree with observed outcome frequencies. Van Calster et al. describe calibration as essential for interpreting risk predictions, often under-reported, and potentially misleading when poor. They stress sample-size limitations for external calibration curves and possible model updating when justified. In this project, Brier score is reported in the training-only ablation, but a Brier score alone does not establish calibration, and Stage B has not received model-specific independent calibration evaluation. Phase 10 studied other baseline/voting configurations; those results are not transferred to Stage B. [12]

Lundberg and Lee's SHAP paper defines additive feature attributions with a unifying framework and theoretical properties. SHAP explains attribution of a fitted model output under feature-dependence/background assumptions; it does not identify causal effects or validate the medical meaning of the variables. The existing SHAP experiment explains the Phase 6 Logistic Regression baseline, not selected Stage B. [13]

Split conformal prediction offers set-valued predictions with finite-sample marginal coverage under exchangeability assumptions. Coverage can fail under distribution shift or if exchangeability is unreasonable. The existing uncertainty study applied conformal sets to soft voting, not Stage B, and its coverage is an exploratory internal estimate. It is not a patient-specific clinical confidence guarantee. [14]

## Synthesis

The literature supports emphasis on (a) a precisely defined endpoint and horizon, (b) verified provenance and label mapping, (c) separation of development from evaluation, (d) discrimination and probability quality, (e) explicit missingness and sample-size limits, and (f) independent validation before transportability claims. It does not support treating a popular classifier, feature selection, ensemble, SHAP, or their combination as novelty by itself.

The most defensible contribution available from the completed experiments is a transparent, reproducible, small-sample benchmark analysis of the user-supplied file, including leakage-aware comparisons, a staged ablation that exposes metric tradeoffs, and explicit documentation that the full combination did not win and dataset semantics remain uncertain. This is bounded empirical evidence, not a novel clinical prediction method. Dataset uncertainty is a constraint that must be resolved before disease-specific claims.

## Evidence extraction for key sources

| Citation | Dataset/population | Method and evaluation | Main reported finding | Limitation relevant here |
|---|---|---|---|---|
| UCI Heart Disease record [1] | Four legacy sites; Cleveland commonly 303 records; 76 raw attributes and a commonly used 14-column subset | Dataset documentation and target/data dictionary, not a comparative experiment | Target num is disease diagnosis coded 0–4; Cleveland ML studies often group 0 versus 1–4 | It does not establish provenance or encoding of the supplied output column |
| Detrano et al. [2] | Patients referred for coronary angiography across international clinical settings | Development/application of a probability algorithm for coronary artery disease diagnosis; study-specific diagnostic performance | Historical clinical study underlying a widely used benchmark context | Referral cohort and diagnosis task are not future community heart-attack risk prediction |
| Krittanawong et al. [3] | 103 cardiovascular cohorts, 3,377,318 total individuals; endpoints include CAD, stroke, heart failure, arrhythmia | Systematic review/meta-analysis; hierarchical summary ROC, pooled discrimination/sensitivity/specificity | CAD boosting subgroup pooled AUC 0.88 (95% CI 0.84–0.91) | Heterogeneous endpoints/data/models; pooled result is not comparable to this small benchmark; clinical translation remains difficult |
| ASCVD ML-vs-risk-score review [4] | 16 retrospective primary-prevention studies; 3,302,515 people | Systematic review/meta-analysis comparing ML and traditional risk scores; PROBAST assessment | Pooled top-ML C-statistic 0.773 vs 0.759 for traditional scores; only 3 studies externally validated; 11 high risk of bias | Primary-prevention future-risk target differs from present diagnostic-code classification |
| Heart-disease implementation study [5] | UCI Cleveland, 303 records | Multiple classifiers; single train/test evaluation reported | Random Forest accuracy 88.5%, ROC-AUC 0.92 | One split and selected features; not protocol-aligned with this project's CV and not independent external validation |
| Heart-disease ML systematic review [6] | Empirical heart-disease prediction literature; includes Cleveland/Statlog benchmark papers | Systematic review of datasets, algorithms, tasks, evaluation, and limitations | Identifies repeated small benchmark data and need for generalization/interpretability work | Review findings depend on its search/inclusion and quality appraisal; does not prove an exact gap here |
| CVD AI prediction systematic review [7] | Cardiovascular risk models across general and special populations | Database review to July 2021; model characteristics and PROBAST appraisal; proposes independent-validation screen | Identifies weaknesses in design, reporting, evaluation, and replicability | Broad CVD population/tasks; review does not validate this project's model |
| Riley et al. [8] | Methodological paper; no single cohort | Sample-size framework for binary, time-to-event, and continuous clinical prediction development | Requirements depend on prevalence, candidate parameters, and target performance; resampling can use development data efficiently | Guidance cannot compensate for a small or nonrepresentative source dataset |
| Andaur Navarro et al. [9] | Sample of 152 supervised-ML model studies across clinical domains | Systematic review; PROBAST risk-of-bias appraisal | 88% of development studies and 79% of external validations were high overall risk of bias | Not cardiovascular-specific; broad methodological findings, not a score for this project |
| Collins et al. [10] | Reporting guideline; no patient cohort | Consensus checklist for clinical prediction model development/validation using regression or ML | 27 main reporting items, including open-science guidance | Reporting guideline, not a model development recipe, quality grade, or validation study |
| Moons et al. [11] | Appraisal framework; no patient cohort | PROBAST+AI framework for development quality, evaluation risk of bias, applicability | Separates development quality from performance-evaluation bias and considers fairness | Appraisal tool; application does not itself remove bias or demonstrate utility |
| Van Calster et al. [12] | Methodological commentary, not a new cohort experiment | Conceptual/methodological review of calibration assessment and updating | Calibration is essential and often under-attended; small external samples limit calibration-curve reliability | Does not establish calibration of this project's selected model |
| Lundberg and Lee [13] | General ML explanation methods; not a clinical validation dataset | Theoretical unification and methods for additive feature attribution | Introduces SHAP and desirable attribution properties | Attribution is model-dependent and not causal evidence or proof of validity |
| Vovk et al. [14] | General exchangeable-data theory | Conformal prediction and algorithmic learning framework | Formalizes set-valued prediction with distribution-free marginal validity under assumptions | Guarantees depend on exchangeability; do not imply calibrated clinical risk or survive arbitrary shift |

## References

[1] A. Janosi, W. Steinbrunn, M. Pfisterer, and R. Detrano, “Heart Disease,” UCI Machine Learning Repository, DOI: [10.24432/C52P4X](https://doi.org/10.24432/C52P4X). Dataset record/data dictionary: [UCI repository](https://archive.ics.uci.edu/dataset/45/heart+disease).

[2] R. Detrano et al., “International application of a new probability algorithm for the diagnosis of coronary artery disease,” *The American Journal of Cardiology*, vol. 64, no. 5, pp. 304–310, 1989. DOI: [10.1016/0002-9149(89)90524-9](https://doi.org/10.1016/0002-9149(89)90524-9).

[3] C. Krittanawong et al., “Machine learning prediction in cardiovascular diseases: a meta-analysis,” *Scientific Reports*, vol. 10, 16057, 2020. DOI: [10.1038/s41598-020-72685-1](https://doi.org/10.1038/s41598-020-72685-1).

[4] “Machine-learning versus traditional approaches for atherosclerotic cardiovascular risk prognostication in primary prevention cohorts: a systematic review and meta-analysis,” *European Heart Journal—Quality of Care and Clinical Outcomes*, vol. 9, no. 4, pp. 310–322, 2023. DOI: [10.1093/ehjqcco/qcad017](https://doi.org/10.1093/ehjqcco/qcad017).

[5] K. Karthick, S. K. Aruna, R. Samikannu, R. Kuppusamy, Y. Teekaraman, and A. R. Thelkar, “Implementation of a Heart Disease Risk Prediction Model Using Machine Learning,” *Computational and Mathematical Methods in Medicine*, vol. 2022, Art. 6517716, 2022. DOI: [10.1155/2022/6517716](https://doi.org/10.1155/2022/6517716). Metrics are mentioned only to characterize one published benchmark; protocols differ.

[6] T. Banerjee and İ. Paçal, “A systematic review of machine learning in heart disease prediction,” *Turkish Journal of Biology*, vol. 49, no. 5, pp. 600–634, 2025. DOI: [10.55730/1300-0152.2766](https://doi.org/10.55730/1300-0152.2766).

[7] Y. Cai, Y.-Q. Cai, L.-Y. Tang, et al., “Artificial intelligence in the risk prediction models of cardiovascular disease and development of an independent validation screening tool: a systematic review,” *BMC Medicine*, vol. 22, no. 1, Art. 56, 2024. DOI: [10.1186/s12916-024-03273-7](https://doi.org/10.1186/s12916-024-03273-7).

[8] R. D. Riley et al., “Calculating the sample size required for developing a clinical prediction model,” *BMJ*, vol. 368, m441, 2020. DOI: [10.1136/bmj.m441](https://doi.org/10.1136/bmj.m441).

[9] C. L. Andaur Navarro et al., “Risk of bias in studies on prediction models developed using supervised machine learning techniques: systematic review,” *BMJ*, vol. 375, n2281, 2021. DOI: [10.1136/bmj.n2281](https://doi.org/10.1136/bmj.n2281).

[10] G. S. Collins et al., “TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods,” *BMJ*, vol. 385, q902, 2024. DOI: [10.1136/bmj.q902](https://doi.org/10.1136/bmj.q902).

[11] K. G. M. Moons et al., “PROBAST+AI: an updated quality, risk of bias, and applicability assessment tool for prediction models using regression or artificial intelligence methods,” *BMJ*, vol. 388, e082505, 2025. DOI: [10.1136/bmj-2024-082505](https://doi.org/10.1136/bmj-2024-082505).

[12] B. Van Calster, D. J. McLernon, M. van Smeden, L. Wynants, and E. W. Steyerberg, “Calibration: the Achilles heel of predictive analytics,” *BMC Medicine*, vol. 17, 230, 2019. DOI: [10.1186/s12916-019-1466-7](https://doi.org/10.1186/s12916-019-1466-7).

[13] S. M. Lundberg and S.-I. Lee, “A Unified Approach to Interpreting Model Predictions,” in *Advances in Neural Information Processing Systems 30*, 2017. [Proceedings paper](https://proceedings.neurips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions).

[14] V. Vovk, A. Gammerman, and G. Shafer, *Algorithmic Learning in a Random World*. Springer, 2005. DOI: [10.1007/b106715](https://doi.org/10.1007/b106715).
