# Phase 15 — Final model selection

## Selected model

The final research candidate is Stage B: the Phase 4 fold-local preprocessing pipeline, mutual-information `SelectKBest(k=15)`, and Logistic Regression (`C=1.0`, `max_iter=2000`). The complete raw-input pipeline is saved at `models/final_model.joblib`; the fitted selector is also saved separately at `models/feature_selector.joblib`. The independent preprocessing artifact from Phase 4 remains at `models/preprocessing_pipeline.joblib`.

## Selection evidence

Selection was based on training-only five-fold OOF results from Phase 14, not the saved holdout. Stage B had pooled ROC-AUC 0.916, average precision 0.922, and Brier score 0.112, the best values among the six candidates for those probability/ranking measures. It had accuracy 0.846 and F1 0.861, below the all-feature Logistic Regression baseline (Stage A: accuracy 0.859, F1 0.872). Stage A, therefore, remains a reasonable simpler comparator. The tuned and selected integrated soft vote (Stage F) had accuracy 0.834, ROC-AUC 0.885, average precision 0.871, and Brier 0.126; the ensemble was not selected. This decision emphasizes ranking and probability quality for a research risk-estimation candidate and accepts the observed accuracy/F1 tradeoff. The CV comparisons are small-sample internal estimates and do not prove superiority.

Phase 10 found no consistently helpful calibration method for the tested baseline models/soft vote. Its calibration study did not evaluate calibrated Stage B directly, so no post-hoc calibration is attached to the final model. Phase 11's conformal and selective prediction results apply to soft voting, not the selected Stage B model; no uncertainty guarantee is attributed to the final model. No clinical threshold is set. The model artifact accepts raw feature columns and returns probabilities for dataset classes 0 and 1; the meaning of class 1 remains unverified.

## Final fit and validation boundary

After selection, the full Stage B pipeline was refit on the 241 rows in the saved training partition. The 61 saved holdout rows were not used for selection, tuning, fitting, calibration, feature ranking, or threshold setting. The runner reloads the serialized artifact and validates the probability interface on training rows only. This interface check is not an independent performance evaluation.

The fitted selector retained: `chol`, `thalachh`, `oldpeak`, `sex_0`, `sex_1`, `cp_0`, `cp_3`, `exng_0`, `exng_1`, `slp_1`, `slp_2`, `caa_0`, `caa_4`, `thall_2`, and `thall_3`. Selection is an algorithmic ranking outcome, not evidence of causal or clinical importance.

## Limitations

The primary dataset is small, its provenance and target direction remain unresolved, and Phase 13 found a target-code discrepancy relative to the historical UCI comparison cohort. The target endpoint is not established as acute heart attack or prospective cardiovascular risk. The final artifact is a research/educational model only; clinical validity, fairness, and deployment readiness have not been established.
