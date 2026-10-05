"""Score the fixed Phase 6 model on the independent UCI Hungary cohort."""
from pathlib import Path

from src.external_validation import run_external_validation

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    table, report = run_external_validation(
        ROOT / "data/external/uci_heart_disease_hungarian.data",
        ROOT / "models/preprocessing_pipeline.joblib",
        ROOT / "models/baselines/logistic_regression.joblib",
        ROOT / "results/tables/external_validation.csv",
        ROOT / "results/metrics/external_validation.json",
    )
    print(table.to_string(index=False))
    print("\nData-quality and provenance summary:")
    print({k: report["external_dataset"][k] for k in (
        "source_rows", "source_target_counts", "binary_target_counts",
        "source_missing_feature_values", "missing_feature_values_after_source_sentinel_mapping",
        "source_feature_rows_with_any_missing", "feature_rows_with_any_missing", "source_sha256",
    )})
