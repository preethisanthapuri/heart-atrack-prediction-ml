"""Run Phase 5 baseline classifiers on the saved Phase 4 train/test split."""
from pathlib import Path
from src.evaluation import run_baseline_screening

ROOT = Path(__file__).resolve().parent
if __name__ == "__main__":
    results = run_baseline_screening(
        ROOT / "data/processed/cleaned_dataset.csv",
        ROOT / "models/preprocessing_pipeline.joblib",
        ROOT / "results/metrics/preprocessing_split.json",
        ROOT / "models/baselines",
        ROOT / "results/tables/baseline_screening.csv",
    )
    print(results[["model", "holdout_accuracy", "fit_seconds"]].to_string(index=False))
    print("Preliminary screening only; formal evaluation is Phase 6.")
