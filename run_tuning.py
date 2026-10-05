"""Run Phase 8 hyperparameter searches on the saved training split only."""
from pathlib import Path

from src.tuning import run_tuning

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    summary, candidates = run_tuning(
        ROOT / "data/processed/cleaned_dataset.csv",
        ROOT / "results/metrics/preprocessing_split.json",
        ROOT / "results/tables/tuning_summary.csv",
        ROOT / "results/tables/tuning_results.csv",
        n_splits=5,
        random_state=42,
        n_iter=12,
    )
    print("Best configurations by cross-validated ROC-AUC (search-selection estimates):")
    print(summary.to_string(index=False))
    print(f"Evaluated {len(candidates)} hyperparameter configurations across {summary['model'].nunique()} models.")
