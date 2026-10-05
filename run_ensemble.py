"""Compare soft voting and stacking on training-only cross-validation folds."""
from pathlib import Path

from src.ensemble import run_ensemble

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    summary, folds = run_ensemble(
        ROOT / "data/processed/cleaned_dataset.csv",
        ROOT / "results/metrics/preprocessing_split.json",
        ROOT / "results/tables/ensemble_comparison.csv",
        ROOT / "results/tables/ensemble_folds.csv",
        n_splits=5,
        random_state=42,
    )
    print("Individual models and ensembles (five-fold CV mean / sample SD):")
    print(summary.to_string(index=False))
    print(f"Saved {len(folds)} per-fold metric observations.")
