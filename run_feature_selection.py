"""Run Phase 7 fold-local feature-selection comparisons."""
from pathlib import Path

from src.feature_selection import run_feature_selection

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    comparison, features = run_feature_selection(
        ROOT / "data/processed/cleaned_dataset.csv",
        ROOT / "results/metrics/preprocessing_split.json",
        ROOT / "results/tables/feature_selection_comparison.csv",
        ROOT / "results/tables/feature_selection.csv",
        n_splits=5,
        random_state=42,
        selection_size=15,
    )
    print("Training-only stratified 5-fold comparison (mean / sample SD):")
    print(comparison.to_string(index=False))
    print(f"Saved feature ranks/fold-selection stability for {len(features)} method-feature rows.")
