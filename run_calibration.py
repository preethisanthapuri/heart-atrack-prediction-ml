"""Compare uncalibrated and calibrated out-of-fold probabilities."""
from pathlib import Path

from src.calibration import run_calibration

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    summary, _, figure = run_calibration(
        ROOT / "data/processed/cleaned_dataset.csv",
        ROOT / "results/metrics/preprocessing_split.json",
        ROOT / "results/tables/calibration.csv",
        ROOT / "results/tables/calibration_curve.csv",
        ROOT / "results/figures/calibration/calibration_curves.png",
        n_splits=5,
        inner_folds=3,
        random_state=42,
        n_bins=8,
    )
    print("Out-of-fold calibration (lower Brier is better; ROC-AUC retained):")
    print(summary.to_string(index=False))
    print(f"Saved calibration plot: {figure}")
