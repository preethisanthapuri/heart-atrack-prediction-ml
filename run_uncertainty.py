"""Run training-only conformal and selective-risk uncertainty evaluation."""
from pathlib import Path

from src.uncertainty import run_uncertainty

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    summary, risk, _ = run_uncertainty(
        ROOT / "data/processed/cleaned_dataset.csv",
        ROOT / "results/metrics/preprocessing_split.json",
        ROOT / "results/tables/uncertainty.csv",
        ROOT / "results/tables/uncertainty_risk_coverage.csv",
        ROOT / "results/tables/uncertainty_predictions.csv",
        n_splits=5,
        calibration_fraction=0.25,
        alphas=(0.10, 0.20),
        random_state=42,
    )
    print("Split-conformal set and selective-risk summary:")
    print(summary.to_string(index=False))
    print("\nConfidence risk-coverage curve:")
    print(risk.to_string(index=False))
