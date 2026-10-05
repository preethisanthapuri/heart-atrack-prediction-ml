"""Run Phase 6 holdout evaluation and stratified cross-validation."""
from pathlib import Path
from src.evaluation import run_model_evaluation

ROOT = Path(__file__).resolve().parent
if __name__ == "__main__":
    result = run_model_evaluation(
        ROOT / "data/processed/cleaned_dataset.csv",
        ROOT / "models/preprocessing_pipeline.joblib",
        ROOT / "results/metrics/preprocessing_split.json",
        ROOT / "results/figures/models",
        ROOT / "results/tables",
        cv_folds=5,
        cv_random_state=42,
    )
    print("Holdout comparison:")
    print(result["holdout_results"].to_string(index=False))
    print("\n5-fold CV mean ± SD:")
    print(result["cross_validation_results"].to_string(index=False))
    print(f"Generated {len(result['figure_paths'])} evaluation figures")
