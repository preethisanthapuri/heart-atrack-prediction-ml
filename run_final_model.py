"""Select and save the Phase 15 final research model."""
from pathlib import Path
from src.final_model import run_final_model

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    metadata = run_final_model(
        ROOT / "data/processed/cleaned_dataset.csv",
        ROOT / "results/metrics/preprocessing_split.json",
        ROOT / "results/tables/ablation.csv",
        ROOT / "models/final_model.joblib",
        ROOT / "models/feature_selector.joblib",
        ROOT / "results/metrics/final_model.json",
    )
    print(f"Selected Stage {metadata['selected_stage']}: {metadata['selection_rationale']}")
    print(f"Training rows: {metadata['training_rows']}; holdout not used: {metadata['holdout_rows_not_used']}")
    print(f"Selected features ({metadata['selected_feature_count']}):")
    for name in metadata["selected_transformed_features"]:
        print(f"  - {name}")
    print(f"Saved raw-input model: {metadata['model_artifact']}")
