"""Run Phase 4 preprocessing on the cleaned dataset."""
from pathlib import Path
from src.preprocessing import run_preprocessing

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    metadata = run_preprocessing(
        ROOT / "data" / "processed" / "cleaned_dataset.csv",
        ROOT / "models" / "preprocessing_pipeline.joblib",
        ROOT / "results" / "metrics" / "preprocessing_split.json",
    )
    print(f"Train rows: {metadata['train_rows']}; test rows: {metadata['test_rows']}")
    print(f"Training class counts: {metadata['train_class_counts']}")
    print(f"Test class counts: {metadata['test_class_counts']}")
    print(f"Transformed features: {metadata['transformed_feature_count']}")
    print(f"Saved fitted preprocessing pipeline to {metadata['pipeline_path']}")
