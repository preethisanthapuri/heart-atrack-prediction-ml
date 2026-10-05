"""Generate training-partition SHAP explanations for the saved baseline."""
from pathlib import Path

from src.explainability import run_explainability

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    result = run_explainability(
        ROOT / "data/processed/cleaned_dataset.csv",
        ROOT / "models/preprocessing_pipeline.joblib",
        ROOT / "models/baselines/logistic_regression.joblib",
        ROOT / "results/metrics/preprocessing_split.json",
        ROOT / "results/figures/shap",
        ROOT / "results/tables",
        ROOT / "results/metrics/shap_local_prediction.json",
    )
    print("Global mean absolute SHAP importance (log-odds):")
    print(result["global_importance"].head(10).to_string(index=False))
    print("\nSelected training-row explanation metadata:")
    print(result["local_prediction"])
    print("\nSaved figures:")
    for path in result["figure_paths"]:
        print(path)
