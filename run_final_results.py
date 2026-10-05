"""Validate prior experiment artifacts and assemble the Phase 17 final results."""
from pathlib import Path
from src.final_results import create_final_results

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    report = create_final_results(ROOT)
    print("Final-results package assembled from recorded outputs.")
    print(f"Selected candidate: Stage {report['selected_candidate']['stage']} — {report['selected_candidate']['model']}")
    print(f"Training rows: {report['evaluation_boundaries']['training_cv_rows']}; saved holdout rows: {report['evaluation_boundaries']['saved_holdout_rows']}")
    print("Pooled training OOF metrics:")
    for metric, value in report["selected_candidate"]["pooled_training_oof_metrics"].items():
        print(f"  {metric}: {value:.4f}")
    print("Figures:")
    for path in report["figures"]:
        print(f"  {path}")
