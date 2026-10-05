"""Run Phase 14 staged ablation and descriptive subgroup evaluation."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from src.ablation import run_ablation_experiment
from src.fairness import subgroup_fairness_tables
from src.preprocessing import FEATURES, TARGET_COLUMN

ROOT = Path(__file__).resolve().parent


def main() -> None:
    frame = pd.read_csv(ROOT / "data/processed/cleaned_dataset.csv")
    metadata = json.loads((ROOT / "results/metrics/preprocessing_split.json").read_text(encoding="utf-8"))
    if metadata.get("fitted_on") != "training split only":
        raise ValueError("Saved split metadata does not confirm training-only fitting")
    train_indices, test_indices = metadata["train_indices"], metadata["test_indices"]
    if set(train_indices) & set(test_indices) or len(train_indices) + len(test_indices) != len(frame):
        raise ValueError("Saved split indices overlap or do not cover the dataset")
    X_train = frame.loc[train_indices, FEATURES]
    y_train = frame.loc[train_indices, TARGET_COLUMN]
    summary, folds, oof = run_ablation_experiment(X_train, y_train)
    fairness, disparity = subgroup_fairness_tables(frame, train_indices, oof)
    tables = ROOT / "results/tables"
    metrics_dir = ROOT / "results/metrics"
    tables.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables / "ablation.csv", index=False)
    folds.to_csv(tables / "ablation_folds.csv", index=False)
    fairness.to_csv(tables / "fairness.csv", index=False)
    disparity.to_csv(tables / "fairness_disparity.csv", index=False)
    payload = {"training_rows": len(train_indices), "holdout_rows_not_used": len(test_indices),
               "outer_cv_folds": 5, "inner_tuning_folds": 3,
               "target_semantics": "numeric dataset classes only; clinical direction unverified",
               "ablation": summary.replace({float("nan"): None}).to_dict(orient="records"),
               "fairness_scope": "descriptive training-only out-of-fold subgroup metrics"}
    (metrics_dir / "phase14.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print("Ablation summary (pooled OOF metrics):")
    print(summary[["stage", "accuracy", "sensitivity", "specificity", "f1", "roc_auc", "pr_auc", "brier"]].to_string(index=False))
    print("\nSubgroup metrics:")
    print(fairness.to_string(index=False))
    print("\nDisparity ranges (descriptive):")
    print(disparity.to_string(index=False))


if __name__ == "__main__":
    main()
