"""Preliminary holdout screening for Phase 5 baseline models.

Comprehensive classification metrics, cross-validation, curves and confusion
matrices are reserved for Phase 6.
"""
from __future__ import annotations
from pathlib import Path
from time import perf_counter
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from src.preprocessing import FEATURES, TARGET_COLUMN


def screen_baseline_models(
    frame: pd.DataFrame,
    split_metadata: dict[str, Any],
    preprocessing_pipeline: Any,
    model_factories: dict[str, Any],
    output_dir: str | Path,
) -> pd.DataFrame:
    """Fit candidate baselines on the saved training split and record holdout accuracy."""
    if split_metadata.get("fitted_on") != "training split only":
        raise ValueError("Preprocessing artifact is not documented as training-split-only")
    train_indices = split_metadata["train_indices"]
    test_indices = split_metadata["test_indices"]
    if set(train_indices) & set(test_indices):
        raise ValueError("Training and test indices overlap")
    if len(train_indices) + len(test_indices) != len(frame):
        raise ValueError("Saved split indices do not cover this dataset")
    X = frame[FEATURES]
    y = frame[TARGET_COLUMN]
    X_train, X_test = X.loc[train_indices], X.loc[test_indices]
    y_train, y_test = y.loc[train_indices], y.loc[test_indices]

    X_train_transformed = preprocessing_pipeline.transform(X_train)
    X_test_transformed = preprocessing_pipeline.transform(X_test)
    if not np.isfinite(X_train_transformed).all() or not np.isfinite(X_test_transformed).all():
        raise ValueError("The saved preprocessing pipeline produced non-finite values")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, model in model_factories.items():
        started = perf_counter()
        model.fit(X_train_transformed, y_train)
        fit_seconds = perf_counter() - started
        predictions = model.predict(X_test_transformed)
        model_file = output_dir / f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')}.joblib"
        joblib.dump(model, model_file)
        rows.append({
            "model": name,
            "holdout_accuracy": float(accuracy_score(y_test, predictions)),
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
            "predicted_output_0": int(np.sum(predictions == 0)),
            "predicted_output_1": int(np.sum(predictions == 1)),
            "fit_seconds": float(fit_seconds),
            "model_artifact": str(model_file),
            "status": "completed",
        })
    results = pd.DataFrame(rows).sort_values(
        ["holdout_accuracy", "model"], ascending=[False, True], ignore_index=True
    )
    return results


def run_baseline_screening(
    cleaned_data_path: str | Path,
    preprocessing_path: str | Path,
    split_metadata_path: str | Path,
    model_output_dir: str | Path,
    results_path: str | Path,
) -> pd.DataFrame:
    """Load Phase 2/4 artifacts, train baselines, and save aggregate screening results."""
    frame = pd.read_csv(cleaned_data_path)
    metadata = pd.read_json(split_metadata_path, typ="series").to_dict()
    pipeline = joblib.load(preprocessing_path)
    from src.models import build_baseline_models
    results = screen_baseline_models(frame, metadata, pipeline, build_baseline_models(), model_output_dir)
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)
    return results
