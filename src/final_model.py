"""Evidence-based Phase 15 final model selection and training-partition fit."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from functools import partial

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing import FEATURES, TARGET_COLUMN, build_preprocessing_pipeline

SEED = 42
SELECTED_STAGE = "B"
SELECTED_FEATURES = 15
SELECTION_RATIONALE = (
    "Stage B (fold-local mutual-information SelectKBest, k=15, Logistic Regression) was selected "
    "as the probability-oriented research model. In Phase 14 pooled training OOF results it had "
    "the highest ROC-AUC (0.9163) and average precision (0.9221), and lowest Brier score (0.1123). "
    "Stage A had higher accuracy/F1; Stage F was weaker on ranking and Brier. This is a tradeoff, "
    "not evidence of clinical superiority."
)


def build_final_pipeline(selection_size: int = SELECTED_FEATURES) -> Pipeline:
    """Build a raw-schema to class-probability pipeline; all fit steps are train-local."""
    return Pipeline([
        ("preprocessor", build_preprocessing_pipeline().named_steps["preprocessor"]),
        ("selector", SelectKBest(
            score_func=partial(mutual_info_classif, random_state=SEED),
            k=selection_size,
        )),
        ("classifier", LogisticRegression(max_iter=2000, random_state=SEED)),
    ])


def fit_final_model(
    frame: pd.DataFrame,
    split_metadata: dict[str, Any],
    *,
    model_path: str | Path,
    selector_path: str | Path,
    metadata_path: str | Path,
    ablation_results_path: str | Path,
) -> dict[str, Any]:
    """Select from recorded CV evidence and fit artifacts on saved training rows only.

    Holdout rows are validated against split metadata and never accessed for fitting,
    model selection, calibration, or threshold setting.
    """
    if split_metadata.get("fitted_on") != "training split only":
        raise ValueError("Saved split metadata must document training-only preprocessing")
    train_indices = [int(index) for index in split_metadata["train_indices"]]
    test_indices = [int(index) for index in split_metadata["test_indices"]]
    if set(train_indices) & set(test_indices):
        raise ValueError("Training and holdout row indices overlap")
    if set(train_indices) | set(test_indices) != set(map(int, frame.index)):
        raise ValueError("Saved split indices do not exactly cover the source frame")
    if not set(FEATURES + [TARGET_COLUMN]).issubset(frame.columns):
        raise ValueError("Cleaned data is missing required model columns")
    y = frame.loc[train_indices, TARGET_COLUMN]
    if set(y.unique()) != {0, 1}:
        raise ValueError("Final model expects target classes 0 and 1")

    ablation = pd.read_csv(ablation_results_path)
    if SELECTED_STAGE not in set(ablation["stage"]):
        raise ValueError("Phase 14 results do not contain the selected Stage B")
    row_b = ablation.loc[ablation["stage"] == SELECTED_STAGE].iloc[0]
    pipeline = build_final_pipeline()
    pipeline.fit(frame.loc[train_indices, FEATURES], y)
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    support = pipeline.named_steps["selector"].get_support()
    selected_names = feature_names[support].astype(str).tolist()
    if len(selected_names) != SELECTED_FEATURES:
        raise RuntimeError(f"Expected {SELECTED_FEATURES} selected features, got {len(selected_names)}")

    model_path, selector_path, metadata_path = map(Path, (model_path, selector_path, metadata_path))
    for path in (model_path, selector_path, metadata_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    joblib.dump(pipeline.named_steps["selector"], selector_path)
    metadata = {
        "project_name": "heart-atrack-prediction-ml",
        "selected_stage": SELECTED_STAGE,
        "selection_rationale": SELECTION_RATIONALE,
        "training_rows": len(train_indices),
        "holdout_rows_not_used": len(test_indices),
        "fitted_on": "saved primary training partition only",
        "target_column": TARGET_COLUMN,
        "target_classes": [0, 1],
        "positive_probability_column": "class 1 (dataset code; clinical meaning unverified)",
        "classification_cutoff": None,
        "threshold_note": "No clinical threshold selected; downstream users receive class probabilities.",
        "calibration": "No post-hoc calibrator selected; Phase 10 did not show a consistent benefit and Phase 14 did not evaluate calibration specifically for Stage B.",
        "uncertainty": "No Stage-B-specific uncertainty method was established; Phase 11 results apply to soft voting only.",
        "selected_feature_count": len(selected_names),
        "selected_transformed_features": selected_names,
        "pipeline_steps": ["median/mode imputation, standard scaling/one-hot encoding", "mutual-information SelectKBest(k=15)", "Logistic Regression (C=1.0, max_iter=2000)"],
        "phase14_stage_b_pooled_training_oof": {
            metric: float(row_b[metric]) for metric in
            ["accuracy", "precision", "sensitivity", "specificity", "f1", "roc_auc", "pr_auc", "brier", "log_loss"]
        },
        "model_artifact": "models/final_model.joblib",
        "selector_artifact": "models/feature_selector.joblib",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    # Check the saved artifact's public raw-input prediction interface without scoring the holdout.
    loaded = joblib.load(model_path)
    train_probabilities = loaded.predict_proba(frame.loc[train_indices, FEATURES])
    if train_probabilities.shape != (len(train_indices), 2) or not np.isfinite(train_probabilities).all():
        raise RuntimeError("Saved final pipeline failed training-row probability interface validation")
    if not np.allclose(train_probabilities.sum(axis=1), 1.0):
        raise RuntimeError("Saved final pipeline probabilities do not sum to one")
    return metadata


def run_final_model(
    cleaned_data_path: str | Path,
    split_metadata_path: str | Path,
    ablation_results_path: str | Path,
    model_path: str | Path,
    selector_path: str | Path,
    metadata_path: str | Path,
) -> dict[str, Any]:
    frame = pd.read_csv(cleaned_data_path)
    metadata = json.loads(Path(split_metadata_path).read_text(encoding="utf-8"))
    return fit_final_model(
        frame, metadata, model_path=model_path, selector_path=selector_path,
        metadata_path=metadata_path, ablation_results_path=ablation_results_path,
    )
