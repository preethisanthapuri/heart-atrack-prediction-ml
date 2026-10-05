"""Leakage-safe preprocessing and reproducible stratified splitting."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_NAME = "heart-atrack-prediction-ml"
TARGET_COLUMN = "output"
NUMERIC_FEATURES = ["age", "trtbps", "chol", "thalachh", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exng", "slp", "caa", "thall"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessing_pipeline() -> Pipeline:
    """Build an unfitted transformer pipeline for the discovered feature schema."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("preprocessor", transformer)])


def _validate_dataset(frame: pd.DataFrame) -> None:
    expected = set(FEATURES + [TARGET_COLUMN])
    missing = expected - set(frame.columns)
    unexpected = set(frame.columns) - expected
    if missing or unexpected:
        raise ValueError(f"Unexpected preprocessing schema: missing={sorted(missing)}, extra={sorted(unexpected)}")
    if frame[TARGET_COLUMN].isna().any():
        raise ValueError(f"Target {TARGET_COLUMN!r} cannot contain missing values")
    if frame[TARGET_COLUMN].nunique() != 2:
        raise ValueError("Stratified binary preprocessing expects exactly two target classes")


def fit_and_save_preprocessing(
    frame: pd.DataFrame,
    model_path: str | Path,
    metadata_path: str | Path,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Split, fit preprocessing on training rows only, and save the fitted pipeline."""
    _validate_dataset(frame)
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    X = frame[FEATURES].copy()
    y = frame[TARGET_COLUMN].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = build_preprocessing_pipeline()
    pipeline.fit(X_train)
    # Transform both splits after fitting to verify deployment-compatible output.
    X_train_processed = pipeline.transform(X_train)
    X_test_processed = pipeline.transform(X_test)
    if not np.isfinite(X_train_processed).all() or not np.isfinite(X_test_processed).all():
        raise ValueError("Preprocessing produced non-finite transformed values")
    if X_train_processed.shape[1] != X_test_processed.shape[1]:
        raise RuntimeError("Train and test transforms produced inconsistent feature dimensions")

    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    names = pipeline.get_feature_names_out().tolist()
    metadata: dict[str, Any] = {
        "project_name": PROJECT_NAME,
        "target_column": TARGET_COLUMN,
        "source_rows": int(len(frame)),
        "feature_count_before_encoding": len(FEATURES),
        "transformed_feature_count": len(names),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "transformed_feature_names": names,
        "test_size_requested": test_size,
        "random_state": random_state,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_class_counts": {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
        "test_class_counts": {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()},
        "train_indices": X_train.index.astype(int).tolist(),
        "test_indices": X_test.index.astype(int).tolist(),
        "fitted_on": "training split only",
        "pipeline_path": str(model_path),
        "preprocessing_steps": {
            "numeric": ["median imputation", "standard scaling"],
            "categorical": ["most-frequent imputation", "one-hot encoding; unknown categories ignored"],
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def run_preprocessing(
    input_path: str | Path,
    model_path: str | Path,
    metadata_path: str | Path,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Load the cleaned CSV and create the saved training-fitted preprocessor."""
    frame = pd.read_csv(input_path)
    return fit_and_save_preprocessing(
        frame, model_path, metadata_path, test_size=test_size, random_state=random_state
    )
