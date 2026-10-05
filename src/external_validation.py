"""Independent-site validation for the fixed Cleveland-trained baseline model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.preprocessing import FEATURES, TARGET_COLUMN

UCI_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
RENAME = {
    "trestbps": "trtbps", "thalach": "thalachh", "exang": "exng",
    "slope": "slp", "ca": "caa", "thal": "thall", "num": TARGET_COLUMN,
}


def load_and_map_uci_site(path: str | Path) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Load one processed UCI site and map published codes to the project schema.

    The class mapping follows the UCI definition: 0 means no angiographic
    disease; values 1--4 indicate presence and are collapsed to positive.
    Chest-pain and ST-slope codes are shifted to match this project's
    zero-based Cleveland-derived encoding. Thal codes use the UCI codebook.
    """
    path = Path(path)
    raw = pd.read_csv(path, header=None, names=UCI_COLUMNS, na_values="?", dtype=float)
    if raw.shape[1] != len(UCI_COLUMNS):
        raise ValueError(f"Expected {len(UCI_COLUMNS)} source columns; got {raw.shape[1]}")
    if raw["num"].isna().any():
        raw = raw.loc[raw["num"].notna()].copy()
    observed_labels = set(raw["num"].dropna().astype(int).unique())
    if not observed_labels <= {0, 1, 2, 3, 4}:
        raise ValueError(f"Unexpected UCI target codes: {sorted(observed_labels)}")

    frame = raw.rename(columns=RENAME).copy()
    source_missing = raw.isna().sum()
    # These recodings reproduce the values in the supplied Cleveland-derived
    # file; they are not the conventional UCI integer encodings. Their mapping
    # was checked against the overlapping Cleveland site records before scoring.
    frame["cp"] = frame["cp"].map({1.0: 3.0, 2.0: 1.0, 3.0: 2.0, 4.0: 0.0})
    frame["restecg"] = frame["restecg"].map({0.0: 1.0, 1.0: 2.0, 2.0: 0.0})
    frame["slp"] = frame["slp"].map({1.0: 2.0, 2.0: 1.0, 3.0: 0.0})
    # The supplied data encodes the source's missing ca/thal markers as 4/0.
    frame["caa"] = frame["caa"].fillna(4.0)
    frame["thall"] = frame["thall"].map({3.0: 2.0, 6.0: 1.0, 7.0: 3.0}).fillna(0.0)
    # Row-level comparison to the official Cleveland cohort showed the supplied
    # `output` orientation is reversed relative to UCI `num`: num=0 -> output=1.
    y = (frame.pop(TARGET_COLUMN) == 0).astype(int).rename(TARGET_COLUMN)
    X = frame[FEATURES].copy()
    details = {
        "source_file": path.name,
        "source_rows": int(len(raw)),
        "source_columns": UCI_COLUMNS,
        "target_source_column": "num",
        "source_target_counts": {str(int(k)): int(v) for k, v in raw["num"].value_counts().sort_index().items()},
        "binary_target_counts": {str(int(k)): int(v) for k, v in y.value_counts().sort_index().items()},
        "source_missing_feature_values": {
            RENAME.get(k, k): int(v) for k, v in source_missing.drop("num").items()
        },
        "source_feature_rows_with_any_missing": int(raw.drop(columns="num").isna().any(axis=1).sum()),
        "missing_feature_values_after_source_sentinel_mapping": {k: int(v) for k, v in X.isna().sum().items()},
        "feature_rows_with_any_missing": int(X.isna().any(axis=1).sum()),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "category_mapping": {
            "cp": "UCI 1,2,3,4 -> project cp 3,1,2,0; empirically matched to supplied Cleveland-derived encoding",
            "restecg": "UCI 0,1,2 -> project restecg 1,2,0; empirically matched to supplied Cleveland-derived encoding",
            "slp": "UCI 1,2,3 -> project slp 2,1,0; empirically matched to supplied Cleveland-derived encoding",
            "caa": "UCI ca 0-3 preserved; missing mapped to project sentinel 4",
            "thall": "UCI thal 3,6,7 -> project thall 2,1,3; missing mapped to project sentinel 0",
            "output": "UCI num 0 -> project output 1; UCI num 1-4 -> project output 0; empirically matched to supplied Cleveland-derived target orientation",
        },
    }
    if list(X.columns) != FEATURES:
        raise RuntimeError("Mapped external feature order differs from the trained model schema")
    return X, y, details


def evaluate_external_site(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessing_path: str | Path,
    model_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score a fixed saved model on an external cohort without refitting anything."""
    if list(X.columns) != FEATURES:
        raise ValueError("External feature schema/order does not match the trained model")
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError("External target must contain both binary classes with no missing labels")
    preprocessor = joblib.load(preprocessing_path)
    model = joblib.load(model_path)
    transformed = preprocessor.transform(X)
    if not np.isfinite(transformed).all():
        raise ValueError("Saved preprocessing did not resolve all external missing values")
    probabilities = model.predict_proba(transformed)[:, list(model.classes_).index(1)]
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    metrics = {
        "n": int(len(y)),
        "class_0_n": int((y == 0).sum()),
        "class_1_n": int((y == 1).sum()),
        "accuracy": float(accuracy_score(y, predictions)),
        "precision_class_1": float(precision_score(y, predictions, zero_division=0)),
        "sensitivity_class_1": float(recall_score(y, predictions, zero_division=0)),
        "specificity_class_0": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "f1_class_1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "pr_auc_average_precision": float(average_precision_score(y, probabilities)),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "classification_threshold": 0.5,
        "threshold_note": "Fixed conventional threshold for descriptive class metrics; not optimized or clinically selected.",
    }
    return pd.DataFrame([metrics]), metrics


def run_external_validation(
    external_path: str | Path,
    preprocessing_path: str | Path,
    model_path: str | Path,
    table_path: str | Path,
    report_path: str | Path,
    *,
    site_name: str = "Hungary",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    X, y, details = load_and_map_uci_site(external_path)
    table, metrics = evaluate_external_site(X, y, preprocessing_path, model_path)
    table.insert(0, "external_site", site_name)
    table_path = Path(table_path)
    report_path = Path(report_path)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)
    report = {
        "project_name": "heart-atrack-prediction-ml",
        "validation_type": "independent-site external validation",
        "external_site": site_name,
        "model": "saved Phase 6 Logistic Regression baseline; no refitting or recalibration",
        "preprocessing": "saved Phase 4 pipeline fitted only on the primary training split",
        "primary_dataset_source_identity": "not independently confirmed; external cohort selected as a different UCI collection site, Hungary",
        "external_dataset": details,
        "metrics": metrics,
        "clinical_limitations": [
            "The endpoint is UCI angiographic heart-disease status, not acute heart attack or prospective cardiovascular risk.",
            "The primary CSV provenance and equivalence of measurement protocols are not confirmed.",
            "This is a historical external cohort from the same UCI dataset family, not contemporary clinical validation.",
            "Missing ca/thal values use source-compatible sentinels; remaining missing values use primary-training imputation. Missingness may be informative and cohort-dependent.",
            "The supplied output orientation appears reversed relative to UCI num (output=1 aligns with num=0), with a one-record class-count discrepancy; clinical class interpretation requires provider confirmation.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return table, report
