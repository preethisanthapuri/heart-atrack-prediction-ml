"""SHAP explanations for the transparent Logistic Regression baseline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.preprocessing import FEATURES, TARGET_COLUMN

RANDOM_SEED = 42


def compute_shap_explanations(
    estimator: Any,
    transformed_training_data: Any,
    feature_names: list[str] | np.ndarray,
    *,
    background_size: int = 100,
    random_state: int = RANDOM_SEED,
) -> Any:
    """Compute exact linear-model SHAP values in log-odds space."""
    import shap

    values = np.asarray(transformed_training_data, dtype=float)
    names = list(feature_names)
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("Feature matrix and feature_names dimensions do not match")
    if not np.isfinite(values).all():
        raise ValueError("Transformed data must contain only finite values")
    if background_size < 1:
        raise ValueError("background_size must be positive")
    if not hasattr(estimator, "coef_") or not hasattr(estimator, "decision_function"):
        raise TypeError("This SHAP routine requires a fitted linear classifier")
    if len(estimator.classes_) != 2 or set(estimator.classes_) != {0, 1}:
        raise ValueError("Expected a fitted binary classifier with classes 0 and 1")

    rng = np.random.default_rng(random_state)
    count = min(background_size, len(values))
    background_indices = np.sort(rng.choice(len(values), size=count, replace=False))
    background = values[background_indices]
    explainer = shap.LinearExplainer(estimator, background)
    explanation = explainer(values)
    explanation.feature_names = names
    if explanation.values.shape != values.shape:
        raise RuntimeError(f"Unexpected SHAP output shape: {explanation.values.shape}")
    return explanation


def summarize_shap_values(
    explanation: Any,
    transformed_data: Any,
    feature_names: list[str] | np.ndarray,
    *,
    local_row: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create model-level mean-absolute and selected-row SHAP contribution tables."""
    values = np.asarray(explanation.values, dtype=float)
    data = np.asarray(transformed_data, dtype=float)
    names = list(feature_names)
    if values.shape != data.shape or values.shape[1] != len(names):
        raise ValueError("SHAP values, transformed data, and feature names must align")
    if not 0 <= local_row < len(data):
        raise IndexError("local_row is outside the explanation data")
    global_table = pd.DataFrame({
        "feature": names,
        "mean_absolute_shap_log_odds": np.mean(np.abs(values), axis=0),
        "mean_shap_log_odds": np.mean(values, axis=0),
    }).sort_values("mean_absolute_shap_log_odds", ascending=False, ignore_index=True)
    local_table = pd.DataFrame({
        "feature": names,
        "transformed_feature_value": data[local_row],
        "shap_value_log_odds": values[local_row],
    })
    local_table["absolute_shap_value"] = np.abs(local_table["shap_value_log_odds"])
    local_table = local_table.sort_values("absolute_shap_value", ascending=False, ignore_index=True)
    return global_table, local_table


def _write_shap_plots(explanation: Any, output_dir: str | Path, local_row: int) -> list[Path]:
    import shap

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    shap.summary_plot(explanation.values, explanation.data, feature_names=explanation.feature_names,
                      show=False, max_display=20)
    fig = plt.gcf()
    fig.set_size_inches(10, 7)
    path = output_dir / "shap_summary.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(path)

    shap.summary_plot(explanation.values, explanation.data, feature_names=explanation.feature_names,
                      plot_type="bar", show=False, max_display=20)
    fig = plt.gcf()
    fig.set_size_inches(9, 7)
    path = output_dir / "shap_feature_importance.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(path)

    local = explanation[local_row]
    shap.plots.waterfall(local, max_display=15, show=False)
    fig = plt.gcf()
    fig.set_size_inches(10, 7)
    path = output_dir / "shap_local_waterfall.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(path)
    return output_paths


def run_explainability(
    cleaned_data_path: str | Path,
    preprocessing_path: str | Path,
    model_path: str | Path,
    split_metadata_path: str | Path,
    figures_dir: str | Path,
    tables_dir: str | Path,
    local_prediction_path: str | Path,
    *,
    background_size: int = 100,
    random_state: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Explain the saved Phase 6 Logistic Regression using training rows only."""
    frame = pd.read_csv(cleaned_data_path)
    with Path(split_metadata_path).open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("fitted_on") != "training split only":
        raise ValueError("Saved preprocessing metadata is not marked training-only")
    train_indices = metadata["train_indices"]
    X_train = frame.loc[train_indices, FEATURES]

    preprocessing = joblib.load(preprocessing_path)
    estimator = joblib.load(model_path)
    transformed = preprocessing.transform(X_train)
    feature_names = preprocessing.get_feature_names_out().tolist()
    if not np.isfinite(transformed).all():
        raise ValueError("Preprocessing produced non-finite values")
    probabilities = estimator.predict_proba(transformed)[:, list(estimator.classes_).index(1)]
    # Explain a deterministic training example closest to the training-set median score.
    local_row = int(np.argsort(np.abs(probabilities - np.median(probabilities)), kind="stable")[0])
    explanation = compute_shap_explanations(
        estimator,
        transformed,
        feature_names,
        background_size=background_size,
        random_state=random_state,
    )
    global_table, local_table = summarize_shap_values(
        explanation,
        transformed,
        feature_names,
        local_row=local_row,
    )
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    global_path = tables_dir / "shap_global_importance.csv"
    local_path = tables_dir / "shap_local_explanation.csv"
    global_table.to_csv(global_path, index=False)
    local_table.to_csv(local_path, index=False)

    local_prediction = {
        "model": "Phase 6 Logistic Regression baseline",
        "target_column": TARGET_COLUMN,
        "explained_target_class": 1,
        "training_row_position": local_row,
        "training_dataset_index": int(train_indices[local_row]),
        "class1_probability": float(probabilities[local_row]),
        "predicted_class_at_0_5": int(probabilities[local_row] >= 0.5),
        "explainer": "SHAP LinearExplainer",
        "shap_output_scale": "log-odds",
        "background_rows": min(background_size, len(transformed)),
        "training_rows_explained": len(transformed),
        "interpretation_note": "SHAP values describe this fitted model's output contributions; they do not establish causal or clinical effects.",
    }
    local_prediction_path = Path(local_prediction_path)
    local_prediction_path.parent.mkdir(parents=True, exist_ok=True)
    local_prediction_path.write_text(json.dumps(local_prediction, indent=2), encoding="utf-8")
    figure_paths = _write_shap_plots(explanation, figures_dir, local_row)
    return {
        "global_importance": global_table,
        "local_explanation": local_table,
        "local_prediction": local_prediction,
        "figure_paths": figure_paths,
        "global_table_path": global_path,
        "local_table_path": local_path,
        "local_prediction_path": local_prediction_path,
    }
