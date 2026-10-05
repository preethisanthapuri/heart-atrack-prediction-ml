"""Leakage-safe probability calibration experiments for Phase 10."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.evaluation import build_model_pipeline
from src.models import build_baseline_models
from src.preprocessing import FEATURES, TARGET_COLUMN

RANDOM_SEED = 42
CALIBRATION_METHODS = ("uncalibrated", "sigmoid", "isotonic")
BASE_MODELS = ("Logistic Regression", "K-Nearest Neighbors", "Random Forest", "Soft Voting")


def _base_estimators() -> dict[str, Any]:
    baselines = build_baseline_models()
    base = {
        "Logistic Regression": build_model_pipeline(baselines["Logistic Regression"]),
        "K-Nearest Neighbors": build_model_pipeline(baselines["K-Nearest Neighbors"]),
        "Random Forest": build_model_pipeline(baselines["Random Forest"]),
    }
    vote = VotingClassifier(
        estimators=[
            ("logistic_regression", base["Logistic Regression"]),
            ("knn", base["K-Nearest Neighbors"]),
            ("random_forest", base["Random Forest"]),
        ],
        voting="soft",
        n_jobs=1,
    )
    return {**base, "Soft Voting": vote}


def _fit_variant(
    estimator: Any,
    method: str,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    *,
    inner_folds: int,
    random_state: int,
) -> Any:
    if method == "uncalibrated":
        fitted = estimator
    else:
        inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=random_state)
        fitted = CalibratedClassifierCV(
            estimator=estimator,
            method=method,
            cv=inner_cv,
            ensemble=True,
        )
    fitted.fit(X_fit, y_fit)
    return fitted


def _positive_probability(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    probabilities = estimator.predict_proba(X)
    positive_index = list(estimator.classes_).index(1)
    return np.asarray(probabilities[:, positive_index], dtype=float)


def _curve_table(y_true: np.ndarray, probability: np.ndarray, model: str, method: str, n_bins: int) -> list[dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(probability, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for index in range(n_bins):
        selected = bin_ids == index
        if selected.any():
            rows.append({
                "model": model,
                "method": method,
                "bin": index + 1,
                "n": int(selected.sum()),
                "mean_predicted_probability": float(probability[selected].mean()),
                "observed_fraction_class1": float(y_true[selected].mean()),
            })
    return rows


def _plot_calibration_curves(curves: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    axes = axes.ravel()
    styles = {"uncalibrated": "o-", "sigmoid": "s-", "isotonic": "^-"}
    for ax, model in zip(axes, BASE_MODELS):
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Ideal")
        subset = curves.loc[curves["model"] == model]
        for method in CALIBRATION_METHODS:
            points = subset.loc[subset["method"] == method].sort_values("mean_predicted_probability")
            if len(points):
                ax.plot(
                    points["mean_predicted_probability"],
                    points["observed_fraction_class1"],
                    styles[method],
                    label=method.capitalize(),
                    markersize=4,
                )
        ax.set(title=model, xlabel="Mean predicted probability", ylabel="Observed class-1 fraction", xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize="small", loc="upper left")
    fig.suptitle("Out-of-fold probability calibration (training partition)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_calibration_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_splits: int = 5,
    inner_folds: int = 3,
    random_state: int = RANDOM_SEED,
    n_bins: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Obtain outer-fold out-of-fold probabilities for raw and calibrated candidates.

    The outer validation fold is never used to fit the base estimator or its calibrator.
    """
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train lengths must match")
    if set(pd.unique(y_train)) != {0, 1}:
        raise ValueError("Calibration expects target classes 0 and 1")
    if n_splits < 2 or inner_folds < 2 or n_bins < 2:
        raise ValueError("n_splits, inner_folds, and n_bins must each be at least 2")
    if y_train.value_counts().min() < n_splits:
        raise ValueError("Each class must have at least n_splits observations")

    candidates = _base_estimators()
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    probabilities: dict[tuple[str, str], np.ndarray] = {
        (model, method): np.full(len(y_train), np.nan, dtype=float)
        for model in BASE_MODELS for method in CALIBRATION_METHODS
    }
    fold_scores: dict[tuple[str, str], list[dict[str, float]]] = {
        key: [] for key in probabilities
    }
    for fit_positions, validation_positions in splitter.split(X_train, y_train):
        X_fit, X_validation = X_train.iloc[fit_positions], X_train.iloc[validation_positions]
        y_fit, y_validation = y_train.iloc[fit_positions], y_train.iloc[validation_positions]
        if y_fit.value_counts().min() < inner_folds:
            raise ValueError("Outer training fold has too few class members for inner calibration CV")
        for model_name, base_estimator in candidates.items():
            for method in CALIBRATION_METHODS:
                fitted = _fit_variant(
                    base_estimator,
                    method,
                    X_fit,
                    y_fit,
                    inner_folds=inner_folds,
                    random_state=random_state,
                )
                positive_probability = _positive_probability(fitted, X_validation)
                probabilities[(model_name, method)][validation_positions] = positive_probability
                fold_scores[(model_name, method)].append({
                    "brier_score": float(brier_score_loss(y_validation, positive_probability)),
                    "roc_auc": float(roc_auc_score(y_validation, positive_probability)),
                })

    summary_rows = []
    curve_rows: list[dict[str, Any]] = []
    y_array = y_train.to_numpy(dtype=int)
    for (model_name, method), probability in probabilities.items():
        if not np.isfinite(probability).all():
            raise RuntimeError(f"Missing out-of-fold probabilities for {model_name}/{method}")
        scores = fold_scores[(model_name, method)]
        brier_values = np.asarray([score["brier_score"] for score in scores])
        auc_values = np.asarray([score["roc_auc"] for score in scores])
        summary_rows.append({
            "model": model_name,
            "method": method,
            "outer_folds": n_splits,
            "inner_calibration_folds": inner_folds if method != "uncalibrated" else 0,
            "brier_mean": float(brier_values.mean()),
            "brier_std": float(brier_values.std(ddof=1)),
            "brier_oof_pooled": float(brier_score_loss(y_array, probability)),
            "roc_auc_mean": float(auc_values.mean()),
            "roc_auc_std": float(auc_values.std(ddof=1)),
            "roc_auc_oof_pooled": float(roc_auc_score(y_array, probability)),
            "class1_prevalence": float(y_array.mean()),
        })
        curve_rows.extend(_curve_table(y_array, probability, model_name, method, n_bins))

    return pd.DataFrame(summary_rows), pd.DataFrame(curve_rows)


def run_calibration(
    cleaned_data_path: str | Path,
    split_metadata_path: str | Path,
    summary_path: str | Path,
    curve_path: str | Path,
    figure_path: str | Path,
    *,
    n_splits: int = 5,
    inner_folds: int = 3,
    random_state: int = RANDOM_SEED,
    n_bins: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Load only training rows; save calibration summaries, curve points, and figure."""
    import json

    frame = pd.read_csv(cleaned_data_path)
    with Path(split_metadata_path).open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("fitted_on") != "training split only":
        raise ValueError("Saved split metadata is not marked training-only")
    train_indices = metadata["train_indices"]
    summary, curves = run_calibration_experiment(
        frame.loc[train_indices, FEATURES],
        frame.loc[train_indices, TARGET_COLUMN],
        n_splits=n_splits,
        inner_folds=inner_folds,
        random_state=random_state,
        n_bins=n_bins,
    )
    summary_output, curve_output = Path(summary_path), Path(curve_path)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    curve_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_output, index=False)
    curves.to_csv(curve_output, index=False)
    figure = _plot_calibration_curves(curves, figure_path)
    return summary, curves, figure
