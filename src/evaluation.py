"""Baseline screening and formal Phase 6 model evaluation."""
from __future__ import annotations
import json
from pathlib import Path
from time import perf_counter
from typing import Any
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from src.preprocessing import FEATURES, TARGET_COLUMN, build_preprocessing_pipeline
from src.models import build_baseline_models

CV_RANDOM_SEED = 42
CV_FOLDS = 5
METRIC_NAMES = ["accuracy", "precision", "sensitivity", "specificity", "f1", "roc_auc", "pr_auc"]


def specificity_score(y_true: Any, y_pred: Any) -> float:
    """Return class-0 true-negative rate; the target classes must be 0 and 1."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    denominator = matrix[0, 0] + matrix[0, 1]
    return float(matrix[0, 0] / denominator) if denominator else 0.0


def screen_baseline_models(
    frame: pd.DataFrame,
    split_metadata: dict[str, Any],
    preprocessing_pipeline: Any,
    model_factories: dict[str, Any],
    models_dir: str | Path,
) -> pd.DataFrame:
    """Phase 5 preliminary accuracy screen retained for backward compatibility."""
    if split_metadata.get("fitted_on") != "training split only":
        raise ValueError("Preprocessing artifact is not documented as training-split-only")
    train_indices, test_indices = split_metadata["train_indices"], split_metadata["test_indices"]
    if set(train_indices) & set(test_indices) or len(train_indices) + len(test_indices) != len(frame):
        raise ValueError("Saved split indices overlap or do not cover this dataset")
    X_train = preprocessing_pipeline.transform(frame.loc[train_indices, FEATURES])
    X_test = preprocessing_pipeline.transform(frame.loc[test_indices, FEATURES])
    y_train = frame.loc[train_indices, TARGET_COLUMN]
    y_test = frame.loc[test_indices, TARGET_COLUMN]
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, estimator in model_factories.items():
        fitted = clone(estimator)
        started = perf_counter()
        fitted.fit(X_train, y_train)
        elapsed = perf_counter() - started
        artifact = models_dir / f"{_slug(name)}.joblib"
        joblib.dump(fitted, artifact)
        rows.append({
            "model": name,
            "holdout_accuracy": float(accuracy_score(y_test, fitted.predict(X_test))),
            "fit_seconds": float(elapsed),
            "model_artifact": str(artifact),
            "status": "completed",
        })
    return pd.DataFrame(rows)


def run_baseline_screening(
    cleaned_data_path: str | Path,
    preprocessing_path: str | Path,
    split_metadata_path: str | Path,
    models_dir: str | Path,
    results_path: str | Path,
) -> pd.DataFrame:
    """Load Phase 2/4 artifacts and run the preliminary Phase 5 screen."""
    frame = pd.read_csv(cleaned_data_path)
    with Path(split_metadata_path).open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    preprocessing = joblib.load(preprocessing_path)
    results = screen_baseline_models(frame, metadata, preprocessing, build_baseline_models(), models_dir)
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)
    return results


def binary_classification_metrics(y_true: Any, y_pred: Any, y_score: Any) -> dict[str, float]:
    """Calculate standard binary metrics using class 1 as the positive label."""
    classes = set(np.unique(y_true).tolist())
    if classes != {0, 1}:
        raise ValueError(f"Expected target classes 0 and 1, received {sorted(classes)}")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": specificity_score(y_true, y_pred),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }


def build_model_pipeline(estimator: Any) -> Pipeline:
    """Build a fresh preprocessing+estimator pipeline for fold-local CV fitting."""
    return Pipeline([
        ("preprocessor", build_preprocessing_pipeline().named_steps["preprocessor"]),
        ("classifier", clone(estimator)),
    ])


def cross_validate_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_factories: dict[str, Any] | None = None,
    *,
    n_splits: int = CV_FOLDS,
    random_state: int = CV_RANDOM_SEED,
) -> pd.DataFrame:
    """Run stratified CV on raw training data, fitting preprocessing within each fold."""
    factories = model_factories or build_baseline_models()
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if y_train.value_counts().min() < n_splits:
        raise ValueError("Each class must have at least n_splits observations")
    scorer = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, pos_label=1, zero_division=0),
        "sensitivity": "recall",
        "specificity": make_scorer(specificity_score),
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []
    for name, estimator in factories.items():
        pipeline = build_model_pipeline(estimator)
        scores = cross_validate(
            pipeline, X_train, y_train, scoring=scorer, cv=cv,
            n_jobs=1, return_train_score=False, error_score="raise",
        )
        row: dict[str, Any] = {"model": name, "folds": n_splits}
        for metric in METRIC_NAMES:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def _positive_class_scores(estimator: Any, X: Any) -> tuple[np.ndarray, str]:
    """Get continuous scores oriented toward target class 1."""
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(X)
        positive_index = list(estimator.classes_).index(1)
        return probabilities[:, positive_index], "predict_proba"
    decision = np.asarray(estimator.decision_function(X))
    if decision.ndim == 2:
        positive_index = list(estimator.classes_).index(1)
        decision = decision[:, positive_index]
    elif list(estimator.classes_).index(1) == 0:
        decision = -decision
    return decision, "decision_function"


def _save_evaluation_figures(
    y_true: pd.Series,
    predictions: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
    metrics: dict[str, dict[str, float]],
    output_dir: str | Path,
) -> list[Path]:
    """Save one confusion matrix per model and aggregate ROC/PR curves."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    sns.set_theme(style="whitegrid", context="notebook")
    for name, predicted in predictions.items():
        matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, square=True,
                    xticklabels=["0", "1"], yticklabels=["0", "1"], ax=ax)
        ax.set(title=f"{name} — confusion matrix", xlabel="Predicted output class", ylabel="Observed output class")
        path = output_dir / f"confusion_matrix_{_slug(name)}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, score in scores.items():
        fpr, tpr, _ = roc_curve(y_true, score, pos_label=1)
        ax.plot(fpr, tpr, label=f"{name} (AUC={metrics[name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate", ylabel="True positive rate",
           title="ROC curves — holdout set")
    ax.legend(loc="lower right", fontsize="small")
    path = output_dir / "roc_curves.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, score in scores.items():
        precision, recall, _ = precision_recall_curve(y_true, score, pos_label=1)
        ax.plot(recall, precision, label=f"{name} (AP={metrics[name]['pr_auc']:.3f})")
    baseline = float(np.mean(y_true == 1))
    ax.axhline(baseline, linestyle="--", color="grey", label=f"Class-1 prevalence={baseline:.3f}")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall (class 1)", ylabel="Precision (class 1)",
           title="Precision-recall curves — holdout set")
    ax.legend(loc="lower left", fontsize="small")
    path = output_dir / "precision_recall_curves.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def _slug(value: str) -> str:
    return "_".join(value.lower().replace("(", "").replace(")", "").replace("-", " ").split())


def evaluate_models(
    frame: pd.DataFrame,
    split_metadata: dict[str, Any],
    preprocessing_pipeline: Any,
    model_factories: dict[str, Any] | None = None,
    *,
    cv_folds: int = CV_FOLDS,
    cv_random_state: int = CV_RANDOM_SEED,
    figures_dir: str | Path,
    tables_dir: str | Path,
) -> dict[str, Any]:
    """Evaluate baselines on the saved holdout and training-only stratified CV."""
    if split_metadata.get("fitted_on") != "training split only":
        raise ValueError("Preprocessing artifact is not documented as training-split-only")
    train_indices, test_indices = split_metadata["train_indices"], split_metadata["test_indices"]
    if set(train_indices) & set(test_indices) or len(train_indices) + len(test_indices) != len(frame):
        raise ValueError("Saved split indices overlap or do not cover this dataset")
    if set(frame[TARGET_COLUMN].unique()) != {0, 1}:
        raise ValueError("Expected binary target values 0 and 1")
    factories = model_factories or build_baseline_models()
    X = frame[FEATURES]
    y = frame[TARGET_COLUMN]
    X_train, X_test = X.loc[train_indices], X.loc[test_indices]
    y_train, y_test = y.loc[train_indices], y.loc[test_indices]

    # This artifact was fitted on the training split in Phase 4; test rows are transform-only.
    X_train_transformed = preprocessing_pipeline.transform(X_train)
    X_test_transformed = preprocessing_pipeline.transform(X_test)
    if not np.isfinite(X_train_transformed).all() or not np.isfinite(X_test_transformed).all():
        raise ValueError("The saved preprocessing pipeline produced non-finite values")

    holdout_rows = []
    fitted_models: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    scores: dict[str, np.ndarray] = {}
    metrics_by_model: dict[str, dict[str, float]] = {}
    for name, base_model in factories.items():
        model = clone(base_model)
        started = perf_counter()
        model.fit(X_train_transformed, y_train)
        fit_seconds = perf_counter() - started
        predicted = model.predict(X_test_transformed)
        score, score_source = _positive_class_scores(model, X_test_transformed)
        model_metrics = binary_classification_metrics(y_test, predicted, score)
        holdout_rows.append({
            "model": name, **model_metrics, "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)), "fit_seconds": float(fit_seconds),
            "score_source": score_source,
        })
        fitted_models[name] = model
        predictions[name] = predicted
        scores[name] = score
        metrics_by_model[name] = model_metrics

    cv_results = cross_validate_baselines(
        X_train, y_train, factories, n_splits=cv_folds, random_state=cv_random_state
    )
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    holdout_table = pd.DataFrame(holdout_rows)
    cv_path = tables_dir / "cross_validation.csv"
    holdout_path = tables_dir / "model_comparison.csv"
    holdout_table.to_csv(holdout_path, index=False)
    cv_results.to_csv(cv_path, index=False)
    figure_paths = _save_evaluation_figures(y_test, predictions, scores, metrics_by_model, figures_dir)
    return {
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "cv_folds": cv_folds, "holdout_results": holdout_table,
        "cross_validation_results": cv_results,
        "figure_paths": figure_paths, "holdout_table_path": holdout_path,
        "cross_validation_path": cv_path,
    }


def run_model_evaluation(
    cleaned_data_path: str | Path,
    preprocessing_path: str | Path,
    split_metadata_path: str | Path,
    figures_dir: str | Path,
    tables_dir: str | Path,
    *, cv_folds: int = CV_FOLDS, cv_random_state: int = CV_RANDOM_SEED,
) -> dict[str, Any]:
    """Load local Phase 2/4 artifacts and run Phase 6 holdout and CV evaluation."""
    frame = pd.read_csv(cleaned_data_path)
    with Path(split_metadata_path).open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    preprocessing = joblib.load(preprocessing_path)
    return evaluate_models(
        frame, metadata, preprocessing, cv_folds=cv_folds,
        cv_random_state=cv_random_state, figures_dir=figures_dir, tables_dir=tables_dir,
    )
