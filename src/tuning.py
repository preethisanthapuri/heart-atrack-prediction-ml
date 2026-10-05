"""Leakage-safe Phase 8 hyperparameter searches on the training partition."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)

from src.models import build_baseline_models
from src.preprocessing import FEATURES, TARGET_COLUMN
from src.feature_selection import build_selection_pipeline

RANDOM_SEED = 42
SEARCH_SCORING = {
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
    "f1": "f1",
    "average_precision": "average_precision",
}


def _search_definitions(n_iter: int) -> list[tuple[str, str, dict[str, list[Any]], int | None]]:
    """Predeclare modest search spaces for Phase 6-supported baseline models."""
    if n_iter < 1:
        raise ValueError("n_iter must be positive")
    return [
        (
            "Logistic Regression",
            "GridSearchCV",
            {"classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
             "classifier__class_weight": [None, "balanced"]},
            None,
        ),
        (
            "Support Vector Classifier (RBF)",
            "GridSearchCV",
            {"classifier__C": [0.1, 1.0, 10.0, 100.0],
             "classifier__gamma": ["scale", 0.001, 0.01, 0.1, 1.0],
             "classifier__class_weight": [None, "balanced"]},
            None,
        ),
        (
            "K-Nearest Neighbors",
            "RandomizedSearchCV",
            {"classifier__n_neighbors": [3, 5, 7, 9, 11, 13],
             "classifier__weights": ["uniform", "distance"],
             "classifier__p": [1, 2]},
            n_iter,
        ),
    ]


def tune_training_partition(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_splits: int = 5,
    random_state: int = RANDOM_SEED,
    n_iter: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Tune selected models using stratified CV and fold-local preprocessing only.

    ROC-AUC is the predeclared refit criterion for all searches. Returned best CV scores
    are search-selection estimates, not unbiased final performance estimates.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train lengths must match")
    if set(pd.unique(y_train)) != {0, 1}:
        raise ValueError("Tuning expects target classes 0 and 1")
    if y_train.value_counts().min() < n_splits:
        raise ValueError("Each target class must have at least n_splits observations")

    candidates = build_baseline_models()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    summary_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    best_estimators: dict[str, Any] = {}

    for model_name, search_type, parameters, iterations in _search_definitions(n_iter):
        pipeline = build_selection_pipeline(candidates[model_name])
        common = {
            "estimator": pipeline,
            "param_distributions" if search_type == "RandomizedSearchCV" else "param_grid": parameters,
            "scoring": SEARCH_SCORING,
            "refit": "roc_auc",
            "cv": cv,
            "n_jobs": 1,
            "return_train_score": False,
            "error_score": "raise",
        }
        if search_type == "GridSearchCV":
            search = GridSearchCV(**common)
        else:
            search = RandomizedSearchCV(
                **common,
                n_iter=iterations,
                random_state=random_state,
            )
        search.fit(X_train, y_train)
        best_estimators[model_name] = search.best_estimator_
        result_table = pd.DataFrame(search.cv_results_)
        for _, result in result_table.iterrows():
            candidate_rows.append({
                "model": model_name,
                "search_type": search_type,
                "rank_by_roc_auc": int(result["rank_test_roc_auc"]),
                "mean_roc_auc": float(result["mean_test_roc_auc"]),
                "std_roc_auc": float(result["std_test_roc_auc"]),
                "mean_accuracy": float(result["mean_test_accuracy"]),
                "mean_f1": float(result["mean_test_f1"]),
                "mean_average_precision": float(result["mean_test_average_precision"]),
                "parameters": json.dumps(result["params"], sort_keys=True),
            })
        best = result_table.iloc[search.best_index_]
        summary_rows.append({
            "model": model_name,
            "search_type": search_type,
            "cv_folds": n_splits,
            "candidate_count": len(result_table),
            "best_mean_roc_auc": float(best["mean_test_roc_auc"]),
            "best_roc_auc_sd": float(best["std_test_roc_auc"]),
            "best_mean_accuracy": float(best["mean_test_accuracy"]),
            "best_mean_f1": float(best["mean_test_f1"]),
            "best_mean_average_precision": float(best["mean_test_average_precision"]),
            "best_parameters": json.dumps(search.best_params_, sort_keys=True),
            "refit_criterion": "mean cross-validated ROC-AUC",
        })

    return pd.DataFrame(summary_rows), pd.DataFrame(candidate_rows), best_estimators


def run_tuning(
    cleaned_data_path: str | Path,
    split_metadata_path: str | Path,
    summary_path: str | Path,
    candidates_path: str | Path,
    *,
    n_splits: int = 5,
    random_state: int = RANDOM_SEED,
    n_iter: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only training rows from the Phase 4 split and persist actual search results."""
    frame = pd.read_csv(cleaned_data_path)
    with Path(split_metadata_path).open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("fitted_on") != "training split only":
        raise ValueError("Saved split metadata is not marked training-only")
    train_indices = metadata["train_indices"]
    summary, candidates, _ = tune_training_partition(
        frame.loc[train_indices, FEATURES],
        frame.loc[train_indices, TARGET_COLUMN],
        n_splits=n_splits,
        random_state=random_state,
        n_iter=n_iter,
    )
    for table, path in ((summary, summary_path), (candidates, candidates_path)):
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output, index=False)
    return summary, candidates
