"""Training-only ensemble experiments for Phase 9."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.evaluation import METRIC_NAMES, specificity_score, build_model_pipeline
from src.models import build_baseline_models
from src.preprocessing import FEATURES, TARGET_COLUMN

RANDOM_SEED = 42


def _candidate_estimators(n_splits: int, random_state: int) -> dict[str, Any]:
    """Build selected Phase 6 single models plus probability-capable soft voting and stacking."""
    baselines = build_baseline_models()
    individual = {
        "Logistic Regression": baselines["Logistic Regression"],
        "K-Nearest Neighbors": baselines["K-Nearest Neighbors"],
        "Support Vector Classifier (RBF)": baselines["Support Vector Classifier (RBF)"],
        "Random Forest": baselines["Random Forest"],
    }
    voting_members = [
        ("logistic_regression", build_model_pipeline(individual["Logistic Regression"])),
        ("knn", build_model_pipeline(individual["K-Nearest Neighbors"])),
        ("random_forest", build_model_pipeline(individual["Random Forest"])),
    ]
    stack_members = [
        ("logistic_regression", build_model_pipeline(individual["Logistic Regression"])),
        ("knn", build_model_pipeline(individual["K-Nearest Neighbors"])),
        ("rbf_svc", build_model_pipeline(individual["Support Vector Classifier (RBF)"])),
    ]
    internal_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    estimators: dict[str, Any] = {
        name: build_model_pipeline(model) for name, model in individual.items()
    }
    estimators["Soft Voting"] = VotingClassifier(
        estimators=voting_members,
        voting="soft",
        weights=None,
        n_jobs=1,
    )
    estimators["Stacking"] = StackingClassifier(
        estimators=stack_members,
        final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
        stack_method="auto",
        cv=internal_cv,
        n_jobs=1,
        passthrough=False,
    )
    return estimators


def run_ensemble_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_splits: int = 5,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare individual classifiers and ensembles using identical outer CV folds.

    Only training rows are accepted. The stacking meta-estimator uses internal CV within
    each outer training fold; preprocessing pipelines are part of every base estimator.
    """
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train lengths must match")
    if set(pd.unique(y_train)) != {0, 1}:
        raise ValueError("Ensemble evaluation expects target classes 0 and 1")
    if n_splits < 2 or y_train.value_counts().min() < n_splits:
        raise ValueError("Each class must have at least n_splits observations and n_splits must be >= 2")

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, pos_label=1, zero_division=0),
        "sensitivity": "recall",
        "specificity": make_scorer(specificity_score),
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for name, estimator in _candidate_estimators(n_splits, random_state).items():
        results = cross_validate(
            estimator,
            X_train,
            y_train,
            scoring=scoring,
            cv=cv,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
        )
        summary: dict[str, Any] = {"model": name, "folds": n_splits}
        for metric in METRIC_NAMES:
            values = results[f"test_{metric}"]
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values, ddof=1))
            for fold_number, value in enumerate(values, start=1):
                fold_rows.append({"model": name, "fold": fold_number, "metric": metric, "value": float(value)})
        summary_rows.append(summary)
    return pd.DataFrame(summary_rows), pd.DataFrame(fold_rows)


def run_ensemble(
    cleaned_data_path: str | Path,
    split_metadata_path: str | Path,
    summary_path: str | Path,
    folds_path: str | Path,
    *,
    n_splits: int = 5,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only the Phase 4 training partition and save actual ensemble CV results."""
    import json

    frame = pd.read_csv(cleaned_data_path)
    with Path(split_metadata_path).open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("fitted_on") != "training split only":
        raise ValueError("Saved split metadata is not marked training-only")
    train_indices = metadata["train_indices"]
    summary, folds = run_ensemble_experiment(
        frame.loc[train_indices, FEATURES],
        frame.loc[train_indices, TARGET_COLUMN],
        n_splits=n_splits,
        random_state=random_state,
    )
    for table, path in ((summary, summary_path), (folds, folds_path)):
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output, index=False)
    return summary, folds
