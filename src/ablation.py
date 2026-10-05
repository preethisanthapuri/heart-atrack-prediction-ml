"""Leakage-safe staged ablation on the primary training partition only."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from functools import partial

from src.evaluation import binary_classification_metrics, build_model_pipeline
from src.preprocessing import FEATURES, TARGET_COLUMN, build_preprocessing_pipeline
from src.ensemble import _candidate_estimators

SEED = 42
STAGES = {
    "A": "Baseline Logistic Regression with all features",
    "B": "Logistic Regression with fold-local mutual-information selection (k=15)",
    "C": "B plus inner-CV tuning of Logistic Regression C",
    "D": "Untuned soft-voting ensemble (Logistic Regression, KNN, Random Forest)",
    "E": "D plus sigmoid probability calibration within outer training folds",
    "F": "Integrated tuned/selected soft vote with disjoint sigmoid calibration subset",
}


def _lr(k: int | None = None) -> Pipeline:
    steps: list[tuple[str, Any]] = [("preprocessor", build_preprocessing_pipeline().named_steps["preprocessor"])]
    if k is not None:
        steps.append(("selector", SelectKBest(partial(mutual_info_classif, random_state=SEED), k=k)))
    steps.append(("classifier", LogisticRegression(max_iter=2000, random_state=SEED)))
    return Pipeline(steps)


def _best_estimator(estimator: Any, grid: dict[str, list[Any]], X: pd.DataFrame,
                    y: pd.Series, inner_splits: int, random_state: int) -> Any:
    search = GridSearchCV(estimator, grid, scoring="roc_auc",
                          cv=StratifiedKFold(inner_splits, shuffle=True, random_state=random_state),
                          n_jobs=1, refit=True, error_score="raise")
    search.fit(X, y)
    return search.best_estimator_


def _positive_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(X)
    return probabilities[:, list(model.classes_).index(1)]


def _platt_calibrate(calibration_scores: np.ndarray, y: pd.Series,
                     test_scores: np.ndarray) -> np.ndarray:
    """Fit a sigmoid mapping on a disjoint calibration subset, using score logits."""
    epsilon = 1e-6
    logits = np.log(np.clip(calibration_scores, epsilon, 1-epsilon) /
                    (1-np.clip(calibration_scores, epsilon, 1-epsilon))).reshape(-1, 1)
    test_logits = np.log(np.clip(test_scores, epsilon, 1-epsilon) /
                         (1-np.clip(test_scores, epsilon, 1-epsilon))).reshape(-1, 1)
    calibrator = LogisticRegression(max_iter=1000, random_state=SEED)
    calibrator.fit(logits, y)
    return calibrator.predict_proba(test_logits)[:, list(calibrator.classes_).index(1)]


def run_ablation_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_splits: int = 5,
    inner_splits: int = 3,
    selection_size: int = 15,
    random_state: int = SEED,
    rf_trees: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run staged ablations and return summary, fold metrics, and OOF predictions.

    The function accepts training rows only. Each selection/tuning/calibration operation is
    contained within the corresponding outer training fold; the saved holdout is never read.
    """
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train lengths must match")
    if set(pd.unique(y_train)) != {0, 1}:
        raise ValueError("Ablation expects target classes 0 and 1")
    if n_splits < 2 or inner_splits < 2 or y_train.value_counts().min() < n_splits:
        raise ValueError("Need at least two folds and enough examples per class")
    if selection_size < 1 or selection_size > 30:
        raise ValueError("selection_size must be between 1 and 30 encoded features")

    cv = StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
    probabilities = {stage: np.full(len(X_train), np.nan) for stage in STAGES}
    fold_rows: list[dict[str, Any]] = []
    for fold, (fit_pos, valid_pos) in enumerate(cv.split(X_train, y_train), start=1):
        X_fit, X_valid = X_train.iloc[fit_pos], X_train.iloc[valid_pos]
        y_fit, y_valid = y_train.iloc[fit_pos], y_train.iloc[valid_pos]
        candidates: dict[str, Any] = {}

        candidates["A"] = _lr()
        candidates["A"].fit(X_fit, y_fit)

        candidates["B"] = _lr(selection_size)
        candidates["B"].fit(X_fit, y_fit)

        candidates["C"] = _best_estimator(
            _lr(selection_size), {"classifier__C": [0.01, 0.1, 1.0, 10.0]},
            X_fit, y_fit, inner_splits, random_state + fold)

        # Keep D and E aligned with the established Phase 9 voting specification.
        phase9 = _candidate_estimators(n_splits=inner_splits, random_state=random_state + fold)
        candidates["D"] = phase9["Soft Voting"].fit(X_fit, y_fit)
        candidates["E"] = CalibratedClassifierCV(
            estimator=clone(phase9["Soft Voting"]), method="sigmoid",
            cv=StratifiedKFold(inner_splits, shuffle=True, random_state=random_state + fold),
            ensemble=True,
        ).fit(X_fit, y_fit)

        # F: tune each selected voter member on development rows; calibrate on a disjoint subset.
        split = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state + fold + 1000)
        dev_pos, cal_pos = next(split.split(X_fit, y_fit))
        X_dev, X_cal = X_fit.iloc[dev_pos], X_fit.iloc[cal_pos]
        y_dev, y_cal = y_fit.iloc[dev_pos], y_fit.iloc[cal_pos]
        tuned_lr = _best_estimator(
            _lr(selection_size), {"classifier__C": [0.01, 0.1, 1.0, 10.0]},
            X_dev, y_dev, inner_splits, random_state + fold + 2000)
        tuned_knn = _best_estimator(
            build_model_pipeline(KNeighborsClassifier()),
            {"classifier__n_neighbors": [3, 5, 9, 13],
             "classifier__weights": ["uniform", "distance"],
             "classifier__p": [1, 2]},
            X_dev, y_dev, inner_splits, random_state + fold + 3000)
        rf = build_model_pipeline(RandomForestClassifier(
            n_estimators=rf_trees, random_state=SEED, n_jobs=1))
        integrated = VotingClassifier(
            estimators=[("selected_lr", tuned_lr), ("tuned_knn", tuned_knn), ("rf", rf)],
            voting="soft", n_jobs=1)
        integrated.fit(X_dev, y_dev)
        cal_scores = _positive_probability(integrated, X_cal)
        test_scores = _positive_probability(integrated, X_valid)
        probabilities["F"][valid_pos] = _platt_calibrate(cal_scores, y_cal, test_scores)

        for stage, model in candidates.items():
            probabilities[stage][valid_pos] = _positive_probability(model, X_valid)
        for stage, scores in probabilities.items():
            if np.isfinite(scores[valid_pos]).all():
                metric = binary_classification_metrics(y_valid, (scores[valid_pos] >= 0.5).astype(int), scores[valid_pos])
                fold_rows.append({"stage": stage, "fold": fold, **metric,
                                  "brier": float(np.mean((scores[valid_pos] - y_valid.to_numpy()) ** 2)),
                                  "log_loss": float(-np.mean(y_valid.to_numpy() * np.log(np.clip(scores[valid_pos], 1e-15, 1)) +
                                                             (1-y_valid.to_numpy()) * np.log(np.clip(1-scores[valid_pos], 1e-15, 1))))})

    if any(not np.isfinite(scores).all() for scores in probabilities.values()):
        raise RuntimeError("Some out-of-fold predictions were not generated")
    predictions = pd.DataFrame({"row_index": X_train.index.to_numpy(),
                                "y_true": y_train.to_numpy(),
                                **{f"probability_{k}": v for k, v in probabilities.items()}})
    fold_table = pd.DataFrame(fold_rows)
    summary_rows = []
    for stage, description in STAGES.items():
        scores = probabilities[stage]
        y = y_train.to_numpy()
        hard = (scores >= 0.5).astype(int)
        metrics = binary_classification_metrics(y, hard, scores)
        stage_folds = fold_table[fold_table.stage == stage]
        summary_rows.append({"stage": stage, "description": description, "folds": n_splits,
                             **metrics,
                             "brier": float(brier_score_loss(y, scores)),
                             "log_loss": float(log_loss(y, scores, labels=[0, 1])),
                             **{f"{m}_fold_mean": float(stage_folds[m].mean()) for m in ["accuracy", "sensitivity", "specificity", "f1", "roc_auc", "pr_auc", "brier"]},
                             **{f"{m}_fold_std": float(stage_folds[m].std(ddof=1)) for m in ["accuracy", "sensitivity", "specificity", "f1", "roc_auc", "pr_auc", "brier"]}})
    return pd.DataFrame(summary_rows), fold_table, predictions
