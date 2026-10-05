"""Leakage-safe feature-selection experiments for Phase 7."""
from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from src.evaluation import METRIC_NAMES, binary_classification_metrics
from src.preprocessing import FEATURES, TARGET_COLUMN, build_preprocessing_pipeline

RANDOM_SEED = 42
DEFAULT_SELECTION_SIZE = 15


def build_selectors(selection_size: int = DEFAULT_SELECTION_SIZE) -> dict[str, Any]:
    """Return predeclared, fixed-size selector methods for the 30 encoded inputs."""
    if selection_size < 1:
        raise ValueError("selection_size must be positive")
    return {
        "Mutual information": lambda: SelectKBest(
            score_func=partial(mutual_info_classif, random_state=RANDOM_SEED), k=selection_size
        ),
        "ANOVA SelectKBest": lambda: SelectKBest(score_func=f_classif, k=selection_size),
        "RFE (Logistic Regression)": lambda: RFE(
            estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
            n_features_to_select=selection_size,
            step=1,
        ),
        "Model-based (Extra Trees)": lambda: SelectFromModel(
            estimator=ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=1),
            threshold=-np.inf,
            max_features=selection_size,
        ),
    }


def build_selection_pipeline(estimator: Any, selector: Any | None = None) -> Pipeline:
    """Construct a fresh pipeline so imputation/encoding/scaling and selection fit in-fold."""
    steps: list[tuple[str, Any]] = [
        ("preprocessor", build_preprocessing_pipeline().named_steps["preprocessor"])
    ]
    if selector is not None:
        steps.append(("selector", selector))
    steps.append(("classifier", clone(estimator)))
    return Pipeline(steps)


def _estimators() -> dict[str, Any]:
    # LR led CV ROC-AUC and KNN led CV accuracy in Phase 6; both are retained for comparison.
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, weights="uniform"),
    }


def _cv_score(y_true: pd.Series, estimator: Pipeline, X: pd.DataFrame) -> dict[str, float]:
    prediction = estimator.predict(X)
    probability = estimator.predict_proba(X)
    class_one_index = list(estimator.classes_).index(1)
    return binary_classification_metrics(y_true, prediction, probability[:, class_one_index])


def _feature_ranks(selector: Any, feature_names: np.ndarray) -> dict[str, float]:
    """Return method-specific descending importance ranks (1 is best)."""
    if hasattr(selector, "scores_"):
        scores = np.nan_to_num(np.asarray(selector.scores_, dtype=float), nan=-np.inf)
        order = np.argsort(-scores, kind="stable")
    elif hasattr(selector, "ranking_"):
        order = np.argsort(np.asarray(selector.ranking_), kind="stable")
    elif hasattr(selector, "estimator_") and hasattr(selector.estimator_, "feature_importances_"):
        order = np.argsort(-selector.estimator_.feature_importances_, kind="stable")
    else:
        order = np.arange(len(feature_names))
    return {str(feature_names[index]): float(rank + 1) for rank, index in enumerate(order)}


def run_feature_selection_experiment(
    frame: pd.DataFrame,
    train_indices: list[int],
    *,
    n_splits: int = 5,
    random_state: int = RANDOM_SEED,
    selection_size: int = DEFAULT_SELECTION_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare all encoded features and selectors using training-only stratified CV.

    The test partition is intentionally not accepted by this function. Selector methods,
    feature budget, classifier set, and CV design are fixed before evaluation.
    """
    X = frame.loc[train_indices, FEATURES]
    y = frame.loc[train_indices, TARGET_COLUMN]
    if set(y.unique()) != {0, 1}:
        raise ValueError("Feature selection expects binary target values 0 and 1")
    if y.value_counts().min() < n_splits or n_splits < 2:
        raise ValueError("Each class must have at least n_splits rows and n_splits must be >= 2")

    selectors = build_selectors(selection_size)
    classifiers = _estimators()
    full_preprocessor = build_preprocessing_pipeline().named_steps["preprocessor"]
    full_preprocessor.fit(X)
    feature_names = full_preprocessor.get_feature_names_out()
    feature_count = len(feature_names)
    if selection_size > feature_count:
        raise ValueError(f"selection_size={selection_size} exceeds the {feature_count} encoded training features")
    configurations: list[tuple[str, Any | None]] = [("All features", None)]
    configurations.extend((name, make_selector()) for name, make_selector in selectors.items())
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    metric_rows: list[dict[str, Any]] = []
    selected_by_method: dict[str, list[list[str]]] = {name: [] for name in selectors}
    method_rankings: dict[str, list[dict[str, float]]] = {name: [] for name in selectors}

    for method_name, selector_template in configurations:
        for classifier_name, estimator in classifiers.items():
            fold_metrics: list[dict[str, float]] = []
            for fit_positions, valid_positions in cv.split(X, y):
                X_fit, X_valid = X.iloc[fit_positions], X.iloc[valid_positions]
                y_fit, y_valid = y.iloc[fit_positions], y.iloc[valid_positions]
                selector = clone(selector_template) if selector_template is not None else None
                pipeline = build_selection_pipeline(estimator, selector)
                pipeline.fit(X_fit, y_fit)
                fold_metrics.append(_cv_score(y_valid, pipeline, X_valid))

                if selector_template is not None and classifier_name == "Logistic Regression":
                    fitted_selector = pipeline.named_steps["selector"]
                    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
                    selected_by_method[method_name].append(names[fitted_selector.get_support()].astype(str).tolist())
                    method_rankings[method_name].append(_feature_ranks(fitted_selector, names))

            row: dict[str, Any] = {
                "selection_method": method_name,
                "classifier": classifier_name,
                "selected_feature_count": feature_count if method_name == "All features" else selection_size,
                "folds": n_splits,
            }
            for metric in METRIC_NAMES:
                values = np.asarray([result[metric] for result in fold_metrics])
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=1))
            metric_rows.append(row)

    # Fit each selector on all training rows to report its final training-only ranking.
    transformed = full_preprocessor.transform(X)
    feature_rows: list[dict[str, Any]] = []
    for method_name, make_selector in selectors.items():
        selector = make_selector()
        selector.fit(transformed, y)
        selected_mask = selector.get_support()
        full_training_selected = set(feature_names[selected_mask].astype(str))
        full_ranks = _feature_ranks(selector, feature_names)
        occurrences: dict[str, int] = {str(name): 0 for name in feature_names}
        for fold_names in selected_by_method[method_name]:
            for name in fold_names:
                occurrences[name] = occurrences.get(name, 0) + 1
        for name in feature_names.astype(str):
            feature_rows.append({
                "selection_method": method_name,
                "encoded_feature": name,
                "selected_folds": occurrences.get(name, 0),
                "fold_selection_rate": occurrences.get(name, 0) / n_splits,
                "full_training_selected": name in full_training_selected,
                "full_training_rank": full_ranks[name],
                "selection_size": selection_size,
            })

    return pd.DataFrame(metric_rows), pd.DataFrame(feature_rows)


def run_feature_selection(
    cleaned_data_path: str | Path,
    split_metadata_path: str | Path,
    comparison_path: str | Path,
    feature_path: str | Path,
    *,
    n_splits: int = 5,
    random_state: int = RANDOM_SEED,
    selection_size: int = DEFAULT_SELECTION_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load local Phase 2/4 data and persist training-only feature selection results."""
    import json

    frame = pd.read_csv(cleaned_data_path)
    with Path(split_metadata_path).open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("fitted_on") != "training split only":
        raise ValueError("Saved split metadata is not marked training-only")
    comparison, features = run_feature_selection_experiment(
        frame,
        metadata["train_indices"],
        n_splits=n_splits,
        random_state=random_state,
        selection_size=selection_size,
    )
    for table, path in ((comparison, comparison_path), (features, feature_path)):
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output, index=False)
    return comparison, features
