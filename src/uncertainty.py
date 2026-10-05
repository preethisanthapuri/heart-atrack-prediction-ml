"""Training-only uncertainty assessment via split conformal sets and selective risk."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.evaluation import build_model_pipeline
from src.models import build_baseline_models
from src.preprocessing import FEATURES, TARGET_COLUMN

RANDOM_SEED = 42
DEFAULT_ALPHAS = (0.10, 0.20)
RISK_COVERAGE_LEVELS = (0.50, 0.60, 0.70, 0.80, 0.90, 1.00)


def build_uncertainty_model() -> VotingClassifier:
    """Use the Phase 9 soft-voting candidate; probabilities are native model outputs."""
    models = build_baseline_models()
    return VotingClassifier(
        estimators=[
            ("logistic_regression", build_model_pipeline(models["Logistic Regression"])),
            ("knn", build_model_pipeline(models["K-Nearest Neighbors"])),
            ("random_forest", build_model_pipeline(models["Random Forest"])),
        ],
        voting="soft",
        n_jobs=1,
    )


def conformal_quantile(calibration_scores: Any, alpha: float) -> float:
    """Finite-sample split-conformal quantile with the conservative +1 correction."""
    values = np.sort(np.asarray(calibration_scores, dtype=float))
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Calibration scores must be a non-empty finite one-dimensional array")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    rank = int(np.ceil((len(values) + 1) * (1 - alpha)))
    return 1.0 if rank > len(values) else float(values[rank - 1])


def conformal_prediction_sets(probabilities: Any, quantile: float) -> list[tuple[int, ...]]:
    """Build binary label sets using nonconformity score 1 - class probability."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("Expected an n-by-2 probability matrix for classes 0 and 1")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    return [
        tuple(label for label in (0, 1) if 1 - row[label] <= quantile)
        for row in probabilities
    ]


def _positive_probabilities(model: Any, X: pd.DataFrame) -> np.ndarray:
    probability_matrix = model.predict_proba(X)
    class_order = list(model.classes_)
    if set(class_order) != {0, 1}:
        raise ValueError(f"Expected classes 0 and 1; observed {class_order}")
    return probability_matrix[:, class_order.index(1)]


def _risk_coverage_table(
    y_true: np.ndarray,
    prediction: np.ndarray,
    confidence: np.ndarray,
) -> list[dict[str, float | int]]:
    order = np.argsort(-confidence, kind="stable")
    rows = []
    for requested in RISK_COVERAGE_LEVELS:
        accepted = max(1, int(np.ceil(requested * len(y_true))))
        selected = order[:accepted]
        rows.append({
            "requested_coverage": requested,
            "actual_coverage": float(accepted / len(y_true)),
            "accepted_rows": accepted,
            "error_rate": float(np.mean(prediction[selected] != y_true[selected])),
            "mean_confidence": float(confidence[selected].mean()),
        })
    return rows


def run_uncertainty_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_splits: int = 5,
    calibration_fraction: float = 0.25,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Estimate conformal coverage and confidence-based selective risk by outer CV.

    Each outer fold is split into model-fit and calibration subsets. Its validation rows
    are untouched until prediction. Only training rows are accepted by this API.
    """
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train lengths must match")
    if set(pd.unique(y_train)) != {0, 1}:
        raise ValueError("Uncertainty evaluation expects target classes 0 and 1")
    if n_splits < 2 or y_train.value_counts().min() < n_splits:
        raise ValueError("Each class must have at least n_splits observations and n_splits must be >= 2")
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be between 0 and 1")
    if not alphas or any(not 0 < alpha < 1 for alpha in alphas):
        raise ValueError("Each conformal alpha must be between 0 and 1")

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    probability_oof = np.full(len(y_train), np.nan, dtype=float)
    fold_calibration: dict[int, dict[str, Any]] = {}
    for fold_number, (outer_fit, outer_validation) in enumerate(splitter.split(X_train, y_train), start=1):
        outer_y = y_train.iloc[outer_fit]
        model_positions, calibration_positions = train_test_split(
            np.arange(len(outer_fit)),
            test_size=calibration_fraction,
            random_state=random_state + fold_number,
            stratify=outer_y.to_numpy(),
        )
        fit_positions = outer_fit[model_positions]
        calibration_rows = outer_fit[calibration_positions]
        y_fit, y_cal = y_train.iloc[fit_positions], y_train.iloc[calibration_rows]
        if set(y_fit.unique()) != {0, 1} or set(y_cal.unique()) != {0, 1}:
            raise ValueError("Model-fit and calibration partitions must both contain classes 0 and 1")

        model = build_uncertainty_model()
        model.fit(X_train.iloc[fit_positions], y_fit)
        calibration_probability = _positive_probabilities(model, X_train.iloc[calibration_rows])
        calibration_true = y_cal.to_numpy(dtype=int)
        nonconformity = np.where(calibration_true == 1, 1 - calibration_probability, calibration_probability)
        quantiles = {alpha: conformal_quantile(nonconformity, alpha) for alpha in alphas}

        validation_probability = _positive_probabilities(model, X_train.iloc[outer_validation])
        probability_oof[outer_validation] = validation_probability
        fold_calibration[fold_number] = {
            "calibration_n": len(calibration_rows),
            "quantiles": quantiles,
            "validation_positions": outer_validation,
        }

    if not np.isfinite(probability_oof).all():
        raise RuntimeError("Some training observations did not receive out-of-fold probabilities")
    y_array = y_train.to_numpy(dtype=int)
    predicted = (probability_oof >= 0.5).astype(int)
    confidence = np.maximum(probability_oof, 1 - probability_oof)
    entropy = -(
        np.clip(probability_oof, 1e-12, 1) * np.log2(np.clip(probability_oof, 1e-12, 1))
        + np.clip(1 - probability_oof, 1e-12, 1) * np.log2(np.clip(1 - probability_oof, 1e-12, 1))
    )

    predictions = pd.DataFrame({
        "observed_class": y_array,
        "predicted_class_at_0_5": predicted,
        "class1_probability": probability_oof,
        "max_class_probability": confidence,
        "normalized_entropy": entropy,
        "correct_argmax": predicted == y_array,
    })
    summary_rows = []
    for alpha in alphas:
        per_fold = []
        for fold_number, fold_info in fold_calibration.items():
            positions = fold_info["validation_positions"]
            p = probability_oof[positions]
            true = y_array[positions]
            q = fold_info["quantiles"][alpha]
            matrix = np.column_stack([1 - p, p])
            sets = conformal_prediction_sets(matrix, q)
            set_sizes = np.asarray([len(item) for item in sets])
            covered = np.asarray([int(label in prediction_set) for label, prediction_set in zip(true, sets)])
            singleton = set_sizes == 1
            per_fold.append({
                "coverage": float(covered.mean()),
                "mean_set_size": float(set_sizes.mean()),
                "singleton_rate": float(singleton.mean()),
                "ambiguous_rate": float((set_sizes == 2).mean()),
                "empty_rate": float((set_sizes == 0).mean()),
                "singleton_accuracy": float(np.mean(predicted[positions][singleton] == true[singleton])) if singleton.any() else float("nan"),
            })
            for row_index, pred_set in zip(positions, sets):
                predictions.loc[row_index, f"set_alpha_{alpha:.2f}"] = ",".join(map(str, pred_set))

        pooled_sets = predictions[f"set_alpha_{alpha:.2f}"].fillna("").map(
            lambda value: tuple(int(label) for label in value.split(",") if label != "")
        )
        sizes = pooled_sets.map(len).to_numpy()
        coverage = np.asarray([int(label in pred_set) for label, pred_set in zip(y_array, pooled_sets)])
        for metric in ("coverage", "mean_set_size", "singleton_rate", "ambiguous_rate", "empty_rate", "singleton_accuracy"):
            values = np.asarray([row[metric] for row in per_fold], dtype=float)
            finite = values[np.isfinite(values)]
            mean = float(finite.mean()) if len(finite) else float("nan")
            sd = float(finite.std(ddof=1)) if len(finite) > 1 else float("nan")
            if metric == "coverage":
                coverage_mean, coverage_sd = mean, sd
            elif metric == "mean_set_size":
                set_size_mean, set_size_sd = mean, sd
            elif metric == "singleton_rate":
                singleton_mean, singleton_sd = mean, sd
            elif metric == "ambiguous_rate":
                ambiguous_mean, ambiguous_sd = mean, sd
            elif metric == "empty_rate":
                empty_mean, empty_sd = mean, sd
            else:
                selective_accuracy_mean, selective_accuracy_sd = mean, sd
        summary_rows.append({
            "method": "split conformal; 1 - class probability",
            "alpha": alpha,
            "nominal_coverage": 1 - alpha,
            "outer_folds": n_splits,
            "calibration_fraction": calibration_fraction,
            "calibration_rows_mean": float(np.mean([info["calibration_n"] for info in fold_calibration.values()])),
            "empirical_coverage_fold_mean": coverage_mean,
            "empirical_coverage_fold_sd": coverage_sd,
            "empirical_coverage_oof_pooled": float(coverage.mean()),
            "mean_set_size": set_size_mean,
            "singleton_rate": singleton_mean,
            "ambiguous_rate": ambiguous_mean,
            "empty_rate": empty_mean,
            "singleton_accuracy": selective_accuracy_mean,
            "singleton_accuracy_sd": selective_accuracy_sd,
            "argmax_accuracy": float(np.mean(predicted == y_array)),
            "brier_oof": float(brier_score_loss(y_array, probability_oof)),
            "roc_auc_oof": float(roc_auc_score(y_array, probability_oof)),
            "mean_normalized_entropy": float(entropy.mean()),
        })

    risk_rows = _risk_coverage_table(y_array, predicted, confidence)
    risk_table = pd.DataFrame(risk_rows)
    return pd.DataFrame(summary_rows), risk_table, predictions


def run_uncertainty(
    cleaned_data_path: str | Path,
    split_metadata_path: str | Path,
    summary_path: str | Path,
    risk_coverage_path: str | Path,
    predictions_path: str | Path,
    *,
    n_splits: int = 5,
    calibration_fraction: float = 0.25,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load only training rows and persist conformal/selective uncertainty results."""
    import json

    frame = pd.read_csv(cleaned_data_path)
    with Path(split_metadata_path).open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("fitted_on") != "training split only":
        raise ValueError("Saved split metadata is not marked training-only")
    train_indices = metadata["train_indices"]
    summary, risk, predictions = run_uncertainty_experiment(
        frame.loc[train_indices, FEATURES],
        frame.loc[train_indices, TARGET_COLUMN],
        n_splits=n_splits,
        calibration_fraction=calibration_fraction,
        alphas=alphas,
        random_state=random_state,
    )
    for table, path in (
        (summary, summary_path),
        (risk, risk_coverage_path),
        (predictions, predictions_path),
    ):
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output, index=False)
    return summary, risk, predictions
