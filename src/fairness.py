"""Descriptive subgroup performance for out-of-fold predictions (not clinical fairness proof)."""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             confusion_matrix)
from src.preprocessing import TARGET_COLUMN


def _metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    predicted = (probability >= 0.5).astype(int)
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "precision_class_1": float(precision_score(y, predicted, pos_label=1, zero_division=0)),
        "recall_class_1": float(recall_score(y, predicted, pos_label=1, zero_division=0)),
        "specificity_class_0": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "f1_class_1": float(f1_score(y, predicted, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)) if np.unique(y).size == 2 else float("nan"),
        "average_precision": float(average_precision_score(y, probability)) if np.unique(y).size == 2 else float("nan"),
        "brier": float(brier_score_loss(y, probability)),
    }


def subgroup_fairness_tables(frame: pd.DataFrame, train_indices: list[int],
                             oof_predictions: pd.DataFrame,
                             *, probability_column: str = "probability_F") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize sex-code and descriptive age groups on training-only OOF results.

    Group metrics are descriptive; no significance tests, clinical threshold, or fairness
    guarantee is implied. The source target semantics and subgroup code definitions are unknown.
    """
    needed = {"sex", "age", TARGET_COLUMN}
    if not needed.issubset(frame.columns):
        raise ValueError(f"Missing required subgroup fields: {sorted(needed - set(frame.columns))}")
    if len(oof_predictions) != len(train_indices):
        raise ValueError("OOF prediction count must equal training-row count")
    if oof_predictions["row_index"].duplicated().any():
        raise ValueError("OOF row indices must be unique")
    expected = set(train_indices)
    if set(oof_predictions["row_index"]) != expected:
        raise ValueError("OOF row indices must exactly match the saved training partition")
    joined = frame.loc[train_indices, ["sex", "age", TARGET_COLUMN]].copy()
    joined["row_index"] = joined.index
    joined = joined.merge(oof_predictions[["row_index", "y_true", probability_column]], on="row_index", validate="one_to_one")
    if not np.array_equal(joined[TARGET_COLUMN].to_numpy(), joined["y_true"].to_numpy()):
        raise ValueError("OOF labels do not match the source rows")
    joined["age_group"] = pd.cut(joined["age"], bins=[-np.inf, 49, 59, 69, np.inf],
                                 labels=["<50", "50-59", "60-69", "70+"])
    rows: list[dict[str, Any]] = []
    for group_name, column in [("sex_code", "sex"), ("age_group", "age_group")]:
        for category, subset in joined.groupby(column, observed=True, dropna=False):
            y = subset["y_true"].to_numpy(dtype=int)
            p = subset[probability_column].to_numpy(dtype=float)
            rows.append({"grouping": group_name, "group": str(category), "n": len(subset),
                         "class_0_n": int((y == 0).sum()), "class_1_n": int((y == 1).sum()),
                         **_metrics(y, p)})
    metrics = pd.DataFrame(rows)
    disparity_rows = []
    for grouping, group_data in metrics.groupby("grouping"):
        for metric in ["accuracy", "recall_class_1", "specificity_class_0", "brier"]:
            values = group_data[metric].dropna()
            disparity_rows.append({"grouping": grouping, "metric": metric,
                                   "groups_with_metric": int(len(values)),
                                   "min": float(values.min()) if len(values) else float("nan"),
                                   "max": float(values.max()) if len(values) else float("nan"),
                                   "range": float(values.max() - values.min()) if len(values) else float("nan")})
    return metrics, pd.DataFrame(disparity_rows)
