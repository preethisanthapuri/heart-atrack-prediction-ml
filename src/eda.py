"""Exploratory data analysis for the cleaned heart-disease dataset."""
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

TARGET_COLUMN = "output"
CATEGORICAL_COLUMNS = ["sex", "cp", "fbs", "restecg", "exng", "slp", "caa", "thall"]
NUMERIC_COLUMNS = ["age", "trtbps", "chol", "thalachh", "oldpeak"]

sns.set_theme(style="whitegrid", context="notebook")


def load_cleaned_dataset(path: str | Path) -> pd.DataFrame:
    """Load the Phase 2 dataset and validate the known Phase 1 schema."""
    frame = pd.read_csv(path)
    required = set(CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + [TARGET_COLUMN])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Cleaned dataset is missing expected columns: {sorted(missing)}")
    if frame[TARGET_COLUMN].isna().any():
        raise ValueError(f"Target column {TARGET_COLUMN!r} contains missing values")
    return frame


def build_dataset_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Create one row per feature with schema, missingness and descriptive stats."""
    rows = []
    for column in frame.columns:
        series = frame[column]
        is_numeric = column in NUMERIC_COLUMNS
        row = {
            "feature": column,
            "role": "target" if column == TARGET_COLUMN else "categorical_code" if column in CATEGORICAL_COLUMNS else "numeric",
            "data_type": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "unique_count": int(series.nunique(dropna=False)),
        }
        if is_numeric:
            row.update({
                "mean": float(series.mean()), "std": float(series.std()),
                "median": float(series.median()), "q1": float(series.quantile(.25)),
                "q3": float(series.quantile(.75)), "min": float(series.min()), "max": float(series.max()),
            })
        else:
            row.update({"mean": np.nan, "std": np.nan, "median": np.nan, "q1": np.nan, "q3": np.nan, "min": np.nan, "max": np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def build_classwise_summaries(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return descriptive numerical summaries and categorical level proportions by target."""
    numeric_rows = []
    categorical_rows = []
    for outcome, group in frame.groupby(TARGET_COLUMN, observed=True):
        for column in NUMERIC_COLUMNS:
            values = group[column]
            numeric_rows.append({
                "target_class": outcome, "feature": column, "n": int(values.count()),
                "mean": float(values.mean()), "std": float(values.std()), "median": float(values.median()),
                "q1": float(values.quantile(.25)), "q3": float(values.quantile(.75)),
            })
        for column in CATEGORICAL_COLUMNS:
            counts = group[column].value_counts(dropna=False).sort_index()
            for level, count in counts.items():
                categorical_rows.append({
                    "target_class": outcome, "feature": column, "level": level,
                    "count": int(count), "within_class_percent": float(100 * count / len(group)),
                })
    return pd.DataFrame(numeric_rows), pd.DataFrame(categorical_rows)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_eda_figures(frame: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Create target, feature, class comparison and correlation plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    labels = {0: "output=0", 1: "output=1"}
    plot_frame = frame.copy()
    plot_frame["target_group"] = plot_frame[TARGET_COLUMN].map(labels).fillna(plot_frame[TARGET_COLUMN].astype(str))

    fig, ax = plt.subplots(figsize=(6, 4))
    counts = frame[TARGET_COLUMN].value_counts().sort_index()
    sns.barplot(x=counts.index.astype(str), y=counts.values, hue=counts.index.astype(str), legend=False, ax=ax, palette="Set2")
    for i, count in enumerate(counts.values):
        ax.text(i, count, f"{count} ({count / len(frame):.1%})", ha="center", va="bottom")
    ax.set(title="Target class distribution", xlabel="Output class", ylabel="Records")
    paths.append(_save(fig, output_dir / "target_distribution.png"))

    ncols = 2
    nrows = int(np.ceil(len(NUMERIC_COLUMNS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows))
    for ax, column in zip(axes.flat, NUMERIC_COLUMNS):
        sns.histplot(data=plot_frame, x=column, hue="target_group", bins="auto", element="step", stat="count", common_norm=False, ax=ax)
        ax.set_title(f"{column} distribution by target")
    for ax in axes.flat[len(NUMERIC_COLUMNS):]:
        ax.remove()
    fig.suptitle("Numeric feature distributions by observed target class", y=1.01)
    paths.append(_save(fig, output_dir / "numeric_feature_distributions.png"))

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, column in zip(axes.flat, NUMERIC_COLUMNS):
        sns.boxplot(data=plot_frame, x="target_group", y=column, hue="target_group", legend=False, ax=ax, palette="Set2")
        ax.set_xlabel("Target class")
        ax.set_title(column)
    axes.flat[-1].remove()
    fig.suptitle("Numeric feature box plots by target class")
    paths.append(_save(fig, output_dir / "numeric_boxplots_by_target.png"))

    ncols = 2
    nrows = int(np.ceil(len(CATEGORICAL_COLUMNS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.8 * nrows))
    for ax, column in zip(axes.flat, CATEGORICAL_COLUMNS):
        sns.countplot(data=plot_frame, x=column, hue="target_group", ax=ax, palette="Set2")
        ax.set_title(f"{column} counts by target")
        ax.set_xlabel(f"{column} (encoded category)")
        ax.set_ylabel("Records")
    for ax in axes.flat[len(CATEGORICAL_COLUMNS):]:
        ax.remove()
    fig.suptitle("Encoded categorical feature counts by target class", y=1.005)
    paths.append(_save(fig, output_dir / "categorical_counts_by_target.png"))

    corr_columns = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [TARGET_COLUMN]
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(frame[corr_columns].corr(method="spearman"), cmap="vlag", center=0, vmin=-1, vmax=1,
                square=True, linewidths=.35, cbar_kws={"label": "Spearman correlation"}, ax=ax)
    ax.set_title("Spearman correlations (encoded category codes are ordinal for this display)")
    paths.append(_save(fig, output_dir / "spearman_correlation_heatmap.png"))
    return paths


def run_eda(input_path: str | Path, output_dir: str | Path, tables_dir: str | Path) -> dict[str, object]:
    """Generate Phase 3 descriptive tables and figures from the cleaned CSV."""
    frame = load_cleaned_dataset(input_path)
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary = build_dataset_summary(frame)
    numeric, categorical = build_classwise_summaries(frame)
    summary_path = tables_dir / "dataset_summary.csv"
    numeric_path = tables_dir / "classwise_numeric_summary.csv"
    categorical_path = tables_dir / "classwise_categorical_distribution.csv"
    summary.to_csv(summary_path, index=False)
    numeric.to_csv(numeric_path, index=False)
    categorical.to_csv(categorical_path, index=False)
    figures = generate_eda_figures(frame, output_dir)
    return {
        "rows": int(len(frame)), "columns": int(frame.shape[1]),
        "target_distribution": {str(k): int(v) for k, v in frame[TARGET_COLUMN].value_counts().sort_index().items()},
        "table_paths": [summary_path, numeric_path, categorical_path], "figure_paths": figures,
    }

