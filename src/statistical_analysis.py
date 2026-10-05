"""Exploratory univariate association tests for Phase 3."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
from src.eda import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, TARGET_COLUMN, load_cleaned_dataset

RANDOM_SEED = 20261005
PERMUTATIONS = 9999


def _bh_adjust(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg false-discovery-rate adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result.tolist()


def _permutation_chi_square(codes: np.ndarray, target: np.ndarray, levels: np.ndarray,
                            permutations: int = PERMUTATIONS, seed: int = RANDOM_SEED) -> tuple[float, float]:
    """Pearson chi-square statistic and Monte Carlo p-value under shuffled labels."""
    n_levels = len(levels)
    level_index = pd.Categorical(codes, categories=levels).codes
    target_classes = np.sort(np.unique(target))
    target_index = pd.Categorical(target, categories=target_classes).codes
    n_classes = len(target_classes)
    observed = np.bincount(level_index * n_classes + target_index, minlength=n_levels * n_classes).reshape(n_levels, n_classes)
    row_totals, col_totals = observed.sum(axis=1), observed.sum(axis=0)
    expected = np.outer(row_totals, col_totals) / observed.sum()
    statistic = float(np.sum((observed - expected) ** 2 / expected))
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(permutations):
        shuffled = rng.permutation(target_index)
        table = np.bincount(level_index * n_classes + shuffled, minlength=n_levels * n_classes).reshape(n_levels, n_classes)
        trial = float(np.sum((table - expected) ** 2 / expected))
        extreme += trial >= statistic - 1e-12
    return statistic, (extreme + 1) / (permutations + 1)


def run_statistical_tests(frame: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    """Run two-group rank tests and encoded-category association tests with FDR correction."""
    classes = sorted(frame[target_column].dropna().unique())
    if len(classes) != 2:
        raise ValueError(f"Expected a binary target; found {len(classes)} classes")
    negative, positive = classes
    rows = []
    for feature in NUMERIC_COLUMNS:
        group0 = frame.loc[frame[target_column] == negative, feature].dropna().to_numpy(dtype=float)
        group1 = frame.loc[frame[target_column] == positive, feature].dropna().to_numpy(dtype=float)
        test = mannwhitneyu(group1, group0, alternative="two-sided", method="auto")
        effect = 2 * float(test.statistic) / (len(group1) * len(group0)) - 1
        rows.append({
            "feature": feature, "feature_type": "continuous", "test": "Mann-Whitney U (two-sided)",
            "class0_n": len(group0), "class0_median": float(np.median(group0)),
            "class0_q1": float(np.quantile(group0, .25)), "class0_q3": float(np.quantile(group0, .75)),
            "class1_n": len(group1), "class1_median": float(np.median(group1)),
            "class1_q1": float(np.quantile(group1, .25)), "class1_q3": float(np.quantile(group1, .75)),
            "test_statistic": float(test.statistic), "effect_size": effect,
            "effect_size_name": "rank-biserial correlation (class1 vs class0)", "p_value": float(test.pvalue),
            "min_expected_count": np.nan, "sparse_expected_cells": 0,
        })
    for feature in CATEGORICAL_COLUMNS:
        table = pd.crosstab(frame[feature], frame[target_column]).reindex(columns=classes, fill_value=0)
        chi = chi2_contingency(table, correction=False)
        expected = chi.expected_freq
        low_expected = int(np.sum(expected < 5))
        if table.shape == (2, 2) and low_expected:
            odds_ratio, p_value = fisher_exact(table.to_numpy(), alternative="two-sided")
            statistic, test_name = float(odds_ratio), "Fisher exact (two-sided)"
        elif low_expected:
            statistic, p_value = _permutation_chi_square(frame[feature].to_numpy(), frame[target_column].to_numpy(), table.index.to_numpy())
            test_name = f"Pearson chi-square permutation ({PERMUTATIONS} shuffles)"
        else:
            statistic, p_value = float(chi.statistic), float(chi.pvalue)
            test_name = "Pearson chi-square"
        n = int(table.to_numpy().sum())
        min_dim = min(table.shape[0] - 1, table.shape[1] - 1)
        cramers_v = float(np.sqrt(chi.statistic / (n * min_dim))) if min_dim > 0 and n > 0 else np.nan
        rows.append({
            "feature": feature, "feature_type": "encoded categorical", "test": test_name,
            "class0_n": int(table[negative].sum()), "class0_median": np.nan,
            "class0_q1": np.nan, "class0_q3": np.nan, "class1_n": int(table[positive].sum()),
            "class1_median": np.nan, "class1_q1": np.nan, "class1_q3": np.nan,
            "test_statistic": statistic, "effect_size": cramers_v,
            "effect_size_name": "Cramer's V", "p_value": float(p_value),
            "min_expected_count": float(np.min(expected)), "sparse_expected_cells": low_expected,
        })
    frame_out = pd.DataFrame(rows)
    frame_out["q_value_bh"] = _bh_adjust(frame_out["p_value"].tolist())
    frame_out["fdr_0_05"] = frame_out["q_value_bh"] < .05
    return frame_out.sort_values("p_value", ignore_index=True)


def write_statistical_interpretation(results: pd.DataFrame, path: str | Path) -> Path:
    """Write actual exploratory findings, methods, and interpretation limits."""
    path = Path(path)
    significant = results.loc[results["fdr_0_05"], "feature"].tolist()
    class0_n = int(results["class0_n"].iloc[0])
    class1_n = int(results["class1_n"].iloc[0])
    lines = [
        "# Exploratory Statistical Analysis", "",
        "**Project:** `heart-atrack-prediction-ml`", "",
        f"The cleaned dataset contains {class0_n + class1_n} observations: `output=0` n={class0_n}; `output=1` n={class1_n}.",
        "Continuous features use two-sided Mann-Whitney U tests; encoded categorical features use Pearson chi-square, Fisher exact for sparse 2×2 tables, and a label-permutation Pearson test for sparse larger tables.",
        "Benjamini-Hochberg false-discovery-rate adjustment is applied across all 13 predictor tests. Continuous-feature effects are rank-biserial correlations (class 1 vs class 0); categorical effects are Cramer's V.", "",
        f"Predictors with BH-adjusted q < 0.05: {', '.join(significant) if significant else 'none'} ({len(significant)} of {len(results)}).", "",
        "## Test results", "",
        "| Feature | Type | Test | Effect size | p-value | BH q-value | FDR < 0.05 |",
        "|---|---|---|---:|---:|---:|:---:|",
    ]
    for _, row in results.sort_values("q_value_bh").iterrows():
        lines.append(
            f"| {row['feature']} | {row['feature_type']} | {row['test']} | {row['effect_size']:.3f} | "
            f"{row['p_value']:.3g} | {row['q_value_bh']:.3g} | {'Yes' if row['fdr_0_05'] else 'No'} |"
        )
    lines += [
        "", "## Interpretation limits", "",
        "These are univariate associations with the observed dataset label. They do not establish causal effects, independent predictors, model performance, or clinical validity. The meaning of `output` has not been independently confirmed from dataset provenance, so results are stated only as comparisons between encoded classes 0 and 1.",
        "The single dataset is small; sparse tables used exact/permutation procedures as noted in the table. Correlation or statistical significance should not be interpreted as clinical importance. Replication on an independently collected dataset is needed.", "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_statistical_analysis(input_path: str | Path, output_path: str | Path, interpretation_path: str | Path) -> pd.DataFrame:
    frame = load_cleaned_dataset(input_path)
    results = run_statistical_tests(frame)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    write_statistical_interpretation(results, interpretation_path)
    return results

