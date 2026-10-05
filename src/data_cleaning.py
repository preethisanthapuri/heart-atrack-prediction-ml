"""Dataset cleaning and audit utilities for the heart disease project."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

EXPECTED_COLUMNS = ["age", "sex", "cp", "trtbps", "chol", "fbs", "restecg", "thalachh", "exng", "oldpeak", "slp", "caa", "thall", "output"]
TARGET_COLUMN = "output"
PROJECT_NAME = "heart-atrack-prediction-ml"
CODE_DOMAINS = {
    "sex": {0, 1}, "cp": {0, 1, 2, 3}, "fbs": {0, 1}, "restecg": {0, 1, 2},
    "exng": {0, 1}, "slp": {0, 1, 2}, "caa": {0, 1, 2, 3, 4}, "thall": {0, 1, 2, 3},
    TARGET_COLUMN: {0, 1},
}
POSITIVE_MEASUREMENTS = ["age", "trtbps", "chol", "thalachh"]
CONTINUOUS_FOR_OUTLIERS = ["age", "trtbps", "chol", "thalachh", "oldpeak"]


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def audit_dataset(path: str | Path, target_candidate: str | None = None) -> dict[str, Any]:
    """Summarize schema and data-quality signals without changing the input."""
    path = Path(path)
    frame = pd.read_csv(path)
    if target_candidate is not None and target_candidate not in frame.columns:
        raise ValueError(f"Target candidate {target_candidate!r} is not a column")
    target = frame[target_candidate] if target_candidate else None
    return {
        "file_name": path.name, "file_type": path.suffix.lower(), "rows": int(frame.shape[0]),
        "columns_count": int(frame.shape[1]), "column_names": list(frame.columns),
        "data_types": {c: str(t) for c, t in frame.dtypes.items()},
        "missing_values": {c: int(n) for c, n in frame.isna().sum().items()},
        "duplicate_records": int(frame.duplicated().sum()),
        "unique_counts": {c: int(n) for c, n in frame.nunique(dropna=False).items()},
        "value_counts": {c: {str(_json_value(k)): int(v) for k, v in frame[c].value_counts(dropna=False).items()} for c in frame.columns},
        "numerical_columns": frame.select_dtypes(include="number").columns.tolist(),
        "categorical_columns": frame.select_dtypes(exclude="number").columns.tolist(),
        "identifier_candidates": [c for c in frame.columns if frame[c].nunique(dropna=False) == len(frame)],
        "target_candidate": target_candidate,
        "target_class_distribution": ({str(_json_value(k)): int(v) for k, v in target.value_counts(dropna=False).items()} if target is not None else None),
        "potential_leakage_columns": [],
        "leakage_review_note": "No leakage column can be established from schema statistics alone; review provenance and feature timing.",
    }


def _outlier_summary(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    summary = {}
    for column in CONTINUOUS_FOR_OUTLIERS:
        q1, q3 = frame[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flagged = frame.loc[(frame[column] < lower) | (frame[column] > upper), column]
        summary[column] = {"q1": float(q1), "q3": float(q3), "lower_fence": float(lower), "upper_fence": float(upper),
                           "flag_count": int(len(flagged)), "flagged_values": sorted(flagged.tolist())}
    return summary


def clean_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply conservative, documented cleaning rules; outliers are retained."""
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(frame.columns))
    extra_columns = sorted(set(frame.columns) - set(EXPECTED_COLUMNS))
    if missing_columns or extra_columns:
        raise ValueError(f"Schema mismatch; missing={missing_columns}, unexpected={extra_columns}")
    cleaned = frame[EXPECTED_COLUMNS].copy()
    before_rows = len(cleaned)
    original_types = {c: str(t) for c, t in cleaned.dtypes.items()}
    conversion_failures: dict[str, int] = {}
    for column in EXPECTED_COLUMNS:
        converted = pd.to_numeric(cleaned[column], errors="coerce")
        conversion_failures[column] = int((converted.isna() & cleaned[column].notna()).sum())
        cleaned[column] = converted
    # A target cannot be imputed. Rows with absent/invalid labels cannot be used.
    target_missing = int(cleaned[TARGET_COLUMN].isna().sum())
    cleaned = cleaned.loc[cleaned[TARGET_COLUMN].notna()].copy()
    # Encoded predictors use their mode; continuous predictors use their median.
    imputed = {}
    for column in [c for c in EXPECTED_COLUMNS if c != TARGET_COLUMN]:
        count = int(cleaned[column].isna().sum())
        imputed[column] = count
        if count:
            if column in CODE_DOMAINS:
                modes = cleaned[column].mode(dropna=True)
                fill_value = modes.iloc[0] if not modes.empty else np.nan
            else:
                fill_value = cleaned[column].median()
            cleaned[column] = cleaned[column].fillna(fill_value)
    invalid_reasons: dict[str, int] = {}
    invalid = pd.Series(False, index=cleaned.index)
    for column, valid_values in CODE_DOMAINS.items():
        bad = ~cleaned[column].isin(valid_values)
        invalid_reasons[f"{column}_outside_code_domain"] = int(bad.sum())
        invalid |= bad
    for column in POSITIVE_MEASUREMENTS:
        bad = cleaned[column] <= 0
        invalid_reasons[f"{column}_not_positive"] = int(bad.sum())
        invalid |= bad
    bad_oldpeak = cleaned["oldpeak"] < 0
    invalid_reasons["oldpeak_negative"] = int(bad_oldpeak.sum())
    invalid |= bad_oldpeak
    non_finite = ~np.isfinite(cleaned.select_dtypes(include="number")).all(axis=1)
    invalid_reasons["non_finite_values"] = int(non_finite.sum())
    invalid |= non_finite
    invalid_rows = int(invalid.sum())
    cleaned = cleaned.loc[~invalid].copy()
    duplicate_count = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates(keep="first").reset_index(drop=True)
    constant_features = [c for c in cleaned.columns if c != TARGET_COLUMN and cleaned[c].nunique(dropna=False) <= 1]
    # Do not discard columns automatically: constants are reported for review.
    outliers = _outlier_summary(cleaned)
    report = {
        "input_rows": int(before_rows), "output_rows": int(len(cleaned)), "input_columns": EXPECTED_COLUMNS,
        "output_columns": list(cleaned.columns), "input_data_types": original_types,
        "output_data_types": {c: str(t) for c, t in cleaned.dtypes.items()},
        "numeric_conversion_failures": conversion_failures, "target_missing_rows_removed": target_missing,
        "predictor_missing_values_imputed_count": imputed,
        "imputation_strategy": "mode for encoded code columns; median for continuous numeric predictors",
        "invalid_value_counts_by_rule": invalid_reasons, "rows_with_any_invalid_value_removed": invalid_rows,
        "exact_duplicate_rows_removed": duplicate_count, "constant_predictor_columns": constant_features,
        "outlier_investigation_iqr": outliers,
        "decisions": [
            "Preserved the 14-column dataset schema and column order; rejected schema drift.",
            "Converted columns to numeric where possible; missing target rows are excluded because labels cannot be imputed.",
            "Imputed missing predictors if present (mode for encoded code columns, median for continuous predictors); none were present in this supplied dataset.",
            "Removed rows only for values outside the dataset's encoded code domains, non-positive measured values, negative oldpeak, or non-finite values.",
            "Removed exact duplicate records, retaining the first occurrence.",
            "Investigated continuous-variable outliers with the 1.5×IQR rule and retained them; IQR flags alone are not evidence of invalid measurements.",
            "Did not remove constant predictors automatically; reported any for review.",
        ],
    }
    return cleaned, report


def clean_csv(input_path: str | Path, output_path: str | Path, report_path: str | Path) -> dict[str, Any]:
    """Clean a CSV, save the processed data and a machine-readable decision log."""
    source = Path(input_path)
    output = Path(output_path)
    report_file = Path(report_path)
    original = pd.read_csv(source)
    cleaned, report = clean_dataset(original)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output, index=False)
    report["project_name"] = PROJECT_NAME
    report["source_file"] = source.name
    report["output_file"] = str(output)
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def write_audit_report(audit: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = output_dir / "dataset_audit.json", output_dir / "dataset_audit.md"
    json_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    lines = ["# Initial Dataset Audit", "", f"- File: `{audit['file_name']}` ({audit['file_type']})", f"- Shape: {audit['rows']} rows × {audit['columns_count']} columns", f"- Duplicate records: {audit['duplicate_records']}", f"- Target candidate: `{audit['target_candidate']}`", "", "## Columns", "", "| Column | Type | Missing | Unique | Values |", "|---|---:|---:|---:|---|"]
    for col in audit["column_names"]:
        values = ", ".join(f"{k}: {v}" for k, v in audit["value_counts"][col].items())
        if len(values) > 140: values = values[:137] + "..."
        lines.append(f"| `{col}` | {audit['data_types'][col]} | {audit['missing_values'][col]} | {audit['unique_counts'][col]} | {values} |")
    lines += ["", "## Target distribution", "", str(audit["target_class_distribution"]), "", "## Initial review notes", "", audit["leakage_review_note"], "", "The source CSV was not modified. No cleaning or modeling was performed in this audit.", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def write_cleaning_report(report: dict[str, Any], path: str | Path) -> Path:
    """Write a concise human-readable cleaning report from the cleaning log."""
    path = Path(path)
    lines = ["# Dataset Cleaning Report", "", "**Project:** `heart-atrack-prediction-ml`", "", f"- Source: `{report['source_file']}`", f"- Output: `{report['output_file']}`", f"- Rows: {report['input_rows']} → {report['output_rows']}", f"- Exact duplicates removed: {report['exact_duplicate_rows_removed']}", f"- Rows with invalid values removed: {report['rows_with_any_invalid_value_removed']}", f"- Rows removed for missing target: {report['target_missing_rows_removed']}", f"- Constant predictors: {report['constant_predictor_columns']}", "", "## IQR investigation (flags retained)", "", "| Feature | Lower fence | Upper fence | Flag count | Values |", "|---|---:|---:|---:|---|"]
    for column, values in report["outlier_investigation_iqr"].items():
        lines.append(f"| {column} | {values['lower_fence']:.2f} | {values['upper_fence']:.2f} | {values['flag_count']} | {values['flagged_values']} |")
    lines += ["", "## Cleaning decisions", ""] + [f"- {decision}" for decision in report["decisions"]] + [""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path







