"""Validate and consolidate experiment outputs without recomputing or inventing metrics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REQUIRED_TABLES = {
    "Dataset description": "dataset_summary.csv",
    "Class-wise numerical summaries": "classwise_numeric_summary.csv",
    "Class-wise categorical distribution": "classwise_categorical_distribution.csv",
    "Exploratory statistical tests": "statistical_tests.csv",
    "Holdout model comparison": "model_comparison.csv",
    "Training cross-validation": "cross_validation.csv",
    "Feature selection comparison": "feature_selection_comparison.csv",
    "Feature selection stability": "feature_selection.csv",
    "Hyperparameter search results": "tuning_results.csv",
    "Hyperparameter search summary": "tuning_summary.csv",
    "Ensemble comparison": "ensemble_comparison.csv",
    "Probability calibration": "calibration.csv",
    "Uncertainty assessment": "uncertainty.csv",
    "External-site evaluation": "external_validation.csv",
    "Ablation study": "ablation.csv",
    "Ablation per-fold results": "ablation_folds.csv",
    "Subgroup analysis": "fairness.csv",
    "Subgroup disparity ranges": "fairness_disparity.csv",
}


def _read_required(tables_dir: Path) -> dict[str, pd.DataFrame]:
    missing = [filename for filename in REQUIRED_TABLES.values() if not (tables_dir / filename).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase 1–16 result tables: {missing}")
    return {label: pd.read_csv(tables_dir / filename) for label, filename in REQUIRED_TABLES.items()}


def build_final_stage_table(ablation: pd.DataFrame, final_metadata: dict[str, Any]) -> pd.DataFrame:
    """Return the recorded Stage A–F metrics with fold summaries and final selection marker."""
    required = {"stage", "accuracy", "sensitivity", "specificity", "f1", "roc_auc", "pr_auc", "brier"}
    if not required.issubset(ablation.columns):
        raise ValueError(f"Ablation table missing columns: {sorted(required - set(ablation.columns))}")
    if set(ablation.stage.astype(str)) != set("ABCDEF"):
        raise ValueError("Expected exactly ablation stages A through F")
    selected = str(final_metadata.get("selected_stage"))
    if selected not in set(ablation.stage.astype(str)):
        raise ValueError("Final model stage does not match the ablation table")
    result = ablation.copy()
    result["selected_final_candidate"] = result.stage.astype(str).eq(selected)
    result["metric_basis"] = "Pooled out-of-fold predictions; training partition only"
    return result


def _plot_ablation(stages: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("accuracy_fold_mean", "accuracy_fold_std", "Accuracy", "Higher is better"),
        ("roc_auc_fold_mean", "roc_auc_fold_std", "ROC-AUC", "Higher is better"),
        ("brier_fold_mean", "brier_fold_std", "Brier score", "Lower is better"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))
    fig.patch.set_facecolor("white")
    for ax, (mean_col, std_col, label, direction) in zip(axes, metrics):
        for stage_index, (_, row) in enumerate(stages.iterrows()):
            selected = bool(row["selected_final_candidate"])
            color = "#127c72" if selected else "#94a3a1"
            ax.errorbar(
                row[mean_col], stage_index, xerr=row[std_col], fmt="o",
                color=color, ecolor=color, elinewidth=1.6, capsize=3,
                markersize=7 if selected else 5,
            )
        ax.set_title(label, loc="left", fontsize=12, weight="bold", color="#1e353b")
        ax.set_yticks(list(range(6)), list("ABCDEF"))
        ax.invert_yaxis()
        ax.set_xlabel(f"Mean ± SD ({direction.lower()})")
        ax.grid(axis="x", color="#e5ecea", linewidth=.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_ylabel("Phase 14 stage (B is selected final candidate)")
    fig.suptitle("Training-only ablation comparison", x=.06, ha="left", y=1.04,
                 fontsize=16, weight="bold", color="#162a31")
    fig.text(.06, -.015,
             "Five stratified outer folds · bars show fold-to-fold SD, not confidence intervals · class codes only",
             fontsize=9, color="#68777a")
    fig.tight_layout()
    paths = [output_dir / "ablation_tradeoffs.png", output_dir / "ablation_tradeoffs.pdf"]
    fig.savefig(paths[0], dpi=300, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def _plot_baseline_cv(cv: pd.DataFrame, output_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ordered = cv.sort_values("roc_auc_mean", ascending=True)
    positions = np.arange(len(ordered))
    ax.errorbar(ordered.roc_auc_mean, positions, xerr=ordered.roc_auc_std,
                fmt="o", color="#147d73", ecolor="#9ab9b2", capsize=3, markersize=6)
    ax.set_yticks(positions, ordered.model)
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("Mean ROC-AUC ± sample SD across five folds")
    ax.set_title("Baseline classifiers · training partition", loc="left", fontsize=14, weight="bold")
    ax.grid(axis="x", color="#e5ecea")
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.text(.01, -.02, "Fold variability is descriptive; error bars are not confidence intervals. Target class 1 is an unverified dataset code.", fontsize=8.5, color="#68777a")
    fig.tight_layout()
    paths = [output_dir / "baseline_cv_roc_auc.png", output_dir / "baseline_cv_roc_auc.pdf"]
    fig.savefig(paths[0], dpi=300, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def _tidy_evidence(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(experiment: str, candidate: str, metric: str, value: Any, basis: str,
            n: int | None = None, variability: Any = None, caveat: str = "") -> None:
        if pd.isna(value):
            return
        rows.append({"experiment": experiment, "candidate": candidate, "metric": metric,
                     "estimate": float(value), "variability_sd": None if pd.isna(variability) else float(variability),
                     "n": n, "evaluation_basis": basis, "interpretation_caveat": caveat})

    for _, row in tables["Training cross-validation"].iterrows():
        for metric in ["accuracy", "sensitivity", "specificity", "f1", "roc_auc", "pr_auc"]:
            add("Baseline CV", row.model, metric, row[f"{metric}_mean"], "5-fold training CV; mean ± SD",
                variability=row[f"{metric}_std"], caveat="Internal cross-validation; small primary dataset.")
    for _, row in tables["Holdout model comparison"].iterrows():
        for metric in ["accuracy", "precision", "sensitivity", "specificity", "f1", "roc_auc", "pr_auc"]:
            add("Holdout (descriptive only)", row.model, metric, row[metric],
                "Saved 61-row holdout; previously screened in Phase 5", n=int(row.test_rows),
                caveat="Not an untouched final estimate; use as descriptive screening output only.")
    for _, row in tables["Feature selection comparison"].iterrows():
        label = f"{row.selection_method} / {row.classifier}"
        for metric in ["accuracy", "sensitivity", "specificity", "f1", "roc_auc", "pr_auc"]:
            add("Feature selection", label, metric, row[f"{metric}_mean"], "5-fold training CV; mean ± SD",
                variability=row[f"{metric}_std"], caveat="Feature selection refit within each fold.")
    for _, row in tables["Hyperparameter search summary"].iterrows():
        add("Hyperparameter search", row.model, "best_mean_roc_auc", row.best_mean_roc_auc,
            f"{row.search_type}; {int(row.cv_folds)}-fold candidate selection", variability=row.best_roc_auc_sd,
            caveat="Best search score is selected across candidates; selection optimism likely.")
    for _, row in tables["Ensemble comparison"].iterrows():
        for metric in ["accuracy", "sensitivity", "specificity", "f1", "roc_auc", "pr_auc"]:
            add("Ensemble comparison", row.model, metric, row[f"{metric}_mean"], "5-fold training CV; mean ± SD",
                variability=row[f"{metric}_std"], caveat="Internal CV; no independent replication.")
    for _, row in tables["Probability calibration"].iterrows():
        for metric in ["brier", "roc_auc"]:
            add("Calibration", f"{row.model} / {row.method}", metric,
                row[f"{metric}_mean"], "5 outer-fold training OOF; mean ± SD",
                variability=row[f"{metric}_std"], caveat="Calibration experiment candidates; not all are final-model-specific.")
    for _, row in tables["Uncertainty assessment"].iterrows():
        for metric, column in [("empirical_coverage_oof", "empirical_coverage_oof_pooled"),
                               ("mean_set_size", "mean_set_size"), ("empty_rate", "empty_rate"),
                               ("argmax_accuracy", "argmax_accuracy")]:
            add("Uncertainty", f"soft voting alpha={row.alpha:g}", metric, row[column],
                "Pooled training OOF", n=241,
                caveat="Split-conformal and selective estimates apply to soft voting, not final Stage B.")
    for _, row in tables["Ablation study"].iterrows():
        for metric in ["accuracy", "sensitivity", "specificity", "f1", "roc_auc", "pr_auc", "brier"]:
            add("Ablation", f"Stage {row.stage}", metric, row[metric], "Pooled 5-fold training OOF",
                n=241, variability=row.get(f"{metric}_fold_std"),
                caveat="Dataset classes only; pooled OOF estimate on one small dataset.")
    external = tables["External-site evaluation"].iloc[0]
    for metric, column in [("accuracy", "accuracy"), ("sensitivity_class_1", "sensitivity_class_1"),
                           ("specificity_class_0", "specificity_class_0"), ("roc_auc", "roc_auc"),
                           ("average_precision", "pr_auc_average_precision"), ("brier", "brier_score")]:
        add("External site", str(external.external_site), metric, external[column],
            "Frozen model; single historical site; n=294", n=int(external.n),
            caveat="Class-code direction unresolved; not prospective/clinical validation.")
    for _, row in tables["Subgroup analysis"].iterrows():
        for metric in ["accuracy", "recall_class_1", "specificity_class_0", "brier"]:
            add("Subgroup (descriptive)", f"{row.grouping}={row.group}", metric, row[metric],
                "Stage F training OOF subset", n=int(row.n),
                caveat="No fairness inference; class meaning unverified; groups are small.")
    return pd.DataFrame(rows)


def _write_paper_report(root: Path, tables: dict[str, pd.DataFrame],
                        stages: pd.DataFrame, payload: dict[str, Any],
                        test_results: dict[str, Any]) -> Path:
    report_path = root / "paper/final_results.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    final_stage = stages.loc[stages.selected_final_candidate].iloc[0]
    baseline = tables["Training cross-validation"].sort_values("roc_auc_mean", ascending=False)
    external = tables["External-site evaluation"].iloc[0]
    conformal = tables["Uncertainty assessment"]
    vote_cal = tables["Probability calibration"].loc[
        tables["Probability calibration"]["model"] == "Soft Voting"
    ]
    ablation_lines = [
        "| Stage | Accuracy | Sensitivity | Specificity | F1 | ROC-AUC | Average precision | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in stages.iterrows():
        marker = " (selected)" if row.selected_final_candidate else ""
        ablation_lines.append(
            f"| {row.stage}{marker} | {row.accuracy:.3f} | {row.sensitivity:.3f} | "
            f"{row.specificity:.3f} | {row.f1:.3f} | {row.roc_auc:.3f} | "
            f"{row.pr_auc:.3f} | {row.brier:.3f} |"
        )
    baseline_lines = [
        "| Model | Accuracy, mean ± SD | ROC-AUC, mean ± SD | Average precision, mean ± SD |",
        "|---|---:|---:|---:|",
    ]
    for _, row in baseline.iterrows():
        baseline_lines.append(
            f"| {row.model} | {row.accuracy_mean:.3f} ± {row.accuracy_std:.3f} | "
            f"{row.roc_auc_mean:.3f} ± {row.roc_auc_std:.3f} | "
            f"{row.pr_auc_mean:.3f} ± {row.pr_auc_std:.3f} |"
        )
    calibration_lines = [
        "| Soft-voting probability method | Brier, mean ± SD | ROC-AUC, mean ± SD |",
        "|---|---:|---:|",
    ]
    for _, row in vote_cal.iterrows():
        calibration_lines.append(
            f"| {row.method} | {row.brier_mean:.3f} ± {row.brier_std:.3f} | "
            f"{row.roc_auc_mean:.3f} ± {row.roc_auc_std:.3f} |"
        )
    uncertainty_lines = [
        "| Nominal coverage | Empirical OOF coverage | Mean set size | Ambiguous rate | Empty rate |",
        "|---:|---:|---:|---:|---:|",
    ]
    for _, row in conformal.iterrows():
        uncertainty_lines.append(
            f"| {row.nominal_coverage:.2f} | {row.empirical_coverage_oof_pooled:.3f} | "
            f"{row.mean_set_size:.3f} | {row.ambiguous_rate:.3f} | {row.empty_rate:.3f} |"
        )
    subgroup = tables["Subgroup analysis"]
    fairness_lines = [
        "| Grouping | Code/bin | n | Class-1 recall | Class-0 specificity | Accuracy |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in subgroup.iterrows():
        fairness_lines.append(
            f"| {row.grouping} | {row.group} | {int(row.n)} | {row.recall_class_1:.3f} | "
            f"{row.specificity_class_0:.3f} | {row.accuracy:.3f} |"
        )

    body = f"""# Phase 17 — Testing and final results

## Verification

The project test suite completed with {test_results['passed_test_cases']} passing tests out of {test_results['discovered_test_cases']} discovered, with {test_results['failure_count']} failures and {test_results['error_count']} errors. Detailed test case IDs and the run environment are in `results/metrics/test_results.json`. The Flask API was smoke-tested against the saved Stage B model and returned HTTP 200 with both class probabilities and local contributions. Python modules compiled successfully. Final-results tables and figures were assembled from previously executed experiment artifacts; this consolidation did not refit models or recalculate patient-level predictions.

## Dataset and evaluation boundaries

The provided primary CSV contained {payload['dataset']['raw_rows']} raw rows and {payload['dataset']['raw_columns']} columns. Cleaning removed {payload['dataset']['duplicate_records_removed']} exact duplicate, leaving {payload['dataset']['cleaned_rows']} rows. The target is `output` with numeric dataset classes 0 and 1; label semantics and provenance remain unverified. The saved stratified split has {payload['evaluation_boundaries']['training_cv_rows']} training rows and {payload['evaluation_boundaries']['saved_holdout_rows']} holdout rows.

The model comparison below reports training-only five-fold cross-validation mean ± sample SD. The holdout was previously screened for accuracy in Phase 5, so its metrics are only descriptive and are not an untouched final estimate. Phases 7–16 selection, ablation, and final model fitting used training indices only. The historical Hungary result is separate from internal CV and has substantial feature missingness and unresolved primary-target orientation.

## Baseline cross-validation

{chr(10).join(baseline_lines)}

## Staged ablation

Pooled out-of-fold metrics for stages A–F are shown below. Outer-fold mean and standard deviation are shown in `results/tables/final_ablation_summary.csv`; fold SD is descriptive, not a confidence interval.

{chr(10).join(ablation_lines)}

Stage B (mutual-information selection plus Logistic Regression) was selected as the probability-oriented research candidate. It has the strongest pooled ROC-AUC ({final_stage.roc_auc:.3f}), average precision ({final_stage.pr_auc:.3f}), and Brier score ({final_stage.brier:.3f}) among the staged candidates. Stage A had higher accuracy ({stages.loc[stages.stage == 'A', 'accuracy'].iloc[0]:.3f}) and F1 ({stages.loc[stages.stage == 'A', 'f1'].iloc[0]:.3f}). Stage F did not improve the overall result. This is a metric tradeoff from a small internal sample, not proof of superiority.

## Calibration and uncertainty scope

| Soft voting probability method | Brier, mean ± SD | ROC-AUC, mean ± SD |
|---|---:|---:|
{chr(10).join(calibration_lines[2:])}

The calibration experiment concerned its listed baseline/voting candidates; a calibrated Stage B model was not directly evaluated. No post-hoc calibrator is attached to the final artifact.

{chr(10).join(uncertainty_lines)}

These Phase 11 split-conformal results are for soft voting, not Stage B, and are not a guarantee of future or clinical coverage.

## External site and subgroup results

The frozen Phase 6 baseline scored {int(external.n)} historical UCI Hungary records: accuracy {external.accuracy:.3f}, ROC-AUC {external.roc_auc:.3f}, average precision {external.pr_auc_average_precision:.3f}, and Brier {external.brier_score:.3f}. Project class 1 appears aligned to UCI angiographic absence, with a one-record count discrepancy and unconfirmed source mapping. Reported values are dataset-code metrics, not disease-positive performance.

The following Stage F subgroup estimates are descriptive and have no uncertainty intervals or fairness-inference interpretation:

{chr(10).join(fairness_lines)}

Small group sizes include n=9 for the 70+ bin. Target direction, subgroup codebook, and provenance are not verified; these results do not establish fairness or discriminatory impact.

## Final artifacts

- `results/tables/final_ablation_summary.csv` — staged metrics and fold summaries.
- `results/tables/final_results_evidence.csv` — tidy multi-phase metrics annotated with evaluation scope and caveats.
- `results/tables/final_results_catalog.csv` — final catalog of available phase tables.
- `results/tables/final_figure_catalog.csv` — inventory of generated phase figures (formats and file sizes).
- `results/figures/final_results/ablation_tradeoffs.png` and `.pdf` — accuracy, ROC-AUC, Brier comparison with outer-fold variability.
- `results/figures/final_results/baseline_cv_roc_auc.png` and `.pdf` — baseline training CV ROC-AUC with fold variability.

The figure catalog contains {payload['figure_inventory_count']} generated figures across EDA, model evaluation, calibration, SHAP, and final results.

The full tuning, feature-selection, calibration, uncertainty, external, ablation, and subgroup tables remain under `results/tables/`. Patient-level inputs and fitted model binaries are local-only/ignored by Git.

## Limitations

The project dataset is small. Target meaning and source provenance remain uncertain, the holdout was screened during baseline work, and the external cohort is historical with severe missingness. Calibration and uncertainty evidence are not specific to the selected Stage B model. Model explanations describe associations in model outputs and do not imply causality. No prospective clinical validation or clinical utility study has been performed. The application remains for research and education, not diagnosis.
"""
    report_path.write_text(body, encoding="utf-8")
    return report_path


def create_final_results(project_root: str | Path) -> dict[str, Any]:
    """Validate phase outputs and write publication-ready summary artifacts."""
    root = Path(project_root)
    tables_dir = root / "results/tables"
    metrics_dir = root / "results/metrics"
    figures_dir = root / "results/figures/final_results"
    tables = _read_required(tables_dir)
    final_metadata = json.loads((metrics_dir / "final_model.json").read_text(encoding="utf-8"))
    split = json.loads((metrics_dir / "preprocessing_split.json").read_text(encoding="utf-8"))
    test_results = json.loads((metrics_dir / "test_results.json").read_text(encoding="utf-8"))
    cleaned_audit = json.loads((tables_dir / "dataset_audit.json").read_text(encoding="utf-8"))
    if final_metadata.get("selected_stage") != "B":
        raise ValueError("Final model selection metadata does not specify the documented Stage B")
    if int(final_metadata["training_rows"]) != len(split["train_indices"]) or int(final_metadata["holdout_rows_not_used"]) != len(split["test_indices"]):
        raise ValueError("Final model partition counts do not match the saved split metadata")
    if set(split["train_indices"]) & set(split["test_indices"]):
        raise ValueError("Saved training and holdout indices overlap")
    if not test_results.get("success") or test_results.get("failure_count") or test_results.get("error_count"):
        raise ValueError("Project test results are not clean; run and fix the suite before final results")
    stages = build_final_stage_table(tables["Ablation study"], final_metadata)
    final_b = stages.loc[stages.selected_final_candidate].iloc[0]
    expected_metrics = final_metadata["phase14_stage_b_pooled_training_oof"]
    for metric, value in expected_metrics.items():
        if metric in final_b and not np.isclose(float(final_b[metric]), float(value), rtol=1e-8, atol=1e-10):
            raise ValueError(f"Final model metadata and Phase 14 table disagree for {metric}")

    stage_path = tables_dir / "final_ablation_summary.csv"
    evidence_path = tables_dir / "final_results_evidence.csv"
    catalog_path = tables_dir / "final_results_catalog.csv"
    stages.to_csv(stage_path, index=False)
    evidence = _tidy_evidence(tables)
    evidence.to_csv(evidence_path, index=False)
    catalog_rows = []
    for label, filename in REQUIRED_TABLES.items():
        table = tables[label]
        catalog_rows.append({"artifact": filename, "description": label,
                             "rows": len(table), "columns": len(table.columns),
                             "contents_patient_level_records": False})
    catalog_rows += [
        {"artifact": stage_path.name, "description": "Consolidated Phase 14 stage metrics plus fold summary", "rows": len(stages), "columns": len(stages.columns), "contents_patient_level_records": False},
        {"artifact": evidence_path.name, "description": "Tidy multi-phase metric catalog with evaluation scope/caveats", "rows": len(evidence), "columns": len(evidence.columns), "contents_patient_level_records": False},
    ]
    pd.DataFrame(catalog_rows).to_csv(catalog_path, index=False)

    figure_paths = _plot_ablation(stages, figures_dir) + _plot_baseline_cv(tables["Training cross-validation"], figures_dir)
    figure_inventory = sorted(path for path in (root / "results/figures").rglob("*") if path.is_file())
    figure_catalog = pd.DataFrame([{
        "artifact": str(path.relative_to(root)), "format": path.suffix.lower().lstrip("."),
        "size_bytes": path.stat().st_size, "contents_patient_level_records": False,
    } for path in figure_inventory])
    figure_catalog_path = tables_dir / "final_figure_catalog.csv"
    figure_catalog.to_csv(figure_catalog_path, index=False)
    payload = {
        "project_name": "heart-atrack-prediction-ml",
        "status": "final results assembled from recorded experiment outputs; no new model fitting",
        "verification": {"test_cases": int(test_results["discovered_test_cases"]),
                         "passed": int(test_results["passed_test_cases"]),
                         "failures": int(test_results["failure_count"]), "errors": int(test_results["error_count"]),
                         "python_version": test_results["python_version"]},
        "dataset": {"raw_rows": int(cleaned_audit["rows"]), "raw_columns": int(cleaned_audit["columns_count"]),
                    "cleaned_rows": int(final_metadata["training_rows"] + final_metadata["holdout_rows_not_used"]),
                    "duplicate_records_removed": int(cleaned_audit["duplicate_records"]),
                    "target_column": "output", "target_meaning": "unverified dataset codes 0 and 1"},
        "evaluation_boundaries": {
            "training_cv_rows": int(final_metadata["training_rows"]),
            "saved_holdout_rows": int(final_metadata["holdout_rows_not_used"]),
            "final_selection_fit_rows": int(final_metadata["training_rows"]),
            "holdout_note": "Baseline holdout results exist but the split was screened in Phase 5; treat as descriptive only. Phases 7-16 final selection/evaluation did not use holdout rows.",
        },
        "selected_candidate": {"stage": str(final_b.stage), "model": "Mutual-information SelectKBest(k=15) + Logistic Regression",
                               "pooled_training_oof_metrics": {k: float(expected_metrics[k]) for k in expected_metrics}},
        "ablation_conclusion": "Stage B led on ROC-AUC, average precision, and Brier; Stage A led on accuracy and F1; Stage F did not outperform the simpler candidates overall.",
        "calibration_note": "No Stage-B-specific post-hoc calibration claim; Phase 10 assessed other baseline/voting variants.",
        "uncertainty_note": "Phase 11 estimates apply to soft voting and are not transferred to the selected Stage B model.",
        "external_validation_note": "Historical UCI Hungary site evaluation is exploratory; primary data provenance/target orientation remain unresolved.",
        "subgroup_analysis_note": "Descriptive class-coded subgroup metrics only; not a fairness certification.",
        "tables": {"stage_summary": str(stage_path.relative_to(root)), "tidy_evidence": str(evidence_path.relative_to(root)),
                   "catalog": str(catalog_path.relative_to(root)),
                   "figure_catalog": str(figure_catalog_path.relative_to(root))},
        "figures": [str(path.relative_to(root)) for path in figure_paths],
        "figure_inventory_count": len(figure_catalog),
    }
    (metrics_dir / "final_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    report_path = _write_paper_report(root, tables, stages, payload, test_results)
    payload["report"] = str(report_path.relative_to(root))
    (metrics_dir / "final_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return payload
