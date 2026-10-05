"""Execute conservative Phase 2 cleaning on the supplied raw dataset."""
from pathlib import Path
from src.data_cleaning import clean_csv, write_cleaning_report

ROOT = Path(__file__).resolve().parent
if __name__ == "__main__":
    report = clean_csv(ROOT / "data/raw/heart_disease.csv", ROOT / "data/processed/cleaned_dataset.csv", ROOT / "results/metrics/cleaning_report.json")
    report_path = write_cleaning_report(report, ROOT / "results/tables/cleaning_report.md")
    print(f"Cleaned {report['input_rows']} rows to {report['output_rows']}; removed {report['exact_duplicate_rows_removed']} duplicate(s) and {report['rows_with_any_invalid_value_removed']} invalid row(s).")
    print(report_path)
