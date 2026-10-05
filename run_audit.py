"""Run the Phase 1 dataset audit. No cleaning or modeling is performed."""
from pathlib import Path
from src.data_loader import load_dataset
from src.data_cleaning import audit_dataset, write_audit_report

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "heart_disease.csv"

if __name__ == "__main__":
    frame = load_dataset(DATA_PATH)
    # `output` is the only column named like a binary outcome in the inspected file.
    target = "output" if "output" in frame.columns else None
    audit = audit_dataset(DATA_PATH, target_candidate=target)
    for path in write_audit_report(audit, ROOT / "results" / "tables"):
        print(path)
    print(f"Audited {audit['rows']} rows and {audit['columns_count']} columns; target candidate={target!r}")
