"""Run Phase 3 EDA and statistical analysis on the cleaned dataset."""
from pathlib import Path
from src.eda import run_eda
from src.statistical_analysis import run_statistical_analysis

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "cleaned_dataset.csv"

if __name__ == "__main__":
    eda = run_eda(DATA_PATH, ROOT / "results" / "figures" / "eda", ROOT / "results" / "tables")
    stats = run_statistical_analysis(
        DATA_PATH,
        ROOT / "results" / "tables" / "statistical_tests.csv",
        ROOT / "results" / "tables" / "statistical_analysis.md",
    )
    print(f"Analyzed {eda['rows']} rows × {eda['columns']} columns")
    print(f"Target distribution: {eda['target_distribution']}")
    print(f"Generated {len(eda['figure_paths'])} figures and {len(eda['table_paths']) + 1} tables")
    print("FDR-significant exploratory features:", stats.loc[stats["fdr_0_05"], "feature"].tolist())
