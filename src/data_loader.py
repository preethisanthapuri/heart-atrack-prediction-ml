"""Dataset loading helpers for the heart disease project."""
from pathlib import Path
import pandas as pd

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "heart_disease.csv"


def load_dataset(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load a CSV dataset without modifying the source file."""
    return pd.read_csv(path)
