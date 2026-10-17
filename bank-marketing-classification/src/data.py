"""Data loading helpers for the UCI Bank Marketing dataset."""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

# Both dataset versions, with their 10% samples.
DATASETS = {
    "bank-additional-full": "bank-additional-full.csv",  # 41,188 rows, 20 features + 5 macro indicators
    "bank-additional": "bank-additional.csv",            # 4,119-row sample of the above
    "bank-full": "bank-full.csv",                        # 45,211 rows, 16 features (older version)
    "bank": "bank.csv",                                  # 4,521-row sample of the above
}

TARGET = "y"
# Only known after the call ends, so it leaks the target (see bank-additional-names.txt).
LEAKY_COLUMNS = ["duration"]


def load_raw(name: str = "bank-additional-full", drop_leaky: bool = True) -> pd.DataFrame:
    """Load a raw CSV (semicolon-separated) and encode the target as 0/1."""
    df = pd.read_csv(RAW_DIR / DATASETS[name], sep=";")
    df[TARGET] = (df[TARGET] == "yes").astype(int)
    if drop_leaky:
        df = df.drop(columns=LEAKY_COLUMNS)
    return df


def split_xy(df: pd.DataFrame):
    return df.drop(columns=[TARGET]), df[TARGET]
