from pathlib import Path
import json
import pandas as pd


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, **kwargs)


def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if str(path).endswith(".gz") else None
    df.to_csv(path, index=index, compression=compression)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=str)


def parse_percent(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype("string").str.replace("%", "", regex=False), errors="coerce") / 100


def parse_price(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.replace(r"[$,]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")
