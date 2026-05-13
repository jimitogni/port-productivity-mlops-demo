from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.settings import get_settings


def load_port_productivity_data(path: str | Path | None = None) -> pd.DataFrame:
    settings = get_settings()
    data_path = Path(path or settings.raw_data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    return pd.read_csv(data_path)


def load_latest_predictions(predictions_dir: str | Path | None = None) -> pd.DataFrame:
    settings = get_settings()
    directory = Path(predictions_dir or settings.predictions_dir)
    files = sorted(directory.glob("predictions_*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.read_csv(files[-1])

