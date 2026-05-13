from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import get_settings
from src.data.generate_synthetic_data import COMMODITIES, SHIFTS
from src.utils.dates import FORECAST_HORIZONS, horizon_to_days


NUMERIC_FEATURES = [
    "number_of_trains_waiting",
    "number_of_wagons",
    "tons_scheduled",
    "queue_time_hours",
    "equipment_availability",
    "rain_mm",
    "harvest_season_flag",
    "operational_restriction_flag",
    "day_of_week_sin",
    "day_of_week_cos",
    "is_weekend",
    "forecast_horizon_days",
    "terminal_lag_1_productivity",
    "terminal_rolling_7d_productivity",
]
TARGET_COLUMN = "actual_productivity_tons_hour"


def expected_feature_columns() -> list[str]:
    settings = get_settings()
    columns = list(NUMERIC_FEATURES)
    columns.extend([f"terminal_id__{terminal}" for terminal in settings.expected_terminals])
    columns.extend([f"commodity_type__{commodity}" for commodity in COMMODITIES])
    columns.extend([f"shift__{shift}" for shift in SHIFTS])
    columns.extend([f"forecast_horizon__{horizon}" for horizon in FORECAST_HORIZONS])
    return columns


def _prepare_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["operation_date"] = pd.to_datetime(prepared["operation_date"])
    prepared["day_of_week"] = prepared["operation_date"].dt.weekday
    prepared["day_of_week_sin"] = np.sin(2 * np.pi * prepared["day_of_week"] / 7)
    prepared["day_of_week_cos"] = np.cos(2 * np.pi * prepared["day_of_week"] / 7)
    prepared["is_weekend"] = prepared["day_of_week"].isin([5, 6]).astype(int)
    if "forecast_horizon" not in prepared.columns:
        prepared["forecast_horizon"] = "D+1"
    prepared["forecast_horizon_days"] = prepared["forecast_horizon"].map(horizon_to_days).astype(int)
    return prepared


def add_training_history_features(df: pd.DataFrame) -> pd.DataFrame:
    prepared = _prepare_base_columns(df)
    prepared = prepared.sort_values(["terminal_id", "operation_date"]).copy()
    grouped = prepared.groupby("terminal_id", group_keys=False)[TARGET_COLUMN]
    prepared["terminal_lag_1_productivity"] = grouped.shift(1)
    prepared["terminal_rolling_7d_productivity"] = grouped.transform(
        lambda series: series.shift(1).rolling(window=7, min_periods=1).mean()
    )
    fallback = float(prepared[TARGET_COLUMN].mean())
    prepared["terminal_lag_1_productivity"] = prepared["terminal_lag_1_productivity"].fillna(fallback)
    prepared["terminal_rolling_7d_productivity"] = prepared["terminal_rolling_7d_productivity"].fillna(fallback)
    return prepared


def _latest_terminal_history(history_df: pd.DataFrame) -> pd.DataFrame:
    history = add_training_history_features(history_df)
    history = history.sort_values(["terminal_id", "operation_date"])
    latest = history.groupby("terminal_id", as_index=False).tail(1)
    return latest[
        [
            "terminal_id",
            "actual_productivity_tons_hour",
            "terminal_rolling_7d_productivity",
        ]
    ].rename(columns={"actual_productivity_tons_hour": "terminal_lag_1_productivity"})


def add_inference_history_features(df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    prepared = _prepare_base_columns(df)
    latest = _latest_terminal_history(history_df)
    prepared = prepared.merge(latest, on="terminal_id", how="left")
    fallback = float(history_df[TARGET_COLUMN].mean()) if TARGET_COLUMN in history_df.columns else 700.0
    prepared["terminal_lag_1_productivity"] = prepared["terminal_lag_1_productivity"].fillna(fallback)
    prepared["terminal_rolling_7d_productivity"] = prepared["terminal_rolling_7d_productivity"].fillna(fallback)
    return prepared


def _one_hot_features(df: pd.DataFrame) -> pd.DataFrame:
    settings = get_settings()
    encoded = df.copy()
    encoded["terminal_id"] = pd.Categorical(encoded["terminal_id"], categories=settings.expected_terminals)
    encoded["commodity_type"] = pd.Categorical(encoded["commodity_type"], categories=COMMODITIES)
    encoded["shift"] = pd.Categorical(encoded["shift"], categories=SHIFTS)
    encoded["forecast_horizon"] = pd.Categorical(encoded["forecast_horizon"], categories=FORECAST_HORIZONS)
    encoded = pd.get_dummies(
        encoded,
        columns=["terminal_id", "commodity_type", "shift", "forecast_horizon"],
        prefix_sep="__",
        dtype=float,
    )
    feature_columns = expected_feature_columns()
    for column in feature_columns:
        if column not in encoded.columns:
            encoded[column] = 0.0
    return encoded[feature_columns].astype(float)


def build_training_dataset(df: pd.DataFrame) -> pd.DataFrame:
    history_enriched = add_training_history_features(df)
    rows = []
    for horizon in FORECAST_HORIZONS:
        copy = history_enriched.copy()
        copy["forecast_horizon"] = horizon
        rows.append(copy)
    expanded = pd.concat(rows, ignore_index=True)
    prepared = _prepare_base_columns(expanded)
    features = _one_hot_features(prepared)
    features[TARGET_COLUMN] = prepared[TARGET_COLUMN].astype(float).values
    features["operation_date"] = prepared["operation_date"].dt.strftime("%Y-%m-%d").values
    features["terminal_id"] = prepared["terminal_id"].astype(str).values
    features["forecast_horizon"] = prepared["forecast_horizon"].astype(str).values
    return features


def build_inference_features(df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    prepared = add_inference_history_features(df, history_df)
    features = _one_hot_features(prepared)
    metadata_columns = [
        "execution_date",
        "forecast_date",
        "forecast_horizon",
        "operation_date",
        "terminal_id",
        "commodity_type",
        "shift",
    ]
    for column in metadata_columns:
        if column in prepared.columns:
            features[column] = prepared[column].astype(str).values
    return features


def save_feature_columns(path: str | Path | None = None) -> Path:
    settings = get_settings()
    output = Path(path or settings.models_dir / "feature_columns.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(expected_feature_columns(), indent=2))
    return output


def main() -> None:
    from src.data.load_data import load_port_productivity_data

    settings = get_settings()
    df = load_port_productivity_data()
    features = build_training_dataset(df)
    output = settings.data_dir / "processed" / "training_features.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output, index=False)
    save_feature_columns()
    print(f"Wrote {len(features)} feature rows to {output}")


if __name__ == "__main__":
    main()
