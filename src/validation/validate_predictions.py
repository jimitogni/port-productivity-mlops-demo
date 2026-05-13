from __future__ import annotations

import pandas as pd

from src.config.settings import get_settings
from src.utils.dates import FORECAST_HORIZONS
from src.validation.validate_input_data import ValidationError


REQUIRED_PREDICTION_COLUMNS = {
    "prediction_id",
    "run_id",
    "execution_date",
    "forecast_date",
    "forecast_horizon",
    "terminal_id",
    "predicted_productivity_tons_hour",
    "model_name",
    "model_version",
    "feature_version",
    "pipeline_version",
    "created_at",
}


def validate_predictions(
    df: pd.DataFrame,
    expected_terminals: tuple[str, ...] | None = None,
    expected_horizons: tuple[str, ...] = FORECAST_HORIZONS,
) -> pd.DataFrame:
    settings = get_settings()
    errors: list[str] = []
    missing = sorted(REQUIRED_PREDICTION_COLUMNS.difference(df.columns))
    if missing:
        errors.append(f"Missing required prediction columns: {missing}")
    if df.empty:
        errors.append("Prediction output is empty")
    if errors:
        raise ValidationError("; ".join(errors))

    predictions = pd.to_numeric(df["predicted_productivity_tons_hour"], errors="coerce")
    if predictions.isna().any():
        errors.append("Predictions contain null or non-numeric values")
    if (predictions < 0).any():
        errors.append("Predictions contain negative values")
    if ((predictions < 50) | (predictions > 2000)).any():
        errors.append("Predictions outside reasonable operational range [50, 2000]")

    terminals = set(expected_terminals or settings.expected_terminals)
    observed_terminals = set(df["terminal_id"].dropna().astype(str))
    missing_terminals = sorted(terminals.difference(observed_terminals))
    if missing_terminals:
        errors.append(f"Expected terminals missing from predictions: {missing_terminals}")

    observed_horizons = set(df["forecast_horizon"].dropna().astype(str))
    missing_horizons = sorted(set(expected_horizons).difference(observed_horizons))
    if missing_horizons:
        errors.append(f"Expected horizons missing from predictions: {missing_horizons}")

    if errors:
        raise ValidationError("; ".join(errors))
    return df

