from __future__ import annotations

import pandas as pd
import pytest

from src.utils.dates import FORECAST_HORIZONS
from src.validation.validate_input_data import ValidationError
from src.validation.validate_predictions import validate_predictions


def _valid_prediction_frame() -> pd.DataFrame:
    rows = []
    for terminal_id in ["T1", "T2", "T3", "T4"]:
        for horizon in FORECAST_HORIZONS:
            rows.append(
                {
                    "prediction_id": f"{terminal_id}-{horizon}",
                    "run_id": "run-1",
                    "execution_date": "2026-05-12",
                    "forecast_date": "2026-05-13",
                    "forecast_horizon": horizon,
                    "terminal_id": terminal_id,
                    "predicted_productivity_tons_hour": 750.0,
                    "model_name": "port_productivity_forecaster",
                    "model_version": "1",
                    "feature_version": "v1",
                    "pipeline_version": "v1",
                    "created_at": "2026-05-12T00:00:00Z",
                }
            )
    return pd.DataFrame(rows)


def test_valid_predictions_pass_validation():
    df = _valid_prediction_frame()
    assert len(validate_predictions(df)) == 12


def test_negative_predictions_fail_validation():
    df = _valid_prediction_frame()
    df.loc[0, "predicted_productivity_tons_hour"] = -1
    with pytest.raises(ValidationError, match="negative"):
        validate_predictions(df)

