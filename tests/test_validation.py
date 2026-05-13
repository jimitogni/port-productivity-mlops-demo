from __future__ import annotations

import pytest

from src.data.generate_synthetic_data import generate_daily_operational_forecast, generate_synthetic_data
from src.validation.validate_input_data import ValidationError, validate_input_data


def test_valid_training_data_passes_validation():
    df = generate_synthetic_data("2024-01-01", "2024-01-05", seed=13)
    validated = validate_input_data(df, require_target=True)
    assert len(validated) == len(df)


def test_missing_terminal_fails_validation():
    df = generate_daily_operational_forecast("2024-01-05", scenario="missing_terminal", seed=14)
    with pytest.raises(ValidationError, match="Expected terminals missing"):
        validate_input_data(df, require_target=False)


def test_invalid_equipment_availability_fails_validation():
    df = generate_synthetic_data("2024-01-01", "2024-01-05", seed=15)
    df.loc[0, "equipment_availability"] = 1.4
    with pytest.raises(ValidationError, match="equipment_availability"):
        validate_input_data(df, require_target=True)

