from __future__ import annotations

import pandas as pd

from src.config.settings import get_settings


class ValidationError(ValueError):
    """Raised when data validation fails."""


REQUIRED_INPUT_COLUMNS = {
    "operation_date",
    "terminal_id",
    "commodity_type",
    "number_of_trains_waiting",
    "number_of_wagons",
    "tons_scheduled",
    "queue_time_hours",
    "equipment_availability",
    "rain_mm",
    "shift",
    "day_of_week",
    "harvest_season_flag",
    "operational_restriction_flag",
}
TARGET_COLUMN = "actual_productivity_tons_hour"


def validate_input_data(
    df: pd.DataFrame,
    require_target: bool = False,
    expected_terminals: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    settings = get_settings()
    required = set(REQUIRED_INPUT_COLUMNS)
    if require_target:
        required.add(TARGET_COLUMN)
    missing = sorted(required.difference(df.columns))
    errors: list[str] = []
    if missing:
        errors.append(f"Missing required columns: {missing}")
    if df.empty:
        errors.append("Input data is empty")
    if errors:
        raise ValidationError("; ".join(errors))

    checked = df.copy()
    checked["operation_date"] = pd.to_datetime(checked["operation_date"], errors="coerce")
    if checked["operation_date"].isna().any():
        errors.append("operation_date contains invalid values")
    if checked["terminal_id"].isna().any():
        errors.append("terminal_id cannot be null")
    terminals = set(expected_terminals or settings.expected_terminals)
    observed_terminals = set(checked["terminal_id"].dropna().astype(str))
    missing_terminals = sorted(terminals.difference(observed_terminals))
    if missing_terminals:
        errors.append(f"Expected terminals missing: {missing_terminals}")

    numeric_ranges = {
        "number_of_trains_waiting": (0, 80),
        "number_of_wagons": (1, 350),
        "tons_scheduled": (1, 40000),
        "queue_time_hours": (0, 96),
        "equipment_availability": (0, 1),
        "rain_mm": (0, 500),
        "day_of_week": (0, 6),
        "harvest_season_flag": (0, 1),
        "operational_restriction_flag": (0, 1),
    }
    if require_target:
        numeric_ranges[TARGET_COLUMN] = (0, 2500)

    for column, (lower, upper) in numeric_ranges.items():
        values = pd.to_numeric(checked[column], errors="coerce")
        if values.isna().any():
            errors.append(f"{column} contains non-numeric values")
        if ((values < lower) | (values > upper)).any():
            errors.append(f"{column} outside expected range [{lower}, {upper}]")

    if errors:
        raise ValidationError("; ".join(errors))
    return checked


def main() -> None:
    from src.data.load_data import load_port_productivity_data

    df = load_port_productivity_data()
    validate_input_data(df, require_target=True)
    print(f"Validated {len(df)} rows")


if __name__ == "__main__":
    main()

