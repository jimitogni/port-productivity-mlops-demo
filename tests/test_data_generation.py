from __future__ import annotations

from src.data.generate_synthetic_data import generate_synthetic_data


def test_synthetic_data_contains_required_columns():
    df = generate_synthetic_data("2024-01-01", "2024-01-10", seed=7)
    assert len(df) == 40
    assert {
        "operation_date",
        "terminal_id",
        "commodity_type",
        "rain_mm",
        "equipment_availability",
        "queue_time_hours",
        "actual_productivity_tons_hour",
    }.issubset(df.columns)
    assert set(df["terminal_id"]) == {"T1", "T2", "T3", "T4"}


def test_drift_scenario_reduces_average_productivity():
    normal = generate_synthetic_data("2026-01-01", "2026-03-31", seed=9)
    drift = generate_synthetic_data(
        "2026-01-01",
        "2026-03-31",
        scenario="drift",
        drift_start_date="2026-01-01",
        seed=9,
    )
    assert drift["actual_productivity_tons_hour"].mean() < normal["actual_productivity_tons_hour"].mean()

