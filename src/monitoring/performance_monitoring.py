from __future__ import annotations

import pandas as pd


def _relative_change(current: float, reference: float) -> float:
    if abs(reference) < 1e-9:
        return 0.0
    return abs(current - reference) / abs(reference)


def detect_data_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.25) -> tuple[bool, dict[str, float]]:
    metrics: dict[str, float] = {}
    drifted = False
    for column in ["rain_mm", "equipment_availability", "queue_time_hours", "number_of_trains_waiting"]:
        reference_mean = float(pd.to_numeric(reference_df[column], errors="coerce").mean())
        current_mean = float(pd.to_numeric(current_df[column], errors="coerce").mean())
        change = _relative_change(current_mean, reference_mean)
        metrics[f"{column}_relative_change"] = change
        drifted = drifted or change > threshold
    return drifted, metrics


def detect_prediction_drift(reference_df: pd.DataFrame, predictions_df: pd.DataFrame, threshold: float = 0.25) -> tuple[bool, dict[str, float]]:
    reference_mean = float(reference_df["actual_productivity_tons_hour"].mean())
    prediction_mean = float(predictions_df["predicted_productivity_tons_hour"].mean())
    change = _relative_change(prediction_mean, reference_mean)
    return change > threshold, {
        "reference_productivity_mean": reference_mean,
        "prediction_mean": prediction_mean,
        "prediction_relative_change": change,
    }

