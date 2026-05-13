from __future__ import annotations

import html
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import get_settings
from src.monitoring.performance_monitoring import detect_data_drift, detect_prediction_drift
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)

TARGET_COLUMN = "actual_productivity_tons_hour"
PREDICTION_COLUMN = "predicted_productivity_tons_hour"


def _relative_change(current: float, reference: float) -> float:
    if abs(reference) < 1e-9:
        return 0.0
    return abs(current - reference) / abs(reference)


def _feature_distribution_changes(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    changes: dict[str, dict[str, float]] = {}
    common_columns = [column for column in current_df.columns if column in reference_df.columns]
    for column in common_columns:
        reference_values = pd.to_numeric(reference_df[column], errors="coerce")
        current_values = pd.to_numeric(current_df[column], errors="coerce")
        if reference_values.notna().sum() == 0 or current_values.notna().sum() == 0:
            continue
        reference_mean = float(reference_values.mean())
        current_mean = float(current_values.mean())
        changes[column] = {
            "reference_mean": round(reference_mean, 4),
            "current_mean": round(current_mean, 4),
            "relative_change": round(_relative_change(current_mean, reference_mean), 4),
            "reference_std": round(float(reference_values.std(ddof=0)), 4),
            "current_std": round(float(current_values.std(ddof=0)), 4),
        }
    return changes


def _regression_performance_if_available(
    current_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> dict[str, float] | None:
    if TARGET_COLUMN not in current_df.columns or PREDICTION_COLUMN not in predictions_df.columns:
        return None
    actual = pd.to_numeric(current_df[TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
    predicted = pd.to_numeric(predictions_df[PREDICTION_COLUMN], errors="coerce").to_numpy(dtype=float)
    length = min(len(actual), len(predicted))
    if length == 0:
        return None
    actual = actual[:length]
    predicted = predicted[:length]
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if not mask.any():
        return None
    actual = actual[mask]
    predicted = predicted[mask]
    errors = actual - predicted
    mae = float(np.mean(np.abs(errors)))
    rmse = float(math.sqrt(np.mean(np.square(errors))))
    denominator = float(np.sum(np.square(actual - np.mean(actual))))
    r2 = 0.0 if denominator < 1e-9 else float(1 - np.sum(np.square(errors)) / denominator)
    safe = np.abs(actual) > 1e-6
    mape = float(np.mean(np.abs(errors[safe] / actual[safe])) * 100) if safe.any() else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def _fallback_html_report(title: str, sections: dict[str, object], output: Path) -> Path:
    rows = []
    for key, value in sections.items():
        rows.append(f"<tr><th>{html.escape(str(key))}</th><td><pre>{html.escape(str(value))}</pre></td></tr>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title>"
        "<style>body{font-family:Arial,sans-serif;margin:32px}table{border-collapse:collapse}"
        "th,td{border:1px solid #ddd;padding:8px;text-align:left}pre{white-space:pre-wrap}</style>"
        f"</head><body><h1>{html.escape(title)}</h1><table>{''.join(rows)}</table></body></html>"
    )
    return output


def create_training_report(reference_df: pd.DataFrame, output_name: str = "training_data_report.html") -> Path:
    settings = get_settings()
    output = settings.reports_dir / "evidently" / output_name
    try:
        from evidently.legacy.metric_preset import DataQualityPreset
        from evidently.legacy.report import Report

        report = Report(metrics=[DataQualityPreset()])
        report.run(reference_data=None, current_data=reference_df)
        report.save_html(str(output))
        return output
    except Exception as exc:
        LOGGER.info("Using fallback training report because Evidently failed or is unavailable: %s", exc)
        return _fallback_html_report(
            "Training Data Report",
            {
                "rows": len(reference_df),
                "columns": list(reference_df.columns),
                "numeric_summary": reference_df.select_dtypes(include="number").describe().round(3).to_dict(),
                "feature_distribution_baseline": _feature_distribution_changes(reference_df, reference_df),
            },
            output,
        )


def create_daily_monitoring_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    execution_date: str,
) -> tuple[Path, dict[str, float | int]]:
    settings = get_settings()
    output = settings.reports_dir / "evidently" / f"daily_prediction_monitoring_{execution_date}.html"
    drift_output = settings.reports_dir / "evidently" / f"drift_report_{execution_date}.html"
    prediction_drift_output = settings.reports_dir / "evidently" / f"prediction_drift_report_{execution_date}.html"
    data_drift, data_drift_metrics = detect_data_drift(reference_df, current_df)
    prediction_drift, prediction_drift_metrics = detect_prediction_drift(reference_df, predictions_df)
    feature_distribution_changes = _feature_distribution_changes(reference_df, current_df)
    regression_performance = _regression_performance_if_available(current_df, predictions_df)
    metrics = {
        "data_drift_detected": int(data_drift),
        "prediction_drift_detected": int(prediction_drift),
        **data_drift_metrics,
        **prediction_drift_metrics,
    }
    if regression_performance:
        metrics.update({f"regression_{key}": value for key, value in regression_performance.items()})

    try:
        from evidently.legacy.metric_preset import DataDriftPreset
        from evidently.legacy.report import Report

        drift_columns = [
            column for column in current_df.columns if column in reference_df.columns
        ]
        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=reference_df[drift_columns],
            current_data=current_df[drift_columns],
        )
        report.save_html(str(drift_output))
    except Exception as exc:
        LOGGER.info("Using fallback drift report because Evidently failed or is unavailable: %s", exc)
        _fallback_html_report(
            "Data Drift Report",
            {
                "drift_metrics": metrics,
                "feature_distribution_changes": feature_distribution_changes,
            },
            drift_output,
        )

    _fallback_html_report(
        "Prediction Drift Report",
        {
            "prediction_drift_detected": int(prediction_drift),
            "prediction_drift_metrics": prediction_drift_metrics,
            "prediction_summary": predictions_df[PREDICTION_COLUMN].describe().round(3).to_dict(),
        },
        prediction_drift_output,
    )

    if regression_performance:
        regression_output = settings.reports_dir / "evidently" / f"regression_performance_{execution_date}.html"
        _fallback_html_report("Regression Performance Report", regression_performance, regression_output)

    _fallback_html_report(
        "Daily Prediction Monitoring",
        {
            "execution_date": execution_date,
            "input_rows": len(current_df),
            "prediction_rows": len(predictions_df),
            "metrics": metrics,
            "feature_distribution_changes": feature_distribution_changes,
            "regression_performance_when_actuals_available": regression_performance or "actuals unavailable for this batch",
            "prediction_summary": predictions_df.describe(include="all").to_dict(),
        },
        output,
    )
    return output, metrics
