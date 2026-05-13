from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config.settings import get_settings
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)


def _write_sql_if_configured(df: pd.DataFrame, table_name: str) -> None:
    settings = get_settings()
    if not settings.database_url:
        return
    try:
        from sqlalchemy import create_engine
    except ImportError as exc:
        raise RuntimeError("DATABASE_URL is set but sqlalchemy is not installed") from exc
    engine = create_engine(settings.database_url)
    with engine.begin() as connection:
        df.to_sql(table_name, connection, if_exists="append", index=False)
    LOGGER.info("Inserted %s rows into %s", len(df), table_name)


def save_predictions(df: pd.DataFrame, execution_date: str) -> Path:
    settings = get_settings()
    settings.predictions_dir.mkdir(parents=True, exist_ok=True)
    output = settings.predictions_dir / f"predictions_{execution_date}.csv"
    df.to_csv(output, index=False)
    df.to_csv(settings.predictions_dir / "latest_predictions.csv", index=False)
    _write_sql_if_configured(df, "predictions")
    LOGGER.info("Saved predictions to %s", output)
    return output


def save_execution_metadata(metadata: dict[str, Any]) -> Path:
    settings = get_settings()
    settings.monitoring_dir.mkdir(parents=True, exist_ok=True)
    execution_date = str(metadata["execution_date"])
    output = settings.monitoring_dir / f"execution_metadata_{execution_date}.csv"
    df = pd.DataFrame([metadata])
    df.to_csv(output, index=False)
    df.to_csv(settings.monitoring_dir / "latest_execution_metadata.csv", index=False)
    _write_sql_if_configured(df, "execution_metadata")
    LOGGER.info("Saved execution metadata to %s", output)
    return output


def save_monitoring_metrics(
    metrics: dict[str, float | int],
    run_id: str,
    execution_date: str,
) -> Path:
    settings = get_settings()
    settings.monitoring_dir.mkdir(parents=True, exist_ok=True)
    created_at = pd.Timestamp.utcnow().isoformat()
    rows = [
        {
            "run_id": run_id,
            "execution_date": execution_date,
            "metric_name": name,
            "metric_value": float(value),
            "created_at": created_at,
        }
        for name, value in metrics.items()
    ]
    df = pd.DataFrame(rows)
    output = settings.monitoring_dir / f"monitoring_metrics_{execution_date}.csv"
    df.to_csv(output, index=False)
    df.to_csv(settings.monitoring_dir / "latest_monitoring_metrics.csv", index=False)
    _write_sql_if_configured(df, "monitoring_metrics")
    LOGGER.info("Saved monitoring metrics to %s", output)
    return output


def save_model_performance(
    metrics: dict[str, float],
    terminal_metrics: dict[str, dict[str, float]],
    run_id: str,
    model_name: str,
    model_version: str,
    evaluation_date: str,
) -> Path:
    settings = get_settings()
    settings.monitoring_dir.mkdir(parents=True, exist_ok=True)
    created_at = pd.Timestamp.utcnow().isoformat()

    def row(
        metric_values: dict[str, float],
        terminal_id: str | None = None,
        forecast_horizon: str | None = None,
    ) -> dict[str, Any]:
        return {
            "model_name": model_name,
            "model_version": str(model_version),
            "run_id": run_id,
            "evaluation_date": evaluation_date,
            "terminal_id": terminal_id,
            "forecast_horizon": forecast_horizon,
            "mae": metric_values.get("mae"),
            "rmse": metric_values.get("rmse"),
            "r2": metric_values.get("r2"),
            "mape": metric_values.get("mape"),
            "created_at": created_at,
        }

    rows: list[dict[str, Any]] = [row(metrics)]
    for key, values in terminal_metrics.items():
        if str(key).startswith("D+"):
            rows.append(row(values, forecast_horizon=str(key)))
        else:
            rows.append(row(values, terminal_id=str(key)))

    df = pd.DataFrame(rows)
    safe_run_id = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in run_id)
    output = settings.monitoring_dir / f"model_performance_{safe_run_id}.csv"
    df.to_csv(output, index=False)
    df.to_csv(settings.monitoring_dir / "latest_model_performance.csv", index=False)
    _write_sql_if_configured(df, "model_performance")
    LOGGER.info("Saved model performance metrics to %s", output)
    return output
