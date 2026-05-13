from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from time import perf_counter

import pandas as pd

from src.config.settings import get_settings
from src.data.generate_synthetic_data import generate_daily_operational_forecast, generate_synthetic_data, save_dataset
from src.data.load_data import load_port_productivity_data
from src.data.save_data import save_execution_metadata, save_monitoring_metrics, save_predictions
from src.features.build_features import build_inference_features
from src.models.predict import load_registered_model, predict_with_bundle
from src.models.register_model import _load_registry
from src.monitoring.evidently_report import create_daily_monitoring_report
from src.monitoring.prometheus_metrics import write_prometheus_metrics
from src.utils.dates import parse_date
from src.utils.logging import get_logger
from src.validation.validate_input_data import ValidationError, validate_input_data
from src.validation.validate_predictions import validate_predictions


LOGGER = get_logger(__name__)


def _ensure_history_exists(execution_date: str) -> pd.DataFrame:
    settings = get_settings()
    if not settings.raw_data_path.exists():
        LOGGER.info("Historical data missing; generating synthetic history up to %s", execution_date)
        history = generate_synthetic_data("2024-01-01", execution_date)
        save_dataset(history, settings.raw_data_path)
    return load_port_productivity_data(settings.raw_data_path)


def _prediction_frame(
    input_df: pd.DataFrame,
    predictions: pd.Series,
    run_id: str,
    execution_date: str,
    model_name: str,
    model_version: str,
) -> pd.DataFrame:
    settings = get_settings()
    created_at = pd.Timestamp.utcnow().isoformat()
    output = input_df[["forecast_date", "forecast_horizon", "terminal_id"]].copy()
    output["prediction_id"] = [str(uuid.uuid4()) for _ in range(len(output))]
    output["run_id"] = run_id
    output["execution_date"] = execution_date
    output["predicted_productivity_tons_hour"] = predictions.round(2).values
    output["model_name"] = model_name
    output["model_version"] = str(model_version)
    output["feature_version"] = settings.feature_version
    output["pipeline_version"] = settings.pipeline_version
    output["created_at"] = created_at
    return output[
        [
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
        ]
    ]


def _model_metric_defaults(model_version: str) -> dict[str, float]:
    try:
        registry = _load_registry()
        metadata = registry.get("versions", {}).get(str(model_version), {})
        metrics = metadata.get("metrics", {})
        return {
            "model_mae": float(metrics.get("mae", 0)),
            "model_rmse": float(metrics.get("rmse", 0)),
        }
    except Exception:
        return {"model_mae": 0.0, "model_rmse": 0.0}


def run_daily_prediction_pipeline(
    execution_date: str,
    input_path: str | Path | None = None,
    scenario: str = "normal",
) -> dict[str, object]:
    settings = get_settings()
    run_id = str(uuid.uuid4())
    start = pd.Timestamp.utcnow()
    timer = perf_counter()
    metadata = {
        "run_id": run_id,
        "pipeline_name": "port_productivity_daily_prediction_pipeline",
        "execution_date": execution_date,
        "start_time": start.isoformat(),
        "end_time": None,
        "status": "running",
        "input_rows": 0,
        "prediction_rows": 0,
        "model_name": settings.model_name,
        "model_version": None,
        "error_message": "",
        "created_at": start.isoformat(),
    }
    validation_failures = 0
    try:
        parse_date(execution_date)
        history_df = _ensure_history_exists(execution_date)
        validate_input_data(history_df, require_target=True)
        if input_path:
            input_df = pd.read_csv(input_path)
        else:
            input_df = generate_daily_operational_forecast(execution_date, scenario=scenario)
            input_output = settings.data_dir / "processed" / f"daily_operational_{execution_date}.csv"
            save_dataset(input_df, input_output)
        metadata["input_rows"] = len(input_df)
        validate_input_data(input_df, require_target=False)
        features = build_inference_features(input_df, history_df)
        bundle = load_registered_model()
        predictions = predict_with_bundle(bundle, features)
        prediction_df = _prediction_frame(
            input_df=input_df,
            predictions=predictions,
            run_id=run_id,
            execution_date=execution_date,
            model_name=bundle.model_name,
            model_version=bundle.model_version,
        )
        validate_predictions(prediction_df)
        prediction_path = save_predictions(prediction_df, execution_date)
        metadata["prediction_rows"] = len(prediction_df)
        metadata["model_version"] = bundle.model_version
        report_path, monitoring_metrics = create_daily_monitoring_report(
            reference_df=history_df,
            current_df=input_df,
            predictions_df=prediction_df,
            execution_date=execution_date,
        )
        metadata["status"] = "success"
        duration = perf_counter() - timer
        model_metrics = _model_metric_defaults(bundle.model_version)
        metrics = {
            "pipeline_success": 1,
            "pipeline_duration_seconds": duration,
            "input_rows_total": int(metadata["input_rows"]),
            "prediction_rows_total": int(metadata["prediction_rows"]),
            "validation_failures_total": validation_failures,
            "model_mae": model_metrics["model_mae"],
            "model_rmse": model_metrics["model_rmse"],
            "data_drift_detected": int(monitoring_metrics["data_drift_detected"]),
            "prediction_drift_detected": int(monitoring_metrics["prediction_drift_detected"]),
            "current_model_version": int(float(bundle.model_version)) if str(bundle.model_version).isdigit() else 0,
        }
        metrics.update({key: value for key, value in monitoring_metrics.items() if key not in metrics})
        write_prometheus_metrics(metrics)
        monitoring_metrics_path = save_monitoring_metrics(metrics, run_id=run_id, execution_date=execution_date)
        result = {
            "run_id": run_id,
            "status": "success",
            "prediction_path": str(prediction_path),
            "monitoring_report": str(report_path),
            "metrics_path": str(settings.prometheus_metrics_path),
            "monitoring_metrics_path": str(monitoring_metrics_path),
            "model_version": bundle.model_version,
            "prediction_rows": len(prediction_df),
        }
        LOGGER.info("Daily prediction complete: %s", result)
        return result
    except ValidationError as exc:
        validation_failures += 1
        metadata["error_message"] = str(exc)
        raise
    except Exception as exc:
        metadata["error_message"] = str(exc)
        raise
    finally:
        metadata["end_time"] = pd.Timestamp.utcnow().isoformat()
        if metadata["status"] == "running":
            metadata["status"] = "failed"
            metadata["error_message"] = metadata["error_message"] or "See pipeline logs for stack trace"
            failure_metrics = {
                "pipeline_success": 0,
                "pipeline_duration_seconds": perf_counter() - timer,
                "input_rows_total": int(metadata["input_rows"]),
                "prediction_rows_total": int(metadata["prediction_rows"]),
                "validation_failures_total": validation_failures,
                "model_mae": 0,
                "model_rmse": 0,
                "data_drift_detected": 0,
                "prediction_drift_detected": 0,
                "current_model_version": 0,
            }
            write_prometheus_metrics(failure_metrics)
            save_monitoring_metrics(failure_metrics, run_id=run_id, execution_date=execution_date)
        save_execution_metadata(metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run daily D+1/D+2/D+3 batch predictions.")
    parser.add_argument("--execution-date", required=True)
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--scenario", default="normal", choices=["normal", "missing_terminal", "drift", "prediction_drift"])
    args = parser.parse_args()
    result = run_daily_prediction_pipeline(args.execution_date, args.input_path, args.scenario)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
