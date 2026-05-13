from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

import pandas as pd

from src.config.settings import get_settings
from src.data.load_data import load_port_productivity_data
from src.data.save_data import save_model_performance
from src.features.build_features import TARGET_COLUMN, build_training_dataset, expected_feature_columns, save_feature_columns
from src.models.evaluate_model import regression_metrics, terminal_level_metrics
from src.models.register_model import register_local_model, register_mlflow_model
from src.models.train_model import train_baseline_model, train_candidate_model
from src.monitoring.evidently_report import create_training_report
from src.utils.logging import get_logger
from src.validation.validate_input_data import validate_input_data


LOGGER = get_logger(__name__)


def _time_split(features: pd.DataFrame, train_fraction: float = 0.80) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(pd.to_datetime(features["operation_date"]).unique())
    split_index = max(1, int(len(dates) * train_fraction))
    split_date = dates[split_index - 1]
    train_df = features[pd.to_datetime(features["operation_date"]) <= split_date].copy()
    test_df = features[pd.to_datetime(features["operation_date"]) > split_date].copy()
    if test_df.empty:
        raise RuntimeError("Time split produced an empty test set; generate a longer date range")
    return train_df, test_df


def _log_mlflow_run(
    best_model,
    model_type: str,
    metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    terminal_metrics: dict[str, dict[str, float]],
    dataset_path: str,
    training_period: dict[str, str],
    feature_columns_path: Path,
    evidently_report_path: Path | None = None,
) -> str:
    settings = get_settings()
    run_id = str(uuid.uuid4())
    try:
        import mlflow
    except ImportError:
        LOGGER.warning("MLflow is not installed; using local run id %s", run_id)
        return run_id

    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment("port_productivity_training")
        with mlflow.start_run(run_name=f"training-{run_id}") as run:
            run_id = run.info.run_id
            mlflow.set_tag("pipeline", "training")
            mlflow.set_tag("model_name", settings.model_name)
            mlflow.set_tag("model_type", model_type)
            mlflow.log_param("dataset_path", dataset_path)
            mlflow.log_param("training_start_date", training_period["start_date"])
            mlflow.log_param("training_end_date", training_period["end_date"])
            for name, value in baseline_metrics.items():
                mlflow.log_metric(f"baseline_{name}", value)
            for name, value in metrics.items():
                mlflow.log_metric(name, value)
            register_mlflow_model(
                best_model,
                "model",
                feature_columns_path,
                metrics=metrics,
                terminal_metrics=terminal_metrics,
                feature_columns=expected_feature_columns(),
                alias="Candidate",
            )
            if evidently_report_path is not None and evidently_report_path.exists():
                mlflow.log_artifact(str(evidently_report_path), artifact_path="evidently")
    except Exception as exc:
        LOGGER.warning("MLflow logging failed; local registry still captures the model: %s", exc)
    return run_id


def run_training_pipeline(data_path: str | Path | None = None, random_state: int = 42) -> dict[str, object]:
    settings = get_settings()
    data_path = Path(data_path or settings.raw_data_path)
    raw_df = load_port_productivity_data(data_path)
    validated_df = validate_input_data(raw_df, require_target=True)
    features = build_training_dataset(validated_df)
    feature_columns = expected_feature_columns()
    train_df, test_df = _time_split(features)

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN]

    baseline = train_baseline_model(X_train, y_train)
    candidate = train_candidate_model(X_train, y_train, random_state=random_state)
    baseline_predictions = baseline.predict(X_test)
    candidate_predictions = candidate.predict(X_test)
    baseline_metrics = regression_metrics(y_test, baseline_predictions)
    candidate_metrics = regression_metrics(y_test, candidate_predictions)

    if candidate_metrics["rmse"] <= baseline_metrics["rmse"]:
        best_model = candidate
        best_metrics = candidate_metrics
        model_type = "RandomForestRegressor"
        best_predictions = candidate_predictions
    else:
        best_model = baseline
        best_metrics = baseline_metrics
        model_type = "DummyRegressor"
        best_predictions = baseline_predictions

    terminal_metrics = terminal_level_metrics(test_df[["terminal_id", "forecast_horizon"]], y_test, best_predictions)
    feature_columns_path = save_feature_columns()
    training_period = {
        "start_date": str(validated_df["operation_date"].min().date()),
        "end_date": str(validated_df["operation_date"].max().date()),
    }
    report_path = create_training_report(validated_df)
    run_id = _log_mlflow_run(
        best_model,
        model_type,
        best_metrics,
        baseline_metrics,
        terminal_metrics,
        str(data_path),
        training_period,
        feature_columns_path,
        evidently_report_path=report_path,
    )
    local_version = register_local_model(
        best_model,
        metrics=best_metrics,
        terminal_metrics=terminal_metrics,
        feature_columns=feature_columns,
        run_id=run_id,
        dataset_path=str(data_path),
        training_period=training_period,
        model_type=model_type,
        alias="Candidate",
    )
    performance_path = save_model_performance(
        metrics=best_metrics,
        terminal_metrics=terminal_metrics,
        run_id=run_id,
        model_name=settings.model_name,
        model_version=local_version,
        evaluation_date=training_period["end_date"],
    )
    metrics_path = settings.models_dir / "latest_training_metrics.json"
    result = {
        "run_id": run_id,
        "model_name": settings.model_name,
        "local_model_version": local_version,
        "model_type": model_type,
        "metrics": best_metrics,
        "baseline_metrics": baseline_metrics,
        "training_period": training_period,
        "training_rows": len(train_df),
        "test_rows": len(test_df),
        "evidently_report": str(report_path),
        "model_performance_path": str(performance_path),
    }
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    LOGGER.info("Training complete: %s", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the port productivity training pipeline.")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    result = run_training_pipeline(args.data_path, args.random_state)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
