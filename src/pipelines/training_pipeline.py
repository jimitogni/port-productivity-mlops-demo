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
    all_models: dict[str, object],
    all_metrics: dict[str, dict[str, float]],
    winner_type: str,
    terminal_metrics: dict[str, dict[str, float]],
    dataset_path: str,
    training_period: dict[str, str],
    feature_columns_path: Path,
    evidently_report_path: Path | None = None,
    best_params: dict | None = None,
) -> str:
    """Log one parent MLflow run with a nested child run per trained model.

    The child runs (DummyRegressor, RandomForestRegressor, XGBRegressor) each
    carry the same held-out test-set metrics, so they can be selected and
    compared side-by-side in the MLflow UI. The winning model is registered to
    the model registry from its own child run with the ``Candidate`` alias;
    the losers still get their artifact logged for inspection.
    """
    settings = get_settings()
    run_id = str(uuid.uuid4())
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        LOGGER.warning("MLflow is not installed; using local run id %s", run_id)
        return run_id

    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment("port_productivity_training")
        with mlflow.start_run(run_name=f"training-{run_id}") as parent_run:
            run_id = parent_run.info.run_id
            mlflow.set_tag("pipeline", "training")
            mlflow.set_tag("model_name", settings.model_name)
            mlflow.set_tag("winner_model_type", winner_type)
            mlflow.log_param("dataset_path", dataset_path)
            mlflow.log_param("training_start_date", training_period["start_date"])
            mlflow.log_param("training_end_date", training_period["end_date"])
            if best_params:
                mlflow.log_params({k: str(v) for k, v in best_params.items() if v is not None})
            # Parent run carries every model's metrics prefixed by type, so it
            # doubles as a one-glance summary of the comparison.
            for name, metrics in all_metrics.items():
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{name}_{metric_name}", value)

            for name, model in all_models.items():
                is_winner = name == winner_type
                with mlflow.start_run(run_name=name, nested=True):
                    mlflow.set_tag("pipeline", "training")
                    mlflow.set_tag("model_type", name)
                    mlflow.set_tag("winner", str(is_winner).lower())
                    mlflow.log_param("dataset_path", dataset_path)
                    for metric_name, value in all_metrics[name].items():
                        mlflow.log_metric(metric_name, value)
                    if is_winner:
                        # Register the selected model (and log its artifact)
                        # from its own child run, with the Candidate alias.
                        register_mlflow_model(
                            model,
                            "model",
                            feature_columns_path,
                            metrics=all_metrics[name],
                            terminal_metrics=terminal_metrics,
                            feature_columns=expected_feature_columns(),
                            alias="Candidate",
                        )
                    else:
                        # Losers are not registered, but their artifact is
                        # still logged so the comparison is fully inspectable.
                        try:
                            mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
                        except Exception as exc:
                            LOGGER.warning("Could not log %s artifact: %s", name, exc)

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
    candidates, candidate_type, best_params = train_candidate_model(X_train, y_train, random_state=random_state)

    # Score every model on the same held-out test set so the MLflow child runs
    # are a fair, apples-to-apples comparison.
    all_models = {"DummyRegressor": baseline, **candidates}
    all_metrics = {
        name: regression_metrics(y_test, model.predict(X_test))
        for name, model in all_models.items()
    }
    baseline_metrics = all_metrics["DummyRegressor"]
    candidate_metrics = all_metrics[candidate_type]

    # Candidate must beat the mean-baseline on test RMSE to be selected.
    model_type = candidate_type if candidate_metrics["rmse"] <= baseline_metrics["rmse"] else "DummyRegressor"
    best_model = all_models[model_type]
    best_metrics = all_metrics[model_type]
    best_predictions = best_model.predict(X_test)

    terminal_metrics = terminal_level_metrics(test_df[["terminal_id", "forecast_horizon"]], y_test, best_predictions)
    feature_columns_path = save_feature_columns()
    training_period = {
        "start_date": str(validated_df["operation_date"].min().date()),
        "end_date": str(validated_df["operation_date"].max().date()),
    }
    report_path = create_training_report(validated_df)
    run_id = _log_mlflow_run(
        all_models,
        all_metrics,
        model_type,
        terminal_metrics,
        str(data_path),
        training_period,
        feature_columns_path,
        evidently_report_path=report_path,
        best_params=best_params,
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
