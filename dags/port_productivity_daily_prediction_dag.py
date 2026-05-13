from __future__ import annotations

from datetime import datetime
from pathlib import Path


try:
    from airflow.decorators import dag, task
except Exception:
    dag = None
    task = None


if dag:

    @dag(
        dag_id="port_productivity_daily_prediction_pipeline",
        start_date=datetime(2024, 1, 1),
        schedule="@daily",
        catchup=False,
        tags=["mlops", "port-productivity", "prediction"],
        default_args={"owner": "mlops", "retries": 1},
        doc_md=__doc__,
    )
    def port_productivity_daily_prediction_pipeline():
        @task(task_id="generate_or_load_daily_operational_data")
        def generate_or_load_daily_operational_data(execution_date: str) -> str:
            from src.config.settings import get_settings
            from src.data.generate_synthetic_data import generate_daily_operational_forecast, save_dataset

            settings = get_settings()
            output = settings.data_dir / "processed" / f"daily_operational_{execution_date}.csv"
            df = generate_daily_operational_forecast(execution_date)
            save_dataset(df, output)
            return str(output)

        @task(task_id="validate_input_data")
        def validate_input_data_task(input_path: str) -> str:
            import pandas as pd

            from src.validation.validate_input_data import validate_input_data

            validate_input_data(pd.read_csv(input_path), require_target=False)
            return input_path

        @task(task_id="build_inference_features")
        def build_inference_features_task(input_path: str) -> str:
            import pandas as pd

            from src.config.settings import get_settings
            from src.data.load_data import load_port_productivity_data
            from src.features.build_features import build_inference_features

            settings = get_settings()
            features = build_inference_features(pd.read_csv(input_path), load_port_productivity_data())
            output = settings.data_dir / "processed" / f"inference_features_{Path(input_path).stem}.csv"
            features.to_csv(output, index=False)
            return str(output)

        @task(task_id="load_production_model_from_mlflow")
        def load_production_model_from_mlflow() -> dict:
            from src.models.predict import load_registered_model

            bundle = load_registered_model()
            return {"model_name": bundle.model_name, "model_version": bundle.model_version, "source": bundle.source}

        @task(task_id="generate_predictions_for_d_plus_1_d_plus_2_d_plus_3")
        def generate_predictions_for_d_plus_1_d_plus_2_d_plus_3(execution_date: str, input_path: str, model_info: dict) -> dict:
            from src.pipelines.daily_prediction_pipeline import run_daily_prediction_pipeline

            if not model_info.get("model_version"):
                raise RuntimeError("Model version missing")
            return run_daily_prediction_pipeline(execution_date=execution_date, input_path=input_path)

        @task(task_id="validate_predictions")
        def validate_predictions_task(result: dict) -> dict:
            import pandas as pd

            from src.validation.validate_predictions import validate_predictions

            validate_predictions(pd.read_csv(result["prediction_path"]))
            return result

        @task(task_id="save_predictions")
        def save_predictions_task(result: dict) -> str:
            prediction_path = result["prediction_path"]
            if not Path(prediction_path).exists():
                raise RuntimeError("Prediction file was not created")
            return prediction_path

        @task(task_id="save_execution_metadata")
        def save_execution_metadata_task(result: dict) -> str:
            from src.config.settings import get_settings

            settings = get_settings()
            latest = settings.monitoring_dir / "latest_execution_metadata.csv"
            if not latest.exists():
                raise RuntimeError("Execution metadata file was not created")
            return str(latest)

        @task(task_id="generate_evidently_monitoring_report")
        def generate_evidently_monitoring_report(result: dict) -> str:
            report = result["monitoring_report"]
            if not Path(report).exists():
                raise RuntimeError("Monitoring report was not created")
            return report

        @task(task_id="export_prometheus_metrics")
        def export_prometheus_metrics(result: dict) -> str:
            metrics_path = result["metrics_path"]
            if not Path(metrics_path).exists():
                raise RuntimeError("Prometheus metrics file was not created")
            return metrics_path

        input_path = generate_or_load_daily_operational_data("{{ ds }}")
        validated_input = validate_input_data_task(input_path)
        build_inference_features_task(validated_input)
        model_info = load_production_model_from_mlflow()
        result = generate_predictions_for_d_plus_1_d_plus_2_d_plus_3("{{ ds }}", validated_input, model_info)
        validated_result = validate_predictions_task(result)
        save_predictions_task(validated_result)
        save_execution_metadata_task(validated_result)
        generate_evidently_monitoring_report(validated_result)
        export_prometheus_metrics(validated_result)

    port_productivity_daily_prediction_pipeline()

else:
    port_productivity_daily_prediction_pipeline = None

