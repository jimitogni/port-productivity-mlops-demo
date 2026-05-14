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
        dag_id="port_productivity_training_pipeline",
        start_date=datetime(2024, 1, 1),
        schedule="@weekly",
        catchup=False,
        tags=["mlops", "port-productivity", "training"],
        default_args={"owner": "mlops", "retries": 1},
        doc_md=__doc__,
    )
    def port_productivity_training_pipeline():
        @task(task_id="generate_synthetic_historical_data")
        def generate_synthetic_historical_data(**context) -> str:
            from src.config.settings import get_settings
            from src.data.generate_synthetic_data import generate_synthetic_data, save_dataset

            # Optional overrides via "Trigger DAG w/ config" (dag_run.conf):
            #   end_date: extend the range past today for more rows (data only,
            #             not the scheduler's logical date — future dates OK here)
            #   scenario: "normal" | "drift" | "prediction_drift" | "missing_terminal"
            #   seed:     different sample over the same range
            # No conf -> identical behaviour to before ({{ ds }}, normal, seed 42).
            conf = (context.get("dag_run").conf or {}) if context.get("dag_run") else {}
            end_date = conf.get("end_date", context["ds"])
            scenario = conf.get("scenario", "normal")
            seed = int(conf.get("seed", 42))

            settings = get_settings()
            df = generate_synthetic_data(
                "2024-01-01", end_date, scenario=scenario, seed=seed
            )
            return str(save_dataset(df, settings.raw_data_path))

        @task(task_id="validate_training_data")
        def validate_training_data(data_path: str) -> str:
            from src.data.load_data import load_port_productivity_data
            from src.validation.validate_input_data import validate_input_data

            df = load_port_productivity_data(data_path)
            validate_input_data(df, require_target=True)
            return data_path

        @task(task_id="build_training_features")
        def build_training_features(data_path: str) -> str:
            from src.config.settings import get_settings
            from src.data.load_data import load_port_productivity_data
            from src.features.build_features import build_training_dataset, save_feature_columns

            settings = get_settings()
            features = build_training_dataset(load_port_productivity_data(data_path))
            output = settings.data_dir / "processed" / "training_features.csv"
            output.parent.mkdir(parents=True, exist_ok=True)
            features.to_csv(output, index=False)
            save_feature_columns()
            return str(output)

        @task(task_id="train_candidate_models")
        def train_candidate_models(data_path: str) -> dict:
            from src.pipelines.training_pipeline import run_training_pipeline

            return run_training_pipeline(data_path)

        @task(task_id="evaluate_models")
        def evaluate_models(training_result: dict) -> dict:
            if "metrics" not in training_result:
                raise RuntimeError("Training result missing metrics")
            return training_result

        @task(task_id="log_experiments_to_mlflow")
        def log_experiments_to_mlflow(training_result: dict) -> dict:
            if not training_result.get("run_id"):
                raise RuntimeError("Training result missing MLflow/local run_id")
            return training_result

        @task(task_id="register_best_model")
        def register_best_model(training_result: dict) -> str:
            version = training_result.get("local_model_version")
            if not version:
                raise RuntimeError("Training result missing registered local model version")
            return str(version)

        @task(task_id="generate_evidently_training_report")
        def generate_evidently_training_report(training_result: dict) -> str:
            report = training_result.get("evidently_report")
            if not report or not Path(report).exists():
                raise RuntimeError("Training Evidently report was not created")
            return str(report)

        data_path = generate_synthetic_historical_data()
        validated_path = validate_training_data(data_path)
        build_training_features(validated_path)
        result = train_candidate_models(validated_path)
        evaluated = evaluate_models(result)
        logged = log_experiments_to_mlflow(evaluated)
        register_best_model(logged) >> generate_evidently_training_report(logged)

    port_productivity_training_pipeline()

else:
    port_productivity_training_pipeline = None

