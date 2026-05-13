from __future__ import annotations

from kfp import dsl


IMAGE = "ghcr.io/OWNER/port-productivity-mlops-demo:latest"


@dsl.pipeline(name="port-productivity-training")
def port_productivity_training_pipeline(
    start_date: str = "2024-01-01",
    end_date: str = "2026-05-12",
    data_path: str = "/app/data/raw/port_productivity.csv",
):
    generate = dsl.ContainerOp(
        name="generate-data",
        image=IMAGE,
        command=["python", "-m", "src.data.generate_synthetic_data"],
        arguments=["--start-date", start_date, "--end-date", end_date, "--output", data_path],
    )
    validate = dsl.ContainerOp(
        name="validate-data",
        image=IMAGE,
        command=["python", "-m", "src.validation.validate_input_data"],
    ).after(generate)
    build_features = dsl.ContainerOp(
        name="build-features",
        image=IMAGE,
        command=["python", "-m", "src.features.build_features"],
    ).after(validate)
    train = dsl.ContainerOp(
        name="train-model",
        image=IMAGE,
        command=["python", "-m", "src.pipelines.training_pipeline"],
        arguments=["--data-path", data_path],
    ).after(build_features)
    dsl.ContainerOp(
        name="register-and-report",
        image=IMAGE,
        command=["python", "-m", "src.models.promote_model"],
    ).after(train)


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(port_productivity_training_pipeline, "port_productivity_training_pipeline.yaml")

