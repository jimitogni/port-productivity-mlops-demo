# Port Productivity MLOps Demo

Production-like MLOps demo for forecasting port terminal discharge productivity. The business process runs on D0 and produces D+1, D+2, and D+3 productivity forecasts by terminal.

## Business Problem

Carlos built a promising notebook model, but the notebook depends on manual execution and has weak traceability, validation, monitoring, reproducibility, and deployment discipline. This project turns that experiment into a pragmatic production-style pipeline.

## Architecture Overview

- Synthetic data generator creates historical terminal operations.
- Training pipeline validates data, builds features, trains baseline and candidate models, logs to MLflow, and registers the best model.
- Controlled promotion moves a Candidate model to Champion/Production only after checks pass.
- Airflow is the core orchestrator for daily batch prediction.
- Daily prediction always writes local CSV artifacts and also stores predictions, execution metadata, model performance, and monitoring metrics in PostgreSQL when `DATABASE_URL` exists.
- Evidently and fallback HTML monitoring reports are saved under `reports/evidently/`.
- Prometheus metrics are generated with `prometheus-client` when available, written in text format, and exposed by the FastAPI demo service.
- Grafana can import `monitoring/grafana/port_productivity_dashboard.json`.
- Docker Compose integrates with the existing homelab Traefik network.
- Kubernetes, KServe, and Kubeflow are optional extension paths.

## Why Airflow Is Core

The requirement is a scheduled daily D0 batch forecast. Airflow gives the clearest first production version: explicit DAGs, retries, logs, reruns, and operational ownership without adding a Kubernetes platform dependency to the core workflow.

## Why MLflow Is Used

MLflow records experiments, parameters, metrics, artifacts, model versions, and aliases. The project also keeps a local registry fallback under `models/registry.json` so the demo works when a remote registry is not available.

## Why Validation Matters

Input validation catches missing terminals, invalid dates, impossible numeric ranges, and equipment availability outside `[0, 1]`. Prediction validation catches missing horizons, missing terminals, null values, negative predictions, and unreasonable productivity values.

## Why Evidently, Prometheus, And Grafana Are Used

Evidently produces human-readable reports for training data and data drift, with fallback HTML reports for prediction drift, feature distribution changes, and regression performance when actuals are available. Prometheus-compatible metrics provide machine-readable operational signals. Grafana gives a dashboard view for interview demos and operational monitoring.

## Why GitHub Actions Is Used

CI runs tests, Python compilation, DAG parsing, and a small training smoke test. Image build workflow pushes to GHCR. Model promotion has a production environment gate so promotion is deliberate.

## How To Run Locally

```bash
cd /home/jimi/projects/port-productivity-mlops-demo
make setup
make generate-data
make train
make promote-model
make predict
```

If dependencies are not installed locally:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The Docker image and GitHub Actions workflow use Python 3.12. Python 3.11 is also suitable.

The Makefile can create the local environment when Python 3.12 and venv support are installed:

```bash
make install
make doctor
```

If the host does not have a suitable Python/pip setup, run the same normal demo through Docker:

```bash
make docker-demo-normal
```

The regular Make targets also fall back to Docker automatically when local Python dependencies are missing. Docker fallback runs with `PROJECT_ROOT=/app` and `MLFLOW_TRACKING_URI=file:///app/mlruns` unless `DOCKER_MLFLOW_TRACKING_URI` is set.

## Airflow DAGs

DAG files live in `dags/`:

- `port_productivity_training_pipeline`
- `port_productivity_daily_prediction_pipeline`

Validate parsing:

```bash
make validate-dags
```

Copy or mount `dags/` into the existing Airflow DAG folder and set environment variables from `.env.example`.

## MLflow Experiments And Registry

Set `MLFLOW_TRACKING_URI` in `.env`. On this homelab, direct local host access is usually:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Public UI through Traefik is documented in the homelab notes as:

```text
http://mlflow-jimitogni.duckdns.org:8888/
```

Registered model name:

```text
port_productivity_forecaster
```

Aliases:

- `Candidate`
- `Challenger`
- `Champion`
- `Production`

Inspect the local fallback registry:

```bash
cat models/registry.json
cat models/latest_training_metrics.json
```

## Prediction Output

Predictions are saved under `data/predictions/` and include:

- `run_id`
- `execution_date`
- `forecast_date`
- `forecast_horizon`
- `terminal_id`
- `predicted_productivity_tons_hour`
- `model_name`
- `model_version`
- `feature_version`
- `pipeline_version`

Execution metadata, model performance snapshots, and monitoring metrics are saved under `data/monitoring/`. If `DATABASE_URL` is set, the same records are also appended to PostgreSQL tables from `sql/create_tables.sql`.

## Evidently Reports

Open generated HTML files from:

```text
reports/evidently/
```

Common files:

- `training_data_report.html`
- `daily_prediction_monitoring_<execution_date>.html`
- `drift_report_<execution_date>.html`
- `prediction_drift_report_<execution_date>.html`
- `regression_performance_<execution_date>.html` when actuals are available

## Prometheus And Grafana

Batch metrics are written to:

```text
data/monitoring/port_productivity_metrics.prom
```

The API exposes:

```bash
curl http://127.0.0.1:8015/metrics
```

Import this Grafana dashboard:

```text
monitoring/grafana/port_productivity_dashboard.json
```

Prometheus can scrape the API `/metrics` endpoint, or you can point a node-exporter textfile collector at `data/monitoring/port_productivity_metrics.prom`.

## API

Run locally:

```bash
make run-api
```

Endpoints:

- `GET /health`
- `GET /metrics`
- `GET /predictions/latest`
- `POST /predict`

Smoke test:

```bash
python scripts/smoke_test_api.py --base-url http://127.0.0.1:8015
```

## Docker Compose And Homelab Traefik

The compose file does not own MLflow, PostgreSQL, Prometheus, Grafana, Airflow, or Traefik. It only runs this project API and attaches to the existing external `traefik_proxy` network.

It does not bind public host ports. Public access goes through the existing host Traefik on server port `80`, while the router exposes public `:8888`.

```bash
make compose-up
curl -H 'Host: jimitogni.duckdns.org:8888' http://127.0.0.1:80/ports-mlops/health
```

Public URL:

```text
http://jimitogni.duckdns.org:8888/ports-mlops/health
```

Optional HTTPS route uses the existing `websecure` entrypoint on host port `8443`:

```text
https://jimitogni.duckdns.org:8443/ports-mlops/health
```

## Kubernetes

Kubernetes files live under `k8s/` and support k3s, kind, or minikube. They are optional:

```bash
make k8s-apply
make k8s-status
make k8s-logs
make k8s-delete
```

Replace `ghcr.io/OWNER/port-productivity-mlops-demo:latest` before applying.

Kubernetes is optional because the business process is a daily batch forecast and Airflow already fits that requirement. The Kubernetes manifests show how the same container can run as an API Deployment, daily CronJob, and training Job when the platform needs cluster-native execution.

## KServe

KServe files live under `kserve/`. This is useful for online inference while Airflow remains the core batch process. The provided manifest uses the same FastAPI container as a custom predictor because feature engineering and MLflow alias lookup are part of the serving path.

Test details and prerequisites are documented in `kserve/README.md`.

## Kubeflow Pipelines

Kubeflow files live under `kubeflow/`. This is a future enterprise option for Kubernetes-native ML workflows. It does not replace Airflow in the first version.

Install optional Kubeflow dependencies only when compiling that extension:

```bash
pip install -r requirements-kubeflow.txt
python kubeflow/pipeline.py
```

## GitHub Actions

Workflows:

- `.github/workflows/ci.yml`
- `.github/workflows/build-and-push.yml`
- `.github/workflows/model-promotion.yml`
- `.github/workflows/deploy-homelab.yml`

On every push to `main`, CI runs first and the image build workflow pushes `latest` and the git SHA tag to GHCR. When the image build workflow succeeds, `deploy-homelab.yml` automatically deploys over SSH to the homelab by packaging the repository, uploading it to `DEPLOY_PATH`, running Docker Compose, and checking `/ports-mlops/health`.

Manual deployment remains available from the Actions tab through `Deploy Homelab`, with `ssh` as the default mode. A self-hosted runner mode is included as an optional fallback.

Required deployment secrets for SSH mode:

- `HOMELAB_HOST`
- `HOMELAB_USER`
- `HOMELAB_SSH_KEY`
- `HOMELAB_SSH_PORT`
- `DEPLOY_PATH`

Optional secrets:

- `MLFLOW_TRACKING_URI`
- `DATABASE_URL`

The deployment does not commit `.env`. On the server, the workflow preserves an existing `.env`; if one is missing it copies `.env.example`, so production values should be edited on the server after the first deployment.

## Demo Scenarios

```bash
make demo-normal
make demo-missing-terminal
make demo-drift
make demo-prediction-drift
make demo-train-challenger
make demo-promote-model
make demo-failed-validation
make demo-recovery
```

These demonstrate happy path, failed validation, data drift, prediction drift, retraining, controlled promotion, and recovery after a validation failure.

## What Is Still Manual

- Connecting the project DAGs into the existing Airflow deployment.
- Configuring Prometheus scrape targets for the API `/metrics` endpoint or textfile collector.
- Importing the Grafana dashboard.
- Creating Kubernetes secrets for cluster deployment.
- Choosing production thresholds for promotion after business review.

## Troubleshooting

- `No registered model was found`: run `make train` and `make promote-model`.
- `Expected terminals missing`: the input data is incomplete; this is intentional in `make demo-missing-terminal`.
- MLflow connection errors: set `MLFLOW_TRACKING_URI` or use the local file fallback.
- PostgreSQL errors: unset `DATABASE_URL` to use local CSV storage.
- Traefik path returns 404: confirm the container is on `traefik_proxy` and the request uses `Host: jimitogni.duckdns.org:8888`.

## Interview Presentation Script

- Start with the business problem.
- Explain that Carlos’s notebook is the experiment.
- Explain that the production code is modular, testable, and reproducible.
- Show Airflow DAGs.
- Show MLflow experiments and model registry.
- Show prediction outputs with run_id and model_version.
- Show validation checks.
- Show Evidently reports.
- Show Prometheus/Grafana metrics.
- Show GitHub Actions CI/CD.
- Explain controlled model promotion.
- Then show Kubernetes as an extension because the company uses Kubernetes.
- Show KServe or the FastAPI deployment as an online inference extension.
- Explain that Kubeflow is optional and useful for Kubernetes-native ML platforms.
- Emphasize that the core solution is not overengineered: Airflow is used for the daily batch requirement, Kubernetes is prepared as an enterprise evolution path.

## How To Explain Kubernetes Without Sounding Overengineered

"The core business requirement is a daily D0 batch forecast for D+1 onward, so Airflow is the most pragmatic orchestrator for the first production version. However, since the target environment uses Kubernetes, I containerized the components and prepared an optional Kubernetes deployment path. This allows the same solution to evolve into Kubernetes-native serving with KServe or workflow orchestration with Kubeflow Pipelines when the business needs scale, online inference, or stronger platform standardization."
