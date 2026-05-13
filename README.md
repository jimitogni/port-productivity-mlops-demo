# Port Productivity MLOps Demo

Production-style MLOps pipeline that forecasts daily discharge productivity (tons/hour) per port terminal. The business process runs on D0 and produces D+1, D+2, and D+3 forecasts. The repository turns a notebook prototype into a reproducible, observable, and deployable pipeline integrated with the existing homelab MLflow, Prometheus, Grafana, and Traefik stack.

## Table of Contents

- [Business Problem](#business-problem)
- [Architecture](#architecture)
- [Tools, Libraries, and Techniques](#tools-libraries-and-techniques)
- [Project Layout](#project-layout)
- [Quickstart (Docker-only path)](#quickstart-docker-only-path)
- [Local Python Path](#local-python-path)
- [End-to-End Pipeline](#end-to-end-pipeline)
- [URLs and Endpoints](#urls-and-endpoints)
- [Configuration](#configuration)
- [Airflow DAGs](#airflow-dags)
- [Kubernetes / KServe / Kubeflow (optional)](#kubernetes--kserve--kubeflow-optional)
- [CI/CD with GitHub Actions](#cicd-with-github-actions)
- [Demo Scenarios](#demo-scenarios)
- [Troubleshooting](#troubleshooting)

---

## Business Problem

A notebook prototype produced reasonable forecasts but lacked traceability, validation, monitoring, reproducibility, and deployment discipline. This project replaces that notebook with a daily batch pipeline that:

- Generates synthetic operational history for terminals `T1–T4`.
- Validates input data against business rules.
- Builds features, trains a baseline (`DummyRegressor`) and a candidate (`RandomForestRegressor`), and registers the better model.
- Promotes only after explicit gating (`Production` alias).
- Produces D+1/D+2/D+3 productivity forecasts daily, with traceable `run_id`, `model_version`, `feature_version`, and `pipeline_version` on every row.
- Emits Prometheus metrics, Evidently HTML reports, and structured CSV monitoring records on every run.

## Architecture

```
┌──────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│ Synthetic Data   │──▶│ Training Pipeline    │──▶│ MLflow (Postgres)    │
│ Generator        │   │ (sklearn + Evidently)│   │ experiments/registry │
└──────────────────┘   └──────────────────────┘   └──────────────────────┘
        │                       │                            │
        │                       ▼                            ▼
        │              ┌──────────────────┐         ┌──────────────────┐
        │              │ Local Registry   │         │ Promotion (CI/CD)│
        │              │ models/registry  │         │ Production alias │
        │              └──────────────────┘         └──────────────────┘
        ▼                                                    │
┌──────────────────┐   ┌──────────────────────┐              ▼
│ Airflow DAG      │──▶│ Daily Prediction     │──▶ predictions/*.csv
│ (D0 → D+1..D+3)  │   │ Pipeline             │──▶ Evidently HTML reports
└──────────────────┘   └──────────────────────┘──▶ Prometheus textfile
                                  │                Prometheus client metrics
                                  ▼
                       ┌──────────────────────┐
                       │ FastAPI service      │──▶ Traefik (homelab)
                       │ /health /metrics …   │   public path /ports-mlops
                       └──────────────────────┘
```

## Tools, Libraries, and Techniques

### Machine Learning

| Concern | Choice | Why |
|---|---|---|
| Baseline model | `sklearn.dummy.DummyRegressor` | Sanity floor for any candidate |
| Candidate model | `sklearn.ensemble.RandomForestRegressor` | Strong tabular baseline, no GPU required |
| Train/test split | Time-based (80/20 by `operation_date`) | Reflects real forecasting use case, avoids leakage |
| Evaluation | MAE, RMSE, R², MAPE — overall + per terminal × horizon | Operational stakeholders care about per-terminal accuracy |
| Feature engineering | `src/features/build_features.py` (`v1`) | Versioned via `FEATURE_VERSION` env var |
| Reproducibility | `random_state=42`, pinned requirements, deterministic synthetic data | Re-runs match |

### MLOps

| Concern | Tool | Notes |
|---|---|---|
| Experiment tracking | **MLflow** (`mlflow>=2.13,<4`) | Backed by Postgres on the homelab; experiment `port_productivity_training` |
| Model registry | MLflow Registered Models + local JSON fallback | `port_productivity_forecaster`; aliases `Candidate` / `Challenger` / `Champion` / `Production` |
| Artifact storage | MLflow `--serve-artifacts` (named volume `mlflow_artifacts`) | Evidently HTML reports logged per run under `evidently/` |
| Orchestration | **Airflow** DAGs in `dags/` | Validated with `make validate-dags` |
| Containerization | **Docker** (`docker/Dockerfile`) + **Docker Compose** (`docker-compose.demo.yml`) | API attaches to external `traefik_proxy` network |
| Reverse proxy | **Traefik** (provided by homelab) | HTTP entrypoint `web`, HTTPS `websecure`, cert resolver `duckdns` |
| CI/CD | **GitHub Actions** | CI, image build → GHCR, manual promotion gate, SSH deploy |
| Optional cluster path | **Kubernetes** (`k8s/`), **KServe** (`kserve/`), **Kubeflow Pipelines** (`kubeflow/`) | Same image, cluster-native execution |

### Monitoring and Observability

| Layer | Tool | Output |
|---|---|---|
| Data quality (training) | **Evidently** `DataQualityPreset` (legacy API on v0.7.x) | `reports/evidently/training_data_report.html`, also logged as MLflow artifact |
| Data drift (daily) | **Evidently** `DataDriftPreset` | `reports/evidently/drift_report_<date>.html` |
| Prediction drift | Custom report + fallback HTML | `reports/evidently/prediction_drift_report_<date>.html` |
| Operational metrics | **prometheus-client** | API `/metrics` endpoint + textfile `data/monitoring/port_productivity_metrics.prom` |
| Dashboards | **Grafana** | Import `monitoring/grafana/port_productivity_dashboard.json` |
| Structured records | CSV + (optional) PostgreSQL | `data/monitoring/{model_performance,monitoring_metrics,execution_metadata}_<date>.csv`, schema in `sql/create_tables.sql` |
| Health checks | FastAPI `/health` + Docker `HEALTHCHECK` | Used by Traefik routing |
| Validation guard rails | `src/validation/` (pydantic + business rules) | Catches missing terminals, bad ranges, null predictions, etc. |

### Supporting Libraries

- **FastAPI** + **uvicorn** — serves predictions and metrics.
- **Pydantic v2** — request/response validation.
- **SQLAlchemy** + **psycopg2-binary** — optional PostgreSQL writes when `DATABASE_URL` is set.
- **pandas**, **numpy** — feature engineering and IO.
- **joblib** — model serialization for the local registry fallback.
- **python-dotenv** — `.env` loading.
- **pytest** — unit and integration tests under `tests/`.

## Project Layout

```
src/
├── api/                FastAPI service
├── config/             Settings + dotenv loader
├── data/               Synthetic data generator, IO helpers
├── features/           Versioned feature builder
├── models/             Train, predict, register, promote
├── monitoring/         Evidently + Prometheus emitters
├── pipelines/          training_pipeline + daily_prediction_pipeline
├── utils/              Logging, date helpers
└── validation/         Input + prediction validators
dags/                   Airflow DAGs
docker/                 Dockerfile
docker-compose.demo.yml Single-service compose attaching to traefik_proxy + mlflow_default
k8s/                    Optional Kubernetes manifests (Deployment, CronJob, Job, Service, Ingress)
kserve/                 Optional KServe InferenceService and custom predictor
kubeflow/               Optional Kubeflow Pipelines + components
monitoring/grafana/     Grafana dashboard JSON
sql/                    PostgreSQL schema for monitoring tables
scripts/                Smoke tests, DAG validation, export helper
tests/                  pytest suite
```

## Quickstart (Docker-only path)

This is the path that works without installing Python locally. Every target builds the image and runs it on the homelab `mlflow_default` Docker network so the container can reach `http://mlflow_server:5000` directly.

```bash
cd /home/jimi/projects/port-productivity-mlops-demo
make setup

# Full happy-path demo (test → generate → train → promote → predict)
make docker-demo-normal
```

Or step-by-step:

```bash
make docker-generate-data     # synthetic data for 2024-01-01 → TODAY (Makefile var)
make docker-train             # logs to MLflow experiment "port_productivity_training"
make docker-promote-model     # sets Production alias on the latest Candidate
make docker-predict           # writes predictions + Evidently HTML + Prometheus textfile
make docker-run-api           # runs FastAPI on host port 8015
```

Each MLflow-using target attaches `--network mlflow_default` so `mlflow_server` resolves. Override with `DOCKER_MLFLOW_TRACKING_URI` and `DOCKER_MLFLOW_NETWORK` if you point at a different MLflow.

## Local Python Path

Python 3.12 is the reference version (Docker image, GitHub Actions). 3.11 also works.

```bash
make venv          # creates .venv with Python 3.12
make install       # pip install -r requirements.txt
make doctor        # verifies imports

make generate-data
make train
make promote-model
make predict
make run-api       # http://127.0.0.1:8015
```

The Make targets automatically fall back to the Docker variants if local Python deps are missing.

## End-to-End Pipeline

### 1. Generate Synthetic Data

`src/data/generate_synthetic_data.py` produces `data/raw/port_productivity.csv` with terminals `T1–T4`, commodity types, queue/equipment availability, weather, and harvest flags.

```bash
make docker-generate-data
```

### 2. Train

`src/pipelines/training_pipeline.py`:

1. Loads + validates raw data (`src/validation/validate_input_data.py`).
2. Builds features (`src/features/build_features.py`).
3. Time-splits 80/20 by `operation_date`.
4. Trains `DummyRegressor` (baseline) and `RandomForestRegressor` (candidate).
5. Picks the lower-RMSE model.
6. Computes overall and per-terminal-per-horizon metrics.
7. Creates the Evidently training data report.
8. Opens an MLflow run, logs params, metrics, tags, the sklearn model, and the Evidently HTML as artifact `evidently/training_data_report.html`.
9. Writes `models/registry.json` and `models/latest_training_metrics.json` for the local fallback path.

```bash
make docker-train
# 🏃 View run training-… at: http://mlflow_server:5000/#/experiments/3/runs/<run_id>
```

### 3. Promote

`src/models/promote_model.py` moves the latest `Candidate` to `Champion` / `Production` in both the MLflow registry and the local JSON registry. Used both via Make and via the `model-promotion.yml` GitHub Actions workflow (which has a `production` environment gate).

```bash
make docker-promote-model
```

### 4. Predict

`src/pipelines/daily_prediction_pipeline.py`:

1. Synthesizes the daily operational batch for `--execution-date`.
2. Loads the model via MLflow alias `Production` (falls back to `local:Production`).
3. Builds inference features against historical context.
4. Validates predictions (null, negative, unreasonable values, missing terminals).
5. Writes `data/predictions/predictions_<date>.csv` and `latest_predictions.csv`.
6. Generates daily Evidently drift report (`drift_report_<date>.html`) plus fallback HTML for prediction drift and the monitoring summary.
7. Writes Prometheus textfile metrics and CSV monitoring records.
8. Optionally appends to PostgreSQL tables when `DATABASE_URL` is configured.

```bash
make docker-predict
```

### 5. Serve the API

```bash
make docker-run-api    # binds host :8015 → container :8000
# or
make run-api           # local uvicorn
```

API endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check (used by Docker + Traefik) |
| GET | `/metrics` | Prometheus exposition format |
| GET | `/predictions/latest` | Last batch from `data/predictions/latest_predictions.csv` |
| POST | `/predict` | On-demand inference (Pydantic request body) |

Smoke test:

```bash
python scripts/smoke_test_api.py --base-url http://127.0.0.1:8015
```

## URLs and Endpoints

### This project (when deployed via compose / homelab Traefik)

| URL | Purpose |
|---|---|
| `http://jimitogni.duckdns.org:8888/ports-mlops/health` | Public health check |
| `http://jimitogni.duckdns.org:8888/ports-mlops/metrics` | Public Prometheus metrics |
| `http://jimitogni.duckdns.org:8888/ports-mlops/predictions/latest` | Public latest batch |
| `http://jimitogni.duckdns.org:8888/ports-mlops/predict` (POST) | Public on-demand inference |
| `https://jimitogni.duckdns.org:8443/ports-mlops/...` | Same routes via HTTPS (router port 8443) |
| `http://127.0.0.1:8015/...` | Local `make run-api` / `make docker-run-api` |
| `http://127.0.0.1:80/ports-mlops/...` (with `Host: jimitogni.duckdns.org:8888`) | Server-local Traefik check |

The path prefix is `${PUBLIC_PATH_PREFIX:-/ports-mlops}` (Traefik strip-prefix middleware removes it before forwarding).

### Homelab platform services consumed by this project

| Service | Internal | Public |
|---|---|---|
| MLflow UI | `http://127.0.0.1:5000` (Host-header validated) | `http://mlflow-jimitogni.duckdns.org:8888` |
| MLflow API | `http://mlflow_server:5000` (from Docker network `mlflow_default`) | same as UI |
| Grafana | `http://127.0.0.1` via Traefik | `http://grafana-jimitogni.duckdns.org:8888` |
| Prometheus | `http://127.0.0.1` via Traefik | `http://prometheus-jimitogni.duckdns.org:8888` (Basic Auth) |
| Portainer | local docker socket | `http://portainer-jimitogni.duckdns.org:8888` (Basic Auth) |
| Uptime Kuma | container `uptime-kuma:3001` | `http://kuma-jimitogni.duckdns.org:8888` |

Browser microphone or any HTTPS-required feature should use `:8443`. Router forwards public `:8888 → host :80` and public `:8443 → host :8443`.

## Configuration

Default values live in `.env.example`; copy to `.env` before running.

Business thresholds (model promotion gates, alert limits) live in [`thresholds.yml`](thresholds.yml) and are loaded automatically by `promote_model.py` and the monitoring pipeline. Edit that file — not the code — when SLA targets change.

For the full Data Scientist ↔ MLOps collaboration workflow, promotion criteria, rollback runbook, and retraining trigger checklist, see [`CONTRIBUTING.md`](CONTRIBUTING.md).

| Variable | Default | Purpose |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:5000` | Used by local Python runs |
| `DOCKER_MLFLOW_TRACKING_URI` | `http://mlflow_server:5000` | Used by all `make docker-*` targets |
| `DOCKER_MLFLOW_NETWORK` | `mlflow_default` | Docker network attached to MLflow-using targets |
| `DATABASE_URL` | empty | Set to a Postgres URL to also write monitoring records to SQL |
| `MODEL_NAME` | `port_productivity_forecaster` | Registered model name in both registries |
| `EXPECTED_TERMINALS` | `T1,T2,T3,T4` | Validation guard for incoming batches |
| `FEATURE_VERSION` | `v1` | Stamped on every prediction row |
| `PIPELINE_VERSION` | `v1` | Stamped on every prediction row |
| `PROMETHEUS_METRICS_PORT` | `8015` | Local API port |
| `PROMETHEUS_METRICS_PATH` | `data/monitoring/port_productivity_metrics.prom` | Textfile target for node-exporter |
| `PUBLIC_PATH_PREFIX` | `/ports-mlops` | FastAPI `root_path` (also Traefik strip-prefix value) |
| `PUBLIC_APPS_HOST` | `jimitogni.duckdns.org` | Traefik Host rule |
| `TRAEFIK_NETWORK` | `traefik_proxy` | External docker network for the API |
| `TRAEFIK_HTTP_ENTRYPOINT` | `web` | Traefik HTTP entrypoint |
| `TRAEFIK_HTTPS_ENTRYPOINT` | `websecure` | Traefik HTTPS entrypoint |
| `TRAEFIK_CERT_RESOLVER` | `duckdns` | Traefik cert resolver (ACME DNS-01 via DuckDNS) |

## Airflow DAGs

DAG files live in `dags/`:

- `port_productivity_training_dag.py` — runs the training pipeline on a schedule.
- `port_productivity_daily_prediction_dag.py` — runs the daily prediction at D0 for D+1/D+2/D+3. Terminal tasks: `check_drift_and_alert` (logs warnings if data/prediction drift thresholds from `thresholds.yml` are breached) and `generate_operational_report` (writes an HTML + CSV forecast report for Camila, Ana, and Bruna to `reports/daily/`).

Validate parsing:

```bash
make validate-dags
```

Copy `dags/*.py` into the existing Airflow `dags/` folder (or mount it) and configure environment variables from `.env.example`. Airflow on the homelab is not yet running, so this remains a documented integration step.

## Kubernetes / KServe / Kubeflow (optional)

> **These are enterprise extensions, not the proposed solution for this use case.**
> The proposed stack is Docker Compose + Airflow + Traefik on a single host (or a small VM).
> K8s/KServe/Kubeflow are preserved in the repo as documented upgrade paths for when the
> workload outgrows a single machine, but they are not required to demonstrate the full
> MLOps lifecycle (training → validation → promotion gate → monitoring → rollback).

The Kubernetes path is preserved as an enterprise extension. Airflow is the primary orchestrator for the daily batch.

```bash
make k8s-apply        # Deployment + Service + CronJob + Job
make k8s-status
make k8s-logs
make k8s-delete
```

Edit the image reference (`ghcr.io/OWNER/port-productivity-mlops-demo:latest`) before applying. The `k8s/ingress.example.yaml` shows the same `/ports-mlops` path-prefix pattern using an Ingress controller instead of Traefik labels.

- `kserve/` — `InferenceService` that points at the FastAPI container as a custom predictor (keeps feature engineering and MLflow alias lookup in the serving path). See `kserve/README.md`.
- `kubeflow/` — Kubeflow Pipelines `pipeline.py` + reusable components. Install extras with `pip install -r requirements-kubeflow.txt`. See `kubeflow/README.md`.

## CI/CD with GitHub Actions

Workflows under `.github/workflows/`:

| Workflow | Trigger | Purpose |
|---|---|---|
| `ci.yml` | push / PR | pytest, compile, DAG validation, training smoke test |
| `build-and-push.yml` | push to `main` | Build image and push `latest` + git SHA tag to GHCR |
| `model-promotion.yml` | manual | Gated promotion to `Production` (uses `production` GH environment) |
| `deploy-homelab.yml` | after successful image build, or manual | SSH deploy: rsync repo, `make compose-up`, check `/ports-mlops/health` |

Required secrets for SSH deploy:

- `HOMELAB_HOST`, `HOMELAB_USER`, `HOMELAB_SSH_KEY`, `HOMELAB_SSH_PORT`, `DEPLOY_PATH`

Optional secrets:

- `MLFLOW_TRACKING_URI`, `DATABASE_URL`

`.env` is never committed. On first deploy, `.env.example` is copied if `.env` is missing; production values are edited on the server thereafter.

## Demo Scenarios

```bash
make demo-normal              # happy path
make demo-missing-terminal    # validation failure scenario
make demo-drift               # data drift injected
make demo-prediction-drift    # prediction drift scenario
make demo-train-challenger    # train on drifted data → produces a Challenger
make demo-promote-model       # promote latest Candidate
make demo-failed-validation   # alias of demo-missing-terminal
make demo-recovery            # validation failure followed by clean re-run
```

## Troubleshooting

| Symptom | Likely Cause / Fix |
|---|---|
| Training runs but nothing on MLflow UI | Container not on `mlflow_default` network, or `MLFLOW_TRACKING_URI` points at a file path. The Makefile defaults are `http://mlflow_server:5000` and `--network mlflow_default` — check `docker network ls` for `mlflow_default`. |
| `Could not load MLflow alias Production` | No model promoted yet. Run `make docker-promote-model`. Until then, predict falls back to `local:Production`. |
| `evidently.metric_preset` ImportError | This project pins Evidently `>=0.4.30,<0.8`. On 0.7.x the legacy API moved under `evidently.legacy.*`; the imports in `src/monitoring/evidently_report.py` already use it. |
| Evidently drift report falls back | The reference dataframe must share columns with the current batch — `create_daily_monitoring_report` already intersects columns. If you change the feature builder, keep that intersection. |
| 404 on `http://jimitogni.duckdns.org:8888/ports-mlops/...` | Container not on `traefik_proxy` network, or `Host` header missing in a local curl. Use `curl -H 'Host: jimitogni.duckdns.org:8888' http://127.0.0.1:80/ports-mlops/health`. |
| MLflow returns `Invalid Host header` on `127.0.0.1:5000` | The homelab MLflow has `--allowed-hosts`. Add `Host: mlflow-jimitogni.duckdns.org` or use the public URL. |
| `Expected terminals missing` | Intentional in `make demo-missing-terminal`; otherwise input batch is incomplete. |
| PostgreSQL connection errors | Unset `DATABASE_URL` to disable the SQL writer; the pipeline keeps writing CSV. |
| Permission denied removing `mlruns/` | Old artifacts may be root-owned (created from a container). Run `docker run --rm -v "$PWD":/work alpine rm -rf /work/mlruns`. |

## What Is Still Manual

- Wiring the project DAGs into a running Airflow instance.
- Adding a Prometheus scrape job for the API `/metrics` endpoint (or pointing node-exporter textfile at `data/monitoring/port_productivity_metrics.prom`).
- Importing the Grafana dashboard JSON.
- Choosing production thresholds before flipping the promotion gate.
- Creating Kubernetes secrets if using the cluster path.
