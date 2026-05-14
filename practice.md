# Interview Practice — Port Productivity MLOps Pipeline

End-to-end walkthrough of the project from raw data to production, monitoring, and rollback.
Use this to narrate the full lifecycle in an interview.

---

## The Business Problem

A Brazilian port operator (think Rumo Logística / Santos port complex) needs daily
forecasts of discharge productivity (tons/hour) for 4 terminals (T1–T4) across 3
forecast horizons: **D+1** (tomorrow), **D+2**, **D+3**.

Operations teams (Camila, Ana) use D+1 to set crew schedules and terminal priorities.
Commercial team (Bruna) uses D+1/D+2 to assess SLA risk with customers.

The starting point was a notebook that produced reasonable numbers but had no
traceability, no validation, no monitoring, and no way to reproduce or deploy.
This project replaces that notebook with a production-grade MLOps pipeline.

**Key metrics the business cares about:**
- MAE D+1 ≤ 15 t/h (tight — drives train retention decisions)
- MAE D+2 ≤ 18 t/h
- MAE D+3 ≤ 22 t/h
- Overall MAPE ≤ 15%, R² ≥ 0.65

These live in `thresholds.yml` — not hardcoded — so Camila/Ana can change them
without touching code.

---

## Step 1 — Synthetic Data Generation

**File:** `src/data/generate_synthetic_data.py`

**Why synthetic?** The real dataset is proprietary. For the demo, we generate realistic
port operations data with seasonality, weekend effects, weather noise, and terminal
variability. This proves the pipeline works end-to-end without needing real data.

**What it generates:**
- `port_productivity.csv` — historical data (2 years) for training
- Daily operational snapshots for inference (features only, no target)

**Tools used:**
- `pandas` — data manipulation, CSV I/O
- `numpy` — random number generation, seasonal patterns

**Interview point:** "I used synthetic data to decouple pipeline development from data
access. The schema, validation rules, and feature logic are identical to what the real
data would require."

---

## Step 2 — Data Validation

**File:** `src/validation/validate_input_data.py`, `src/validation/validate_predictions.py`

Before any feature engineering or model inference, data is validated at the boundary.

**What is checked:**
- Required columns are present (terminal_id, date, berth_occupancy_rate, etc.)
- No nulls in critical columns
- terminal_id values are within the expected set (T1–T4)
- Numeric ranges are plausible (e.g., productivity > 0)
- For inference: no target column leakage

**Tools used:**
- `pandas` — schema and value checks
- Custom `ValidationError` exception — lets the pipeline fail fast with a clear message

**Why this matters (interview point):** "Validating at the boundary is a core MLOps
practice. A model that silently ingests bad data produces bad predictions with no
warning. We surface the error at the earliest possible point."

---

## Step 3 — Feature Engineering

**File:** `src/features/build_features.py`

Transforms raw operational data into model-ready features.

**Features built:**
- Rolling averages (3-day, 7-day productivity per terminal)
- Day-of-week and month-of-year encodings
- Berth occupancy lag features
- Terminal-level historical averages
- Forecast horizon encoding (D+1, D+2, D+3 become numeric)

**Feature versioning (`FEATURE_VERSION`):**
Every prediction row is stamped with `FEATURE_VERSION` (e.g., `v1`). If the feature
logic changes (new column, different lag window), the version is bumped in `.env` so
historical predictions remain comparable.

**Tools used:**
- `pandas`, `numpy` — rolling windows, lag computation, one-hot encoding

**Interview point:** "Feature versioning is critical when you're comparing predictions
made weeks apart. Without it, you can't tell if a shift in forecast quality is due to
real-world change or a feature change you forgot about."

---

## Step 4 — Model Training

**File:** `src/pipelines/training_pipeline.py`, `src/models/train_model.py`

**What happens:**
1. Load historical data, split train/test by time (not random — to avoid leakage)
2. Train a **baseline model** (`DummyRegressor` — predicts the mean)
3. Train a **candidate model** (`RandomForestRegressor`)
4. Evaluate both on the held-out test set: MAE, RMSE, R², MAPE per terminal per horizon
5. Register the better model in both the **local registry** and **MLflow**

**Time-based split (critical interview point):** "We never use random split for time
series. Training on future data to predict the past produces artificially good metrics
that collapse in production. The split is always by date."

**Tools used:**
- `scikit-learn` — `RandomForestRegressor`, `DummyRegressor`, metrics
- `joblib` — serialize the trained model to disk
- `MLflow` — log parameters, metrics, artifacts (model file, Evidently HTML report)
- `xgboost` — available as an alternative estimator

**Per-terminal per-horizon metrics:** The model computes not just overall MAE but
terminal-specific metrics for each forecast horizon. These become `terminal_metrics`
stored in the registry and checked during promotion.

---

## Step 5 — Experiment Tracking with MLflow

**Files:** `src/models/train_model.py`, `src/models/register_model.py`
**Server:** homelab MLflow at `http://mlflow_server:5000`

Every training run logs to MLflow:
- **Parameters:** algorithm, hyperparameters, feature version, pipeline version
- **Metrics:** MAE, RMSE, R², MAPE (overall and per terminal/horizon)
- **Artifacts:** the serialized model (`model.joblib`), Evidently training data quality report
- **Tags:** validation_status, feature_columns, terminal_metrics (JSON)

**MLflow concepts used:**
- `mlflow.start_run()` — opens a tracked experiment run
- `mlflow.log_params()` / `mlflow.log_metrics()` — structured logging
- `mlflow.log_artifact()` — attach files (model, reports)
- **Model Registry** — register models with versioned aliases: `Candidate`, `Champion`, `Production`

**Why MLflow?** "It gives us full reproducibility: given any run ID, we can retrieve
the exact model, its training data snapshot, all metrics, and the feature version. This
is essential for debugging production regressions."

---

## Step 6 — Model Registry (Local + MLflow)

**File:** `src/models/register_model.py`
**Registry file:** `models/registry.json`

We maintain **two registries** in parallel:
1. **Local JSON registry** (`models/registry.json`) — works without MLflow, fast, inspectable
2. **MLflow Model Registry** — cloud-capable, UI-friendly, supports alias-based serving

**Aliases:**
- `Candidate` — freshly trained model, not yet promoted
- `Champion` — current Production model (previous winner)
- `Production` — alias that the serving layer reads

**Why dual registry?** "The local registry means the serving pipeline works even if
MLflow is down. It's also human-readable JSON — you can audit every registered model's
metadata with `cat models/registry.json`."

---

## Step 7 — Model Promotion with Gates

**File:** `src/models/promote_model.py`
**Config:** `thresholds.yml`

Promotion is not automatic — it requires passing a **gate** defined in `thresholds.yml`.

**Checks run automatically:**
1. `validation_status == "passed"` — model was validated after training
2. Feature metadata present — model knows what columns it was trained on
3. D+1/D+2/D+3 terminal metrics present
4. **Absolute error limits:** MAE D+1 ≤ 15, D+2 ≤ 18, D+3 ≤ 22 t/h
5. **Overall MAPE ≤ 15%, R² ≥ 0.65**
6. **Relative degradation vs Champion:** MAE and RMSE must not regress by more than 5%
7. **Per-terminal degradation:** no single terminal can degrade by more than 10%

If any check fails, the promotion is blocked with a clear error message listing every
failure reason.

**Promotion history** is recorded in `registry.json` (last 20 promotions). This enables:

**Rollback:**
```bash
python -m src.models.promote_model --rollback              # revert to previous version
python -m src.models.promote_model --rollback --target-version 3  # specific version
```

**Interview point:** "Automated gates mean a data scientist can't accidentally push a
model that regresses performance. The thresholds are in a YAML file — operations
agreed on them, they don't require a code change to adjust."

---

## Step 8 — Orchestration with Apache Airflow

**Files:** `dags/port_productivity_training_dag.py`, `dags/port_productivity_daily_prediction_dag.py`
**Setup:** `docker-compose.airflow.yml`

Two DAGs manage the full lifecycle:

### Training DAG (weekly or on-demand)
1. Generate/load historical data
2. Validate input data
3. Build features
4. Train and evaluate model
5. Register as Candidate in local + MLflow registry

### Daily Prediction DAG (runs every day at D0)
1. `generate_or_load_daily_operational_data` — get today's operational snapshot
2. `validate_input_data` — check schema, ranges, terminal IDs
3. `build_inference_features` — apply the same feature logic as training
4. `load_production_model_from_mlflow` — resolve `Production` alias → model version
5. `generate_predictions_for_d_plus_1_d_plus_2_d_plus_3` — run inference
6. `validate_predictions` — check output schema and value ranges
7. `save_predictions` — persist to `data/predictions/`
8. `save_execution_metadata` — audit trail of every run
9. `generate_evidently_monitoring_report` — drift detection report
10. `export_prometheus_metrics` — write `.prom` file for Prometheus scraping
11. `check_drift_and_alert` — **NEW** logs WARN if drift thresholds from `thresholds.yml` are breached
12. `generate_operational_report` — **NEW** writes HTML + CSV report for Camila/Ana/Bruna

**Tools used:**
- `Apache Airflow` — DAG definition, scheduling, task dependency graph, retries
- `@task` decorator — TaskFlow API (cleaner than classic Operators for Python-heavy pipelines)
- Docker Compose — run Airflow locally with postgres + redis + webserver + scheduler + worker

**Interview point:** "Airflow gives us a visual audit of every pipeline run, which tasks
succeeded or failed, how long they took, and the ability to retry individual tasks
without rerunning the whole pipeline. The TaskFlow API makes dependencies explicit
through function return values."

---

## Step 9 — Model Serving with FastAPI

**Files:** `src/api/`
**Port:** `:8015` locally, `/ports-mlops` path prefix in production

The REST API wraps the model for synchronous inference (ad-hoc requests, integrations).

**Endpoints:**
- `GET /ports-mlops/health` — liveness check (used by Traefik + deploy workflow)
- `GET /ports-mlops/metrics` — Prometheus metrics endpoint
- `POST /ports-mlops/predict` — submit operational data, receive D+1/D+2/D+3 forecasts
- `GET /ports-mlops/model-info` — current Production model version and metadata

**Tools used:**
- `FastAPI` — async REST framework, automatic OpenAPI docs
- `pydantic` — request/response schema validation
- `uvicorn` — ASGI server

**Interview point:** "FastAPI gives us automatic validation of request payloads via
Pydantic models and auto-generated Swagger docs at `/docs`. The `root_path` config
means the API works correctly behind Traefik's strip-prefix middleware."

---

## Step 10 — Drift Detection and Monitoring Reports

**File:** `src/monitoring/evidently_report.py`, `src/monitoring/daily_report.py`

After every prediction run, two monitoring outputs are generated:

### Evidently Reports (technical, for MLOps)
- **Training report:** data quality analysis of the training dataset (missing values,
  distribution summaries, outliers)
- **Daily drift report:** compares today's input features against the training
  reference distribution. Flags features with significant shift (PSI > 0.2).
  Also checks prediction distribution shift.
- Output: HTML report in `reports/evidently/`

**Tools used:**
- `Evidently` — open-source ML monitoring library. `DataDriftPreset` computes feature
  drift statistics. The HTML report is attached to the MLflow run as an artifact.

### Operational Report (business, for Camila/Ana/Bruna)
- Traffic-light table: terminal × forecast horizon
  - 🟢 ≥ 600 t/h high productivity
  - 🟡 400–599 t/h attention
  - 🔴 < 400 t/h low productivity
- Drift alerts in Portuguese
- Per-persona usage guide (how Camila reads it vs how Ana reads it vs how Bruna reads it)
- Also exports a CSV (`forecast_summary_<date>.csv`) for Bruna's commercial tooling
- Output: `reports/daily/operational_report_<date>.html`

**Interview point:** "Evidently separates what MLOps engineers need (statistical drift
report) from what operations needs (plain-language traffic-light table). The business
report answers 'should I worry about tomorrow's schedule?' without requiring the reader
to understand PSI or Jensen-Shannon divergence."

---

## Step 11 — Metrics Export with Prometheus

**File:** `src/monitoring/prometheus_metrics.py`
**Output:** `data/monitoring/port_productivity_metrics.prom`

After each prediction run, key metrics are written to a `.prom` textfile:

```
port_productivity_pipeline_success 1
port_productivity_model_mae 12.4
port_productivity_data_drift_detected 0
port_productivity_prediction_drift_detected 0
port_productivity_current_model_version 5
```

**Tools used:**
- `prometheus-client` — Prometheus Python client, textfile format
- `node-exporter` (on the homelab) — scrapes the `.prom` file and exposes it to Prometheus

**Interview point:** "We use the textfile collector pattern instead of a running
Prometheus exporter because the pipeline is a batch job, not a long-running service.
The pipeline writes the file; node-exporter exposes it. This decouples metric
production from metric scraping."

---

## Step 12 — Alerting Rules (Prometheus Alertmanager)

**File:** `monitoring/prometheus/alerts.yml`

Six alert rules defined:

| Alert | Condition | Severity |
|---|---|---|
| `PipelineFailure` | `pipeline_success == 0` for 5m | critical |
| `PipelineMissedRun` | No metric for 26+ hours | warning |
| `DataDriftDetected` | `data_drift_detected == 1` | warning |
| `PredictionDriftDetected` | `prediction_drift_detected == 1` | warning |
| `ModelMAEDegradation` | Live MAE > 20% above MAE 7 days ago | warning |
| `StaleModel` | No version change in 30 days | info |

**Interview point:** "Alert thresholds (30% drift share, 20% MAE degradation) are in
`thresholds.yml` and inform the DAG's `check_drift_and_alert` task at runtime. The
Prometheus alert rules are the paging layer — they fire even if Airflow is down.
Two layers of alerting with different blast radii."

---

## Step 13 — Visualization with Grafana

**Files:** `monitoring/grafana/dashboards/`

Dashboards display:
- Pipeline success/failure over time
- Live model MAE trend
- Data drift share per day
- Prediction distribution (mean, p25, p75) over time
- Current model version

**Tools used:**
- `Grafana` — dashboard visualization connected to Prometheus datasource
- Dashboards defined as JSON (importable, version-controlled)

**Interview point:** "The Grafana dashboard is the single pane of glass for the on-call
engineer. If `ModelMAEDegradation` fires at 2am, the dashboard immediately shows
whether it's a data quality problem, a drift problem, or a genuine model degradation."

---

## Step 14 — Containerization with Docker

**Files:** `Dockerfile`, `docker-compose.yml`, `docker-compose.airflow.yml`

Every component runs in Docker:
- `port-productivity-mlops-demo` — main app image (FastAPI + pipeline scripts)
- `mlflow_server` — experiment tracking server
- `airflow_scheduler`, `airflow_webserver`, `airflow_worker`, `airflow_postgres`, `airflow_redis`
- `prometheus`, `grafana` — monitoring stack

**Tools used:**
- `Docker` — image build, container isolation
- `Docker Compose` — multi-container orchestration for local + homelab deployment
- `GHCR` (GitHub Container Registry) — stores built images, tagged by git SHA and `latest`

**Interview point:** "Docker Compose is the right tool for a single-host deployment.
We consciously chose it over Kubernetes for this use case — fewer moving parts, simpler
ops, same container image. K8s/KServe/Kubeflow manifests exist in the repo as
documented enterprise upgrade paths, not the proposed solution."

---

## Step 15 — CI/CD with GitHub Actions

**Files:** `.github/workflows/`

Four workflows:

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | Every push / PR | pytest, Python compile check, DAG parse validation, training smoke test |
| `build-and-push.yml` | Push to `main` | Build Docker image, push to GHCR with `latest` + git SHA tags |
| `model-promotion.yml` | Manual (button) | Run `promote_model.py` — gates apply, requires `production` GH environment approval |
| `deploy-homelab.yml` | After build, or manual | SSH to homelab, rsync repo, `make compose-up`, verify `/ports-mlops/health` |

**Tools used:**
- `GitHub Actions` — CI/CD runner
- `GitHub Environments` — the `production` environment requires a manual approval before
  the promotion workflow proceeds (human-in-the-loop gate)
- `SSH + rsync` — deploy artifacts to the homelab server

**Interview point:** "The promotion workflow requires a human approval via GitHub
Environments. This is the governance layer — even if all automated checks pass, a
person reviews before the model goes live. Automated gates + human approval = defense
in depth."

---

## Step 16 — Reverse Proxy and Public Routing with Traefik

**Files:** `docker-compose.yml` (Traefik labels)
**Homelab setup:** External port `8888` → host port `80` → Traefik → containers

Traefik routes public traffic to the FastAPI container:
- Rule: `Host(jimitogni.duckdns.org) && PathPrefix(/ports-mlops)`
- Strip-prefix middleware removes `/ports-mlops` before forwarding
- TLS: ACME DNS-01 challenge via DuckDNS (automatic HTTPS cert)

**Tools used:**
- `Traefik` — cloud-native reverse proxy, reads routing config from Docker labels
- `DuckDNS` — dynamic DNS for homelab public URL
- `Let's Encrypt` (via ACME) — free TLS certificates

**Interview point:** "Traefik's Docker provider means routing config lives in the
`docker-compose.yml` labels alongside the service definition. No separate nginx.conf
to keep in sync. When the container starts, Traefik auto-discovers it."

---

## Step 17 — Collaboration Workflow (DS ↔ MLOps)

**File:** `CONTRIBUTING.md`

Defines how Carlos (Data Scientist) and the MLOps engineer work together:

1. Carlos experiments locally, logs to MLflow
2. Carlos opens a PR: must include updated `build_features.py` (bumped `FEATURE_VERSION` if features changed) + updated `requirements.txt`
3. CI validates the PR: tests, compile, DAG check, smoke test
4. MLOps engineer reviews: no leakage, no hardcoded paths, no fixed-version deps
5. PR merges → training pipeline runs → model lands as `Candidate`
6. Promotion is manual + gated (GitHub environment approval)
7. Post-promotion: MLOps watches Grafana. If `DataDriftDetected` fires for 2 consecutive days → retraining cycle starts

**Interview point:** "The CONTRIBUTING.md makes the implicit explicit. A new MLOps
engineer joining the team knows exactly what to check on a Carlos PR without asking.
The retraining checklist tells you when to retrain vs when to wait."

---

## The Full Data Flow (one sentence per step)

```
Synthetic data (CSV)
  → validate schema/ranges
  → build lag/rolling features
  → time-based train/test split
  → train RandomForest + DummyRegressor baseline
  → log run to MLflow (params, metrics, model artifact, Evidently report)
  → register as Candidate in local registry + MLflow registry
  → promotion gate checks all thresholds.yml rules
  → if passed: alias promoted to Champion/Production, history recorded
  → daily prediction DAG loads Production model
  → runs inference for D+1/D+2/D+3 across T1–T4
  → validates predictions
  → writes Evidently drift report (HTML)
  → writes Prometheus .prom file
  → checks drift thresholds → logs WARN if breached
  → writes Portuguese HTML+CSV report for Camila/Ana/Bruna
  → FastAPI serves predictions on-demand
  → Traefik routes HTTPS traffic to FastAPI
  → Prometheus scrapes .prom file via node-exporter
  → Grafana dashboards visualize trends
  → Alertmanager fires if MAE degrades or drift persists
  → if alert fires → rollback or retrain cycle begins
```

---

## Tools Summary Table

| Tool / Library | Layer | What it does in this project |
|---|---|---|
| `pandas` | Data | Data loading, transformation, CSV I/O |
| `numpy` | Data | Synthetic data generation, numerical ops |
| `scikit-learn` | ML | `RandomForestRegressor`, `DummyRegressor`, MAE/RMSE/R² metrics |
| `xgboost` | ML | Alternative gradient-boosted tree estimator |
| `joblib` | ML | Serialize/deserialize model to disk |
| `MLflow` | Tracking | Experiment logging, model registry, alias-based serving |
| `Evidently` | Monitoring | Data drift detection (PSI), prediction drift, HTML reports |
| `FastAPI` | Serving | REST API for on-demand predictions |
| `pydantic` | Serving | Request/response schema validation |
| `uvicorn` | Serving | ASGI server for FastAPI |
| `prometheus-client` | Observability | Write Prometheus textfile metrics |
| `Prometheus` | Observability | Metrics collection and alerting rules |
| `Grafana` | Observability | Dashboard visualization of metrics |
| `Apache Airflow` | Orchestration | DAG scheduling, task dependency, retries |
| `Docker` | Infrastructure | Container build and isolation |
| `Docker Compose` | Infrastructure | Multi-container local + homelab orchestration |
| `GitHub Actions` | CI/CD | Tests, build, promotion, deploy workflows |
| `GHCR` | CI/CD | Container image registry |
| `Traefik` | Routing | Reverse proxy, TLS termination, path routing |
| `DuckDNS` | Networking | Dynamic DNS for homelab public URL |
| `thresholds.yml` | Governance | Single source of truth for business SLA thresholds |
| `CONTRIBUTING.md` | Process | DS↔MLOps collaboration protocol |

---

## Common Interview Questions and Short Answers

**Q: Why not use a random train/test split?**
A: Time-series data has temporal dependencies. A random split lets the model learn from
future data to predict the past, producing metrics that collapse in production. We
always split by date.

**Q: How do you prevent a bad model from going to production?**
A: Three layers — automated `promotion_checks()` enforcing `thresholds.yml` rules,
MLflow model registry requiring explicit alias promotion, and a GitHub Environment
approval gate that requires human sign-off before the workflow proceeds.

**Q: How do you detect when the model degrades in production?**
A: Evidently drift reports run daily and feed `data_drift_detected` / `prediction_drift_detected`
metrics to Prometheus. The `ModelMAEDegradation` Prometheus alert fires if live MAE
grows >20% above the value from 7 days ago. The Airflow `check_drift_and_alert` task
logs warnings immediately after each run.

**Q: How do you roll back a bad model?**
A: `python -m src.models.promote_model --rollback` reads the last 20 promotion history
entries in `registry.json`, reverts the `Production` and `Champion` aliases to the
previous version, and records the rollback in the history. Takes under a second.

**Q: Why Docker Compose instead of Kubernetes?**
A: The workload runs on a single homelab server. Docker Compose is the right tool for
that scope — fewer moving parts, easier debugging, same container image. K8s/KServe
manifests exist in the repo as documented enterprise upgrade paths for when the
workload outgrows a single machine.

**Q: What is feature versioning and why does it matter?**
A: Every prediction row is stamped with `FEATURE_VERSION`. If we add a feature column
or change lag windows, we bump the version. Without it, you can't compare predictions
made before and after a feature change — you'd be comparing apples to oranges in
your drift reports.

**Q: How does the business consume forecasts?**
A: Two ways. The daily Airflow pipeline writes an HTML report (traffic-light table,
Portuguese, per-persona guidance) and a CSV to `reports/daily/` — zero technical
knowledge required. For integrations, the FastAPI endpoint at `/ports-mlops/predict`
accepts a JSON payload and returns structured D+1/D+2/D+3 forecasts.
