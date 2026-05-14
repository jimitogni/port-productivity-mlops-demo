 ---
  Port Productivity MLOps Demo — End-to-End Explained
  
  Business problem: A port operator wants to forecast daily discharge productivity (tons/hour) for terminals T1–T4,
  three days ahead (D+1, D+2, D+3). The original work was a notebook with no traceability, no monitoring, and no
  deployment. This project replaces it with a production-style MLOps pipeline.

  ---
  Step 1 — Synthetic Data Generation

  File: src/data/generate_synthetic_data.py

  Since there's no real port sensor feed, synthetic data is generated that mimics the real world: terminals T1–T4,
  commodity types, queue depth, equipment availability, weather conditions, and seasonal harvest flags.

  Output: data/raw/port_productivity.csv — a multi-year historical dataset (2024-01-01 to today).

  Tool: pandas + numpy — data generation, manipulation, and CSV serialisation.

  ---
  Step 2 — Input Validation

  File: src/validation/validate_input_data.py

  Before any model ever sees data, business rules are enforced: are all terminals present? Are productivity values
  in a sane range? Are dates contiguous?

  Tool: pydantic v2 — schema and type validation at the boundary between raw data and the ML system. Catches
  problems early rather than letting a bad row corrupt a model silently.

  ---
  Step 3 — Feature Engineering

  File: src/features/build_features.py

  Raw columns are transformed into model-ready features: lag features, rolling averages, day-of-week, terminal
  encodings, etc. The version is stamped as FEATURE_VERSION=v1 so every downstream prediction row is traceable back
  to exactly which feature logic produced it.
  
  Tools:
  - pandas — all transformations
  - numpy — numerical operations
  - joblib — saves the feature column list to models/feature_columns.json so inference uses exactly the same columns
   as training

  ---
  Step 4 — Training Pipeline
  
  File: src/pipelines/training_pipeline.py

  This is the main training orchestration. It runs in sequence:

  1. Load + validate data
  2. Build features
  3. Time-based 80/20 train/test split (sorted by operation_date, not shuffled) — this mirrors the real forecasting
  scenario and prevents data leakage
  4. Train two models:
    - sklearn DummyRegressor — the baseline (always predicts the mean); anything worse than this is useless
    - sklearn RandomForestRegressor — the candidate; strong on tabular data, interpretable, no GPU needed
  5. Pick the winner by lowest RMSE
  6. Evaluate with MAE, RMSE, R², and MAPE — both overall and per-terminal per-horizon so operations stakeholders
  can see which terminal is performing worst

  Tools:
  - scikit-learn — DummyRegressor, RandomForestRegressor, train/test splitting, all metric functions
  - pandas / numpy — data wrangling during training

  ---
  Step 5 — Experiment Tracking with MLflow
  
  File: src/pipelines/training_pipeline.py (MLflow calls inside)

  Every training run opens an MLflow run that logs:
  - Parameters (model type, feature version, train/test sizes)
  - Metrics (MAE, RMSE, R², MAPE — per terminal and overall)
  - Tags (pipeline_version, feature_version)
  - The trained sklearn model as an artifact
  - The Evidently training data quality report as an HTML artifact under evidently/
  
  Tool: MLflow (backed by PostgreSQL on the homelab at mlflow_server:5000)
  - Acts as the single source of truth for all experiments
  - UI at http://mlflow-jimitogni.duckdns.org:8888
  - Postgres stores metadata; named volume mlflow_artifacts stores model files and HTML reports

  ---
  Step 6 — Model Registration & Promotion

  Files: src/models/register_model.py, src/models/promote_model.py

  After training, the model is registered under the name port_productivity_forecaster. It gets a Candidate alias.
  Promotion to Production is a separate explicit gate — it never happens automatically.

  The registry has two layers:
  - MLflow Registered Models (primary) — uses aliases Candidate / Challenger / Champion / Production
  - Local JSON fallback (models/registry.json) — so the prediction pipeline still works even if MLflow is
  unreachable
  
  Tool: MLflow Model Registry — versioned model store with aliases; joblib for local serialisation fallback.

  CI/CD gate: The model-promotion.yml GitHub Actions workflow requires a manual approval via a production GitHub
  environment before the Production alias is moved.

  ---
  Step 7 — Orchestration with Airflow
  
  Files: dags/port_productivity_training_dag.py, dags/port_productivity_daily_prediction_dag.py

  Two DAGs automate the entire pipeline on a schedule:

  Training DAG (weekly):
  generate_synthetic_historical_data
    → validate_training_data
    → build_training_features
    → train_candidate_models
    → evaluate_models
    → log_experiments_to_mlflow
    → register_best_model + generate_evidently_training_report
    
  Daily Prediction DAG (daily at D0, produces D+1/D+2/D+3 forecasts):
  generate_daily_operational_data
    → validate_daily_data
    → generate_predictions
    → validate_predictions
    → save_predictions
    → generate_monitoring_report
    → update_prometheus_metrics
    
  Tool: Apache Airflow — task dependency management, retry logic, scheduling, and execution logging. Running locally
   via docker-compose.airflow.yml.

  ---
  Step 8 — Daily Prediction Pipeline
  
  File: src/pipelines/daily_prediction_pipeline.py

  Every day (D0) the pipeline:

  1. Synthesises the current day's operational batch
  2. Loads the Production model from MLflow (falls back to local JSON if MLflow is down)
  3. Builds inference features identically to training
  4. Validates predictions — catches null values, negatives, implausible ranges, and missing terminals before
  writing anything
  5. Writes data/predictions/predictions_<date>.csv and latest_predictions.csv
  6. Generates Evidently drift reports
  7. Writes Prometheus metrics
  8. Optionally appends to PostgreSQL if DATABASE_URL is set

  Every output row carries run_id, model_version, feature_version, and pipeline_version — full lineage traceability.

  File: src/validation/validate_predictions.py — post-prediction guard.

  ---
  Step 9 — Monitoring & Observability
  
  Three layers run on every prediction cycle:

  Data / Model Quality — Evidently

  File: src/monitoring/evidently_report.py
  
  - Training report: DataQualityPreset — checks for nulls, distributions, outliers in training data. Saved as HTML
  and logged to MLflow.
  - Daily drift report: DataDriftPreset — compares today's operational batch against the training reference
  distribution. Flags when input data starts drifting.
  - Prediction drift report: checks if the prediction distribution is shifting over time.

  Tool: Evidently (v0.7.x legacy API under evidently.legacy.*) — open-source ML observability library that produces
  human-readable HTML reports and structured metrics.

  Operational Metrics — Prometheus

  File: src/monitoring/prometheus_metrics.py

  Emits metrics in two ways:
  - Push: writes a .prom textfile to data/monitoring/port_productivity_metrics.prom — readable by a Prometheus
  node_exporter textfile collector
  - Pull: exposed live at the FastAPI /metrics endpoint — scraped by the homelab Prometheus
  
  Metrics include prediction counts, MAE per terminal, pipeline run duration, model version tags.

  Tool: prometheus-client — Python library for the Prometheus exposition format.

  Structured Records — CSV / PostgreSQL

  File: src/monitoring/performance_monitoring.py
  
  Every run writes three CSV files:
  - model_performance_<run_id>.csv — metrics per run
  - monitoring_metrics_<date>.csv — drift and data quality scores
  - execution_metadata_<date>.csv — pipeline runtime, versions
  
  Optionally writes to PostgreSQL (schema in sql/create_tables.sql) when DATABASE_URL is configured.

  Tools: pandas (CSV), SQLAlchemy + psycopg2-binary (PostgreSQL).

  Dashboards — Grafana
  
  File: monitoring/grafana/port_productivity_dashboard.json

  Pre-built Grafana dashboard that visualises all the Prometheus metrics. Import the JSON into the homelab Grafana
  at http://grafana-jimitogni.duckdns.org:8888.

  Tool: Grafana — running on the homelab, shared with other projects (customer churn, English voice tutor).

  ---
  Step 10 — Serving the API

  File: src/api/main.py

  A FastAPI service exposes:

  ┌─────────────────────────┬───────────────────────────────────────────────────────────────────┐
  │        Endpoint         │                              Purpose                              │
  ├─────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ GET /health             │ Liveness check — used by Docker's HEALTHCHECK and Traefik routing │
  ├─────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ GET /metrics            │ Prometheus metrics in exposition format                           │
  ├─────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ GET /predictions/latest │ Returns the last batch from latest_predictions.csv                │
  ├─────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ POST /predict           │ On-demand inference with a Pydantic-validated request body        │
  └─────────────────────────┴───────────────────────────────────────────────────────────────────┘

  Tools:
  - FastAPI — async Python web framework, automatic OpenAPI docs
  - uvicorn — ASGI server
  - Pydantic v2 — request/response schema validation

  ---
  Step 11 — Containerisation & Reverse Proxy

  Files: docker/Dockerfile, docker-compose.demo.yml

  The API is packaged into a Docker image and published to GHCR (GitHub Container Registry) as
  ghcr.io/jimitogni/port-productivity-mlops-demo:latest.

  The compose file attaches the container to two Docker networks:
  - traefik_proxy — so Traefik can route public traffic to it
  - mlflow_default — so the container can reach mlflow_server:5000 directly by container name
  
  Traefik labels on the container define the routing rules:
  - Host: jimitogni.duckdns.org
  - Path prefix: /ports-mlops (stripped before forwarding to FastAPI)
  - HTTP and HTTPS routers, with Let's Encrypt cert via DuckDNS DNS challenge

  Tools:
  - Docker — containerisation
  - Docker Compose — multi-container orchestration on the homelab
  - Traefik v3 — reverse proxy; handles SSL termination, routing, and health-based load balancing
  
  Public URL: http://jimitogni.duckdns.org:8888/ports-mlops/health

  ---
  Step 12 — CI/CD with GitHub Actions

  Files: .github/workflows/

  Workflow: ci.yml
  Trigger: Every push / PR
  What it does: Runs pytest, compiles all Python, validates Airflow DAGs, runs a training smoke test
  ────────────────────────────────────────
  Workflow: build-and-push.yml
  Trigger: Push to main
  What it does: Builds the Docker image and pushes latest + git SHA tag to GHCR
  ────────────────────────────────────────
  Workflow: model-promotion.yml
  Trigger: Manual
  What it does: Gated promotion: requires approval from the production GitHub environment before moving the
    Production alias
  ────────────────────────────────────────
  Workflow: deploy-homelab.yml
  Trigger: After successful build
  What it does: SSHes into the homelab, extracts the repo tarball, runs docker compose up, and health-checks the
    running service

  Tools:
  - GitHub Actions — CI/CD runner
  - pytest — unit and integration tests
  - GHCR (GitHub Container Registry) — image storage

  ---
  Optional Enterprise Extensions
  
  The project also includes stubs for a Kubernetes path:
  - k8s/ — standard Kubernetes Deployment, CronJob, Service, and Ingress manifests
  - kserve/ — InferenceService wrapping the FastAPI container as a custom KServe predictor
  - kubeflow/ — Kubeflow Pipelines DAG equivalent of the Airflow training pipeline
  
  These exist to show the same business logic can run in a cluster without rewriting the core code.

  ---
  Full Tool/Library Summary

  Layer: Data
  Tool: pandas, numpy   
  Role: Generation, transformation, IO
  ────────────────────────────────────────            
  Layer: Validation
  Tool: pydantic v2     
  Role: Schema enforcement at data boundaries
  ────────────────────────────────────────            
  Layer: ML
  Tool: scikit-learn    
  Role: Models, splitting, metrics
  ────────────────────────────────────────            
  Layer: Serialisation
  Tool: joblib          
  Role: Model and feature column persistence
  ────────────────────────────────────────
  Layer: Experiment tracking
  Tool: MLflow
  Role: Runs, params, metrics, artifacts, registry
  ────────────────────────────────────────            
  Layer: DB backend
  Tool: PostgreSQL      
  Role: MLflow metadata store + optional monitoring tables
  ────────────────────────────────────────            
  Layer: DB client
  Tool: SQLAlchemy, psycopg2-binary
  Role: ORM and Postgres driver
  ────────────────────────────────────────
  Layer: Orchestration
  Tool: Apache Airflow  
  Role: DAG scheduling, retries, task dependencies
  ────────────────────────────────────────            
  Layer: Data observability
  Tool: Evidently       
  Role: Drift detection, data quality HTML reports
  ────────────────────────────────────────            
  Layer: Metrics
  Tool: prometheus-client 
  Role: Prometheus exposition (pull + textfile push)
  ────────────────────────────────────────
  Layer: Dashboards
  Tool: Grafana
  Role: Visualisation of Prometheus metrics
  ────────────────────────────────────────
  Layer: API
  Tool: FastAPI, uvicorn
  Role: Serving predictions and metrics
  ────────────────────────────────────────
  Layer: Containerisation
  Tool: Docker, Docker Compose
  Role: Packaging and local orchestration
  ────────────────────────────────────────
  Layer: Reverse proxy
  Tool: Traefik
  Role: Routing, SSL termination, health checks
  ────────────────────────────────────────
  Layer: DNS
  Tool: DuckDNS
  Role: Dynamic DNS for homelab public URLs
  ────────────────────────────────────────
  Layer: CI/CD
  Tool: GitHub Actions, GHCR
  Role: Testing, image build, gated deployment
  ────────────────────────────────────────
  Layer: Config
  Tool: python-dotenv
  Role: .env loading
  ────────────────────────────────────────
  Layer: Testing
  Tool: pytest
  Role: Unit and integration tests
  ────────────────────────────────────────
  Layer: Optional cluster
  Tool: Kubernetes, KServe, Kubeflow Pipelines
  Role: Enterprise execution path

