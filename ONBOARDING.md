# ONBOARDING — Interview Runbook & Session Handoff

> Handoff doc written 2026-05-14. Travels with the repo via `git pull`.
> **This machine is not the homelab.** The pipeline runs on the homelab; from a
> laptop you *trigger* (git push / GitHub UI) and *observe* (public URLs).

---

## 1. What this project is

`port-productivity-mlops-demo` — an MLOps demo that forecasts rail/port
terminal productivity (tons/hour). It's a **regression** problem, so metrics are
MAE / RMSE / R² / MAPE (not accuracy/precision/recall). Full pipeline:
generate synthetic data → train 3 models (Dummy / RandomForest / XGBoost) →
promote the winner → daily batch predict → serve via FastAPI → monitor with
Evidently + Prometheus. CI/CD via GitHub Actions; deployed to a homelab behind
Traefik on duckdns.

---

## 2. Laptop setup (one time)

```bash
git clone git@github.com:jimitogni/port-productivity-mlops-demo.git
cd port-productivity-mlops-demo
```

That's it — the laptop only needs the repo + a browser + `git`. Do **not** run
`make docker-demo-normal` on the laptop: it needs the homelab's `mlflow_default`
Docker network and `mlflow_server`, which only exist on the homelab.

---

## 3. Interview runbook (laptop = observer/trigger)

### A. Show the deployed app (public URLs)

| What | URL |
|---|---|
| App health | `http://jimitogni.duckdns.org:8888/ports-mlops/health` |
| Prometheus metrics | `http://jimitogni.duckdns.org:8888/ports-mlops/metrics` |
| Latest predictions | `http://jimitogni.duckdns.org:8888/ports-mlops/predictions/latest` |
| On-demand predict | `POST http://jimitogni.duckdns.org:8888/ports-mlops/predict` |
| MLflow UI (experiments + registry) | `http://mlflow-jimitogni.duckdns.org:8888` |
| Airflow UI | `http://airflow-jimitogni.duckdns.org:8888/` |
| Grafana | `http://grafana-jimitogni.duckdns.org:8888` |
| Prometheus | `http://prometheus-jimitogni.duckdns.org:8888` (Basic Auth) |

`POST /predict` example body:
```json
{"terminal_id":"T1","commodity_type":"soybean","number_of_trains_waiting":10,
 "number_of_wagons":120,"tons_scheduled":8000,"queue_time_hours":5.5,
 "equipment_availability":0.87,"rain_mm":12.0,"shift":"day",
 "harvest_season_flag":1,"operational_restriction_flag":0,"forecast_horizon":"D+1"}
```

### B. Show CI/CD — trigger a deploy

Any push to `main` runs **CI → Build and Push Image → Deploy Homelab**
automatically. To demo it live, push a trivial change and watch the three
workflows in the GitHub Actions tab.

### C. Trigger Model Promotion

GitHub → **Actions → Model Promotion → Run workflow**. It runs on the
self-hosted homelab runner and promotes the current `Candidate` →
`Champion`/`Production` in the MLflow registry. If the `production` environment
has a required-reviewer gate, approve it when prompted (call it out as a
"production gate" — it's intentional).

> For a *new* version to promote, a fresh `Candidate` must exist. If you want to
> show promotion moving a new model, training has to run on the homelab first
> (`make docker-train` there). Otherwise promotion re-affirms the current one.

---

## 4. What was fixed in the 2026-05-14 prep session

All on `main`, all deploys green. Commits: `7f8b14c`, `7d2b063`, `1e3f6e3`,
`874aa65`, `774b21c`.

1. **Deploy failed — `tar: Cannot change mode` on `./models`.** Containers run
   as root and re-own bind-mounted dirs; tar couldn't chmod them on extract.
   Fix: deploy bundle is now code-only — `data/`, `reports/`, `models/`,
   `mlruns/` excluded; the app's `ensure_directories()` recreates them.
2. **Local dir ownership** (`data/`/`models/`/`reports/` owned by `50000:root`)
   restored to `jimi:jimi`.
3. **`/predictions/latest` served stale data** — it picked the prediction file
   whose filename date sorted last, ignoring the canonical
   `latest_predictions.csv`. Fixed to read `latest_predictions.csv`.
4. **Airflow 404 after every deploy.** `docker-compose.demo.yml` and
   `docker-compose.airflow.yml` share one Compose project, so the deploy's
   `--remove-orphans` deleted the Airflow containers each time. Fix: dropped
   `--remove-orphans`.
5. **`model-promotion.yml` was a no-op** — ran on a GitHub-hosted runner that
   can't reach the homelab MLflow. Fix: `runs-on: self-hosted` calling
   `make docker-promote-model`.
6. End-to-end demo + all 4 API endpoints verified working on the homelab.

---

## 5. Known tech debt (good interview talking points)

- **Containers run as root** and re-own bind-mounted host dirs. Proper fix: a
  non-root container user matching the host UID.
- **Demo and Airflow compose files share one Compose project** (no explicit
  `name:`). Should be separate projects — not changed yet because renaming
  orphans Airflow's stateful volumes.
- **Reference Python is 3.12**; the local-Python path needs a 3.12 venv. The
  Docker-only path is the portable one.

---

## 6. If you do end up running the pipeline on the homelab

```bash
make clean                 # clear stale predictions / model files
make setup
make docker-demo-normal    # test → generate → train → promote → predict (~3 min)
```
This is verified working **on the homelab only** (needs `mlflow_default` network).
