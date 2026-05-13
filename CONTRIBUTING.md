# Contributing — MLOps Workflow

This document describes how Carlos (Data Scientist) and the MLOps engineer collaborate across the full model lifecycle: from experiment to production and back.

---

## Roles and responsibilities

| Role | Person | Owns |
|------|--------|------|
| Data Scientist | Carlos | Feature logic, algorithm selection, hyperparameter search, experiment notebooks |
| MLOps Engineer | MLOps team | Pipeline code, orchestration, deployment, monitoring, registry |
| Operations | Camila / Ana | Defines acceptable forecast error (see `thresholds.yml`) |
| Commercial | Bruna | Reviews SLA risk before Production promotion |

---

## Model development workflow

```
Carlos experiments          MLOps operationalises        Operations validates
──────────────────         ──────────────────────────   ────────────────────
notebook / local run   →   PR against main (feature/)   review thresholds.yml
log to MLflow              CI runs tests + DAG check     approve production env
register as Candidate      promotion checks pass          model goes Production
                           deploy via GitHub Actions
```

### Step-by-step

1. **Carlos explores** locally or in a notebook. He logs every run to MLflow experiment `port_productivity_training` with `mlflow.start_run()`.

2. **Carlos opens a PR** against `main` when he is satisfied with a candidate. The PR must include:
   - Updated `src/features/build_features.py` (if features changed) with an incremented `FEATURE_VERSION`
   - Updated `requirements.txt` if new libraries are added
   - A short description of what changed and why (MAE improvement, new feature rationale)

3. **CI validates the PR** (`ci.yml`): pytest, compile check, DAG validation, training smoke test.

4. **MLOps engineer reviews** the PR, checking:
   - Feature version bumped if `build_features.py` changed
   - No data leakage (train/test split remains time-based)
   - No new hard-coded paths or secrets
   - Requirements pinned to a range, not a fixed version

5. **PR merges → training pipeline runs** (Airflow weekly or triggered manually via `make docker-train`). The resulting model lands in the local registry and MLflow as `Candidate`.

6. **Promotion is manual and gated** (`model-promotion.yml` workflow):
   - Requires `production` GitHub environment approval
   - `promote_model.py` runs `promotion_checks()` automatically against `thresholds.yml`
   - If checks fail, the workflow fails with a clear reason — Carlos and MLOps review together

7. **Post-promotion monitoring**: MLOps watches Grafana + Prometheus alerts. If `DataDriftDetected` or `ModelMAEDegradation` fires, Carlos is notified and the cycle restarts.

---

## Feature version policy

| Change | Action required |
|--------|----------------|
| New feature column added | Bump `FEATURE_VERSION` (e.g. `v1` → `v2`) in `.env` / `thresholds.yml` comment |
| Existing feature logic changed | Bump `FEATURE_VERSION` |
| Bug fix that doesn't change column set | No bump needed |
| Hyperparameter or algorithm change only | No bump needed |

**Why it matters:** every prediction row carries `feature_version`. If you change feature logic without bumping the version, historical predictions become incomparable.

---

## Promotion criteria (sourced from `thresholds.yml`)

A Candidate model must satisfy **all** of the following before it can be promoted to Production:

| Check | Threshold | Rationale |
|-------|-----------|-----------|
| MAE D+1 | ≤ 15 t/h | Drives train retention decisions (Camila) |
| MAE D+2 | ≤ 18 t/h | Contingency planning horizon (Ana) |
| MAE D+3 | ≤ 22 t/h | Commercial buffer (Bruna) |
| Overall MAPE | ≤ 15% | Relative accuracy floor |
| R² | ≥ 0.65 | Minimum predictive power |
| MAE vs Champion | ≤ +5% | Must not regress vs current Production |
| Per-terminal MAE vs Champion | ≤ +10% | No single terminal can degrade silently |

Thresholds are in [`thresholds.yml`](thresholds.yml) and loaded automatically by `promote_model.py`.

---

## Rollback runbook

If a recently promoted model causes operational issues:

```bash
# Option A — rollback via local registry (fast, no MLflow needed)
python -m src.models.promote_model --rollback

# Option B — rollback a specific version
python -m src.models.promote_model --rollback --target-version 3

# Verify which version is now Production
cat models/registry.json | python -m json.tool | grep -A2 '"Production"'
```

After rollback:
1. Open a GitHub issue describing the symptom
2. Carlos investigates root cause in MLflow experiment history
3. A new Candidate is prepared before the next promotion attempt

---

## Retraining trigger checklist

Initiate a retraining cycle when **any** of the following are true:

- [ ] `DataDriftDetected` alert fired for 2 consecutive days
- [ ] `ModelMAEDegradation` alert fired (live MAE > 20% above training MAE)
- [ ] `StaleModel` alert fired (no promotion in 30 days during active harvest season)
- [ ] Carlos identifies a structural change in port operations (new terminal, commodity mix shift)
- [ ] Business requests updated SLA thresholds (update `thresholds.yml` first, then retrain)

---

## Local development commands

```bash
make generate-data    # generate synthetic training data
make train            # run training pipeline, register Candidate
make promote-model    # promote Candidate → Production (runs checks)
make predict          # run daily prediction for today
make run-api          # start FastAPI locally on :8015

# Docker variants (recommended, mirrors production)
make docker-train
make docker-promote-model
make docker-predict
make docker-run-api
```
