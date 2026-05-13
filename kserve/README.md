# KServe Extension

KServe is useful when the business adds online inference requirements. The core use case in this interview case remains Airflow batch prediction.

Prerequisites:

- A Kubernetes cluster with KServe installed.
- An image pushed to a registry the cluster can pull.
- `k8s/secret.yaml` created from `k8s/secret.example.yaml`.
- MLflow reachable from the cluster, or a local registry volume mounted into the custom predictor image.

This extension serves the same model through the FastAPI predictor container:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
cp k8s/secret.example.yaml k8s/secret.yaml
# edit k8s/secret.yaml before applying
kubectl apply -f k8s/secret.yaml
kubectl apply -f kserve/inferenceservice.yaml
```

Check readiness:

```bash
kubectl -n port-productivity-mlops get inferenceservice port-productivity-forecaster
```

Example request:

```bash
curl -X POST http://<kserve-host>/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "terminal_id": "T1",
    "commodity_type": "soybean",
    "number_of_trains_waiting": 10,
    "number_of_wagons": 120,
    "tons_scheduled": 8000,
    "queue_time_hours": 5.5,
    "equipment_availability": 0.87,
    "rain_mm": 12.0,
    "shift": "day",
    "harvest_season_flag": 1,
    "operational_restriction_flag": 0,
    "forecast_horizon": "D+1"
  }'
```

Expected response:

```json
{
  "terminal_id": "T1",
  "forecast_horizon": "D+1",
  "predicted_productivity_tons_hour": 780.4,
  "model_name": "port_productivity_forecaster",
  "model_version": "3"
}
```

The fallback custom container is intentional: it keeps the model, feature engineering, and MLflow alias lookup together for the demo. A pure sklearn-server deployment is possible after exporting a self-contained artifact.
