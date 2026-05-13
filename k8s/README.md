# Kubernetes Extension

Kubernetes is optional for this demo. The core production workflow is Airflow batch orchestration because the business requirement is a daily D0 forecast for D+1, D+2, and D+3.

Use these manifests when you want to show how the same container can run on k3s, kind, or minikube:

- `deployment-api.yaml` runs the FastAPI demo inference API.
- `service-api.yaml` exposes `/health`, `/metrics`, `/predictions/latest`, and `/predict` inside the cluster.
- `cronjob-daily-prediction.yaml` runs the daily batch prediction pipeline as a Kubernetes CronJob.
- `job-training.yaml` runs a one-off training job.
- `ingress.example.yaml` shows a Traefik ingress path for `/ports-mlops`.

Before applying:

1. Replace `ghcr.io/OWNER/port-productivity-mlops-demo:latest` with your image.
2. Copy `secret.example.yaml` to `secret.yaml` and set `MLFLOW_TRACKING_URI` and `DATABASE_URL`.
3. Make sure MLflow and PostgreSQL are reachable from the cluster, or use file-based storage for a local demo.

Commands:

```bash
make k8s-apply
make k8s-status
make k8s-logs
make k8s-delete
```

For k3s with Traefik, prefer adapting the ingress to your cluster routing. For this homelab Docker Compose remains simpler because the existing public routing is already handled by the host Traefik on port `80`, with public traffic coming from router port `8888`.

