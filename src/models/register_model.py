from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config.settings import get_settings
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)


def _registry_path() -> Path:
    return get_settings().models_dir / "registry.json"


def _load_registry() -> dict[str, Any]:
    path = _registry_path()
    if not path.exists():
        return {"aliases": {}, "versions": {}}
    return json.loads(path.read_text())


def _save_registry(registry: dict[str, Any]) -> None:
    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2, sort_keys=True))


def _next_local_version(registry: dict[str, Any]) -> str:
    if not registry["versions"]:
        return "1"
    numeric_versions = [int(version) for version in registry["versions"].keys() if str(version).isdigit()]
    return str(max(numeric_versions, default=0) + 1)


def register_local_model(
    model: Any,
    metrics: dict[str, float],
    terminal_metrics: dict[str, dict[str, float]],
    feature_columns: list[str],
    run_id: str,
    dataset_path: str,
    training_period: dict[str, str],
    model_type: str,
    alias: str = "Candidate",
) -> str:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("joblib is required to persist the local model registry") from exc

    settings = get_settings()
    registry = _load_registry()
    version = _next_local_version(registry)
    version_dir = settings.models_dir / settings.model_name / version
    version_dir.mkdir(parents=True, exist_ok=True)
    model_path = version_dir / "model.joblib"
    joblib.dump(model, model_path)

    metadata = {
        "model_name": settings.model_name,
        "model_version": version,
        "model_type": model_type,
        "run_id": run_id,
        "metrics": metrics,
        "terminal_metrics": terminal_metrics,
        "feature_columns": feature_columns,
        "dataset_path": dataset_path,
        "training_period": training_period,
        "validation_status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
    }
    (version_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    registry["versions"][version] = metadata
    registry["aliases"][alias] = version
    if "Champion" not in registry["aliases"] and "Production" not in registry["aliases"]:
        registry["aliases"]["Champion"] = version
        registry["aliases"]["Production"] = version
        LOGGER.info("Bootstrapped Champion/Production aliases to local model version %s", version)
    _save_registry(registry)
    shutil.copy2(model_path, settings.models_dir / "latest_model.joblib")
    LOGGER.info("Registered local model version %s as %s", version, alias)
    return version


def register_mlflow_model(
    model: Any,
    artifact_path: str,
    feature_columns_path: Path,
    metrics: dict[str, float] | None = None,
    terminal_metrics: dict[str, dict[str, float]] | None = None,
    feature_columns: list[str] | None = None,
    alias: str = "Candidate",
) -> str | None:
    settings = get_settings()
    try:
        import mlflow
        import mlflow.sklearn
        from mlflow.tracking import MlflowClient
    except ImportError:
        LOGGER.warning("MLflow is not installed; skipped MLflow model registration")
        return None

    try:
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            registered_model_name=settings.model_name,
        )
        mlflow.log_artifact(str(feature_columns_path), artifact_path="features")
        version = getattr(model_info, "registered_model_version", None)
        if version:
            client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
            try:
                client.set_registered_model_alias(settings.model_name, alias, str(version))
            except Exception:
                client.set_model_version_tag(settings.model_name, str(version), "alias", alias)
            client.set_model_version_tag(settings.model_name, str(version), "validation_status", "passed")
            for key, value in (metrics or {}).items():
                client.set_model_version_tag(settings.model_name, str(version), f"metric_{key}", str(value))
            client.set_model_version_tag(
                settings.model_name,
                str(version),
                "feature_columns",
                json.dumps(feature_columns or []),
            )
            client.set_model_version_tag(
                settings.model_name,
                str(version),
                "terminal_metrics",
                json.dumps(terminal_metrics or {}),
            )
            LOGGER.info("Registered MLflow model version %s as %s", version, alias)
            return str(version)
    except Exception as exc:
        LOGGER.warning("MLflow model registration failed; local registry remains available: %s", exc)
    return None


def get_local_model_metadata(alias: str = "Champion") -> dict[str, Any] | None:
    registry = _load_registry()
    version = registry.get("aliases", {}).get(alias)
    if not version:
        return None
    return registry.get("versions", {}).get(str(version))
