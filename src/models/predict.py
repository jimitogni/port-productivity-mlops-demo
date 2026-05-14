from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.settings import get_settings
from src.features.build_features import expected_feature_columns
from src.models.register_model import _load_registry
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)


@dataclass
class ModelBundle:
    model: Any
    model_name: str
    model_version: str
    feature_columns: list[str]
    source: str


def _load_from_mlflow(alias: str) -> ModelBundle | None:
    settings = get_settings()
    try:
        import mlflow.pyfunc
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        return None
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        model_uri = f"models:/{settings.model_name}@{alias}"
        model = mlflow.pyfunc.load_model(model_uri)
        client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        version = client.get_model_version_by_alias(settings.model_name, alias)
        return ModelBundle(
            model=model,
            model_name=settings.model_name,
            model_version=str(version.version),
            feature_columns=expected_feature_columns(),
            source=f"mlflow:{alias}",
        )
    except Exception as exc:
        LOGGER.info("Could not load MLflow alias %s: %s", alias, exc)
        return None


def _load_from_local_registry(alias: str) -> ModelBundle | None:
    settings = get_settings()
    registry = _load_registry()
    version = registry.get("aliases", {}).get(alias)
    if not version:
        return None
    metadata = registry["versions"][str(version)]
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("joblib is required to load the local registered model") from exc

    # registry.json stores model_path as an absolute path written by whichever
    # container ran training (e.g. /opt/airflow/project/models/...), which won't
    # exist in another container. Reconstruct the path from the current
    # settings.models_dir, falling back to the stored path then latest_model.joblib.
    candidate_paths = [
        settings.models_dir / settings.model_name / str(version) / "model.joblib",
        Path(metadata["model_path"]),
        settings.models_dir / "latest_model.joblib",
    ]
    model_path = next((p for p in candidate_paths if p.exists()), None)
    if model_path is None:
        LOGGER.info("Local registry alias %s: no model file found in %s", alias, candidate_paths)
        return None
    try:
        model = joblib.load(model_path)
    except Exception as exc:
        LOGGER.info("Could not load local model %s from %s: %s", alias, model_path, exc)
        return None
    return ModelBundle(
        model=model,
        model_name=settings.model_name,
        model_version=str(version),
        feature_columns=metadata.get("feature_columns", expected_feature_columns()),
        source=f"local:{alias}",
    )


def load_registered_model(preferred_aliases: tuple[str, ...] = ("Production", "Champion", "Candidate", "Challenger")) -> ModelBundle:
    for alias in preferred_aliases:
        bundle = _load_from_mlflow(alias) or _load_from_local_registry(alias)
        if bundle:
            LOGGER.info("Loaded model %s version %s from %s", bundle.model_name, bundle.model_version, bundle.source)
            return bundle
    raise RuntimeError(
        "No registered model was found. Run `make train` and optionally `make promote-model` first."
    )


def predict_with_bundle(bundle: ModelBundle, features: pd.DataFrame) -> pd.Series:
    X = features[bundle.feature_columns].copy()
    predictions = bundle.model.predict(X)
    return pd.Series(predictions, index=features.index, name="predicted_productivity_tons_hour")
