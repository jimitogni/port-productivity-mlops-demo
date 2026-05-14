from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _load_thresholds(project_root: Path) -> dict[str, Any]:
    """Load thresholds.yml; return defaults if file is missing or PyYAML unavailable."""
    path = project_root / "thresholds.yml"
    if path.exists():
        try:
            import yaml
            return yaml.safe_load(path.read_text()) or {}
        except Exception:
            pass
    return {
        "promotion_gate": {
            "max_mae_d1": 15.0, "max_mae_d2": 18.0, "max_mae_d3": 22.0,
            "max_mape": 0.15, "min_r2": 0.65,
            "max_mae_degradation_pct": 0.05, "max_rmse_degradation_pct": 0.05,
            "max_terminal_mae_degradation_pct": 0.10,
        },
        "alert_thresholds": {
            "data_drift_share": 0.30, "prediction_mean_shift_pct": 0.25,
            "mae_degradation_pct": 0.20, "consecutive_drift_days": 2,
        },
    }


def _load_dotenv(project_root: Path) -> None:
    """Small dotenv loader so core settings do not require python-dotenv."""
    env_file = project_root / ".env"
    if not env_file.exists():
        return
    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _project_root() -> Path:
    env_root = os.getenv("PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    reports_dir: Path
    models_dir: Path
    raw_data_path: Path
    predictions_dir: Path
    monitoring_dir: Path
    mlflow_tracking_uri: str
    model_name: str
    expected_terminals: tuple[str, ...]
    prometheus_metrics_port: int
    prometheus_metrics_path: Path
    feature_version: str
    pipeline_version: str
    database_url: str | None
    public_path_prefix: str
    thresholds: dict[str, Any] = None  # type: ignore[assignment]

    def promotion_gate(self) -> dict[str, Any]:
        return (self.thresholds or {}).get("promotion_gate", {})

    def alert_thresholds(self) -> dict[str, Any]:
        return (self.thresholds or {}).get("alert_thresholds", {})

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.predictions_dir,
            self.monitoring_dir,
            self.reports_dir,
            self.reports_dir / "evidently",
            self.reports_dir / "daily",
            self.models_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
            # Keep shared dirs group-writable so containers running under different
            # UIDs but the common root group (e.g. the API as root and Airflow as
            # uid 50000) can both write reports/predictions/metrics.
            try:
                path.chmod(0o775)
            except (PermissionError, OSError):
                pass


def get_settings() -> Settings:
    root = _project_root()
    _load_dotenv(root)
    thresholds = _load_thresholds(root)
    data_dir = Path(os.getenv("DATA_DIR", root / "data")).expanduser().resolve()
    reports_dir = Path(os.getenv("REPORTS_DIR", root / "reports")).expanduser().resolve()
    models_dir = Path(os.getenv("MODELS_DIR", root / "models")).expanduser().resolve()
    expected = tuple(
        item.strip()
        for item in os.getenv("EXPECTED_TERMINALS", "T1,T2,T3,T4").split(",")
        if item.strip()
    )
    metrics_path = Path(
        os.getenv("PROMETHEUS_METRICS_PATH", data_dir / "monitoring" / "port_productivity_metrics.prom")
    ).expanduser().resolve()
    settings = Settings(
        project_root=root,
        data_dir=data_dir,
        reports_dir=reports_dir,
        models_dir=models_dir,
        raw_data_path=Path(
            os.getenv("RAW_DATA_PATH", data_dir / "raw" / "port_productivity.csv")
        ).expanduser().resolve(),
        predictions_dir=Path(
            os.getenv("PREDICTIONS_DIR", data_dir / "predictions")
        ).expanduser().resolve(),
        monitoring_dir=Path(
            os.getenv("MONITORING_DIR", data_dir / "monitoring")
        ).expanduser().resolve(),
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", f"file://{root / 'mlruns'}"),
        model_name=os.getenv("MODEL_NAME", "port_productivity_forecaster"),
        expected_terminals=expected,
        prometheus_metrics_port=int(os.getenv("PROMETHEUS_METRICS_PORT", "8015")),
        prometheus_metrics_path=metrics_path,
        feature_version=os.getenv("FEATURE_VERSION", "v1"),
        pipeline_version=os.getenv("PIPELINE_VERSION", "v1"),
        database_url=os.getenv("DATABASE_URL") or None,
        public_path_prefix=os.getenv("PUBLIC_PATH_PREFIX", "/ports-mlops"),
        thresholds=thresholds,
    )
    settings.ensure_directories()
    return settings

