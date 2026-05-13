from __future__ import annotations

from pathlib import Path
import re
from typing import Mapping

from src.config.settings import get_settings


METRIC_HELP = {
    "pipeline_success": "Whether the latest pipeline execution succeeded.",
    "pipeline_duration_seconds": "Latest pipeline execution duration in seconds.",
    "input_rows_total": "Number of input rows processed by the latest execution.",
    "prediction_rows_total": "Number of prediction rows produced by the latest execution.",
    "validation_failures_total": "Validation failures observed in the latest execution.",
    "model_mae": "Latest model mean absolute error.",
    "model_rmse": "Latest model root mean squared error.",
    "data_drift_detected": "Whether data drift was detected in the latest execution.",
    "prediction_drift_detected": "Whether prediction drift was detected in the latest execution.",
    "current_model_version": "Current model version used by the latest execution.",
}


def _metric_name(name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_:]", "_", name)
    if not re.match(r"^[a-zA-Z_:]", sanitized):
        sanitized = f"metric_{sanitized}"
    return f"port_productivity_{sanitized}"


def _render_prometheus_metrics_fallback(metrics: Mapping[str, float | int]) -> str:
    lines: list[str] = []
    for name, value in metrics.items():
        metric_name = _metric_name(name)
        help_text = METRIC_HELP.get(name, metric_name)
        lines.append(f"# HELP {metric_name} {help_text}")
        lines.append(f"# TYPE {metric_name} gauge")
        lines.append(f"{metric_name} {float(value)}")
    return "\n".join(lines) + "\n"


def render_prometheus_metrics(metrics: Mapping[str, float | int]) -> str:
    try:
        from prometheus_client import CollectorRegistry, Gauge, generate_latest
    except Exception:
        return _render_prometheus_metrics_fallback(metrics)

    registry = CollectorRegistry()
    for name, value in metrics.items():
        Gauge(_metric_name(name), METRIC_HELP.get(name, _metric_name(name)), registry=registry).set(float(value))
    return generate_latest(registry).decode("utf-8")


def write_prometheus_metrics(metrics: Mapping[str, float | int], path: str | Path | None = None) -> Path:
    settings = get_settings()
    output = Path(path or settings.prometheus_metrics_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_prometheus_metrics(metrics))
    return output


def read_latest_metrics_text() -> str:
    settings = get_settings()
    if settings.prometheus_metrics_path.exists():
        return settings.prometheus_metrics_path.read_text()
    return render_prometheus_metrics({"pipeline_success": 0})
