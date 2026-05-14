from __future__ import annotations

import html
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from src.config.settings import get_settings
from src.data.generate_synthetic_data import generate_synthetic_data
from src.data.load_data import load_latest_predictions, load_port_productivity_data
from src.features.build_features import build_inference_features
from src.models.predict import load_registered_model, predict_with_bundle
from src.monitoring.prometheus_metrics import read_latest_metrics_text
from src.utils.dates import forecast_date


app = FastAPI(
    title="Port Productivity MLOps Demo",
    version="1.0.0",
    root_path=get_settings().public_path_prefix,
)


class PredictionRequest(BaseModel):
    terminal_id: str
    commodity_type: str
    number_of_trains_waiting: int = Field(ge=0)
    number_of_wagons: int = Field(gt=0)
    tons_scheduled: float = Field(gt=0)
    queue_time_hours: float = Field(ge=0)
    equipment_availability: float = Field(ge=0, le=1)
    rain_mm: float = Field(ge=0)
    shift: str
    harvest_season_flag: int = Field(ge=0, le=1)
    operational_restriction_flag: int = Field(ge=0, le=1)
    forecast_horizon: str = "D+1"
    execution_date: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "port-productivity-mlops-demo"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(read_latest_metrics_text(), media_type="text/plain; version=0.0.4")


@app.get("/predictions/latest")
def latest_predictions() -> list[dict[str, object]]:
    df = load_latest_predictions()
    return df.to_dict(orient="records")


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict[str, object]:
    settings = get_settings()
    execution_date = payload.execution_date or date.today().isoformat()
    forecast_day = forecast_date(execution_date, payload.forecast_horizon)
    row = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    row["execution_date"] = execution_date
    row["forecast_date"] = forecast_day.isoformat()
    row["operation_date"] = forecast_day.isoformat()
    row["day_of_week"] = forecast_day.weekday()
    input_df = pd.DataFrame([row])
    try:
        history_df = load_port_productivity_data(settings.raw_data_path)
    except FileNotFoundError:
        history_df = generate_synthetic_data("2024-01-01", execution_date)
    try:
        features = build_inference_features(input_df, history_df)
        bundle = load_registered_model()
        prediction = float(predict_with_bundle(bundle, features).iloc[0])
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "terminal_id": payload.terminal_id,
        "forecast_horizon": payload.forecast_horizon,
        "predicted_productivity_tons_hour": round(prediction, 2),
        "model_name": bundle.model_name,
        "model_version": bundle.model_version,
    }


_REPORTS_INDEX_CSS = """
body { font-family: Arial, sans-serif; margin: 32px; background: #f8f9fa; color: #212529; }
h1 { color: #003366; margin-bottom: 4px; }
h2 { color: #003366; font-size: 18px; border-bottom: 2px solid #e9ecef; padding-bottom: 6px;
     margin-top: 28px; }
.subtitle { color: #6c757d; font-size: 14px; margin-bottom: 24px; }
ul { list-style: none; padding: 0; }
li { padding: 8px 0; border-bottom: 1px solid #e9ecef; }
a { color: #0066cc; text-decoration: none; }
a:hover { text-decoration: underline; }
.size { color: #6c757d; font-size: 12px; margin-left: 8px; }
.empty { color: #6c757d; font-style: italic; }
"""


def _render_reports_index(base: Path, rel: str) -> str:
    """Render an HTML directory listing for the reports tree."""
    # Absolute href base so links work regardless of trailing slash / depth.
    prefix = get_settings().public_path_prefix.rstrip("/")
    href_base = f"{prefix}/reports"
    current = (base / rel).resolve()
    rows = ""
    if rel:
        parent = "/".join(rel.rstrip("/").split("/")[:-1])
        parent_href = f"{href_base}/{parent}".rstrip("/")
        rows += f'<li><a href="{parent_href}">⬑ ..</a></li>'
    entries = sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name))
    for entry in entries:
        entry_rel = f"{rel}/{entry.name}".strip("/")
        href = f"{href_base}/{entry_rel}"
        if entry.is_dir():
            rows += f'<li>📁 <a href="{href}">{html.escape(entry.name)}/</a></li>'
        else:
            size_kb = entry.stat().st_size / 1024
            rows += (
                f'<li>📄 <a href="{href}">{html.escape(entry.name)}</a>'
                f'<span class="size">{size_kb:.1f} KB</span></li>'
            )
    if not rows:
        rows = '<li class="empty">No reports generated yet.</li>'
    title = f"/{rel}" if rel else "/"
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Port Productivity — Reports {html.escape(title)}</title>
<style>{_REPORTS_INDEX_CSS}</style></head>
<body>
  <h1>📦 Port Productivity — Reports</h1>
  <div class="subtitle">Index of <strong>{html.escape(title)}</strong> &nbsp;|&nbsp;
    generated {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
  <ul>{rows}</ul>
</body>
</html>"""


@app.get("/reports", response_class=HTMLResponse)
@app.get("/reports/{file_path:path}")
def reports(file_path: str = ""):
    """Browse and download operational + Evidently reports."""
    base = get_settings().reports_dir.resolve()
    target = (base / file_path).resolve()
    if base != target and base not in target.parents:
        raise HTTPException(status_code=404, detail="Not found")
    if not target.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    if target.is_dir():
        return HTMLResponse(_render_reports_index(base, file_path.strip("/")))
    return FileResponse(target)
