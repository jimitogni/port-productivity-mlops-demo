from __future__ import annotations

from datetime import date

import pandas as pd
from fastapi import FastAPI, HTTPException, Response
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
