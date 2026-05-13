from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import get_settings
from src.utils.dates import FORECAST_HORIZONS, forecast_date, parse_date
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)

COMMODITIES = ("soybean", "corn", "sugar", "fertilizer", "iron_ore")
SHIFTS = ("day", "night", "swing")
TERMINAL_BASELINES = {
    "T1": 860.0,
    "T2": 790.0,
    "T3": 735.0,
    "T4": 685.0,
}
COMMODITY_MULTIPLIERS = {
    "soybean": 1.02,
    "corn": 0.96,
    "sugar": 0.88,
    "fertilizer": 0.82,
    "iron_ore": 1.10,
}
SHIFT_MULTIPLIERS = {
    "day": 1.00,
    "swing": 0.97,
    "night": 0.92,
}


def _date_range(start_date: date, end_date: date) -> list[date]:
    days = (end_date - start_date).days
    if days < 0:
        raise ValueError("end_date must be greater than or equal to start_date")
    return [start_date + timedelta(days=offset) for offset in range(days + 1)]


def _is_harvest_season(day: date) -> int:
    return int(day.month in {2, 3, 4, 5, 9, 10})


def _scenario_flags(day: date, scenario: str, drift_start_date: date | None) -> tuple[bool, bool]:
    data_drift = scenario in {"drift", "prediction_drift"} and (
        drift_start_date is None or day >= drift_start_date
    )
    prediction_drift = scenario == "prediction_drift" and (
        drift_start_date is None or day >= drift_start_date
    )
    return data_drift, prediction_drift


def _sample_row(
    rng: np.random.Generator,
    operation_date: date,
    terminal_id: str,
    scenario: str,
    drift_start_date: date | None,
) -> dict[str, object]:
    data_drift, prediction_drift = _scenario_flags(operation_date, scenario, drift_start_date)
    harvest = _is_harvest_season(operation_date)
    commodity = str(rng.choice(COMMODITIES, p=[0.30, 0.24, 0.16, 0.16, 0.14]))
    shift = str(rng.choice(SHIFTS, p=[0.50, 0.25, 0.25]))

    rain_shape = 1.6 if harvest else 1.2
    rain_scale = 8.0 if data_drift else 5.0
    rain_mm = float(np.round(rng.gamma(rain_shape, rain_scale), 2))
    trains_waiting = int(max(0, rng.poisson(5 + harvest * 2 + data_drift * 3)))
    wagons = int(max(30, rng.normal(110 + trains_waiting * 3, 18)))
    tons_scheduled = float(np.round(wagons * rng.normal(67, 6), 2))
    queue_time = float(np.round(max(0, rng.normal(2.5 + trains_waiting * 0.42 + data_drift * 1.8, 1.25)), 2))
    equipment_availability = float(
        np.round(np.clip(rng.normal(0.88 - data_drift * 0.10, 0.055), 0.45, 0.99), 3)
    )
    restriction = int(rng.random() < (0.06 + harvest * 0.05 + data_drift * 0.09))

    baseline = TERMINAL_BASELINES[terminal_id]
    baseline *= COMMODITY_MULTIPLIERS[commodity]
    baseline *= SHIFT_MULTIPLIERS[shift]
    if data_drift:
        baseline *= 0.92
    if prediction_drift:
        baseline *= 0.86

    noise_sigma = 36 + harvest * 34 + data_drift * 42
    productivity = (
        baseline
        + (equipment_availability - 0.85) * 410
        - rain_mm * 4.2
        - queue_time * 9.5
        - trains_waiting * 5.0
        - restriction * 95
        - harvest * 28
        + rng.normal(0, noise_sigma)
    )
    productivity = float(np.round(np.clip(productivity, 80, 1500), 2))

    return {
        "operation_date": operation_date.isoformat(),
        "terminal_id": terminal_id,
        "commodity_type": commodity,
        "number_of_trains_waiting": trains_waiting,
        "number_of_wagons": wagons,
        "tons_scheduled": tons_scheduled,
        "queue_time_hours": queue_time,
        "equipment_availability": equipment_availability,
        "rain_mm": rain_mm,
        "shift": shift,
        "day_of_week": operation_date.weekday(),
        "harvest_season_flag": harvest,
        "operational_restriction_flag": restriction,
        "actual_productivity_tons_hour": productivity,
    }


def generate_synthetic_data(
    start_date: str | date,
    end_date: str | date,
    terminals: tuple[str, ...] | list[str] | None = None,
    scenario: str = "normal",
    drift_start_date: str | date | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    settings = get_settings()
    selected_terminals = tuple(terminals or settings.expected_terminals)
    if scenario == "missing_terminal" and selected_terminals:
        selected_terminals = selected_terminals[:-1]
    rng = np.random.default_rng(seed)
    drift_day = parse_date(drift_start_date) if drift_start_date else None
    rows = [
        _sample_row(rng, day, terminal_id, scenario, drift_day)
        for day in _date_range(parse_date(start_date), parse_date(end_date))
        for terminal_id in selected_terminals
    ]
    return pd.DataFrame(rows)


def generate_daily_operational_forecast(
    execution_date: str | date,
    terminals: tuple[str, ...] | list[str] | None = None,
    scenario: str = "normal",
    seed: int = 123,
) -> pd.DataFrame:
    settings = get_settings()
    exec_day = parse_date(execution_date)
    rows: list[pd.DataFrame] = []
    for horizon in FORECAST_HORIZONS:
        day = forecast_date(exec_day, horizon)
        daily = generate_synthetic_data(
            start_date=day,
            end_date=day,
            terminals=terminals or settings.expected_terminals,
            scenario=scenario,
            drift_start_date=day if scenario in {"drift", "prediction_drift"} else None,
            seed=seed + int(horizon.split("+")[1]),
        )
        daily["execution_date"] = exec_day.isoformat()
        daily["forecast_date"] = day.isoformat()
        daily["forecast_horizon"] = horizon
        rows.append(daily.drop(columns=["actual_productivity_tons_hour"], errors="ignore"))
    return pd.concat(rows, ignore_index=True)


def save_dataset(df: pd.DataFrame, output: str | Path) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    LOGGER.info("Wrote %s rows to %s", len(df), output_path)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic port productivity data.")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--scenario", default="normal", choices=["normal", "missing_terminal", "drift", "prediction_drift"])
    parser.add_argument("--drift-start-date", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    settings = get_settings()
    args = build_arg_parser().parse_args()
    df = generate_synthetic_data(
        start_date=args.start_date,
        end_date=args.end_date,
        scenario=args.scenario,
        drift_start_date=args.drift_start_date,
        seed=args.seed,
    )
    save_dataset(df, args.output or settings.raw_data_path)


if __name__ == "__main__":
    main()

