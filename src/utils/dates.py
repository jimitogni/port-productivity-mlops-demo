from __future__ import annotations

from datetime import date, datetime, timedelta, timezone


FORECAST_HORIZONS = ("D+1", "D+2", "D+3")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def horizon_to_days(horizon: str) -> int:
    if not horizon.startswith("D+"):
        raise ValueError(f"Invalid forecast horizon: {horizon}")
    return int(horizon.split("+", 1)[1])


def forecast_date(execution_date: str | date | datetime, horizon: str) -> date:
    return parse_date(execution_date) + timedelta(days=horizon_to_days(horizon))

