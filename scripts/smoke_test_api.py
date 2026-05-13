from __future__ import annotations

import argparse
import json
import urllib.request


EXAMPLE_PAYLOAD = {
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
    "forecast_horizon": "D+1",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the FastAPI demo service.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8015")
    args = parser.parse_args()
    with urllib.request.urlopen(f"{args.base_url}/health", timeout=10) as response:
        print(response.read().decode())
    request = urllib.request.Request(
        f"{args.base_url}/predict",
        data=json.dumps(EXAMPLE_PAYLOAD).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        print(response.read().decode())


if __name__ == "__main__":
    main()

