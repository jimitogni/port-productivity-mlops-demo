from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    metrics_path = Path("models/latest_training_metrics.json")
    if not metrics_path.exists():
        raise RuntimeError("Training metrics were not produced")
    metrics = json.loads(metrics_path.read_text())
    print(json.dumps(metrics["metrics"], indent=2))


if __name__ == "__main__":
    main()

