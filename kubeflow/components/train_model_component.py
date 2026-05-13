from __future__ import annotations

import subprocess


def main() -> None:
    subprocess.run(["python", "-m", "src.pipelines.training_pipeline", "--data-path", "data/raw/port_productivity.csv"], check=True)


if __name__ == "__main__":
    main()

