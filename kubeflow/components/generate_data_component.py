from __future__ import annotations

import subprocess


def main() -> None:
    subprocess.run(
        [
            "python",
            "-m",
            "src.data.generate_synthetic_data",
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2026-05-12",
            "--output",
            "data/raw/port_productivity.csv",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()

