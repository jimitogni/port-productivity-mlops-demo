from __future__ import annotations

import subprocess


def main() -> None:
    subprocess.run(["python", "-m", "src.validation.validate_input_data"], check=True)


if __name__ == "__main__":
    main()

