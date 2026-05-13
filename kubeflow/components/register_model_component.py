from __future__ import annotations

import subprocess


def main() -> None:
    subprocess.run(["python", "-m", "src.models.promote_model"], check=True)


if __name__ == "__main__":
    main()

