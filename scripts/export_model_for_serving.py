from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from src.config.settings import get_settings
from src.models.predict import load_registered_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the registered model artifact for simple serving.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    settings = get_settings()
    output_dir = Path(args.output_dir or settings.models_dir / "serving")
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_registered_model()
    if bundle.source.startswith("local:"):
        registry = json.loads((settings.models_dir / "registry.json").read_text())
        metadata = registry["versions"][bundle.model_version]
        shutil.copy2(metadata["model_path"], output_dir / "model.joblib")
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    else:
        (output_dir / "metadata.json").write_text(
            json.dumps({"model_name": bundle.model_name, "model_version": bundle.model_version, "source": bundle.source}, indent=2)
        )
    print(f"Exported serving metadata to {output_dir}")


if __name__ == "__main__":
    main()

