from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def import_file(path: Path) -> None:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    dag_dir = project_root / "dags"
    dag_files = sorted(dag_dir.glob("*.py"))
    if not dag_files:
        raise RuntimeError(f"No DAG files found under {dag_dir}")
    for dag_file in dag_files:
        import_file(dag_file)
        print(f"OK {dag_file.name}")


if __name__ == "__main__":
    main()

