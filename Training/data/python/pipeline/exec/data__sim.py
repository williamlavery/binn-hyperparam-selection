"""Command-line entrypoint for configured synthetic data-generation sweeps.

Contents
--------
- main: build config and execute the synthetic data-generation pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "Training").is_dir():
        sys.path.insert(0, str(parent))
        sys.path.insert(0, str(parent / "Training"))
        break

from Training.pipeline_runtime import ensure_project_imports

ensure_project_imports(Path(__file__))

from Training.data.python.pipeline.components.runner import run_data_pipeline
from Training.data.python.pipeline.config.data__config import build_config


def main() -> None:
    run_data_pipeline(build_config())


if __name__ == "__main__":
    main()
