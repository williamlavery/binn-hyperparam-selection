from __future__ import annotations

import os
import sys
from pathlib import Path


def find_project_root(start_path: str | os.PathLike[str] | None = None) -> Path:
    current = Path(start_path or os.getcwd()).resolve()
    if current.is_file():
        current = current.parent

    while True:
        if (current / "Training").is_dir():
            return current
        parent = current.parent
        if parent == current:
            raise FileNotFoundError(
                f"Could not find a project root containing 'Training' above "
                f"{start_path or os.getcwd()!r}."
            )
        current = parent


def ensure_project_imports(start_path: str | os.PathLike[str] | None = None) -> Path:
    project_root = find_project_root(start_path)
    project_parent = str(project_root.parent)
    training_root = str(project_root / "Training")
    for path in (project_parent, training_root):
        if path not in sys.path:
            sys.path.insert(0, path)
    return project_root


def print_nested(values: dict, indent: int = 2, omit_keys: set[str] | None = None) -> None:
    omit_keys = omit_keys or set()
    for key, value in values.items():
        if key in omit_keys:
            continue
        if isinstance(value, dict):
            print(" " * indent + f"{key}:")
            print_nested(value, indent + 4, omit_keys)
        else:
            print(" " * indent + f"{key:<27} = {value}")
