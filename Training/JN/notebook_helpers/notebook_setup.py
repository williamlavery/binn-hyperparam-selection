"""Notebook bootstrap helpers shared across the case notebooks.

Contents
--------
- NotebookPaths: immutable container for key project directories.
- find_project_root: locate the repository root containing `Training`.
- bootstrap_notebook: register notebook import paths and return core paths.
- register_notebook_pickle_globals: register legacy classes for notebook unpickling.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NotebookPaths:
    project_root: Path
    training_root: Path
    notebook_root: Path
    data_root: Path
    binn_root: Path


def find_project_root(start_path: str | Path | None = None) -> Path:
    current = Path(start_path or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent

    while True:
        if (current / "Training").is_dir():
            return current
        parent = current.parent
        if parent == current:
            raise FileNotFoundError(
                f"Could not find a project root containing 'Training' above {start_path or Path.cwd()!r}."
            )
        current = parent


def bootstrap_notebook(start_path: str | Path | None = None) -> NotebookPaths:
    project_root = find_project_root(start_path)
    training_root = project_root / "Training"
    notebook_root = training_root / "JN"

    for path in (project_root, training_root, notebook_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    return NotebookPaths(
        project_root=project_root,
        training_root=training_root,
        notebook_root=notebook_root,
        data_root=training_root / "data" / "dataObj",
        binn_root=training_root / "binn",
    )


def register_notebook_pickle_globals() -> None:
    import __main__

    from Training.binn.python.Modules.Models import register_legacy_pickle_globals
    from Training.data.python.Modules.dataClass import Data, OriginalData

    register_legacy_pickle_globals()
    __main__.Data = Data
    __main__.OriginalData = OriginalData
