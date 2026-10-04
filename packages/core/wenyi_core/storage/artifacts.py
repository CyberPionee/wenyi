"""File implementation of the backend-neutral JSON artifact port.

Keys are relative to a run; Review and subtitle services never open state paths.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import RLock
from typing import Any

_MAX_PATH = 260


def _long_path(path: str) -> str:
    """Qualify a Windows path with the ``\\\\?\\`` prefix only when it exceeds MAX_PATH.

    Precision keys nest chapter, fingerprint and call identifiers, so an absolute path
    below a long temporary directory can pass 260 characters even though every parent
    directory exists. Shorter paths keep their original form so callers observe the same
    path identity; other platforms and already-qualified paths are returned unchanged.
    """
    if os.name != "nt" or path.startswith("\\\\?\\"):
        return path
    absolute = os.path.abspath(path)
    if len(absolute) < _MAX_PATH:
        return path
    if absolute.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute[2:]
    return "\\\\?\\" + absolute


class FileArtifacts:
    def __init__(self, run_dir: str):
        self._artifact_root = os.path.abspath(run_dir)
        self._artifact_lock = RLock()

    def _artifact_path(self, key: str) -> Path:
        # Validate the key structurally. resolve()/is_relative_to() is racy on Windows
        # while concurrent writers create directories under the run root, and resolve()
        # itself fails on paths beyond MAX_PATH, so containment is checked lexically.
        if not key or Path(key).is_absolute() or ".." in Path(key).parts:
            raise ValueError("Artifact key must remain within the run")
        path = Path(self._artifact_root, key)
        normalized = Path(os.path.normcase(str(path)))
        if not normalized.is_relative_to(Path(os.path.normcase(self._artifact_root))):
            raise ValueError("Artifact key must remain within the run")
        return path

    def read_artifact(self, key: str) -> Any | None:
        try:
            value = Path(_long_path(str(self._artifact_path(key)))).read_text(encoding="utf-8")
            return json.loads(value)
        except (OSError, json.JSONDecodeError):
            return None

    def write_artifact(self, key: str, value: Any) -> None:
        path = str(self._artifact_path(key))
        with self._artifact_lock:
            os.makedirs(_long_path(os.path.dirname(path) or "."), exist_ok=True)
            # Qualify the temporary name, which is longer than the final path.
            tmp = _long_path(path + ".tmp")
            with open(tmp, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(value, ensure_ascii=False, indent=2))
            os.replace(tmp, _long_path(path))

    def delete_artifact(self, key: str) -> None:
        Path(_long_path(str(self._artifact_path(key)))).unlink(missing_ok=True)

    def list_artifacts(self, prefix: str = "") -> list[str]:
        base = Path(self._artifact_root).resolve()
        parent = prefix.rpartition("/")[0]
        directory = base if not parent else self._artifact_path(parent)
        if not directory.is_dir():
            return []
        keys = (
            path.relative_to(base).as_posix() for path in directory.rglob("*") if path.is_file()
        )
        return sorted(key for key in keys if key.startswith(prefix))

    def append_artifact_record(self, key: str, record: dict) -> None:
        path = str(self._artifact_path(key))
        with self._artifact_lock:
            os.makedirs(_long_path(os.path.dirname(path)), exist_ok=True)
            with open(_long_path(path), "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def read_artifact_records(self, key: str) -> list[dict]:
        try:
            content = Path(_long_path(str(self._artifact_path(key)))).read_text(encoding="utf-8")
        except OSError:
            return []
        rows = []
        for line in content.splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows
