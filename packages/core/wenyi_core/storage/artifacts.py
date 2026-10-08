"""File implementation of the backend-neutral JSON artifact port.

Keys are relative to a run; Review and subtitle services never open state paths.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import RLock
from typing import Any


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

    @staticmethod
    def _io_path(path: Path) -> str:
        """Return a string the Win32 API can open even past MAX_PATH.

        Precision drafts live under a 64-char source hash, so a deep state
        directory can push ``<draft>.json.tmp`` past the 260-character limit and
        Windows then reports a bare "No such file or directory". The ``\\\\?\\``
        prefix switches to the NT namespace, which has no such limit.
        """
        text = str(path)
        if os.name != "nt" or text.startswith("\\\\?\\"):
            return text
        return "\\\\?\\" + os.path.abspath(text)

    def read_artifact(self, key: str) -> Any | None:
        try:
            return json.loads(
                Path(self._io_path(self._artifact_path(key))).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return None

    def write_artifact(self, key: str, value: Any) -> None:
        path = self._artifact_path(key)
        os.makedirs(self._io_path(path.parent), exist_ok=True)
        with self._artifact_lock:
            tmp = path.with_name(path.name + ".tmp")
            Path(self._io_path(tmp)).write_text(
                json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(self._io_path(tmp), self._io_path(path))

    def delete_artifact(self, key: str) -> None:
        Path(self._io_path(self._artifact_path(key))).unlink(missing_ok=True)

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
        path = self._artifact_path(key)
        os.makedirs(self._io_path(path.parent), exist_ok=True)
        with self._artifact_lock, open(self._io_path(path), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def read_artifact_records(self, key: str) -> list[dict]:
        try:
            lines = (
                Path(self._io_path(self._artifact_path(key)))
                .read_text(encoding="utf-8")
                .splitlines()
            )
        except OSError:
            return []
        rows = []
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows
