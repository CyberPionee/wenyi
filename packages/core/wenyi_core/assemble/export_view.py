"""Disposable chapter view used only for export.
Deterministic postprocessing changes only in-memory copies returned by load_chapter and
never writes back to RunStore.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from types import MappingProxyType
from typing import Any, TypeAlias

from ..i18n.policy.models import ExportTextInput, ExportTextResult, PolicyContext, PolicyPlan
from ..i18n.policy.resolver import resolve_policy
from ..ingest.models import Chapter
from ..pipeline.runstore import ExportSnapshotStore, RunStore
from ..postprocess.export_text import (
    apply_operations,
    boundary_map,
    logical_ranges,
    normalize_chinese,
)
from ..storage.protocol import Storage
from .writer_common import _manifest_target_lang

TEXT_HANDLERS: MappingProxyType[str, Callable[[ExportTextInput], ExportTextResult]] = (
    MappingProxyType(
        {
            "punctuation.zh_cn": normalize_chinese,
        }
    )
)


def _target_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _boundary_map(before: str, after: str) -> list[int]:
    """Map original character boundaries to transformed boundaries for annotation and style
    offsets.
    """
    return list(boundary_map(before, after))


def _remap_metadata_offsets(
    metadata: object, before: str, after: str, mapping: tuple[int, ...] | None = None
) -> None:
    """Remap export-copy offsets only when placements match the formal translation."""
    if not isinstance(metadata, dict) or metadata.get("target_digest") != _target_digest(before):
        return
    mapping = mapping if mapping is not None else boundary_map(before, after)
    raw_placements = metadata.get("placements")
    if isinstance(raw_placements, list):
        for placement in raw_placements:
            if not isinstance(placement, dict):
                continue
            start = placement.get("target_start")
            end = placement.get("target_end")
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
            ):
                continue
            start = min(max(start, 0), len(before))
            end = min(max(end, start), len(before))
            placement["target_start"] = mapping[start]
            placement["target_end"] = mapping[end]
    metadata["target_digest"] = _target_digest(after)


class ExportViewStore(RunStore):
    """Overlay read-only export transformations on RunStore and delegate other capabilities."""

    # Mutators inherited from RunStore/FileArtifacts or forwarded via __getattr__.
    _READ_ONLY_MUTATORS = frozenset(
        {
            "begin_initialization",
            "finish_initialization",
            "stage_document",
            "set_chapter_status",
            "save_manifest",
            "save_chapter",
            "save_chapter_with_status",
            "save_context",
            "save_annotation_contexts",
            "save_analysis",
            "save_report",
            "save_usage",
            "prepare_usage_commit",
            "recover_usage",
            "record_timing",
            "log_event",
            "write_artifact",
            "delete_artifact",
            "append_artifact_record",
            "upsert_term",
            "resolve_term",
            "delete_term",
            "mark_conflicts_resolved",
            "_write_json",
        }
    )

    def __init__(
        self,
        store: Storage | ExportSnapshotStore,
        *,
        punctuation_normalize: bool,
        plan: PolicyPlan | None = None,
    ) -> None:
        super().__init__(store.run_dir, create=False)
        self._store = store
        self._chapters: dict[int, Chapter] = {}
        manifest = store.load_manifest()
        self.language_policy = plan or resolve_policy(
            PolicyContext(
                manifest.get("source_lang", "auto"),
                _manifest_target_lang(manifest),
                phase="export",
                format="epub",
                punctuation_normalize=punctuation_normalize,
                source_identity=manifest.get("source_sha256", ""),
            )
        )

    def load_manifest(self) -> dict:
        return self._store.load_manifest()

    def load_chapter(self, ci: int) -> Chapter:
        if ci in self._chapters:
            return self._chapters[ci].model_copy(deep=True)
        chapter = self._store.load_chapter(ci).model_copy(deep=True)
        operations = [op for op in self.language_policy.operations if op.point == "export.text"]
        if not operations:
            self._chapters[ci] = chapter.model_copy(deep=True)
            return chapter

        segments = chapter.text_segments
        continuations = tuple(segment.cont for segment in segments)
        request = ExportTextInput(
            tuple(segment.index for segment in segments),
            tuple(
                segment.target if segment.target != segment.source else None for segment in segments
            ),
            continuations,
            logical_ranges(continuations),
        )
        current, maps = apply_operations(
            request,
            TEXT_HANDLERS,
            tuple(operation.id for operation in operations),
        )
        for (position, end), mapping in zip(request.logical_ranges, maps):
            before = "".join(segment.target or "" for segment in segments[position:end])
            after = "".join(target or "" for target in current.targets[position:end])
            if any(segment.target == segment.source for segment in segments[position:end]):
                continue
            if before != after:
                metadata = segments[position].meta
                _remap_metadata_offsets(metadata.get("epub_annotations"), before, after, mapping)
                _remap_metadata_offsets(metadata.get("docx_styles"), before, after, mapping)

        for segment, target in zip(segments, current.targets):
            if segment.target is not None and segment.target != segment.source:
                segment.target = target
        self._chapters[ci] = chapter.model_copy(deep=True)
        return chapter

    def prepare(self) -> None:
        """Validate every transformation before opening or replacing an output file."""
        for row in self.load_manifest()["chapters"]:
            self.load_chapter(row["index"])

    def __getattr__(self, name: str) -> Any:
        if name in ExportViewStore._READ_ONLY_MUTATORS:
            raise RuntimeError(f"Export view is read-only; refusing to {name}")
        return getattr(self._store, name)

    def _deny_write(self, name: str) -> None:
        raise RuntimeError(f"Export view is read-only; refusing to {name}")

    def begin_initialization(self, source_hash: str) -> None:
        self._deny_write("begin_initialization")

    def finish_initialization(self) -> None:
        self._deny_write("finish_initialization")

    def stage_document(self, *args: Any, **kwargs: Any) -> dict:
        self._deny_write("stage_document")
        raise AssertionError("unreachable")

    def set_chapter_status(self, ci: int, status: str) -> None:
        self._deny_write("set_chapter_status")

    def save_manifest(self, manifest: dict) -> None:
        self._deny_write("save_manifest")

    def save_chapter(self, chapter) -> None:
        self._deny_write("save_chapter")

    def save_chapter_with_status(self, chapter, status) -> None:
        self._deny_write("save_chapter_with_status")

    def save_context(self, data: dict) -> None:
        self._deny_write("save_context")

    def save_annotation_contexts(self, data: dict) -> None:
        self._deny_write("save_annotation_contexts")

    def save_analysis(self, data: dict) -> None:
        self._deny_write("save_analysis")

    def save_report(self, data: dict) -> None:
        self._deny_write("save_report")

    def save_usage(self, data: dict) -> None:
        self._deny_write("save_usage")

    def prepare_usage_commit(self, ledgers: dict) -> None:
        self._deny_write("prepare_usage_commit")

    def recover_usage(self) -> None:
        self._deny_write("recover_usage")

    def record_timing(self, record: dict) -> dict:
        self._deny_write("record_timing")
        raise AssertionError("unreachable")

    def log_event(self, event: str, **data: Any) -> None:
        self._deny_write("log_event")

    def write_artifact(self, key: str, value: Any) -> None:
        self._deny_write("write_artifact")

    def delete_artifact(self, key: str) -> None:
        self._deny_write("delete_artifact")

    def append_artifact_record(self, key: str, record: dict) -> None:
        self._deny_write("append_artifact_record")

    def upsert_term(self, term, chapter: int | None = None) -> str:
        self._deny_write("upsert_term")
        raise AssertionError("unreachable")

    def resolve_term(self, source: str, target: str) -> bool:
        self._deny_write("resolve_term")
        raise AssertionError("unreachable")

    def delete_term(self, source: str) -> bool:
        self._deny_write("delete_term")
        raise AssertionError("unreachable")

    def mark_conflicts_resolved(self, source: str) -> None:
        self._deny_write("mark_conflicts_resolved")

    def _write_json(self, path: str, data) -> None:
        self._deny_write("_write_json")


AssembleStore: TypeAlias = Storage | ExportViewStore | ExportSnapshotStore
