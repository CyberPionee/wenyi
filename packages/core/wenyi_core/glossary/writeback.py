"""Source-anchored rewrite of formal targets when a glossary target changes.

Rewriting is deliberately conservative: a segment is only rewritten when its source
mentions the term (or one of its aliases) and its target still carries a previous
mapping. Segments that do not match are left alone instead of guessing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..ingest.models import Chapter
from ..storage.protocol import Storage
from .store import GlossaryTerm, source_matches_text, term_match_sources


@dataclass(frozen=True)
class TargetRewrite:
    """One formal paragraph that has to take the settled rendering of a term."""

    chapter: int
    text_index: int
    segment_index: int
    before: str
    after: str


def rewrite_target_text(target: str, old_target: str, new_target: str) -> str | None:
    """Return the rewritten target, or None when nothing changed."""
    if not (old_target or "").strip() or not (new_target or "").strip():
        return None
    if old_target == new_target:
        return None
    if old_target not in (target or ""):
        return None
    return target.replace(old_target, new_target)


def plan_conflict_writeback(
    chapters: Sequence[Chapter],
    *,
    term_source: str,
    term: GlossaryTerm,
    rejected_targets: Sequence[str],
    chosen_target: str,
) -> list[TargetRewrite]:
    """Return every formal paragraph that still carries a rejected rendering for this term.

    Settling a terminology conflict decides which rendering the whole book uses, so the plan
    starts from every rejected candidate rather than only the previously established one:
    choosing the established target still has to fix the passages that were translated with
    the rejected proposal. Planning stays separate from writing so a caller can record the
    decision before any formal target changes.
    """
    losing = [
        target
        for target in dict.fromkeys(rejected_targets)
        if (target or "").strip() and target != chosen_target
    ]
    if not (chosen_target or "").strip() or not losing:
        return []
    # Match the original source spelling plus any aliases of the settled term.
    matcher = GlossaryTerm(
        source=term_source,
        target=chosen_target,
        type=term.type,
        aliases=list(term.aliases or []),
    )
    keys = [key for key in term_match_sources(matcher) if (key or "").strip()]
    if not keys:
        return []

    plan: list[TargetRewrite] = []
    for chapter in chapters:
        for text_index, segment in enumerate(chapter.text_segments):
            target = segment.target or ""
            if not target.strip():
                continue
            replaced = target
            for rejected in losing:
                rewritten = rewrite_target_text(replaced, rejected, chosen_target)
                if rewritten is not None:
                    replaced = rewritten
            if replaced == target:
                continue
            if not any(source_matches_text(key, segment.source or "") for key in keys):
                continue
            plan.append(
                TargetRewrite(
                    chapter=chapter.index,
                    text_index=text_index,
                    segment_index=segment.index,
                    before=target,
                    after=replaced,
                )
            )
    return plan


def apply_conflict_writeback(
    store: Storage,
    *,
    term_source: str,
    term: GlossaryTerm,
    rejected_targets: Sequence[str],
    chosen_target: str,
) -> dict[str, Any]:
    """Rewrite every formal segment that still carries a rejected rendering for this term.

    An operator's edit or conflict decision uses this path: the decision is already durable in
    the glossary, so the plan is applied in one pass and the summary reports what moved.
    """
    return apply_target_rewrites(
        store,
        plan_conflict_writeback(
            load_chapters(store),
            term_source=term_source,
            term=term,
            rejected_targets=rejected_targets,
            chosen_target=chosen_target,
        ),
        term_source=term_source,
        rejected_targets=rejected_targets,
        chosen_target=chosen_target,
    )


def apply_target_rewrites(
    store: Storage,
    rewrites: Sequence[TargetRewrite],
    *,
    term_source: str,
    rejected_targets: Sequence[str],
    chosen_target: str,
) -> dict[str, Any]:
    """Write planned rewrites back to the chapters they belong to."""
    summary: dict[str, Any] = {
        "source": term_source,
        "old_targets": list(rejected_targets),
        "new_target": chosen_target,
        "segments_replaced": len(rewrites),
        "matched_segments": len(rewrites),
        "chapters_touched": len({rewrite.chapter for rewrite in rewrites}),
    }
    by_chapter: dict[int, list[TargetRewrite]] = {}
    for rewrite in rewrites:
        by_chapter.setdefault(rewrite.chapter, []).append(rewrite)
    for chapter_index, chapter_rewrites in sorted(by_chapter.items()):
        chapter = store.load_chapter(chapter_index)
        text_segments = chapter.text_segments
        for rewrite in chapter_rewrites:
            if not 0 <= rewrite.text_index < len(text_segments):
                continue
            text_segments[rewrite.text_index].target = rewrite.after
        store.save_chapter(chapter)
    return summary


def apply_term_writeback(
    store: Storage,
    *,
    term_source: str,
    term: GlossaryTerm,
    old_target: str,
    new_target: str,
) -> dict[str, Any]:
    """Rewrite every formal segment that still uses the old mapping for this term.

    ``term_source`` is the source spelling that appears in the text; it may differ from
    ``term.source`` when the edit renamed the source entry. Only segments whose source
    mentions the term are touched, so an unrelated passage that happens to contain the
    old target string is never rewritten.
    """
    return apply_conflict_writeback(
        store,
        term_source=term_source,
        term=term,
        rejected_targets=[old_target],
        chosen_target=new_target,
    )


def load_chapters(store: Storage) -> list[Chapter]:
    """Load every chapter the manifest lists, in manifest order."""
    manifest = store.load_manifest()
    chapters: list[Chapter] = []
    for row in manifest.get("chapters", []):
        index = row.get("index")
        if isinstance(index, int):
            chapters.append(store.load_chapter(index))
    return chapters
