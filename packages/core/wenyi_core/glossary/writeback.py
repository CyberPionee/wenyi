"""Source-anchored rewrite of formal targets when a glossary target changes.

Rewriting is deliberately conservative: a segment is only rewritten when its source
mentions the term (or one of its aliases) and its target still carries the previous
mapping. Segments that do not match are left alone instead of guessing.
"""

from __future__ import annotations

from typing import Any

from ..ingest.models import Chapter
from ..storage.protocol import Storage
from .store import GlossaryTerm, source_matches_text, term_match_sources


def rewrite_target_text(target: str, old_target: str, new_target: str) -> str | None:
    """Return the rewritten target, or None when nothing changed."""
    if not (old_target or "").strip() or not (new_target or "").strip():
        return None
    if old_target == new_target:
        return None
    if old_target not in (target or ""):
        return None
    return target.replace(old_target, new_target)


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
    summary: dict[str, Any] = {
        "source": term_source,
        "old_target": old_target,
        "new_target": new_target,
        "segments_replaced": 0,
        "chapters_touched": 0,
        "matched_segments": 0,
    }
    if not (old_target or "").strip() or not (new_target or "").strip() or old_target == new_target:
        return summary
    # Match the original source spelling plus any aliases of the edited term.
    matcher = GlossaryTerm(
        source=term_source,
        target=new_target,
        type=term.type,
        aliases=list(term.aliases or []),
    )
    keys = [key for key in term_match_sources(matcher) if (key or "").strip()]
    if not keys:
        return summary

    manifest = store.load_manifest()
    for row in manifest.get("chapters", []):
        index = row.get("index")
        if not isinstance(index, int):
            continue
        chapter: Chapter = store.load_chapter(index)
        dirty = False
        for segment in chapter.text_segments:
            target = segment.target or ""
            if not target.strip() or old_target not in target:
                continue
            if not any(source_matches_text(key, segment.source or "") for key in keys):
                continue
            summary["matched_segments"] += 1
            rewritten = rewrite_target_text(target, old_target, new_target)
            if rewritten is None or rewritten == target:
                continue
            segment.target = rewritten
            dirty = True
            summary["segments_replaced"] += 1
        if dirty:
            store.save_chapter(chapter)
            summary["chapters_touched"] += 1
    return summary
