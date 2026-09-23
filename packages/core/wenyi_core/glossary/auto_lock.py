"""Pure helpers for constrained automatic glossary locking (B2').

Auto-lock never overwrites an established non-empty target. It only fills a
missing mapping when history alignment succeeded, the source recurred at least
twice book-wide, there is no open conflict, and the type is on the whitelist.
"""

from __future__ import annotations

from collections.abc import Sequence

from .store import TYPE_PERSON, TYPE_TERM, GlossaryTerm

AUTO_LOCK_TYPES = frozenset(
    {
        TYPE_PERSON,
        "place",
        "organization",
        TYPE_TERM,
        "technique",
        "item",
        "setting",
    }
)


def can_auto_lock(
    proposed: GlossaryTerm,
    *,
    history_aligned: bool,
    occurrences: int,
    has_open_conflict: bool,
    always_types: Sequence[str] | None = None,
    min_occurrences: int = 2,
) -> bool:
    """Return whether a proposed term may be auto-locked under the four gates."""
    types = frozenset(always_types) if always_types is not None else AUTO_LOCK_TYPES
    return (
        history_aligned
        and occurrences >= min_occurrences
        and not has_open_conflict
        and proposed.type in types
        and bool((proposed.target or "").strip())
    )


def should_write_auto_lock(existing: GlossaryTerm | None, proposed: GlossaryTerm) -> bool:
    """Never overwrite an established mapping; only fill a missing or empty target."""
    if not (proposed.target or "").strip():
        return False
    if existing is None:
        return True
    if (existing.target or "").strip() and existing.status == "ok":
        return False
    return True
