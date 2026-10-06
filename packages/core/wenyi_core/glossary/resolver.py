"""CLI helpers for human resolution of glossary conflicts.
GlossaryStore.upsert_term detects conflicts automatically. These wrappers choose the final
translation and mark related conflicts resolved.
"""

from __future__ import annotations

from typing import Any

from ..storage.protocol import Storage
from .store import GlossaryStore


def resolve(store: GlossaryStore, source: str, target: str) -> bool:
    """Resolve a source term's final translation and clear conflicts; report whether the term
    exists.
    """
    if not store.resolve_term(source, target):
        return False
    store.mark_conflicts_resolved(source)
    return True


def keep_current_terms(store: Storage) -> list[dict[str, Any]]:
    """Settle every open conflict in favour of each term's established target.

    Returns one record per settled term: its source, the target that now stands, the rejected
    proposals, and the settled term. A term an operator locked is already excluded from the
    open list, so this only closes disagreements nobody has decided, and each settlement locks
    the term so the same proposal cannot reopen a conflict later. A conflict whose term or
    target is gone stays open for a human to inspect rather than being discarded.

    Rewriting translations belongs to the caller: it touches chapters, so it has to run after
    the glossary lock is released.
    """
    proposals: dict[str, list[str]] = {}
    for conflict in store.open_conflicts():
        source = str(conflict.get("source") or "")
        if source:
            proposals.setdefault(source, []).append(str(conflict.get("proposed_target") or ""))

    settled: list[dict[str, Any]] = []
    for source, proposed in proposals.items():
        term = store.get_term(source)
        if term is None or not (term.target or "").strip():
            continue
        if not store.resolve_term(term.source, term.target):
            continue
        store.mark_conflicts_resolved(term.source)
        settled.append(
            {
                "source": term.source,
                "target": term.target,
                "rejected": [
                    value
                    for value in dict.fromkeys(proposed)
                    if (value or "").strip() and value != term.target
                ],
                "term": term,
            }
        )
    return settled
