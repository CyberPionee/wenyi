"""Universal flexible injection of existing terms into glossary extraction prompts.

One scoring + budget algorithm adapts to short and long books. Short books usually
fit the full glossary under the budget; long books keep only high-relevance terms.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .store import (
    TYPE_APPELLATION,
    TYPE_FIXED_EXPR,
    TYPE_HONORIFIC,
    TYPE_PERSON,
    TYPE_SPEECH,
    GlossaryOccurrenceMatcher,
    GlossaryTerm,
    source_matches_text,
    term_match_sources,
)

_ENTITY_TYPES = frozenset({"person", "place", "organization"})
_CORE_TYPES = frozenset({TYPE_PERSON, "place", "organization"})
_SOURCE_ONLY_TYPES = frozenset({TYPE_APPELLATION, TYPE_HONORIFIC, TYPE_SPEECH, TYPE_FIXED_EXPR})


@dataclass(frozen=True)
class InjectionCandidate:
    term: GlossaryTerm
    score: int
    hit: bool
    core: bool
    recent: bool
    forced: bool


def _batch_hit(term: GlossaryTerm, batch_text: str) -> bool:
    if not batch_text:
        return False
    for key in term_match_sources(term):
        if key and source_matches_text(key, batch_text):
            return True
    return False


def score_term(
    term: GlossaryTerm,
    *,
    batch_text: str = "",
    core: bool = False,
    recent: bool = False,
    frequency: int = 0,
    forced: bool = False,
) -> InjectionCandidate:
    """Score one term for extraction-time injection."""
    hit = _batch_hit(term, batch_text)
    score = 0
    if hit:
        score += 100
    if core:
        score += 60
    if recent:
        score += 40
    score += max(0, min(int(frequency or 0), 20))
    if term.type in _ENTITY_TYPES:
        score += 15
    elif term.type in _SOURCE_ONLY_TYPES and not hit:
        score += 0
    return InjectionCandidate(
        term=term,
        score=score,
        hit=hit,
        core=core,
        recent=recent,
        forced=forced,
    )


def select_extraction_terms(
    terms: Sequence[GlossaryTerm],
    *,
    batch_text: str = "",
    source_corpus: str = "",
    budget_chars: int = 4000,
    core_max: int = 12,
    recent_max: int = 20,
    min_terms: int = 5,
    mode: str = "smart",
    open_conflict_sources: Iterable[str] = (),
    recent_sources: Iterable[str] = (),
    always_types: Sequence[str] = (TYPE_PERSON,),
    core_min_occurrences: int = 3,
) -> list[GlossaryTerm]:
    """Pick existing terms to inject into an extraction prompt.

    Modes:
    - smart: score + budget fill with fallbacks
    - all: every term (debug)
    - hit_only: batch hits and forced conflicts only
    """
    if not terms:
        return []
    if mode == "all":
        return list(terms)

    forced_set = {str(s) for s in open_conflict_sources if str(s).strip()}
    recent_set = {str(s) for s in recent_sources if str(s).strip()}
    matcher = GlossaryOccurrenceMatcher(source_corpus) if source_corpus else None
    always = frozenset(always_types or (TYPE_PERSON,))

    def is_core(term: GlossaryTerm) -> bool:
        if term.type not in always and term.type not in _CORE_TYPES:
            return False
        if matcher is None:
            return term.type in always
        return bool(matcher.recurring_terms([term], min_occurrences=core_min_occurrences))

    def frequency(term: GlossaryTerm) -> int:
        if matcher is None:
            return 0
        matched = matcher.recurring_terms([term], min_occurrences=1)
        return 20 if matched else 1

    candidates = [
        score_term(
            term,
            batch_text=batch_text,
            core=is_core(term),
            recent=term.source in recent_set,
            frequency=frequency(term),
            forced=term.source in forced_set,
        )
        for term in terms
    ]
    if mode == "hit_only":
        picked = [c for c in candidates if c.hit or c.forced]
        picked.sort(key=lambda c: (not c.forced, -c.score, c.term.source))
        return [c.term for c in picked]

    # smart
    candidates.sort(key=lambda c: (not c.forced, -c.score, c.term.source))
    selected: list[InjectionCandidate] = []
    selected_keys: set[str] = set()
    budget_left = max(0, int(budget_chars))

    def try_add(candidate: InjectionCandidate) -> bool:
        nonlocal budget_left
        if candidate.term.source in selected_keys:
            return True
        line = _render_line(candidate.term)
        cost = len(line) + 1
        if selected and cost > budget_left:
            return False
        selected.append(candidate)
        selected_keys.add(candidate.term.source)
        budget_left = max(0, budget_left - cost)
        return True

    for candidate in candidates:
        if candidate.forced:
            try_add(candidate)

    core_count = 0
    recent_count = 0
    for candidate in candidates:
        if candidate.term.source in selected_keys:
            continue
        if candidate.hit:
            if not try_add(candidate):
                break
            continue
        if candidate.core and core_count < core_max:
            if try_add(candidate):
                core_count += 1
                continue
        if candidate.recent and recent_count < recent_max:
            if try_add(candidate):
                recent_count += 1
                continue
        if candidate.core or candidate.recent:
            continue
        # Remaining score-ordered fill while budget allows.
        if not try_add(candidate):
            break

    if len(selected) < max(0, min_terms):
        for candidate in candidates:
            if len(selected) >= max(0, min_terms):
                break
            try_add(candidate)

    return [c.term for c in selected]


def _render_line(term: GlossaryTerm) -> str:
    """Mirror prompts.render_glossary single-line size without importing UI helpers."""
    extra = []
    if term.gender:
        extra.append(term.gender)
    if term.reading:
        extra.append(f"Pronunciation: {term.reading}")
    tag = f"({term.type}{(', ' + ', '.join(extra)) if extra else ''})"
    alias = f" [Aliases:  {', '.join(term.aliases)}]" if term.aliases else ""
    return f"- {term.source} → {term.target}{tag}{alias}"
