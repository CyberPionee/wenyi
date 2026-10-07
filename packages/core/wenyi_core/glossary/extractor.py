"""Extract glossary terms with an economical model and persist actual translations.
Extract proper names from source/target pairs after translation. GlossaryStore.upsert_term
records alternate translations as conflicts for human resolution.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, replace

from ..agents import prompts
from ..agents.base import Agent
from ..config import Config
from ..llm.base import LLMClient
from ..storage.protocol import Storage
from .auto_lock import can_auto_lock, should_write_auto_lock
from .injection import select_extraction_terms
from .store import (
    TYPE_PERSON,
    TYPE_TERM,
    GlossaryStore,
    GlossaryTerm,
    corpus_matcher,
    source_matches_text,
)


def _text(value: object, default: str = "") -> str:
    """Normalize scalar model fields to strings."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return default


@dataclass(frozen=True)
class TranslatedSegmentEvidence:
    """A translated paragraph and its book position for tracing a new term's first translation."""

    chapter: int
    segment: int
    source: str
    target: str


class GlossaryExtractor(Agent):
    def __init__(self, client: LLMClient, config: Config):
        super().__init__(client, config)

    def _recurring_sources(self, source_corpus: str | None, terms: list[GlossaryTerm]) -> set[str]:
        """Return the sources occurring at least twice in the corpus.

        The corpus matcher is shared across callers and memoizes the count per term, so a run
        pays for each term once instead of re-scanning the book for it on every chapter. The
        recurrence gate itself is min_occurrences=2.
        """
        if not source_corpus:
            return set()
        matcher = corpus_matcher(source_corpus)
        return {term.source for term in terms if matcher.occurrence_count(term) >= 2}

    def extract(
        self, source_text: str, target_text: str, existing: list[GlossaryTerm]
    ) -> list[GlossaryTerm]:
        """Extract valid terms from source/target pairs and normalize model field types."""
        system = self.render("glossary_extractor_system", src=self.src, tgt=self.tgt)
        user = self.render(
            "glossary_extractor_user",
            src=self.src,
            tgt=self.tgt,
            glossary=prompts.render_glossary(
                existing,
                include_note=False,
                max_note_chars=0,
                source_lang=self.src,
            ),
            source=source_text,
            target=target_text,
        )
        raw = self._ask_json(system, user, operation="glossary.extract", key="terms", default=[])
        terms: list[GlossaryTerm] = []
        for d in self.dict_items(raw, operation="glossary.extract", field="terms"):
            source = _text(d.get("source"))
            target = _text(d.get("target"))
            if not source or not target:
                continue
            raw_aliases = d.get("aliases")
            aliases = raw_aliases if isinstance(raw_aliases, list) else []
            gender = _text(d.get("gender"))
            terms.append(
                GlossaryTerm(
                    source=source,
                    target=target,
                    reading=_text(d.get("reading")) if self.src == "ja" else "",
                    type=_text(d.get("type"), TYPE_TERM),
                    gender=gender,
                    aliases=[alias for a in aliases if (alias := _text(a))],
                    note=_text(d.get("note")),
                )
            )
        return terms

    @staticmethod
    def _drop_unmatched_sources(
        terms: list[GlossaryTerm], source_text: str
    ) -> tuple[list[GlossaryTerm], int]:
        """Keep only candidates whose source actually occurs in the text they came from.

        A glossary entry exists to match text, so a candidate whose source appears nowhere
        cannot do anything: it never injects, it never matches a segment, and it misreports
        the chapter the term first appeared in. One run stored 22 such rows, including
        Chinese spellings of Japanese sources, mixed-script corruptions, misspellings and
        invented sentence fragments. The model paraphrasing its own source is the cause, not
        a legitimate alias, so the source text decides instead of trusting the candidate.
        """
        if not source_text.strip():
            return terms, 0
        kept = [
            term
            for term in terms
            if any(
                source_matches_text(key, source_text)
                for key in [term.source, *term.aliases]
                if (key or "").strip()
            )
        ]
        return kept, len(terms) - len(kept)

    @staticmethod
    def _first_occurrences(
        terms: list[GlossaryTerm],
        store: Storage | GlossaryStore,
        history: Iterable[TranslatedSegmentEvidence],
        before: tuple[int, int],
    ) -> dict[str, TranslatedSegmentEvidence]:
        """Find the first translated paragraph before the given position for terms not yet
        stored.
        """
        pending = {term.source for term in terms if store.get_term(term.source) is None}
        if not pending:
            return {}

        first: dict[str, TranslatedSegmentEvidence] = {}
        ordered_history = sorted(history, key=lambda item: (item.chapter, item.segment))
        for evidence in ordered_history:
            if (evidence.chapter, evidence.segment) >= before:
                continue
            for source in pending:
                if source in first:
                    continue
                if source_matches_text(source, evidence.source):
                    first[source] = evidence
            if len(first) == len(pending):
                break
        return first

    def _align_with_first_occurrences(
        self,
        terms: list[GlossaryTerm],
        occurrences: dict[str, TranslatedSegmentEvidence],
    ) -> tuple[list[GlossaryTerm], int, int]:
        """Align candidates with their first translations; defer terms whose historical mapping
        is uncertain.
        """
        if not occurrences:
            return terms, 0, 0

        candidates = []
        for term in terms:
            evidence = occurrences.get(term.source)
            if evidence is None:
                continue
            candidates.append(
                {
                    "source": term.source,
                    "proposed_target": term.target,
                    "first_occurrence": {
                        "chapter": evidence.chapter,
                        "segment": evidence.segment,
                        "source": evidence.source,
                        "target": evidence.target,
                    },
                }
            )

        system = self.render("glossary_history_system", src=self.src, tgt=self.tgt)
        user = self.render(
            "glossary_history_user",
            src=self.src,
            tgt=self.tgt,
            candidates_json=json.dumps(candidates, ensure_ascii=False, indent=2),
        )
        raw = self._ask_json(
            system, user, operation="glossary.align_history", key="terms", default=[]
        )
        resolved = {
            source: target
            for item in self.dict_items(raw, operation="glossary.align_history", field="terms")
            if (source := _text(item.get("source"))) in occurrences
            and (target := _text(item.get("target")))
        }

        aligned: list[GlossaryTerm] = []
        unresolved = 0
        for term in terms:
            if term.source not in occurrences:
                aligned.append(term)
                continue
            target = resolved.get(term.source)
            if not target:
                unresolved += 1
                continue
            aligned.append(replace(term, target=target))
        return aligned, len(resolved), unresolved

    def extract_and_store(
        self,
        store: Storage | GlossaryStore,
        source_text: str,
        target_text: str,
        chapter: int,
        *,
        history: Iterable[TranslatedSegmentEvidence] = (),
        before: tuple[int, int] | None = None,
        source_corpus: str | None = None,
        on_auto_lock=None,
    ) -> dict[str, int]:
        """Extract and store terms, preferring the translation at their first historical
        occurrence.
        history contains translated evidence only. If a new term appears before the supplied
        position, align target against its first source/target pair. Defer uncertain
        mappings instead of locking a later candidate into the glossary and contaminating
        subsequent text.
        With source_corpus, inject only existing terms occurring at least twice in the
        source. Low-frequency terms remain stored but do not repeatedly consume extraction
        context.
        When auto-lock gates pass (history aligned, recurring, no open conflict, whitelist
        type), lock the mapping without overwriting an established non-empty target and
        notify ``on_auto_lock(source, target)``.
        """
        all_existing = store.all_terms()
        open_conflicts = {
            str(row.get("source") or "")
            for row in (store.open_conflicts() if hasattr(store, "open_conflicts") else [])
        }
        pipeline = self.config.pipeline
        recent_n = max(0, int(getattr(pipeline, "glossary_extract_recent_max", 20)))
        recent_sources = [t.source for t in all_existing[-recent_n:]] if recent_n else []
        existing = select_extraction_terms(
            all_existing,
            batch_text=f"{source_text}\n{target_text}",
            source_corpus=source_corpus or "",
            budget_chars=int(getattr(pipeline, "glossary_extract_budget_chars", 4000)),
            core_max=int(getattr(pipeline, "glossary_extract_core_max", 12)),
            recent_max=recent_n,
            min_terms=int(getattr(pipeline, "glossary_extract_min_terms", 5)),
            mode=str(getattr(pipeline, "glossary_extract_inject", "smart")),
            open_conflict_sources=open_conflicts,
            recent_sources=recent_sources,
            always_types=tuple(getattr(pipeline, "glossary_always_types", (TYPE_PERSON,)))
            or (TYPE_PERSON,),
            core_min_occurrences=int(getattr(pipeline, "glossary_always_min_occurrences", 3)),
        )
        terms = self.extract(source_text, target_text, existing)
        terms, unmatched = self._drop_unmatched_sources(terms, source_text)
        occurrences = (
            self._first_occurrences(terms, store, history, before) if before is not None else {}
        )
        terms, aligned, unresolved = self._align_with_first_occurrences(terms, occurrences)
        summary = {
            "inserted": 0,
            "conflict": 0,
            "unchanged": 0,
            "history_matched": len(occurrences),
            "history_aligned": aligned,
            "history_unresolved": unresolved,
            "auto_locked": 0,
            "injected_terms": len(existing),
            "source_not_in_text": unmatched,
        }
        recurring = self._recurring_sources(source_corpus, terms)
        for t in terms:
            evidence = occurrences.get(t.source)
            t.first_chapter = evidence.chapter if evidence is not None else chapter
            prior = store.get_term(t.source) if hasattr(store, "get_term") else None
            result = store.upsert_term(t, chapter=chapter)
            summary[result] = summary.get(result, 0) + 1
            history_aligned = evidence is not None and bool((t.target or "").strip())
            if not source_corpus:
                # Without a corpus the recurrence gate cannot be verified; do not auto-lock.
                continue
            occurrences_n = 2 if t.source in recurring else 1
            # Only pre-existing open conflicts block auto-lock. A same-run "conflict"
            # result still allows a restricted resolve that keeps the established target.
            has_open_conflict = t.source in open_conflicts
            if not can_auto_lock(
                t,
                history_aligned=history_aligned,
                occurrences=occurrences_n,
                has_open_conflict=has_open_conflict,
            ):
                continue
            if result == "conflict":
                # Restricted resolve: keep the established target; never overwrite it. The port
                # owns this call, so every backend settles the same conflict. Guarding it on a
                # concrete store type silently left every HTTP and SQLite run with an open
                # conflict that the acceptance gate then refused to export.
                store.mark_conflicts_resolved(t.source)
                locked_target = (prior.target if prior is not None else t.target) or t.target
            elif not should_write_auto_lock(prior, t) and result not in {
                "inserted",
                "updated",
                "unchanged",
            }:
                continue
            else:
                locked_target = t.target
            summary["auto_locked"] = summary.get("auto_locked", 0) + 1
            if on_auto_lock is not None:
                on_auto_lock(t.source, locked_target or t.target)
        return summary

    def finalize_chapter_glossary(
        self,
        store: Storage | GlossaryStore,
        chapter: int,
        *,
        history: Iterable[TranslatedSegmentEvidence] = (),
        before: tuple[int, int] | None = None,
        source_corpus: str | None = None,
        on_auto_lock=None,
    ) -> dict[str, int]:
        """Local chapter close-out without another model extraction call.

        Batch extraction already ran per batch. This only fills empty targets from
        historical evidence (first matching source, no LLM) and applies auto-lock
        gates to existing terms so chapter end does not re-send the full chapter
        through the LLM.
        """
        summary = {
            "inserted": 0,
            "conflict": 0,
            "unchanged": 0,
            "updated": 0,
            "history_filled": 0,
            "auto_locked": 0,
            "llm_calls": 0,
            "mode": "local_finalize",
        }
        all_existing = store.all_terms()
        if not all_existing:
            return summary
        ordered_history = sorted(history, key=lambda item: (item.chapter, item.segment))
        recurring = self._recurring_sources(source_corpus, all_existing)
        open_conflicts = {
            str(row.get("source") or "")
            for row in (store.open_conflicts() if hasattr(store, "open_conflicts") else [])
        }
        for t in all_existing:
            prior = store.get_term(t.source) if hasattr(store, "get_term") else t
            target = (prior.target or t.target or "").strip()
            # This pass reports a lock only for a mapping it supplied itself. Every term that
            # already carried a target is a mapping an earlier pass established, so re-reporting
            # it here emitted one auto-lock per chapter per established term.
            already_established = bool((t.target or "").strip())
            if not target:
                # Local fill: first historical source match provides the mapping text.
                for evidence in ordered_history:
                    if before is not None and (evidence.chapter, evidence.segment) >= before:
                        continue
                    if source_matches_text(t.source, evidence.source):
                        target = (evidence.target or "").strip()
                        if target:
                            filled = replace(t, target=target)
                            result = store.upsert_term(filled, chapter=chapter)
                            summary[result] = summary.get(result, 0) + 1
                            summary["history_filled"] += 1
                            t = filled
                            prior = filled
                        break
            if not target:
                summary["unchanged"] += 1
                continue
            if not source_corpus:
                summary["unchanged"] += 1
                continue
            occurrences_n = 2 if t.source in recurring else 1
            has_open_conflict = t.source in open_conflicts
            if not already_established and can_auto_lock(
                t,
                history_aligned=True,
                occurrences=occurrences_n,
                has_open_conflict=has_open_conflict,
            ):
                summary["auto_locked"] = summary.get("auto_locked", 0) + 1
                if on_auto_lock is not None:
                    on_auto_lock(t.source, target)
        return summary
