"""Judge source terms that share one target and split the ones that must read apart.

Distinct sources mapped to a single rendering may be one entity, distinct entities that
happen to share a name, or one entity whose source-side distinction the target no longer
preserves. After each chapter's extraction this service finds those groups, has each judged
against sampled passages, and applies the verdict: a source judged distinct moves to its own
wording in the glossary and every passage that mentions it is rewritten to match. Groups the
passages cannot decide wait for a later pass with wider context.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..agents.glossary_disambiguator import GlossaryTargetDisambiguator
from ..glossary.store import GlossaryTerm, source_matches_text, term_match_sources
from ..glossary.writeback import (
    apply_target_rewrites,
    load_chapters,
    plan_conflict_writeback,
)
from ..review.evidence import BookEvidenceIndex
from ..review.run_store import ReviewRunStore
from ..storage.protocol import Storage
from .review_checkpoint import ReviewTraceStore

if TYPE_CHECKING:
    from .runtime import PipelineRuntime

ProgressFn = Callable[[int, int, str], None]

_INDEX = "glossary-disambiguation/judgements.json"
_SAMPLED_OCCURRENCES = 12
_VERDICT_TAG = "[target-disambiguation]"


def _collision_id(target: str, sources: list[str]) -> str:
    """Give each collision a stable id without leaking term characters into artifacts."""
    digest = hashlib.sha256(f"{target}|{','.join(sorted(sources))}".encode("utf-8")).hexdigest()[
        :16
    ]
    return f"collision-{digest}"


def _empty_summary() -> dict[str, Any]:
    return {"collisions": 0, "judged": 0, "unresolved": 0, "renderings_applied": 0, "reason": ""}


def _verdict_note(
    judgement: dict[str, Any], others: list[str], *, wording: str, changed: bool
) -> str:
    """One short line for the term note; the full reason stays in the event and artifact."""
    listed = ", ".join(others)
    if changed:
        return f"{_VERDICT_TAG} Distinguished from {listed}; now rendered as {wording}."
    if judgement.get("same_entity") is True:
        return f"{_VERDICT_TAG} Same entity as {listed}; unified rendering kept."
    return f"{_VERDICT_TAG} Distinct from {listed}; shared rendering kept."


class GlossaryDisambiguationService:
    """Apply same-target collision verdicts, splitting the renderings that must read apart."""

    def __init__(self, runtime: PipelineRuntime):
        self._runtime = runtime

    def run(self, store: Storage, *, progress: ProgressFn | None = None) -> dict[str, Any]:
        """Judge every unjudged collision and apply its renderings.

        Called after each chapter's extraction so a split lands before the next chapter is
        translated, and again before review as a whole-book backstop.
        """
        summary = _empty_summary()
        if not self._runtime.config.pipeline.glossary_target_disambiguation:
            summary["reason"] = "disabled"
            return summary
        resumed = self.resume_pending(store, progress=progress)
        if resumed is not None:
            return resumed
        groups = self._collision_groups(store)
        if not groups:
            return summary
        summary["collisions"] = len(groups)
        debug = ReviewRunStore(store.run_dir, storage=store, kind="glossary-disambiguation")
        debug.start(
            reviewed_content_digest="glossary-disambiguation",
            metadata={"kind": "glossary_disambiguation", "collisions": len(groups)},
        )
        evidence = BookEvidenceIndex(
            load_chapters(store),
            list(store.all_terms()),
            store.load_analysis() or {},
        )
        judge = GlossaryTargetDisambiguator(
            self._runtime.client,
            self._runtime.config,
            evidence,
            ReviewTraceStore(debug),
        )
        judgements: list[dict[str, Any]] = []
        unresolved: list[dict[str, Any]] = []
        for done, group in enumerate(groups, start=1):
            if progress:
                progress(done - 1, len(groups), "Judging shared-target term collisions")
            outcome = judge.judge(group)
            if outcome["status"] == "judged":
                judgements.append({**outcome, "sources": [s["source"] for s in group["sources"]]})
            else:
                unresolved.append(
                    {
                        "collision_id": outcome["collision_id"],
                        "target": outcome["target"],
                        "reason": outcome["reason"],
                    }
                )
            if progress:
                progress(done, len(groups), "Judging shared-target term collisions")
        summary["judged"] = len(judgements)
        summary["unresolved"] = len(unresolved)
        if not judgements:
            store.write_artifact(
                _INDEX, {"status": "completed", "judgements": judgements, "unresolved": unresolved}
            )
            return summary
        # Persist the verdicts before touching notes so an interruption resumes from this index.
        store.write_artifact(
            _INDEX, {"status": "judged", "judgements": judgements, "unresolved": unresolved}
        )
        summary["renderings_applied"] = self._apply(store, judgements, progress=progress)
        store.write_artifact(
            _INDEX, {"status": "completed", "judgements": judgements, "unresolved": unresolved}
        )
        store.log_event("glossary_disambiguation_finished", **summary)
        return summary

    def resume_pending(
        self, store: Storage, *, progress: ProgressFn | None = None
    ) -> dict[str, Any] | None:
        """Finish an interrupted run from its recorded verdicts."""
        index = store.read_artifact(_INDEX)
        if not isinstance(index, dict) or index.get("status") != "judged":
            return None
        judgements = [item for item in index.get("judgements") or [] if isinstance(item, dict)]
        unresolved = [item for item in index.get("unresolved") or [] if isinstance(item, dict)]
        summary = _empty_summary()
        summary["resumed"] = True
        summary["collisions"] = len(judgements)
        summary["judged"] = len(judgements)
        summary["unresolved"] = len(unresolved)
        store.log_event(
            "glossary_disambiguation_resumed",
            collision_ids=[str(item.get("collision_id") or "") for item in judgements],
        )
        summary["renderings_applied"] = self._apply(store, judgements, progress=progress)
        store.write_artifact(
            _INDEX, {"status": "completed", "judgements": judgements, "unresolved": unresolved}
        )
        store.log_event("glossary_disambiguation_finished", **summary)
        return summary

    # -- internals ----------------------------------------------------------

    def _collision_groups(self, store: Storage) -> list[dict[str, Any]]:
        """Group distinct sources that share one target, skipping already-judged collisions."""
        done_index = store.read_artifact(_INDEX)
        done_ids: set[str] = set()
        if isinstance(done_index, dict):
            # Only judged groups are settled. An unresolved group stays eligible so the next
            # chapter's extraction can retry it with wider context.
            for item in done_index.get("judgements") or []:
                if isinstance(item, dict) and item.get("collision_id"):
                    done_ids.add(str(item["collision_id"]))
        by_target: dict[str, list[GlossaryTerm]] = {}
        for term in store.all_terms():
            target = (term.target or "").strip()
            source = (term.source or "").strip()
            if target and source:
                by_target.setdefault(target, []).append(term)
        groups: list[dict[str, Any]] = []
        for target, terms in by_target.items():
            sources = sorted({term.source for term in terms})
            if len(sources) < 2:
                continue
            collision_id = _collision_id(target, sources)
            if collision_id in done_ids:
                continue
            groups.append(
                {
                    "collision_id": collision_id,
                    "target": target,
                    "sources": [
                        {
                            "source": term.source,
                            "occurrences": self._occurrences(store, term.source, term),
                        }
                        for term in sorted(terms, key=lambda item: item.source)
                    ],
                }
            )
        return groups

    @staticmethod
    def _occurrences(store: Storage, source: str, term: GlossaryTerm) -> list[dict[str, int]]:
        """Locate a bounded, book-spread sample of passages that mention the source."""
        matcher = GlossaryTerm(
            source=source,
            target=term.target,
            type=term.type,
            aliases=list(term.aliases or []),
        )
        keys = [key for key in term_match_sources(matcher) if (key or "").strip()]
        if not keys:
            return []
        hits: list[dict[str, int]] = []
        for chapter in load_chapters(store):
            for text_index, segment in enumerate(chapter.text_segments):
                if not (segment.source or "").strip():
                    continue
                if any(source_matches_text(key, segment.source or "") for key in keys):
                    hits.append({"chapter": chapter.index, "index": text_index})
        if len(hits) <= _SAMPLED_OCCURRENCES:
            return hits
        # Spread the sample across the book so the judge sees how each spelling behaves later,
        # not only where it first appears.
        step = len(hits) / _SAMPLED_OCCURRENCES
        return [
            hits[min(len(hits) - 1, int(position * step))]
            for position in range(_SAMPLED_OCCURRENCES)
        ]

    def _apply(
        self,
        store: Storage,
        judgements: list[dict[str, Any]],
        *,
        progress: ProgressFn | None,
    ) -> int:
        """Apply judged renderings: replace changed targets in the glossary and the text."""
        applied = 0
        total = len(judgements)
        for done, judgement in enumerate(judgements, start=1):
            if progress:
                progress(done - 1, total, "Applying judged renderings")
            sources = [str(value) for value in judgement.get("sources") or [] if value]
            renderings = {
                str(entry.get("source") or ""): str(entry.get("target") or "")
                for entry in judgement.get("renderings") or []
                if isinstance(entry, dict)
            }
            for source in sources:
                term = store.get_term(source)
                if term is None:
                    continue
                others = [value for value in sources if value != source]
                old_wording = (term.target or "").strip()
                new_wording = renderings.get(source, "")
                changed = bool(new_wording) and new_wording != old_wording
                if changed:
                    # Source-anchored rewrite: only passages mentioning this source that still
                    # carry the old wording move to the new one, exactly like conflict settling.
                    plan = plan_conflict_writeback(
                        load_chapters(store),
                        term_source=source,
                        term=term,
                        rejected_targets=[old_wording],
                        chosen_target=new_wording,
                    )
                    if plan:
                        apply_target_rewrites(
                            store,
                            plan,
                            term_source=source,
                            rejected_targets=[old_wording],
                            chosen_target=new_wording,
                        )
                    with store.state_lock():
                        store.resolve_term(source, new_wording)
                    applied += 1
                note = _verdict_note(
                    judgement,
                    others,
                    wording=new_wording if changed else old_wording,
                    changed=changed,
                )
                if _VERDICT_TAG not in (term.note or ""):
                    combined = f"{term.note}\n{note}" if (term.note or "").strip() else note
                    with store.state_lock():
                        store.upsert_term(
                            GlossaryTerm(
                                source=term.source,
                                target=new_wording if changed else old_wording,
                                reading=term.reading,
                                type=term.type,
                                gender=term.gender,
                                aliases=list(term.aliases or []),
                                note=combined,
                                status=term.status,
                                first_chapter=term.first_chapter,
                            ),
                            chapter=term.first_chapter,
                        )
                store.log_event(
                    "glossary_target_disambiguated",
                    source=source,
                    collision_id=str(judgement.get("collision_id") or ""),
                    same_entity=judgement.get("same_entity"),
                    old_target=old_wording,
                    new_target=new_wording if changed else old_wording,
                    changed=changed,
                    reason=str(judgement.get("reason") or ""),
                )
        if progress:
            progress(total, total, "Applying judged renderings")
        return applied
