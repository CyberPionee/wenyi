"""Settle open terminology conflicts from book context once translation is complete.

The arbiter's decisions are recorded before any formal target changes, so an interrupted run
finishes from its own index instead of paying for another model call and possibly deciding
differently. Glossary writes live here rather than in the review Autofix publisher, which must
never modify the glossary.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..agents.glossary_arbiter import GlossaryConflictArbiter
from ..glossary.store import GlossaryTerm, source_matches_text, term_match_sources
from ..glossary.writeback import apply_target_rewrites, load_chapters, plan_conflict_writeback
from ..review.evidence import BookEvidenceIndex
from ..review.run_store import ReviewRunStore
from ..storage.protocol import Storage
from .review_checkpoint import ReviewTraceStore

if TYPE_CHECKING:
    from .runtime import PipelineRuntime

ProgressFn = Callable[[int, int, str], None]

_INDEX = "glossary-arbitration/decisions.json"
_SAMPLED_OCCURRENCES = 12


def _conflict_id(source: str) -> str:
    """Give each term a stable arbitration id without leaking its characters into artifacts."""
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    return f"glossary-{digest}"


def _empty_summary() -> dict[str, Any]:
    return {"conflicts": 0, "decided": 0, "undecided": 0, "segments_replaced": 0, "reason": ""}


class GlossaryArbitrationService:
    """Decide terminology conflicts from the term's use in the book and write the result back."""

    def __init__(self, runtime: PipelineRuntime):
        self._runtime = runtime

    def run(self, store: Storage, *, progress: ProgressFn | None = None) -> dict[str, Any]:
        """Settle every open conflict once the whole book has been translated."""
        summary = _empty_summary()
        if not self._runtime.config.pipeline.glossary_conflict_arbitration:
            summary["reason"] = "disabled"
            return summary
        resumed = self.resume_pending(store, progress=progress)
        if resumed is not None:
            return resumed
        groups = self._conflict_groups(store)
        if not groups:
            return summary
        summary["conflicts"] = len(groups)
        debug = ReviewRunStore(store.run_dir, storage=store, kind="glossary-arbitration")
        debug.start(
            reviewed_content_digest="glossary-arbitration",
            metadata={"kind": "glossary_arbitration", "conflicts": len(groups)},
        )
        evidence = BookEvidenceIndex(
            load_chapters(store),
            list(store.all_terms()),
            store.load_analysis() or {},
        )
        arbiter = GlossaryConflictArbiter(
            self._runtime.client,
            self._runtime.config,
            evidence,
            ReviewTraceStore(debug),
        )
        decisions: list[dict[str, Any]] = []
        undecided: list[dict[str, Any]] = []
        for done, group in enumerate(groups, start=1):
            if progress:
                progress(done - 1, len(groups), "Settling terminology conflicts")
            outcome = arbiter.arbitrate(group)
            if outcome["status"] == "decided":
                candidates = [group["established_target"], *group["proposed_targets"]]
                decisions.append(
                    {
                        "source": outcome["source"],
                        "target": outcome["recommended_target"],
                        "rejected": [
                            value for value in candidates if value != outcome["recommended_target"]
                        ],
                        "reason": outcome["reason"],
                    }
                )
            else:
                undecided.append({"source": outcome["source"], "reason": outcome["reason"]})
            if progress:
                progress(done, len(groups), "Settling terminology conflicts")
        summary["decided"] = len(decisions)
        summary["undecided"] = len(undecided)
        if not decisions:
            return summary
        # Persist the decisions before a formal target changes so an interruption can finish them.
        store.write_artifact(
            _INDEX, {"status": "decided", "decisions": decisions, "undecided": undecided}
        )
        summary["segments_replaced"] = self._apply(store, decisions, progress=progress)
        store.write_artifact(
            _INDEX, {"status": "completed", "decisions": [], "undecided": undecided}
        )
        store.log_event("glossary_arbitration_finished", **summary)
        return summary

    def resume_pending(
        self, store: Storage, *, progress: ProgressFn | None = None
    ) -> dict[str, Any] | None:
        """Finish an interrupted arbitration from its recorded decisions."""
        index = store.read_artifact(_INDEX)
        if not isinstance(index, dict) or index.get("status") != "decided":
            return None
        decisions = [item for item in index.get("decisions") or [] if isinstance(item, dict)]
        undecided = [item for item in index.get("undecided") or [] if isinstance(item, dict)]
        summary = _empty_summary()
        summary["resumed"] = True
        summary["conflicts"] = len(decisions)
        summary["decided"] = len(decisions)
        summary["undecided"] = len(undecided)
        store.log_event(
            "glossary_arbitration_resumed",
            sources=[str(item.get("source") or "") for item in decisions],
        )
        summary["segments_replaced"] = self._apply(store, decisions, progress=progress)
        store.write_artifact(
            _INDEX, {"status": "completed", "decisions": [], "undecided": undecided}
        )
        store.log_event("glossary_arbitration_finished", **summary)
        return summary

    # -- internals ----------------------------------------------------------

    def _conflict_groups(self, store: Storage) -> list[dict[str, Any]]:
        """Group open conflicts by term, keeping the established rendering first."""
        proposals: dict[str, list[str]] = {}
        for conflict in store.open_conflicts():
            source = str(conflict.get("source") or "")
            proposed = str(conflict.get("proposed_target") or "")
            if source and proposed.strip():
                proposals.setdefault(source, []).append(proposed)
        groups: list[dict[str, Any]] = []
        for source, proposed in proposals.items():
            term = store.get_term(source)
            if term is None or not (term.target or "").strip():
                continue
            groups.append(
                {
                    "conflict_id": _conflict_id(source),
                    "source": source,
                    "established_target": term.target,
                    "proposed_targets": list(dict.fromkeys(proposed)),
                    "occurrences": self._occurrences(store, source, term),
                }
            )
        return groups

    @staticmethod
    def _occurrences(store: Storage, source: str, term: GlossaryTerm) -> list[dict[str, int]]:
        """Locate a bounded, book-spread sample of the translated passages that mention the term."""
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
                if not (segment.target or "").strip():
                    continue
                if any(source_matches_text(key, segment.source or "") for key in keys):
                    hits.append({"chapter": chapter.index, "index": text_index})
        if len(hits) <= _SAMPLED_OCCURRENCES:
            return hits
        # Spread the sample across the book: clustering at the start would hide how the term
        # behaves later, which is exactly what the arbiter has to weigh.
        step = len(hits) / _SAMPLED_OCCURRENCES
        return [
            hits[min(len(hits) - 1, int(position * step))]
            for position in range(_SAMPLED_OCCURRENCES)
        ]

    def _apply(
        self,
        store: Storage,
        decisions: list[dict[str, Any]],
        *,
        progress: ProgressFn | None,
    ) -> int:
        """Write each settled rendering back, then lock the term and close its conflicts."""
        replaced = 0
        total = len(decisions)
        for done, decision in enumerate(decisions, start=1):
            if progress:
                progress(done - 1, total, "Writing settled terminology back")
            source = str(decision.get("source") or "")
            chosen = str(decision.get("target") or "")
            rejected = [value for value in decision.get("rejected") or [] if isinstance(value, str)]
            term = store.get_term(source) if source else None
            if term is None or not chosen:
                continue
            count = 0
            plan = plan_conflict_writeback(
                load_chapters(store),
                term_source=source,
                term=term,
                rejected_targets=rejected,
                chosen_target=chosen,
            )
            if plan:
                count = int(
                    apply_target_rewrites(
                        store,
                        plan,
                        term_source=source,
                        rejected_targets=rejected,
                        chosen_target=chosen,
                    )["segments_replaced"]
                )
                replaced += count
            # The glossary decision lands after its text so an interruption resumes from the index.
            with store.state_lock():
                store.resolve_term(source, chosen)
                store.mark_conflicts_resolved(source)
                store.log_event(
                    "glossary_conflict_arbitrated",
                    source=source,
                    target=chosen,
                    rejected=rejected,
                    segments_replaced=count,
                    reason=str(decision.get("reason") or ""),
                )
        if progress:
            progress(total, total, "Writing settled terminology back")
        return replaced
