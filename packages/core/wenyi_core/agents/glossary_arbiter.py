"""Prepare terminology-conflict arbitration prompts and validate the settled rendering."""

from __future__ import annotations

import json
from typing import Any

from ..config import Config
from ..i18n.prompts import render
from ..llm.base import LLMClient
from ..review.contracts import EvidenceQueries, ReviewTrace
from ..review.models import clean_text, normalize_value
from .review_actions import ReviewActionLoop, ReviewLoopProtocolError, validate_evidence_refs

_MAX_ARBITRATION_SAMPLE_TEXT = 1200
_MAX_ARBITRATION_PAYLOAD_BYTES = 96_000


def conflict_candidates(conflict: dict[str, Any]) -> list[str]:
    """Return the distinct renderings the arbiter may choose from, established target first."""
    values = [conflict.get("established_target"), *(conflict.get("proposed_targets") or [])]
    return [value for value in dict.fromkeys(values) if isinstance(value, str) and value.strip()]


class GlossaryConflictArbiter:
    """Decide one terminology conflict from the term's use across the whole book.

    The choice stays inside the supplied candidates: an arbiter that could invent a rendering
    would change terminology without the operator ever seeing the option. A conflict it cannot
    separate is returned as undecided so a human still owns that decision.
    """

    def __init__(
        self,
        client: LLMClient,
        config: Config,
        evidence: EvidenceQueries,
        trace: ReviewTrace,
    ):
        self.config = config
        # The arbitration belongs to the translate workflow and settles terminology for the
        # translated text, so it renders the translation-phase policy. Hashing its templates into
        # the review phase instead would make a terminology-prompt edit invalidate a completed
        # review and force a full re-run of it.
        self.language_policy = config.language_policy("translation")
        self.evidence = evidence
        self.trace = trace
        self._loop = ReviewActionLoop(client, config, evidence, trace)

    def arbitrate(self, conflict: dict[str, Any]) -> dict[str, Any]:
        """Settle one conflict; retain it for a human when the evidence cannot separate them."""
        conflict_id = str(conflict["conflict_id"])
        source = str(conflict["source"])

        def undecided(reason: str, refs: set[str] | None = None) -> dict[str, Any]:
            """Build a conservative result that leaves the conflict open and records why."""
            self.trace.log_event(
                "glossary_arbitration_undecided",
                conflict_id=conflict_id,
                source=source,
                reason=reason,
            )
            return {
                "conflict_id": conflict_id,
                "source": source,
                "status": "unresolved",
                "recommended_target": "",
                "reason": reason,
                "evidence_refs": sorted(refs or set()),
            }

        candidates = conflict_candidates(conflict)
        if len(candidates) < 2:
            return undecided("A conflict needs two distinct candidates to arbitrate.")

        sampled_refs: set[str] = set()
        samples: list[dict[str, Any]] = []
        for occurrence in conflict.get("occurrences") or []:
            chapter = occurrence.get("chapter")
            index = occurrence.get("index")
            if not isinstance(chapter, int) or not isinstance(index, int):
                continue
            segment = self.evidence.segment_ref(chapter, index)
            if segment is not None:
                sampled_refs.add(segment.ref)
            samples.append(
                {
                    "chapter": chapter,
                    "index": index,
                    "segment_ref": segment.ref if segment is not None else "",
                    "source": (
                        segment.source[:_MAX_ARBITRATION_SAMPLE_TEXT] if segment is not None else ""
                    ),
                    "target": (
                        segment.target[:_MAX_ARBITRATION_SAMPLE_TEXT] if segment is not None else ""
                    ),
                }
            )
        if not samples:
            return undecided("The term has no located occurrence to judge from.")

        compact = {
            "conflict_id": conflict_id,
            "source": source,
            "candidates": candidates,
            "occurrences": samples,
        }
        compact_json = json.dumps(compact, ensure_ascii=False, indent=2)
        if len(compact_json.encode("utf-8")) > _MAX_ARBITRATION_PAYLOAD_BYTES:
            return undecided("The sampled occurrences exceed the input size limit.", sampled_refs)

        system = render(
            "glossary_arbiter_system",
            plan=self.language_policy,
            src=self.config.source_lang,
            tgt=self.config.target_lang,
            max_evidence_rounds=self.config.pipeline.review_agent_max_evidence_rounds,
        )
        user = render(
            "glossary_arbiter_user",
            plan=self.language_policy,
            src=self.config.source_lang,
            tgt=self.config.target_lang,
            conflict_json=compact_json,
        )
        # Preauthorize only the sampled refs whose text appears in the prompt; anything the
        # arbiter wants beyond them has to be requested through the evidence tools.
        allowed_refs = set(sampled_refs)

        def validate_final(data: dict[str, Any], valid_refs: set[str]) -> dict[str, Any]:
            if data.get("conflict_id") != conflict_id:
                raise ReviewLoopProtocolError("conflict_id_mismatch")
            status = data.get("status")
            if status not in {"decided", "unresolved"}:
                raise ReviewLoopProtocolError("invalid_arbitration_status")
            recommended = clean_text(data.get("recommended_target"))
            reason = clean_text(data.get("reason"))
            if not reason:
                raise ReviewLoopProtocolError("arbitration_without_reason")
            if status == "decided":
                if not recommended:
                    raise ReviewLoopProtocolError("decision_without_target")
                chosen = next(
                    (
                        value
                        for value in candidates
                        if normalize_value(value) == normalize_value(recommended)
                    ),
                    None,
                )
                if chosen is None:
                    raise ReviewLoopProtocolError("recommended_target_not_candidate")
                recommended = chosen
            else:
                recommended = ""
            return {
                "conflict_id": conflict_id,
                "source": source,
                "status": status,
                "recommended_target": recommended,
                "reason": reason,
                "evidence_refs": validate_evidence_refs(data.get("evidence_refs"), valid_refs),
            }

        result, reason = self._loop.run(
            agent_id=f"glossary-arbiter-{conflict_id}",
            system=system,
            user=user,
            stage="glossary.arbitrate",
            allowed_refs=allowed_refs,
            validate_final=validate_final,
        )
        if result is not None:
            return result
        return undecided(f"Arbitration agent did not complete: {reason}", allowed_refs)
