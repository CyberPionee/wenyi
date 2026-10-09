"""Prepare target-collision judgement prompts and validate the outcome.

Distinct source terms that share one target are judged from their use in the book: whether
they name the same entity, and whether the source distinction still matters. The judge
records that verdict with its evidence; it never proposes a replacement rendering.
"""

from __future__ import annotations

import json
from typing import Any

from ..config import Config
from ..i18n.prompts import render
from ..llm.base import LLMClient
from ..review.contracts import EvidenceQueries, ReviewTrace
from ..review.models import clean_text
from .review_actions import ReviewActionLoop, ReviewLoopProtocolError, validate_evidence_refs

_MAX_SAMPLE_TEXT = 1200
_MAX_PAYLOAD_BYTES = 96_000


class GlossaryTargetDisambiguator:
    """Judge one group of sources that resolve to a single target.

    The verdict stays binary and evidence-backed: a collision the passages cannot settle is
    returned as unresolved so a human keeps that decision. Nothing here writes to the glossary.
    """

    def __init__(
        self,
        client: LLMClient,
        config: Config,
        evidence: EvidenceQueries,
        trace: ReviewTrace,
    ):
        self.config = config
        # Same phase as the conflict arbiter: this judge settles terminology questions for the
        # translated text and runs in the translate workflow, so it renders the translation plan.
        self.language_policy = config.language_policy("translation")
        self.evidence = evidence
        self.trace = trace
        self._loop = ReviewActionLoop(client, config, evidence, trace)

    def judge(self, collision: dict[str, Any]) -> dict[str, Any]:
        """Judge one collision group; keep it open for a human when the passages cannot decide."""
        collision_id = str(collision["collision_id"])
        target = str(collision.get("target") or "")

        def unresolved(reason: str, refs: set[str] | None = None) -> dict[str, Any]:
            """Build a conservative result that leaves the collision open and records why."""
            self.trace.log_event(
                "glossary_disambiguation_unresolved",
                collision_id=collision_id,
                target=target,
                reason=reason,
            )
            return {
                "collision_id": collision_id,
                "target": target,
                "status": "unresolved",
                "same_entity": None,
                "needs_distinction": None,
                "reason": reason,
                "evidence_refs": sorted(refs or set()),
            }

        sampled_refs: set[str] = set()
        sources: list[dict[str, Any]] = []
        for entry in collision.get("sources") or []:
            source = str(entry.get("source") or "")
            if not source:
                continue
            samples: list[dict[str, Any]] = []
            for occurrence in entry.get("occurrences") or []:
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
                            segment.source[:_MAX_SAMPLE_TEXT] if segment is not None else ""
                        ),
                        "target": (
                            segment.target[:_MAX_SAMPLE_TEXT] if segment is not None else ""
                        ),
                    }
                )
            sources.append({"source": source, "occurrences": samples})
        if not sources or all(not entry["occurrences"] for entry in sources):
            return unresolved("The group has no located occurrence to judge from.")

        compact = {
            "collision_id": collision_id,
            "target": target,
            "sources": sources,
        }
        compact_json = json.dumps(compact, ensure_ascii=False, indent=2)
        if len(compact_json.encode("utf-8")) > _MAX_PAYLOAD_BYTES:
            return unresolved("The sampled occurrences exceed the input size limit.", sampled_refs)

        system = render(
            "glossary_disambiguation_system",
            plan=self.language_policy,
            src=self.config.source_lang,
            tgt=self.config.target_lang,
            max_evidence_rounds=self.config.pipeline.review_agent_max_evidence_rounds,
        )
        user = render(
            "glossary_disambiguation_user",
            plan=self.language_policy,
            src=self.config.source_lang,
            tgt=self.config.target_lang,
            collision_json=compact_json,
        )
        # Preauthorize only the sampled refs whose text appears in the prompt; anything the
        # judge wants beyond them has to be requested through the evidence tools.
        allowed_refs = set(sampled_refs)

        def validate_final(data: dict[str, Any], valid_refs: set[str]) -> dict[str, Any]:
            if data.get("collision_id") != collision_id:
                raise ReviewLoopProtocolError("collision_id_mismatch")
            status = data.get("status")
            if status not in {"judged", "unresolved"}:
                raise ReviewLoopProtocolError("invalid_disambiguation_status")
            reason = clean_text(data.get("reason"))
            if not reason:
                raise ReviewLoopProtocolError("disambiguation_without_reason")
            same_entity = data.get("same_entity")
            needs_distinction = data.get("needs_distinction")
            if status == "judged":
                if not isinstance(same_entity, bool) or not isinstance(needs_distinction, bool):
                    raise ReviewLoopProtocolError("judgement_without_verdict")
                # A distinction only applies when the sources are one entity; two distinct
                # entities already differ by identity, so the flag would carry no meaning.
                if not same_entity and needs_distinction:
                    raise ReviewLoopProtocolError("distinction_without_same_entity")
            else:
                same_entity = None
                needs_distinction = None
            return {
                "collision_id": collision_id,
                "target": target,
                "status": status,
                "same_entity": same_entity,
                "needs_distinction": needs_distinction,
                "reason": reason,
                "evidence_refs": validate_evidence_refs(data.get("evidence_refs"), valid_refs),
            }

        result, reason = self._loop.run(
            agent_id=f"glossary-disambiguator-{collision_id}",
            system=system,
            user=user,
            stage="glossary.disambiguate",
            allowed_refs=allowed_refs,
            validate_final=validate_final,
        )
        if result is not None:
            return result
        return unresolved(f"Disambiguation agent did not complete: {reason}", allowed_refs)
