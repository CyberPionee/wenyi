"""Review agent using the cheap tier.
Compare source and translation paragraph by paragraph for omissions, additions,
mistranslations, glossary violations and pronoun errors.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..llm.json_parser import parse_json_result
from . import prompts
from .base import Agent


class ReviewOutputError(ValueError):
    """Structured review output error that can be retried with a smaller input."""

    def __init__(self, reason: str):
        super().__init__(f"Review output protocol error: {reason}")
        self.reason = reason


@dataclass(frozen=True)
class ReviewResult:
    """Structured result of one review call, including whether local JSON repair was used."""

    issues: list[dict[str, Any]]
    soft_findings: list[dict[str, Any]] = field(default_factory=list)
    repaired: bool = False


class Reviewer(Agent):
    policy_phase = "review"

    def review(
        self, sources: list[str], targets: list[str], glossary_terms=None
    ) -> list[dict[str, Any]]:
        """Return issue dictionaries containing index, type, detail and suggestion."""
        return self.review_result(sources, targets, glossary_terms).issues

    def review_result(
        self,
        sources: list[str],
        targets: list[str],
        glossary_terms=None,
        *,
        style: str = "",
        book_synopsis: str = "",
        chapter_digest: str = "",
        trace: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> ReviewResult:
        """Return issues and soft findings with recovery metadata.

        Soft findings are uncertain notes; they never enter fix/autofix/arbiter.
        """
        if not sources:
            return ReviewResult([], soft_findings=[])
        system = self.render("reviewer_system", src=self.src, tgt=self.tgt, n=len(sources))
        user = self.render(
            "reviewer_user",
            src=self.src,
            tgt=self.tgt,
            glossary=prompts.render_glossary(
                glossary_terms or [],
                max_note_chars=self.config.pipeline.glossary_note_chars,
            ),
            n=len(sources),
            pairs=prompts.numbered_pairs(sources, targets),
            style=prompts.clip_context_block(style, 500),
            book_synopsis=prompts.clip_context_block(book_synopsis, 600),
            chapter_digest=prompts.clip_context_block(chapter_digest, 400),
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if trace:
            trace("request", {"messages": [dict(message) for message in messages]})
        try:
            text = self.client.complete(
                messages,
                operation="review.scan",
                json_mode=True,
            )
        except Exception as error:
            if trace:
                trace(
                    "error",
                    {
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )
            raise
        if trace:
            trace("response", {"raw_response": text})
        try:
            parsed = parse_json_result(text)
        except ValueError:
            raise ReviewOutputError("malformed_json") from None
        data = parsed.value
        repaired = parsed.repaired
        if trace:
            trace(
                "parsed",
                {
                    "value": data,
                    "json_repaired": repaired,
                },
            )

        if not isinstance(data, dict):
            raise ReviewOutputError("response_not_object")
        if not parsed.safe_for_complete_payload:
            # Structural repair means the payload may be an invented fragment.
            raise ReviewOutputError("unsafe_json_repair")
        reviewed_segments = data.get("reviewed_segments")
        if (
            isinstance(reviewed_segments, bool)
            or not isinstance(reviewed_segments, int)
            or reviewed_segments != len(sources)
        ):
            raise ReviewOutputError("reviewed_segments_mismatch")
        if data.get("complete") is not True:
            raise ReviewOutputError("completion_marker_missing")

        issues = data.get("issues")
        if not isinstance(issues, list):
            raise ReviewOutputError("issues_not_list")
        if any(not isinstance(item, dict) for item in issues):
            raise ReviewOutputError("issue_not_object")
        validated = self._validate_issues(issues, len(sources))
        raw_soft = data.get("soft_findings", [])
        if raw_soft is None:
            raw_soft = []
        if not isinstance(raw_soft, list):
            raise ReviewOutputError("soft_findings_not_list")
        if any(not isinstance(item, dict) for item in raw_soft):
            raise ReviewOutputError("soft_finding_not_object")
        soft_findings = self._validate_soft_findings(raw_soft, len(sources))
        return ReviewResult(validated, soft_findings=soft_findings, repaired=repaired)

    @staticmethod
    def _issue_types() -> set[str]:
        """Keep in sync with review_loop._ISSUE_TYPES."""
        return {
            "missing",
            "added",
            "mistranslation",
            "terminology",
            "pronoun",
            "voice",
            "style",
        }

    @staticmethod
    def _validate_issues(
        issues: list[dict[str, Any]],
        segment_count: int,
    ) -> list[dict[str, Any]]:
        """Normalize and validate every candidate so malformed fields cannot silently mean no
        issues.
        """
        allowed_types = Reviewer._issue_types()
        for item in issues:
            index = item.get("index")
            if isinstance(index, str):
                try:
                    index = int(index.strip())
                except ValueError:
                    index = None
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < segment_count
            ):
                raise ReviewOutputError("invalid_issue_index")
            if item.get("type") not in allowed_types:
                raise ReviewOutputError("invalid_issue_type")
            for field_name in ("detail", "suggestion"):
                value = item.get(field_name)
                if not isinstance(value, str) or not value.strip():
                    raise ReviewOutputError(f"invalid_issue_{field_name}")
        return [
            {
                "index": int(str(item["index"]).strip()),
                "type": item["type"],
                "detail": item["detail"].strip(),
                "suggestion": item["suggestion"].strip(),
            }
            for item in issues
        ]

    @staticmethod
    def _validate_soft_findings(
        findings: list[dict[str, Any]],
        segment_count: int,
    ) -> list[dict[str, Any]]:
        """Normalize uncertain notes. Suggestion is optional; empty findings are dropped."""
        allowed_types = Reviewer._issue_types()
        validated: list[dict[str, Any]] = []
        for item in findings:
            detail = item.get("detail")
            if not isinstance(detail, str) or not detail.strip():
                continue
            finding_type = item.get("type")
            if finding_type is not None and finding_type not in allowed_types:
                continue
            raw_index = item.get("index")
            index: int | None = None
            if raw_index is not None and not isinstance(raw_index, bool):
                try:
                    parsed = int(str(raw_index).strip())
                except ValueError:
                    parsed = None
                if parsed is not None and 0 <= parsed < segment_count:
                    index = parsed
            suggestion = item.get("suggestion")
            validated.append(
                {
                    "index": index,
                    "type": str(finding_type or ""),
                    "detail": detail.strip(),
                    "suggestion": suggestion.strip() if isinstance(suggestion, str) else "",
                }
            )
        return validated
