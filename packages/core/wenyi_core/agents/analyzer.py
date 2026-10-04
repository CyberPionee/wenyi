"""Global analysis agent using the strong tier.
Read sample chapters to produce a style guide, character reference with gender and voice,
and initial term candidates. Seed characters and terms into the glossary as a consistent
reference for the book.
"""

from __future__ import annotations

from typing import Any

from ..glossary.store import TYPE_PERSON, GlossaryStore, GlossaryTerm
from ..i18n.metadata import normalize_gender, normalize_term_type
from ..llm.retrying import TruncatedResponseError
from ..storage.protocol import Storage
from .base import Agent

# Output budgets per attempt: the registered budget first, then a larger one when thinking
# tokens exhaust it. The first response raises a length stop instead of a partial answer.
_OUTPUT_BUDGETS: tuple[int | None, ...] = (None, 12288, 16384)


def _text(value: Any, default: str = "") -> str:
    """Normalize model fields to text; fall back for non-scalar values such as nested objects."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return default


class Analyzer(Agent):
    policy_phase = "analysis"

    def analyze(self, sample_text: str) -> dict[str, Any]:
        """Analyze samples and return type-checked style, character and terminology data.

        Retry up to 3 times when the answer is incomplete (thinking tokens can exhaust the
        output budget mid-JSON while finish_reason still says stop).
        """
        system = self.render("analyzer_system", src=self.src, tgt=self.tgt)
        user = self.render("analyzer_user", src=self.src, tgt=self.tgt, sample=sample_text)
        data: dict[str, Any] = {}
        for attempt, budget in enumerate(_OUTPUT_BUDGETS):
            try:
                # No default: propagate analysis failures for the caller to handle.
                data = self._ask_json(system, user, operation="analysis.style", max_tokens=budget)
            except TruncatedResponseError as error:
                if attempt == len(_OUTPUT_BUDGETS) - 1:
                    raise TruncatedResponseError(
                        "analysis.style",
                        "Style analysis was truncated at the output limit on every attempt; "
                        "raise max_output_tokens or lower reasoning_effort",
                    ) from error
                continue
            if not isinstance(data, dict):
                data = {}
            # Accept a list of prose bullets as well as the requested string. Never stringify objects.
            if isinstance(data.get("style_guide"), list):
                data["style_guide"] = "\n".join(
                    item.strip()
                    for item in data["style_guide"]
                    if isinstance(item, str) and item.strip()
                )
            for key in (
                "genre",
                "tone",
                "style_guide",
                "narration",
                "pacing",
                "register",
                "dialogue_style",
                "rhetoric",
            ):
                data[key] = _text(data.get(key))
            style = data.get("style_guide", "")
            if not style or style.endswith((".", "!", "?", "。", "！", "？")):
                break
            # Mid-sentence cut: retry the call with a larger budget.
        data["characters"] = self.dict_items(
            data.get("characters"), operation="analysis.style", field="characters"
        )
        data["terms"] = self.dict_items(
            data.get("terms"), operation="analysis.style", field="terms"
        )
        for character in data["characters"]:
            character["gender"] = normalize_gender(_text(character.get("gender")))
        for term in data["terms"]:
            term["type"] = normalize_term_type(_text(term.get("type")))
        return data

    def seed_glossary(self, store: Storage | GlossaryStore, analysis: dict[str, Any]) -> int:
        """Seed analyzed characters and terms into the glossary; return the entry count."""
        count = 0
        for ch in self.dict_items(
            analysis.get("characters"), operation="analysis.style", field="characters"
        ):
            source = _text(ch.get("source"))
            target = _text(ch.get("target"))
            if not source or not target:
                continue
            store.upsert_term(
                GlossaryTerm(
                    source=source,
                    target=target,
                    reading=_text(ch.get("reading")),
                    type=TYPE_PERSON,
                    gender=_text(ch.get("gender")),
                    note=_text(ch.get("note")),
                    first_chapter=0,
                ),
                chapter=0,
            )
            count += 1
        for tm in self.dict_items(analysis.get("terms"), operation="analysis.style", field="terms"):
            source = _text(tm.get("source"))
            target = _text(tm.get("target"))
            if not source or not target:
                continue
            store.upsert_term(
                GlossaryTerm(
                    source=source,
                    target=target,
                    reading=_text(tm.get("reading")),
                    type=normalize_term_type(_text(tm.get("type"))),
                    note=_text(tm.get("note")),
                    first_chapter=0,
                ),
                chapter=0,
            )
            count += 1
        return count

    def style_brief(self, analysis: dict[str, Any]) -> str:
        """Condense analysis into a style and character brief for the translator."""
        lines = []
        if analysis.get("genre"):
            lines.append(f"Genre: {analysis['genre']}")
        if analysis.get("tone"):
            lines.append(f"Tone: {analysis['tone']}")
        if analysis.get("style_guide"):
            lines.append(f"Style guide: {analysis['style_guide']}")
        # Include only style dimensions supported by the model's analysis.
        for key, tag in (
            ("narration", "Narration"),
            ("pacing", "Pacing"),
            ("register", "Register"),
            ("dialogue_style", "Dialogue style"),
            ("rhetoric", "Rhetoric"),
        ):
            if analysis.get(key):
                lines.append(f"{tag}: {analysis[key]}")
        chars = self.dict_items(
            analysis.get("characters"), operation="analysis.style", field="characters"
        )
        if chars:
            lines.append("Characters: ")
            for c in chars:
                gender = normalize_gender(_text(c.get("gender")))
                g = f", {gender}" if gender else ""
                note = f", {c.get('note')}" if c.get("note") else ""
                lines.append(
                    f"  - {c.get('target') or c.get('source', '')} ({c.get('source', '')}{g}{note})"
                )
        return "\n".join(lines)
