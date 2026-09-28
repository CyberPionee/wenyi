"""Opt-in C-batch quality passes that never write formal chapter targets.

Each pass is disabled by default. Enabled passes record analysis/events only so
automatic changes still require the existing shadow → Autofix publication chain.
"""

from __future__ import annotations

from typing import Any

from ..glossary.store import GlossaryTerm
from ..i18n.prompts import render
from . import prompts
from .base import Agent


class QualityPassAgent(Agent):
    """Self-revision, editorial notes, final polish candidates, self-check and back-translation."""

    def self_revise(
        self,
        sources: list[str],
        targets: list[str],
        *,
        style: str = "",
        glossary_terms: list[GlossaryTerm] | None = None,
    ) -> list[str]:
        """Return revised drafts; unchanged input on count mismatch or failure."""
        if not targets:
            return []
        n = len(targets)
        system = render("self_revision_system", src=self.src, tgt=self.tgt, n=n)
        user = render(
            "self_revision_user",
            src=self.src,
            tgt=self.tgt,
            style=style or "(none)",
            glossary=prompts.render_glossary(glossary_terms or []),
            n=n,
            pairs=prompts.numbered_pairs(list(sources), list(targets)),
        )
        items = self._ask_json(
            system, user, operation="quality.self_revision", key="revised", default=None
        )
        if isinstance(items, list) and len(items) == n:
            return [str(x) for x in items]
        return list(targets)

    def editorial_notes(
        self,
        pairs: list[tuple[str, str]],
        *,
        style: str = "",
        book_synopsis: str = "",
        max_notes: int = 12,
    ) -> list[str]:
        """Return whole-book editorial notes in the target language."""
        if not pairs:
            return []
        system = render("editorial_pass_system", src=self.src, tgt=self.tgt)
        user = render(
            "editorial_pass_user",
            src=self.src,
            tgt=self.tgt,
            style=style or "(none)",
            book_synopsis=book_synopsis or "(none)",
            pairs=prompts.numbered_pairs([s for s, _ in pairs], [t for _, t in pairs]),
            n=max_notes,
        )
        items = self._ask_json(system, user, operation="quality.editorial", key="notes", default=[])
        if not isinstance(items, list):
            return []
        return [str(item) for item in items[:max_notes]]

    def final_polish(
        self,
        targets: list[str],
        *,
        style: str = "",
        glossary_terms: list[GlossaryTerm] | None = None,
    ) -> list[str]:
        """Return polished candidates; unchanged input on count mismatch or failure."""
        if not targets:
            return []
        n = len(targets)
        system = render("final_polish_system", src=self.src, tgt=self.tgt, n=n)
        user = render(
            "final_polish_user",
            src=self.src,
            tgt=self.tgt,
            style=style or "(none)",
            glossary=prompts.render_glossary(glossary_terms or []),
            n=n,
            numbered_target=prompts.numbered(list(targets)),
        )
        items = self._ask_json(
            system, user, operation="quality.final_polish", key="polished", default=None
        )
        if isinstance(items, list) and len(items) == n:
            return [str(x) for x in items]
        return list(targets)

    def chapter_selfcheck(
        self,
        sources: list[str],
        targets: list[str],
        *,
        glossary_terms: list[GlossaryTerm] | None = None,
    ) -> list[dict[str, Any]]:
        """Return cheap LLM findings for one chapter; never writes targets."""
        if not targets:
            return []
        n = len(targets)
        system = render("chapter_selfcheck_system", src=self.src, tgt=self.tgt, n=n)
        user = render(
            "chapter_selfcheck_user",
            src=self.src,
            tgt=self.tgt,
            glossary=prompts.render_glossary(glossary_terms or []),
            n=n,
            pairs=prompts.numbered_pairs(list(sources), list(targets)),
        )
        items = self._ask_json(
            system, user, operation="quality.chapter_selfcheck", key="findings", default=[]
        )
        findings: list[dict[str, Any]] = []
        for item in self.dict_items(items, operation="quality.chapter_selfcheck", field="findings"):
            detail = item.get("detail")
            if not isinstance(detail, str) or not detail.strip():
                continue
            raw_index = item.get("index")
            index = None
            if (
                isinstance(raw_index, int)
                and not isinstance(raw_index, bool)
                and 0 <= raw_index < n
            ):
                index = raw_index
            findings.append(
                {
                    "index": index,
                    "kind": str(item.get("kind") or "unknown"),
                    "detail": detail.strip(),
                }
            )
        return findings

    def back_translate(self, targets: list[str]) -> list[str]:
        """Back-translate drafts into the source language for QA comparison."""
        if not targets:
            return []
        n = len(targets)
        system = render("back_translation_system", src=self.src, tgt=self.tgt, n=n)
        user = render(
            "back_translation_user",
            src=self.src,
            tgt=self.tgt,
            n=n,
            numbered_target=prompts.numbered(list(targets)),
        )
        items = self._ask_json(
            system, user, operation="quality.back_translation", key="back", default=None
        )
        if isinstance(items, list) and len(items) == n:
            return [str(x) for x in items]
        return []
