"""Book-understanding prescan agent using an economical tier.
Read the source before translation. Store one target-language digest per chapter in
chapter.meta["source_digest"], then combine digests and preliminary analysis into a
whole-book synopsis.
Inject both as a stable prompt prefix so translators know the plot, character arcs,
foreshadowing and revelations before translating early chapters. The fixed global prefix
supports cache reuse. Use grouped map-reduce merging for long books to bound prompt length.
"""

from __future__ import annotations

import re

from ..glossary.store import GlossaryTerm
from ..i18n.prompts import render
from . import prompts
from .base import Agent

# Character budget for one digest merge; group and recursively merge larger inputs.
_REDUCE_BUDGET = 12000

# Heuristic markers for front/back matter that should never receive the story template.
_NON_STORY_PATTERNS = re.compile(
    r"(?i)(all rights reserved|copyright\s*©|publisher|isbn|dedication|table of contents|"
    r"colophon|acknowledg|for my (mother|father|wife|husband)|^to [a-z ]+$|"
    r"^\s*©|little, brown|hachette)"
)


class Synopsizer(Agent):
    def digest_chapter(
        self,
        source_text: str,
        glossary_terms: list[GlossaryTerm] | None = None,
    ) -> str:
        """Summarize one source chapter in the target language; return empty on empty input or
        failure.  Glossary terms steer name translations to match the established mapping.
        """
        if not source_text.strip():
            return ""
        glossary = prompts.render_glossary(
            glossary_terms or [],
            max_note_chars=self.config.pipeline.glossary_note_chars,
        )
        if _looks_non_story(source_text):
            system = render("chapter_digest_system", src=self.src, tgt=self.tgt)
            user = render(
                "chapter_digest_user",
                src=self.src,
                tgt=self.tgt,
                source=source_text[:2000],
                glossary=glossary,
            )
            result = self._ask_text(system, user, operation="synopsis.chapter")
            return _collapse_to_sentence(result)
        system = render("chapter_digest_system", src=self.src, tgt=self.tgt)
        user = render(
            "chapter_digest_user",
            src=self.src,
            tgt=self.tgt,
            source=source_text[:8000],
            glossary=glossary,
        )
        # Use the fast tier with output headroom above the language-specific digest budget.
        return self._ask_text(system, user, operation="synopsis.chapter")

    def book_synopsis(
        self,
        digests: list[str],
        analysis_brief: str,
        glossary_terms: list[GlossaryTerm] | None = None,
    ) -> str:
        """Combine chapter digests and analysis into a book synopsis; use map-reduce for long
        inputs.  Glossary terms keep character names consistent with the established mapping.
        """
        items = [d.strip() for d in digests if d and d.strip()]
        if not items:
            return ""
        while True:
            groups = self._group(items, _REDUCE_BUDGET)
            if len(groups) == 1:
                return self._synth(groups[0], analysis_brief, glossary_terms)
            # Summarize each group first, then merge those summaries in the next round.
            # Every group must produce a summary: silently dropping a failed group would
            # yield a synopsis that omits whole stretches of the book.
            summaries = [self._synth(g, analysis_brief, glossary_terms) for g in groups]
            if not all(s.strip() for s in summaries):
                return ""
            items = summaries

    # Internal helpers.
    @staticmethod
    def _group(items: list[str], budget: int) -> list[list[str]]:
        """Greedily group strings by character budget, keeping joined groups near or below
        budget.
        """
        groups: list[list[str]] = []
        cur: list[str] = []
        size = 0
        for it in items:
            if cur and size + len(it) > budget:
                groups.append(cur)
                cur, size = [], 0
            cur.append(it)
            size += len(it) + 1
        if cur:
            groups.append(cur)
        return groups

    def _synth(
        self,
        digests: list[str],
        analysis_brief: str,
        glossary_terms: list[GlossaryTerm] | None = None,
    ) -> str:
        """Merge one group of chapter digests and style analysis into a higher-level synopsis."""
        numbered = "\n".join(f"[{i}] {d}" for i, d in enumerate(digests))
        glossary = prompts.render_glossary(
            glossary_terms or [],
            max_note_chars=self.config.pipeline.glossary_note_chars,
        )
        system = render("book_synopsis_system", src=self.src, tgt=self.tgt)
        user = render(
            "book_synopsis_user",
            src=self.src,
            tgt=self.tgt,
            analysis=analysis_brief or "(none)",
            digests=numbered,
            glossary=glossary,
        )
        # Thinking tokens can randomly exhaust the output budget; retry truncation.
        return self._ask_synopsis_book(system, user)

    def _ask_synopsis_book(self, system: str, user: str) -> str:
        """Call synopsis.book with retries; truncated thinking budgets are not final answers."""
        for _attempt in range(3):
            try:
                return (
                    self.client.complete(
                        [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        operation="synopsis.book",
                    )
                    or ""
                ).strip()
            except RuntimeError as error:
                if "truncated" not in str(error).lower():
                    raise
        return ""


def _looks_non_story(source_text: str) -> bool:
    """Heuristic: very short text or known front/back-matter markers."""
    stripped = source_text.strip()
    if len(stripped) < 300:
        return True
    return bool(_NON_STORY_PATTERNS.search(stripped[:800]))


def _collapse_to_sentence(text: str) -> str:
    """Reduce a template-shaped digest to a single identifying sentence.

    If the model still emitted section headings, take the first non-empty body
    line under any heading; otherwise return the first non-empty line.
    """
    lines = [ln.strip() for ln in (text or "").split("\n") if ln.strip()]
    if not lines:
        return ""
    for line in lines:
        if line.startswith("##"):
            continue
        return line
    return lines[0]
