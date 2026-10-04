"""Polishing agent using the strong tier.
Improve literary quality in the target language without changing information or paragraph
count. Prefer appending a polish user turn to the translation conversation so shared
prompt prefixes stay cacheable. Preserve the original translation on alignment failure so
polishing cannot drop paragraphs.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..glossary.store import GlossaryTerm
from . import prompts
from .base import Agent, Messages

# Heuristic: CJK characters dominate target-language prose in zh/ja/ko.
_CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿]")


def _looks_like_source_leak(source: str, polished: str) -> bool:
    """Return True when *polished* is suspiciously similar to *source*.

    Guards against the model returning the source-language paragraph as the
    "polished" result instead of the target-language translation.  Short strings
    (numbers, names, single words) are exempt because they legitimately stay
    identical across languages.
    """
    src = (source or "").strip()
    pol = (polished or "").strip()
    if not src or not pol:
        return False
    # Very short text is almost always preserved as-is (numbers, names, etc.).
    if len(src) < 15 or len(pol) < 15:
        return False
    if src == pol:
        return True

    def _norm(t: str) -> str:
        return re.sub(r"\s+", "", t).lower()

    ns, np_ = _norm(src), _norm(pol)
    if ns == np_:
        return True
    if not ns or not np_:
        return False
    # If the shorter string is almost fully contained in the longer one, it is a leak.
    shorter, longer = (ns, np_) if len(ns) <= len(np_) else (np_, ns)
    if len(shorter) >= 10 and shorter in longer:
        return True
    # Character-overlap ratio: >0.85 means the "polished" text is essentially the source.
    from collections import Counter

    cs, cp = Counter(ns), Counter(np_)
    overlap = sum((cs & cp).values())
    if overlap / max(len(ns), len(np_)) > 0.85:
        return True
    return False


class Polisher(Agent):
    def polish(
        self,
        targets: list[str],
        *,
        sources: Sequence[str] | None = None,
        glossary_terms: list[GlossaryTerm] | None = None,
        style: str = "",
        next_source: str = "",
    ) -> list[str]:
        """Polish an aligned list in a fresh conversation; return input unchanged on failure.

        When ``sources`` is provided, the user prompt pairs each source paragraph with its
        target so the editor can keep meaning and register; otherwise only targets are sent.
        """
        if not targets:
            return []
        n = len(targets)
        system = self.render("polisher_system", src=self.src, tgt=self.tgt, n=n)
        pairs = (
            prompts.numbered_pairs(list(sources), targets)
            if sources is not None and len(sources) == n
            else prompts.numbered(targets)
        )
        user = self.render(
            "polisher_user",
            src=self.src,
            tgt=self.tgt,
            glossary=prompts.render_glossary(
                glossary_terms or [],
                max_note_chars=self.config.pipeline.glossary_note_chars,
            ),
            style=style or "(none)",
            n=n,
            pairs=pairs,
            next_source=prompts.render_source_reference(next_source),
        )
        items = self._ask_json(system, user, operation="polish.body", key="polished", default=None)
        if isinstance(items, list) and len(items) == n:
            result = [str(x) for x in items]
            # Reject the batch if any item is essentially the source text.
            if sources is not None and len(sources) == n:
                for s, t in zip(sources, result):
                    if _looks_like_source_leak(s, t):
                        return list(targets)
            return result
        return list(targets)

    def polish_continue(
        self,
        turn: Messages,
        *,
        n: int,
        sources: Sequence[str] | None = None,
        next_source: str = "",
    ) -> list[str] | None:
        """Append a polish user turn to a successful translation transcript.

        Returns the polished list on success, or ``None`` when the continuation fails so the
        caller can fall back to a standalone polish call or keep the raw translations.
        """
        if n <= 0 or len(turn) < 3:
            return None
        continue_user = self.render(
            "polisher_continue_user",
            src=self.src,
            tgt=self.tgt,
            n=n,
            next_source=prompts.render_source_reference(next_source),
        )
        messages: Messages = [
            *turn,
            {"role": "user", "content": continue_user},
        ]
        items = self._ask_json_messages(
            messages,
            operation="polish.body",
            key="polished",
            default=None,
        )
        if isinstance(items, list) and len(items) == n:
            result = [str(x) for x in items]
            if sources is not None and len(sources) == n:
                for s, t in zip(sources, result):
                    if _looks_like_source_leak(s, t):
                        return None
            return result
        return None
