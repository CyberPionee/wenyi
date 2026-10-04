"""Recent translated paragraphs for local continuity of pronouns, address and tone.
Source prescans provide a fixed whole-book synopsis and chapter digest as stable prompt
prefixes. This module manages only the recent-translation suffix that changes each batch,
complementing that global context.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RollingContext:
    recent_targets: list[str] = field(default_factory=list)
    recent_pairs: list[dict[str, str]] = field(default_factory=list)
    max_recent_keep: int = 40  # Maximum number of recent translated paragraphs to retain.

    def render(self, n_recent: int, *, with_source: bool = True) -> str:
        """Return the last n_recent translations as text; the template supplies its context
        heading. When ``with_source`` and paired history exist, render Source/Translation
        lines; legacy target-only history degrades to translations alone.
        """
        if n_recent <= 0:
            return ""
        if with_source and self.recent_pairs:
            tail = self.recent_pairs[-n_recent:]
            lines: list[str] = []
            for pair in tail:
                lines.append(f"Source: {pair.get('source', '')}")
                lines.append(f"Translation: {pair.get('target', '')}")
            return "\n".join(lines)
        tail_targets = self.recent_targets[-n_recent:]
        return "\n".join(tail_targets)

    def add_targets(self, targets: list[str]) -> None:
        """Append nonempty translations without source pairing (legacy/resume)."""
        self.recent_targets.extend(t for t in targets if t and t.strip())
        if len(self.recent_targets) > self.max_recent_keep:
            self.recent_targets = self.recent_targets[-self.max_recent_keep :]

    def add_pairs(self, sources: list[str], targets: list[str]) -> None:
        """Append nonempty source-target pairs and retain only the configured tail.

        Empty or whitespace-only targets never enter the pair list. ``recent_targets`` is
        kept in sync so legacy readers and ``to_dict`` still see translations alone.
        """
        for source, target in zip(sources, targets):
            text = target or ""
            if not text.strip():
                continue
            self.recent_pairs.append({"source": source or "", "target": text})
        if len(self.recent_pairs) > self.max_recent_keep:
            self.recent_pairs = self.recent_pairs[-self.max_recent_keep :]
        self.recent_targets = [pair["target"] for pair in self.recent_pairs]

    def to_dict(self) -> dict:
        """Serialize recent pairs and the legacy target list together."""
        if self.recent_pairs:
            recent_targets = [pair["target"] for pair in self.recent_pairs]
        else:
            recent_targets = self.recent_targets
        return {
            "recent_targets": recent_targets,
            "recent_pairs": self.recent_pairs,
            "max_recent_keep": self.max_recent_keep,
        }

    @classmethod
    def from_dict(
        cls,
        d: dict,
        *,
        min_recent_keep: int = 0,
    ) -> RollingContext:
        """Restore persisted context with at least the capacity required by current
        configuration. Legacy JSON that only has ``recent_targets`` loads as target-only
        history and renders without source lines.
        """
        persisted = d.get("max_recent_keep", 40)
        max_recent_keep = persisted if isinstance(persisted, int) else 40
        max_recent_keep = max(max_recent_keep, min_recent_keep)
        raw_pairs = d.get("recent_pairs") or []
        recent_pairs: list[dict[str, str]] = []
        for item in raw_pairs:
            if not isinstance(item, dict):
                continue
            source = item.get("source") or ""
            target = item.get("target") or ""
            if not str(target).strip():
                continue
            recent_pairs.append({"source": str(source), "target": str(target)})
        recent_pairs = recent_pairs[-max_recent_keep:]
        recent_targets_raw = d.get("recent_targets", []) or []
        if recent_pairs:
            recent_targets = [pair["target"] for pair in recent_pairs]
        else:
            recent_targets = [str(t) for t in recent_targets_raw if str(t).strip()]
            recent_targets = recent_targets[-max_recent_keep:]
        return cls(
            recent_targets=recent_targets,
            recent_pairs=recent_pairs,
            max_recent_keep=max_recent_keep,
        )
