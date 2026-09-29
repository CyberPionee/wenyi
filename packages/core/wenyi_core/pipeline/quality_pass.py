"""Opt-in C-batch quality passes after translation.

Enabled passes write only analysis/events. Formal chapter targets still change
exclusively through the Autofix publication chain.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..storage.protocol import Storage

if TYPE_CHECKING:
    from .runtime import PipelineRuntime

ProgressFn = Callable[[int, int, str], None]


class QualityPassService:
    """Run disabled-by-default self-revision / editorial / polish / self-check / back-translation."""

    def __init__(self, runtime: PipelineRuntime):
        self._runtime = runtime

    def run_after_translate(
        self,
        store: Storage,
        *,
        progress: ProgressFn | None = None,
    ) -> dict[str, Any]:
        """Execute every enabled quality pass; never write chapter targets."""
        cfg = self._runtime.config.pipeline
        enabled = {
            "self_revision": cfg.self_revision,
            "editorial_pass": cfg.editorial_pass,
            "final_polish": cfg.final_polish,
            "chapter_selfcheck": cfg.chapter_selfcheck,
            "back_translation": cfg.back_translation,
        }
        if not any(enabled.values()):
            return {}
        manifest = store.load_manifest()
        chapters = [
            store.load_chapter(row["index"])
            for row in manifest.get("chapters", [])
            if isinstance(row.get("index"), int)
        ]
        style = self._runtime.analyzer.style_brief(store.load_analysis() or {})
        analysis = store.load_analysis() or {}
        book_synopsis = str(analysis.get("book_synopsis") or "")
        terms = store.all_terms()
        result: dict[str, Any] = {}
        agent = self._runtime.quality_pass

        sampled_pairs: list[tuple[str, str]] = []
        chapter_notes: list[dict[str, Any]] = []
        revision_notes: list[dict[str, Any]] = []
        polish_notes: list[dict[str, Any]] = []
        back_notes: list[dict[str, Any]] = []
        for chapter in chapters:
            segments = [s for s in chapter.text_segments if (s.target or "").strip()]
            if not segments:
                continue
            sources = [s.source for s in segments]
            targets = [s.target or "" for s in segments]
            sampled_pairs.extend(list(zip(sources, targets))[:8])
            if cfg.self_revision:
                revised = agent.self_revise(sources, targets, style=style, glossary_terms=terms)
                for index, (before, after) in enumerate(zip(targets, revised)):
                    if after != before:
                        revision_notes.append(
                            {
                                "chapter": chapter.index,
                                "index": segments[index].index,
                                "suggested": after,
                            }
                        )
            if cfg.final_polish:
                polished = agent.final_polish(targets, style=style, glossary_terms=terms)
                for index, (before, after) in enumerate(zip(targets, polished)):
                    if after != before:
                        polish_notes.append(
                            {
                                "chapter": chapter.index,
                                "index": segments[index].index,
                                "suggested": after,
                            }
                        )
            if cfg.chapter_selfcheck:
                for finding in agent.chapter_selfcheck(sources, targets, glossary_terms=terms):
                    finding = dict(finding)
                    finding["chapter"] = chapter.index
                    local = finding.get("index")
                    if isinstance(local, int):
                        finding["index"] = segments[local].index
                    chapter_notes.append(finding)
            if cfg.back_translation:
                backs = agent.back_translate(targets[:4])
                for source, back in zip(sources[:4], backs):
                    back_notes.append(
                        {
                            "chapter": chapter.index,
                            "source_preview": source[:80],
                            "back_preview": back[:80],
                        }
                    )
        if cfg.editorial_pass:
            result["editorial_notes"] = agent.editorial_notes(
                sampled_pairs[:24], style=style, book_synopsis=book_synopsis
            )
        if revision_notes:
            result["self_revision_notes"] = revision_notes
        if polish_notes:
            result["final_polish_notes"] = polish_notes
        if chapter_notes:
            result["chapter_selfcheck_findings"] = chapter_notes
        if back_notes:
            result["back_translation_notes"] = back_notes
        if result:
            analysis = store.load_analysis() or {}
            analysis["quality_pass"] = result
            store.save_analysis(analysis)
            store.log_event(
                "quality_pass_finished",
                enabled={key: value for key, value in enabled.items() if value},
                counts={
                    key: len(value) if isinstance(value, list) else 1
                    for key, value in result.items()
                },
            )
        mode = getattr(cfg, "decision_anchors", "off")
        if mode != "off":
            from .decision_anchors import update_decision_anchors

            update_decision_anchors(store, mode=mode)
        return result
