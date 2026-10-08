"""Opt-in C-batch quality passes after translation.

Enabled passes write only analysis/events. Formal chapter targets still change
exclusively through the Autofix publication chain.
"""

from __future__ import annotations

import json
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
        from .tuning import QUALITY_PASS_KEYS, quality_pass_plan

        passes, coverage = quality_pass_plan(
            cfg.quality_passes, tier=cfg.autonomy_tier, configured=cfg.model_dump()
        )
        if not passes:
            return {}
        enabled = {key: key in passes for key in QUALITY_PASS_KEYS}
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

        # These passes run after translation, where the only other progress signal is the word
        # count translation already saturated, so every step reports its own position.
        staged = [
            (key, label)
            for key, label in (
                ("self_revision", "self revision"),
                ("final_polish", "final polish"),
                ("chapter_selfcheck", "chapter self-check"),
                ("back_translation", "back-translation"),
            )
            if enabled[key]
        ]
        active = [
            chapter
            for chapter in chapters
            if any((segment.target or "").strip() for segment in chapter.text_segments)
        ]
        if coverage == "risk" and active:
            # Deterministic scans decide which chapters deserve the expensive passes; a chapter
            # no scan flagged keeps its translation as it stands.
            from .evaluation import select_risk_segments

            flagged = {
                item.chapter for item in select_risk_segments(active, terms=terms, sample_ratio=0.0)
            }
            active = [chapter for chapter in active if chapter.index in flagged]
        # A pass that already produced its notes is not repeated. "auto" turns passes on by
        # default, so a resumed or repeated run must not spend their calls again.
        recorded = {
            str(index): {key for key in value if isinstance(key, str)}
            for index, value in (analysis.get("quality_pass_done") or {}).items()
            if isinstance(value, list)
        }
        work = [
            (
                chapter,
                [
                    key
                    for key, _label in staged
                    if key not in recorded.get(str(chapter.index), set())
                ],
            )
            for chapter in active
        ]
        work = [(chapter, keys) for chapter, keys in work if keys]
        editorial_pending = enabled["editorial_pass"] and "editorial_pass" not in recorded
        total = sum(len(keys) for _chapter, keys in work) + (1 if editorial_pending else 0)
        done = 0
        failed: list[str] = []

        def report(label: str) -> None:
            if progress is not None:
                progress(done, total, label)

        sampled_pairs: list[tuple[str, str]] = []
        chapter_notes: list[dict[str, Any]] = []
        revision_notes: list[dict[str, Any]] = []
        polish_notes: list[dict[str, Any]] = []
        back_notes: list[dict[str, Any]] = []
        for position, (chapter, pending) in enumerate(work, start=1):
            where = f" · chapter {position}/{len(work)}"
            text_segments = list(chapter.text_segments)
            segments = [s for s in text_segments if (s.target or "").strip()]
            if not segments:
                continue
            # Autofix locations are positions within text_segments, not Segment.index.
            positions = [text_segments.index(s) for s in segments]
            sources = [s.source for s in segments]
            targets = [s.target or "" for s in segments]
            sampled_pairs.extend(list(zip(sources, targets))[:8])
            completed: set[str] = set()
            if "self_revision" in pending:
                report(f"Quality pass · self revision{where}")
                try:
                    revised = agent.self_revise(
                        sources,
                        targets,
                        style=style,
                        glossary_terms=terms,
                        raise_on_failure=True,
                    )
                except Exception:
                    failed.append("self_revision")
                else:
                    for index, (before, after) in enumerate(zip(targets, revised)):
                        if after != before:
                            revision_notes.append(
                                {
                                    "chapter": chapter.index,
                                    "index": positions[index],
                                    "suggested": after,
                                }
                            )
                    completed.add("self_revision")
                done += 1
            if "final_polish" in pending:
                report(f"Quality pass · final polish{where}")
                try:
                    polished = agent.final_polish(
                        targets, style=style, glossary_terms=terms, raise_on_failure=True
                    )
                except Exception:
                    failed.append("final_polish")
                else:
                    for index, (before, after) in enumerate(zip(targets, polished)):
                        if after != before:
                            polish_notes.append(
                                {
                                    "chapter": chapter.index,
                                    "index": positions[index],
                                    "suggested": after,
                                }
                            )
                    completed.add("final_polish")
                done += 1
            if "chapter_selfcheck" in pending:
                report(f"Quality pass · chapter self-check{where}")
                try:
                    findings = agent.chapter_selfcheck(
                        sources, targets, glossary_terms=terms, raise_on_failure=True
                    )
                except Exception:
                    failed.append("chapter_selfcheck")
                else:
                    for finding in findings:
                        finding = dict(finding)
                        finding["chapter"] = chapter.index
                        local = finding.get("index")
                        if isinstance(local, int) and 0 <= local < len(positions):
                            finding["index"] = positions[local]
                        else:
                            finding.pop("index", None)
                        chapter_notes.append(finding)
                    completed.add("chapter_selfcheck")
                done += 1
            if "back_translation" in pending:
                report(f"Quality pass · back-translation{where}")
                try:
                    backs = agent.back_translate(targets[:4], raise_on_failure=True)
                except Exception:
                    failed.append("back_translation")
                else:
                    from .evaluation import back_translation_similarity

                    for position, (source, target, back) in enumerate(
                        zip(sources[:4], targets[:4], backs)
                    ):
                        back_notes.append(
                            {
                                "chapter": chapter.index,
                                "index": positions[position],
                                "source_preview": source[:80],
                                "back_preview": back[:80],
                                "score": round(back_translation_similarity(source, back), 4),
                            }
                        )
                    completed.add("back_translation")
                done += 1
            recorded[str(chapter.index)] = recorded.get(str(chapter.index), set()) | completed
        if editorial_pending:
            report("Quality pass · editorial notes")
            try:
                result["editorial_notes"] = agent.editorial_notes(
                    sampled_pairs[:24],
                    style=style,
                    book_synopsis=book_synopsis,
                    raise_on_failure=True,
                )
            except Exception:
                failed.append("editorial_pass")
            else:
                recorded["editorial_pass"] = {"editorial_pass"}
            done += 1
        if revision_notes:
            result["self_revision_notes"] = revision_notes
        if polish_notes:
            result["final_polish_notes"] = polish_notes
        if chapter_notes:
            result["chapter_selfcheck_findings"] = chapter_notes
        if back_notes:
            result["back_translation_notes"] = back_notes
        # The checkpoint is written even when no pass produced a note: otherwise a repeated run
        # would pay for every pass again, which is what "auto" turns on by default.
        analysis = store.load_analysis() or {}
        checkpoint = {key: sorted(value) for key, value in recorded.items() if value}
        if checkpoint:
            analysis["quality_pass_done"] = checkpoint
        if result:
            # Merge per key: a resumed run only covers the chapters it still owes, so replacing
            # the record wholesale would drop the notes the checkpoint already claims are done.
            merged = dict(analysis.get("quality_pass") or {})
            for key, value in result.items():
                if not isinstance(value, list):
                    merged[key] = value
                    continue
                existing = list(merged.get(key) or [])
                seen = {json.dumps(item, sort_keys=True, ensure_ascii=False) for item in existing}
                for item in value:
                    marker = json.dumps(item, sort_keys=True, ensure_ascii=False)
                    if marker not in seen:
                        existing.append(item)
                        seen.add(marker)
                merged[key] = existing
            analysis["quality_pass"] = merged
        if checkpoint or result:
            store.save_analysis(analysis)
            store.log_event(
                "quality_pass_finished",
                mode=cfg.quality_passes,
                coverage=coverage,
                chapters=len(work),
                failed=sorted(failed),
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
        # Reported last so a pause requested during the final pass cannot lose the recorded notes.
        report("Quality pass complete")
        return result
