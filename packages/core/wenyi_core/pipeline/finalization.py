"""Report and assembly finalization services.
ReportService owns glossary lifetime, build_report, report.json and related events.
AssemblyService exports monolingual/bilingual products from live state or immutable
snapshots and forwards format options.
Standalone assembly avoids the long run lock: capture a snapshot under the assembly/state
locks, release the short state lock before rendering, and validate source hashes before and
after. Full-workflow assembly uses live state under its existing run lock plus the assembly
lock to serialize output writers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from ..glossary.store import GlossaryStore
from ..storage.protocol import Storage
from .language_policies import persist_plan
from .runstore import source_sha256

if TYPE_CHECKING:
    from .runtime import PipelineRuntime

ProgressFn = Callable[[int, int, str], None]


class ReportService:
    """Domain service for glossary lifetime and report generation."""

    def __init__(self, runtime: PipelineRuntime):
        self._runtime = runtime

    @contextmanager
    def glossary_scope(
        self, store: Storage, needed: bool
    ) -> Iterator[Storage | GlossaryStore | None]:
        """Borrow the injected glossary; its lifetime belongs to the caller."""
        yield store if needed else None

    def _run_evaluation(
        self,
        store: Storage,
        glossary: Storage | GlossaryStore,
        *,
        strict: bool,
    ) -> tuple[dict[str, Any], Any]:
        """Run one L0-L3 evaluation pass and return its payload plus the result object."""
        from ..assemble.report import build_report
        from .evaluation import EvaluationService
        from .tuning import describe_tuning, evaluation_policy, history_entries

        pipeline = self._runtime.config.pipeline
        pipeline_map = pipeline.model_dump()
        pre_report = build_report(store, glossary, strict_auto_qa=strict)
        # Recorded runs may move a threshold between its tier floor and the configured cap.
        history = history_entries(store.read_artifact("evaluation_history.json") or [])
        policy = evaluation_policy(pipeline_map, tier=str(pipeline.autonomy_tier), history=history)
        service = EvaluationService(
            store,
            risk_sample_ratio=policy["risk_sample_ratio"],
            judge_sample_ratio=policy["judge_sample_ratio"],
            bt_score_min=policy["bt_score_min"],
            judge_score_min=policy["judge_score_min"],
            l2_min_consistency=policy["l2_min_consistency"],
            block_on_l0_only=bool(policy["block_on_l0_only"]),
            risk_back_translation=bool(pipeline.risk_back_translation),
            quality_judge=bool(pipeline.quality_judge),
        )
        terms = glossary.all_terms() if hasattr(glossary, "all_terms") else []
        agent = self._runtime.quality_pass

        def _back(targets: list[str]) -> list[str]:
            return agent.back_translate(targets)

        def _judge(pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
            style = self._runtime.analyzer.style_brief(store.load_analysis() or {})
            first = agent.quality_judge(pairs, style=style)
            if not bool(getattr(pipeline, "quality_judge_dual", False)):
                return first
            # Dual judging reduces single-rater noise by averaging two independent passes.
            second = agent.quality_judge(pairs, style=style)
            merged: list[dict[str, Any]] = []
            for position in range(min(len(first), len(second), len(pairs))):
                left = first[position] if isinstance(first[position], dict) else {}
                right = second[position] if isinstance(second[position], dict) else {}
                scores = [
                    float(value)
                    for value in (left.get("score"), right.get("score"))
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                ]
                record = dict(left)
                if scores:
                    record["score"] = round(sum(scores) / len(scores), 2)
                record["judge_passes"] = len(scores)
                merged.append(record)
            return merged

        evaluation = service.run(
            l0=pre_report.get("auto_qa") or {},
            terms=terms,
            back_translate=_back if pipeline.risk_back_translation else None,
            judge=_judge if pipeline.quality_judge else None,
        )
        payload = evaluation.to_dict()
        payload["tuning"] = describe_tuning(
            pipeline=pipeline_map,
            run_plan=self._runtime.run_tuning,
            evaluation=policy,
        )
        store.log_event(
            "evaluation_finished",
            passed=bool((evaluation.machine_gate or {}).get("passed")),
            blocking=bool((evaluation.machine_gate or {}).get("blocking")),
            risk_count=len(evaluation.risk_segments),
            bt_count=len(evaluation.back_translation),
            judge_count=len(evaluation.judge_scores),
        )
        return payload, evaluation

    @staticmethod
    def _record_evaluation_trend(store: Storage, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Append a compact summary to the evaluation history and return the recent tail.

        The history is a rolling list so the report can show whether the machine gate is
        improving without asking anyone to diff runs by hand.
        """
        from datetime import datetime

        gate = payload.get("machine_gate") or {}
        entry = {
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            "passed": bool(gate.get("passed")),
            "blocking": bool(gate.get("blocking")),
            "tier": gate.get("tier") or gate.get("block_on_l0_only"),
            "l2_consistency_rate": gate.get("l2_consistency_rate"),
            "bt_low_count": gate.get("bt_low_count"),
            "judge_avg": gate.get("judge_avg"),
            "l0_residual_finding_count": gate.get("l0_residual_finding_count"),
            # Distribution and bar per run, so later runs can calibrate the thresholds.
            "bt_sample_count": gate.get("bt_sample_count"),
            "bt_p10": gate.get("bt_p10"),
            "bt_score_min": gate.get("bt_score_min"),
            "judge_sample_count": gate.get("judge_sample_count"),
            "judge_p10": gate.get("judge_p10"),
            "judge_score_min": gate.get("judge_score_min"),
        }
        history = store.read_artifact("evaluation_history.json")
        if not isinstance(history, list):
            history = []
        history.append(entry)
        history = history[-20:]
        store.write_artifact("evaluation_history.json", history)
        return history

    def build_and_save(
        self,
        store: Storage,
        glossary: Storage | GlossaryStore,
        *,
        progress: ProgressFn | None = None,
    ) -> dict[str, Any]:
        """Generate and persist report.json and record the corresponding event."""
        from ..assemble.report import build_report
        from .evaluation_redo import EvaluationRedoService

        if progress:
            progress(0, 0, "Generating report…")
        strict = bool(getattr(self._runtime.config.pipeline, "auto_qa_strict", False))
        pipeline = self._runtime.config.pipeline
        evaluation_payload: dict[str, Any] | None = None
        redo_summary: dict[str, Any] = {"rounds": 0, "published_segment_count": 0}
        if getattr(pipeline, "evaluation_enabled", True):
            if progress:
                progress(0, 0, "Running machine evaluation…")
            evaluation_payload, evaluation = self._run_evaluation(store, glossary, strict=strict)
            # Autonomous redo: repair failing findings through the Autofix channel and
            # re-evaluate. Publishes only when review_autofix is allowed; otherwise the
            # gate just reports.
            max_rounds = int(getattr(pipeline, "max_auto_redo_rounds", 0) or 0)
            if (
                max_rounds > 0
                and getattr(pipeline, "review_autofix", True)
                and evaluation_payload.get("machine_gate", {}).get("passed") is False
            ):
                redo = EvaluationRedoService(self._runtime)
                for round_number in range(1, max_rounds + 1):
                    if progress:
                        progress(0, 0, f"Automatic revision round {round_number}")
                    summary = redo.repair_once(store, evaluation_payload, progress=progress)
                    redo_summary["rounds"] = round_number
                    redo_summary["published_segment_count"] += int(
                        summary.get("published_segment_count") or 0
                    )
                    store.log_event(
                        "evaluation_redo_round",
                        round=round_number,
                        **{key: value for key, value in summary.items() if key != "reason"},
                    )
                    if summary.get("published_segment_count", 0) == 0:
                        break
                    evaluation_payload, evaluation = self._run_evaluation(
                        store, glossary, strict=strict
                    )
                    if evaluation_payload.get("machine_gate", {}).get("passed"):
                        break
                store.log_event(
                    "evaluation_redo_finished",
                    rounds=redo_summary["rounds"],
                    published_segment_count=redo_summary["published_segment_count"],
                    passed=bool(evaluation_payload.get("machine_gate", {}).get("passed")),
                )
        if evaluation_payload is not None:
            evaluation_payload["auto_redo"] = redo_summary
            evaluation_payload["history"] = self._record_evaluation_trend(store, evaluation_payload)
        report = build_report(
            store,
            glossary,
            strict_auto_qa=strict,
            evaluation=evaluation_payload,
        )
        assert report is not None
        store.save_report(report)
        store.log_event("report_saved", artifact="report.json")
        auto_qa = report.get("auto_qa") or {}
        store.log_event(
            "auto_qa_finished",
            passed=bool(auto_qa.get("passed")),
            blocking=bool(auto_qa.get("blocking")),
            empty_target_count=int(auto_qa.get("empty_target_count") or 0),
            open_conflict_count=int(auto_qa.get("open_conflict_count") or 0),
            residual_finding_count=int(auto_qa.get("residual_finding_count") or 0),
            open_issue_count=int(auto_qa.get("open_issue_count") or 0),
        )
        return report


class AssemblyService:
    """Domain service for live-state and read-only snapshot exports."""

    def __init__(self, runtime: PipelineRuntime):
        self._runtime = runtime

    def assemble_outputs(
        self,
        store: Storage,
        *,
        input_path: str,
        progress: ProgressFn | None,
        out_format: str,
        out_path: str | None,
        pdf_engine: str,
        policy_store: Storage,
    ) -> list[str]:
        """Generate every configured artifact from live state or a read-only snapshot."""
        from ..assemble.policy import export_plan
        from ..assemble.writer import assemble
        from ..assemble.writer_common import bilingual_out_path

        if getattr(self._runtime.config.pipeline, "auto_qa_strict", False):
            from ..assemble.report import build_report

            report = build_report(store, store, strict_auto_qa=True)
            qa = report.get("auto_qa") or {}
            saved = store.load_report() or {}
            gate = saved.get("machine_gate") or {}
            if qa.get("blocking") or gate.get("blocking"):
                raise ValueError(
                    "auto_qa_strict is enabled and residual or evaluation findings remain; "
                    "resolve empty targets, glossary conflicts, open issues or low evaluation "
                    "scores before export"
                )
        if progress:
            progress(0, 0, "Assembling translation…")
        out_cfg = self._runtime.config.output
        do_mono, do_bilingual = out_cfg.mono, out_cfg.bilingual
        if not do_mono and not do_bilingual:
            do_mono = True
        plans = {}
        for bilingual in ([False] if do_mono else []) + ([True] if do_bilingual else []):
            plans[bilingual] = export_plan(
                store,
                out_format,
                pdf_engine=pdf_engine,
                punctuation_normalize=out_cfg.punctuation_normalize,
                bilingual=bilingual,
                order=out_cfg.bilingual_order if bilingual else "target_first",
                preserve_source_style=out_cfg.bilingual_preserve_source_style
                if bilingual
                else False,
                about_page=out_cfg.about_page,
            )
        for plan in plans.values():
            persist_plan(policy_store, plan)

        outputs: list[str] = []
        if do_mono:
            outputs.append(
                assemble(
                    store,
                    input_path,
                    out_path=out_path,
                    out_format=out_format,
                    bilingual=False,
                    about_page=out_cfg.about_page,
                    pdf_engine=pdf_engine,
                    babeldoc_timeout=self._runtime.config.pipeline.babeldoc_timeout,
                    punctuation_normalize=out_cfg.punctuation_normalize,
                    language_policy=plans[False],
                )
            )
        if do_bilingual:
            bi_out_path = bilingual_out_path(out_path) if out_path else None
            outputs.append(
                assemble(
                    store,
                    input_path,
                    out_path=bi_out_path,
                    out_format=out_format,
                    bilingual=True,
                    order=out_cfg.bilingual_order,
                    preserve_source_style=out_cfg.bilingual_preserve_source_style,
                    about_page=out_cfg.about_page,
                    pdf_engine=pdf_engine,
                    babeldoc_timeout=self._runtime.config.pipeline.babeldoc_timeout,
                    punctuation_normalize=out_cfg.punctuation_normalize,
                    language_policy=plans[True],
                )
            )
        return outputs

    def assemble_live(
        self,
        store: Storage,
        *,
        input_path: str,
        progress: ProgressFn | None,
        out_format: str | None,
        out_path: str | None,
        pdf_engine: str,
    ) -> list[str]:
        """Export under the book run lock, adding the assembly lock to serialize output
        writers.
        """
        from ..assemble.writer_common import default_output_format

        with store.assemble_lock():
            # Export rereads the source template; validate before and after to detect replacement during the run.
            self._runtime.ensure_store_source(store, input_path)
            if out_format is None:
                out_format = default_output_format(store.load_manifest())
            outputs = self.assemble_outputs(
                store,
                input_path=input_path,
                progress=progress,
                out_format=out_format,
                out_path=out_path,
                pdf_engine=pdf_engine,
                policy_store=store,
            )
            self._runtime.ensure_store_source(store, input_path)
        self._runtime.log_event(store, "assembled", outputs=outputs, out_format=out_format)
        return outputs

    def assemble_snapshot(
        self,
        store: Storage,
        *,
        input_path: str,
        progress: ProgressFn | None,
        out_format: str | None,
        out_path: str | None,
        pdf_engine: str,
    ) -> list[str]:
        """Capture an immutable snapshot under the assembly lock and validate source hashes
        around rendering.
        """
        from ..assemble.writer_common import default_output_format

        with store.assemble_lock():
            snapshot = store.create_export_snapshot(actual_sha256=source_sha256(input_path))
            self._runtime.apply_manifest_languages(snapshot.load_manifest())
            if out_format is None:
                out_format = default_output_format(snapshot.load_manifest())

            # The source may change while waiting for another export; validate again immediately before rendering.
            self._runtime.ensure_store_source(store, input_path)
            outputs = self.assemble_outputs(
                snapshot,
                input_path=input_path,
                progress=progress,
                out_format=out_format,
                out_path=out_path,
                pdf_engine=pdf_engine,
                policy_store=store,
            )
            # The source template is an export input; validate afterward so mid-render replacement cannot succeed.
            self._runtime.ensure_store_source(store, input_path)
        self._runtime.log_event(store, "assembled", outputs=outputs, out_format=out_format)
        return outputs
