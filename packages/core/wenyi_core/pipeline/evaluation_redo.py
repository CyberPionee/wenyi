"""Repair loop driven by machine-evaluation failures.

Reuses the existing Autofix shadow → verify → publish channel: failing evaluation
findings become Autofix issues, the fixer revises whole paragraphs, and publication
writes formal targets through the recoverable index. Nothing here invents a second
write path.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..llm.routing import inference_snapshot
from ..review.models import ReviewOutcome
from ..review.run_store import ReviewRunStore
from ..storage.protocol import Storage
from .annotations import AnnotationService
from .autofix_candidates import AutofixCandidateService
from .autofix_plan import prepare_identity, save_plan
from .autofix_publish import AutofixPublisher
from .docx_styles import DocxStyleService
from .evaluation import evaluation_low_score_issues, quality_pass_notes_to_issues

if TYPE_CHECKING:
    from .runtime import PipelineRuntime

ProgressFn = Callable[[int, int, str], None]


class EvaluationRedoService:
    """Turn failed machine-gate findings into one Autofix repair round."""

    def __init__(self, runtime: PipelineRuntime):
        self._runtime = runtime
        self._annotations = AnnotationService(runtime)
        self._publisher = AutofixPublisher(self._annotations, DocxStyleService(runtime))
        self._candidates = AutofixCandidateService(
            runtime.config, runtime.client, runtime.analyzer.style_brief
        )

    def _build_issues(self, store: Storage, evaluation: dict[str, Any]) -> list[dict[str, Any]]:
        pipeline = self._runtime.config.pipeline
        issues = evaluation_low_score_issues(
            evaluation,
            bt_min=float(getattr(pipeline, "bt_score_min", 0.45)),
            judge_min=float(getattr(pipeline, "judge_score_min", 3.5)),
        )
        analysis = store.load_analysis() or {}
        quality = analysis.get("quality_pass") or {}
        quality_issues = quality_pass_notes_to_issues(
            quality, bt_min=float(getattr(pipeline, "bt_score_min", 0.45))
        )
        # Merge without duplicating locations already covered by an evaluation issue.
        seen = {(issue["chapter"], issue["index"]) for issue in issues}
        for note in quality_issues:
            location = (note["chapter"], note["index"])
            if location in seen:
                continue
            issues.append(note)
            seen.add(location)
        return issues

    def repair_once(
        self,
        store: Storage,
        evaluation: dict[str, Any],
        *,
        progress: ProgressFn | None = None,
    ) -> dict[str, Any]:
        """Run one Autofix repair round for failing findings; return a small summary."""
        # An interrupted round whose plan is already saved resumes by publishing that plan:
        # planning (verification) is the expensive half and its plan is the durable artifact.
        resumed = self._resumable_round(store)
        if resumed is not None:
            debug, index = resumed
            self._publisher.apply(store, debug, index, progress=progress)
            failed = sum(
                1 for record in index.get("records", []) if record.get("status") == "failed"
            )
            summary = {
                "issue_count": len(index.get("records") or []),
                "published_segment_count": len(index.get("locations") or []),
                "failed_issue_count": failed,
                "review_id": debug.review_id,
                "resumed": True,
            }
            store.log_event(
                "evaluation_redo_finished",
                review_id=debug.review_id,
                resumed=True,
                issue_count=summary["issue_count"],
                published_segment_count=summary["published_segment_count"],
                failed_issue_count=failed,
            )
            return summary
        issues = self._build_issues(store, evaluation)
        summary: dict[str, Any] = {
            "issue_count": len(issues),
            "published_segment_count": 0,
            "failed_issue_count": 0,
            "review_id": None,
        }
        if not issues:
            summary["reason"] = "no_candidates"
            return summary

        manifest = store.load_manifest()
        chapters = [
            store.load_chapter(row["index"])
            for row in manifest.get("chapters", [])
            if isinstance(row.get("index"), int)
        ]
        analysis = store.load_analysis() or {}
        all_terms = store.all_terms() if hasattr(store, "all_terms") else []
        debug = ReviewRunStore(store.run_dir, storage=store, kind="evaluation-redo")
        debug.start(
            reviewed_content_digest="evaluation-redo",
            metadata={"kind": "evaluation_redo", "issue_count": len(issues)},
        )
        result = {
            "issues": issues,
            "changes": [],
            "summary": {"issue_count": len(issues), "change_count": 0},
        }
        outcome = ReviewOutcome(run_dir=debug.run_dir, result=result, usage={})
        candidates = self._candidates.prepare(
            chapters, analysis, outcome, all_terms, debug, progress=progress
        )
        index = dict(
            save_plan(
                chapters,
                candidates,
                prepare_identity(debug, self._runtime.llm_config),
                debug,
                outcome,
            )
        )
        self._publisher.apply(store, debug, index, progress=progress)
        failed = sum(1 for record in index.get("records", []) if record.get("status") == "failed")
        summary["published_segment_count"] = len(index.get("locations") or [])
        summary["failed_issue_count"] = failed
        summary["review_id"] = debug.review_id
        store.log_event(
            "evaluation_redo_finished",
            review_id=debug.review_id,
            issue_count=len(issues),
            published_segment_count=summary["published_segment_count"],
            failed_issue_count=failed,
        )
        return summary

    def _resumable_round(self, store: Storage) -> tuple[ReviewRunStore, dict[str, Any]] | None:
        """Return an interrupted redo round whose plan is already saved.

        The plan (locations + records, written before publishing) is the durable artifact of
        a round, and publishing it is idempotent: every location carries its baseline, so
        paragraphs already published or edited since are skipped. A pause between planning
        and publishing therefore resumes the round instead of re-running every verification
        call. A round whose planning models differ is not reused — it is re-verified with the
        current routes.
        """
        current = inference_snapshot(self._runtime.llm_config, ("autofix.verify", "autofix.fix"))
        names = sorted(
            {
                key.split("/")[1]
                for key in store.list_artifacts("reviews/")
                if len(key.split("/")) > 2 and key.split("/")[1].startswith("evaluation-redo-")
            },
            reverse=True,
        )
        for name in names:
            result = store.read_artifact(f"reviews/{name}/result.json")
            if not isinstance(result, dict) or not ReviewRunStore.is_resumable_status(
                result.get("status")
            ):
                continue
            index = store.read_artifact(f"reviews/{name}/autofix/index.json")
            if not isinstance(index, dict) or not index.get("locations"):
                continue
            if index.get("inference") != current:
                continue
            debug = ReviewRunStore._from_existing(
                os.path.join(store.run_dir, "reviews", name), name, storage=store
            )
            debug.log_event(
                "evaluation_redo_resumed",
                review_id=name,
                location_count=len(index["locations"]),
            )
            return debug, index
        return None
