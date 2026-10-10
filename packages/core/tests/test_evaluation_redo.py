"""Tests for evaluation-driven Autofix repair rounds."""

from __future__ import annotations

import json
import os
import re
import tempfile
import unittest

from wenyi_core.config import Config
from wenyi_core.glossary.store import GlossaryTerm
from wenyi_core.ingest.models import Chapter, Document, Segment
from wenyi_core.pipeline.evaluation_redo import EvaluationRedoService
from wenyi_core.pipeline.finalization import ReportService
from wenyi_core.pipeline.runtime import PipelineRuntime
from wenyi_core.storage.file import FileStorage

from .fake_llm import FakeClient, routing_handler


def _config(state_dir: str) -> Config:
    return Config.from_dict(
        {
            "language": {"source": "ja", "target": "zh"},
            "llm": {"preset": "fake"},
            "paths": {"state_dir": state_dir},
            "pipeline": {"review": False, "polish": False, "review_autofix": True},
        }
    )


def _store(directory: str) -> FileStorage:
    store = FileStorage(os.path.join(directory, "state"))
    doc = Document(
        title="T",
        fmt="text",
        source_lang="ja",
        target_lang="zh",
        chapters=[
            Chapter(
                index=0,
                title="c",
                segments=[
                    Segment(index=0, source="ドルフィン・ホテルへ行く。", target="去了那家旅馆。"),
                    Segment(index=1, source="普通の文。", target="普通的一句。"),
                ],
            )
        ],
    )
    store.begin_initialization("b" * 64)
    manifest = store.stage_document(doc, source_hash="b" * 64)
    manifest["initialized"] = True
    store.save_manifest(manifest)
    store.upsert_term(GlossaryTerm(source="ドルフィン・ホテル", target="海豚酒店", type="place"))
    return store


def _evaluation_with_drift() -> dict:
    return {
        "l2": {
            "checked": 1,
            "drifted": 1,
            "consistency_rate": 0.0,
            "items": [
                {
                    "chapter": 0,
                    "index": 0,
                    "source_term": "ドルフィン・ホテル",
                    "expected_target": "海豚酒店",
                    "missing_targets": ["海豚酒店"],
                }
            ],
        },
        "back_translation": [],
        "judge_scores": [],
        "machine_gate": {"passed": False, "blocking": True},
    }


def _confirming_handler(messages, tier, json_mode):
    """Confirm every agent candidate so the fixer can revise, and return a corrected paragraph."""
    system = messages[0]["content"]
    user = messages[-1]["content"]
    if "evidence-based review agent" in system:
        ids = re.findall(r'"candidate_id"\s*:\s*"([^"]*)"', user)
        decisions = [
            {
                "candidate_id": cid,
                "verdict": "confirmed",
                "detail": "确认问题",
                "suggestion": "使用术语表译法",
                "reason": "",
                "consistency": {},
                "evidence_refs": [],
            }
            for cid in ids
        ]
        return json.dumps(
            {"action": "final", "decisions": decisions, "new_issues": [], "complete": True},
            ensure_ascii=False,
        )
    if "cautious revision editor" in system:
        ref = re.search(r"segment_ref:\s*(\S+)", user)
        digest = re.search(r"before_hash:\s*([0-9a-f]{64})", user)
        ids = re.findall(r'"issue_id"\s*:\s*"([^"]*)"', user) or ["eval_l2:0:0:0"]
        return json.dumps(
            {
                "segment_ref": ref.group(1) if ref else "",
                "before_hash": digest.group(1) if digest else "0" * 64,
                "issue_ids": ids,
                "replacement": "去了海豚酒店。",
                "complete": True,
            },
            ensure_ascii=False,
        )
    return routing_handler(messages, tier, json_mode)


def _dismissing_handler(messages, tier, json_mode):
    """Reject every agent candidate, so the round verifies and publishes nothing."""
    system = messages[0]["content"]
    user = messages[-1]["content"]
    if "evidence-based review agent" in system:
        ids = re.findall(r'"candidate_id"\s*:\s*"([^"]*)"', user)
        decisions = [
            {
                "candidate_id": cid,
                "verdict": "dismissed",
                "detail": "译文无误",
                "suggestion": "",
                "reason": "原文此处本就没有术语漂移",
                "consistency": {},
                "evidence_refs": [],
            }
            for cid in ids
        ]
        return json.dumps(
            {"action": "final", "decisions": decisions, "new_issues": [], "complete": True},
            ensure_ascii=False,
        )
    return routing_handler(messages, tier, json_mode)


class RedoBudgetTests(unittest.TestCase):
    def test_spent_redo_rounds_survive_a_rerun(self):
        """A paused report step continues its remaining rounds instead of restarting the budget."""
        with tempfile.TemporaryDirectory() as d:
            store = _store(d)
            try:
                manifest = store.load_manifest()
                for row in manifest["chapters"]:
                    row["status"] = "done"
                store.save_manifest(manifest)
                config = _config(os.path.join(d, "state"))
                config.pipeline.max_auto_redo_rounds = 1
                service = ReportService(PipelineRuntime(config, client=FakeClient(routing_handler)))
                identity = service._redo_identity(
                    store.load_manifest(),
                    max_rounds=1,
                    tier=config.pipeline.autonomy_tier,
                )
                service._record_redo_rounds(store, identity, 1)

                service.build_and_save(store, store)

                # The budget for this source was already spent, so nothing ran again.
                self.assertEqual(store.list_events(event_type="evaluation_redo_round"), [])
                exhausted = store.list_events(event_type="evaluation_redo_exhausted")
                self.assertEqual(len(exhausted), 1)
                self.assertEqual(exhausted[0]["rounds"], 1)
                self.assertEqual(exhausted[0]["max_rounds"], 1)
            finally:
                store.close()

    def test_a_fruitless_round_closes_the_budget(self):
        """A round that dismissed every candidate must not let a later run start the next one.

        The verifier judged text that has not changed, so a resume can only reach the same
        verdict. Recording only the round just spent let it start the next round and pay for a
        decision already made.
        """
        with tempfile.TemporaryDirectory() as d:
            store = _store(d)
            try:
                manifest = store.load_manifest()
                for row in manifest["chapters"]:
                    row["status"] = "done"
                store.save_manifest(manifest)
                config = _config(os.path.join(d, "state"))
                config.pipeline.max_auto_redo_rounds = 2
                service = ReportService(
                    PipelineRuntime(config, client=FakeClient(_dismissing_handler))
                )

                service.build_and_save(store, store)
                rounds = store.list_events(event_type="evaluation_redo_round")
                self.assertEqual([row["round"] for row in rounds], [1])
                self.assertEqual({row["published_segment_count"] for row in rounds}, {0})
                self.assertEqual({row["failed_issue_count"] for row in rounds}, {0})

                service.build_and_save(store, store)
                self.assertEqual(
                    [row["round"] for row in store.list_events(event_type="evaluation_redo_round")],
                    [1],
                )
                exhausted = store.list_events(event_type="evaluation_redo_exhausted")
                self.assertEqual(len(exhausted), 1)
                self.assertEqual(exhausted[0]["max_rounds"], 2)
            finally:
                store.close()

    def test_a_failed_round_leaves_the_budget_for_a_retry(self):
        """A round whose verification failed is a model failure, not a decision about the text.

        Closing the budget on it would make one transient agent failure burn every remaining
        repair round for that source.
        """
        with tempfile.TemporaryDirectory() as d:
            store = _store(d)
            try:
                manifest = store.load_manifest()
                for row in manifest["chapters"]:
                    row["status"] = "done"
                store.save_manifest(manifest)
                config = _config(os.path.join(d, "state"))
                config.pipeline.max_auto_redo_rounds = 2
                service = ReportService(PipelineRuntime(config, client=FakeClient(routing_handler)))

                service.build_and_save(store, store)
                rounds = store.list_events(event_type="evaluation_redo_round")
                self.assertEqual([row["round"] for row in rounds], [1])
                self.assertEqual({row["published_segment_count"] for row in rounds}, {0})
                self.assertEqual({row["failed_issue_count"] for row in rounds}, {1})
                self.assertEqual(store.list_events(event_type="evaluation_redo_exhausted"), [])

                service.build_and_save(store, store)
                self.assertEqual(
                    [row["round"] for row in store.list_events(event_type="evaluation_redo_round")],
                    [1, 2],
                )
            finally:
                store.close()


class RedoCandidateTests(unittest.TestCase):
    def test_build_issues_maps_l2_drift_and_quality_notes(self):
        with tempfile.TemporaryDirectory() as d:
            store = _store(d)
            try:
                analysis = store.load_analysis() or {}
                analysis["quality_pass"] = {
                    "self_revision_notes": [
                        {"chapter": 0, "index": 1, "suggested": "更好的句子。"},
                        {"chapter": 0, "index": 0, "suggested": "重复位置应被跳过。"},
                    ]
                }
                store.save_analysis(analysis)
                runtime = PipelineRuntime(
                    _config(os.path.join(d, "state2")),
                    client=FakeClient(handler=_confirming_handler),
                )
                redo = EvaluationRedoService(runtime)
                issues = redo._build_issues(store, _evaluation_with_drift())
                self.assertEqual({issue["index"] for issue in issues}, {0, 1})
                self.assertTrue(any(issue["issue_key"].startswith("eval_l2") for issue in issues))
            finally:
                store.close()

    def test_no_candidates_short_circuits(self):
        with tempfile.TemporaryDirectory() as d:
            store = _store(d)
            try:
                runtime = PipelineRuntime(
                    _config(os.path.join(d, "state2")),
                    client=FakeClient(handler=_confirming_handler),
                )
                redo = EvaluationRedoService(runtime)
                evaluation = {
                    "l2": {"checked": 0, "drifted": 0, "consistency_rate": 1.0, "items": []},
                    "back_translation": [],
                    "judge_scores": [],
                    "machine_gate": {"passed": True, "blocking": False},
                }
                summary = redo.repair_once(store, evaluation)
                self.assertEqual(summary["issue_count"], 0)
                self.assertEqual(summary["published_segment_count"], 0)
                self.assertEqual(summary["reason"], "no_candidates")
            finally:
                store.close()

    def test_repairs_drift_through_autofix_channel(self):
        with tempfile.TemporaryDirectory() as d:
            store = _store(d)
            try:
                runtime = PipelineRuntime(
                    _config(os.path.join(d, "state2")),
                    client=FakeClient(handler=_confirming_handler),
                )
                redo = EvaluationRedoService(runtime)
                summary = redo.repair_once(store, _evaluation_with_drift())
                self.assertEqual(summary["issue_count"], 1)
                self.assertIsNotNone(summary["review_id"])
                self.assertEqual(store.load_chapter(0).segments[0].target, "去了海豚酒店。")
            finally:
                store.close()


class RedoRoundResumeTests(unittest.TestCase):
    def test_an_interrupted_round_resumes_from_its_saved_plan(self):
        """A round interrupted after planning publishes that plan instead of re-verifying.

        Planning (verification) is the expensive half and the saved plan is its durable
        artifact, so a pause between planning and publishing must resume the round rather
        than re-run every verification call.
        """
        from wenyi_core.llm.routing import inference_snapshot
        from wenyi_core.review.models import text_hash
        from wenyi_core.review.run_store import ReviewRunStore

        with tempfile.TemporaryDirectory() as d:
            store = _store(d)
            try:
                manifest = store.load_manifest()
                for row in manifest["chapters"]:
                    row["status"] = "done"
                store.save_manifest(manifest)
                config = _config(os.path.join(d, "state"))
                runtime = PipelineRuntime(config, client=FakeClient(_confirming_handler))
                service = EvaluationRedoService(runtime)

                # A redo round paused after its plan was written: running status + a plan.
                debug = ReviewRunStore(store.run_dir, storage=store, kind="evaluation-redo")
                debug.start(
                    reviewed_content_digest="evaluation-redo",
                    metadata={"kind": "evaluation_redo"},
                )
                inference = inference_snapshot(
                    runtime.llm_config, ("autofix.verify", "autofix.fix")
                )
                before_text = "去了那家旅馆。"
                debug.write_json(
                    "autofix/index.json",
                    {
                        "version": 1,
                        "inference": inference,
                        "review_id": debug.review_id,
                        "status": "applying",
                        "records": [],
                        "locations": [
                            {
                                "chapter": 0,
                                "index": 0,
                                "segment_ref": "ch0:text0:seg0",
                                "before": before_text,
                                "before_hash": text_hash(before_text),
                                "target": "去了海豚酒店。",
                                "target_hash": text_hash("去了海豚酒店。"),
                                "record_ids": [],
                                "status": "pending",
                                "alignment_status": "pending",
                            }
                        ],
                    },
                )

                resumed = service._resumable_round(store)
                self.assertIsNotNone(resumed)
                resumed_debug, index = resumed
                self.assertEqual(resumed_debug.review_id, debug.review_id)
                self.assertEqual(len(index["locations"]), 1)

                before_calls = len(runtime.client.calls)
                summary = service.repair_once(store, {})

                self.assertTrue(summary.get("resumed"))
                self.assertEqual(summary["published_segment_count"], 1)
                # No verification was re-run: the saved plan was published directly.
                self.assertEqual(len(runtime.client.calls), before_calls)
                self.assertEqual(store.load_chapter(0).segments[0].target, "去了海豚酒店。")
            finally:
                store.close()

    def test_a_round_with_changed_planning_models_is_not_reused(self):
        """A plan verified with other routes is re-verified, never published blindly."""
        from wenyi_core.review.run_store import ReviewRunStore

        with tempfile.TemporaryDirectory() as d:
            store = _store(d)
            try:
                config = _config(os.path.join(d, "state"))
                service = EvaluationRedoService(
                    PipelineRuntime(config, client=FakeClient(routing_handler))
                )
                debug = ReviewRunStore(store.run_dir, storage=store, kind="evaluation-redo")
                debug.start(
                    reviewed_content_digest="evaluation-redo",
                    metadata={"kind": "evaluation_redo"},
                )
                debug.write_json(
                    "autofix/index.json",
                    {
                        "version": 1,
                        "inference": {"autofix.verify": {"model": "other"}},
                        "review_id": debug.review_id,
                        "status": "applying",
                        "records": [],
                        "locations": [{"chapter": 0, "index": 0}],
                    },
                )

                self.assertIsNone(service._resumable_round(store))
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
