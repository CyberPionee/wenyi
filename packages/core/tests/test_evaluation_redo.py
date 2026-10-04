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


if __name__ == "__main__":
    unittest.main()
