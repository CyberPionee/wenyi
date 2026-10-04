"""Opt-in C-batch quality passes write analysis/events only."""

from __future__ import annotations

import os
import tempfile
import unittest

from wenyi_core.agents.quality_pass import QualityPassAgent
from wenyi_core.config import Config
from wenyi_core.llm.providers.fake import FakeClient
from wenyi_core.pipeline.orchestrator import Orchestrator

from tests.fake_llm import routing_handler
from tests.sample_data import write_sample_txt


def _cfg(state: str) -> Config:
    return Config.from_dict(
        {
            "language": {"source": "en", "target": "zh"},
            "llm": {"preset": "fake"},
            "paths": {"state_dir": state},
            "pipeline": {
                "review": False,
                "polish": False,
                "book_understanding": False,
                "annotation_alignment": False,
                "self_revision": False,
                "editorial_pass": False,
                "final_polish": False,
                "chapter_selfcheck": False,
                "back_translation": False,
            },
        }
    )


class TestQualityPassAgent(unittest.TestCase):
    def test_self_revise_and_final_polish_and_back(self):
        cfg = _cfg(tempfile.mkdtemp())
        agent = QualityPassAgent(FakeClient(handler=routing_handler), cfg)
        revised = agent.self_revise(["a", "b"], ["甲", "乙"], style="s")
        self.assertEqual(len(revised), 2)
        polished = agent.final_polish(["甲", "乙"])
        self.assertEqual(len(polished), 2)
        backs = agent.back_translate(["甲", "乙"])
        self.assertEqual(len(backs), 2)
        notes = agent.editorial_notes([("a", "甲")], style="s", book_synopsis="syn")
        self.assertTrue(notes)
        findings = agent.chapter_selfcheck(["a"], ["甲"])
        self.assertIsInstance(findings, list)

    def test_self_revise_count_mismatch_keeps_input(self):
        def handler(messages, tier, json_mode):
            return '{"revised":["only-one"]}'

        cfg = _cfg(tempfile.mkdtemp())
        agent = QualityPassAgent(FakeClient(handler=handler), cfg)
        self.assertEqual(agent.self_revise(["a", "b"], ["甲", "乙"]), ["甲", "乙"])


class TestQualityPassService(unittest.TestCase):
    def test_all_flags_off_make_no_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            txt = os.path.join(directory, "novel.txt")
            write_sample_txt(txt)
            cfg = _cfg(os.path.join(directory, "state"))
            client = FakeClient(handler=routing_handler)
            orch = Orchestrator(cfg, client=client)
            orch.run(txt)
            calls_before = len(client.calls)
            result = orch._quality_pass.run_after_translate(orch._preparation.locate_existing(txt))
            self.assertEqual(result, {})
            self.assertEqual(len(client.calls), calls_before)

    def test_enabled_passes_write_analysis_only(self):
        with tempfile.TemporaryDirectory() as directory:
            txt = os.path.join(directory, "novel.txt")
            write_sample_txt(txt)
            cfg = _cfg(os.path.join(directory, "state"))
            cfg.pipeline.self_revision = True
            cfg.pipeline.editorial_pass = True
            cfg.pipeline.final_polish = True
            cfg.pipeline.chapter_selfcheck = True
            cfg.pipeline.back_translation = True
            client = FakeClient(handler=routing_handler)
            orch = Orchestrator(cfg, client=client)
            store = orch.run(txt)
            chapter_before = store.load_chapter(0)
            targets_before = [s.target for s in chapter_before.text_segments]
            result = orch._quality_pass.run_after_translate(store)
            self.assertIn("editorial_notes", result)
            self.assertTrue(result.get("self_revision_notes") or result.get("final_polish_notes"))
            analysis = store.load_analysis() or {}
            self.assertIn("quality_pass", analysis)
            chapter_after = store.load_chapter(0)
            self.assertEqual(
                [s.target for s in chapter_after.text_segments],
                targets_before,
            )
            events = open(store.event_log_path, encoding="utf-8").read()
            self.assertIn("quality_pass_finished", events)


if __name__ == "__main__":
    unittest.main()
