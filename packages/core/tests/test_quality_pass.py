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
                # These tests cover the individual switches, which apply in "manual" mode; the
                # default "auto" mode derives the passes and risk-gates chapters instead.
                "quality_passes": "manual",
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

    def test_system_prompts_do_not_vary_with_the_paragraph_count(self):
        """A per-call count in the system message destroys the request's prefix cache.

        The system message is the request's first block, so interpolating the chapter's
        paragraph count there made every call's prompt differ from its first tokens. The stable
        style and glossary block after it (tens of thousands of tokens) could then only be
        reused when two chapters happened to share the same count.
        """
        cfg = _cfg(tempfile.mkdtemp())
        agent = QualityPassAgent(FakeClient(handler=routing_handler), cfg)
        calls: list[dict] = []
        agent.client = FakeClient(
            handler=lambda messages, tier, json_mode: (
                calls.append({"system": messages[0]["content"], "user": messages[-1]["content"]})
                or routing_handler(messages, tier, json_mode)
            )
        )
        agent.self_revise(["a", "b"], ["甲", "乙"])
        agent.final_polish(["甲", "乙"])
        agent.chapter_selfcheck(["a", "b"], ["甲", "乙"])
        agent.self_revise(["a", "b", "c"], ["甲", "乙", "丙"])
        agent.final_polish(["甲", "乙", "丙"])
        agent.chapter_selfcheck(["a", "b", "c"], ["甲", "乙", "丙"])

        by_stage = {
            "self_revision": (0, 3),
            "final_polish": (1, 4),
            "chapter_selfcheck": (2, 5),
        }
        for stage, (short, long) in by_stage.items():
            self.assertEqual(
                calls[short]["system"],
                calls[long]["system"],
                f"{stage} system prompt changes with the paragraph count",
            )
            self.assertNotEqual(calls[short]["user"], calls[long]["user"])
            # The count still reaches the model, from the user message.
            self.assertIn("3", calls[long]["user"])


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

    def test_each_pass_reports_its_own_progress(self):
        """Translation saturates the word-count bar, so every pass reports its own position."""
        with tempfile.TemporaryDirectory() as directory:
            txt = os.path.join(directory, "novel.txt")
            write_sample_txt(txt)
            cfg = _cfg(os.path.join(directory, "state"))
            cfg.pipeline.self_revision = True
            cfg.pipeline.final_polish = True
            orch = Orchestrator(cfg, client=FakeClient(handler=routing_handler))

            seen: list[tuple[int, int, str]] = []
            orch.run(
                txt,
                progress=lambda done, total, label: seen.append((done, total, label)),
            )

            labels = [label for _done, _total, label in seen]
            passes = [item for item in seen if item[2].startswith("Quality pass")]
            self.assertTrue(
                any(label.startswith("Quality pass · self revision") for label in labels)
            )
            self.assertTrue(
                any(label.startswith("Quality pass · final polish") for label in labels)
            )
            # Disabled passes report nothing at all.
            self.assertFalse(any("editorial" in label for label in labels))
            self.assertFalse(any("back-translation" in label for label in labels))
            self.assertEqual(passes[-1][2], "Quality pass complete")
            totals = {total for _done, total, _label in passes}
            self.assertEqual(len(totals), 1)
            (total,) = totals
            self.assertEqual([done for done, _t, _l in passes], list(range(total + 1)))

    def test_a_completed_pass_is_not_repeated(self):
        """The notes are a checkpoint: a repeated run must not spend the same calls again."""
        with tempfile.TemporaryDirectory() as directory:
            txt = os.path.join(directory, "novel.txt")
            write_sample_txt(txt)
            cfg = _cfg(os.path.join(directory, "state"))
            cfg.pipeline.chapter_selfcheck = True
            client = FakeClient(handler=routing_handler)
            orch = Orchestrator(cfg, client=client)
            store = orch.run(txt)

            orch._quality_pass.run_after_translate(store)
            calls_after_first = len(client.calls)
            self.assertIn("quality_pass_done", store.load_analysis() or {})

            self.assertEqual(orch._quality_pass.run_after_translate(store), {})
            self.assertEqual(len(client.calls), calls_after_first)

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
            # The run itself executes the passes; a later call is a checkpoint no-op.
            analysis = store.load_analysis() or {}
            result = analysis.get("quality_pass") or {}
            self.assertIn("editorial_notes", result)
            self.assertTrue(result.get("self_revision_notes") or result.get("final_polish_notes"))
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
