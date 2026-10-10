"""Opt-in C-batch quality passes write analysis/events only."""

from __future__ import annotations

import os
import tempfile
import unittest

from wenyi_core.agents.quality_pass import QualityPassAgent
from wenyi_core.config import Config
from wenyi_core.llm.providers.fake import FakeClient
from wenyi_core.pipeline.orchestrator import Orchestrator
from wenyi_core.pipeline.quality_pass import _even_sample, _map_editorial_findings

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


class TestEditorialSampling(unittest.TestCase):
    def test_even_sample_spreads_across_the_book(self):
        """The sample covers the whole book, not just the opening chapters."""
        pairs = [(f"s{i}", f"t{i}") for i in range(100)]
        locations = [(i // 10, i) for i in range(100)]
        sampled, picked = _even_sample(pairs, locations)
        self.assertEqual(len(sampled), 24)
        self.assertEqual(picked[0], (0, 0))
        self.assertEqual(picked[-1], (9, 95))  # reaches the last chapter
        steps = [index for _, index in picked]
        self.assertEqual(steps, sorted(steps))
        self.assertGreater(steps[1] - steps[0], 3)  # evenly spaced, not packed at the head

    def test_even_sample_keeps_short_books_whole(self):
        pairs = [("s", "t")] * 5
        locations = [(0, i) for i in range(5)]
        sampled, picked = _even_sample(pairs, locations)
        self.assertEqual(sampled, pairs)
        self.assertEqual(picked, locations)

    def test_findings_map_sample_numbers_and_drop_outsiders(self):
        locations = [(2, 7), (4, 0), (6, 13)]
        mapped = _map_editorial_findings(
            [
                {"pair": 1, "detail": "repeat", "suggested": "fixed text"},
                {"pair": 9, "detail": "beyond sample", "suggested": "x"},  # dropped
                {"pair": -1, "detail": "negative", "suggested": "x"},  # dropped
                {"pair": 0, "detail": "no suggestion", "suggested": "  "},  # dropped
            ],
            locations,
        )
        self.assertEqual(
            mapped,
            [{"chapter": 4, "index": 0, "detail": "repeat", "suggested": "fixed text"}],
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
        payload = agent.editorial_notes([("a", "甲")], style="s", book_synopsis="syn")
        self.assertIn("notes", payload)
        self.assertIn("findings", payload)
        self.assertIsInstance(payload["notes"], list)
        self.assertIsInstance(payload["findings"], list)
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

    def test_a_failed_pass_is_not_recorded_as_done(self):
        """A provider failure must not be checkpointed as a completed pass, otherwise a resumed
        run keeps an empty result instead of retrying it."""
        with tempfile.TemporaryDirectory() as directory:
            txt = os.path.join(directory, "novel.txt")
            write_sample_txt(txt)
            cfg = _cfg(os.path.join(directory, "state"))
            orch = Orchestrator(cfg, client=FakeClient(handler=routing_handler))
            store = orch.run(txt)

            cfg.pipeline.chapter_selfcheck = True
            agent = orch._runtime.quality_pass

            def boom(*_args, **_kwargs):
                raise RuntimeError("provider down")

            agent.chapter_selfcheck = boom
            try:
                orch._quality_pass.run_after_translate(store)
            finally:
                del agent.chapter_selfcheck

            recorded = (store.load_analysis() or {}).get("quality_pass_done") or {}
            self.assertNotIn("chapter_selfcheck", recorded.get("0") or [])

    def test_a_failed_pass_is_recorded_and_skipped_on_the_next_run(self):
        """A repeatedly failing pass must not block resume.

        The failure lands in its own ledger so the pass is skipped next time; otherwise a
        pass that keeps failing (e.g. a truncating upstream model) re-runs for minutes on
        every resume and stalls every stage queued behind it.
        """
        with tempfile.TemporaryDirectory() as directory:
            txt = os.path.join(directory, "novel.txt")
            write_sample_txt(txt)
            cfg = _cfg(os.path.join(directory, "state"))
            orch = Orchestrator(cfg, client=FakeClient(handler=routing_handler))
            store = orch.run(txt)

            cfg.pipeline.chapter_selfcheck = True
            agent = orch._runtime.quality_pass
            calls = {"n": 0}

            def boom(*_args, **_kwargs):
                calls["n"] += 1
                raise RuntimeError("provider down")

            agent.chapter_selfcheck = boom
            try:
                orch._quality_pass.run_after_translate(store)
                analysis = store.load_analysis() or {}
                # Not a completed pass: the done ledger keeps its old meaning.
                self.assertNotIn(
                    "chapter_selfcheck", (analysis.get("quality_pass_done") or {}).get("0") or []
                )
                # The failure is recorded so the next run skips it instead of blocking.
                self.assertIn(
                    "chapter_selfcheck", (analysis.get("quality_pass_failed") or {}).get("0") or []
                )
                first = calls["n"]
                # One attempt per chapter that still owes the pass (the sample book has
                # more than one chapter).
                self.assertGreaterEqual(first, 1)
                self.assertIn(
                    "chapter_selfcheck", (analysis.get("quality_pass_failed") or {}).get("0") or []
                )
                orch._quality_pass.run_after_translate(store)
                self.assertEqual(calls["n"], first)  # skipped on resume
            finally:
                del agent.chapter_selfcheck

    def test_a_repeated_run_keeps_notes_from_an_earlier_run(self):
        """Merging per key must not drop the notes an earlier run already recorded."""
        with tempfile.TemporaryDirectory() as directory:
            txt = os.path.join(directory, "novel.txt")
            write_sample_txt(txt)
            cfg = _cfg(os.path.join(directory, "state"))
            cfg.pipeline.self_revision = True
            orch = Orchestrator(cfg, client=FakeClient(handler=routing_handler))
            store = orch.run(txt)

            analysis = store.load_analysis() or {}
            analysis["quality_pass"] = {
                "self_revision_notes": [{"chapter": 99, "index": 0, "suggested": "kept"}]
            }
            analysis.pop("quality_pass_done", None)
            store.save_analysis(analysis)

            orch._quality_pass.run_after_translate(store)

            merged = (store.load_analysis() or {}).get("quality_pass") or {}
            self.assertIn(
                {"chapter": 99, "index": 0, "suggested": "kept"},
                merged.get("self_revision_notes") or [],
            )


if __name__ == "__main__":
    unittest.main()
