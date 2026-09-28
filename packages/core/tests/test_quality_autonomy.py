"""Tests for residual sweep candidates, auto-lock events and opt-in pilot self-check."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from wenyi_core.config import Config
from wenyi_core.glossary.extractor import GlossaryExtractor, TranslatedSegmentEvidence
from wenyi_core.glossary.store import GlossaryStore, GlossaryTerm
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
            "segment": {"max_tokens_per_batch": 2000},
            "pipeline": {
                "review": False,
                "polish": False,
                "book_understanding": True,
                "annotation_alignment": False,
                "pilot": False,
            },
        }
    )


class TestTermAutoLock(unittest.TestCase):
    def test_extract_and_store_emits_auto_lock_callback(self):
        def handler(messages, tier, json_mode):
            system = messages[0]["content"]
            if "terminology" in system:
                return json.dumps(
                    {
                        "terms": [
                            {
                                "source": "Ann",
                                "target": "安",
                                "type": "person",
                                "gender": "female",
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            return "{}"

        cfg = _cfg(tempfile.mkdtemp())
        client = FakeClient(handler=handler)
        store = GlossaryStore(os.path.join(tempfile.mkdtemp(), "g.db"))
        try:
            locked: list[tuple[str, str]] = []
            summary = GlossaryExtractor(client, cfg).extract_and_store(
                store,
                "Ann met Ann again",
                "安又见了安",
                chapter=0,
                history=[
                    TranslatedSegmentEvidence(0, 0, "Ann met Ann again", "安又见了安"),
                ],
                before=(0, 1),
                source_corpus="Ann met Ann again",
                on_auto_lock=lambda source, target: locked.append((source, target)),
            )
            self.assertEqual(summary["auto_locked"], 1)
            self.assertEqual(locked, [("Ann", "安")])
            self.assertEqual(store.get_term("Ann").target, "安")
        finally:
            store.close()

    def test_auto_lock_never_overwrites_established_target(self):
        def handler(messages, tier, json_mode):
            system = messages[0]["content"]
            if "terminology" in system:
                return json.dumps(
                    {
                        "terms": [
                            {
                                "source": "Ann",
                                "target": "安娜",
                                "type": "person",
                                "gender": "female",
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            return "{}"

        cfg = _cfg(tempfile.mkdtemp())
        client = FakeClient(handler=handler)
        store = GlossaryStore(os.path.join(tempfile.mkdtemp(), "g.db"))
        try:
            store.upsert_term(GlossaryTerm(source="Ann", target="安", type="person"), chapter=0)
            locked: list[tuple[str, str]] = []
            GlossaryExtractor(client, cfg).extract_and_store(
                store,
                "Ann left",
                "安娜 left",
                chapter=1,
                history=[
                    TranslatedSegmentEvidence(0, 0, "Ann left", "安娜 left"),
                ],
                before=(1, 1),
                source_corpus="Ann left Ann",
                on_auto_lock=lambda source, target: locked.append((source, target)),
            )
            self.assertEqual(store.get_term("Ann").target, "安")
            for source, target in locked:
                self.assertEqual(source, "Ann")
                self.assertEqual(target, "安")
        finally:
            store.close()

    def test_auto_lock_requires_source_corpus_for_recurrence(self):
        def handler(messages, tier, json_mode):
            system = messages[0]["content"]
            if "terminology" in system:
                return json.dumps(
                    {
                        "terms": [
                            {
                                "source": "Ann",
                                "target": "安",
                                "type": "person",
                                "gender": "female",
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            return "{}"

        cfg = _cfg(tempfile.mkdtemp())
        client = FakeClient(handler=handler)
        store = GlossaryStore(os.path.join(tempfile.mkdtemp(), "g.db"))
        try:
            locked: list[tuple[str, str]] = []
            summary = GlossaryExtractor(client, cfg).extract_and_store(
                store,
                "Ann met Ann again",
                "安又见了安",
                chapter=0,
                history=[
                    TranslatedSegmentEvidence(0, 0, "Ann met Ann again", "安又见了安"),
                ],
                before=(0, 1),
                source_corpus=None,
                on_auto_lock=lambda source, target: locked.append((source, target)),
            )
            self.assertEqual(summary["auto_locked"], 0)
            self.assertEqual(locked, [])
        finally:
            store.close()


class TestPilotSelfCheck(unittest.TestCase):
    def test_pilot_writes_analysis_only_and_logs_events(self):
        with tempfile.TemporaryDirectory() as directory:
            txt = os.path.join(directory, "novel.txt")
            write_sample_txt(txt)
            cfg = _cfg(os.path.join(directory, "state"))
            cfg.pipeline.pilot = True
            client = FakeClient(handler=routing_handler)
            orch = Orchestrator(cfg, client=client)
            store = orch.prepare(txt)
            orch._preparation.run_pilot(store, synopsis="overview")
            analysis = store.load_analysis() or {}
            self.assertIn("pilot", analysis)
            self.assertEqual(analysis["pilot"]["chapter"], 0)
            events = open(store.event_log_path, encoding="utf-8").read()
            self.assertIn("pilot_selfcheck_started", events)
            self.assertTrue(
                "pilot_selfcheck_finished" in events or "pilot_selfcheck_degraded" in events
            )
            # Pilot must not publish chapter targets.
            chapter = store.load_chapter(0)
            self.assertTrue(all(segment.target is None for segment in chapter.text_segments))

    def test_pilot_default_off_skips_trial(self):
        with tempfile.TemporaryDirectory() as directory:
            txt = os.path.join(directory, "novel.txt")
            write_sample_txt(txt)
            cfg = _cfg(os.path.join(directory, "state"))
            self.assertFalse(cfg.pipeline.pilot)
            client = FakeClient(handler=routing_handler)
            orch = Orchestrator(cfg, client=client)
            store = orch.prepare(txt)
            events = open(store.event_log_path, encoding="utf-8").read()
            self.assertNotIn("pilot_selfcheck_started", events)

    def test_pilot_degrade_disables_polish(self):
        with tempfile.TemporaryDirectory() as directory:
            txt = os.path.join(directory, "novel.txt")
            with open(txt, "w", encoding="utf-8") as handle:
                handle.write("# One\n\nChapter 12 starts in 2024 and Ann left.\n")
            cfg = _cfg(os.path.join(directory, "state"))
            cfg.pipeline.pilot = True
            cfg.pipeline.polish = True

            def handler(messages, tier, json_mode):
                system = messages[0]["content"]
                if "literary translator" in system:
                    # Omit source numbers to force residual findings.
                    return '{"translations":["Chapter starts and left."]}'
                return routing_handler(messages, tier, json_mode)

            orch = Orchestrator(cfg, client=FakeClient(handler=handler))
            store = orch.prepare(txt)
            orch._preparation.run_pilot(store, synopsis="overview")
            self.assertFalse(cfg.pipeline.polish)
            analysis = store.load_analysis() or {}
            self.assertTrue(analysis.get("pilot", {}).get("polish_disabled"))
            events = open(store.event_log_path, encoding="utf-8").read()
            self.assertIn("pilot_selfcheck_degraded", events)


if __name__ == "__main__":
    unittest.main()
