"""Tests for residual sweep candidates and auto-lock events."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from wenyi_core.agents.synopsis import _collapse_to_sentence, _looks_non_story
from wenyi_core.config import Config
from wenyi_core.glossary.extractor import GlossaryExtractor, TranslatedSegmentEvidence
from wenyi_core.glossary.store import GlossaryStore, GlossaryTerm
from wenyi_core.llm.providers.fake import FakeClient


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


class TestNonStoryDigest(unittest.TestCase):
    def test_short_text_is_non_story(self):
        self.assertTrue(_looks_non_story("TO MY MOTHER"))

    def test_copyright_markers_are_non_story(self):
        self.assertTrue(_looks_non_story("Copyright 2019 Little, Brown and Company. All rights reserved."))

    def test_long_narrative_is_story(self):
        text = "Holden walked down the street and thought about his life at Pencey. " * 20
        self.assertFalse(_looks_non_story(text))

    def test_collapse_strips_headings_keeps_first_body_line(self):
        result = _collapse_to_sentence("## Plot\n版权页。\n## Characters\n无角色出场。")
        self.assertEqual(result, "版权页。")

    def test_collapse_plain_text_passthrough(self):
        self.assertEqual(_collapse_to_sentence("题献给作者母亲。"), "题献给作者母亲。")

    def test_collapse_empty_returns_empty(self):
        self.assertEqual(_collapse_to_sentence(""), "")


if __name__ == "__main__":
    unittest.main()
