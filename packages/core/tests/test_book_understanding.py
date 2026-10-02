"""Book-understanding completeness and recovery tests."""

from __future__ import annotations

import os
import tempfile
import unittest

from wenyi_core.config import Config
from wenyi_core.pipeline.orchestrator import Orchestrator
from wenyi_core.pipeline.preparation import _synopsis_complete

from .fake_llm import FakeClient, routing_handler

_DIGEST = "## Plot\n甲说话。\n## Characters\n甲\n## Foreshadowing\n无\n## Address\n无"


def _config(state_dir: str, **pipeline_overrides) -> Config:
    return Config.from_dict(
        {
            "language": {"source": "ja", "target": "zh"},
            "llm": {"preset": "fake"},
            "paths": {"state_dir": state_dir},
            "pipeline": {"review": False, "polish": False, **pipeline_overrides},
        }
    )


def _write_book(directory: str) -> str:
    txt = os.path.join(directory, "novel.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write("# 第一章\n\n" + "甲说了一句话。" * 40 + "\n\n# 第二章\n\n" + "乙又说了话。" * 40)
    return txt


def _handler(fail_ordinals: set[int], synopsis: str = "这是一段完整的全书概要。"):
    """Wrap routing_handler, failing digestion for the given call ordinals."""
    state = {"digest_calls": 0}

    def handler(messages, tier, json_mode):
        system = messages[0]["content"]
        if "chapter digest writer" in system:
            ordinal = state["digest_calls"]
            state["digest_calls"] += 1
            if ordinal in fail_ordinals:
                return ""
            return _DIGEST
        if "whole-book synopsis writer" in system:
            return synopsis
        return routing_handler(messages, tier, json_mode)

    return handler


class SynopsisCompleteTests(unittest.TestCase):
    def test_complete_requires_terminator(self):
        self.assertFalse(_synopsis_complete(""))
        self.assertFalse(_synopsis_complete("故事从第一章开始，主角名叫"))
        self.assertTrue(_synopsis_complete("故事结束了。"))


class DigestRequirementTests(unittest.TestCase):
    def test_failed_digest_blocks_translation_and_is_not_cached(self):
        with tempfile.TemporaryDirectory() as d:
            txt = _write_book(d)
            cfg = _config(os.path.join(d, "state"))
            orch = Orchestrator(cfg, client=FakeClient(handler=_handler({0})))

            with self.assertRaisesRegex(ValueError, "digests could not be generated"):
                orch.run(txt)

            # The failed chapter is not cached: a later run retries instead of reusing "".
            store = orch._preparation.locate_existing(txt)
            metas = [
                store.load_chapter(c["index"]).meta
                for c in store.load_manifest().get("chapters", [])
            ]
            self.assertTrue(any(not str(m.get("source_digest") or "").strip() for m in metas))
            store.close()

    def test_all_digests_succeed_then_translation_runs(self):
        with tempfile.TemporaryDirectory() as d:
            txt = _write_book(d)
            cfg = _config(os.path.join(d, "state"))
            orch = Orchestrator(cfg, client=FakeClient(handler=_handler(set())))

            store = orch.run(txt)

            metas = [
                store.load_chapter(c["index"]).meta
                for c in store.load_manifest().get("chapters", [])
            ]
            self.assertTrue(all(str(m.get("source_digest") or "").strip() for m in metas))
            store.close()


class SynopsisFailureTests(unittest.TestCase):
    def test_incomplete_synopsis_does_not_overwrite_existing(self):
        with tempfile.TemporaryDirectory() as d:
            txt = _write_book(d)
            cfg = _config(os.path.join(d, "state"))
            orch = Orchestrator(cfg, client=FakeClient(handler=_handler(set())))

            store = orch._preparation.prepare(txt)
            analysis = store.load_analysis() or {}
            analysis["book_synopsis"] = "旧的可用的概要。"
            analysis["book_synopsis_v"] = 3
            analysis["book_synopsis_gf"] = {}
            store.save_analysis(analysis)

            # A truncated regeneration must not replace the working synopsis.
            orch = Orchestrator(
                cfg, client=FakeClient(handler=_handler(set(), synopsis="这是一段被截断的概要"))
            )
            orch._preparation.ensure_understanding(store)

            self.assertEqual(
                (store.load_analysis() or {}).get("book_synopsis"),
                "旧的可用的概要。",
            )
            store.close()


if __name__ == "__main__":
    unittest.main()
