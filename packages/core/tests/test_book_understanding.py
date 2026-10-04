"""Book-understanding completeness and recovery tests."""

from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from wenyi_core.agents.synopsis import Synopsizer
from wenyi_core.config import Config
from wenyi_core.ingest.models import Chapter, Document, Segment
from wenyi_core.llm.providers.fake import FakeClient
from wenyi_core.pipeline.orchestrator import Orchestrator
from wenyi_core.pipeline.preparation import PreparationService, _synopsis_complete
from wenyi_core.storage.file import FileStorage

from .fake_llm import routing_handler


def _service(tmp_path, chapters=None):
    source = tmp_path / "source.txt"
    source.write_text("Temporary source text.", encoding="utf-8")
    store = FileStorage(str(tmp_path / "state"))
    store.init_from_document(
        Document(
            source_lang="en",
            target_lang="zh",
            fmt="text",
            source_path=str(source),
            chapters=chapters
            or [
                Chapter(index=0, segments=[Segment(index=0, source="Opening.")]),
                Chapter(index=1, segments=[Segment(index=0, source="Ending.")]),
            ],
        )
    )
    store.save_analysis({"style_guide": "Restrained."})
    runtime = SimpleNamespace(
        config=Config.from_dict({"llm": {"preset": "fake"}}),
        synopsizer=Mock(),
        analyzer=Mock(),
    )
    runtime.synopsizer.digest_chapter.return_value = "Chapter digest."
    runtime.synopsizer.book_synopsis.return_value = "Whole-book synopsis."
    runtime.analyzer.style_brief.return_value = "Restrained."
    return PreparationService(runtime), store, runtime


def test_legacy_state_without_policy_stamps_resumes_without_regeneration(tmp_path):
    """State written before policy tracking keeps its digests and synopsis on resume.

    Those artifacts already carry the glossary snapshot this run validates, and the upgrade
    leaves their prompts untouched, so the first resumed run reuses them and stamps them.
    """
    service, store, runtime = _service(tmp_path)
    service.ensure_understanding(store)
    assert runtime.synopsizer.digest_chapter.call_count == 2

    for index in (0, 1):
        chapter = store.load_chapter(index)
        chapter.meta.pop("source_digest_policy", None)
        store.save_chapter(chapter)
    analysis = store.load_analysis()
    analysis.pop("book_synopsis_inputs", None)
    store.save_analysis(analysis)

    runtime.synopsizer.digest_chapter.reset_mock()
    assert service.ensure_understanding(store) == "Whole-book synopsis."
    assert runtime.synopsizer.digest_chapter.call_count == 0
    assert runtime.synopsizer.book_synopsis.call_count == 1  # First call above, not again.


def test_changed_policy_regenerates_an_existing_digest(tmp_path):
    """Once a digest carries a stamp, a different prompt revision makes it stale."""
    service, store, runtime = _service(tmp_path)
    service.ensure_understanding(store)
    chapter = store.load_chapter(0)
    chapter.meta["source_digest_policy"] = "some-other-revision"
    store.save_chapter(chapter)

    runtime.synopsizer.digest_chapter.reset_mock()
    service.ensure_understanding(store)
    assert runtime.synopsizer.digest_chapter.call_count == 1
    assert store.load_chapter(0).meta["source_digest_policy"] != "some-other-revision"


def test_book_synopsis_transport_failure_has_empty_fallback():
    def handler(messages, tier, json_mode):
        raise TimeoutError("provider unavailable")

    assert Synopsizer(FakeClient(handler=handler), Config()).book_synopsis(["Digest."], "") == ""


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


def test_failed_regeneration_keeps_existing_analysis_but_does_not_inject_stale_synopsis(tmp_path):
    service, store, runtime = _service(tmp_path)
    service.ensure_understanding(store)
    runtime.analyzer.style_brief.return_value = "New style."
    runtime.synopsizer.book_synopsis.return_value = ""
    assert service.ensure_understanding(store) == ""
    assert store.load_analysis()["book_synopsis"] == "Whole-book synopsis."
    runtime.synopsizer.book_synopsis.return_value = "Revised synopsis."
    assert service.ensure_understanding(store) == "Revised synopsis."


def test_legacy_truncated_digest_and_unverified_synopsis_are_regenerated(tmp_path):
    service, store, runtime = _service(tmp_path)
    chapter = store.load_chapter(0)
    chapter.meta["source_digest"] = "Legacy unfinished sentence"
    store.save_chapter(chapter)
    chapter = store.load_chapter(1)
    chapter.meta["source_digest"] = "Legacy complete digest."
    store.save_chapter(chapter)
    store.save_analysis({"style_guide": "Restrained.", "book_synopsis": "Legacy synopsis."})
    assert service.ensure_understanding(store) == "Whole-book synopsis."
    assert runtime.synopsizer.digest_chapter.call_count == 2
    # This branch prescans after the style analysis, so digests receive the seeded glossary
    # terms that keep character names aligned with the established mapping.
    assert sorted(call_.args[0] for call_ in runtime.synopsizer.digest_chapter.call_args_list) == [
        "Ending.",
        "Opening.",
    ]
    synopsis_args = runtime.synopsizer.book_synopsis.call_args.args
    assert synopsis_args[:2] == (["Chapter digest.", "Chapter digest."], "Restrained.")
