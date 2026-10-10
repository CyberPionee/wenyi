"""Preparation tests for language normalization and style-sample selection."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

import pytest
from wenyi_core.config import Config
from wenyi_core.llm.providers.fake import FakeClient
from wenyi_core.pipeline import preparation
from wenyi_core.pipeline.orchestrator import Orchestrator
from wenyi_core.pipeline.runstore import slugify, source_sha256
from wenyi_core.storage.file import FileStorage

from tests.fake_llm import routing_handler
from tests.pipeline_fixtures import fake_pipeline_config
from tests.sample_data import write_sample_epub, write_sample_txt


class TestLocateExistingStore(unittest.TestCase):
    def test_epub_locate_uses_peek_title_without_load_document(self):
        """Locate EPUB state through OPF title only, avoiding repeated full-book annotation."""
        with tempfile.TemporaryDirectory() as directory:
            epub = os.path.join(directory, "sample.epub")
            write_sample_epub(epub)
            digest = source_sha256(epub)
            # Use the same slug rule for the sample EPUB's OPF title as preparation does.
            store = FileStorage(
                os.path.join(directory, "state", slugify("サンプル小説"), "targets", "zh"),
            )
            store.save_manifest(
                {
                    "title": "サンプル小説",
                    "fmt": "epub",
                    "source_path": epub,
                    "source_sha256": digest,
                    "source_lang": "ja",
                    "target_lang": "zh",
                    "chapters": [],
                }
            )
            cfg = fake_pipeline_config(os.path.join(directory, "state"))
            orch = Orchestrator(cfg, client=FakeClient())

            with patch(
                "wenyi_core.pipeline.preparation.load_document",
                side_effect=AssertionError("locate 不应调用 load_document"),
            ):
                located = orch._preparation.locate_existing(epub)

            self.assertEqual(located.run_dir, store.run_dir)
            self.assertTrue(located.exists())


def _preparation_inputs(tmp_path, book_understanding=True):
    source = tmp_path / "book.txt"
    write_sample_txt(str(source))
    config = Config.from_dict(
        {
            "language": {"source": "ja", "target": "zh"},
            "llm": {"preset": "fake"},
            "pipeline": {
                "book_understanding": book_understanding,
                "prescan_concurrency": 2,
                "review": False,
                "polish": False,
            },
            "paths": {"state_dir": str(tmp_path / "state")},
        }
    )
    return source, config, FileStorage(str(tmp_path / "run"))


@pytest.mark.parametrize("entry_point", ["prepare", "prepare_for_translation", "run"])
@pytest.mark.parametrize("book_understanding", [False, True])
def test_initialization_analyses_style_before_any_chapter_prescan(
    tmp_path, entry_point, book_understanding
):
    source, config, store = _preparation_inputs(tmp_path, book_understanding)
    digest = "DIGEST-ONLY-CONTENT"
    ordering: list[str] = []

    def handler(messages, tier, json_mode):
        system = messages[0]["content"]
        if "chapter digest writer" in system:
            ordering.append("digest")
            return digest
        if "pre-translation analyst" in system:
            ordering.append("style")
            assert not store.exists(), "The initialization manifest must commit last"
            chapter = store.load_chapter(0)
            # This branch prescans during translation so digests can use the seeded terms,
            # so no digest exists yet and the analysis reads source samples.
            assert not chapter.meta.get("source_digest")
            assert digest not in messages[-1]["content"], "Style analysis reads source samples"
        return routing_handler(messages, tier, json_mode)

    client = FakeClient(handler=handler)
    getattr(Orchestrator(config, client=client, storage=store), entry_point)(str(source))
    assert ordering
    assert ordering[0] == "style"
    # `prepare` stops after initialization, so digests arrive only on the translating entry
    # points; the style analysis itself never sees a digest either way.
    if book_understanding and entry_point != "prepare":
        assert "digest" in ordering
    else:
        assert ordering == ["style"]


def test_prescan_failure_proceeds_without_digests_and_retries_them(tmp_path, monkeypatch):
    """A failing digest no longer blocks the book: resume must always make progress.

    Translation proceeds with an empty digest for the affected chapter, the failure is
    recorded, and nothing is cached so the next understanding run retries it.
    """
    source, config, store = _preparation_inputs(tmp_path)
    orchestrator = Orchestrator(config, client=FakeClient(handler=routing_handler), storage=store)
    store = orchestrator.prepare(str(source))

    def failing(messages, tier, json_mode):
        if "chapter digest writer" in messages[0]["content"]:
            return ""
        return routing_handler(messages, tier, json_mode)

    monkeypatch.setattr(preparation, "_DIGEST_RETRY_PAUSE_SECONDS", 0.0)
    Orchestrator(config, client=FakeClient(handler=failing), storage=store).prepare_for_translation(
        str(source)
    )
    chapters = [store.load_chapter(index) for index in (0, 1)]
    assert not any(chapter.meta.get("source_digest") for chapter in chapters)
    events = store.list_events(event_type="book_understanding_proceeded_without_digests")
    assert events, "the run must record that it continued without digests"
    # A failed prescan must not cache the failure: retrying succeeds and saves digests.
    client = FakeClient(handler=routing_handler)
    Orchestrator(config, client=client, storage=store).prepare_for_translation(str(source))
    assert all(store.load_chapter(index).meta.get("source_digest") for index in (0, 1))


def test_interrupted_prescan_resumes_without_repeating_prescan_or_style(tmp_path):
    source, config, store = _preparation_inputs(tmp_path)
    orchestrator = Orchestrator(config, client=FakeClient(handler=routing_handler), storage=store)
    store = orchestrator.prepare(str(source))

    first = {"count": 0}

    def interrupted(messages, tier, json_mode):
        if "chapter digest writer" in messages[0]["content"]:
            first["count"] += 1
            if first["count"] == 2:
                raise KeyboardInterrupt("during the prescan")
        return routing_handler(messages, tier, json_mode)

    with pytest.raises(KeyboardInterrupt, match="during the prescan"):
        Orchestrator(
            config, client=FakeClient(handler=interrupted), storage=store
        ).prepare_for_translation(str(source))

    resumed = FakeClient(handler=routing_handler)
    Orchestrator(config, client=resumed, storage=store).prepare_for_translation(str(source))
    # Only the chapter whose digest never landed is requested again, and the style
    # analysis is not repeated; the whole-book synopsis still has to be produced.
    operations = [call["operation"] for call in resumed.calls]
    assert operations.count("synopsis.chapter") == 1
    assert operations[-1] == "synopsis.book"
    assert "analysis.style" not in operations
    chapters = [store.load_chapter(index) for index in (0, 1)]
    assert all(chapter.meta.get("source_digest") for chapter in chapters)


def test_style_failure_leaves_initialization_uncommitted(tmp_path):
    source, config, store = _preparation_inputs(tmp_path)

    def handler(messages, tier, json_mode):
        if "pre-translation analyst" in messages[0]["content"]:
            raise RuntimeError("style analysis failed")
        return routing_handler(messages, tier, json_mode)

    with pytest.raises(RuntimeError, match="style analysis failed"):
        Orchestrator(config, client=FakeClient(handler=handler), storage=store).prepare(str(source))
    assert not store.exists()
    assert store.load_analysis() is None

    client = FakeClient(handler=routing_handler)
    Orchestrator(config, client=client, storage=store).prepare_for_translation(str(source))
    assert store.exists()
    assert store.load_analysis()["style_guide"]
    assert all(store.load_chapter(index).meta.get("source_digest") for index in (0, 1))
