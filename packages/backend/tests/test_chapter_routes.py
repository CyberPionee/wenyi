"""Chapter queue routing delegates durable identity and rollback to start_job."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException
from wenyi_backend.routers import chapters
from wenyi_core.config import Config
from wenyi_core.ingest.models import Chapter, Segment
from wenyi_core.storage.file import FileStorage


def _project(**overrides) -> dict:
    return {"id": "project-1", "fmt": "text", "source_lang": "ja", "target_lang": "zh", **overrides}


def test_chapter_payload_returns_a_display_copy_without_touching_stored_targets(
    tmp_path, monkeypatch
):
    store = FileStorage(str(tmp_path / "book"))
    store.save_chapter(
        Chapter(
            index=0,
            segments=[
                Segment(index=0, source="原文", target="「你好」"),
                Segment(index=1, source="ない", target=None),
            ],
        )
    )
    monkeypatch.setattr(
        chapters,
        "effective_config",
        lambda project: Config.from_dict({"language": {"target": "zh"}}),
    )

    payload = chapters.chapter_payload(_project(), store, 0)

    assert [segment["target"] for segment in payload["segments"]] == ["「你好」", None]
    assert [segment["display_target"] for segment in payload["segments"]] == ["“你好”", None]
    assert [segment.target for segment in store.load_chapter(0).text_segments] == ["「你好」", None]


def test_chapter_payload_keeps_stored_targets_for_other_target_languages(tmp_path, monkeypatch):
    store = FileStorage(str(tmp_path / "book"))
    store.save_chapter(
        Chapter(index=0, segments=[Segment(index=0, source="原文", target="「你好」")])
    )
    monkeypatch.setattr(
        chapters,
        "effective_config",
        lambda project: Config.from_dict({"language": {"target": "en"}}),
    )

    payload = chapters.chapter_payload(_project(target_lang="en"), store, 0)

    assert payload["segments"][0]["display_target"] == "「你好」"


def test_translate_chapter_enqueues_only_requested_chapter(monkeypatch):
    enqueued = []
    monkeypatch.setattr(chapters, "require_project", lambda pid: {"id": pid, "fmt": "text"})
    monkeypatch.setattr(
        chapters.dal, "chapter_summaries", lambda pid: [{"index": 2, "status": "pending"}]
    )

    async def start(pid, kind, *, params):
        enqueued.append((pid, kind, params))
        return {"job_id": "chapter-job", "project_id": pid, "kind": kind}

    monkeypatch.setattr(chapters, "start_job", start)
    result = asyncio.run(chapters.translate_chapter("project-1", 2))
    assert enqueued == [("project-1", "chapter_translation", {"chapter_index": 2})]
    assert result["kind"] == "chapter_translation"


@pytest.mark.parametrize("entries,code", [([], 404), ([{"index": 0, "status": "done"}], 409)])
def test_translate_chapter_rejects_missing_or_completed_chapter(monkeypatch, entries, code):
    monkeypatch.setattr(chapters, "require_project", lambda pid: {"id": pid, "fmt": "text"})
    monkeypatch.setattr(chapters.dal, "chapter_summaries", lambda pid: entries)
    with pytest.raises(HTTPException) as error:
        asyncio.run(chapters.translate_chapter("project-1", 0))
    assert error.value.status_code == code


def test_translate_chapter_propagates_queue_failure(monkeypatch):
    monkeypatch.setattr(chapters, "require_project", lambda pid: {"id": pid, "fmt": "text"})
    monkeypatch.setattr(
        chapters.dal, "chapter_summaries", lambda pid: [{"index": 1, "status": "pending"}]
    )

    async def fail(*args, **kwargs):
        raise HTTPException(503, "queue unavailable")

    monkeypatch.setattr(chapters, "start_job", fail)
    with pytest.raises(HTTPException) as error:
        asyncio.run(chapters.translate_chapter("project-1", 1))
    assert error.value.status_code == 503
