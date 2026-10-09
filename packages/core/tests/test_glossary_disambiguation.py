"""Offline same-target collision judgement tests."""

from __future__ import annotations

import json
import re
from pathlib import Path

from wenyi_core.config import Config
from wenyi_core.glossary.store import GlossaryTerm
from wenyi_core.ingest.models import Chapter, Segment
from wenyi_core.llm.providers.fake import FakeClient
from wenyi_core.pipeline.glossary_disambiguation import _collision_id
from wenyi_core.pipeline.orchestrator import Orchestrator
from wenyi_core.pipeline.runstore import STATUS_DONE
from wenyi_core.storage.file import FileStorage

COLLISION_ID = re.compile(r'"collision_id":\s*"([^"]+)"')


def _config(state_dir: str, **pipeline) -> Config:
    return Config.from_dict(
        {
            "language": {"source": "ja", "target": "zh"},
            "llm": {
                "preset": "fake",
                "models": {"default_strong": {"provider": "default", "model": "p"}},
            },
            "pipeline": {"review_concurrency": 1, **pipeline},
            "output": {"punctuation_normalize": False},
            "paths": {"state_dir": state_dir},
        }
    )


def _store(directory: str) -> FileStorage:
    """Two spellings of one hotel, both mapped to a single target."""
    store = FileStorage(str(Path(directory, "state", "book")))
    store.save_chapter(
        Chapter(
            index=0,
            title="第一章",
            segments=[
                Segment(index=0, source="いるかホテルへ行く", target="去海豚宾馆。"),
                Segment(index=1, source="いるかホテルは古い", target="海豚宾馆很旧。"),
                Segment(index=2, source="ドルフィン・ホテルは白い", target="海豚宾馆是白色的。"),
            ],
        )
    )
    store.save_manifest(
        {
            "title": "book",
            "source_lang": "ja",
            "target_lang": "zh",
            "source_sha256": "0" * 64,
            "chapters": [{"index": 0, "status": STATUS_DONE}],
        }
    )
    store.upsert_term(
        GlossaryTerm(source="いるかホテル", target="海豚宾馆", type="place"), chapter=0
    )
    store.upsert_term(
        GlossaryTerm(source="ドルフィン・ホテル", target="海豚宾馆", type="place"), chapter=0
    )
    return store


def _client(status: str, *, same_entity: bool | None, needs_distinction: bool | None) -> FakeClient:
    """Reply as the disambiguation agent does, echoing the supplied collision id."""

    def handler(messages, _tier, _json_mode):
        payload = json.dumps(
            {
                "action": "final",
                "collision_id": COLLISION_ID.search(messages[-1]["content"]).group(1),
                "status": status,
                "same_entity": same_entity,
                "needs_distinction": needs_distinction,
                "reason": "Both spellings refer to the same hotel throughout the passages.",
                "evidence_refs": [],
                "complete": True,
            },
            ensure_ascii=False,
        )
        return payload

    return FakeClient(handler)


def _run(orchestrator: Orchestrator, store: FileStorage) -> dict:
    return orchestrator._glossary_disambiguation.run(store)


def test_a_shared_entity_is_noted_and_the_target_stays_untouched(tmp_path):
    """The verdict lands in the note; the rendering this book uses is never rewritten."""
    store = _store(str(tmp_path))
    orchestrator = Orchestrator(
        _config(str(tmp_path)),
        client=_client("judged", same_entity=True, needs_distinction=False),
        storage=store,
    )

    summary = _run(orchestrator, store)

    assert summary["collisions"] == 1
    assert summary["judged"] == 1
    assert summary["notes_written"] == 2
    assert store.get_term("いるかホテル").target == "海豚宾馆"
    assert store.get_term("ドルフィン・ホテル").target == "海豚宾馆"
    note = store.get_term("いるかホテル").note
    assert "[target-disambiguation]" in note
    assert "ドルフィン・ホテル" in note
    assert "shared target is kept" in note
    assert store.read_artifact("glossary-disambiguation/judgements.json")["status"] == "completed"
    events = store.list_events(event_type="glossary_target_disambiguated")
    assert len(events) == 2
    assert events[0]["same_entity"] is True
    assert events[0]["needs_distinction"] is False


def test_an_unpreserved_distinction_is_recorded_as_such(tmp_path):
    """same_entity with needs_distinction says the shared target dropped the difference."""
    store = _store(str(tmp_path))
    orchestrator = Orchestrator(
        _config(str(tmp_path)),
        client=_client("judged", same_entity=True, needs_distinction=True),
        storage=store,
    )

    _run(orchestrator, store)

    note = store.get_term("ドルフィン・ホテル").note
    assert "source distinction is not preserved" in note
    # Still no rewrite: the note informs later passes, the operator decides the split.
    assert store.get_term("ドルフィン・ホテル").target == "海豚宾馆"


def test_distinct_entities_sharing_a_target_are_noted_as_distinct(tmp_path):
    store = _store(str(tmp_path))
    orchestrator = Orchestrator(
        _config(str(tmp_path)),
        client=_client("judged", same_entity=False, needs_distinction=False),
        storage=store,
    )

    _run(orchestrator, store)

    note = store.get_term("いるかホテル").note
    assert "Distinct from" in note
    assert store.get_term("いるかホテル").target == "海豚宾馆"


def test_an_unresolved_collision_leaves_the_note_alone(tmp_path):
    """A collision the passages cannot decide stays open for a human."""
    store = _store(str(tmp_path))
    orchestrator = Orchestrator(
        _config(str(tmp_path)),
        client=_client("unresolved", same_entity=None, needs_distinction=None),
        storage=store,
    )

    summary = _run(orchestrator, store)

    assert summary["judged"] == 0
    assert summary["unresolved"] == 1
    assert summary["notes_written"] == 0
    assert store.get_term("いるかホテル").note == ""
    index = store.read_artifact("glossary-disambiguation/judgements.json")
    assert index["status"] == "completed"
    assert index["unresolved"]


def test_disambiguation_can_be_disabled(tmp_path):
    store = _store(str(tmp_path))
    client = _client("judged", same_entity=True, needs_distinction=False)
    orchestrator = Orchestrator(
        _config(str(tmp_path), glossary_target_disambiguation=False),
        client=client,
        storage=store,
    )

    summary = _run(orchestrator, store)

    assert summary["reason"] == "disabled"
    assert client.calls == []
    assert store.get_term("いるかホテル").note == ""


def test_a_recorded_verdict_resumes_without_asking_the_model_again(tmp_path):
    """An interrupted run finishes from its own index rather than paying for another call."""
    store = _store(str(tmp_path))
    store.write_artifact(
        "glossary-disambiguation/judgements.json",
        {
            "status": "judged",
            "judgements": [
                {
                    "collision_id": "collision-recorded",
                    "target": "海豚宾馆",
                    "status": "judged",
                    "same_entity": True,
                    "needs_distinction": False,
                    "reason": "recorded",
                    "evidence_refs": [],
                    "sources": ["いるかホテル", "ドルフィン・ホテル"],
                }
            ],
            "unresolved": [],
        },
    )
    client = _client("judged", same_entity=True, needs_distinction=False)
    orchestrator = Orchestrator(_config(str(tmp_path)), client=client, storage=store)

    summary = orchestrator._glossary_disambiguation.resume_pending(store)

    assert summary is not None
    assert summary["resumed"] is True
    assert summary["notes_written"] == 2
    assert client.calls == []
    assert "[target-disambiguation]" in store.get_term("いるかホテル").note


def test_a_judged_collision_is_not_judged_twice(tmp_path):
    """A kept verdict id skips an already-settled group without paying for another call."""
    store = _store(str(tmp_path))
    real_id = _collision_id("海豚宾馆", ["ドルフィン・ホテル", "いるかホテル"])
    store.write_artifact(
        "glossary-disambiguation/judgements.json",
        {
            "status": "completed",
            "judgements": [],
            "unresolved": [{"collision_id": real_id}],
        },
    )
    client = _client("judged", same_entity=True, needs_distinction=False)
    orchestrator = Orchestrator(_config(str(tmp_path)), client=client, storage=store)

    groups = orchestrator._glossary_disambiguation._collision_groups(store)

    assert groups == []
    assert client.calls == []
