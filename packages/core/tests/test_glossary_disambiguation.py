"""Offline same-target collision judgement tests."""

from __future__ import annotations

import json
import re
from pathlib import Path

from wenyi_core.config import Config
from wenyi_core.glossary.store import GlossaryTerm
from wenyi_core.ingest.models import Chapter, Segment
from wenyi_core.llm.providers.fake import FakeClient
from wenyi_core.pipeline.glossary_disambiguation import (
    _MAX_GROUPS_PER_RUN,
    _MAX_UNRESOLVED_ATTEMPTS,
    _collision_id,
)
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


def _client(
    status: str, *, same_entity: bool | None, renderings: dict[str, str] | None
) -> FakeClient:
    """Reply as the disambiguation agent does, echoing the supplied collision id."""

    def handler(messages, _tier, _json_mode):
        payload = json.dumps(
            {
                "action": "final",
                "collision_id": COLLISION_ID.search(messages[-1]["content"]).group(1),
                "status": status,
                "same_entity": same_entity,
                "renderings": [
                    {"source": source, "target": wording}
                    for source, wording in (renderings or {}).items()
                ],
                "reason": "Both spellings refer to the same hotel throughout the passages.",
                "evidence_refs": [],
                "complete": True,
            },
            ensure_ascii=False,
        )
        return payload

    return FakeClient(handler)


def _targets(store: FileStorage) -> list[str]:
    return [segment.target for segment in store.load_chapter(0).text_segments]


def _run(orchestrator: Orchestrator, store: FileStorage) -> dict:
    return orchestrator._glossary_disambiguation.run(store)


def test_a_judged_split_changes_the_term_and_the_passages(tmp_path):
    """A source judged distinct moves to its own wording, in the glossary and in the text."""
    store = _store(str(tmp_path))
    orchestrator = Orchestrator(
        _config(str(tmp_path)),
        client=_client(
            "judged",
            same_entity=False,
            renderings={"いるかホテル": "海豚宾馆", "ドルフィン・ホテル": "海豚旅店"},
        ),
        storage=store,
    )

    summary = _run(orchestrator, store)

    assert summary["collisions"] == 1
    assert summary["judged"] == 1
    assert summary["renderings_applied"] == 1
    # The glossary moves only for the source the judge split off.
    assert store.get_term("いるかホテル").target == "海豚宾馆"
    assert store.get_term("ドルフィン・ホテル").target == "海豚旅店"
    # The passage that mentions that source is rewritten; passages of the other stay.
    assert _targets(store) == ["去海豚宾馆。", "海豚宾馆很旧。", "海豚旅店是白色的。"]
    events = store.list_events(event_type="glossary_target_disambiguated")
    assert len(events) == 2
    split = next(event for event in events if event["source"] == "ドルフィン・ホテル")
    assert split["changed"] is True
    assert split["old_target"] == "海豚宾馆"
    assert split["new_target"] == "海豚旅店"
    assert "[target-disambiguation]" in store.get_term("ドルフィン・ホテル").note


def test_a_unified_group_keeps_one_rendering_and_changes_nothing(tmp_path):
    """same_entity with identical renderings is a no-op on the text."""
    store = _store(str(tmp_path))
    orchestrator = Orchestrator(
        _config(str(tmp_path)),
        client=_client(
            "judged",
            same_entity=True,
            renderings={"いるかホテル": "海豚宾馆", "ドルフィン・ホテル": "海豚宾馆"},
        ),
        storage=store,
    )

    summary = _run(orchestrator, store)

    assert summary["renderings_applied"] == 0
    assert store.get_term("いるかホテル").target == "海豚宾馆"
    assert store.get_term("ドルフィン・ホテル").target == "海豚宾馆"
    assert _targets(store) == ["去海豚宾馆。", "海豚宾馆很旧。", "海豚宾馆是白色的。"]
    assert "unified rendering kept" in store.get_term("いるかホテル").note


def test_an_unresolved_collision_waits_and_stays_retryable(tmp_path):
    """A collision the passages cannot decide writes nothing and is retried next chapter."""
    store = _store(str(tmp_path))
    orchestrator = Orchestrator(
        _config(str(tmp_path)),
        client=_client("unresolved", same_entity=None, renderings=None),
        storage=store,
    )

    summary = _run(orchestrator, store)

    assert summary["judged"] == 0
    assert summary["unresolved"] == 1
    assert summary["renderings_applied"] == 0
    assert store.get_term("いるかホテル").note == ""
    # Unresolved ids never block a retry: the group is still detected next time.
    groups = orchestrator._glossary_disambiguation._collision_groups(store)
    assert len(groups) == 1


def test_disambiguation_can_be_disabled(tmp_path):
    store = _store(str(tmp_path))
    client = _client("judged", same_entity=False, renderings={"ドルフィン・ホテル": "海豚旅店"})
    orchestrator = Orchestrator(
        _config(str(tmp_path), glossary_target_disambiguation=False),
        client=client,
        storage=store,
    )

    summary = _run(orchestrator, store)

    assert summary["reason"] == "disabled"
    assert client.calls == []
    assert store.get_term("ドルフィン・ホテル").target == "海豚宾馆"
    assert _targets(store)[2] == "海豚宾馆是白色的。"


def test_a_recorded_verdict_resumes_without_asking_the_model_again(tmp_path):
    """An interrupted run applies its recorded renderings without paying for another call."""
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
                    "same_entity": False,
                    "renderings": [
                        {"source": "いるかホテル", "target": "海豚宾馆"},
                        {"source": "ドルフィン・ホテル", "target": "海豚旅店"},
                    ],
                    "reason": "recorded",
                    "evidence_refs": [],
                    "sources": ["いるかホテル", "ドルフィン・ホテル"],
                }
            ],
            "unresolved": [],
        },
    )
    client = _client("judged", same_entity=False, renderings={"ドルフィン・ホテル": "海豚旅店"})
    orchestrator = Orchestrator(_config(str(tmp_path)), client=client, storage=store)

    summary = orchestrator._glossary_disambiguation.resume_pending(store)

    assert summary is not None
    assert summary["resumed"] is True
    assert summary["renderings_applied"] == 1
    assert client.calls == []
    assert store.get_term("ドルフィン・ホテル").target == "海豚旅店"
    assert _targets(store)[2] == "海豚旅店是白色的。"


def test_a_completed_run_is_not_rejudged_on_the_next_chapter(tmp_path):
    """The verdict ids stay in the completed index so a later run cannot repeat the calls."""
    store = _store(str(tmp_path))
    client = _client(
        "judged",
        same_entity=True,
        renderings={"いるかホテル": "海豚宾馆", "ドルフィン・ホテル": "海豚宾馆"},
    )
    orchestrator = Orchestrator(_config(str(tmp_path)), client=client, storage=store)

    first = _run(orchestrator, store)
    assert first["judged"] == 1
    spent = len(client.calls)

    second = _run(orchestrator, store)

    assert second["collisions"] == 0
    assert len(client.calls) == spent


def test_a_kept_verdict_id_skips_the_group(tmp_path):
    """A stored judged id keeps the next run from re-detecting the settled group."""
    store = _store(str(tmp_path))
    real_id = _collision_id("海豚宾馆", ["ドルフィン・ホテル", "いるかホテル"])
    store.write_artifact(
        "glossary-disambiguation/judgements.json",
        {"status": "completed", "judgements": [{"collision_id": real_id}], "unresolved": []},
    )
    client = _client("judged", same_entity=True, renderings=None)
    orchestrator = Orchestrator(_config(str(tmp_path)), client=client, storage=store)

    groups = orchestrator._glossary_disambiguation._collision_groups(store)

    assert groups == []
    assert client.calls == []


def test_judged_ids_accumulate_across_runs(tmp_path):
    """Regression: a group judged in an earlier run is never judged again by a later one.

    Each run used to overwrite the index with only its own batch, so ids judged chapters ago
    fell out and the next run re-judged them, burning tokens on settled collisions.
    """
    store = _store(str(tmp_path))
    first_client = _client(
        "judged",
        same_entity=True,
        renderings={"いるかホテル": "海豚宾馆", "ドルフィン・ホテル": "海豚宾馆"},
    )
    first_orch = Orchestrator(_config(str(tmp_path)), client=first_client, storage=store)
    first = _run(first_orch, store)
    assert first["judged"] == 1
    first_id = _collision_id("海豚宾馆", ["ドルフィン・ホテル", "いるかホテル"])
    index = store.read_artifact("glossary-disambiguation/judgements.json")
    assert [item["collision_id"] for item in index["judgements"]] == [first_id]

    # A later chapter extracts a second source sharing the same target: a new group appears,
    # while the settled one must stay settled.
    store.upsert_term(
        GlossaryTerm(source="Blue Whale Hotel", target="海豚宾馆", type="place"), chapter=0
    )
    second_client = _client(
        "judged",
        same_entity=False,
        renderings={
            "いるかホテル": "海豚宾馆",
            "ドルフィン・ホテル": "海豚宾馆",
            "Blue Whale Hotel": "蓝鲸宾馆",
        },
    )
    second_orch = Orchestrator(_config(str(tmp_path)), client=second_client, storage=store)
    second = _run(second_orch, store)

    # Only the new group is judged; the settled one is not re-asked.
    assert second["collisions"] == 1
    assert second["judged"] == 1
    index = store.read_artifact("glossary-disambiguation/judgements.json")
    judged_ids = [item["collision_id"] for item in index["judgements"]]
    assert len(judged_ids) == 2, judged_ids  # accumulated, not overwritten
    assert first_id in judged_ids


def test_unresolved_groups_stop_retrying_after_max_attempts(tmp_path):
    """An undecided group retries with wider context, then exhausts and waits for a human."""
    store = _store(str(tmp_path))
    client = _client("unresolved", same_entity=None, renderings=None)
    orchestrator = Orchestrator(_config(str(tmp_path)), client=client, storage=store)

    for attempt in range(1, _MAX_UNRESOLVED_ATTEMPTS + 1):
        summary = _run(orchestrator, store)
        assert summary["collisions"] == 1, attempt
        index = store.read_artifact("glossary-disambiguation/judgements.json")
        assert index["unresolved"][0]["attempts"] == attempt

    spent = len(client.calls)
    summary = _run(orchestrator, store)
    assert summary["collisions"] == 0  # exhausted: no further model calls
    assert len(client.calls) == spent


def test_one_run_judges_at_most_the_group_cap(tmp_path):
    """A burst of new collisions is split across close-outs instead of stalling translation."""
    store = _store(str(tmp_path))  # two sources sharing one target -> one group
    renderings = {"いるかホテル": "海豚宾馆", "ドルフィン・ホテル": "海豚宾馆"}
    source_parts = []
    for i in range(6):
        # One group per target: each target needs at least two distinct sources.
        store.upsert_term(
            GlossaryTerm(source=f"Place {i} left", target=f"译名{i}", type="place"), chapter=0
        )
        store.upsert_term(
            GlossaryTerm(source=f"Place {i} right", target=f"译名{i}", type="place"), chapter=0
        )
        renderings[f"Place {i} left"] = f"译名{i}"
        renderings[f"Place {i} right"] = f"译名{i}"
        source_parts.append(f"Place {i} left and Place {i} right opened")
    # The judge needs located occurrences: every new source must appear in the book text.
    store.save_chapter(
        Chapter(
            index=0,
            title="第一章",
            segments=[
                Segment(
                    index=0,
                    source=" / ".join(source_parts),
                    target=" / ".join(f"译文{i}" for i in range(6)),
                ),
                Segment(index=1, source="いるかホテルへ行く", target="去海豚宾馆。"),
            ],
        )
    )
    client = _client("judged", same_entity=True, renderings=renderings)
    orchestrator = Orchestrator(_config(str(tmp_path)), client=client, storage=store)

    summary = _run(orchestrator, store)

    assert summary["collisions"] == _MAX_GROUPS_PER_RUN
    assert summary["deferred"] == 1  # seven groups, cap six
    assert len(client.calls) == _MAX_GROUPS_PER_RUN
