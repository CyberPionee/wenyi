"""Offline terminology-conflict arbitration tests."""

from __future__ import annotations

import json
import re
from pathlib import Path

from wenyi_core.config import Config
from wenyi_core.glossary.store import MANUAL_STATUS, GlossaryTerm
from wenyi_core.ingest.models import Chapter, Segment
from wenyi_core.llm.providers.fake import FakeClient
from wenyi_core.pipeline.orchestrator import Orchestrator
from wenyi_core.pipeline.runstore import STATUS_DONE
from wenyi_core.storage.file import FileStorage

CONFLICT_ID = re.compile(r'"conflict_id":\s*"([^"]+)"')


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
    """One chapter holding both renderings plus an unrelated passage of the other name."""
    store = FileStorage(str(Path(directory, "state", "book")))
    store.save_chapter(
        Chapter(
            index=0,
            title="第一章",
            segments=[
                Segment(index=0, source="いるかホテルへ行く", target="去海豚旅店。"),
                Segment(index=1, source="いるかホテルは古い", target="海豚酒店很旧。"),
                Segment(index=2, source="ドルフィン・ホテルは白い", target="海豚酒店是白色的。"),
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
    store.upsert_term(GlossaryTerm(source="いるかホテル", target="海豚旅店"), chapter=0)
    assert (
        store.upsert_term(GlossaryTerm(source="いるかホテル", target="海豚酒店"), chapter=1)
        == "conflict"
    )
    return store


def _client(status: str, target: str = "") -> FakeClient:
    """Reply as the arbiter agent does, echoing the supplied conflict id."""

    def handler(messages, _tier, _json_mode):
        payload = json.dumps(
            {
                "action": "final",
                "conflict_id": CONFLICT_ID.search(messages[-1]["content"]).group(1),
                "status": status,
                "recommended_target": target,
                "reason": "The book names this hotel one way throughout.",
                "evidence_refs": [],
                "complete": True,
            },
            ensure_ascii=False,
        )
        return payload

    return FakeClient(handler)


def _targets(store: FileStorage) -> list[str]:
    return [segment.target for segment in store.load_chapter(0).text_segments]


def _events(store: FileStorage, kind: str) -> list[dict]:
    return store.list_events(event_type=kind)


def test_arbitration_settles_the_conflict_and_writes_the_rejected_rendering_back(tmp_path):
    """Keeping the established target still has to fix the passage that used the proposal."""
    store = _store(str(tmp_path))
    orchestrator = Orchestrator(
        _config(str(tmp_path)), client=_client("decided", "海豚旅店"), storage=store
    )

    summary = orchestrator._glossary_arbitration.run(store)

    assert summary["conflicts"] == 1
    assert summary["decided"] == 1
    assert summary["segments_replaced"] == 1
    term = store.get_term("いるかホテル")
    assert term is not None
    assert term.target == "海豚旅店"
    # Settling is a decision, so it locks the term like any operator action.
    assert term.status == MANUAL_STATUS
    assert store.open_conflicts() == []
    assert _targets(store) == [
        "去海豚旅店。",
        "海豚旅店很旧。",
        # Its source never mentions the term, so the same string elsewhere is left alone.
        "海豚酒店是白色的。",
    ]
    settled = _events(store, "glossary_conflict_arbitrated")
    assert len(settled) == 1
    assert settled[0]["target"] == "海豚旅店"
    assert settled[0]["rejected"] == ["海豚酒店"]
    assert store.read_artifact("glossary-arbitration/decisions.json")["status"] == "completed"


def test_arbitration_accepting_the_proposal_rewrites_the_established_rendering(tmp_path):
    """The other direction: the proposal wins, so the established rendering leaves the text."""
    store = _store(str(tmp_path))
    orchestrator = Orchestrator(
        _config(str(tmp_path)), client=_client("decided", "海豚酒店"), storage=store
    )

    summary = orchestrator._glossary_arbitration.run(store)

    assert summary["segments_replaced"] == 1
    assert store.get_term("いるかホテル").target == "海豚酒店"
    assert _targets(store) == ["去海豚酒店。", "海豚酒店很旧。", "海豚酒店是白色的。"]


def test_an_undecided_conflict_stays_open_with_its_reason(tmp_path):
    """Evidence that cannot separate the candidates leaves the decision to a human."""
    store = _store(str(tmp_path))
    orchestrator = Orchestrator(_config(str(tmp_path)), client=_client("unresolved"), storage=store)

    summary = orchestrator._glossary_arbitration.run(store)

    assert summary == {
        "conflicts": 1,
        "decided": 0,
        "undecided": 1,
        "segments_replaced": 0,
        "reason": "",
    }
    assert len(store.open_conflicts()) == 1
    assert _targets(store)[1] == "海豚酒店很旧。"


def test_arbitration_can_be_disabled(tmp_path):
    store = _store(str(tmp_path))
    client = _client("decided", "海豚旅店")
    orchestrator = Orchestrator(
        _config(str(tmp_path), glossary_conflict_arbitration=False), client=client, storage=store
    )

    summary = orchestrator._glossary_arbitration.run(store)

    assert summary["reason"] == "disabled"
    assert client.calls == []
    assert len(store.open_conflicts()) == 1


def test_a_recorded_decision_resumes_without_asking_the_model_again(tmp_path):
    """An interrupted run finishes from its own index rather than paying for another call."""
    store = _store(str(tmp_path))
    store.write_artifact(
        "glossary-arbitration/decisions.json",
        {
            "status": "decided",
            "decisions": [
                {
                    "source": "いるかホテル",
                    "target": "海豚旅店",
                    "rejected": ["海豚酒店"],
                    "reason": "recorded",
                }
            ],
            "undecided": [],
        },
    )
    client = _client("decided", "海豚旅店")
    orchestrator = Orchestrator(_config(str(tmp_path)), client=client, storage=store)

    summary = orchestrator._glossary_arbitration.resume_pending(store)

    assert summary is not None
    assert summary["resumed"] is True
    assert summary["segments_replaced"] == 1
    assert client.calls == []
    assert store.open_conflicts() == []
    assert _targets(store)[1] == "海豚旅店很旧。"
