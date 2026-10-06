"""Configuration endpoints expose routing tools and workflow statistics."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from wenyi_backend.application import create_app
from wenyi_backend.project_service import effective_config
from wenyi_backend.routers import configuration
from wenyi_core.config import Config
from wenyi_core.llm.usage import UsageSample, UsageTracker, empty_usage


def test_model_comparison_endpoints_are_not_available(backend_context):
    app = create_app(backend_context)
    client = TestClient(app)
    try:
        assert client.post("/projects/test/models/compare", json={}).status_code == 404
        assert client.get("/projects/test/models/comparisons/test").status_code == 404
    finally:
        client.close()
    schema = app.openapi()
    assert "ModelCompareRequest" not in schema["components"]["schemas"]
    assert "ModelMessage" not in schema["components"]["schemas"]
    assert "/projects/{pid}/models/compare" not in schema["paths"]
    assert "/projects/{pid}/models/comparisons/{job_id}" not in schema["paths"]
    assert "/projects/{pid}/models" in schema["paths"]
    assert "/projects/{pid}/models/check" in schema["paths"]


def test_precision_project_yaml_validates_mode_and_required_polish():
    defaults = Config.from_dict({"llm": {"preset": "fake"}})
    project = {
        "id": "precision-test",
        "source_lang": "en",
        "target_lang": "zh",
        "config": {"pipeline": {"translation_mode": "best_of_three", "polish": True}},
    }
    valid = effective_config(
        project,
        defaults=defaults,
        document={"pipeline": {"translation_mode": "best_of_three", "polish": True}},
    )
    assert valid.pipeline.translation_mode == "best_of_three"
    for pipeline in (
        {"translation_mode": "best_of_three", "polish": False},
        {"translation_mode": "invalid"},
        {"translation_mode": "standard"},
    ):
        with pytest.raises(ValueError):
            effective_config(project, defaults=defaults, document={"pipeline": pipeline})

    for document in ({}, {"pipeline": {"review": False}}):
        restored = effective_config(project, defaults=defaults, document=document)
        assert restored.pipeline.translation_mode == "best_of_three"
        assert restored.pipeline.polish is True


def test_legacy_quick_project_read_preserves_saved_configuration():
    from copy import deepcopy

    project = {"id": "legacy", "strategy": {"template": "快速出稿"}}
    before = deepcopy(project)
    config = effective_config(project, defaults=Config.from_dict({"llm": {"preset": "fake"}}))
    assert not any(
        getattr(config.pipeline, key)
        for key in ("book_understanding", "polish", "review", "review_autofix")
    )
    assert project == before
    project["config"] = {"pipeline": {"polish": True}}
    assert effective_config(
        project, defaults=Config.from_dict({"llm": {"preset": "fake"}})
    ).pipeline.polish


def test_workflow_keeps_frozen_quick_snapshot_independent(monkeypatch):
    from copy import deepcopy

    snapshot = {
        "pipeline": dict.fromkeys(
            ("book_understanding", "polish", "review", "review_autofix"), False
        )
    }
    before = deepcopy(snapshot)
    monkeypatch.setattr(configuration, "require_project", lambda _: {"id": "legacy"})
    monkeypatch.setattr(
        configuration.dal,
        "list_jobs",
        lambda _: [
            {"kind": "translation", "status": "queued", "params": {"config_snapshot": snapshot}}
        ],
    )
    result = configuration.workflow("legacy")
    assert result["source"] == "snapshot"
    assert all(
        not stage["enabled"] for stage in result["stages"] if stage["id"] in snapshot["pipeline"]
    )
    assert snapshot == before


@pytest.mark.parametrize("initialized", [False, True])
def test_project_stats_read_workflow_usage_and_timing(monkeypatch, initialized):
    tracker = UsageTracker()
    tracker.record(
        "fast", UsageSample(prompt_tokens=2, completion_tokens=3, total_tokens=5), "Translator"
    )
    usage = tracker.summary() if initialized else None
    timing = {"runs": [{"id": "translation", "elapsed_seconds": 8}], "total_seconds": 8}

    def read_artifact(key):
        assert key == "timing.json"
        return timing if initialized else None

    storage = SimpleNamespace(load_usage=lambda: usage, read_artifact=read_artifact)
    monkeypatch.setattr(configuration, "require_project", lambda pid: {"id": pid})
    monkeypatch.setattr(configuration, "storage_for", lambda pid: storage)
    expected = {
        "usage": usage or empty_usage(),
        "timing": timing if initialized else {"runs": [], "total_seconds": 0},
    }
    assert configuration.project_stats("test") == expected
    assert configuration.project_stats("test") == expected


@pytest.mark.parametrize(
    ("initialized", "column_source", "manifest_source", "expected_source"),
    [
        # An initialized project recovers its detected language from the manifest, so a saved
        # ``auto`` never overwrites the direction already in use.
        (True, "auto", "ja", "ja"),
        # Without a recorded manifest language there is nothing to recover.
        (True, "auto", None, "auto"),
        # An uninitialized project has no detected language to keep.
        (False, "auto", "ja", "auto"),
        (False, "en", None, "auto"),
    ],
)
def test_saving_auto_source_keeps_the_detected_project_direction(
    monkeypatch, initialized, column_source, manifest_source, expected_source
):
    """An initialized project keeps its detected language when the saved config asks for auto."""
    project = {
        "id": "p",
        "initialized": initialized,
        "source_lang": column_source,
        "target_lang": "zh",
    }
    saved: list[tuple[str, str]] = []
    config = SimpleNamespace(source_lang="auto", target_lang="zh")

    @contextmanager
    def project_write(pid):
        assert pid == "p"
        yield (
            project,
            SimpleNamespace(
                load_manifest=lambda: {"source_lang": manifest_source, "target_lang": "zh"}
            ),
        )

    @contextmanager
    def registry_guard():
        yield SimpleNamespace(execute=lambda *args: None)

    monkeypatch.setattr(configuration, "project_write", project_write)
    monkeypatch.setattr(configuration, "registry_guard", registry_guard)
    monkeypatch.setattr(
        configuration, "load_settings", lambda **kwargs: SimpleNamespace(config=None)
    )
    monkeypatch.setattr(configuration, "effective_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(
        configuration,
        "project_document",
        lambda cfg: {"language": {"source": cfg.source_lang}},
    )
    monkeypatch.setattr(configuration, "config_response", lambda project, cfg: {"effective": {}})
    monkeypatch.setattr(
        configuration,
        "dal",
        SimpleNamespace(
            set_project_config=lambda *args, **kwargs: None,
            set_project_languages=lambda pid, source, target, connection=None: saved.append(
                (source, target)
            ),
        ),
    )

    response = configuration.save_config("p", SimpleNamespace(yaml="language: {source: auto}"))

    assert response == {"effective": {}}
    assert saved == [(expected_source, "zh")]
