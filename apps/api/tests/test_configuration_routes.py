"""Configuration endpoints expose routing tools and workflow statistics."""

from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from wenyi_api import main
from wenyi_api.routers import configuration
from wenyi_core.llm.usage import UsageSample, UsageTracker, empty_usage


def test_model_comparison_endpoints_are_not_available(monkeypatch):
    monkeypatch.setattr(main, "settings", replace(main.settings, api_token=None))
    app = main.create_app()
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
    "initialized,column_source,manifest_source,expected_source",
    [
        (True, "en", "en", "en"),
        (True, "auto", "en", "en"),
        (False, "en", None, "auto"),
    ],
)
def test_saving_auto_source_keeps_the_detected_project_direction(
    monkeypatch, initialized, column_source, manifest_source, expected_source
):
    project = {
        "id": "p",
        "initialized": initialized,
        "source_lang": column_source,
        "target_lang": "zh",
    }
    statements = []
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
        yield SimpleNamespace(execute=lambda *args: statements.append(args))

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
        configuration, "dal", SimpleNamespace(set_project_config=lambda *args, **kwargs: None)
    )

    response = configuration.save_config("p", SimpleNamespace(yaml="language: {source: auto}"))

    assert response == {"effective": {}}
    assert statements[0][1][:2] == (expected_source, "zh")
