"""Responses API provider: request shape, response reading and usage accounting.

Gateways expose some models only on this endpoint and reject Chat Completions for them with
ModelProtocolUnsupported, so the protocol is a separate provider kind rather than a flag.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from wenyi_core.llm.providers._openai_responses import (
    normalize_responses_usage,
    responses_output_text,
    responses_request_kwargs,
)
from wenyi_core.llm.providers.opencode_go import (
    OpenCodeGoOptions,
    OpenCodeGoResponsesClient,
)
from wenyi_core.llm.retrying import EmptyResponseError, TruncatedResponseError
from wenyi_core.llm.router import RoutedLLMClient
from wenyi_core.llm.transport import RequestContext, ResolvedModel

from tests.model_fixtures import model_config

MESSAGES = [
    {"role": "system", "content": "You review translations."},
    {"role": "user", "content": "Check this paragraph."},
]


class _NullScope:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _context() -> RequestContext:
    return RequestContext(
        operation="translation.body",
        tier="strong",
        max_tokens=None,
        emit=lambda *a, **k: None,
        record_usage=lambda *a, **k: None,
        attempt_scope=lambda: _NullScope(),
    )


def _message_item(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text=text)])


def _response(*, output, status="completed", incomplete=None, usage=None, error=None):
    return SimpleNamespace(
        output=list(output),
        status=status,
        incomplete_details=incomplete,
        usage=usage,
        error=error,
    )


def _routed():
    """A routed client whose SDK connection is a stub, so no request leaves the process."""
    client = RoutedLLMClient(model_config(kind="opencode-go-responses"))
    adapter = client.adapter("default")
    calls: list[dict] = []
    return client, adapter, calls


def _respond(adapter, response, calls: list[dict]) -> None:
    def create(**kwargs):
        calls.append(kwargs)
        return response

    adapter._client = SimpleNamespace(responses=SimpleNamespace(create=create))


class TestResponsesRequest:
    def test_system_message_becomes_instructions(self):
        kwargs = responses_request_kwargs("m", MESSAGES, json_mode=False, max_tokens=512)
        assert kwargs["instructions"] == "You review translations."
        assert kwargs["input"] == [{"role": "user", "content": "Check this paragraph."}]
        assert kwargs["max_output_tokens"] == 512
        # Responses rejects max_tokens as an unknown parameter.
        assert "max_tokens" not in kwargs
        assert "messages" not in kwargs

    def test_json_mode_marks_the_format_and_both_messages(self):
        kwargs = responses_request_kwargs("m", MESSAGES, json_mode=True, max_tokens=None)
        assert kwargs["text"] == {"format": {"type": "json_object"}}
        assert "json" in kwargs["instructions"].lower()
        assert "json" in kwargs["input"][-1]["content"].lower()
        assert "max_output_tokens" not in kwargs
        assert MESSAGES[0]["content"] == "You review translations."

    def test_reasoning_and_extra_body_are_forwarded(self):
        kwargs = responses_request_kwargs(
            "m",
            MESSAGES,
            json_mode=False,
            max_tokens=64,
            reasoning={"effort": "minimal"},
            extra_body={"temperature": 0.2},
        )
        assert kwargs["reasoning"] == {"effort": "minimal"}
        assert kwargs["temperature"] == 0.2

    def test_opencode_maps_thinking_onto_reasoning_effort(self):
        client = RoutedLLMClient(model_config(kind="opencode-go-responses")).adapter("default")
        thinking = ResolvedModel(model="m", options=OpenCodeGoOptions(thinking=True))
        built = client._build_request_kwargs(thinking, MESSAGES, json_mode=False, max_tokens=64)
        assert built["reasoning"] == {"effort": "high"}

        # The endpoint cannot switch reasoning off and the gateway rejects "none".
        quiet = ResolvedModel(model="m", options=OpenCodeGoOptions(thinking=False))
        built = client._build_request_kwargs(quiet, MESSAGES, json_mode=False, max_tokens=64)
        assert built["reasoning"] == {"effort": "minimal"}

    def test_thinking_keeps_a_large_output_budget(self):
        """A reasoning model spends the budget before writing, so a small limit returns nothing."""
        options = OpenCodeGoOptions(thinking=True)
        assert OpenCodeGoResponsesClient.output_limit(options, 512, None) == 4096
        with pytest.raises(ValueError, match="Thinking mode requires"):
            OpenCodeGoResponsesClient.output_limit(options, 512, 512)


class TestResponsesOutput:
    def test_assistant_text_is_read_past_reasoning_items(self):
        response = _response(output=[SimpleNamespace(type="reasoning"), _message_item("ok")])
        assert responses_output_text(response) == "ok"

    def test_multiple_text_parts_are_joined(self):
        response = _response(
            output=[
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(type="output_text", text="a"),
                        SimpleNamespace(type="output_text", text="b"),
                    ],
                )
            ]
        )
        assert responses_output_text(response) == "ab"

    def test_sdk_convenience_property_is_the_fallback(self):
        assert responses_output_text(SimpleNamespace(output=[], output_text="p")) == "p"

    def test_missing_output_reads_as_empty(self):
        assert responses_output_text(SimpleNamespace(output=None)) == ""


class TestResponsesUsage:
    def test_input_and_output_tokens_become_prompt_and_completion(self):
        sample = normalize_responses_usage(
            {
                "input_tokens": 100,
                "output_tokens": 40,
                "total_tokens": 140,
                "input_tokens_details": {"cached_tokens": 30},
            }
        )
        assert sample is not None
        assert sample.prompt_tokens == 100
        assert sample.completion_tokens == 40
        assert sample.total_tokens == 140
        assert sample.cache_hit_tokens == 30
        assert sample.cache_miss_tokens == 70

    def test_usage_without_cache_details_reports_no_hits(self):
        sample = normalize_responses_usage({"input_tokens": 10, "output_tokens": 2})
        assert sample is not None
        assert sample.cache_hit_tokens == 0
        assert sample.cache_miss_tokens == 10

    def test_absent_usage_stays_absent(self):
        assert normalize_responses_usage(None) is None


class TestResponsesExecution:
    def test_a_completed_answer_returns_its_text_and_posts_to_responses(self):
        client, adapter, calls = _routed()
        _respond(adapter, _response(output=[_message_item("translated")]), calls)
        assert client.complete(MESSAGES, operation="translation.body") == "translated"
        assert calls and "input" in calls[0] and "messages" not in calls[0]

    def test_an_incomplete_answer_is_a_truncation(self):
        client, adapter, calls = _routed()
        _respond(
            adapter,
            _response(
                output=[],
                status="incomplete",
                incomplete=SimpleNamespace(reason="max_output_tokens"),
            ),
            calls,
        )
        with pytest.raises(TruncatedResponseError, match="max_output_tokens"):
            adapter._request(
                MESSAGES,
                ResolvedModel("m", OpenCodeGoOptions()),
                json_mode=False,
                context=_context(),
            )

    def test_a_failed_answer_is_retryable(self):
        client, adapter, calls = _routed()
        _respond(
            adapter,
            _response(output=[], status="failed", error=SimpleNamespace(message="gave up")),
            calls,
        )
        with pytest.raises(EmptyResponseError, match="gave up"):
            adapter._request(
                MESSAGES,
                ResolvedModel("m", OpenCodeGoOptions()),
                json_mode=False,
                context=_context(),
            )

    def test_an_answer_with_only_reasoning_is_retryable(self):
        client, adapter, calls = _routed()
        _respond(adapter, _response(output=[SimpleNamespace(type="reasoning")]), calls)
        with pytest.raises(EmptyResponseError, match="content is empty"):
            adapter._request(
                MESSAGES,
                ResolvedModel("m", OpenCodeGoOptions()),
                json_mode=False,
                context=_context(),
            )

    def test_usage_reaches_the_ledger(self):
        client, adapter, calls = _routed()
        _respond(
            adapter,
            _response(
                output=[_message_item("ok")],
                usage={
                    "input_tokens": 7,
                    "output_tokens": 3,
                    "total_tokens": 10,
                    "input_tokens_details": {"cached_tokens": 2},
                },
            ),
            calls,
        )
        client.complete(MESSAGES, operation="translation.body")
        totals = client.usage_summary()["totals"]
        assert totals["calls"] == 1
        assert totals["prompt_tokens"] == 7
        assert totals["completion_tokens"] == 3
        assert totals["cache_hit_tokens"] == 2


@pytest.mark.parametrize("field", ["input", "instructions", "text", "reasoning"])
def test_responses_body_fields_cannot_be_replaced_through_extra_body(field):
    """extra_body is merged last, so these must be reserved: otherwise a model profile could
    replace the conversation, the system prompt or the JSON response format."""
    from wenyi_core.llm.configuration import LLMConfig

    with pytest.raises(ValueError, match="Reserved model option"):
        LLMConfig.model_validate(
            {
                "providers": {"go": {"kind": "opencode-go-responses"}},
                "models": {
                    "m": {
                        "provider": "go",
                        "model": "muse-spark-1.3-contributor",
                        "options": {"extra_body": {field: "x"}},
                    }
                },
                "tiers": {"strong": "m", "cheap": "m", "fast": "m"},
            }
        )
