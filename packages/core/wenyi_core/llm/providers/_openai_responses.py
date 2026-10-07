"""OpenAI Responses API wire protocol and single-attempt response handling.

The Responses endpoint takes the system message as ``instructions`` and the rest of the
conversation as ``input``, limits output with ``max_output_tokens``, and answers with a list of
output items in which the assistant text sits inside ``message`` items next to ``reasoning``
ones. Gateways expose some models only here: one of them rejects Chat Completions outright with
``ModelProtocolUnsupported``.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from ..retrying import EmptyResponseError, TruncatedResponseError
from ..transport import Messages, RequestContext, ResolvedModel
from ..usage import UsageSample, make_usage_sample, read_usage_int, read_usage_value
from ._openai_compatible import OpenAICompatibleBaseClient, deep_merge, json_mode_messages

OptionsT = TypeVar("OptionsT", bound=BaseModel)
_JSON_MODE_FORMAT = {"format": {"type": "json_object"}}


def responses_request_kwargs(
    model: str,
    messages: Messages,
    *,
    json_mode: bool,
    max_tokens: int | None,
    reasoning: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build Responses API arguments from the shared message list."""
    request_messages = json_mode_messages(messages) if json_mode else messages
    instructions = "\n\n".join(
        str(message.get("content", ""))
        for message in request_messages
        if message.get("role") == "system" and str(message.get("content", "")).strip()
    )
    kwargs: dict[str, Any] = {
        "model": model,
        "input": [dict(m) for m in request_messages if m.get("role") != "system"],
        "stream": False,
    }
    if instructions:
        kwargs["instructions"] = instructions
    if json_mode:
        kwargs["text"] = deep_merge({}, _JSON_MODE_FORMAT)
    if reasoning:
        kwargs["reasoning"] = deep_merge({}, reasoning)
    if max_tokens is not None:
        kwargs["max_output_tokens"] = max_tokens
    if extra_body:
        kwargs = deep_merge(kwargs, extra_body)
    return kwargs


def responses_output_text(response: Any) -> str:
    """Return the assistant text from a Responses payload, skipping reasoning items."""
    parts: list[str] = []
    for item in getattr(response, "output", None) or []:
        if str(getattr(item, "type", "") or "") != "message":
            continue
        for part in getattr(item, "content", None) or []:
            if str(getattr(part, "type", "") or "") != "output_text":
                continue
            text = getattr(part, "text", None)
            if isinstance(text, str):
                parts.append(text)
    if parts:
        return "".join(parts)
    # The SDK computes the same text as a convenience property when the items are shaped the
    # way it expects; fall back to it rather than reporting an empty answer.
    fallback = getattr(response, "output_text", None)
    return fallback if isinstance(fallback, str) else ""


def normalize_responses_usage(usage: Any) -> UsageSample | None:
    """Normalize Responses token fields into shared usage accounting.

    Responses reports input/output tokens rather than prompt/completion tokens, and puts the
    cached count under input_tokens_details.
    """
    if usage is None:
        return None
    details = read_usage_value(usage, "input_tokens_details")
    input_tokens = read_usage_int(usage, "input_tokens")
    cached = read_usage_value(details, "cached_tokens")
    cache_hit_tokens = read_usage_int(details, "cached_tokens") if cached is not None else 0
    return make_usage_sample(
        {
            "prompt_tokens": input_tokens,
            "completion_tokens": read_usage_int(usage, "output_tokens"),
            "total_tokens": read_usage_int(usage, "total_tokens"),
        },
        cache_hit_tokens=cache_hit_tokens,
        cache_miss_tokens=max(0, input_tokens - cache_hit_tokens),
    )


class OpenAIResponsesBaseClient(OpenAICompatibleBaseClient[OptionsT], Generic[OptionsT]):
    """Reuse one SDK connection over the Responses endpoint."""

    def _normalize_usage(self, usage: Any) -> UsageSample | None:
        return normalize_responses_usage(usage)

    def _request(
        self,
        messages: Messages,
        model: ResolvedModel[OptionsT],
        *,
        json_mode: bool,
        context: RequestContext,
    ) -> str:
        kwargs = self._build_request_kwargs(
            model, messages, json_mode=json_mode, max_tokens=context.max_tokens
        )
        response = self._ensure_client().responses.create(**kwargs)
        context.record_usage(self._normalize_usage(getattr(response, "usage", None)))
        status = str(getattr(response, "status", "") or "").lower()
        if status == "incomplete":
            details = getattr(response, "incomplete_details", None)
            reason = str(getattr(details, "reason", "") or "")
            # A response stopped at the output limit is not a complete answer; a reasoning
            # model can spend the whole budget before writing anything.
            raise TruncatedResponseError(
                f"{self.cfg.kind} response was incomplete ({reason or 'no reason given'})"
            )
        if status == "failed":
            error = getattr(response, "error", None)
            raise EmptyResponseError(
                f"{self.cfg.kind} response failed: {getattr(error, 'message', None) or 'no detail'}"
            )
        content = responses_output_text(response)
        if not content.strip():
            raise EmptyResponseError(f"{self.cfg.kind} response content is empty")
        return content
