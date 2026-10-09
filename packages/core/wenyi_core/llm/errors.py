"""Safe, provider-independent diagnostics for failed model requests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .json_parser import JsonParseError
from .retrying import (
    EmptyResponseError,
    OutputBudgetExhausted,
    ProviderRequestError,
    TruncatedResponseError,
    error_status_code,
    is_transient_upstream,
    retry_reason,
)


@dataclass(frozen=True)
class ProviderFailure:
    """A diagnostic that never includes provider response or request contents."""

    status_code: int | None
    category: str
    message: str

    def log_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "error_category": self.category,
            "error_message": self.message,
        }
        if self.status_code is not None:
            fields["status_code"] = self.status_code
        return fields


_HTTP_FAILURES = {
    400: ("invalid_request", "The provider rejected the request. Check the model configuration."),
    401: ("authentication_failed", "Provider authentication failed. Check the API key."),
    402: (
        "insufficient_balance",
        "The provider reported insufficient balance. Recharge the account or change provider.",
    ),
    403: ("permission_denied", "Provider access was denied. Check account and model permissions."),
    404: (
        "model_or_endpoint_not_found",
        "The provider model or endpoint was not found. Check the model and endpoint configuration.",
    ),
    408: ("timeout", "The provider request timed out. Try again later."),
    422: ("invalid_request", "The provider rejected the request. Check the model configuration."),
    429: ("rate_limited", "The provider rate limit was reached. Try again later."),
}
# Categories that send the operator to inspect their own configuration. A gateway reporting
# its own outage under one of these statuses must not be reported this way.
_CONFIGURATION_CATEGORIES = frozenset(
    {"invalid_request", "authentication_failed", "permission_denied", "model_or_endpoint_not_found"}
)

_RESPONSE_FAILURES = (
    (TruncatedResponseError, "truncated_response", "The provider response was truncated."),
    (EmptyResponseError, "empty_response", "The provider returned no usable text."),
    (JsonParseError, "invalid_json", "The provider response was not valid JSON."),
    (json.JSONDecodeError, "invalid_json", "The provider response was not valid JSON."),
)


def describe_provider_failure(error: BaseException) -> ProviderFailure:
    """Normalize known failures without inspecting or copying raw error messages.

    A gateway that reports its own outage with a client-error status is an availability
    problem, not a request the caller got wrong: the same body that makes the request
    retryable decides the reported category, so the operator is never sent to re-check a
    model configuration that is already correct.
    """
    status = error_status_code(error)
    mapped = _HTTP_FAILURES.get(status) if status is not None else None
    if (
        mapped is not None
        and mapped[0] in _CONFIGURATION_CATEGORIES
        and is_transient_upstream(error)
    ):
        return ProviderFailure(
            status, "provider_unavailable", "The provider is unavailable. Try again later."
        )
    if mapped is not None:
        category, message = mapped
        return ProviderFailure(status, category, message)
    if status is not None:
        if status >= 500:
            return ProviderFailure(
                status, "provider_unavailable", "The provider is unavailable. Try again later."
            )
        return ProviderFailure(status, "http_error", f"The provider returned HTTP {status}.")

    current: BaseException | None = error
    seen: set[int] = set()
    names: list[str] = []
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        names.append(type(current).__name__)
        for error_type, category, message in _RESPONSE_FAILURES:
            if isinstance(current, error_type):
                # The exhausted-budget error is raised by the operation's own retry ladder,
                # and its text names the operator's next step (raise max_output_tokens or
                # lower reasoning_effort). Unlike a bare truncation surfaced while parsing a
                # provider response, that message is ours and safe to show verbatim.
                detail = str(current).strip() if isinstance(current, OutputBudgetExhausted) else ""
                return ProviderFailure(None, category, detail or message)
        current = current.__cause__ or current.__context__

    reason = retry_reason(error)
    if reason == "timeout":
        return ProviderFailure(None, "timeout", "The provider request timed out. Try again later.")
    if reason == "connection":
        return ProviderFailure(
            None, "connection_failed", "Could not connect to the provider. Check connectivity."
        )
    return ProviderFailure(
        None, "unknown_error", f"The provider request failed ({' caused by '.join(names)})."
    )


def operator_failure_message(error: BaseException) -> str:
    """Return the message to persist or display for a failed run.

    A model-request failure is reported through its safe classification: raw SDK text carries
    the provider's response body, which tells the operator nothing they can act on and belongs
    to the provider's own logs. Anything else keeps its own message, because domain errors are
    already written for the operator and the classifier would flatten them. An error that is
    already a safe classification keeps its own text, which names the operation and the error
    type.
    """
    if isinstance(error, ProviderRequestError):
        return str(error)
    failure = describe_provider_failure(error)
    return str(error) if failure.category == "unknown_error" else failure.message


__all__ = ["ProviderFailure", "describe_provider_failure", "operator_failure_message"]
