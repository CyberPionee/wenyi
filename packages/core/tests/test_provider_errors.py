"""Provider diagnostics exclude raw SDK contents and preserve useful failure causes."""

import json
from dataclasses import FrozenInstanceError

import httpx
import pytest
from openai import APIStatusError
from wenyi_core.llm.errors import describe_provider_failure
from wenyi_core.llm.retrying import EmptyResponseError, TruncatedResponseError


def _status_error(status):
    response = httpx.Response(
        status,
        request=httpx.Request("POST", "https://example.invalid"),
    )
    return APIStatusError(
        "PRIVATE_PLACEHOLDER https://user:password@example.invalid?api_key=PRIVATE_PLACEHOLDER",
        response=response,
        body={"prompt": "PRIVATE_PLACEHOLDER", "api_key": "PRIVATE_PLACEHOLDER"},
    )


@pytest.mark.parametrize(
    "status, category",
    [
        (400, "invalid_request"),
        (401, "authentication_failed"),
        (402, "insufficient_balance"),
        (403, "permission_denied"),
        (404, "model_or_endpoint_not_found"),
        (408, "timeout"),
        (422, "invalid_request"),
        (429, "rate_limited"),
        (500, "provider_unavailable"),
        (503, "provider_unavailable"),
        (409, "http_error"),
    ],
)
def test_http_diagnostics_are_safe(status, category):
    failure = describe_provider_failure(_status_error(status))
    assert failure.status_code == status
    assert failure.category == category
    assert failure.log_fields()["status_code"] == status
    assert "PRIVATE_PLACEHOLDER" not in json.dumps(failure.log_fields())
    assert "example.invalid" not in failure.message
    with pytest.raises(FrozenInstanceError):
        setattr(failure, "category", "changed")


@pytest.mark.parametrize(
    "error, category",
    [
        (TimeoutError("PRIVATE_PLACEHOLDER"), "timeout"),
        (httpx.ConnectError("PRIVATE_PLACEHOLDER"), "connection_failed"),
        (EmptyResponseError("PRIVATE_PLACEHOLDER"), "empty_response"),
        (TruncatedResponseError("PRIVATE_PLACEHOLDER"), "truncated_response"),
        (json.JSONDecodeError("PRIVATE_PLACEHOLDER", "PRIVATE_PLACEHOLDER", 0), "invalid_json"),
        (ValueError("PRIVATE_PLACEHOLDER"), "unknown_error"),
    ],
)
def test_chained_diagnostics_are_safe(error, category):
    wrapper = RuntimeError("PRIVATE_PLACEHOLDER")
    wrapper.__cause__ = error
    failure = describe_provider_failure(wrapper)
    assert failure.category == category
    assert "status_code" not in failure.log_fields()
    assert "PRIVATE_PLACEHOLDER" not in json.dumps(failure.log_fields())
    if category == "unknown_error":
        assert "RuntimeError" in failure.message
        assert "ValueError" in failure.message


def test_chained_http402_and_cycle():
    wrapper = RuntimeError("PRIVATE_PLACEHOLDER")
    wrapper.__cause__ = _status_error(402)
    failure = describe_provider_failure(wrapper)
    assert failure.category == "insufficient_balance"
    assert "Recharge" in failure.message and "change provider" in failure.message
    wrapper.__cause__ = wrapper
    assert describe_provider_failure(wrapper).category == "unknown_error"


def _upstream_outage(status: int = 400):
    """A gateway reporting its own outage under a client-error status."""
    response = httpx.Response(
        status,
        text=(
            '{"error": {"type": "server_error", '
            '"message": "Upstream request failed: Model is unavailable."}}'
        ),
        request=httpx.Request("POST", "https://example.invalid"),
    )
    return APIStatusError(
        "PRIVATE_PLACEHOLDER", response=response, body={"key": "PRIVATE_PLACEHOLDER"}
    )


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_gateway_outage_is_not_reported_as_a_configuration_problem(status):
    """The request is retryable, so the diagnostic must not send the operator to the config."""
    failure = describe_provider_failure(_upstream_outage(status))
    assert failure.status_code == status
    assert failure.category == "provider_unavailable"
    assert "unavailable" in failure.message
    assert "configuration" not in failure.message
    assert "PRIVATE_PLACEHOLDER" not in json.dumps(failure.log_fields())


def test_retryable_statuses_keep_their_specific_message():
    """408/429 already say to try again; an outage body must not flatten them."""
    assert describe_provider_failure(_upstream_outage(408)).category == "timeout"
    assert describe_provider_failure(_upstream_outage(429)).category == "rate_limited"


def test_genuine_request_error_still_blames_the_request():
    response = httpx.Response(
        400,
        text='{"error": {"message": "Invalid value for temperature"}}',
        request=httpx.Request("POST", "https://example.invalid"),
    )
    failure = describe_provider_failure(
        APIStatusError("PRIVATE_PLACEHOLDER", response=response, body={})
    )
    assert failure.category == "invalid_request"
    assert "Check the model configuration" in failure.message
