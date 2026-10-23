import httpx2
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
)

from aashir.agent import AgentRequest, ProviderError, run_agent
from tests.helpers import make_failing_openai_client, make_openai_client, sse_chunk


@pytest.mark.parametrize(
    ("provider_type", "provider_code", "expected_category"),
    [
        ("rate_limit_error", None, "rate_limit"),
        ("authentication_error", None, "auth"),
        ("invalid_request_error", "context_length_exceeded", "context_length"),
        ("server_error", None, "provider_unavailable"),
        ("invalid_request_error", "content_policy_violation", "content_policy"),
        ("unrecognized_error", None, "unknown"),
    ],
)
def test_run_agent_normalizes_provider_error_categories(
    provider_type: str,
    provider_code: str | None,
    expected_category: str,
) -> None:
    # Given: a provider error payload with a known or unrecognized type/code.
    error_payload = {"message": "provider failed", "type": provider_type}
    if provider_code is not None:
        error_payload["code"] = provider_code
    body = sse_chunk({}, error=error_payload) + b"data: [DONE]\n\n"
    with (
        make_openai_client(httpx2.ByteStream(body)) as client,
        pytest.raises(ProviderError) as error,
    ):
        events = run_agent(
            client,
            AgentRequest(
                model="test-model", messages=[{"role": "user", "content": "hi"}]
            ),
        )

        # When: the consumer iterates the provider stream.
        list(events)

    # Then: the normalized error retains its category and provider type.
    assert error.value.category == expected_category
    assert error.value.error_type == provider_type


@pytest.mark.parametrize(
    ("error_class", "status", "expected_category"),
    [
        (RateLimitError, 429, "rate_limit"),
        (APIStatusError, 429, "rate_limit"),
        (AuthenticationError, 401, "auth"),
        (PermissionDeniedError, 403, "auth"),
        (APIStatusError, 401, "auth"),
        (APIStatusError, 403, "auth"),
        (InternalServerError, 500, "provider_unavailable"),
        (APIStatusError, 500, "provider_unavailable"),
        (APIStatusError, 503, "provider_unavailable"),
    ],
)
def test_run_agent_normalizes_sdk_status_error_classes(
    error_class: type[APIStatusError],
    status: int,
    expected_category: str,
) -> None:
    # Given: a provider transport that fails with an SDK status error.
    request = httpx2.Request("POST", "https://example.invalid/v1/chat/completions")
    error = error_class(
        "provider failed",
        response=httpx2.Response(status, request=request),
        body=None,
    )

    # When: the agent attempts the provider request.
    with make_failing_openai_client(error) as client:
        events = run_agent(
            client,
            AgentRequest(
                model="test-model", messages=[{"role": "user", "content": "hi"}]
            ),
        )
        with pytest.raises(ProviderError) as raised:
            list(events)

    # Then: the error is normalized with its message and cause preserved.
    assert raised.value.category == expected_category
    assert raised.value.error_type is None
    assert raised.value.message == "provider failed"
    assert raised.value.__cause__ is error


@pytest.mark.parametrize("error_class", [APIConnectionError, APITimeoutError])
def test_run_agent_normalizes_sdk_connection_error_classes(
    error_class: type[APIConnectionError],
) -> None:
    # Given: a provider transport that fails with an SDK connection error.
    request = httpx2.Request("POST", "https://example.invalid/v1/chat/completions")
    error = error_class(request=request)

    # When: the agent attempts the provider request.
    with (
        make_failing_openai_client(error) as client,
        pytest.raises(ProviderError) as raised,
    ):
        events = run_agent(
            client,
            AgentRequest(
                model="test-model", messages=[{"role": "user", "content": "hi"}]
            ),
        )
        list(events)

    # Then: the error is normalized with its message and cause preserved.
    assert raised.value.category == "provider_unavailable"
    assert raised.value.error_type is None
    assert raised.value.message == error.message
    assert raised.value.__cause__ is error
