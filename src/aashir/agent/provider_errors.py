"""Provider error types and normalization for the agent boundary."""

from dataclasses import dataclass
from typing import Literal, assert_never

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    AuthenticationError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
)

type ProviderErrorCategory = Literal[
    "rate_limit",
    "auth",
    "context_length",
    "provider_unavailable",
    "content_policy",
    "protocol",
    "unknown",
]


@dataclass(frozen=True, slots=True)
class ProviderError(Exception):
    """A provider failure normalized independently of the SDK."""

    category: ProviderErrorCategory
    error_type: str | None
    message: str

    def __str__(self) -> str:
        return self.message


def _normalize_provider_error(error: APIError) -> ProviderError:
    """Convert an SDK failure into the provider-neutral agent error contract."""
    match error:
        case RateLimitError() | APIStatusError(status_code=429):
            category: ProviderErrorCategory = "rate_limit"
        case AuthenticationError() | PermissionDeniedError():
            category = "auth"
        case APIStatusError(status_code=status_code) if status_code in {401, 403}:
            category = "auth"
        case APIConnectionError() | InternalServerError():
            category = "provider_unavailable"
        case APIStatusError(status_code=status_code) if status_code >= 500:
            category = "provider_unavailable"
        case APIError():
            error_categories: dict[str, ProviderErrorCategory] = {
                "authentication_error": "auth",
                "content_filter": "content_policy",
                "content_policy_violation": "content_policy",
                "context_length": "context_length",
                "context_length_exceeded": "context_length",
                "insufficient_quota": "rate_limit",
                "invalid_api_key": "auth",
                "permission_denied": "auth",
                "permission_denied_error": "auth",
                "provider_error": "provider_unavailable",
                "provider_unavailable": "provider_unavailable",
                "rate_limit": "rate_limit",
                "rate_limit_error": "rate_limit",
                "rate_limit_exceeded": "rate_limit",
                "server_error": "provider_unavailable",
            }
            category = error_categories.get(
                error.code or "",
                error_categories.get(error.type or "", "unknown"),
            )
        case unreachable:
            assert_never(unreachable)

    return ProviderError(
        category=category,
        error_type=error.type,
        message=error.message,
    )
