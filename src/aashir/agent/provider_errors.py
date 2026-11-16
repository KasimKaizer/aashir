"""Provider error types and normalization for the agent boundary."""

from dataclasses import dataclass
from typing import Final, Literal, assert_never, override

from openai import (
    APIConnectionError,
    APIError,
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
    message: str
    error_type: str | None = None

    @override
    def __str__(self) -> str:
        return self.message


# The SDK raises subclasses for HTTP 401, 403, 429 and 5xx; other failures, including
# errors reported inside a stream, are classified by the provider's code or type.
_CATEGORIES_BY_PROVIDER_NAME: Final[dict[str, ProviderErrorCategory]] = {
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


def _normalize_provider_error(error: APIError) -> ProviderError:
    """Convert an SDK failure into the provider-neutral agent error contract."""
    match error:
        case RateLimitError():
            category: ProviderErrorCategory = "rate_limit"
        case AuthenticationError() | PermissionDeniedError():
            category = "auth"
        case APIConnectionError() | InternalServerError():
            category = "provider_unavailable"
        case APIError():
            category = _CATEGORIES_BY_PROVIDER_NAME.get(
                error.code or "",
                _CATEGORIES_BY_PROVIDER_NAME.get(error.type or "", "unknown"),
            )
        case unreachable:
            assert_never(unreachable)

    return ProviderError(
        category=category,
        error_type=error.type,
        message=error.message,
    )
