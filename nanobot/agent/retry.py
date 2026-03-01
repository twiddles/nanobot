"""Retry helper for transient LLM errors (rate limits, 5xx, timeouts)."""

from __future__ import annotations

import asyncio
import random

from loguru import logger

from nanobot.providers.base import LLMResponse

# Patterns that indicate a NON-retryable error (checked first).
_NON_RETRYABLE = (
    "401", "403", "404",
    "invalid api key", "invalid_api_key",
    "authentication", "unauthorized", "permission denied",
    "invalid request", "invalid_request",
    "context_length_exceeded", "context length",
    "content filtering", "content_filter",
)

# Patterns that indicate a retryable (transient) error.
_RETRYABLE = (
    "ratelimit", "rate_limit", "rate limit",
    "429", "502", "503", "500",
    "server error", "service unavailable", "overloaded",
    "timeout", "timed out",
    "connection error", "connection reset",
    "too many requests", "quota exceeded", "try again",
)


def is_retryable_error(response: LLMResponse) -> bool:
    """Return True if the error response looks transient and worth retrying."""
    if response.finish_reason != "error":
        return False
    content = (response.content or "").lower()
    if any(p in content for p in _NON_RETRYABLE):
        return False
    return any(p in content for p in _RETRYABLE)


async def chat_with_retry(
    provider,
    *,
    max_retries: int,
    retry_backoff_seconds: float,
    **chat_kwargs,
) -> LLMResponse:
    """Call provider.chat() with exponential backoff on transient errors.

    Returns the final LLMResponse (success or last error after exhausting retries).
    Respects asyncio cancellation (e.g. /stop).
    """
    response = await provider.chat(**chat_kwargs)

    if max_retries <= 0 or not is_retryable_error(response):
        return response

    last = response
    for attempt in range(1, max_retries + 1):
        delay = retry_backoff_seconds * (2 ** (attempt - 1))
        delay += random.uniform(0, delay * 0.5)  # jitter

        logger.warning(
            "Transient LLM error (attempt {}/{}), retrying in {:.1f}s: {}",
            attempt, max_retries, delay, (last.content or "")[:120],
        )
        await asyncio.sleep(delay)

        response = await provider.chat(**chat_kwargs)
        if not is_retryable_error(response):
            if response.finish_reason != "error":
                logger.info("LLM retry succeeded on attempt {}/{}", attempt, max_retries)
            return response
        last = response

    logger.error("All {} retries exhausted: {}", max_retries, (last.content or "")[:200])
    return last
