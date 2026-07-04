"""Async retry decorator using tenacity.

Retries only on retryable ProviderErrors. Exponential backoff with jitter.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import ParamSpec, TypeVar

from agentic_rag.config import RetrySettings
from agentic_rag.providers.base import ProviderError, ProviderRateLimitError

P = ParamSpec("P")
T = TypeVar("T")


def with_retries(
    settings: RetrySettings,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator factory for async callables with exponential backoff.

    Retries only when the raised exception is a ProviderError with retryable=True.
    If the exception is ProviderRateLimitError with retry_after set, waits at
    least retry_after seconds before retrying.

    Args:
        settings: RetrySettings with max_attempts, initial_backoff_s, max_backoff_s.

    Returns:
        A decorator for async callables that applies retry logic.
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Exception | None = None
            for attempt in range(settings.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except ProviderError as e:
                    if not e.retryable:
                        raise

                    last_exception = e
                    if attempt < settings.max_attempts - 1:
                        # Calculate backoff with exponential growth and jitter
                        base_backoff = min(
                            settings.initial_backoff_s * (2**attempt),
                            settings.max_backoff_s,
                        )
                        # Add jitter: random between base_backoff and base_backoff * 2
                        jitter = random.uniform(0, base_backoff)
                        wait_time = base_backoff + jitter

                        # If it's a rate limit error with retry_after, respect that
                        if isinstance(e, ProviderRateLimitError) and e.retry_after:
                            wait_time = max(wait_time, e.retry_after)

                        await asyncio.sleep(wait_time)

            # All retries exhausted
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected: no exception recorded after retries")

        return wrapper

    return decorator
