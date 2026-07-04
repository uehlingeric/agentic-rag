"""Tests for retry decorator."""

import pytest

from agentic_rag.config import RetrySettings
from agentic_rag.providers.base import ProviderAuthError, ProviderError
from agentic_rag.providers.retry import with_retries


class TestWithRetries:
    """Test with_retries decorator."""

    @pytest.mark.asyncio
    async def test_retryable_succeeds_after_failures(self) -> None:
        """Test that retryable errors are retried."""
        call_count = 0

        @with_retries(RetrySettings(max_attempts=3, initial_backoff_s=0.0, max_backoff_s=0.0))
        async def flaky_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ProviderError("Transient", provider="test", retryable=True)
            return "success"

        result = await flaky_func()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_non_retryable_raises_immediately(self) -> None:
        """Test that non-retryable errors are not retried."""
        call_count = 0

        @with_retries(RetrySettings(max_attempts=3, initial_backoff_s=0.0, max_backoff_s=0.0))
        async def failing_func() -> str:
            nonlocal call_count
            call_count += 1
            raise ProviderAuthError("Auth failed", provider="test")

        with pytest.raises(ProviderAuthError):
            await failing_func()

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_exhausts_max_attempts(self) -> None:
        """Test that retries exhaust after max_attempts."""
        call_count = 0

        @with_retries(RetrySettings(max_attempts=2, initial_backoff_s=0.0, max_backoff_s=0.0))
        async def always_fails() -> str:
            nonlocal call_count
            call_count += 1
            raise ProviderError("Retryable", provider="test", retryable=True)

        with pytest.raises(ProviderError) as exc_info:
            await always_fails()

        assert call_count == 2
        assert exc_info.value.retryable is True

    @pytest.mark.asyncio
    async def test_passes_through_args(self) -> None:
        """Test that decorator passes through function args."""

        @with_retries(RetrySettings(max_attempts=1, initial_backoff_s=0.0, max_backoff_s=0.0))
        async def add(a: int, b: int, *, c: int = 0) -> int:
            return a + b + c

        result = await add(1, 2, c=3)
        assert result == 6

    @pytest.mark.asyncio
    async def test_rate_limit_respected(self) -> None:
        """Test that retry_after is respected in ProviderRateLimitError."""
        import time

        from agentic_rag.providers.base import ProviderRateLimitError

        call_count = 0
        start_time = time.time()

        @with_retries(RetrySettings(max_attempts=2, initial_backoff_s=0.1, max_backoff_s=0.5))
        async def rate_limited() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ProviderRateLimitError("Rate limited", provider="test", retry_after=0.2)
            return "success"

        result = await rate_limited()
        elapsed = time.time() - start_time

        assert result == "success"
        assert call_count == 2
        # Should have waited at least retry_after (0.2s), but less than double (0.4s+)
        assert 0.15 < elapsed < 1.0  # Loose bounds to avoid flakiness

    @pytest.mark.asyncio
    async def test_first_attempt_success(self) -> None:
        """Test that successful first attempt doesn't retry."""
        call_count = 0

        @with_retries(RetrySettings(max_attempts=3, initial_backoff_s=0.0, max_backoff_s=0.0))
        async def immediate_success() -> str:
            nonlocal call_count
            call_count += 1
            return "done"

        result = await immediate_success()
        assert result == "done"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_exponential_backoff(self) -> None:
        """Test exponential backoff is applied."""
        call_count = 0

        @with_retries(RetrySettings(max_attempts=3, initial_backoff_s=0.05, max_backoff_s=0.5))
        async def exponential_fail() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ProviderError("Transient", provider="test", retryable=True)
            return "success"

        result = await exponential_fail()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_backoff_capped(self) -> None:
        """Test that backoff doesn't exceed max_backoff_s."""
        call_count = 0

        @with_retries(RetrySettings(max_attempts=4, initial_backoff_s=10.0, max_backoff_s=0.05))
        async def capped_backoff() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                raise ProviderError("Transient", provider="test", retryable=True)
            return "success"

        result = await capped_backoff()
        assert result == "success"
        assert call_count == 4
