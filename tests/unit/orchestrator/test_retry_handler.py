"""
Unit tests for RetryHandler.

All asyncio.sleep calls are mocked to keep the test suite fast.
Tests confirm exponential backoff, jitter, retry limits, and async detection.
"""

from __future__ import annotations

import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from distribution.retry_handler import RetryHandler

SAMPLE_CONFIG = {
  "max_attempts": 3,
  "base_delay_seconds": 1.0,
  "max_delay_seconds": 60.0,
  "backoff_factor": 2.0,
}


@pytest.fixture
def handler() -> RetryHandler:
  return RetryHandler(SAMPLE_CONFIG)


class TestInit:
  def test_init_parses_config(self) -> None:
    """RetryHandler reads all four config keys correctly."""
    h = RetryHandler({
      "max_attempts": 5,
      "base_delay_seconds": 2.0,
      "max_delay_seconds": 120.0,
      "backoff_factor": 3.0,
    })
    assert h._max_attempts == 5
    assert h._base_delay == 2.0
    assert h._max_delay == 120.0
    assert h._backoff_factor == 3.0


class TestExecuteWithRetry:
  @pytest.mark.asyncio
  async def test_execute_success_first_attempt(self, handler: RetryHandler) -> None:
    """execute_with_retry returns immediately when func succeeds on first call."""
    async def always_succeeds():
      return "ok"

    result = await handler.execute_with_retry(always_succeeds, (ValueError,))
    assert result == "ok"

  @pytest.mark.asyncio
  async def test_execute_retries_on_retryable_exception(self, handler: RetryHandler) -> None:
    """execute_with_retry retries when func raises a retryable exception."""
    call_count = 0

    async def fails_once():
      nonlocal call_count
      call_count += 1
      if call_count < 2:
        raise ConnectionError("transient failure")
      return "recovered"

    with patch("distribution.retry_handler.asyncio.sleep", new_callable=AsyncMock):
      result = await handler.execute_with_retry(fails_once, (ConnectionError,))

    assert result == "recovered"
    assert call_count == 2

  @pytest.mark.asyncio
  async def test_execute_exhausts_retries_raises_last_exception(
    self, handler: RetryHandler
  ) -> None:
    """execute_with_retry raises the last exception after max_attempts."""
    async def always_fails():
      raise ConnectionError("permanent failure")

    with patch("distribution.retry_handler.asyncio.sleep", new_callable=AsyncMock):
      with pytest.raises(ConnectionError, match="permanent failure"):
        await handler.execute_with_retry(always_fails, (ConnectionError,))

  @pytest.mark.asyncio
  async def test_non_retryable_exception_not_retried(self, handler: RetryHandler) -> None:
    """execute_with_retry propagates non-retryable exceptions immediately."""
    call_count = 0

    async def raises_non_retryable():
      nonlocal call_count
      call_count += 1
      raise ValueError("bad input — not retryable")

    with pytest.raises(ValueError, match="bad input"):
      await handler.execute_with_retry(raises_non_retryable, (ConnectionError,))

    # Should have been called exactly once — no retry for non-retryable.
    assert call_count == 1

  @pytest.mark.asyncio
  async def test_async_callable_detected_and_awaited(self, handler: RetryHandler) -> None:
    """execute_with_retry detects async callables and awaits them properly."""
    async def async_func():
      return "async_result"

    result = await handler.execute_with_retry(async_func, (ValueError,))
    assert result == "async_result"

  @pytest.mark.asyncio
  async def test_sync_callable_supported(self, handler: RetryHandler) -> None:
    """execute_with_retry supports plain synchronous callables."""
    def sync_func():
      return "sync_result"

    result = await handler.execute_with_retry(sync_func, (ValueError,))
    assert result == "sync_result"

  @pytest.mark.asyncio
  async def test_retry_count_does_not_exceed_max_attempts(
    self, handler: RetryHandler
  ) -> None:
    """execute_with_retry calls func at most max_attempts times."""
    call_count = 0

    async def count_calls():
      nonlocal call_count
      call_count += 1
      raise ConnectionError("fail")

    with patch("distribution.retry_handler.asyncio.sleep", new_callable=AsyncMock):
      with pytest.raises(ConnectionError):
        await handler.execute_with_retry(count_calls, (ConnectionError,))

    assert call_count == SAMPLE_CONFIG["max_attempts"]


class TestCalculateDelay:
  def test_calculate_delay_exponential_backoff(self, handler: RetryHandler) -> None:
    """_calculate_delay grows exponentially with each attempt."""
    delay_0 = handler._calculate_delay(0)
    delay_1 = handler._calculate_delay(1)
    delay_2 = handler._calculate_delay(2)

    # Remove jitter ceiling effect by asserting growth direction.
    assert delay_1 > delay_0
    assert delay_2 > delay_1

  def test_calculate_delay_capped_at_max(self, handler: RetryHandler) -> None:
    """_calculate_delay never exceeds max_delay_seconds + 10% jitter cap."""
    # At attempt 100 the raw delay would be astronomical — must be capped.
    delay = handler._calculate_delay(100)
    # Max is 60.0, plus up to 10% jitter = 66.0 at most.
    assert delay <= handler._max_delay * 1.10 + 0.01

  def test_jitter_adds_randomness(self, handler: RetryHandler) -> None:
    """_calculate_delay returns slightly different values on repeated calls."""
    delays = [handler._calculate_delay(0) for _ in range(20)]
    # At least some values should differ due to jitter (not all identical).
    assert len(set(round(d, 6) for d in delays)) > 1
