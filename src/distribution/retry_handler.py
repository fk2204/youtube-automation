"""
RetryHandler — exponential backoff with jitter for async and sync callables.

Used by the orchestrator to wrap platform calls that may hit transient errors
such as 429 rate limits or 503 service-unavailable responses.
"""

from __future__ import annotations

import asyncio
import inspect
import random
from typing import Any, Callable, Tuple, Type

from loguru import logger


class RetryHandler:
  """
  Exponential backoff retry with jitter.

  Config-driven: max_attempts, base_delay_seconds, max_delay_seconds,
  backoff_factor are all read from the config dict supplied at init time.
  """

  def __init__(self, retry_config: dict) -> None:
    """
    Initialize with a retry configuration dict.

    Config schema:
      {
        "max_attempts": 3,
        "base_delay_seconds": 1.0,
        "max_delay_seconds": 60.0,
        "backoff_factor": 2.0,
      }
    """
    self._max_attempts: int = int(retry_config.get("max_attempts", 3))
    self._base_delay: float = float(retry_config.get("base_delay_seconds", 1.0))
    self._max_delay: float = float(retry_config.get("max_delay_seconds", 60.0))
    self._backoff_factor: float = float(retry_config.get("backoff_factor", 2.0))

  async def execute_with_retry(
    self,
    func: Callable,
    retryable_exceptions: Tuple[Type[Exception], ...],
    *args: Any,
    **kwargs: Any,
  ) -> Any:
    """
    Execute func with retry on retryable_exceptions.

    Detects async callables automatically and awaits them.
    Raises the last exception when all attempts are exhausted.
    Non-retryable exceptions propagate immediately without retrying.

    Args:
      func:                 Callable to execute (sync or async).
      retryable_exceptions: Tuple of exception types that warrant a retry.
      *args:                Positional arguments forwarded to func.
      **kwargs:             Keyword arguments forwarded to func.

    Returns:
      Whatever func returns on success.

    Raises:
      Last retryable exception when max_attempts are exhausted.
      Any non-retryable exception on first occurrence.
    """
    is_async = inspect.iscoroutinefunction(func)
    last_exc: Exception = RuntimeError("No attempts made")

    for attempt in range(self._max_attempts):
      try:
        if is_async:
          return await func(*args, **kwargs)
        return func(*args, **kwargs)

      except retryable_exceptions as exc:
        last_exc = exc
        remaining = self._max_attempts - attempt - 1

        if remaining == 0:
          logger.error(
            "Exhausted retries after {n} attempts for {func}.",
            n=self._max_attempts,
            func=getattr(func, "__name__", repr(func)),
          )
          raise

        delay = self._calculate_delay(attempt)
        logger.info(
          "Retry attempt {a}/{m} after {d:.2f}s: {exc}",
          a=attempt + 1,
          m=self._max_attempts,
          d=delay,
          exc=exc,
        )
        await asyncio.sleep(delay)

    raise last_exc

  def _calculate_delay(self, attempt: int) -> float:
    """
    Compute exponential backoff delay with jitter.

    Formula: min(base * factor^attempt, max_delay) * (1 + jitter)
    Jitter: random 0–10% to prevent thundering herd on distributed retries.

    Args:
      attempt: Zero-based attempt index (0 = first retry delay).

    Returns:
      Delay in seconds, capped at max_delay.
    """
    raw_delay = self._base_delay * (self._backoff_factor ** attempt)
    capped = min(raw_delay, self._max_delay)
    jitter = capped * random.uniform(0.0, 0.10)
    return capped + jitter
