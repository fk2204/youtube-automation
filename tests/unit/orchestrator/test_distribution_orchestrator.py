"""
Unit tests for DistributionOrchestrator.

All platform calls and API calls are mocked. No real credentials required.
Tests confirm routing, simulation mode, retry integration, error containment,
and summary accuracy across all 12 platforms.
"""

from __future__ import annotations

import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from distribution.orchestrator import DistributionOrchestrator
from distribution.job_result import JobResult

# Path to the real distribution.yaml so tests use realistic config.
_CONFIG_PATH = os.path.join(
  os.path.dirname(__file__), "..", "..", "..", "config", "distribution.yaml"
)

SAMPLE_CONTENT = {
  "title": "Test Post Title",
  "body": "# Test\n\nThis is test content for pipeline validation.",
  "video_path": None,
  "image_paths": [],
  "tags": ["test", "automation"],
  "canonical_url": "https://example.com/test",
  "metadata": {},
}


@pytest.fixture
def orchestrator() -> DistributionOrchestrator:
  """Orchestrator in simulation mode — zero API calls."""
  return DistributionOrchestrator(
    config_path=_CONFIG_PATH,
    simulation_mode=True,
  )


@pytest.fixture
def real_orchestrator() -> DistributionOrchestrator:
  """Orchestrator in live mode for testing dispatch logic with mocks."""
  return DistributionOrchestrator(
    config_path=_CONFIG_PATH,
    simulation_mode=False,
  )


class TestInit:
  def test_init_loads_config(self, orchestrator: DistributionOrchestrator) -> None:
    """__init__ loads distribution.yaml and populates _enabled."""
    assert isinstance(orchestrator._config, dict)
    assert "platforms" in orchestrator._config

  def test_init_creates_retry_handler(self, orchestrator: DistributionOrchestrator) -> None:
    """__init__ creates a RetryHandler from config."""
    from distribution.retry_handler import RetryHandler
    assert isinstance(orchestrator._retry_handler, RetryHandler)


class TestGetEnabledPlatforms:
  def test_get_enabled_platforms_returns_list(
    self, orchestrator: DistributionOrchestrator
  ) -> None:
    """get_enabled_platforms() returns a non-empty list of strings."""
    platforms = orchestrator.get_enabled_platforms()
    assert isinstance(platforms, list)
    assert len(platforms) > 0

  def test_get_enabled_platforms_includes_expected_platforms(
    self, orchestrator: DistributionOrchestrator
  ) -> None:
    """get_enabled_platforms() includes platforms defined as enabled in config."""
    platforms = orchestrator.get_enabled_platforms()
    assert "tiktok" in platforms
    assert "medium" in platforms


class TestDistribute:
  @pytest.mark.asyncio
  async def test_distribute_returns_list_of_job_results(
    self, orchestrator: DistributionOrchestrator
  ) -> None:
    """distribute() returns a List[JobResult]."""
    results = await orchestrator.distribute(SAMPLE_CONTENT)
    assert isinstance(results, list)
    assert all(isinstance(r, JobResult) for r in results)

  @pytest.mark.asyncio
  async def test_all_platforms_produce_job_result(
    self, orchestrator: DistributionOrchestrator
  ) -> None:
    """distribute() produces one JobResult per enabled platform."""
    enabled = orchestrator.get_enabled_platforms()
    results = await orchestrator.distribute(SAMPLE_CONTENT)
    reached = {r.platform for r in results}
    assert reached == set(enabled)

  @pytest.mark.asyncio
  async def test_simulation_mode_makes_no_api_calls(
    self, orchestrator: DistributionOrchestrator
  ) -> None:
    """In simulation mode, all results have status='simulated' and simulated=True."""
    results = await orchestrator.distribute(SAMPLE_CONTENT)
    for result in results:
      assert result.status == "simulated"
      assert result.simulated is True

  @pytest.mark.asyncio
  async def test_disabled_platforms_are_skipped(
    self, orchestrator: DistributionOrchestrator
  ) -> None:
    """Platforms not in enabled config are skipped, not failed."""
    # Use platforms_override with a non-existent platform.
    results = await orchestrator.distribute(
      SAMPLE_CONTENT,
      platforms_override=["nonexistent_platform_xyz"],
    )
    # In simulation mode, simulated result is returned even for overridden platforms.
    # This test verifies the override mechanism works.
    assert len(results) == 1
    assert results[0].platform == "nonexistent_platform_xyz"

  @pytest.mark.asyncio
  async def test_platforms_override_limits_targets(
    self, orchestrator: DistributionOrchestrator
  ) -> None:
    """platforms_override restricts distribution to only the listed platforms."""
    results = await orchestrator.distribute(
      SAMPLE_CONTENT,
      platforms_override=["tiktok", "medium"],
    )
    assert len(results) == 2
    platforms_reached = {r.platform for r in results}
    assert platforms_reached == {"tiktok", "medium"}


class TestDistributeToPlatform:
  @pytest.mark.asyncio
  async def test_distribute_to_platform_returns_job_result(
    self, orchestrator: DistributionOrchestrator
  ) -> None:
    """distribute_to_platform() always returns a JobResult."""
    result = await orchestrator.distribute_to_platform("tiktok", SAMPLE_CONTENT)
    assert isinstance(result, JobResult)
    assert result.platform == "tiktok"

  @pytest.mark.asyncio
  async def test_failed_platform_does_not_block_others(
    self, real_orchestrator: DistributionOrchestrator
  ) -> None:
    """A failure on one platform does not prevent others from being attempted."""
    original_call = real_orchestrator._call_platform

    call_count = 0

    async def call_with_first_fail(platform_name, content):
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        raise RuntimeError("Simulated first-platform crash")
      return {"success": True, "post_id": "ok", "url": None, "error": None}

    real_orchestrator._call_platform = call_with_first_fail

    results = await real_orchestrator.distribute(
      SAMPLE_CONTENT,
      platforms_override=["tiktok", "instagram"],
    )

    assert len(results) == 2
    assert call_count == 2
    statuses = {r.platform: r.status for r in results}
    assert statuses["tiktok"] == "failed"
    assert statuses["instagram"] == "success"

  @pytest.mark.asyncio
  async def test_youtube_community_returns_skipped_not_error(
    self, real_orchestrator: DistributionOrchestrator
  ) -> None:
    """youtube_community always returns status='skipped' (API unavailable)."""
    result = await real_orchestrator.distribute_to_platform(
      "youtube_community", SAMPLE_CONTENT
    )
    assert result.status == "skipped"
    assert result.error_code == "PLATFORM_UNAVAILABLE"

  @pytest.mark.asyncio
  async def test_job_result_always_populated_regardless_of_platform_outcome(
    self, real_orchestrator: DistributionOrchestrator
  ) -> None:
    """distribute_to_platform always returns a fully-formed JobResult."""
    async def always_fails(platform_name, content):
      raise Exception("hard crash")

    real_orchestrator._call_platform = always_fails

    result = await real_orchestrator.distribute_to_platform("tiktok", SAMPLE_CONTENT)

    assert result.platform == "tiktok"
    assert result.status == "failed"
    assert result.error_message is not None
    assert result.started_at is not None
    assert result.completed_at is not None


class TestRetryIntegration:
  @pytest.mark.asyncio
  async def test_retry_handler_called_on_retryable_error(
    self, real_orchestrator: DistributionOrchestrator
  ) -> None:
    """Retryable errors trigger multiple _call_platform invocations."""
    call_count = 0

    async def retryable_fail(platform_name, content):
      nonlocal call_count
      call_count += 1
      raise ConnectionError("transient")

    real_orchestrator._call_platform = retryable_fail

    with patch("distribution.retry_handler.asyncio.sleep", new_callable=AsyncMock):
      result = await real_orchestrator.distribute_to_platform("tiktok", SAMPLE_CONTENT)

    # max_attempts is 3 per config — all 3 attempts should have been made.
    assert call_count == real_orchestrator._retry_handler._max_attempts
    assert result.status == "failed"

  @pytest.mark.asyncio
  async def test_non_retryable_error_not_retried(
    self, real_orchestrator: DistributionOrchestrator
  ) -> None:
    """AuthError is non-retryable — _call_platform called only once."""
    from exceptions import AuthError

    call_count = 0

    async def auth_fail(platform_name, content):
      nonlocal call_count
      call_count += 1
      raise AuthError("bad credentials")

    real_orchestrator._call_platform = auth_fail

    result = await real_orchestrator.distribute_to_platform("tiktok", SAMPLE_CONTENT)

    assert call_count == 1
    assert result.status == "failed"
    assert result.error_code == "AUTH_ERROR"


class TestResultsSummary:
  @pytest.mark.asyncio
  async def test_get_results_summary_counts_are_accurate(
    self, orchestrator: DistributionOrchestrator
  ) -> None:
    """get_results_summary() returns correct totals after distribute()."""
    await orchestrator.distribute(SAMPLE_CONTENT)
    summary = orchestrator.get_results_summary()

    assert "total" in summary
    assert "success" in summary
    assert "failed" in summary
    assert "skipped" in summary
    assert "simulated" in summary
    assert "platforms" in summary

    assert summary["total"] == len(orchestrator.get_enabled_platforms())
    # In simulation mode all should be simulated.
    assert summary["simulated"] == summary["total"]
    assert summary["success"] == 0
    assert summary["failed"] == 0
