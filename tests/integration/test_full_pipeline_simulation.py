"""
Integration tests for the full distribution pipeline in simulation mode.

No API calls are made. No credentials are required.
Verifies that all 12 platforms are reachable and produce valid JobResult objects.
"""

from __future__ import annotations

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from distribution.orchestrator import DistributionOrchestrator
from distribution.simulation_runner import SimulationRunner
from distribution.job_result import JobResult

_CONFIG_PATH = os.path.join(
  os.path.dirname(__file__), "..", "..", "config", "distribution.yaml"
)

SAMPLE_CONTENT = {
  "title": "Integration Test Post",
  "body": "# Integration Test\n\nFull pipeline simulation content for all 12 platforms.",
  "video_path": None,
  "image_paths": [],
  "tags": ["integration", "simulation", "test"],
  "canonical_url": "https://example.com/integration-test",
  "metadata": {"source": "integration_test"},
}


@pytest.fixture
def runner() -> SimulationRunner:
  return SimulationRunner(config_path=_CONFIG_PATH)


@pytest.fixture
def orchestrator() -> DistributionOrchestrator:
  return DistributionOrchestrator(
    config_path=_CONFIG_PATH,
    simulation_mode=True,
  )


class TestSimulationRunnerAllPlatforms:
  @pytest.mark.asyncio
  async def test_simulation_runner_all_platforms_reached(
    self, runner: SimulationRunner
  ) -> None:
    """SimulationRunner produces a JobResult for every enabled platform."""
    results = await runner.run(SAMPLE_CONTENT)
    runner.assert_all_platforms_reached(results)

  @pytest.mark.asyncio
  async def test_simulation_runner_no_api_calls_made(
    self, runner: SimulationRunner
  ) -> None:
    """All results in simulation mode have simulated=True (zero real API calls)."""
    results = await runner.run(SAMPLE_CONTENT)
    assert len(results) > 0
    for result in results:
      assert result.simulated is True, (
        f"Platform '{result.platform}' returned simulated=False in simulation mode."
      )


class TestOrchestratorFullPipeline:
  @pytest.mark.asyncio
  async def test_distribution_orchestrator_distributes_to_all_enabled(
    self, orchestrator: DistributionOrchestrator
  ) -> None:
    """Orchestrator in simulation mode reaches every enabled platform."""
    enabled = set(orchestrator.get_enabled_platforms())
    results = await orchestrator.distribute(SAMPLE_CONTENT)
    reached = {r.platform for r in results}
    assert reached == enabled

  @pytest.mark.asyncio
  async def test_job_results_complete_no_missing_fields(
    self, orchestrator: DistributionOrchestrator
  ) -> None:
    """Every JobResult has all required fields populated."""
    results = await orchestrator.distribute(SAMPLE_CONTENT)

    for result in results:
      assert hasattr(result, "platform")
      assert hasattr(result, "status")
      assert hasattr(result, "started_at")
      assert hasattr(result, "completed_at")
      assert hasattr(result, "simulated")
      assert hasattr(result, "retry_count")
      assert hasattr(result, "metadata")
      assert result.platform is not None
      assert result.status in ("success", "failed", "skipped", "simulated")
      assert result.started_at is not None
      assert result.completed_at is not None

  @pytest.mark.asyncio
  async def test_unhandled_exceptions_caught_and_returned_in_job_result(
    self, orchestrator: DistributionOrchestrator
  ) -> None:
    """An unexpected exception inside _call_platform produces a failed JobResult, not a raise."""
    live_orchestrator = DistributionOrchestrator(
      config_path=_CONFIG_PATH,
      simulation_mode=False,
    )

    async def explode(platform_name, content):
      raise RuntimeError("Unexpected internal explosion")

    live_orchestrator._call_platform = explode

    results = await live_orchestrator.distribute(
      SAMPLE_CONTENT,
      platforms_override=["tiktok"],
    )

    assert len(results) == 1
    result = results[0]
    assert result.status == "failed"
    assert result.error_code == "UNEXPECTED_ERROR"
    assert "Unexpected internal explosion" in result.error_message
    assert result.completed_at is not None
