"""
SimulationRunner — runs the full distribution pipeline in simulation mode.

No API calls are made. All platforms return mock JobResult objects.
Used for: CI testing, pre-deployment validation, pipeline smoke tests.
"""

from __future__ import annotations

import sys
import os
from typing import List

from loguru import logger

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
  sys.path.insert(0, _SRC_DIR)

from distribution.orchestrator import DistributionOrchestrator
from distribution.job_result import JobResult


class SimulationRunner:
  """
  Runs the full distribution pipeline in simulation mode.

  Initializes DistributionOrchestrator with simulation_mode=True
  so that zero real API calls are made during the run.

  Typical use:
    runner = SimulationRunner()
    results = await runner.run(content_package)
    runner.assert_all_platforms_reached(results)
  """

  def __init__(self, config_path: str = "config/distribution.yaml") -> None:
    """
    Initialize the SimulationRunner.

    Args:
      config_path: Path to distribution.yaml (relative or absolute).
                   Defaults to the project-level config file.
    """
    self._orchestrator = DistributionOrchestrator(
      config_path=config_path,
      simulation_mode=True,
    )
    logger.info("SimulationRunner initialized. All API calls will be suppressed.")

  async def run(self, content_package: dict) -> List[JobResult]:
    """
    Execute the full distribution pipeline in simulation mode.

    No real API calls are made. Every enabled platform returns a
    JobResult with status="simulated".

    Args:
      content_package: Content dict as expected by DistributionOrchestrator.distribute().

    Returns:
      List of JobResult — one per enabled platform.
    """
    logger.info("SimulationRunner.run() started.")
    results = await self._orchestrator.distribute(content_package)
    logger.info(
      "SimulationRunner.run() complete. {n} results collected.",
      n=len(results),
    )
    return results

  def assert_all_platforms_reached(self, results: List[JobResult]) -> bool:
    """
    Assert that every enabled platform produced a JobResult.

    Compares enabled platform names against the platform fields in results.
    Raises AssertionError with the missing platform list if any are absent.

    Args:
      results: List of JobResult objects from run().

    Returns:
      True when all enabled platforms have a corresponding result.

    Raises:
      AssertionError: when one or more enabled platforms are missing from results.
    """
    enabled = set(self._orchestrator.get_enabled_platforms())
    reached = {r.platform for r in results}
    missing = enabled - reached

    if missing:
      raise AssertionError(
        f"The following enabled platforms did not produce a JobResult: "
        f"{sorted(missing)}"
      )

    logger.info(
      "assert_all_platforms_reached: all {n} platforms reached.",
      n=len(enabled),
    )
    return True
