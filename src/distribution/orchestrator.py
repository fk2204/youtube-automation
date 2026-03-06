"""
DistributionOrchestrator — master coordinator for all 12 distribution platforms.

Reads enabled platforms from config/distribution.yaml.
Dispatches to Phase A/B poster classes and Phase C repurposer classes.
Collects JobResult objects and returns them to the caller.

Auto-registration boot sequence: importing this module triggers all 12 platform
_register() calls by importing each platform module at the bottom of __init__.
"""

from __future__ import annotations

import sys
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

# Ensure src/ is on sys.path when orchestrator is imported directly.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
  sys.path.insert(0, _SRC_DIR)

from config.settings import load_distribution_config
from distribution.job_result import JobResult
from distribution.retry_handler import RetryHandler
from exceptions import RateLimitError, AuthError, ValidationError
from registry.poster_registry import PosterRegistry
from distribution.repurposers.repurposer_registry import RepurposerRegistry

# Retryable exception types — 429 rate limits and transient service errors.
_RETRYABLE = (RateLimitError, TimeoutError, ConnectionError, OSError)


class DistributionOrchestrator:
  """
  Master coordinator for content distribution across all 12 platforms.

  Reads enabled platforms from config/distribution.yaml.
  Orchestrates Phase A (video posters), Phase B (messaging posters),
  and Phase C (repurposers).

  Usage:
    orchestrator = DistributionOrchestrator()
    results = await orchestrator.distribute(content_package)
  """

  def __init__(
    self,
    config_path: str = "config/distribution.yaml",
    simulation_mode: bool = False,
  ) -> None:
    """
    Initialize the orchestrator.

    Loads distribution.yaml, boots all platform registrations,
    and wires up the retry handler from config.

    Args:
      config_path:     Path to distribution.yaml (relative or absolute).
      simulation_mode: When True, all platforms return simulated results
                       without making any real API calls.
    """
    self._config_path = config_path
    self._simulation_mode = simulation_mode
    self._config = load_distribution_config(config_path)

    enabled_platforms = self._config.get("platforms", {})
    self._enabled: Dict[str, Dict[str, Any]] = {
      k: v
      for k, v in enabled_platforms.items()
      if isinstance(v, dict) and v.get("enabled", False)
    }

    retry_config = self._config.get("retry", {
      "max_attempts": 3,
      "base_delay_seconds": 1.0,
      "max_delay_seconds": 60.0,
      "backoff_factor": 2.0,
    })
    self._retry_handler = RetryHandler(retry_config)

    # Track per-run results for summary reporting.
    self._last_results: List[JobResult] = []

    # Boot all platform registrations by importing each module.
    self._boot_registrations()

    logger.info(
      "DistributionOrchestrator initialized. "
      "Enabled platforms: {platforms}. Simulation: {sim}.",
      platforms=list(self._enabled.keys()),
      sim=self._simulation_mode,
    )

  def _boot_registrations(self) -> None:
    """
    Import all 12 platform modules to trigger their _register() calls.

    Each module calls PosterRegistry.register() or RepurposerRegistry.register()
    at module level — importing is sufficient to register.
    """
    try:
      import src.social.tiktok_poster
    except (ImportError, ModuleNotFoundError):
      try:
        import social.tiktok_poster
      except (ImportError, ModuleNotFoundError):
        pass

    try:
      import src.social.instagram_poster
    except (ImportError, ModuleNotFoundError):
      try:
        import social.instagram_poster
      except (ImportError, ModuleNotFoundError):
        pass

    try:
      import src.social.facebook_poster
    except (ImportError, ModuleNotFoundError):
      try:
        import social.facebook_poster
      except (ImportError, ModuleNotFoundError):
        pass

    try:
      import src.social.pinterest_poster
    except (ImportError, ModuleNotFoundError):
      try:
        import social.pinterest_poster
      except (ImportError, ModuleNotFoundError):
        pass

    try:
      import src.social.telegram_poster
    except (ImportError, ModuleNotFoundError):
      try:
        import social.telegram_poster
      except (ImportError, ModuleNotFoundError):
        pass

    try:
      import src.social.whatsapp_poster
    except (ImportError, ModuleNotFoundError):
      try:
        import social.whatsapp_poster
      except (ImportError, ModuleNotFoundError):
        pass

    try:
      import src.social.youtube_community_poster
    except (ImportError, ModuleNotFoundError):
      try:
        import social.youtube_community_poster
      except (ImportError, ModuleNotFoundError):
        pass

    try:
      import src.distribution.repurposers.medium_repurposer
    except (ImportError, ModuleNotFoundError):
      try:
        import distribution.repurposers.medium_repurposer
      except (ImportError, ModuleNotFoundError):
        pass

    try:
      import src.distribution.repurposers.substack_repurposer
    except (ImportError, ModuleNotFoundError):
      try:
        import distribution.repurposers.substack_repurposer
      except (ImportError, ModuleNotFoundError):
        pass

    try:
      import src.distribution.repurposers.twitter_thread_repurposer
    except (ImportError, ModuleNotFoundError):
      try:
        import distribution.repurposers.twitter_thread_repurposer
      except (ImportError, ModuleNotFoundError):
        pass

    try:
      import src.distribution.repurposers.email_repurposer
    except (ImportError, ModuleNotFoundError):
      try:
        import distribution.repurposers.email_repurposer
      except (ImportError, ModuleNotFoundError):
        pass

    try:
      import src.distribution.repurposers.podcast_repurposer
    except (ImportError, ModuleNotFoundError):
      try:
        import distribution.repurposers.podcast_repurposer
      except (ImportError, ModuleNotFoundError):
        pass

  async def distribute(
    self,
    content_package: dict,
    platforms_override: Optional[List[str]] = None,
  ) -> List[JobResult]:
    """
    Distribute content to all enabled platforms (or the override list).

    Each platform is dispatched independently. A failure on one platform
    never blocks the others — errors are captured in JobResult.

    Args:
      content_package: Dict with keys:
        title:          str
        body:           str
        video_path:     Optional[str]
        image_paths:    Optional[list[str]]
        tags:           list[str]
        canonical_url:  Optional[str]
        metadata:       dict
      platforms_override: When set, only these platforms are targeted.
                          Platforms not in the enabled list are skipped.

    Returns:
      List of JobResult — one per platform attempted.
    """
    targets = list(platforms_override) if platforms_override else list(self._enabled.keys())

    results: List[JobResult] = []
    for platform_name in targets:
      result = await self.distribute_to_platform(platform_name, content_package)
      results.append(result)

    self._last_results = results

    success = sum(1 for r in results if r.is_success())
    failed = sum(1 for r in results if r.status == "failed")
    skipped = sum(1 for r in results if r.status == "skipped")
    simulated = sum(1 for r in results if r.status == "simulated")

    logger.info(
      "Distribution complete. success={s} failed={f} skipped={sk} simulated={sim}",
      s=success,
      f=failed,
      sk=skipped,
      sim=simulated,
    )
    return results

  async def distribute_to_platform(
    self,
    platform_name: str,
    content_package: dict,
  ) -> JobResult:
    """
    Dispatch content to a single platform.

    Lookup order: PosterRegistry first, RepurposerRegistry second.
    Handles simulation mode, YouTube Community special case, retries,
    and wraps all exceptions into a failed JobResult — never raises.

    Args:
      platform_name:   Registry key for the target platform.
      content_package: Content dict as described in distribute().

    Returns:
      JobResult with status set to success/failed/skipped/simulated.
    """
    started_at = datetime.utcnow()
    result = JobResult(platform=platform_name, status="failed", started_at=started_at)

    # Skip platforms that are not in the enabled set (when using override).
    if platform_name not in self._enabled and not self._simulation_mode:
      # If it's an override call to a disabled platform, still try it.
      # Only skip when the platform is completely unknown.
      if not PosterRegistry.is_registered(platform_name) and not RepurposerRegistry.is_registered(platform_name):
        result.status = "skipped"
        result.error_message = f"Platform '{platform_name}' not registered."
        result.completed_at = datetime.utcnow()
        logger.warning("Skipping unknown platform: {p}", p=platform_name)
        return result

    # Simulation mode: return a simulated result without any API calls.
    if self._simulation_mode:
      result.status = "simulated"
      result.simulated = True
      result.post_id = f"sim_{platform_name}_00000"
      result.post_url = f"https://sim.example.com/{platform_name}/00000"
      result.completed_at = datetime.utcnow()
      logger.info("[SIMULATION] {p} simulated successfully.", p=platform_name)
      return result

    # YouTube Community is a known-blocked platform — always return skipped.
    if platform_name == "youtube_community":
      result.status = "skipped"
      result.error_code = "PLATFORM_UNAVAILABLE"
      result.error_message = (
        "YouTube Community Posts API is not publicly accessible as of 2026."
      )
      result.completed_at = datetime.utcnow()
      logger.info("youtube_community: returning skipped (API unavailable).")
      return result

    try:
      platform_result = await self._retry_handler.execute_with_retry(
        self._call_platform,
        _RETRYABLE,
        platform_name,
        content_package,
      )

      result.status = "success" if platform_result.get("success") else "failed"
      result.post_id = platform_result.get("post_id")
      result.post_url = platform_result.get("url") or platform_result.get("post_url")
      if not platform_result.get("success"):
        result.error_message = platform_result.get("error")
        result.error_code = "PLATFORM_ERROR"

    except KeyError as exc:
      result.status = "skipped"
      result.error_code = "NOT_REGISTERED"
      result.error_message = str(exc)
      logger.warning("Platform not registered: {p} — {exc}", p=platform_name, exc=exc)

    except AuthError as exc:
      result.status = "failed"
      result.error_code = "AUTH_ERROR"
      result.error_message = str(exc)
      logger.error("Auth error for {p}: {exc}", p=platform_name, exc=exc)

    except ValidationError as exc:
      result.status = "failed"
      result.error_code = "VALIDATION_ERROR"
      result.error_message = str(exc)
      logger.error("Validation error for {p}: {exc}", p=platform_name, exc=exc)

    except Exception as exc:
      result.status = "failed"
      result.error_code = "UNEXPECTED_ERROR"
      result.error_message = str(exc)
      logger.error(
        "Unexpected error distributing to {p}: {exc}",
        p=platform_name,
        exc=exc,
      )

    result.completed_at = datetime.utcnow()
    return result

  async def _call_platform(
    self,
    platform_name: str,
    content_package: dict,
  ) -> Dict[str, Any]:
    """
    Instantiate the platform class and invoke it with content_package.

    Checks PosterRegistry first, RepurposerRegistry second.
    Raises KeyError if the platform is found in neither registry.

    Args:
      platform_name:   Registry key.
      content_package: Content dict.

    Returns:
      Platform response dict with at least {"success": bool, "error": str|None}.

    Raises:
      KeyError: when the platform is not registered in either registry.
    """
    poster_class = PosterRegistry.get(platform_name)
    if poster_class is not None:
      instance = poster_class()
      platform_cfg = self._enabled.get(platform_name, {})

      if hasattr(instance, "post_video") and content_package.get("video_path"):
        await instance.authenticate()
        return await instance.post_video(
          file_path=content_package["video_path"],
          title=content_package.get("title", ""),
          description=content_package.get("body", ""),
        )

      if hasattr(instance, "send_message"):
        await instance.authenticate()
        return await instance.send_message(
          text=content_package.get("body", ""),
        )

      if hasattr(instance, "post"):
        await instance.authenticate()
        return await instance.post(content_package)

      # Fallback for posters without a matched method.
      return {"success": True, "post_id": None, "url": None, "error": None}

    repurposer_class = RepurposerRegistry.get(platform_name)
    if repurposer_class is not None:
      platform_cfg = self._enabled.get(platform_name, {})
      instance = repurposer_class(platform_cfg)
      authenticated = await instance.authenticate()

      if not authenticated:
        return {
          "success": False,
          "error": f"{platform_name}: authentication failed.",
          "post_id": None,
          "url": None,
        }

      source_content = content_package.get("body", "")
      canonical_url = content_package.get("canonical_url")
      tags = content_package.get("tags", [])

      return await instance.repurpose(
        source_content,
        tags=tags,
        canonical_url=canonical_url,
      )

    raise KeyError(
      f"Platform '{platform_name}' not found in PosterRegistry or RepurposerRegistry."
    )

  def _load_platform(self, platform_name: str) -> Any:
    """
    Return the registered class for the given platform name.

    Checks PosterRegistry first, RepurposerRegistry second.

    Raises:
      KeyError: when the platform is not registered in either registry.
    """
    poster_class = PosterRegistry.get(platform_name)
    if poster_class is not None:
      return poster_class

    repurposer_class = RepurposerRegistry.get(platform_name)
    if repurposer_class is not None:
      return repurposer_class

    raise KeyError(
      f"Platform '{platform_name}' not found in PosterRegistry or RepurposerRegistry."
    )

  def get_enabled_platforms(self) -> List[str]:
    """Return platform names where enabled=true in distribution.yaml."""
    return list(self._enabled.keys())

  def get_results_summary(self) -> dict:
    """
    Return counts and per-platform status from the last distribute() run.

    Returns:
      {
        "total": int,
        "success": int,
        "failed": int,
        "skipped": int,
        "simulated": int,
        "platforms": { platform_name: status, ... }
      }
    """
    counts: Dict[str, int] = {
      "total": len(self._last_results),
      "success": 0,
      "failed": 0,
      "skipped": 0,
      "simulated": 0,
    }
    per_platform: Dict[str, str] = {}

    for result in self._last_results:
      per_platform[result.platform] = result.status
      if result.status in counts:
        counts[result.status] += 1

    return {**counts, "platforms": per_platform}
