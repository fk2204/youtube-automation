"""
YouTubeCommunityPoster — STUB for YouTube Community Posts API.

BLOCKER: The YouTube Community Posts API is not publicly available.
As of 2026, Google has not opened a programmatic API for Community Posts.
The YouTube Data API v3 does not expose an endpoint for creating, reading,
or managing Community Posts for regular channels.

Status: Unavailable — all methods raise NotImplementedError.

Use simulate() to test the pipeline without real API access.
Monitor https://developers.google.com/youtube/v3/docs for API availability updates.
"""

from __future__ import annotations

from typing import Any, Dict

from loguru import logger

from registry.poster_registry import PosterRegistry
from social.base_poster import BasePoster
from social.messaging_platform import MessagingPlatform

BLOCKER_REASON = (
  "YouTubeCommunityPoster is unavailable: "
  "YouTube Community Posts API is not publicly accessible as of 2026. "
  "Use simulate() for pipeline testing. "
  "Monitor https://developers.google.com/youtube/v3/docs for availability."
)


class YouTubeCommunityPoster(MessagingPlatform, BasePoster):
  """
  STUB poster for YouTube Community Posts.

  All methods raise NotImplementedError until the YouTube Community Posts API
  becomes publicly available. Use simulate() for integration testing.

  No credentials are required — this class will never make API calls.
  """

  platform_name = "youtube_community"

  def __init__(self) -> None:
    # Log a warning immediately — instantiation signals intent to use a blocked platform.
    logger.warning(
      "YouTubeCommunityPoster instantiated but platform is unavailable. "
      "All methods will raise NotImplementedError. Use simulate() for testing."
    )

  async def authenticate(self) -> None:
    """
    Not implemented — YouTube Community Posts API is unavailable.

    Raises:
        NotImplementedError: always.
    """
    raise NotImplementedError(BLOCKER_REASON)

  async def post_community_update(
    self,
    text: str,
    image_url: str = "",
  ) -> Dict[str, Any]:
    """
    Not implemented — YouTube Community Posts API is unavailable.

    Raises:
        NotImplementedError: always.
    """
    raise NotImplementedError(BLOCKER_REASON)

  async def send_video(
    self,
    file_path: str,
    caption: str = "",
  ) -> Dict[str, Any]:
    """
    Not implemented — YouTube Community Posts API is unavailable.

    Raises:
        NotImplementedError: always.
    """
    raise NotImplementedError(BLOCKER_REASON)

  async def send_message(
    self,
    text: str,
  ) -> Dict[str, Any]:
    """
    Not implemented — YouTube Community Posts API is unavailable.

    Raises:
        NotImplementedError: always.
    """
    raise NotImplementedError(BLOCKER_REASON)

  async def get_rate_limit_status(self) -> Dict[str, Any]:
    """
    Not implemented — YouTube Community Posts API is unavailable.

    Raises:
        NotImplementedError: always.
    """
    raise NotImplementedError(BLOCKER_REASON)

  def simulate(self) -> Dict[str, Any]:
    """
    Return a mock success response for pipeline integration testing.

    Does NOT make any API calls. Safe to call without credentials.
    Use this to verify that orchestration logic handles the YouTube
    Community channel correctly before real API access is available.

    Returns:
        Mock success dict matching the shape other messaging posters return.
    """
    logger.info(
      "[SIMULATION] YouTubeCommunityPoster.simulate() called — "
      "returning mock success response."
    )
    return {
      "success": True,
      "message_id": "sim_youtube_community_12345",
      "platform": self.platform_name,
      "error": None,
      "simulated": True,
      "note": BLOCKER_REASON,
    }


def _register() -> None:
  """Register YouTubeCommunityPoster with the global PosterRegistry."""
  PosterRegistry.register("youtube_community", YouTubeCommunityPoster)
  logger.debug("YouTubeCommunityPoster registered (stub — API unavailable).")


_register()
