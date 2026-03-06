"""
PinterestPoster — creates Pins and Video Pins via the Pinterest API v5.

API flows:

Image Pin (create_pin):
  POST /v5/pins with media_source.source_type = "image_url"
  Returns pin_id. No async processing required for image URLs.

Video Pin (create_video_pin):
  1. POST /v5/media — Register the video for upload.
     Returns media_id and upload_url (a pre-signed S3 URL).
  2. PUT to upload_url with the video file bytes.
  3. Poll GET /v5/media/{media_id} until status == "succeeded".
     Status values: "registered" → "processing" → "succeeded" | "failed".
     Polling is required — Pinterest processes video asynchronously.
  4. POST /v5/pins with media_source.media_id = media_id.
     IMPORTANT: cover_image_url is MANDATORY for video pins.
     The API will reject the pin creation request if cover_image_url is absent.

Boards (get_boards):
  GET /v5/boards — Returns paginated list of boards.
  Response includes board_id, name, description, pin_count.

Rate limits:
  - 1000 calls per day per app
  - 10 calls per second burst limit

Credentials (env vars):
  PINTEREST_ACCESS_TOKEN  — OAuth2 access token with pins:read, pins:write,
                              boards:read, video:upload scopes
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

from exceptions import AuthError, PollingTimeoutError, RateLimitError, ValidationError
from registry.poster_registry import PosterRegistry
from social.base_poster import BasePoster
from social.social_platform import SocialPlatform

PINTEREST_API_BASE = "https://api.pinterest.com"

# Polling configuration for video media registration
MAX_POLL_ATTEMPTS = 20
POLL_INTERVAL_SECONDS = 5

# Supported cover image formats (mandatory for video pins)
SUPPORTED_COVER_FORMATS = {".jpg", ".jpeg", ".png"}


class PinterestPoster(SocialPlatform, BasePoster):
  """
  Creates image Pins and Video Pins on Pinterest boards.

  Video pins require a cover_image_url — this is MANDATORY per the API.
  Requests without a cover image are rejected by Pinterest.

  Requires env vars:
    PINTEREST_ACCESS_TOKEN  — OAuth2 token with appropriate scopes
  """

  platform_name = "pinterest"

  def __init__(self) -> None:
    self._access_token: Optional[str] = None
    self._authenticated: bool = False

  async def authenticate(self) -> None:
    """
    Load and validate Pinterest credentials from env vars.

    Raises:
      AuthError: if PINTEREST_ACCESS_TOKEN is missing or token is invalid.
    """
    if self.guard_unconfigured("PINTEREST_ACCESS_TOKEN"):
      raise AuthError(
        "Pinterest credentials not configured. "
        "Set PINTEREST_ACCESS_TOKEN."
      )

    self._access_token = os.environ["PINTEREST_ACCESS_TOKEN"]

    try:
      async with httpx.AsyncClient() as client:
        response = await client.get(
          f"{PINTEREST_API_BASE}/v5/user_account",
          headers=self._auth_headers(),
          timeout=10.0,
        )
      if response.status_code == 401:
        raise AuthError(
          "Pinterest token rejected (401). "
          "Refresh PINTEREST_ACCESS_TOKEN."
        )
      if response.status_code == 429:
        raise RateLimitError("Pinterest rate limit hit during auth check.")
      response.raise_for_status()
    except (AuthError, RateLimitError):
      raise
    except Exception as exc:
      raise AuthError(f"Pinterest auth validation failed: {exc}") from exc

    self._authenticated = True
    logger.info("Pinterest authenticated successfully.")

  def _auth_headers(self) -> Dict[str, str]:
    """Return authorization headers for Pinterest API requests."""
    return {"Authorization": f"Bearer {self._access_token}"}

  async def _poll_media_registration(
    self,
    media_id: str,
    max_attempts: int = MAX_POLL_ATTEMPTS,
    poll_interval: int = POLL_INTERVAL_SECONDS,
  ) -> None:
    """
    Poll Pinterest media registration until status is 'succeeded'.

    Pinterest processes uploaded videos asynchronously. The media
    must reach 'succeeded' status before it can be used in a Pin.

    Status progression: registered → processing → succeeded | failed

    Args:
      media_id:     Media ID returned by the /v5/media registration call.
      max_attempts: Maximum polling attempts before raising timeout.
      poll_interval: Seconds between each poll attempt.

    Raises:
      PollingTimeoutError: if 'succeeded' is not reached within max_attempts.
      ValidationError: if media enters 'failed' status.
    """
    for attempt in range(max_attempts):
      await asyncio.sleep(poll_interval)

      try:
        async with httpx.AsyncClient() as client:
          response = await client.get(
            f"{PINTEREST_API_BASE}/v5/media/{media_id}",
            headers=self._auth_headers(),
            timeout=15.0,
          )
        response.raise_for_status()
      except Exception as exc:
        logger.warning(
          "Pinterest media poll attempt {a} failed for {mid}: {exc}",
          a=attempt + 1,
          mid=media_id,
          exc=exc,
        )
        continue

      status = response.json().get("status", "")
      logger.debug(
        "Pinterest media {mid} status: {s} (attempt {a}/{m})",
        mid=media_id,
        s=status,
        a=attempt + 1,
        m=max_attempts,
      )

      if status == "succeeded":
        logger.info(
          "Pinterest media registration succeeded: {mid}",
          mid=media_id,
        )
        return
      if status == "failed":
        raise ValidationError(
          f"Pinterest media {media_id} registration failed."
        )

    raise PollingTimeoutError(
      f"Pinterest media {media_id} did not reach 'succeeded' "
      f"after {max_attempts} attempts."
    )

  async def create_pin(
    self,
    board_id: str,
    image_url: str,
    title: str,
    description: str = "",
    link: Optional[str] = None,
  ) -> Dict[str, Any]:
    """
    Create an image Pin on a Pinterest board.

    Args:
      board_id:    The target board ID (from get_boards()).
      image_url:   Public URL of the image to pin.
      title:       Pin title (shown below the image).
      description: Optional Pin description.
      link:        Optional destination URL when the Pin is clicked.

    Returns:
      Standard response dict with pin_id as post_id.
    """
    if self.simulation_mode:
      return self._simulate_post(
        {"board_id": board_id, "image_url": image_url, "title": title}
      )

    payload: Dict[str, Any] = {
      "board_id": board_id,
      "title": title,
      "description": description,
      "media_source": {
        "source_type": "image_url",
        "url": image_url,
      },
    }
    if link:
      payload["link"] = link

    try:
      async with httpx.AsyncClient() as client:
        response = await client.post(
          f"{PINTEREST_API_BASE}/v5/pins",
          headers={**self._auth_headers(), "Content-Type": "application/json"},
          json=payload,
          timeout=15.0,
        )
      if response.status_code == 429:
        raise RateLimitError("Pinterest rate limit hit during pin creation.")
      response.raise_for_status()
    except RateLimitError:
      raise
    except Exception as exc:
      logger.error("Pinterest create_pin error: {exc}", exc=exc)
      return self._build_error_response(str(exc))

    pin_id = response.json().get("id", "")
    url = f"https://www.pinterest.com/pin/{pin_id}/"
    logger.info("Pinterest Pin created: {pid}", pid=pin_id)
    return self._build_success_response(post_id=pin_id, url=url)

  async def create_video_pin(
    self,
    board_id: str,
    video_file_path: str,
    cover_image_url: str,
    title: str,
    description: str = "",
  ) -> Dict[str, Any]:
    """
    Create a Video Pin on a Pinterest board.

    Steps:
    1. Register video for upload via /v5/media.
    2. PUT video to the pre-signed upload_url.
    3. Poll /v5/media/{media_id} until 'succeeded'.
    4. POST /v5/pins with cover_image_url (MANDATORY).

    Args:
      board_id:         Target board ID.
      video_file_path:  Absolute path to the video file.
      cover_image_url:  MANDATORY public URL for the video cover image.
                              Pinterest API rejects video pins without this.
      title:            Pin title.
      description:      Optional description.

    Returns:
      Standard response dict.

    Raises:
      ValidationError: if cover_image_url is missing or empty.
      PollingTimeoutError: if video processing times out.
    """
    if not cover_image_url or not cover_image_url.strip():
      raise ValidationError(
        "cover_image_url is mandatory for Pinterest Video Pins. "
        "The API rejects video pins without a cover image."
      )

    if self.simulation_mode:
      return self._simulate_post(
        {
          "board_id": board_id,
          "video_file_path": video_file_path,
          "cover_image_url": cover_image_url,
          "title": title,
        }
      )

    if not os.path.exists(video_file_path):
      raise FileNotFoundError(f"Video file not found: {video_file_path}")

    try:
      # Step 1: Register video
      file_size = os.path.getsize(video_file_path)
      async with httpx.AsyncClient() as client:
        register_response = await client.post(
          f"{PINTEREST_API_BASE}/v5/media",
          headers={**self._auth_headers(), "Content-Type": "application/json"},
          json={"media_type": "video"},
          timeout=15.0,
        )
      if register_response.status_code == 429:
        raise RateLimitError(
          "Pinterest rate limit hit during media registration."
        )
      register_response.raise_for_status()

      register_data = register_response.json()
      media_id = register_data["media_id"]
      upload_url = register_data["upload_url"]
      logger.info(
        "Pinterest media registered: {mid}, uploading {sz} bytes.",
        mid=media_id,
        sz=file_size,
      )

      # Step 2: Upload video to pre-signed URL
      with open(video_file_path, "rb") as video_file:
        async with httpx.AsyncClient() as client:
          upload_response = await client.put(
            upload_url,
            content=video_file.read(),
            headers={"Content-Type": "video/mp4"},
            timeout=300.0,
          )
      upload_response.raise_for_status()
      logger.info("Pinterest video uploaded for media_id={mid}.", mid=media_id)

      # Step 3: Poll for processing completion
      await self._poll_media_registration(media_id)

      # Step 4: Create the Video Pin
      async with httpx.AsyncClient() as client:
        pin_response = await client.post(
          f"{PINTEREST_API_BASE}/v5/pins",
          headers={**self._auth_headers(), "Content-Type": "application/json"},
          json={
            "board_id": board_id,
            "title": title,
            "description": description,
            "cover_image_url": cover_image_url,
            "media_source": {
              "source_type": "video_id",
              "cover_image_url": cover_image_url,
              "media_id": media_id,
            },
          },
          timeout=15.0,
        )
      if pin_response.status_code == 429:
        raise RateLimitError(
          "Pinterest rate limit hit during video pin creation."
        )
      pin_response.raise_for_status()

    except (RateLimitError, PollingTimeoutError, ValidationError):
      raise
    except Exception as exc:
      logger.error("Pinterest create_video_pin error: {exc}", exc=exc)
      return self._build_error_response(str(exc))

    pin_id = pin_response.json().get("id", "")
    url = f"https://www.pinterest.com/pin/{pin_id}/"
    logger.info("Pinterest Video Pin created: {pid}", pid=pin_id)
    return self._build_success_response(post_id=pin_id, url=url)

  async def get_boards(
    self,
    page_size: int = 25,
    bookmark: Optional[str] = None,
  ) -> Dict[str, Any]:
    """
    Retrieve the authenticated user's Pinterest boards.

    Args:
      page_size: Number of boards per page (max 250).
      bookmark:  Pagination cursor from a previous call.

    Returns:
      Dict with "items" (list of board dicts) and "bookmark" for next page.
    """
    if self.simulation_mode:
      return {
        "items": [
          {"id": "sim_board_1", "name": "Simulated Board", "pin_count": 0}
        ],
        "bookmark": None,
      }

    params: Dict[str, Any] = {"page_size": page_size}
    if bookmark:
      params["bookmark"] = bookmark

    try:
      async with httpx.AsyncClient() as client:
        response = await client.get(
          f"{PINTEREST_API_BASE}/v5/boards",
          headers=self._auth_headers(),
          params=params,
          timeout=15.0,
        )
      if response.status_code == 429:
        raise RateLimitError("Pinterest rate limit hit during get_boards.")
      response.raise_for_status()
    except RateLimitError:
      raise
    except Exception as exc:
      logger.error("Pinterest get_boards error: {exc}", exc=exc)
      return {"items": [], "bookmark": None, "error": str(exc)}

    data = response.json()
    boards = data.get("items", [])
    logger.info("Pinterest get_boards returned {n} boards.", n=len(boards))
    return {"items": boards, "bookmark": data.get("bookmark")}

  async def get_rate_limit_status(self) -> Dict[str, Any]:
    """Return Pinterest API rate limit information."""
    return {
      "requests_remaining": "check X-RateLimit-Remaining response header",
      "reset_at": "check X-RateLimit-Reset response header",
      "daily_limit": 1000,
      "burst_limit_per_second": 10,
    }


def _register() -> None:
  """Register PinterestPoster with the global PosterRegistry."""
  PosterRegistry.register("pinterest", PinterestPoster)
  logger.debug("PinterestPoster registered.")


_register()
