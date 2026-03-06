"""
InstagramPoster — posts Reels, carousels, and images via the Instagram Graph API.

API flow for Reels / single image:
  1. POST /v21.0/{ig-user-id}/media — Create a media container.
     Returns container_id (also called creation_id).
  2. Poll GET /v21.0/{container_id}?fields=status_code until status_code == FINISHED.
     Container expires in 24 hours if not published.
     Poll max 10 times (configurable) with 10-second intervals.
     CRITICAL: Wait at least 10 seconds after container creation before publishing.
  3. POST /v21.0/{ig-user-id}/media_publish — Publish the container.
     Returns media_id.
  4. Construct permalink from media_id or fetch from GET /v21.0/{media_id}?fields=permalink.

API flow for carousel:
  Same as above but first create up to 10 child containers (one per item),
  then create a parent carousel container referencing child IDs.
  Max 10 items — enforced in post_carousel().

IMPORTANT — Music limitation:
  The Instagram Graph API does NOT provide access to Instagram's licensed music library.
  Audio must be embedded in the uploaded video file itself.
  Any music in Reels must be owned by the poster or royalty-free.
  This is a permanent API limitation as of 2026, not a temporary restriction.

Rate limits:
  - 200 API calls per hour per user token (shared across all Graph API calls).
  - Media publishing is limited to 50 posts per 24-hour period per account.

Credentials (env vars):
  INSTAGRAM_ACCESS_TOKEN          — Long-lived user or page access token
  INSTAGRAM_BUSINESS_ACCOUNT_ID   — Numeric IG Business / Creator account ID
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

GRAPH_API_BASE = "https://graph.facebook.com/v21.0"

# Polling configuration
MAX_POLL_ATTEMPTS = 10
POLL_INTERVAL_SECONDS = 10  # Instagram requires >= 10 seconds between poll and publish

# Carousel limits
MAX_CAROUSEL_ITEMS = 10

# Supported media formats
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png"}
SUPPORTED_VIDEO_FORMATS = {".mp4", ".mov"}


class InstagramPoster(SocialPlatform, BasePoster):
  """
  Posts Reels, carousels, and images to an Instagram Business/Creator account.

  Requires env vars:
    INSTAGRAM_ACCESS_TOKEN          — Valid Graph API access token
    INSTAGRAM_BUSINESS_ACCOUNT_ID   — IG Business account ID (numeric string)

  IMPORTANT: Licensed music is NOT available via the Graph API.
  Any audio must be embedded in the video file before upload.
  """

  platform_name = "instagram"

  def __init__(self) -> None:
    self._access_token: Optional[str] = None
    self._account_id: Optional[str] = None
    self._authenticated: bool = False

  async def authenticate(self) -> None:
    """
    Load credentials from env vars and verify account access.

    Raises:
      AuthError: if credentials are missing or account is unreachable.
    """
    if self.guard_unconfigured(
      "INSTAGRAM_ACCESS_TOKEN",
      "INSTAGRAM_BUSINESS_ACCOUNT_ID",
    ):
      raise AuthError(
        "Instagram credentials not configured. "
        "Set INSTAGRAM_ACCESS_TOKEN and INSTAGRAM_BUSINESS_ACCOUNT_ID."
      )

    self._access_token = os.environ["INSTAGRAM_ACCESS_TOKEN"]
    self._account_id = os.environ["INSTAGRAM_BUSINESS_ACCOUNT_ID"]

    try:
      async with httpx.AsyncClient() as client:
        response = await client.get(
          f"{GRAPH_API_BASE}/{self._account_id}",
          params={
            "fields": "id,name,username",
            "access_token": self._access_token,
          },
          timeout=10.0,
        )
      if response.status_code == 401:
        raise AuthError(
          "Instagram access token rejected (401). "
          "Refresh INSTAGRAM_ACCESS_TOKEN."
        )
      if response.status_code == 429:
        raise RateLimitError("Instagram rate limit hit during auth check.")
      response.raise_for_status()
    except (AuthError, RateLimitError):
      raise
    except Exception as exc:
      raise AuthError(f"Instagram auth validation failed: {exc}") from exc

    self._authenticated = True
    logger.info("Instagram authenticated for account {id}.", id=self._account_id)

  def _validate_media(self, file_path: str, allowed_formats: set[str]) -> None:
    """
    Validate a media file exists and has a supported format.

    Raises:
      ValidationError: if the format is not in allowed_formats.
      FileNotFoundError: if the file does not exist.
    """
    if not os.path.exists(file_path):
      raise FileNotFoundError(f"Media file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in allowed_formats:
      raise ValidationError(
        f"Unsupported format '{ext}' for Instagram. "
        f"Allowed: {sorted(allowed_formats)}"
      )

  async def _poll_container_status(
    self,
    container_id: str,
    max_attempts: int = MAX_POLL_ATTEMPTS,
    poll_interval: int = POLL_INTERVAL_SECONDS,
  ) -> None:
    """
    Poll the container status until it reaches FINISHED or ERROR.

    Instagram processes uploaded media asynchronously. The container
    must reach status FINISHED before media_publish can be called.
    The container expires 24 hours after creation.

    Args:
      container_id:  The creation_id returned from the media endpoint.
      max_attempts:  How many times to poll before giving up.
      poll_interval: Seconds between each poll attempt.

    Raises:
      PollingTimeoutError: if FINISHED is not reached within max_attempts.
      ValidationError: if the container enters ERROR status.
    """
    for attempt in range(max_attempts):
      await asyncio.sleep(poll_interval)

      try:
        async with httpx.AsyncClient() as client:
          response = await client.get(
            f"{GRAPH_API_BASE}/{container_id}",
            params={
              "fields": "status_code",
              "access_token": self._access_token,
            },
            timeout=15.0,
          )
        response.raise_for_status()
      except Exception as exc:
        logger.warning(
          "Container poll attempt {a} failed for {cid}: {exc}",
          a=attempt + 1,
          cid=container_id,
          exc=exc,
        )
        continue

      status_code = response.json().get("status_code", "")
      logger.debug(
        "Container {cid} status: {s} (attempt {a}/{m})",
        cid=container_id,
        s=status_code,
        a=attempt + 1,
        m=max_attempts,
      )

      if status_code == "FINISHED":
        return
      if status_code == "ERROR":
        raise ValidationError(
          f"Instagram container {container_id} entered ERROR status."
        )

    raise PollingTimeoutError(
      f"Instagram container {container_id} did not reach FINISHED "
      f"after {max_attempts} attempts."
    )

  async def _create_container(self, params: Dict[str, Any]) -> str:
    """
    POST to the media endpoint to create a container.

    Returns:
      container_id (str)

    Raises:
      RateLimitError: on 429 response.
      ValidationError: on API error.
    """
    params["access_token"] = self._access_token

    try:
      async with httpx.AsyncClient() as client:
        response = await client.post(
          f"{GRAPH_API_BASE}/{self._account_id}/media",
          data=params,
          timeout=30.0,
        )
      if response.status_code == 429:
        raise RateLimitError("Instagram rate limit hit during container creation.")
      response.raise_for_status()
    except RateLimitError:
      raise
    except Exception as exc:
      raise ValidationError(
        f"Instagram container creation failed: {exc}"
      ) from exc

    container_id = response.json().get("id")
    logger.info("Instagram container created: {cid}", cid=container_id)
    return container_id

  async def _publish_container(self, container_id: str) -> str:
    """
    Publish a FINISHED container and return the resulting media_id.

    Raises:
      RateLimitError: on 429.
      ValidationError: on API error.
    """
    try:
      async with httpx.AsyncClient() as client:
        response = await client.post(
          f"{GRAPH_API_BASE}/{self._account_id}/media_publish",
          data={
            "creation_id": container_id,
            "access_token": self._access_token,
          },
          timeout=30.0,
        )
      if response.status_code == 429:
        raise RateLimitError("Instagram rate limit hit during media_publish.")
      response.raise_for_status()
    except RateLimitError:
      raise
    except Exception as exc:
      raise ValidationError(
        f"Instagram media_publish failed: {exc}"
      ) from exc

    media_id = response.json().get("id")
    logger.info("Instagram post published: media_id={mid}", mid=media_id)
    return media_id

  async def post_reel(
    self,
    video_url: str,
    caption: str = "",
    cover_url: Optional[str] = None,
  ) -> Dict[str, Any]:
    """
    Post a Reel to Instagram.

    The video_url must be publicly accessible (Instagram fetches it directly).

    IMPORTANT: No licensed music is available via the Graph API.
    Audio must be embedded in the video before upload.

    Args:
      video_url:  Public URL of the video file.
      caption:    Post caption (hashtags and mentions supported).
      cover_url:  Optional public URL for the cover thumbnail.

    Returns:
      Standard response dict with media_id as post_id.
    """
    if self.simulation_mode:
      return self._simulate_post({"video_url": video_url, "caption": caption})

    try:
      container_params: Dict[str, Any] = {
        "media_type": "REELS",
        "video_url": video_url,
        "caption": caption,
      }
      if cover_url:
        container_params["thumb_offset"] = 0
        container_params["cover_url"] = cover_url

      container_id = await self._create_container(container_params)
      await self._poll_container_status(container_id)
      media_id = await self._publish_container(container_id)

      url = f"https://www.instagram.com/p/{media_id}/"
      return self._build_success_response(post_id=media_id, url=url)

    except (RateLimitError, PollingTimeoutError, ValidationError):
      raise
    except Exception as exc:
      logger.error("Instagram post_reel unexpected error: {exc}", exc=exc)
      return self._build_error_response(str(exc))

  async def post_carousel(
    self,
    media_urls: List[str],
    caption: str = "",
    media_type: str = "IMAGE",
  ) -> Dict[str, Any]:
    """
    Post a carousel (album) to Instagram.

    Args:
      media_urls:  List of 2–10 public URLs (images or videos).
      caption:     Caption for the carousel post.
      media_type:  "IMAGE" or "VIDEO" (must be consistent across all items).

    Raises:
      ValidationError: if more than 10 items are provided.

    Returns:
      Standard response dict.
    """
    if len(media_urls) > MAX_CAROUSEL_ITEMS:
      raise ValidationError(
        f"Instagram carousel accepts max {MAX_CAROUSEL_ITEMS} items. "
        f"Received {len(media_urls)}."
      )

    if self.simulation_mode:
      return self._simulate_post({"media_urls": media_urls, "caption": caption})

    try:
      # Create one child container per item.
      child_ids: List[str] = []
      for url in media_urls:
        child_id = await self._create_container(
          {"is_carousel_item": "true", "image_url": url, "media_type": media_type}
        )
        child_ids.append(child_id)

      # Create parent carousel container.
      carousel_id = await self._create_container(
        {
          "media_type": "CAROUSEL",
          "children": ",".join(child_ids),
          "caption": caption,
        }
      )
      await self._poll_container_status(carousel_id)
      media_id = await self._publish_container(carousel_id)

      url = f"https://www.instagram.com/p/{media_id}/"
      return self._build_success_response(post_id=media_id, url=url)

    except (RateLimitError, PollingTimeoutError, ValidationError):
      raise
    except Exception as exc:
      logger.error("Instagram post_carousel unexpected error: {exc}", exc=exc)
      return self._build_error_response(str(exc))

  async def post_image(
    self,
    image_url: str,
    caption: str = "",
  ) -> Dict[str, Any]:
    """
    Post a single image to Instagram.

    Args:
      image_url:  Public URL of the image (JPEG or PNG).
      caption:    Post caption.

    Returns:
      Standard response dict.
    """
    if self.simulation_mode:
      return self._simulate_post({"image_url": image_url, "caption": caption})

    try:
      container_id = await self._create_container(
        {"image_url": image_url, "caption": caption}
      )
      await self._poll_container_status(container_id)
      media_id = await self._publish_container(container_id)

      url = f"https://www.instagram.com/p/{media_id}/"
      return self._build_success_response(post_id=media_id, url=url)

    except (RateLimitError, PollingTimeoutError, ValidationError):
      raise
    except Exception as exc:
      logger.error("Instagram post_image unexpected error: {exc}", exc=exc)
      return self._build_error_response(str(exc))

  async def get_rate_limit_status(self) -> Dict[str, Any]:
    """
    Return Instagram Graph API rate limit info.

    Actual remaining calls are in the X-App-Usage response header.
    """
    return {
      "requests_remaining": "check X-App-Usage header",
      "reset_at": "rolling 1-hour window",
      "window_seconds": 3600,
      "publish_limit_per_24h": 50,
    }


def _register() -> None:
  """Register InstagramPoster with the global PosterRegistry."""
  PosterRegistry.register("instagram", InstagramPoster)
  logger.debug("InstagramPoster registered.")


_register()
