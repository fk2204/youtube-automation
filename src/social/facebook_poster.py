"""
FacebookPoster — posts videos, Reels, text, and links to Facebook Pages.

IMPORTANT: This poster requires a Page Access Token, NOT a User Access Token.
authenticate() enforces this by checking the token type via the Graph API
debug_token endpoint. Passing a User token raises AuthError immediately.

API flows:

Video upload (< 1 GB):
  POST /{page-id}/videos with source or file_url.

Video upload (>= 1 GB) — Resumable upload:
  1. POST /{page-id}/videos?upload_phase=start&file_size={size}
     Returns upload_session_id and start_offset.
  2. POST /{page-id}/videos?upload_phase=transfer&upload_session_id={id}
     with Content-Range and chunk bytes.
     Repeat until start_offset == file_size.
  3. POST /{page-id}/videos?upload_phase=finish&upload_session_id={id}
     with title, description.

Reel upload:
  Uses the /reels/upload endpoint (requires reels.publish permission).

Text / link post:
  POST /{page-id}/feed with message and optional link.

Rate limits:
  - 200 calls per hour per user token (shared Graph API quota)
  - Video uploads count against Page-level publishing quotas

Credentials (env vars):
  FACEBOOK_PAGE_ACCESS_TOKEN  — Page Access Token (NOT User token)
  FACEBOOK_PAGE_ID            — Numeric Facebook Page ID
"""

from __future__ import annotations

import asyncio
import math
import os
from typing import Any, Dict, Optional

import httpx
from loguru import logger

from exceptions import AuthError, FileSizeError, RateLimitError, UploadError, ValidationError
from registry.poster_registry import PosterRegistry
from social.base_poster import BasePoster
from social.social_platform import SocialPlatform

GRAPH_API_BASE = "https://graph.facebook.com/v21.0"

# Resumable upload threshold: files >= this size use the resumable protocol.
RESUMABLE_THRESHOLD_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB

# Resumable chunk size
CHUNK_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB per chunk

# Max video size Facebook accepts
MAX_VIDEO_SIZE_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB


class FacebookPoster(SocialPlatform, BasePoster):
  """
  Posts videos, Reels, text, and links to a Facebook Page.

  Requires a Page Access Token — NOT a User Access Token.
  authenticate() enforces this requirement and will raise AuthError
  if a User token is provided.

  Requires env vars:
    FACEBOOK_PAGE_ACCESS_TOKEN  — Page Access Token
    FACEBOOK_PAGE_ID            — Numeric Page ID
  """

  platform_name = "facebook"

  def __init__(self) -> None:
    self._page_token: Optional[str] = None
    self._page_id: Optional[str] = None
    self._authenticated: bool = False

  async def authenticate(self) -> None:
    """
    Load credentials and enforce Page Access Token requirement.

    Verifies the token type via the Graph API debug_token endpoint.
    Raises AuthError if the token is a User token instead of a Page token.

    Raises:
      AuthError: on missing env vars, invalid token, or User token passed.
    """
    if self.guard_unconfigured(
      "FACEBOOK_PAGE_ACCESS_TOKEN",
      "FACEBOOK_PAGE_ID",
    ):
      raise AuthError(
        "Facebook credentials not configured. "
        "Set FACEBOOK_PAGE_ACCESS_TOKEN and FACEBOOK_PAGE_ID."
      )

    token = os.environ["FACEBOOK_PAGE_ACCESS_TOKEN"]
    page_id = os.environ["FACEBOOK_PAGE_ID"]

    # Inspect the token type — must be PAGE token.
    try:
      async with httpx.AsyncClient() as client:
        response = await client.get(
          f"{GRAPH_API_BASE}/debug_token",
          params={
            "input_token": token,
            "access_token": token,
          },
          timeout=10.0,
        )
      if response.status_code == 401:
        raise AuthError(
          "Facebook token rejected (401). "
          "Ensure FACEBOOK_PAGE_ACCESS_TOKEN is valid."
        )
      if response.status_code == 429:
        raise RateLimitError("Facebook rate limit hit during auth check.")
      response.raise_for_status()
    except (AuthError, RateLimitError):
      raise
    except Exception as exc:
      raise AuthError(f"Facebook auth validation failed: {exc}") from exc

    token_data = response.json().get("data", {})
    token_type = token_data.get("type", "")

    # PAGE tokens have type "PAGE". USER tokens have type "USER".
    # Reject anything that is not a PAGE token.
    if token_type.upper() == "USER":
      raise AuthError(
        "FACEBOOK_PAGE_ACCESS_TOKEN is a User token, not a Page token. "
        "Exchange for a Page Access Token before using FacebookPoster."
      )

    self._page_token = token
    self._page_id = page_id
    self._authenticated = True
    logger.info(
      "Facebook authenticated for page {pid} (token type: {t}).",
      pid=page_id,
      t=token_type,
    )

  async def _resumable_upload(
    self,
    file_path: str,
    title: str,
    description: str,
  ) -> str:
    """
    Upload a large video (>= 1 GB) using Facebook's resumable upload protocol.

    Three-phase: start → transfer (loop) → finish.
    Returns the video_id from the finish phase.

    Raises:
      UploadError: on any phase failure.
    """
    file_size = os.path.getsize(file_path)

    # Phase 1: Start
    try:
      async with httpx.AsyncClient() as client:
        start_response = await client.post(
          f"{GRAPH_API_BASE}/{self._page_id}/videos",
          params={
            "upload_phase": "start",
            "file_size": str(file_size),
            "access_token": self._page_token,
          },
          timeout=30.0,
        )
      start_response.raise_for_status()
    except Exception as exc:
      raise UploadError(
        f"Facebook resumable upload start failed: {exc}"
      ) from exc

    start_data = start_response.json()
    session_id = start_data["upload_session_id"]
    start_offset = int(start_data["start_offset"])
    logger.info(
      "Facebook resumable upload session started: {sid}",
      sid=session_id,
    )

    # Phase 2: Transfer chunks
    total_chunks = math.ceil(file_size / CHUNK_SIZE_BYTES)
    with open(file_path, "rb") as video_file:
      for chunk_index in range(total_chunks):
        chunk_data = video_file.read(CHUNK_SIZE_BYTES)
        end_offset = start_offset + len(chunk_data)
        content_range = f"bytes {start_offset}-{end_offset - 1}/{file_size}"

        try:
          async with httpx.AsyncClient() as client:
            transfer_response = await client.post(
              f"{GRAPH_API_BASE}/{self._page_id}/videos",
              params={
                "upload_phase": "transfer",
                "upload_session_id": session_id,
                "start_offset": str(start_offset),
                "access_token": self._page_token,
              },
              content=chunk_data,
              headers={"Content-Range": content_range},
              timeout=120.0,
            )
          if transfer_response.status_code == 429:
            raise RateLimitError(
              "Facebook rate limit hit during chunk transfer."
            )
          transfer_response.raise_for_status()
        except RateLimitError:
          raise
        except Exception as exc:
          raise UploadError(
            f"Facebook chunk {chunk_index + 1}/{total_chunks} failed: {exc}"
          ) from exc

        start_offset = int(transfer_response.json().get("start_offset", end_offset))
        logger.debug(
          "Facebook chunk {idx}/{total} transferred.",
          idx=chunk_index + 1,
          total=total_chunks,
        )

    # Phase 3: Finish
    try:
      async with httpx.AsyncClient() as client:
        finish_response = await client.post(
          f"{GRAPH_API_BASE}/{self._page_id}/videos",
          params={
            "upload_phase": "finish",
            "upload_session_id": session_id,
            "title": title,
            "description": description,
            "access_token": self._page_token,
          },
          timeout=30.0,
        )
      finish_response.raise_for_status()
    except Exception as exc:
      raise UploadError(
        f"Facebook resumable upload finish failed: {exc}"
      ) from exc

    video_id = finish_response.json().get("video_id", session_id)
    logger.info(
      "Facebook resumable upload finished: video_id={vid}",
      vid=video_id,
    )
    return str(video_id)

  async def post_video(
    self,
    file_path: str,
    title: str,
    description: str = "",
  ) -> Dict[str, Any]:
    """
    Upload and post a video to the Facebook Page.

    Files >= 1 GB use the resumable upload protocol automatically.
    Files < 1 GB are uploaded in a single POST.

    Args:
      file_path:   Absolute path to the video file.
      title:       Video title shown in the Facebook post.
      description: Optional video description.

    Returns:
      Standard response dict with video_id as post_id.

    Raises:
      FileSizeError: if file exceeds 10 GB.
      RateLimitError: on 429 response.
      UploadError: on upload failure.
    """
    if self.simulation_mode:
      return self._simulate_post(
        {"file_path": file_path, "title": title, "description": description}
      )

    if not os.path.exists(file_path):
      raise FileNotFoundError(f"Video file not found: {file_path}")

    file_size = os.path.getsize(file_path)
    if file_size > MAX_VIDEO_SIZE_BYTES:
      raise FileSizeError(
        f"Video size {file_size} bytes exceeds Facebook limit of 10 GB."
      )

    try:
      if file_size >= RESUMABLE_THRESHOLD_BYTES:
        logger.info(
          "File size {s} bytes >= 1 GB, using resumable upload.",
          s=file_size,
        )
        video_id = await self._resumable_upload(file_path, title, description)
      else:
        async with httpx.AsyncClient() as client:
          with open(file_path, "rb") as video_file:
            response = await client.post(
              f"{GRAPH_API_BASE}/{self._page_id}/videos",
              data={
                "title": title,
                "description": description,
                "access_token": self._page_token,
              },
              files={"source": ("video.mp4", video_file, "video/mp4")},
              timeout=300.0,
            )
        if response.status_code == 429:
          raise RateLimitError("Facebook rate limit hit during video upload.")
        response.raise_for_status()
        video_id = str(response.json().get("id"))

      url = f"https://www.facebook.com/{self._page_id}/videos/{video_id}/"
      return self._build_success_response(post_id=video_id, url=url)

    except (FileSizeError, RateLimitError, UploadError):
      raise
    except Exception as exc:
      logger.error("Facebook post_video unexpected error: {exc}", exc=exc)
      return self._build_error_response(str(exc))

  async def post_reel(
    self,
    video_url: str,
    description: str = "",
  ) -> Dict[str, Any]:
    """
    Post a Reel to the Facebook Page using a public video URL.

    Requires the reels.publish permission on the Page token.

    Args:
      video_url:   Publicly accessible URL of the video.
      description: Reel caption.

    Returns:
      Standard response dict.
    """
    if self.simulation_mode:
      return self._simulate_post(
        {"video_url": video_url, "description": description}
      )

    try:
      async with httpx.AsyncClient() as client:
        response = await client.post(
          f"{GRAPH_API_BASE}/{self._page_id}/video_reels",
          data={
            "upload_phase": "finish",
            "video_url": video_url,
            "description": description,
            "access_token": self._page_token,
          },
          timeout=30.0,
        )
      if response.status_code == 429:
        raise RateLimitError("Facebook rate limit hit during Reel post.")
      response.raise_for_status()
    except RateLimitError:
      raise
    except Exception as exc:
      logger.error("Facebook post_reel error: {exc}", exc=exc)
      return self._build_error_response(str(exc))

    reel_id = str(response.json().get("video_id", ""))
    url = f"https://www.facebook.com/reel/{reel_id}"
    return self._build_success_response(post_id=reel_id, url=url)

  async def post_text(
    self,
    message: str,
  ) -> Dict[str, Any]:
    """
    Post a text-only update to the Facebook Page feed.

    Args:
      message: Text content of the post.

    Returns:
      Standard response dict with post_id.
    """
    if self.simulation_mode:
      return self._simulate_post({"message": message})

    try:
      async with httpx.AsyncClient() as client:
        response = await client.post(
          f"{GRAPH_API_BASE}/{self._page_id}/feed",
          data={
            "message": message,
            "access_token": self._page_token,
          },
          timeout=15.0,
        )
      if response.status_code == 429:
        raise RateLimitError("Facebook rate limit hit during text post.")
      response.raise_for_status()
    except RateLimitError:
      raise
    except Exception as exc:
      logger.error("Facebook post_text error: {exc}", exc=exc)
      return self._build_error_response(str(exc))

    post_id = str(response.json().get("id", ""))
    url = f"https://www.facebook.com/{post_id}"
    return self._build_success_response(post_id=post_id, url=url)

  async def post_link(
    self,
    link: str,
    message: str = "",
  ) -> Dict[str, Any]:
    """
    Post a link to the Facebook Page feed.

    Facebook generates a preview card automatically from the URL's Open Graph tags.

    Args:
      link:    The URL to share.
      message: Optional text accompanying the link.

    Returns:
      Standard response dict.
    """
    if self.simulation_mode:
      return self._simulate_post({"link": link, "message": message})

    try:
      async with httpx.AsyncClient() as client:
        response = await client.post(
          f"{GRAPH_API_BASE}/{self._page_id}/feed",
          data={
            "link": link,
            "message": message,
            "access_token": self._page_token,
          },
          timeout=15.0,
        )
      if response.status_code == 429:
        raise RateLimitError("Facebook rate limit hit during link post.")
      response.raise_for_status()
    except RateLimitError:
      raise
    except Exception as exc:
      logger.error("Facebook post_link error: {exc}", exc=exc)
      return self._build_error_response(str(exc))

    post_id = str(response.json().get("id", ""))
    url = f"https://www.facebook.com/{post_id}"
    return self._build_success_response(post_id=post_id, url=url)

  async def get_rate_limit_status(self) -> Dict[str, Any]:
    """Return Facebook Graph API rate limit information."""
    return {
      "requests_remaining": "check X-App-Usage header",
      "reset_at": "rolling 1-hour window",
      "window_seconds": 3600,
    }


def _register() -> None:
  """Register FacebookPoster with the global PosterRegistry."""
  PosterRegistry.register("facebook", FacebookPoster)
  logger.debug("FacebookPoster registered.")


_register()
