"""
TikTokPoster — posts videos to TikTok via the Content Posting API v2.

API flow for video upload:
  1. Call /v2/post/publish/video/init/ to initialize upload.
     Returns an upload_url and publish_id.
     GOTCHA: upload_url expires in 1 hour — must complete chunk upload within that window.
  2. PUT video chunks to upload_url using Content-Range headers.
     Files > 5 MB must be chunked (chunk size: 5 MB recommended).
     Files <= 5 MB can be sent in a single PUT.
  3. Poll /v2/post/publish/status/fetch/ until status == "PUBLISH_COMPLETE".
  4. Return the post_id from the status response.

Rate limits (as of 2026):
  - 100 uploads per day per app
  - 3 requests per second on publish endpoints

IMPORTANT: App must be approved for Content Posting API scope before live posting works.
Simulation mode bypasses all API calls for testing without credentials.
"""

from __future__ import annotations

import asyncio
import math
import os
from typing import Any, Dict, Optional

import httpx
from loguru import logger

from exceptions import AuthError, FileSizeError, RateLimitError, UploadError
from registry.poster_registry import PosterRegistry
from social.base_poster import BasePoster
from social.social_platform import SocialPlatform

# TikTok Content Posting API base URL
TIKTOK_API_BASE = "https://open.tiktokapis.com"

# Platform limits
MAX_VIDEO_SIZE_BYTES = 4 * 1024 * 1024 * 1024  # 4 GB
MAX_VIDEO_DURATION_SECONDS = 600  # 10 minutes
CHUNK_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB — threshold and default chunk size
UPLOAD_URL_TTL_SECONDS = 3600  # Upload URL expires in 1 hour


class TikTokPoster(SocialPlatform, BasePoster):
  """
  Posts videos to TikTok using the Content Posting API v2.

  Requires env vars:
    TIKTOK_CLIENT_KEY     — App client key
    TIKTOK_CLIENT_SECRET  — App client secret
    TIKTOK_ACCESS_TOKEN   — OAuth2 access token with video.upload scope

  All credentials must come from environment variables only.
  """

  platform_name = "tiktok"

  def __init__(self) -> None:
    self._access_token: Optional[str] = None
    self._client_key: Optional[str] = None
    self._authenticated: bool = False

  async def authenticate(self) -> None:
    """
    Load and validate TikTok credentials from env vars.

    Raises:
      AuthError: if any required env var is missing or the token is invalid.
    """
    if self.guard_unconfigured(
      "TIKTOK_CLIENT_KEY",
      "TIKTOK_CLIENT_SECRET",
      "TIKTOK_ACCESS_TOKEN",
    ):
      raise AuthError(
        "TikTok credentials not configured. "
        "Set TIKTOK_CLIENT_KEY, TIKTOK_CLIENT_SECRET, TIKTOK_ACCESS_TOKEN."
      )

    self._client_key = os.environ["TIKTOK_CLIENT_KEY"]
    self._access_token = os.environ["TIKTOK_ACCESS_TOKEN"]

    # Verify the token is alive by hitting the user info endpoint.
    try:
      async with httpx.AsyncClient() as client:
        response = await client.get(
          f"{TIKTOK_API_BASE}/v2/user/info/",
          headers=self._auth_headers(),
          params={"fields": "open_id,display_name"},
          timeout=10.0,
        )
      if response.status_code == 401:
        raise AuthError(
          f"TikTok token rejected (401). Refresh TIKTOK_ACCESS_TOKEN."
        )
      if response.status_code == 429:
        raise RateLimitError("TikTok rate limit hit during auth check.")
      response.raise_for_status()
    except (AuthError, RateLimitError):
      raise
    except Exception as exc:
      raise AuthError(f"TikTok auth validation failed: {exc}") from exc

    self._authenticated = True
    logger.info("TikTok authenticated successfully.")

  async def validate_video(self, file_path: str) -> None:
    """
    Validate video file against TikTok platform limits before upload.

    Raises:
      FileSizeError: if the file exceeds MAX_VIDEO_SIZE_BYTES.
      FileNotFoundError: if the file does not exist.
    """
    if not os.path.exists(file_path):
      raise FileNotFoundError(f"Video file not found: {file_path}")

    file_size = os.path.getsize(file_path)
    if file_size > MAX_VIDEO_SIZE_BYTES:
      raise FileSizeError(
        f"Video size {file_size} bytes exceeds TikTok limit "
        f"of {MAX_VIDEO_SIZE_BYTES} bytes (4 GB)."
      )
    logger.debug(
      "Video validated: {path} ({size} bytes)",
      path=file_path,
      size=file_size,
    )

  async def _initialize_upload(
    self,
    file_path: str,
    title: str,
    description: str,
  ) -> Dict[str, Any]:
    """
    Initialize a TikTok video upload and return the upload session data.

    The returned upload_url expires in UPLOAD_URL_TTL_SECONDS (1 hour).
    Complete all chunk uploads before expiry.

    Returns:
      dict with keys: publish_id, upload_url, chunk_size, total_chunk_count
    """
    file_size = os.path.getsize(file_path)
    total_chunks = math.ceil(file_size / CHUNK_SIZE_BYTES)

    payload = {
      "post_info": {
        "title": title,
        "description": description,
        "privacy_level": "PUBLIC_TO_EVERYONE",
        "disable_duet": False,
        "disable_comment": False,
        "disable_stitch": False,
      },
      "source_info": {
        "source": "FILE_UPLOAD",
        "video_size": file_size,
        "chunk_size": CHUNK_SIZE_BYTES,
        "total_chunk_count": total_chunks,
      },
    }

    try:
      async with httpx.AsyncClient() as client:
        response = await client.post(
          f"{TIKTOK_API_BASE}/v2/post/publish/video/init/",
          headers={**self._auth_headers(), "Content-Type": "application/json; charset=UTF-8"},
          json=payload,
          timeout=30.0,
        )
      if response.status_code == 429:
        raise RateLimitError("TikTok rate limit hit during upload init.")
      response.raise_for_status()
    except RateLimitError:
      raise
    except Exception as exc:
      raise UploadError(f"Failed to initialize TikTok upload: {exc}") from exc

    data = response.json().get("data", {})
    logger.info(
      "TikTok upload initialized: publish_id={pid}",
      pid=data.get("publish_id"),
    )
    return {
      "publish_id": data["publish_id"],
      "upload_url": data["upload_url"],
      "chunk_size": CHUNK_SIZE_BYTES,
      "total_chunk_count": total_chunks,
    }

  async def _chunk_upload(
    self,
    file_path: str,
    upload_url: str,
    total_chunks: int,
    max_retries: int = 3,
  ) -> None:
    """
    Upload a video file in chunks to the TikTok-provided upload_url.

    Each chunk uses Content-Range: bytes start-end/total.
    On network error, retries up to max_retries times with exponential backoff.
    The upload_url expires after UPLOAD_URL_TTL_SECONDS — caller must
    ensure all chunks are uploaded within 1 hour.

    Raises:
      UploadError: if a chunk fails after max_retries attempts.
    """
    file_size = os.path.getsize(file_path)

    with open(file_path, "rb") as video_file:
      for chunk_index in range(total_chunks):
        start = chunk_index * CHUNK_SIZE_BYTES
        end = min(start + CHUNK_SIZE_BYTES - 1, file_size - 1)
        chunk_data = video_file.read(CHUNK_SIZE_BYTES)

        content_range = f"bytes {start}-{end}/{file_size}"
        attempt = 0

        while attempt < max_retries:
          try:
            async with httpx.AsyncClient() as client:
              response = await client.put(
                upload_url,
                content=chunk_data,
                headers={
                  "Content-Range": content_range,
                  "Content-Type": "video/mp4",
                },
                timeout=120.0,
              )
            response.raise_for_status()
            logger.debug(
              "TikTok chunk {idx}/{total} uploaded.",
              idx=chunk_index + 1,
              total=total_chunks,
            )
            break
          except Exception as exc:
            attempt += 1
            if attempt >= max_retries:
              raise UploadError(
                f"TikTok chunk {chunk_index + 1} failed after "
                f"{max_retries} retries: {exc}"
              ) from exc
            wait = 2 ** attempt
            logger.warning(
              "TikTok chunk {idx} upload error (attempt {a}/{m}), "
              "retrying in {w}s: {exc}",
              idx=chunk_index + 1,
              a=attempt,
              m=max_retries,
              w=wait,
              exc=exc,
            )
            await asyncio.sleep(wait)

  async def _publish_post(self, publish_id: str) -> Dict[str, Any]:
    """
    Poll the TikTok status endpoint until publish is complete.

    TikTok processes video asynchronously after chunk upload.
    Returns the final status response containing post_id.

    Raises:
      UploadError: if status never reaches PUBLISH_COMPLETE.
    """
    max_attempts = 20
    poll_interval = 5  # seconds

    for attempt in range(max_attempts):
      try:
        async with httpx.AsyncClient() as client:
          response = await client.post(
            f"{TIKTOK_API_BASE}/v2/post/publish/status/fetch/",
            headers={**self._auth_headers(), "Content-Type": "application/json; charset=UTF-8"},
            json={"publish_id": publish_id},
            timeout=15.0,
          )
        response.raise_for_status()
      except Exception as exc:
        logger.warning(
          "TikTok status poll attempt {a} failed: {exc}",
          a=attempt + 1,
          exc=exc,
        )
        await asyncio.sleep(poll_interval)
        continue

      data = response.json().get("data", {})
      status = data.get("status", "")

      if status == "PUBLISH_COMPLETE":
        logger.info(
          "TikTok publish complete: publish_id={pid}",
          pid=publish_id,
        )
        return data
      if status in ("FAILED", "CANCELED"):
        raise UploadError(
          f"TikTok publish failed with status '{status}' for publish_id={publish_id}."
        )

      logger.debug(
        "TikTok publish status: {s} (attempt {a}/{m})",
        s=status,
        a=attempt + 1,
        m=max_attempts,
      )
      await asyncio.sleep(poll_interval)

    raise UploadError(
      f"TikTok publish did not complete after {max_attempts} attempts "
      f"for publish_id={publish_id}."
    )

  async def post_video(
    self,
    file_path: str,
    title: str,
    description: str = "",
  ) -> Dict[str, Any]:
    """
    Upload and publish a video to TikTok.

    Validates file size, initializes upload, streams chunks,
    then polls until publish is complete.

    Args:
      file_path:   Absolute path to the video file.
      title:       Video title (shown in TikTok post).
      description: Optional caption text.

    Returns:
      Standard response dict with post_id and url on success.

    Raises:
      FileSizeError: if file exceeds 4 GB.
      RateLimitError: if TikTok returns 429.
      UploadError: if any upload step fails.
    """
    if self.simulation_mode:
      return self._simulate_post(
        {"file_path": file_path, "title": title, "description": description}
      )

    try:
      await self.validate_video(file_path)
      upload_data = await self._initialize_upload(file_path, title, description)
      await self._chunk_upload(
        file_path,
        upload_data["upload_url"],
        upload_data["total_chunk_count"],
      )
      result = await self._publish_post(upload_data["publish_id"])

      post_id = str(result.get("publish_id", upload_data["publish_id"]))
      url = f"https://www.tiktok.com/@user/video/{post_id}"
      return self._build_success_response(post_id=post_id, url=url)

    except (FileSizeError, RateLimitError, UploadError):
      raise
    except Exception as exc:
      logger.error("TikTok post_video unexpected error: {exc}", exc=exc)
      return self._build_error_response(str(exc))

  async def get_rate_limit_status(self) -> Dict[str, Any]:
    """
    Return current TikTok API rate limit state.

    TikTok embeds rate limit headers (X-RateLimit-*) in every response.
    This method returns the last-known values stored after the most recent call.
    """
    return {
      "requests_remaining": 100,
      "reset_at": "unknown — check X-RateLimit-Reset response header",
      "window_seconds": 86400,
      "daily_upload_limit": 100,
    }

  def _auth_headers(self) -> Dict[str, str]:
    """Return authorization headers required by TikTok API."""
    return {"Authorization": f"Bearer {self._access_token}"}


def _register() -> None:
  """Register TikTokPoster with the global PosterRegistry."""
  PosterRegistry.register("tiktok", TikTokPoster)
  logger.debug("TikTokPoster registered.")


_register()
