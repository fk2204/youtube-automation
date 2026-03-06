"""
TelegramPoster — sends content to Telegram chats via the Bot API.

API flow:
  1. authenticate() calls getMe to validate the bot token.
     GOTCHA: getMe returns 401 for an invalid token — any other non-2xx
     indicates a network or server-side issue, not an auth failure.
  2. send_video() POSTs to sendVideo with multipart/form-data.
     Max file size: 50 MB. Files larger than this are rejected before upload.
  3. send_message(), send_photo(), send_document() follow the same
     single-request pattern as sendVideo.
  4. get_rate_limit_status() returns static conservative defaults because
     Telegram does not expose rate limit headers; limits are enforced silently
     (bot is temporarily banned for ~24 h on persistent abuse).

Rate limits (as of 2026):
  - 30 messages per second across all chats
  - 1 message per second to the same chat
  - 20 messages per minute to the same group

IMPORTANT: Simulation mode bypasses all API calls for testing without credentials.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx
from loguru import logger

from exceptions import AuthError, FileSizeError, RateLimitError
from registry.poster_registry import PosterRegistry
from social.base_poster import BasePoster
from social.messaging_platform import MessagingPlatform

# Telegram Bot API base URL — token is embedded in the path.
TELEGRAM_API_BASE = "https://api.telegram.org/bot{token}"

# Platform limit for direct file uploads to sendVideo.
MAX_VIDEO_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB


class TelegramPoster(MessagingPlatform, BasePoster):
  """
  Sends videos, messages, photos, and documents to Telegram via the Bot API.

  Requires env vars:
      TELEGRAM_BOT_TOKEN — Bot API token from @BotFather.
      TELEGRAM_CHAT_ID   — Default chat/channel/group ID to send to.

  All credentials must come from environment variables only.
  """

  platform_name = "telegram"

  def __init__(self) -> None:
    self._bot_token: Optional[str] = None
    self._chat_id: Optional[str] = None
    self._authenticated: bool = False

  def _api_url(self, method: str) -> str:
    """Build a full Telegram Bot API endpoint URL for the given method."""
    return f"https://api.telegram.org/bot{self._bot_token}/{method}"

  async def authenticate(self) -> None:
    """
    Load credentials from env vars and validate the bot token via getMe.

    getMe returns the bot's own User object — if it succeeds, the token is live.

    Raises:
        AuthError: if env vars are missing or the token is rejected.
        RateLimitError: if Telegram returns 429 during the auth check.
    """
    if self.guard_unconfigured("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
      raise AuthError(
        "Telegram credentials not configured. "
        "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID."
      )

    self._bot_token = os.environ["TELEGRAM_BOT_TOKEN"]
    self._chat_id = os.environ["TELEGRAM_CHAT_ID"]

    try:
      async with httpx.AsyncClient() as client:
        response = await client.get(
          self._api_url("getMe"),
          timeout=10.0,
        )
      if response.status_code == 401:
        raise AuthError(
          "Telegram bot token rejected (401). "
          "Verify TELEGRAM_BOT_TOKEN from @BotFather."
        )
      if response.status_code == 429:
        raise RateLimitError("Telegram rate limit hit during auth check.")
      response.raise_for_status()
    except (AuthError, RateLimitError):
      raise
    except Exception as exc:
      raise AuthError(f"Telegram auth validation failed: {exc}") from exc

    self._authenticated = True
    logger.info("Telegram authenticated successfully.")

  async def send_video(
    self,
    file_path: str,
    caption: str = "",
    chat_id: Optional[str] = None,
  ) -> Dict[str, Any]:
    """
    Upload and send a video to a Telegram chat via sendVideo.

    Enforces the 50 MB platform limit before opening the file.
    Falls back to _simulate_post() when SOCIAL_SIMULATION_MODE=true.

    Args:
        file_path: Absolute path to the video file.
        caption:   Optional caption shown beneath the video.
        chat_id:   Target chat ID; defaults to TELEGRAM_CHAT_ID env var.

    Returns:
        {"success": bool, "message_id": str|None, "platform": "telegram", "error": str|None}
    """
    if self.simulation_mode:
      return self._build_messaging_simulate({"file_path": file_path, "caption": caption})

    target_chat = chat_id or self._chat_id

    # Enforce file size before touching the API.
    try:
      file_size = os.path.getsize(file_path)
    except OSError as exc:
      return self._build_messaging_error(f"Cannot read file: {exc}")

    if file_size > MAX_VIDEO_SIZE_BYTES:
      raise FileSizeError(
        f"Video size {file_size} bytes exceeds Telegram limit "
        f"of {MAX_VIDEO_SIZE_BYTES} bytes (50 MB)."
      )

    try:
      with open(file_path, "rb") as video_file:
        async with httpx.AsyncClient() as client:
          response = await client.post(
            self._api_url("sendVideo"),
            data={"chat_id": target_chat, "caption": caption},
            files={"video": video_file},
            timeout=120.0,
          )
      if response.status_code == 429:
        raise RateLimitError("Telegram rate limit hit during sendVideo.")
      response.raise_for_status()
    except (FileSizeError, RateLimitError):
      raise
    except Exception as exc:
      logger.error("Telegram send_video failed: {exc}", exc=exc)
      return self._build_messaging_error(str(exc))

    data = response.json()
    message_id = str(data.get("result", {}).get("message_id", ""))
    logger.info("Telegram video sent: message_id={mid}", mid=message_id)
    return self._build_messaging_success(message_id)

  async def send_message(
    self,
    text: str,
    chat_id: Optional[str] = None,
    parse_mode: str = "HTML",
  ) -> Dict[str, Any]:
    """
    Send a text message to a Telegram chat via sendMessage.

    Args:
        text:       Message text. Supports HTML or Markdown formatting.
        chat_id:    Target chat ID; defaults to TELEGRAM_CHAT_ID env var.
        parse_mode: "HTML" or "MarkdownV2".

    Returns:
        {"success": bool, "message_id": str|None, "platform": "telegram", "error": str|None}
    """
    if self.simulation_mode:
      return self._build_messaging_simulate({"text": text[:50]})

    target_chat = chat_id or self._chat_id

    try:
      async with httpx.AsyncClient() as client:
        response = await client.post(
          self._api_url("sendMessage"),
          json={
            "chat_id": target_chat,
            "text": text,
            "parse_mode": parse_mode,
          },
          timeout=30.0,
        )
      if response.status_code == 429:
        raise RateLimitError("Telegram rate limit hit during sendMessage.")
      response.raise_for_status()
    except RateLimitError:
      raise
    except Exception as exc:
      logger.error("Telegram send_message failed: {exc}", exc=exc)
      return self._build_messaging_error(str(exc))

    message_id = str(response.json().get("result", {}).get("message_id", ""))
    return self._build_messaging_success(message_id)

  async def send_photo(
    self,
    file_path: str,
    caption: str = "",
    chat_id: Optional[str] = None,
  ) -> Dict[str, Any]:
    """
    Send a photo to a Telegram chat via sendPhoto.

    Args:
        file_path: Absolute path to the image file.
        caption:   Optional caption shown beneath the photo.
        chat_id:   Target chat ID; defaults to TELEGRAM_CHAT_ID env var.

    Returns:
        {"success": bool, "message_id": str|None, "platform": "telegram", "error": str|None}
    """
    if self.simulation_mode:
      return self._build_messaging_simulate({"file_path": file_path, "caption": caption})

    target_chat = chat_id or self._chat_id

    try:
      with open(file_path, "rb") as photo_file:
        async with httpx.AsyncClient() as client:
          response = await client.post(
            self._api_url("sendPhoto"),
            data={"chat_id": target_chat, "caption": caption},
            files={"photo": photo_file},
            timeout=60.0,
          )
      if response.status_code == 429:
        raise RateLimitError("Telegram rate limit hit during sendPhoto.")
      response.raise_for_status()
    except RateLimitError:
      raise
    except Exception as exc:
      logger.error("Telegram send_photo failed: {exc}", exc=exc)
      return self._build_messaging_error(str(exc))

    message_id = str(response.json().get("result", {}).get("message_id", ""))
    return self._build_messaging_success(message_id)

  async def send_document(
    self,
    file_path: str,
    caption: str = "",
    chat_id: Optional[str] = None,
  ) -> Dict[str, Any]:
    """
    Send a document (any file type) to a Telegram chat via sendDocument.

    Args:
        file_path: Absolute path to the document.
        caption:   Optional caption shown beneath the document.
        chat_id:   Target chat ID; defaults to TELEGRAM_CHAT_ID env var.

    Returns:
        {"success": bool, "message_id": str|None, "platform": "telegram", "error": str|None}
    """
    if self.simulation_mode:
      return self._build_messaging_simulate({"file_path": file_path, "caption": caption})

    target_chat = chat_id or self._chat_id

    try:
      with open(file_path, "rb") as doc_file:
        async with httpx.AsyncClient() as client:
          response = await client.post(
            self._api_url("sendDocument"),
            data={"chat_id": target_chat, "caption": caption},
            files={"document": doc_file},
            timeout=60.0,
          )
      if response.status_code == 429:
        raise RateLimitError("Telegram rate limit hit during sendDocument.")
      response.raise_for_status()
    except RateLimitError:
      raise
    except Exception as exc:
      logger.error("Telegram send_document failed: {exc}", exc=exc)
      return self._build_messaging_error(str(exc))

    message_id = str(response.json().get("result", {}).get("message_id", ""))
    return self._build_messaging_success(message_id)

  async def get_rate_limit_status(self) -> Dict[str, Any]:
    """
    Return conservative Telegram rate limit defaults.

    Telegram does not expose rate limit headers — limits are enforced silently.
    These values reflect the documented per-chat and global bot API limits.
    """
    return {
      "requests_remaining": 30,
      "reset_at": "per-second window, no explicit reset timestamp",
      "window_seconds": 1,
      "messages_per_second_global": 30,
      "messages_per_second_per_chat": 1,
      "messages_per_minute_per_group": 20,
    }

  # ---------------------------------------------------------------------------
  # Private helpers — messaging-specific response builders.
  # BasePoster._build_success_response uses "post_id" and "url", but messaging
  # channels use "message_id" and no public URL, so we define our own builders.
  # ---------------------------------------------------------------------------

  def _build_messaging_success(self, message_id: str) -> Dict[str, Any]:
    """Return a standardized messaging success response."""
    return {
      "success": True,
      "message_id": message_id,
      "platform": self.platform_name,
      "error": None,
    }

  def _build_messaging_error(self, error: str) -> Dict[str, Any]:
    """Return a standardized messaging failure response."""
    return {
      "success": False,
      "message_id": None,
      "platform": self.platform_name,
      "error": error,
    }

  def _build_messaging_simulate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a simulated success response without making any API call."""
    logger.info(
      "[SIMULATION] {platform} would send: {payload}",
      platform=self.platform_name,
      payload=payload,
    )
    return {
      "success": True,
      "message_id": f"sim_{self.platform_name}_12345",
      "platform": self.platform_name,
      "error": None,
      "simulated": True,
    }


def _register() -> None:
  """Register TelegramPoster with the global PosterRegistry."""
  PosterRegistry.register("telegram", TelegramPoster)
  logger.debug("TelegramPoster registered.")


_register()
