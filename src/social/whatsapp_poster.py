"""
WhatsAppPoster — sends content via the Meta WhatsApp Cloud API.

API flow — preferred workflow (Decision Log #4: notification_with_link):
  1. send_notification_with_link() sends a template message containing
     an external URL to the video. The user clicks the link in WhatsApp.
     This is the PREFERRED workflow because:
       - WhatsApp limits direct video uploads to 16 MB.
       - Template messages reliably reach users even on slow connections.
       - External links work for videos of any size hosted on CDN/YouTube.
  2. send_video_direct() performs a direct media upload to the Media API,
     then sends the returned media_id. Raises FileSizeError if > 16 MB.
     Use ONLY when you specifically need the video inline in the chat.

API flow — direct video (send_video_direct):
  1. POST /v19.0/{phone_number_id}/media with multipart/form-data.
     Returns a media_id.
  2. POST /v19.0/{phone_number_id}/messages with type=video and media_id.

Rate limits (as of 2026):
  - 250 conversation-initiated messages per 24 h per phone number (free tier).
  - 1,000+ with Meta Business verification.
  - Template messages: additional per-template rate limits apply.

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

# Meta WhatsApp Cloud API base URL.
WHATSAPP_API_BASE = "https://graph.facebook.com/v19.0"

# Maximum file size for direct video upload to WhatsApp Media API.
MAX_DIRECT_VIDEO_SIZE_BYTES = 16 * 1024 * 1024  # 16 MB

# Preferred workflow identifier — logged and included in responses for auditability.
PREFERRED_WORKFLOW = "notification_with_link"


class WhatsAppPoster(MessagingPlatform, BasePoster):
  """
  Sends messages and media to WhatsApp via the Meta Cloud API.

  Preferred method: send_notification_with_link() — template message with URL.
  Direct video:     send_video_direct()            — inline video, 16 MB max.

  Requires env vars:
      WHATSAPP_PHONE_NUMBER_ID — Sender's WhatsApp phone number ID.
      WHATSAPP_ACCESS_TOKEN    — Meta Cloud API access token.
      WHATSAPP_RECIPIENT_PHONE — Default recipient phone (international format, e.g. +15551234567).

  All credentials must come from environment variables only.
  """

  platform_name = "whatsapp"

  def __init__(self) -> None:
    self._phone_number_id: Optional[str] = None
    self._access_token: Optional[str] = None
    self._recipient_phone: Optional[str] = None
    self._authenticated: bool = False

  def _messages_url(self) -> str:
    """Return the messages endpoint URL for this phone number ID."""
    return f"{WHATSAPP_API_BASE}/{self._phone_number_id}/messages"

  def _media_url(self) -> str:
    """Return the media upload endpoint URL for this phone number ID."""
    return f"{WHATSAPP_API_BASE}/{self._phone_number_id}/media"

  def _auth_headers(self) -> Dict[str, str]:
    """Return authorization headers for Meta Cloud API requests."""
    return {"Authorization": f"Bearer {self._access_token}"}

  async def authenticate(self) -> None:
    """
    Load credentials from env vars and validate via the phone number endpoint.

    Calls GET /{phone_number_id} to confirm the token has access to the
    registered sender number before any messages are attempted.

    Raises:
        AuthError: if env vars are missing or the token is rejected.
        RateLimitError: if Meta returns 429 during the auth check.
    """
    if self.guard_unconfigured(
      "WHATSAPP_PHONE_NUMBER_ID",
      "WHATSAPP_ACCESS_TOKEN",
      "WHATSAPP_RECIPIENT_PHONE",
    ):
      raise AuthError(
        "WhatsApp credentials not configured. "
        "Set WHATSAPP_PHONE_NUMBER_ID, WHATSAPP_ACCESS_TOKEN, "
        "and WHATSAPP_RECIPIENT_PHONE."
      )

    self._phone_number_id = os.environ["WHATSAPP_PHONE_NUMBER_ID"]
    self._access_token = os.environ["WHATSAPP_ACCESS_TOKEN"]
    self._recipient_phone = os.environ["WHATSAPP_RECIPIENT_PHONE"]

    try:
      async with httpx.AsyncClient() as client:
        response = await client.get(
          f"{WHATSAPP_API_BASE}/{self._phone_number_id}",
          headers=self._auth_headers(),
          params={"fields": "id,display_phone_number,verified_name"},
          timeout=10.0,
        )
      if response.status_code == 401:
        raise AuthError(
          "WhatsApp access token rejected (401). "
          "Verify WHATSAPP_ACCESS_TOKEN in Meta Business Manager."
        )
      if response.status_code == 429:
        raise RateLimitError("WhatsApp rate limit hit during auth check.")
      response.raise_for_status()
    except (AuthError, RateLimitError):
      raise
    except Exception as exc:
      raise AuthError(f"WhatsApp auth validation failed: {exc}") from exc

    self._authenticated = True
    logger.info("WhatsApp authenticated successfully.")

  async def send_notification_with_link(
    self,
    video_url: str,
    caption: str,
    template_name: str = "video_notification",
    recipient_phone: Optional[str] = None,
  ) -> Dict[str, Any]:
    """
    Send a template message with an external video URL (preferred workflow).

    This is the PREFERRED method per Decision Log #4.
    The recipient receives a text message with a clickable link to the video.
    Works for videos of any size — no upload size restriction applies.

    Template message format:
        caption + "\\n" + video_url

    Args:
        video_url:       Publicly accessible URL to the video (CDN, YouTube, etc.).
        caption:         Description or call-to-action text.
        template_name:   WhatsApp approved template name (must be pre-approved in Meta).
        recipient_phone: Override for WHATSAPP_RECIPIENT_PHONE env var.

    Returns:
        {"success": bool, "message_id": str|None, "platform": "whatsapp",
         "workflow": "notification_with_link", "error": str|None}
    """
    if self.simulation_mode:
      result = self._build_messaging_simulate(
        {"video_url": video_url, "caption": caption}
      )
      result["workflow"] = PREFERRED_WORKFLOW
      return result

    target_phone = recipient_phone or self._recipient_phone

    payload = {
      "messaging_product": "whatsapp",
      "to": target_phone,
      "type": "template",
      "template": {
        "name": template_name,
        "language": {"code": "en_US"},
        "components": [
          {
            "type": "body",
            "parameters": [
              {"type": "text", "text": caption},
              {"type": "text", "text": video_url},
            ],
          }
        ],
      },
    }

    try:
      async with httpx.AsyncClient() as client:
        response = await client.post(
          self._messages_url(),
          headers={**self._auth_headers(), "Content-Type": "application/json"},
          json=payload,
          timeout=30.0,
        )
      if response.status_code == 429:
        raise RateLimitError(
          "WhatsApp rate limit hit during send_notification_with_link."
        )
      response.raise_for_status()
    except RateLimitError:
      raise
    except Exception as exc:
      logger.error("WhatsApp send_notification_with_link failed: {exc}", exc=exc)
      result = self._build_messaging_error(str(exc))
      result["workflow"] = PREFERRED_WORKFLOW
      return result

    data = response.json()
    message_id = str(
      data.get("messages", [{}])[0].get("id", "")
    )
    logger.info(
      "WhatsApp notification_with_link sent: message_id={mid}",
      mid=message_id,
    )
    result = self._build_messaging_success(message_id)
    result["workflow"] = PREFERRED_WORKFLOW
    return result

  async def send_video_direct(
    self,
    file_path: str,
    caption: str = "",
    recipient_phone: Optional[str] = None,
  ) -> Dict[str, Any]:
    """
    Upload a video directly and send it as an inline media message.

    SECONDARY method. Enforces the 16 MB platform limit via FileSizeError.
    Prefer send_notification_with_link() for videos larger than 16 MB
    or when CDN hosting is available.

    API flow:
      1. POST /{phone_number_id}/media — upload the video, get media_id.
      2. POST /{phone_number_id}/messages — send message with media_id.

    Args:
        file_path:       Absolute path to the video file (must be <= 16 MB).
        caption:         Optional caption shown in the WhatsApp chat.
        recipient_phone: Override for WHATSAPP_RECIPIENT_PHONE env var.

    Raises:
        FileSizeError: if file exceeds 16 MB.

    Returns:
        {"success": bool, "message_id": str|None, "platform": "whatsapp", "error": str|None}
    """
    if self.simulation_mode:
      return self._build_messaging_simulate(
        {"file_path": file_path, "caption": caption}
      )

    # Enforce 16 MB limit before touching the API.
    try:
      file_size = os.path.getsize(file_path)
    except OSError as exc:
      return self._build_messaging_error(f"Cannot read file: {exc}")

    if file_size > MAX_DIRECT_VIDEO_SIZE_BYTES:
      raise FileSizeError(
        f"Video size {file_size} bytes exceeds WhatsApp direct upload limit "
        f"of {MAX_DIRECT_VIDEO_SIZE_BYTES} bytes (16 MB). "
        f"Use send_notification_with_link() instead."
      )

    target_phone = recipient_phone or self._recipient_phone

    # Step 1: Upload media and get media_id.
    try:
      with open(file_path, "rb") as video_file:
        async with httpx.AsyncClient() as client:
          upload_response = await client.post(
            self._media_url(),
            headers=self._auth_headers(),
            files={"file": ("video.mp4", video_file, "video/mp4")},
            data={"messaging_product": "whatsapp"},
            timeout=120.0,
          )
      if upload_response.status_code == 429:
        raise RateLimitError("WhatsApp rate limit hit during media upload.")
      upload_response.raise_for_status()
    except (FileSizeError, RateLimitError):
      raise
    except Exception as exc:
      logger.error("WhatsApp media upload failed: {exc}", exc=exc)
      return self._build_messaging_error(f"Media upload failed: {exc}")

    media_id = upload_response.json().get("id", "")
    logger.debug("WhatsApp media uploaded: media_id={mid}", mid=media_id)

    # Step 2: Send message referencing the uploaded media_id.
    try:
      async with httpx.AsyncClient() as client:
        send_response = await client.post(
          self._messages_url(),
          headers={**self._auth_headers(), "Content-Type": "application/json"},
          json={
            "messaging_product": "whatsapp",
            "to": target_phone,
            "type": "video",
            "video": {"id": media_id, "caption": caption},
          },
          timeout=30.0,
        )
      if send_response.status_code == 429:
        raise RateLimitError("WhatsApp rate limit hit during message send.")
      send_response.raise_for_status()
    except RateLimitError:
      raise
    except Exception as exc:
      logger.error("WhatsApp message send failed after upload: {exc}", exc=exc)
      return self._build_messaging_error(f"Message send failed: {exc}")

    message_id = str(
      send_response.json().get("messages", [{}])[0].get("id", "")
    )
    logger.info(
      "WhatsApp direct video sent: message_id={mid}",
      mid=message_id,
    )
    return self._build_messaging_success(message_id)

  async def send_text(
    self,
    text: str,
    recipient_phone: Optional[str] = None,
  ) -> Dict[str, Any]:
    """
    Send a plain text message to a WhatsApp number.

    Args:
        text:            Message body text.
        recipient_phone: Override for WHATSAPP_RECIPIENT_PHONE env var.

    Returns:
        {"success": bool, "message_id": str|None, "platform": "whatsapp", "error": str|None}
    """
    if self.simulation_mode:
      return self._build_messaging_simulate({"text": text[:50]})

    target_phone = recipient_phone or self._recipient_phone

    try:
      async with httpx.AsyncClient() as client:
        response = await client.post(
          self._messages_url(),
          headers={**self._auth_headers(), "Content-Type": "application/json"},
          json={
            "messaging_product": "whatsapp",
            "to": target_phone,
            "type": "text",
            "text": {"body": text},
          },
          timeout=30.0,
        )
      if response.status_code == 429:
        raise RateLimitError("WhatsApp rate limit hit during send_text.")
      response.raise_for_status()
    except RateLimitError:
      raise
    except Exception as exc:
      logger.error("WhatsApp send_text failed: {exc}", exc=exc)
      return self._build_messaging_error(str(exc))

    message_id = str(response.json().get("messages", [{}])[0].get("id", ""))
    return self._build_messaging_success(message_id)

  async def send_image(
    self,
    file_path: str,
    caption: str = "",
    recipient_phone: Optional[str] = None,
  ) -> Dict[str, Any]:
    """
    Upload and send an image to a WhatsApp number.

    Args:
        file_path:       Absolute path to the image file.
        caption:         Optional caption text.
        recipient_phone: Override for WHATSAPP_RECIPIENT_PHONE env var.

    Returns:
        {"success": bool, "message_id": str|None, "platform": "whatsapp", "error": str|None}
    """
    if self.simulation_mode:
      return self._build_messaging_simulate(
        {"file_path": file_path, "caption": caption}
      )

    target_phone = recipient_phone or self._recipient_phone

    # Step 1: Upload image.
    try:
      with open(file_path, "rb") as image_file:
        async with httpx.AsyncClient() as client:
          upload_response = await client.post(
            self._media_url(),
            headers=self._auth_headers(),
            files={"file": ("image.jpg", image_file, "image/jpeg")},
            data={"messaging_product": "whatsapp"},
            timeout=60.0,
          )
      if upload_response.status_code == 429:
        raise RateLimitError("WhatsApp rate limit hit during image upload.")
      upload_response.raise_for_status()
    except RateLimitError:
      raise
    except Exception as exc:
      logger.error("WhatsApp image upload failed: {exc}", exc=exc)
      return self._build_messaging_error(f"Image upload failed: {exc}")

    media_id = upload_response.json().get("id", "")

    # Step 2: Send image message.
    try:
      async with httpx.AsyncClient() as client:
        send_response = await client.post(
          self._messages_url(),
          headers={**self._auth_headers(), "Content-Type": "application/json"},
          json={
            "messaging_product": "whatsapp",
            "to": target_phone,
            "type": "image",
            "image": {"id": media_id, "caption": caption},
          },
          timeout=30.0,
        )
      if send_response.status_code == 429:
        raise RateLimitError("WhatsApp rate limit hit during image send.")
      send_response.raise_for_status()
    except RateLimitError:
      raise
    except Exception as exc:
      logger.error("WhatsApp send_image message failed: {exc}", exc=exc)
      return self._build_messaging_error(f"Image send failed: {exc}")

    message_id = str(
      send_response.json().get("messages", [{}])[0].get("id", "")
    )
    return self._build_messaging_success(message_id)

  async def get_rate_limit_status(self) -> Dict[str, Any]:
    """
    Return current WhatsApp Cloud API rate limit defaults.

    Meta does not expose per-request rate limit headers for WhatsApp.
    These reflect the documented free-tier conversation limits.
    """
    return {
      "requests_remaining": 250,
      "reset_at": "24-hour rolling window, resets at midnight UTC",
      "window_seconds": 86400,
      "conversations_per_day": 250,
      "note": "Increase limit by completing Meta Business verification.",
    }

  # ---------------------------------------------------------------------------
  # Private helpers — messaging-specific response builders.
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
  """Register WhatsAppPoster with the global PosterRegistry."""
  PosterRegistry.register("whatsapp", WhatsAppPoster)
  logger.debug("WhatsAppPoster registered.")


_register()
