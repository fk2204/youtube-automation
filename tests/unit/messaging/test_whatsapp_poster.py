"""
Unit tests for WhatsAppPoster.

All external API calls and file system operations are mocked.
No real credentials required — tests are fully isolated.

Decision Log #4: send_notification_with_link() is the preferred workflow.
Tests verify this workflow is functional and that send_video_direct()
enforces the 16 MB limit.
"""

import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

# Ensure src/ is on path for direct pytest invocation from project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from exceptions import AuthError, FileSizeError, RateLimitError
from registry.poster_registry import PosterRegistry

# Import triggers _register() at module level.
import social.whatsapp_poster as whatsapp_module
from social.whatsapp_poster import (
  WhatsAppPoster,
  MAX_DIRECT_VIDEO_SIZE_BYTES,
  PREFERRED_WORKFLOW,
)

SAMPLE_VIDEO_URL = "https://cdn.example.com/videos/campaign.mp4"
SAMPLE_FILE_PATH = "/tmp/test_video.mp4"
SAMPLE_SMALL_SIZE = 1 * 1024 * 1024   # 1 MB — within limit
SAMPLE_RECIPIENT = "+15551234567"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def poster() -> WhatsAppPoster:
  return WhatsAppPoster()


@pytest.fixture
def authenticated_poster(poster: WhatsAppPoster) -> WhatsAppPoster:
  """Return a poster with auth state already set (bypasses authenticate())."""
  poster._phone_number_id = "15550001111"
  poster._access_token = "test_access_token"
  poster._recipient_phone = SAMPLE_RECIPIENT
  poster._authenticated = True
  return poster


@pytest.fixture(autouse=True)
def clear_simulation_mode(monkeypatch: pytest.MonkeyPatch) -> None:
  """Ensure simulation mode is off for all tests unless explicitly set."""
  monkeypatch.delenv("SOCIAL_SIMULATION_MODE", raising=False)


# ---------------------------------------------------------------------------
# Registration test
# ---------------------------------------------------------------------------


class TestAuthenticate:
  @pytest.mark.asyncio
  async def test_authenticate_success(
    self,
    poster: WhatsAppPoster,
    monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """authenticate() must set _authenticated=True on valid credentials."""
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "15550001111")
    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "EAA_test_token")
    monkeypatch.setenv("WHATSAPP_RECIPIENT_PHONE", SAMPLE_RECIPIENT)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
      "id": "15550001111",
      "display_phone_number": "+1 555 000 1111",
      "verified_name": "Test Business",
    }

    with patch("social.whatsapp_poster.httpx.AsyncClient") as mock_client_class:
      mock_client = AsyncMock()
      mock_client.__aenter__ = AsyncMock(return_value=mock_client)
      mock_client.__aexit__ = AsyncMock(return_value=False)
      mock_client.get = AsyncMock(return_value=mock_response)
      mock_client_class.return_value = mock_client

      await poster.authenticate()

    assert poster._authenticated is True
    assert poster._access_token == "EAA_test_token"
    assert poster._phone_number_id == "15550001111"

  @pytest.mark.asyncio
  async def test_authenticate_missing_env_raises_auth_error(
    self,
    poster: WhatsAppPoster,
    monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """authenticate() must raise AuthError when env vars are missing."""
    monkeypatch.delenv("WHATSAPP_PHONE_NUMBER_ID", raising=False)
    monkeypatch.delenv("WHATSAPP_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("WHATSAPP_RECIPIENT_PHONE", raising=False)

    with pytest.raises(AuthError):
      await poster.authenticate()

  @pytest.mark.asyncio
  async def test_authenticate_401_raises_auth_error(
    self,
    poster: WhatsAppPoster,
    monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """authenticate() must raise AuthError when Meta API returns 401."""
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "111")
    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "bad_token")
    monkeypatch.setenv("WHATSAPP_RECIPIENT_PHONE", SAMPLE_RECIPIENT)

    mock_response = MagicMock()
    mock_response.status_code = 401

    with patch("social.whatsapp_poster.httpx.AsyncClient") as mock_client_class:
      mock_client = AsyncMock()
      mock_client.__aenter__ = AsyncMock(return_value=mock_client)
      mock_client.__aexit__ = AsyncMock(return_value=False)
      mock_client.get = AsyncMock(return_value=mock_response)
      mock_client_class.return_value = mock_client

      with pytest.raises(AuthError, match="401"):
        await poster.authenticate()


# ---------------------------------------------------------------------------
# send_notification_with_link() tests — preferred workflow (Decision Log #4)
# ---------------------------------------------------------------------------

class TestNotificationWithLink:
  @pytest.mark.asyncio
  async def test_notification_with_link_success_returns_message_id_and_workflow(
    self,
    authenticated_poster: WhatsAppPoster,
  ) -> None:
    """send_notification_with_link() must return message_id and workflow key."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
      "messaging_product": "whatsapp",
      "messages": [{"id": "wamid.abc123"}],
    }

    with patch("social.whatsapp_poster.httpx.AsyncClient") as mock_client_class:
      mock_client = AsyncMock()
      mock_client.__aenter__ = AsyncMock(return_value=mock_client)
      mock_client.__aexit__ = AsyncMock(return_value=False)
      mock_client.post = AsyncMock(return_value=mock_response)
      mock_client_class.return_value = mock_client

      result = await authenticated_poster.send_notification_with_link(
        SAMPLE_VIDEO_URL, "Check out our new video!"
      )

    assert result["success"] is True
    assert result["message_id"] == "wamid.abc123"
    assert result["workflow"] == PREFERRED_WORKFLOW
    assert result["platform"] == "whatsapp"
    assert result["error"] is None

  @pytest.mark.asyncio
  async def test_notification_with_link_simulation_returns_simulated(
    self,
    authenticated_poster: WhatsAppPoster,
    monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """send_notification_with_link() in simulation mode skips API call."""
    monkeypatch.setenv("SOCIAL_SIMULATION_MODE", "true")

    result = await authenticated_poster.send_notification_with_link(
      SAMPLE_VIDEO_URL, "Caption"
    )

    assert result["success"] is True
    assert result.get("simulated") is True
    assert result["workflow"] == PREFERRED_WORKFLOW

  @pytest.mark.asyncio
  async def test_notification_with_link_429_raises_rate_limit_error(
    self,
    authenticated_poster: WhatsAppPoster,
  ) -> None:
    """send_notification_with_link() must raise RateLimitError on 429."""
    mock_response = MagicMock()
    mock_response.status_code = 429

    with patch("social.whatsapp_poster.httpx.AsyncClient") as mock_client_class:
      mock_client = AsyncMock()
      mock_client.__aenter__ = AsyncMock(return_value=mock_client)
      mock_client.__aexit__ = AsyncMock(return_value=False)
      mock_client.post = AsyncMock(return_value=mock_response)
      mock_client_class.return_value = mock_client

      with pytest.raises(RateLimitError):
        await authenticated_poster.send_notification_with_link(
          SAMPLE_VIDEO_URL, "Caption"
        )


# ---------------------------------------------------------------------------
# send_video_direct() tests — 16 MB limit enforced
# ---------------------------------------------------------------------------

class TestSendVideoDirect:
  @pytest.mark.asyncio
  async def test_send_video_direct_exceeds_16mb_raises_file_size_error(
    self,
    authenticated_poster: WhatsAppPoster,
  ) -> None:
    """send_video_direct() must raise FileSizeError when file exceeds 16 MB."""
    oversized = MAX_DIRECT_VIDEO_SIZE_BYTES + 1

    with patch("os.path.getsize", return_value=oversized):
      with pytest.raises(FileSizeError, match="16 MB"):
        await authenticated_poster.send_video_direct(SAMPLE_FILE_PATH)

  @pytest.mark.asyncio
  async def test_send_video_direct_success_within_limit(
    self,
    authenticated_poster: WhatsAppPoster,
  ) -> None:
    """send_video_direct() must return success dict for files within 16 MB."""
    upload_response = MagicMock()
    upload_response.status_code = 200
    upload_response.raise_for_status = MagicMock()
    upload_response.json.return_value = {"id": "media_id_xyz"}

    send_response = MagicMock()
    send_response.status_code = 200
    send_response.raise_for_status = MagicMock()
    send_response.json.return_value = {
      "messaging_product": "whatsapp",
      "messages": [{"id": "wamid.video999"}],
    }

    with (
      patch("os.path.getsize", return_value=SAMPLE_SMALL_SIZE),
      patch("builtins.open", mock_open(read_data=b"v" * SAMPLE_SMALL_SIZE)),
      patch("social.whatsapp_poster.httpx.AsyncClient") as mock_client_class,
    ):
      mock_client = AsyncMock()
      mock_client.__aenter__ = AsyncMock(return_value=mock_client)
      mock_client.__aexit__ = AsyncMock(return_value=False)
      # First call: upload. Second call: send message.
      mock_client.post = AsyncMock(side_effect=[upload_response, send_response])
      mock_client_class.return_value = mock_client

      result = await authenticated_poster.send_video_direct(
        SAMPLE_FILE_PATH, "Direct video caption"
      )

    assert result["success"] is True
    assert result["message_id"] == "wamid.video999"
    assert result["platform"] == "whatsapp"

  @pytest.mark.asyncio
  async def test_send_video_direct_simulation_mode_returns_simulated(
    self,
    authenticated_poster: WhatsAppPoster,
    monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """send_video_direct() in simulation mode skips API and file checks."""
    monkeypatch.setenv("SOCIAL_SIMULATION_MODE", "true")

    result = await authenticated_poster.send_video_direct(SAMPLE_FILE_PATH)

    assert result["success"] is True
    assert result.get("simulated") is True


# ---------------------------------------------------------------------------
# get_rate_limit_status() tests
# ---------------------------------------------------------------------------

class TestRateLimitStatus:
  @pytest.mark.asyncio
  async def test_rate_limit_status_returns_expected_shape(
    self,
    authenticated_poster: WhatsAppPoster,
  ) -> None:
    """get_rate_limit_status() must return dict with required keys."""
    status = await authenticated_poster.get_rate_limit_status()

    assert "requests_remaining" in status
    assert "reset_at" in status
    assert "window_seconds" in status
    assert isinstance(status["requests_remaining"], int)
