"""
Unit tests for TelegramPoster.

All external API calls and file system operations are mocked.
No real credentials required — tests are fully isolated.
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
import social.telegram_poster as telegram_module
from social.telegram_poster import TelegramPoster, MAX_VIDEO_SIZE_BYTES

SAMPLE_FILE_PATH = "/tmp/test_video.mp4"
SAMPLE_SMALL_SIZE = 1 * 1024 * 1024   # 1 MB — within limit
SAMPLE_CHAT_ID = "-100123456789"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def poster() -> TelegramPoster:
  return TelegramPoster()


@pytest.fixture
def authenticated_poster(poster: TelegramPoster) -> TelegramPoster:
  """Return a poster with auth state already set (bypasses authenticate())."""
  poster._bot_token = "test_bot_token"
  poster._chat_id = SAMPLE_CHAT_ID
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
    poster: TelegramPoster,
    monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """authenticate() must set _authenticated=True on valid credentials."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:ABC-test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", SAMPLE_CHAT_ID)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
      "ok": True,
      "result": {"id": 123, "is_bot": True, "username": "test_bot"},
    }

    with patch("social.telegram_poster.httpx.AsyncClient") as mock_client_class:
      mock_client = AsyncMock()
      mock_client.__aenter__ = AsyncMock(return_value=mock_client)
      mock_client.__aexit__ = AsyncMock(return_value=False)
      mock_client.get = AsyncMock(return_value=mock_response)
      mock_client_class.return_value = mock_client

      await poster.authenticate()

    assert poster._authenticated is True
    assert poster._bot_token == "123456:ABC-test-token"
    assert poster._chat_id == SAMPLE_CHAT_ID

  @pytest.mark.asyncio
  async def test_authenticate_failure_missing_env_raises_auth_error(
    self,
    poster: TelegramPoster,
    monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """authenticate() must raise AuthError when env vars are missing."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    with pytest.raises(AuthError):
      await poster.authenticate()

  @pytest.mark.asyncio
  async def test_authenticate_failure_401_raises_auth_error(
    self,
    poster: TelegramPoster,
    monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """authenticate() must raise AuthError when Telegram returns 401."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bad_token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", SAMPLE_CHAT_ID)

    mock_response = MagicMock()
    mock_response.status_code = 401

    with patch("social.telegram_poster.httpx.AsyncClient") as mock_client_class:
      mock_client = AsyncMock()
      mock_client.__aenter__ = AsyncMock(return_value=mock_client)
      mock_client.__aexit__ = AsyncMock(return_value=False)
      mock_client.get = AsyncMock(return_value=mock_response)
      mock_client_class.return_value = mock_client

      with pytest.raises(AuthError, match="401"):
        await poster.authenticate()

  @pytest.mark.asyncio
  async def test_authenticate_429_raises_rate_limit_error(
    self,
    poster: TelegramPoster,
    monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """authenticate() must raise RateLimitError when Telegram returns 429."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "valid_token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", SAMPLE_CHAT_ID)

    mock_response = MagicMock()
    mock_response.status_code = 429

    with patch("social.telegram_poster.httpx.AsyncClient") as mock_client_class:
      mock_client = AsyncMock()
      mock_client.__aenter__ = AsyncMock(return_value=mock_client)
      mock_client.__aexit__ = AsyncMock(return_value=False)
      mock_client.get = AsyncMock(return_value=mock_response)
      mock_client_class.return_value = mock_client

      with pytest.raises(RateLimitError):
        await poster.authenticate()


# ---------------------------------------------------------------------------
# send_video() tests
# ---------------------------------------------------------------------------

class TestSendVideo:
  @pytest.mark.asyncio
  async def test_send_video_success_returns_message_id(
    self,
    authenticated_poster: TelegramPoster,
  ) -> None:
    """send_video() must return success dict with message_id on 200 response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
      "ok": True,
      "result": {"message_id": 42, "chat": {"id": -100123456789}},
    }

    with (
      patch("os.path.getsize", return_value=SAMPLE_SMALL_SIZE),
      patch("builtins.open", mock_open(read_data=b"x" * SAMPLE_SMALL_SIZE)),
      patch("social.telegram_poster.httpx.AsyncClient") as mock_client_class,
    ):
      mock_client = AsyncMock()
      mock_client.__aenter__ = AsyncMock(return_value=mock_client)
      mock_client.__aexit__ = AsyncMock(return_value=False)
      mock_client.post = AsyncMock(return_value=mock_response)
      mock_client_class.return_value = mock_client

      result = await authenticated_poster.send_video(SAMPLE_FILE_PATH, "Test caption")

    assert result["success"] is True
    assert result["message_id"] == "42"
    assert result["platform"] == "telegram"
    assert result["error"] is None

  @pytest.mark.asyncio
  async def test_send_video_exceeds_50mb_raises_file_size_error(
    self,
    authenticated_poster: TelegramPoster,
  ) -> None:
    """send_video() must raise FileSizeError when file exceeds 50 MB."""
    oversized = MAX_VIDEO_SIZE_BYTES + 1

    with patch("os.path.getsize", return_value=oversized):
      with pytest.raises(FileSizeError):
        await authenticated_poster.send_video(SAMPLE_FILE_PATH)

  @pytest.mark.asyncio
  async def test_send_video_simulation_mode_returns_simulated(
    self,
    authenticated_poster: TelegramPoster,
    monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """send_video() in simulation mode must return simulated=True without API calls."""
    monkeypatch.setenv("SOCIAL_SIMULATION_MODE", "true")

    result = await authenticated_poster.send_video(SAMPLE_FILE_PATH, "caption")

    assert result["success"] is True
    assert result.get("simulated") is True
    assert result["platform"] == "telegram"


# ---------------------------------------------------------------------------
# send_message() tests
# ---------------------------------------------------------------------------

class TestSendMessage:
  @pytest.mark.asyncio
  async def test_send_message_success_returns_message_id(
    self,
    authenticated_poster: TelegramPoster,
  ) -> None:
    """send_message() must return success dict with message_id."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
      "ok": True,
      "result": {"message_id": 77},
    }

    with patch("social.telegram_poster.httpx.AsyncClient") as mock_client_class:
      mock_client = AsyncMock()
      mock_client.__aenter__ = AsyncMock(return_value=mock_client)
      mock_client.__aexit__ = AsyncMock(return_value=False)
      mock_client.post = AsyncMock(return_value=mock_response)
      mock_client_class.return_value = mock_client

      result = await authenticated_poster.send_message("Hello, channel!")

    assert result["success"] is True
    assert result["message_id"] == "77"
    assert result["error"] is None

  @pytest.mark.asyncio
  async def test_send_message_429_raises_rate_limit_error(
    self,
    authenticated_poster: TelegramPoster,
  ) -> None:
    """send_message() must raise RateLimitError on 429."""
    mock_response = MagicMock()
    mock_response.status_code = 429

    with patch("social.telegram_poster.httpx.AsyncClient") as mock_client_class:
      mock_client = AsyncMock()
      mock_client.__aenter__ = AsyncMock(return_value=mock_client)
      mock_client.__aexit__ = AsyncMock(return_value=False)
      mock_client.post = AsyncMock(return_value=mock_response)
      mock_client_class.return_value = mock_client

      with pytest.raises(RateLimitError):
        await authenticated_poster.send_message("Rate limited message")


# ---------------------------------------------------------------------------
# send_photo() and send_document() tests
# ---------------------------------------------------------------------------

class TestSendPhotoAndDocument:
  @pytest.mark.asyncio
  async def test_send_photo_success_returns_message_id(
    self,
    authenticated_poster: TelegramPoster,
  ) -> None:
    """send_photo() must return success dict with message_id."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"ok": True, "result": {"message_id": 55}}

    with (
      patch("builtins.open", mock_open(read_data=b"img_data")),
      patch("social.telegram_poster.httpx.AsyncClient") as mock_client_class,
    ):
      mock_client = AsyncMock()
      mock_client.__aenter__ = AsyncMock(return_value=mock_client)
      mock_client.__aexit__ = AsyncMock(return_value=False)
      mock_client.post = AsyncMock(return_value=mock_response)
      mock_client_class.return_value = mock_client

      result = await authenticated_poster.send_photo("/tmp/image.jpg", "Photo caption")

    assert result["success"] is True
    assert result["message_id"] == "55"

  @pytest.mark.asyncio
  async def test_send_document_success_returns_message_id(
    self,
    authenticated_poster: TelegramPoster,
  ) -> None:
    """send_document() must return success dict with message_id."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"ok": True, "result": {"message_id": 88}}

    with (
      patch("builtins.open", mock_open(read_data=b"doc_data")),
      patch("social.telegram_poster.httpx.AsyncClient") as mock_client_class,
    ):
      mock_client = AsyncMock()
      mock_client.__aenter__ = AsyncMock(return_value=mock_client)
      mock_client.__aexit__ = AsyncMock(return_value=False)
      mock_client.post = AsyncMock(return_value=mock_response)
      mock_client_class.return_value = mock_client

      result = await authenticated_poster.send_document("/tmp/file.pdf", "Doc caption")

    assert result["success"] is True
    assert result["message_id"] == "88"


# ---------------------------------------------------------------------------
# get_rate_limit_status() tests
# ---------------------------------------------------------------------------

class TestRateLimitStatus:
  @pytest.mark.asyncio
  async def test_rate_limit_status_returns_expected_shape(
    self,
    authenticated_poster: TelegramPoster,
  ) -> None:
    """get_rate_limit_status() must return dict with required keys."""
    status = await authenticated_poster.get_rate_limit_status()

    assert "requests_remaining" in status
    assert "reset_at" in status
    assert "window_seconds" in status
    assert isinstance(status["requests_remaining"], int)
