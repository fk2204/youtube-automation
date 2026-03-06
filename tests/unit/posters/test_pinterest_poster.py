"""
Unit tests for PinterestPoster.

All API calls and file system operations are mocked.
No real credentials required. Tests verify pin creation,
video pin polling, board listing, cover image requirement, and registration.
"""

import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from exceptions import AuthError, PollingTimeoutError, RateLimitError, ValidationError
from registry.poster_registry import PosterRegistry

import social.pinterest_poster as pinterest_module
from social.pinterest_poster import PinterestPoster, MAX_POLL_ATTEMPTS

SAMPLE_VIDEO_PATH = "/tmp/test_video.mp4"
SAMPLE_IMAGE_URL = "https://cdn.example.com/cover.jpg"
SAMPLE_BOARD_ID = "board_12345"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def poster() -> PinterestPoster:
    return PinterestPoster()


@pytest.fixture
def authenticated_poster(poster: PinterestPoster) -> PinterestPoster:
    poster._access_token = "test_pinterest_token"
    poster._authenticated = True
    return poster


@pytest.fixture(autouse=True)
def clear_simulation_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SOCIAL_SIMULATION_MODE", raising=False)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestAuthenticate:
    @pytest.mark.asyncio
    async def test_authenticate_success(
        self,
        poster: PinterestPoster,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """authenticate() must set _authenticated=True on valid credentials."""
        monkeypatch.setenv("PINTEREST_ACCESS_TOKEN", "valid_token")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"username": "testuser"}

        with patch("social.pinterest_poster.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            await poster.authenticate()

        assert poster._authenticated is True

    @pytest.mark.asyncio
    async def test_authenticate_missing_env_raises_auth_error(
        self,
        poster: PinterestPoster,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """authenticate() must raise AuthError when env var is absent."""
        monkeypatch.delenv("PINTEREST_ACCESS_TOKEN", raising=False)

        with pytest.raises(AuthError):
            await poster.authenticate()


# ---------------------------------------------------------------------------
# create_pin() — success path
# ---------------------------------------------------------------------------

class TestCreatePin:
    @pytest.mark.asyncio
    async def test_create_pin_success(
        self,
        authenticated_poster: PinterestPoster,
    ) -> None:
        """create_pin() must return success dict with pin_id and url."""
        pin_id = "pin_abc_789"

        pin_response = MagicMock()
        pin_response.status_code = 201
        pin_response.raise_for_status = MagicMock()
        pin_response.json.return_value = {"id": pin_id}

        with patch("social.pinterest_poster.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=pin_response)
            mock_client_class.return_value = mock_client

            result = await authenticated_poster.create_pin(
                board_id=SAMPLE_BOARD_ID,
                image_url=SAMPLE_IMAGE_URL,
                title="Test Pin",
                description="A test pin",
            )

        assert result["success"] is True
        assert result["post_id"] == pin_id
        assert result["url"] == f"https://www.pinterest.com/pin/{pin_id}/"
        assert result["error"] is None
        assert result["platform"] == "pinterest"

    @pytest.mark.asyncio
    async def test_create_pin_rate_limit_raises_error(
        self,
        authenticated_poster: PinterestPoster,
    ) -> None:
        """create_pin() must raise RateLimitError on 429."""
        rate_limit_response = MagicMock()
        rate_limit_response.status_code = 429

        with patch("social.pinterest_poster.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=rate_limit_response)
            mock_client_class.return_value = mock_client

            with pytest.raises(RateLimitError):
                await authenticated_poster.create_pin(
                    SAMPLE_BOARD_ID, SAMPLE_IMAGE_URL, "Title"
                )


# ---------------------------------------------------------------------------
# create_video_pin() — cover image requirement + polling
# ---------------------------------------------------------------------------

class TestCreateVideoPin:
    @pytest.mark.asyncio
    async def test_video_pin_requires_cover_image(
        self,
        authenticated_poster: PinterestPoster,
    ) -> None:
        """create_video_pin() must raise ValidationError when cover_image_url is empty."""
        with pytest.raises(ValidationError, match="cover_image_url"):
            await authenticated_poster.create_video_pin(
                board_id=SAMPLE_BOARD_ID,
                video_file_path=SAMPLE_VIDEO_PATH,
                cover_image_url="",   # Empty — must be rejected
                title="Video Pin",
            )

    @pytest.mark.asyncio
    async def test_video_pin_requires_cover_image_not_none(
        self,
        authenticated_poster: PinterestPoster,
    ) -> None:
        """create_video_pin() must raise ValidationError when cover_image_url is None."""
        with pytest.raises(ValidationError, match="cover_image_url"):
            await authenticated_poster.create_video_pin(
                board_id=SAMPLE_BOARD_ID,
                video_file_path=SAMPLE_VIDEO_PATH,
                cover_image_url=None,  # type: ignore[arg-type]
                title="Video Pin",
            )

    @pytest.mark.asyncio
    async def test_video_pin_polls_until_complete(
        self,
        authenticated_poster: PinterestPoster,
    ) -> None:
        """
        create_video_pin() must poll _poll_media_registration() and
        return a success dict when polling completes.
        """
        pin_id = "pin_video_xyz"

        register_response = MagicMock()
        register_response.status_code = 200
        register_response.raise_for_status = MagicMock()
        register_response.json.return_value = {
            "media_id": "media_001",
            "upload_url": "https://s3.amazonaws.com/pinterest-upload/test",
        }

        upload_response = MagicMock()
        upload_response.status_code = 200
        upload_response.raise_for_status = MagicMock()

        pin_create_response = MagicMock()
        pin_create_response.status_code = 201
        pin_create_response.raise_for_status = MagicMock()
        pin_create_response.json.return_value = {"id": pin_id}

        file_size = 5 * 1024 * 1024  # 5 MB

        with (
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=file_size),
            patch("builtins.open", mock_open(read_data=b"x" * file_size)),
            patch("social.pinterest_poster.httpx.AsyncClient") as mock_client_class,
            patch.object(
                authenticated_poster,
                "_poll_media_registration",
                new_callable=AsyncMock,
            ) as mock_poll,
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(
                side_effect=[register_response, pin_create_response]
            )
            mock_client.put = AsyncMock(return_value=upload_response)
            mock_client_class.return_value = mock_client

            result = await authenticated_poster.create_video_pin(
                board_id=SAMPLE_BOARD_ID,
                video_file_path=SAMPLE_VIDEO_PATH,
                cover_image_url=SAMPLE_IMAGE_URL,
                title="Video Pin Test",
            )

        mock_poll.assert_called_once_with("media_001")
        assert result["success"] is True
        assert result["post_id"] == pin_id


# ---------------------------------------------------------------------------
# _poll_media_registration() — timeout and success
# ---------------------------------------------------------------------------

class TestPollMediaRegistration:
    @pytest.mark.asyncio
    async def test_polling_times_out_after_max_attempts(
        self,
        authenticated_poster: PinterestPoster,
    ) -> None:
        """_poll_media_registration() must raise PollingTimeoutError on timeout."""
        processing_response = MagicMock()
        processing_response.status_code = 200
        processing_response.raise_for_status = MagicMock()
        processing_response.json.return_value = {"status": "processing"}

        with (
            patch("social.pinterest_poster.httpx.AsyncClient") as mock_client_class,
            patch("social.pinterest_poster.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=processing_response)
            mock_client_class.return_value = mock_client

            with pytest.raises(PollingTimeoutError):
                await authenticated_poster._poll_media_registration(
                    "media_timeout",
                    max_attempts=3,
                    poll_interval=0,
                )

    @pytest.mark.asyncio
    async def test_polling_succeeds_on_succeeded_status(
        self,
        authenticated_poster: PinterestPoster,
    ) -> None:
        """_poll_media_registration() must return without raising when 'succeeded'."""
        succeeded_response = MagicMock()
        succeeded_response.status_code = 200
        succeeded_response.raise_for_status = MagicMock()
        succeeded_response.json.return_value = {"status": "succeeded"}

        with (
            patch("social.pinterest_poster.httpx.AsyncClient") as mock_client_class,
            patch("social.pinterest_poster.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=succeeded_response)
            mock_client_class.return_value = mock_client

            await authenticated_poster._poll_media_registration(
                "media_ok", max_attempts=3, poll_interval=0
            )


# ---------------------------------------------------------------------------
# get_boards() — board list
# ---------------------------------------------------------------------------

class TestGetBoards:
    @pytest.mark.asyncio
    async def test_get_boards_returns_list(
        self,
        authenticated_poster: PinterestPoster,
    ) -> None:
        """get_boards() must return a dict with 'items' containing board data."""
        boards_data = [
            {"id": "b1", "name": "Board One", "pin_count": 10},
            {"id": "b2", "name": "Board Two", "pin_count": 5},
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"items": boards_data, "bookmark": None}

        with patch("social.pinterest_poster.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await authenticated_poster.get_boards()

        assert "items" in result
        assert len(result["items"]) == 2
        assert result["items"][0]["id"] == "b1"
        assert result["items"][1]["name"] == "Board Two"

    @pytest.mark.asyncio
    async def test_get_boards_simulation_mode(
        self,
        authenticated_poster: PinterestPoster,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """get_boards() in simulation mode must return simulated board data."""
        monkeypatch.setenv("SOCIAL_SIMULATION_MODE", "true")

        result = await authenticated_poster.get_boards()

        assert "items" in result
        assert len(result["items"]) >= 1
        assert result["items"][0]["id"] == "sim_board_1"
