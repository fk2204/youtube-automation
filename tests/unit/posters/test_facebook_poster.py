"""
Unit tests for FacebookPoster.

All API calls and file system operations are mocked.
No real credentials required. Tests verify video upload,
resumable upload triggering, Page token enforcement, and registration.
"""

import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from exceptions import AuthError, FileSizeError, RateLimitError, UploadError
from registry.poster_registry import PosterRegistry

import social.facebook_poster as facebook_module
from social.facebook_poster import (
    FacebookPoster,
    RESUMABLE_THRESHOLD_BYTES,
    MAX_VIDEO_SIZE_BYTES,
    CHUNK_SIZE_BYTES,
)

SAMPLE_FILE_PATH = "/tmp/test_video.mp4"
SAMPLE_VIDEO_URL = "https://cdn.example.com/video.mp4"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def poster() -> FacebookPoster:
    return FacebookPoster()


@pytest.fixture
def authenticated_poster(poster: FacebookPoster) -> FacebookPoster:
    poster._page_token = "EAAtest_page_token"
    poster._page_id = "1234567890"
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
    async def test_authenticate_success_with_page_token(
        self,
        poster: FacebookPoster,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """authenticate() must succeed when debug_token returns type='PAGE'."""
        monkeypatch.setenv("FACEBOOK_PAGE_ACCESS_TOKEN", "EAApage_token")
        monkeypatch.setenv("FACEBOOK_PAGE_ID", "123456")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": {"type": "PAGE", "is_valid": True}
        }

        with patch("social.facebook_poster.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            await poster.authenticate()

        assert poster._authenticated is True
        assert poster._page_token == "EAApage_token"

    @pytest.mark.asyncio
    async def test_page_token_required_raises_auth_error_on_user_token(
        self,
        poster: FacebookPoster,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """authenticate() must raise AuthError when a User token is provided."""
        monkeypatch.setenv("FACEBOOK_PAGE_ACCESS_TOKEN", "EAAuser_token")
        monkeypatch.setenv("FACEBOOK_PAGE_ID", "123456")

        # debug_token says type=USER — this must be rejected.
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": {"type": "USER", "is_valid": True}
        }

        with patch("social.facebook_poster.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            with pytest.raises(AuthError, match="User token"):
                await poster.authenticate()

    @pytest.mark.asyncio
    async def test_authenticate_missing_env_raises_auth_error(
        self,
        poster: FacebookPoster,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """authenticate() must raise AuthError when credentials are absent."""
        monkeypatch.delenv("FACEBOOK_PAGE_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("FACEBOOK_PAGE_ID", raising=False)

        with pytest.raises(AuthError):
            await poster.authenticate()

    @pytest.mark.asyncio
    async def test_authenticate_401_raises_auth_error(
        self,
        poster: FacebookPoster,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """authenticate() must raise AuthError on 401 from debug_token."""
        monkeypatch.setenv("FACEBOOK_PAGE_ACCESS_TOKEN", "bad_token")
        monkeypatch.setenv("FACEBOOK_PAGE_ID", "123")

        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("social.facebook_poster.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            with pytest.raises(AuthError, match="401"):
                await poster.authenticate()


# ---------------------------------------------------------------------------
# post_video() — success and resumable upload triggering
# ---------------------------------------------------------------------------

class TestPostVideo:
    @pytest.mark.asyncio
    async def test_post_video_success(
        self,
        authenticated_poster: FacebookPoster,
    ) -> None:
        """post_video() must return success dict with video_id for small files."""
        video_id = "vid_fb_789"
        small_file_size = 100 * 1024 * 1024  # 100 MB — below resumable threshold

        upload_response = MagicMock()
        upload_response.status_code = 200
        upload_response.raise_for_status = MagicMock()
        upload_response.json.return_value = {"id": video_id}

        with (
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=small_file_size),
            patch("builtins.open", mock_open(read_data=b"x" * 1024)),
            patch("social.facebook_poster.httpx.AsyncClient") as mock_client_class,
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=upload_response)
            mock_client_class.return_value = mock_client

            result = await authenticated_poster.post_video(
                SAMPLE_FILE_PATH, "Test Video"
            )

        assert result["success"] is True
        assert result["post_id"] == video_id
        assert result["url"] is not None
        assert result["error"] is None
        assert result["platform"] == "facebook"

    @pytest.mark.asyncio
    async def test_resumable_upload_triggered_for_large_files(
        self,
        authenticated_poster: FacebookPoster,
    ) -> None:
        """
        post_video() must call _resumable_upload() when file >= RESUMABLE_THRESHOLD_BYTES.
        Verifies the routing decision — _resumable_upload is mocked to return a video_id.
        """
        large_file_size = RESUMABLE_THRESHOLD_BYTES + 1  # just above 1 GB threshold

        with (
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=large_file_size),
            patch.object(
                authenticated_poster,
                "_resumable_upload",
                new_callable=AsyncMock,
                return_value="resumable_vid_001",
            ) as mock_resumable,
        ):
            result = await authenticated_poster.post_video(
                SAMPLE_FILE_PATH, "Large Video"
            )

        mock_resumable.assert_called_once_with(
            SAMPLE_FILE_PATH, "Large Video", ""
        )
        assert result["success"] is True
        assert result["post_id"] == "resumable_vid_001"

    @pytest.mark.asyncio
    async def test_post_video_exceeds_max_size_raises_file_size_error(
        self,
        authenticated_poster: FacebookPoster,
    ) -> None:
        """post_video() must raise FileSizeError when file exceeds 10 GB."""
        with (
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=MAX_VIDEO_SIZE_BYTES + 1),
        ):
            with pytest.raises(FileSizeError):
                await authenticated_poster.post_video(SAMPLE_FILE_PATH, "Too Big")

    @pytest.mark.asyncio
    async def test_post_video_rate_limit_raises_rate_limit_error(
        self,
        authenticated_poster: FacebookPoster,
    ) -> None:
        """post_video() must raise RateLimitError on 429 during upload."""
        small_size = 50 * 1024 * 1024

        rate_limit_response = MagicMock()
        rate_limit_response.status_code = 429
        rate_limit_response.raise_for_status = MagicMock()

        with (
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=small_size),
            patch("builtins.open", mock_open(read_data=b"x" * 1024)),
            patch("social.facebook_poster.httpx.AsyncClient") as mock_client_class,
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=rate_limit_response)
            mock_client_class.return_value = mock_client

            with pytest.raises(RateLimitError):
                await authenticated_poster.post_video(SAMPLE_FILE_PATH, "Title")


# ---------------------------------------------------------------------------
# _resumable_upload() — three-phase test
# ---------------------------------------------------------------------------

class TestResumableUpload:
    @pytest.mark.asyncio
    async def test_resumable_upload_completes_three_phases(
        self,
        authenticated_poster: FacebookPoster,
    ) -> None:
        """
        _resumable_upload() must call start, transfer, and finish phases
        and return the video_id from the finish response.
        """
        file_size = CHUNK_SIZE_BYTES  # exactly one chunk

        start_response = MagicMock()
        start_response.raise_for_status = MagicMock()
        start_response.json.return_value = {
            "upload_session_id": "session_abc",
            "start_offset": 0,
        }

        transfer_response = MagicMock()
        transfer_response.status_code = 200
        transfer_response.raise_for_status = MagicMock()
        transfer_response.json.return_value = {"start_offset": file_size}

        finish_response = MagicMock()
        finish_response.raise_for_status = MagicMock()
        finish_response.json.return_value = {"video_id": "vid_resumable_999"}

        with (
            patch("os.path.getsize", return_value=file_size),
            patch("builtins.open", mock_open(read_data=b"x" * file_size)),
            patch("social.facebook_poster.httpx.AsyncClient") as mock_client_class,
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(
                side_effect=[start_response, transfer_response, finish_response]
            )
            mock_client_class.return_value = mock_client

            video_id = await authenticated_poster._resumable_upload(
                SAMPLE_FILE_PATH, "Title", "Desc"
            )

        assert video_id == "vid_resumable_999"
