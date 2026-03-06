"""
Unit tests for TikTokPoster.

All external API calls and file system operations are mocked.
No real credentials required — tests are fully isolated.
"""

import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

# Ensure src/ is on path for direct pytest invocation from project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from exceptions import AuthError, FileSizeError, RateLimitError, UploadError
from registry.poster_registry import PosterRegistry

# Import triggers _register() at module level.
import social.tiktok_poster as tiktok_module
from social.tiktok_poster import TikTokPoster, MAX_VIDEO_SIZE_BYTES, CHUNK_SIZE_BYTES


SAMPLE_FILE_PATH = "/tmp/test_video.mp4"
SAMPLE_FILE_SIZE_SMALL = 1 * 1024 * 1024   # 1 MB — single PUT
SAMPLE_FILE_SIZE_LARGE = 10 * 1024 * 1024  # 10 MB — requires chunking


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def poster() -> TikTokPoster:
    return TikTokPoster()


@pytest.fixture
def authenticated_poster(poster: TikTokPoster) -> TikTokPoster:
    """Return a poster with auth state already set (bypasses authenticate())."""
    poster._access_token = "test_access_token"
    poster._client_key = "test_client_key"
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
        poster: TikTokPoster,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """authenticate() must set _authenticated=True on valid credentials."""
        monkeypatch.setenv("TIKTOK_CLIENT_KEY", "test_key")
        monkeypatch.setenv("TIKTOK_CLIENT_SECRET", "test_secret")
        monkeypatch.setenv("TIKTOK_ACCESS_TOKEN", "test_token")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"data": {"open_id": "uid_123"}}

        with patch("social.tiktok_poster.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            await poster.authenticate()

        assert poster._authenticated is True
        assert poster._access_token == "test_token"

    @pytest.mark.asyncio
    async def test_authenticate_failure_raises_auth_error_missing_env(
        self,
        poster: TikTokPoster,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """authenticate() must raise AuthError when credentials are missing."""
        monkeypatch.delenv("TIKTOK_CLIENT_KEY", raising=False)
        monkeypatch.delenv("TIKTOK_CLIENT_SECRET", raising=False)
        monkeypatch.delenv("TIKTOK_ACCESS_TOKEN", raising=False)

        with pytest.raises(AuthError):
            await poster.authenticate()

    @pytest.mark.asyncio
    async def test_authenticate_failure_raises_auth_error_on_401(
        self,
        poster: TikTokPoster,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """authenticate() must raise AuthError when API returns 401."""
        monkeypatch.setenv("TIKTOK_CLIENT_KEY", "k")
        monkeypatch.setenv("TIKTOK_CLIENT_SECRET", "s")
        monkeypatch.setenv("TIKTOK_ACCESS_TOKEN", "bad_token")

        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("social.tiktok_poster.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            with pytest.raises(AuthError, match="401"):
                await poster.authenticate()


# ---------------------------------------------------------------------------
# validate_video() / post_video() size limit tests
# ---------------------------------------------------------------------------

class TestVideoSizeLimits:
    @pytest.mark.asyncio
    async def test_post_video_exceeds_size_limit_raises_file_size_error(
        self,
        authenticated_poster: TikTokPoster,
    ) -> None:
        """post_video() must raise FileSizeError when file exceeds 4 GB."""
        oversized = MAX_VIDEO_SIZE_BYTES + 1

        with (
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=oversized),
        ):
            with pytest.raises(FileSizeError):
                await authenticated_poster.post_video(
                    SAMPLE_FILE_PATH, "Test title"
                )

    @pytest.mark.asyncio
    async def test_validate_video_raises_file_not_found(
        self,
        authenticated_poster: TikTokPoster,
    ) -> None:
        """validate_video() must raise FileNotFoundError for non-existent path."""
        with patch("os.path.exists", return_value=False):
            with pytest.raises(FileNotFoundError):
                await authenticated_poster.validate_video("/nonexistent/video.mp4")

    @pytest.mark.asyncio
    async def test_validate_video_passes_within_limit(
        self,
        authenticated_poster: TikTokPoster,
    ) -> None:
        """validate_video() must not raise when file size is within limit."""
        with (
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=SAMPLE_FILE_SIZE_SMALL),
        ):
            # Should complete without raising.
            await authenticated_poster.validate_video(SAMPLE_FILE_PATH)


# ---------------------------------------------------------------------------
# post_video() success path
# ---------------------------------------------------------------------------

class TestPostVideo:
    @pytest.mark.asyncio
    async def test_post_video_success_returns_post_id_and_url(
        self,
        authenticated_poster: TikTokPoster,
    ) -> None:
        """post_video() must return dict with success=True, post_id, and url."""
        publish_id = "pub_abc123"

        init_response = MagicMock()
        init_response.status_code = 200
        init_response.raise_for_status = MagicMock()
        init_response.json.return_value = {
            "data": {
                "publish_id": publish_id,
                "upload_url": "https://upload.tiktok.com/v1/test",
            }
        }

        chunk_response = MagicMock()
        chunk_response.status_code = 206
        chunk_response.raise_for_status = MagicMock()

        status_response = MagicMock()
        status_response.status_code = 200
        status_response.raise_for_status = MagicMock()
        status_response.json.return_value = {
            "data": {"status": "PUBLISH_COMPLETE", "publish_id": publish_id}
        }

        with (
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=SAMPLE_FILE_SIZE_SMALL),
            patch(
                "builtins.open",
                mock_open(read_data=b"x" * SAMPLE_FILE_SIZE_SMALL),
            ),
            patch("social.tiktok_poster.httpx.AsyncClient") as mock_client_class,
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            # post (init), put (chunk), post (status poll)
            mock_client.post = AsyncMock(
                side_effect=[init_response, status_response]
            )
            mock_client.put = AsyncMock(return_value=chunk_response)
            mock_client_class.return_value = mock_client

            result = await authenticated_poster.post_video(
                SAMPLE_FILE_PATH, "My TikTok video"
            )

        assert result["success"] is True
        assert result["post_id"] is not None
        assert result["url"] is not None
        assert result["error"] is None
        assert result["platform"] == "tiktok"


# ---------------------------------------------------------------------------
# _chunk_upload() retry logic
# ---------------------------------------------------------------------------

class TestChunkUpload:
    @pytest.mark.asyncio
    async def test_chunk_upload_retries_on_network_error(
        self,
        authenticated_poster: TikTokPoster,
    ) -> None:
        """_chunk_upload() must retry on network error and succeed on final attempt."""
        file_size = CHUNK_SIZE_BYTES  # exactly one chunk

        success_response = MagicMock()
        success_response.status_code = 206
        success_response.raise_for_status = MagicMock()

        # First attempt raises, second succeeds.
        call_count = {"n": 0}

        async def fake_put(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise httpx.NetworkError("connection reset")
            return success_response

        import httpx  # noqa: PLC0415 — imported inside test intentionally

        with (
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=file_size),
            patch("builtins.open", mock_open(read_data=b"x" * file_size)),
            patch("social.tiktok_poster.httpx.AsyncClient") as mock_client_class,
            patch("social.tiktok_poster.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.put = fake_put
            mock_client_class.return_value = mock_client

            # Should not raise — retried and succeeded.
            await authenticated_poster._chunk_upload(
                SAMPLE_FILE_PATH,
                "https://upload.tiktok.com/test",
                total_chunks=1,
                max_retries=3,
            )

        assert call_count["n"] == 2

    @pytest.mark.asyncio
    async def test_chunk_upload_raises_upload_error_after_max_retries(
        self,
        authenticated_poster: TikTokPoster,
    ) -> None:
        """_chunk_upload() must raise UploadError when all retries are exhausted."""
        file_size = CHUNK_SIZE_BYTES

        import httpx  # noqa: PLC0415

        with (
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=file_size),
            patch("builtins.open", mock_open(read_data=b"x" * file_size)),
            patch("social.tiktok_poster.httpx.AsyncClient") as mock_client_class,
            patch("social.tiktok_poster.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.put = AsyncMock(
                side_effect=httpx.NetworkError("always fails")
            )
            mock_client_class.return_value = mock_client

            with pytest.raises(UploadError):
                await authenticated_poster._chunk_upload(
                    SAMPLE_FILE_PATH,
                    "https://upload.tiktok.com/test",
                    total_chunks=1,
                    max_retries=3,
                )


# ---------------------------------------------------------------------------
# Rate limit tests
# ---------------------------------------------------------------------------

class TestRateLimit:
    @pytest.mark.asyncio
    async def test_rate_limit_exceeded_raises_rate_limit_error_on_init(
        self,
        authenticated_poster: TikTokPoster,
    ) -> None:
        """_initialize_upload() must raise RateLimitError on 429 response."""
        rate_limit_response = MagicMock()
        rate_limit_response.status_code = 429
        rate_limit_response.raise_for_status = MagicMock()

        with (
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=SAMPLE_FILE_SIZE_SMALL),
            patch("social.tiktok_poster.httpx.AsyncClient") as mock_client_class,
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=rate_limit_response)
            mock_client_class.return_value = mock_client

            with pytest.raises(RateLimitError):
                await authenticated_poster._initialize_upload(
                    SAMPLE_FILE_PATH, "title", "desc"
                )

    @pytest.mark.asyncio
    async def test_rate_limit_on_auth_raises_rate_limit_error(
        self,
        poster: TikTokPoster,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """authenticate() must raise RateLimitError when user info endpoint returns 429."""
        monkeypatch.setenv("TIKTOK_CLIENT_KEY", "k")
        monkeypatch.setenv("TIKTOK_CLIENT_SECRET", "s")
        monkeypatch.setenv("TIKTOK_ACCESS_TOKEN", "t")

        mock_response = MagicMock()
        mock_response.status_code = 429

        with patch("social.tiktok_poster.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            with pytest.raises(RateLimitError):
                await poster.authenticate()
