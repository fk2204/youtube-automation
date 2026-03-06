"""
Unit tests for InstagramPoster.

All external API calls are mocked. No real credentials required.
Tests verify container polling logic, carousel limits, format validation,
and registration.
"""

import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from exceptions import AuthError, PollingTimeoutError, RateLimitError, ValidationError
from registry.poster_registry import PosterRegistry

import social.instagram_poster as instagram_module
from social.instagram_poster import (
    InstagramPoster,
    MAX_CAROUSEL_ITEMS,
    MAX_POLL_ATTEMPTS,
)

SAMPLE_VIDEO_URL = "https://cdn.example.com/video.mp4"
SAMPLE_IMAGE_URL = "https://cdn.example.com/image.jpg"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def poster() -> InstagramPoster:
    return InstagramPoster()


@pytest.fixture
def authenticated_poster(poster: InstagramPoster) -> InstagramPoster:
    poster._access_token = "test_access_token"
    poster._account_id = "12345678"
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
        poster: InstagramPoster,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """authenticate() must set _authenticated=True on valid credentials."""
        monkeypatch.setenv("INSTAGRAM_ACCESS_TOKEN", "valid_token")
        monkeypatch.setenv("INSTAGRAM_BUSINESS_ACCOUNT_ID", "9876543")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"id": "9876543", "name": "Test Account"}

        with patch("social.instagram_poster.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            await poster.authenticate()

        assert poster._authenticated is True

    @pytest.mark.asyncio
    async def test_authenticate_failure_missing_env(
        self,
        poster: InstagramPoster,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """authenticate() must raise AuthError when env vars are missing."""
        monkeypatch.delenv("INSTAGRAM_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("INSTAGRAM_BUSINESS_ACCOUNT_ID", raising=False)

        with pytest.raises(AuthError):
            await poster.authenticate()


# ---------------------------------------------------------------------------
# _poll_container_status() — polling logic
# ---------------------------------------------------------------------------

class TestContainerPolling:
    @pytest.mark.asyncio
    async def test_container_polling_times_out_after_max_attempts(
        self,
        authenticated_poster: InstagramPoster,
    ) -> None:
        """
        _poll_container_status() must raise PollingTimeoutError when
        FINISHED is never returned within MAX_POLL_ATTEMPTS attempts.
        """
        # Every poll returns IN_PROGRESS — never FINISHED.
        in_progress_response = MagicMock()
        in_progress_response.status_code = 200
        in_progress_response.raise_for_status = MagicMock()
        in_progress_response.json.return_value = {"status_code": "IN_PROGRESS"}

        with (
            patch("social.instagram_poster.httpx.AsyncClient") as mock_client_class,
            patch("social.instagram_poster.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=in_progress_response)
            mock_client_class.return_value = mock_client

            with pytest.raises(PollingTimeoutError):
                await authenticated_poster._poll_container_status(
                    "container_xyz",
                    max_attempts=MAX_POLL_ATTEMPTS,
                    poll_interval=0,
                )

    @pytest.mark.asyncio
    async def test_container_polling_succeeds_on_finished(
        self,
        authenticated_poster: InstagramPoster,
    ) -> None:
        """_poll_container_status() must return without raising when FINISHED."""
        finished_response = MagicMock()
        finished_response.status_code = 200
        finished_response.raise_for_status = MagicMock()
        finished_response.json.return_value = {"status_code": "FINISHED"}

        with (
            patch("social.instagram_poster.httpx.AsyncClient") as mock_client_class,
            patch("social.instagram_poster.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=finished_response)
            mock_client_class.return_value = mock_client

            # Should not raise.
            await authenticated_poster._poll_container_status(
                "container_abc",
                max_attempts=3,
                poll_interval=0,
            )

    @pytest.mark.asyncio
    async def test_container_error_status_raises_validation_error(
        self,
        authenticated_poster: InstagramPoster,
    ) -> None:
        """_poll_container_status() must raise ValidationError on ERROR status."""
        error_response = MagicMock()
        error_response.status_code = 200
        error_response.raise_for_status = MagicMock()
        error_response.json.return_value = {"status_code": "ERROR"}

        with (
            patch("social.instagram_poster.httpx.AsyncClient") as mock_client_class,
            patch("social.instagram_poster.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=error_response)
            mock_client_class.return_value = mock_client

            with pytest.raises(ValidationError):
                await authenticated_poster._poll_container_status(
                    "container_err",
                    max_attempts=3,
                    poll_interval=0,
                )


# ---------------------------------------------------------------------------
# post_reel() success path
# ---------------------------------------------------------------------------

class TestPostReel:
    @pytest.mark.asyncio
    async def test_reel_post_returns_media_id_and_permalink(
        self,
        authenticated_poster: InstagramPoster,
    ) -> None:
        """post_reel() must return dict with success=True and media_id."""
        media_id = "media_abc123"

        create_response = MagicMock()
        create_response.status_code = 200
        create_response.raise_for_status = MagicMock()
        create_response.json.return_value = {"id": "container_001"}

        finished_response = MagicMock()
        finished_response.status_code = 200
        finished_response.raise_for_status = MagicMock()
        finished_response.json.return_value = {"status_code": "FINISHED"}

        publish_response = MagicMock()
        publish_response.status_code = 200
        publish_response.raise_for_status = MagicMock()
        publish_response.json.return_value = {"id": media_id}

        with (
            patch("social.instagram_poster.httpx.AsyncClient") as mock_client_class,
            patch("social.instagram_poster.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(
                side_effect=[create_response, publish_response]
            )
            mock_client.get = AsyncMock(return_value=finished_response)
            mock_client_class.return_value = mock_client

            result = await authenticated_poster.post_reel(
                SAMPLE_VIDEO_URL, caption="#test"
            )

        assert result["success"] is True
        assert result["post_id"] == media_id
        assert result["url"] is not None
        assert result["error"] is None
        assert result["platform"] == "instagram"


# ---------------------------------------------------------------------------
# post_carousel() — item limit validation
# ---------------------------------------------------------------------------

class TestPostCarousel:
    @pytest.mark.asyncio
    async def test_carousel_rejects_more_than_10_items(
        self,
        authenticated_poster: InstagramPoster,
    ) -> None:
        """post_carousel() must raise ValidationError for > 10 items."""
        too_many_urls = [f"https://cdn.example.com/img{i}.jpg" for i in range(11)]

        with pytest.raises(ValidationError, match="10"):
            await authenticated_poster.post_carousel(too_many_urls, caption="test")

    @pytest.mark.asyncio
    async def test_carousel_accepts_exactly_10_items_in_simulation(
        self,
        authenticated_poster: InstagramPoster,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """post_carousel() must not raise for exactly 10 items."""
        monkeypatch.setenv("SOCIAL_SIMULATION_MODE", "true")
        exactly_10 = [f"https://cdn.example.com/img{i}.jpg" for i in range(10)]

        result = await authenticated_poster.post_carousel(exactly_10)
        assert result["success"] is True


# ---------------------------------------------------------------------------
# _validate_media() — format validation
# ---------------------------------------------------------------------------

class TestValidateMedia:
    def test_invalid_format_raises_validation_error(
        self,
        poster: InstagramPoster,
    ) -> None:
        """_validate_media() must raise ValidationError for unsupported format."""
        with patch("os.path.exists", return_value=True):
            with pytest.raises(ValidationError, match="bmp"):
                poster._validate_media("/tmp/image.bmp", {".jpg", ".jpeg", ".png"})

    def test_valid_format_passes(self, poster: InstagramPoster) -> None:
        """_validate_media() must not raise for supported format."""
        with patch("os.path.exists", return_value=True):
            poster._validate_media("/tmp/image.jpg", {".jpg", ".jpeg", ".png"})

    def test_missing_file_raises_file_not_found(
        self,
        poster: InstagramPoster,
    ) -> None:
        """_validate_media() must raise FileNotFoundError for missing file."""
        with patch("os.path.exists", return_value=False):
            with pytest.raises(FileNotFoundError):
                poster._validate_media("/tmp/missing.jpg", {".jpg"})
