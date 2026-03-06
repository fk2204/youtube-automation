"""
Unit tests for MediumRepurposer.

All external API calls are mocked. No real credentials required.
Tests are fully isolated from the network and file system.
"""

import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from distribution.repurposers.repurposer_registry import RepurposerRegistry
import distribution.repurposers.medium_repurposer as medium_module
from distribution.repurposers.medium_repurposer import MediumRepurposer, MAX_TAGS

SAMPLE_CONFIG = {"default_status": "draft"}
SAMPLE_CONTENT = "# My Article\n\nThis is the body of the article."


@pytest.fixture
def repurposer() -> MediumRepurposer:
    return MediumRepurposer(SAMPLE_CONFIG)


@pytest.fixture
def authenticated_repurposer(repurposer: MediumRepurposer) -> MediumRepurposer:
    """Return a repurposer with auth state already set."""
    repurposer._token = "test_medium_token"
    repurposer._author_id = "author_abc123"
    repurposer._authenticated = True
    return repurposer


@pytest.fixture(autouse=True)
def clear_simulation_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SOCIAL_SIMULATION_MODE", raising=False)



class TestAuthenticate:
    @pytest.mark.asyncio
    async def test_authenticate_success_returns_true(
        self,
        repurposer: MediumRepurposer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """authenticate() returns True and sets _authenticated on valid token."""
        monkeypatch.setenv("MEDIUM_INTEGRATION_TOKEN", "valid_token_123")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"data": {"id": "author_xyz", "name": "Test Author"}}

        with patch("distribution.repurposers.medium_repurposer.httpx.AsyncClient") as mock_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_class.return_value = mock_client

            result = await repurposer.authenticate()

        assert result is True
        assert repurposer._authenticated is True
        assert repurposer._token == "valid_token_123"
        assert repurposer._author_id == "author_xyz"

    @pytest.mark.asyncio
    async def test_authenticate_failure_missing_env_returns_false(
        self,
        repurposer: MediumRepurposer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """authenticate() returns False when MEDIUM_INTEGRATION_TOKEN is missing."""
        monkeypatch.delenv("MEDIUM_INTEGRATION_TOKEN", raising=False)
        result = await repurposer.authenticate()
        assert result is False
        assert repurposer._authenticated is False

    @pytest.mark.asyncio
    async def test_authenticate_failure_401_returns_false(
        self,
        repurposer: MediumRepurposer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """authenticate() returns False when API returns 401."""
        monkeypatch.setenv("MEDIUM_INTEGRATION_TOKEN", "bad_token")

        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("distribution.repurposers.medium_repurposer.httpx.AsyncClient") as mock_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_class.return_value = mock_client

            result = await repurposer.authenticate()

        assert result is False


class TestRepurpose:
    @pytest.mark.asyncio
    async def test_repurpose_success_returns_post_id_and_url(
        self,
        authenticated_repurposer: MediumRepurposer,
    ) -> None:
        """repurpose() returns success dict with post_id and url."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "id": "post_abc123",
                "url": "https://medium.com/@user/my-article-abc123",
                "publishStatus": "draft",
            }
        }

        with patch("distribution.repurposers.medium_repurposer.httpx.AsyncClient") as mock_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_class.return_value = mock_client

            result = await authenticated_repurposer.repurpose(SAMPLE_CONTENT)

        assert result["success"] is True
        assert result["post_id"] == "post_abc123"
        assert result["url"] == "https://medium.com/@user/my-article-abc123"
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_repurpose_default_status_is_draft(
        self,
        authenticated_repurposer: MediumRepurposer,
    ) -> None:
        """repurpose() uses 'draft' as default status."""
        captured_payload: dict = {}

        async def capture_post(*args, **kwargs):
            captured_payload.update(kwargs.get("json", {}))
            mock_resp = MagicMock()
            mock_resp.status_code = 201
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {"data": {"id": "p1", "url": "https://medium.com/p1", "publishStatus": "draft"}}
            return mock_resp

        with patch("distribution.repurposers.medium_repurposer.httpx.AsyncClient") as mock_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = capture_post
            mock_class.return_value = mock_client

            result = await authenticated_repurposer.repurpose(SAMPLE_CONTENT)

        assert captured_payload.get("publishStatus") == "draft"

    @pytest.mark.asyncio
    async def test_repurpose_tags_capped_at_five(
        self,
        authenticated_repurposer: MediumRepurposer,
    ) -> None:
        """repurpose() silently drops tags beyond MAX_TAGS (5)."""
        captured_payload: dict = {}

        async def capture_post(*args, **kwargs):
            captured_payload.update(kwargs.get("json", {}))
            mock_resp = MagicMock()
            mock_resp.status_code = 201
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {"data": {"id": "p2", "url": "https://medium.com/p2", "publishStatus": "draft"}}
            return mock_resp

        too_many_tags = ["a", "b", "c", "d", "e", "f", "g"]

        with patch("distribution.repurposers.medium_repurposer.httpx.AsyncClient") as mock_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = capture_post
            mock_class.return_value = mock_client

            result = await authenticated_repurposer.repurpose(
                SAMPLE_CONTENT, tags=too_many_tags
            )

        assert len(captured_payload.get("tags", [])) == MAX_TAGS

    @pytest.mark.asyncio
    async def test_repurpose_not_authenticated_returns_error(
        self,
        repurposer: MediumRepurposer,
    ) -> None:
        """repurpose() returns error dict when not authenticated."""
        result = await repurposer.repurpose(SAMPLE_CONTENT)
        assert result["success"] is False
        assert result["error"] is not None


class TestGetPublications:
    @pytest.mark.asyncio
    async def test_get_publications_returns_list(
        self,
        authenticated_repurposer: MediumRepurposer,
    ) -> None:
        """get_publications() returns list of publication dicts."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": [{"id": "pub1", "name": "My Publication"}]
        }

        with patch("distribution.repurposers.medium_repurposer.httpx.AsyncClient") as mock_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_class.return_value = mock_client

            pubs = await authenticated_repurposer.get_publications()

        assert isinstance(pubs, list)
        assert len(pubs) == 1
        assert pubs[0]["id"] == "pub1"
