"""
Unit tests for SubstackRepurposer.

All file system operations are mocked. No real disk writes occur.
Tests verify content_only automation level, HTML generation, and draft path.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, mock_open, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from distribution.repurposers.repurposer_registry import RepurposerRegistry
import distribution.repurposers.substack_repurposer as substack_module
from distribution.repurposers.substack_repurposer import SubstackRepurposer

SAMPLE_CONFIG = {"export_path": "/tmp/substack_test_exports"}
SAMPLE_CONTENT = "# Test Article\n\nThis is a paragraph.\n\n## Subheading\n\nMore content here."


@pytest.fixture
def repurposer() -> SubstackRepurposer:
    return SubstackRepurposer(SAMPLE_CONFIG)


@pytest.fixture
def authenticated_repurposer(repurposer: SubstackRepurposer) -> SubstackRepurposer:
    """Return a repurposer with auth state already set."""
    repurposer._authenticated = True
    return repurposer


@pytest.fixture(autouse=True)
def clear_simulation_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SOCIAL_SIMULATION_MODE", raising=False)



class TestAuthenticate:
    @pytest.mark.asyncio
    async def test_authenticate_success_writable_path(
        self,
        repurposer: SubstackRepurposer,
    ) -> None:
        """authenticate() returns True when export_path is writable."""
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("os.access", return_value=True),
        ):
            result = await repurposer.authenticate()

        assert result is True
        assert repurposer._authenticated is True

    @pytest.mark.asyncio
    async def test_authenticate_fails_not_writable(
        self,
        repurposer: SubstackRepurposer,
    ) -> None:
        """authenticate() returns False when export_path is not writable."""
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("os.access", return_value=False),
        ):
            result = await repurposer.authenticate()

        assert result is False
        assert repurposer._authenticated is False

    @pytest.mark.asyncio
    async def test_authenticate_fails_empty_export_path(self) -> None:
        """authenticate() returns False when export_path is not configured."""
        repurposer = SubstackRepurposer({"export_path": ""})
        result = await repurposer.authenticate()
        assert result is False

    @pytest.mark.asyncio
    async def test_authenticate_creates_missing_directory(
        self,
        repurposer: SubstackRepurposer,
    ) -> None:
        """authenticate() creates the export_path directory if it does not exist."""
        with (
            patch("pathlib.Path.exists", return_value=False),
            patch("pathlib.Path.mkdir") as mock_mkdir,
            patch("os.access", return_value=True),
        ):
            result = await repurposer.authenticate()

        mock_mkdir.assert_called_once()
        assert result is True


class TestRepurpose:
    @pytest.mark.asyncio
    async def test_repurpose_writes_html_file(
        self,
        authenticated_repurposer: SubstackRepurposer,
    ) -> None:
        """repurpose() writes an HTML file to export_path."""
        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            result = await authenticated_repurposer.repurpose(SAMPLE_CONTENT)

        mock_file.assert_called_once()
        assert result["success"] is True
        assert result["draft_path"] is not None
        assert result["draft_path"].endswith(".html")

    @pytest.mark.asyncio
    async def test_repurpose_returns_draft_path(
        self,
        authenticated_repurposer: SubstackRepurposer,
    ) -> None:
        """repurpose() result contains 'draft_path' key."""
        with patch("builtins.open", mock_open()):
            result = await authenticated_repurposer.repurpose(SAMPLE_CONTENT)

        assert "draft_path" in result
        assert result["draft_path"] is not None

    @pytest.mark.asyncio
    async def test_repurpose_action_required_is_manual_publish(
        self,
        authenticated_repurposer: SubstackRepurposer,
    ) -> None:
        """repurpose() result always has action_required='manual_publish'."""
        with patch("builtins.open", mock_open()):
            result = await authenticated_repurposer.repurpose(SAMPLE_CONTENT)

        assert result["action_required"] == "manual_publish"

    @pytest.mark.asyncio
    async def test_repurpose_status_is_draft_saved(
        self,
        authenticated_repurposer: SubstackRepurposer,
    ) -> None:
        """repurpose() result status is 'draft_saved'."""
        with patch("builtins.open", mock_open()):
            result = await authenticated_repurposer.repurpose(SAMPLE_CONTENT)

        assert result["status"] == "draft_saved"

    @pytest.mark.asyncio
    async def test_repurpose_no_api_calls_made(
        self,
        authenticated_repurposer: SubstackRepurposer,
    ) -> None:
        """repurpose() must not call httpx or any external HTTP client."""
        with (
            patch("builtins.open", mock_open()),
            patch("httpx.AsyncClient") as mock_http,
        ):
            await authenticated_repurposer.repurpose(SAMPLE_CONTENT)

        mock_http.assert_not_called()

    @pytest.mark.asyncio
    async def test_repurpose_html_contains_heading(
        self,
        authenticated_repurposer: SubstackRepurposer,
    ) -> None:
        """HTML output contains the article heading as <h1>."""
        written_content: list[str] = []

        def capture_write(data: str) -> None:
            written_content.append(data)

        m = mock_open()
        m.return_value.__enter__.return_value.write = capture_write

        with patch("builtins.open", m):
            await authenticated_repurposer.repurpose(SAMPLE_CONTENT)

        combined = "".join(written_content)
        assert "<h1>" in combined or "Test Article" in combined


class TestTransformToSubstackHtml:
    def test_transform_produces_valid_html(self, repurposer: SubstackRepurposer) -> None:
        """_transform_to_substack_html() returns a complete HTML document."""
        html = repurposer._transform_to_substack_html("# Hello\n\nParagraph.")
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "<body>" in html
        assert "</body>" in html

    def test_transform_h1_from_heading(self, repurposer: SubstackRepurposer) -> None:
        """# heading maps to <h1>."""
        html = repurposer._transform_to_substack_html("# My Title\n\nBody.")
        assert "<h1>My Title</h1>" in html

    def test_transform_h2_from_subheading(self, repurposer: SubstackRepurposer) -> None:
        """## heading maps to <h2>."""
        html = repurposer._transform_to_substack_html("## Section\n\nContent.")
        assert "<h2>Section</h2>" in html
