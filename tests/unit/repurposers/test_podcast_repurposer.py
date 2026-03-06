"""
Unit tests for PodcastRepurposer.

All external API calls and file I/O are mocked.
No real credentials required, no actual audio files written.
Tests verify TTS provider auth, audio generation, RSS entry generation,
and intro/outro toggle behavior.
"""

import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from distribution.repurposers.repurposer_registry import RepurposerRegistry
import distribution.repurposers.podcast_repurposer as podcast_module
from distribution.repurposers.podcast_repurposer import PodcastRepurposer

ELEVENLABS_CONFIG = {"tts_provider": "elevenlabs", "podcast_title": "Test Podcast"}
OPENAI_TTS_CONFIG = {"tts_provider": "openai_tts", "podcast_title": "Test Podcast"}
GOOGLE_TTS_CONFIG = {"tts_provider": "google_tts", "podcast_title": "Test Podcast"}

SAMPLE_CONTENT = (
    "# Episode Title\n\n"
    "Welcome to this episode. Today we cover important topics. "
    "The content is well-structured and informative. "
    "We hope you enjoy the listen."
)
SAMPLE_AUDIO_PATH = "/tmp/test_podcast_episode.mp3"


@pytest.fixture(autouse=True)
def clear_simulation_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SOCIAL_SIMULATION_MODE", raising=False)



class TestAuthenticateElevenLabs:
    @pytest.mark.asyncio
    async def test_authenticate_elevenlabs_success(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """authenticate() returns True when ElevenLabs API key is valid."""
        monkeypatch.setenv("ELEVENLABS_API_KEY", "el_test_key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"subscription": {"tier": "free"}}

        with patch("distribution.repurposers.podcast_repurposer.httpx.AsyncClient") as mock_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_class.return_value = mock_client

            rp = PodcastRepurposer(ELEVENLABS_CONFIG)
            result = await rp.authenticate()

        assert result is True
        assert rp._authenticated is True

    @pytest.mark.asyncio
    async def test_authenticate_elevenlabs_returns_false_on_missing_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """authenticate() returns False when ELEVENLABS_API_KEY is not set."""
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        rp = PodcastRepurposer(ELEVENLABS_CONFIG)
        result = await rp.authenticate()
        assert result is False


class TestAuthenticateOpenAiTts:
    @pytest.mark.asyncio
    async def test_authenticate_openai_tts_success(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """authenticate() returns True when OpenAI API key is valid."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test_openai_key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"data": [{"id": "tts-1"}]}

        with patch("distribution.repurposers.podcast_repurposer.httpx.AsyncClient") as mock_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_class.return_value = mock_client

            rp = PodcastRepurposer(OPENAI_TTS_CONFIG)
            result = await rp.authenticate()

        assert result is True

    @pytest.mark.asyncio
    async def test_authenticate_openai_tts_returns_false_on_missing_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """authenticate() returns False when OPENAI_API_KEY is not set."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        rp = PodcastRepurposer(OPENAI_TTS_CONFIG)
        result = await rp.authenticate()
        assert result is False


class TestAuthenticateGoogleTts:
    @pytest.mark.asyncio
    async def test_authenticate_google_tts_success(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """authenticate() returns True when Google TTS API key is valid."""
        monkeypatch.setenv("GOOGLE_TTS_API_KEY", "google_test_key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"voices": []}

        with patch("distribution.repurposers.podcast_repurposer.httpx.AsyncClient") as mock_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_class.return_value = mock_client

            rp = PodcastRepurposer(GOOGLE_TTS_CONFIG)
            result = await rp.authenticate()

        assert result is True


class TestRepurpose:
    def _make_authenticated_repurposer(self, config: dict) -> PodcastRepurposer:
        rp = PodcastRepurposer(config)
        rp._tts_api_key = "test_key"
        rp._authenticated = True
        return rp

    @pytest.mark.asyncio
    async def test_repurpose_returns_audio_and_rss_paths(self) -> None:
        """repurpose() returns audio_path and rss_entry_path on success."""
        rp = self._make_authenticated_repurposer(ELEVENLABS_CONFIG)

        fake_audio = b"fake_mp3_bytes"

        with (
            patch.object(rp, "_call_tts", new_callable=AsyncMock, return_value=fake_audio),
            patch("pathlib.Path.mkdir"),
            patch("builtins.open", mock_open()),
        ):
            result = await rp.repurpose(
                SAMPLE_CONTENT,
                output_audio_path=SAMPLE_AUDIO_PATH,
            )

        assert result["success"] is True
        assert result["audio_path"] == SAMPLE_AUDIO_PATH
        assert result["rss_entry_path"] is not None
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_repurpose_duration_seconds_is_positive_int(self) -> None:
        """repurpose() returns a positive integer for duration_seconds."""
        rp = self._make_authenticated_repurposer(ELEVENLABS_CONFIG)
        fake_audio = b"fake_mp3_bytes"

        with (
            patch.object(rp, "_call_tts", new_callable=AsyncMock, return_value=fake_audio),
            patch("pathlib.Path.mkdir"),
            patch("builtins.open", mock_open()),
        ):
            result = await rp.repurpose(
                SAMPLE_CONTENT,
                output_audio_path=SAMPLE_AUDIO_PATH,
            )

        assert isinstance(result["duration_seconds"], int)
        assert result["duration_seconds"] > 0

    @pytest.mark.asyncio
    async def test_repurpose_not_authenticated_returns_error(self) -> None:
        """repurpose() returns error dict when not authenticated."""
        rp = PodcastRepurposer(ELEVENLABS_CONFIG)
        result = await rp.repurpose(SAMPLE_CONTENT, output_audio_path=SAMPLE_AUDIO_PATH)
        assert result["success"] is False
        assert result["error"] is not None

    @pytest.mark.asyncio
    async def test_repurpose_missing_output_path_returns_error(self) -> None:
        """repurpose() returns error when output_audio_path is empty."""
        rp = self._make_authenticated_repurposer(ELEVENLABS_CONFIG)
        result = await rp.repurpose(SAMPLE_CONTENT, output_audio_path="")
        assert result["success"] is False
        assert "output_audio_path" in result["error"]


class TestGenerateScript:
    def test_intro_outro_added_when_enabled(self) -> None:
        """_generate_script() includes intro and outro when add_intro_outro=True."""
        rp = PodcastRepurposer(ELEVENLABS_CONFIG)
        script = rp._generate_script("Main content here.", add_intro_outro=True)
        assert "Welcome" in script
        assert "Thanks for listening" in script

    def test_intro_outro_omitted_when_disabled(self) -> None:
        """_generate_script() omits intro/outro when add_intro_outro=False."""
        rp = PodcastRepurposer(ELEVENLABS_CONFIG)
        script = rp._generate_script("Main content here.", add_intro_outro=False)
        assert "Welcome" not in script
        assert "Thanks for listening" not in script

    def test_markdown_headings_stripped(self) -> None:
        """_generate_script() removes # heading markers from script."""
        rp = PodcastRepurposer(ELEVENLABS_CONFIG)
        script = rp._generate_script("# My Title\nBody text.", add_intro_outro=False)
        assert "#" not in script
        assert "My Title" in script


class TestGenerateRssEntry:
    def test_rss_entry_is_valid_xml_item(self) -> None:
        """_generate_rss_entry() returns a string containing <item> tags."""
        rp = PodcastRepurposer(ELEVENLABS_CONFIG)
        rss = rp._generate_rss_entry(
            title="Episode 1",
            audio_path="/tmp/ep1.mp3",
            duration_seconds=120,
            description="Test description.",
        )
        assert "<item>" in rss
        assert "</item>" in rss

    def test_rss_entry_contains_title(self) -> None:
        """_generate_rss_entry() includes the episode title."""
        rp = PodcastRepurposer(ELEVENLABS_CONFIG)
        rss = rp._generate_rss_entry(
            title="My Episode Title",
            audio_path="/tmp/ep.mp3",
            duration_seconds=60,
            description="Desc.",
        )
        assert "My Episode Title" in rss

    def test_rss_entry_contains_enclosure(self) -> None:
        """_generate_rss_entry() includes an <enclosure> element for the audio."""
        rp = PodcastRepurposer(ELEVENLABS_CONFIG)
        rss = rp._generate_rss_entry(
            title="Ep",
            audio_path="/tmp/ep.mp3",
            duration_seconds=90,
            description="Desc.",
        )
        assert "<enclosure" in rss
        assert "audio/mpeg" in rss
