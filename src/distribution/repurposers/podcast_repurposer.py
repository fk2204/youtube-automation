"""
PodcastRepurposer — generates audio and RSS entries from written content.

AUTOMATION_LEVEL: content_only

Apple Podcasts and Spotify do not provide write APIs for automated publishing.
This repurposer does NOT submit to any podcast directory. Instead it:
  1. Generates a podcast script from source content (_generate_script).
  2. Calls a TTS provider to synthesize audio (_call_tts).
  3. Generates an RSS entry XML file (_generate_rss_entry).
  4. Returns paths to both files so the operator can host them and
     submit the RSS feed to directories once manually.

Supported TTS providers:
  - elevenlabs:  ElevenLabs Speech API (high quality, paid)
  - openai_tts:  OpenAI TTS API (gpt-4o-mini-tts or tts-1)
  - google_tts:  Google Cloud Text-to-Speech API

Decision from Wave 2 Blueprint: podcast = content_only, no directory submission.

Operator workflow after repurpose():
  1. Host audio_path on a public CDN/server.
  2. Add rss_entry_path content to your RSS feed XML.
  3. Submit RSS feed URL to Apple Podcasts / Spotify once (they auto-poll after).

Credentials (env vars):
  ElevenLabs:  ELEVENLABS_API_KEY
  OpenAI TTS:  OPENAI_API_KEY
  Google TTS:  GOOGLE_TTS_API_KEY (or GOOGLE_APPLICATION_CREDENTIALS path)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from email.utils import formatdate
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

from distribution.repurposers.base_repurposer import BaseRepurposer
from distribution.repurposers.repurposer_registry import RepurposerRegistry

ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1"
OPENAI_API_BASE = "https://api.openai.com/v1"
GOOGLE_TTS_API_BASE = "https://texttospeech.googleapis.com/v1"

SUPPORTED_TTS_PROVIDERS = {"elevenlabs", "openai_tts", "google_tts"}

# Default ElevenLabs voice ID (Rachel — neutral, clear narration)
DEFAULT_ELEVENLABS_VOICE = "21m00Tcm4TlvDq8ikWAM"

# Default OpenAI voice
DEFAULT_OPENAI_VOICE = "alloy"


class PodcastRepurposer(BaseRepurposer):
    """
    Generates podcast audio and RSS entries from written content.

    Does NOT submit to Apple Podcasts or Spotify — see module docstring.

    Config keys:
        tts_provider:  "elevenlabs" | "openai_tts" | "google_tts"
        podcast_title: str — show title for RSS metadata
        podcast_description: str — show description for RSS metadata
        podcast_author: str — author name
        podcast_email:  str — author email for iTunes metadata
    """

    PLATFORM_NAME = "podcast"
    AUTOMATION_LEVEL = "content_only"

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize with config dict.

        Config keys:
            tts_provider: str — one of SUPPORTED_TTS_PROVIDERS
        """
        super().__init__(config)
        self._tts_provider: str = str(config.get("tts_provider", "elevenlabs")).lower()
        self._tts_api_key: Optional[str] = None

    @property
    def platform_name(self) -> str:
        return self.PLATFORM_NAME

    async def authenticate(self) -> bool:
        """
        Validate TTS provider credentials from env vars.

        Returns:
            True on success, False on failure (never raises).
        """
        try:
            if self._tts_provider == "elevenlabs":
                return await self._authenticate_elevenlabs()
            if self._tts_provider == "openai_tts":
                return await self._authenticate_openai_tts()
            if self._tts_provider == "google_tts":
                return await self._authenticate_google_tts()
        except Exception as exc:
            logger.error(
                "PodcastRepurposer ({p}) authenticate error: {exc}",
                p=self._tts_provider,
                exc=exc,
            )
            return False
        logger.error(
            "PodcastRepurposer: unsupported tts_provider '{p}'.",
            p=self._tts_provider,
        )
        return False

    async def repurpose(
        self,
        source_content: str,
        output_audio_path: str = "",
        voice_id: Optional[str] = None,
        add_intro_outro: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate podcast audio and RSS entry from source content.

        Args:
            source_content:   Article body to convert to audio.
            output_audio_path: Absolute path where the audio file will be saved.
            voice_id:         Provider-specific voice ID. Uses provider default if None.
            add_intro_outro:  If True, wraps script with intro/outro sentences.

        Returns:
            {
                "success": bool,
                "audio_path": str | None,
                "rss_entry_path": str | None,
                "duration_seconds": int,
                "error": str | None,
            }
        """
        if self.simulation_mode:
            sim = self._simulate_repurpose({
                "tts_provider": self._tts_provider,
                "add_intro_outro": add_intro_outro,
            })
            sim.update({
                "audio_path": "/tmp/podcast_sim.mp3",
                "rss_entry_path": "/tmp/podcast_sim_rss.xml",
                "duration_seconds": 60,
            })
            return sim

        if not self._authenticated:
            return self._build_error_response(
                "Not authenticated. Call authenticate() first.",
                extra={"audio_path": None, "rss_entry_path": None, "duration_seconds": 0},
            )

        if not output_audio_path:
            return self._build_error_response(
                "output_audio_path is required.",
                extra={"audio_path": None, "rss_entry_path": None, "duration_seconds": 0},
            )

        script = self._generate_script(source_content, add_intro_outro=add_intro_outro)

        try:
            audio_bytes = await self._call_tts(script, voice_id=voice_id)
        except Exception as exc:
            logger.error("PodcastRepurposer TTS call failed: {exc}", exc=exc)
            return self._build_error_response(
                str(exc),
                extra={"audio_path": None, "rss_entry_path": None, "duration_seconds": 0},
            )

        try:
            Path(output_audio_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_audio_path, "wb") as f:
                f.write(audio_bytes)
        except Exception as exc:
            logger.error("PodcastRepurposer: failed to write audio: {exc}", exc=exc)
            return self._build_error_response(
                str(exc),
                extra={"audio_path": None, "rss_entry_path": None, "duration_seconds": 0},
            )

        # Estimate duration: ~150 words per minute for TTS
        word_count = len(script.split())
        duration_seconds = max(1, int((word_count / 150) * 60))

        rss_path = output_audio_path.replace(".mp3", "_rss_entry.xml")
        rss_xml = self._generate_rss_entry(
            title=self._extract_title(source_content),
            audio_path=output_audio_path,
            duration_seconds=duration_seconds,
            description=source_content[:300],
        )

        try:
            with open(rss_path, "w", encoding="utf-8") as f:
                f.write(rss_xml)
        except Exception as exc:
            logger.error("PodcastRepurposer: failed to write RSS entry: {exc}", exc=exc)
            return self._build_error_response(
                str(exc),
                extra={
                    "audio_path": output_audio_path,
                    "rss_entry_path": None,
                    "duration_seconds": duration_seconds,
                },
            )

        logger.info(
            "PodcastRepurposer: audio={a}, rss={r}, duration={d}s",
            a=output_audio_path,
            r=rss_path,
            d=duration_seconds,
        )
        return {
            "success": True,
            "audio_path": output_audio_path,
            "rss_entry_path": rss_path,
            "duration_seconds": duration_seconds,
            "error": None,
        }

    def _generate_script(self, content: str, add_intro_outro: bool) -> str:
        """
        Generate a podcast narration script from source content.

        Cleans markdown syntax and optionally adds intro/outro sentences.
        """
        lines: List[str] = []
        for line in content.strip().splitlines():
            stripped = line.strip()
            # Remove markdown heading markers.
            if stripped.startswith("#"):
                stripped = stripped.lstrip("#").strip()
            if stripped:
                lines.append(stripped)

        body = " ".join(lines)

        if not add_intro_outro:
            return body

        podcast_title = self._config.get("podcast_title", "the show")
        intro = f"Welcome to {podcast_title}. Today's episode: "
        outro = " That's all for today. Thanks for listening."
        return intro + body + outro

    async def _call_tts(
        self,
        script: str,
        voice_id: Optional[str] = None,
    ) -> bytes:
        """
        Call the configured TTS provider and return audio bytes.

        Args:
            script:   Full narration script text.
            voice_id: Provider-specific voice identifier.

        Returns:
            Raw audio bytes (MP3 format).

        Raises:
            Exception: on API failure (caller handles and returns error response).
        """
        if self._tts_provider == "elevenlabs":
            return await self._tts_elevenlabs(script, voice_id)
        if self._tts_provider == "openai_tts":
            return await self._tts_openai(script, voice_id)
        if self._tts_provider == "google_tts":
            return await self._tts_google(script, voice_id)
        raise ValueError(f"Unsupported TTS provider: {self._tts_provider}")

    def _generate_rss_entry(
        self,
        title: str,
        audio_path: str,
        duration_seconds: int,
        description: str,
    ) -> str:
        """
        Generate an RSS <item> XML snippet for the episode.

        The caller is responsible for embedding this into their full RSS feed.
        Does NOT generate a complete RSS feed — only the <item> element.

        Format follows Apple Podcasts RSS requirements:
        https://podcasters.apple.com/support/823-podcast-requirements
        """
        pub_date = formatdate(usegmt=True)
        audio_filename = Path(audio_path).name
        audio_url = self._config.get("audio_base_url", "https://example.com/podcast/audio")
        full_audio_url = f"{audio_url.rstrip('/')}/{audio_filename}"

        minutes = duration_seconds // 60
        seconds = duration_seconds % 60
        duration_str = f"{minutes:02d}:{seconds:02d}"

        safe_title = self._xml_escape(title)
        safe_description = self._xml_escape(description)

        return (
            f"<item>\n"
            f"  <title>{safe_title}</title>\n"
            f"  <description>{safe_description}</description>\n"
            f"  <pubDate>{pub_date}</pubDate>\n"
            f"  <enclosure url=\"{full_audio_url}\" type=\"audio/mpeg\" length=\"0\"/>\n"
            f"  <itunes:duration>{duration_str}</itunes:duration>\n"
            f"  <itunes:explicit>no</itunes:explicit>\n"
            f"  <guid isPermaLink=\"false\">{full_audio_url}</guid>\n"
            f"</item>"
        )

    async def _tts_elevenlabs(self, script: str, voice_id: Optional[str]) -> bytes:
        """Call ElevenLabs Text-to-Speech API and return MP3 bytes."""
        vid = voice_id or DEFAULT_ELEVENLABS_VOICE
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ELEVENLABS_API_BASE}/text-to-speech/{vid}",
                headers={
                    "xi-api-key": self._tts_api_key or "",
                    "Content-Type": "application/json",
                    "Accept": "audio/mpeg",
                },
                json={
                    "text": script,
                    "model_id": "eleven_monolingual_v1",
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
                },
                timeout=120.0,
            )
        response.raise_for_status()
        return response.content

    async def _tts_openai(self, script: str, voice_id: Optional[str]) -> bytes:
        """Call OpenAI TTS API and return MP3 bytes."""
        voice = voice_id or DEFAULT_OPENAI_VOICE
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OPENAI_API_BASE}/audio/speech",
                headers={
                    "Authorization": f"Bearer {self._tts_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "tts-1",
                    "input": script,
                    "voice": voice,
                },
                timeout=120.0,
            )
        response.raise_for_status()
        return response.content

    async def _tts_google(self, script: str, voice_id: Optional[str]) -> bytes:
        """Call Google Cloud Text-to-Speech API and return MP3 bytes."""
        voice_name = voice_id or "en-US-Standard-A"
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{GOOGLE_TTS_API_BASE}/text:synthesize",
                params={"key": self._tts_api_key},
                json={
                    "input": {"text": script},
                    "voice": {"languageCode": "en-US", "name": voice_name},
                    "audioConfig": {"audioEncoding": "MP3"},
                },
                timeout=120.0,
            )
        response.raise_for_status()
        import base64
        audio_content = response.json().get("audioContent", "")
        return base64.b64decode(audio_content)

    async def _authenticate_elevenlabs(self) -> bool:
        """Validate ElevenLabs API key via GET /v1/user."""
        if self.guard_unconfigured_env("ELEVENLABS_API_KEY"):
            return False
        key = os.environ["ELEVENLABS_API_KEY"]
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{ELEVENLABS_API_BASE}/user",
                    headers={"xi-api-key": key},
                    timeout=10.0,
                )
            if response.status_code in (401, 403):
                logger.error("ElevenLabs API key rejected ({s}).", s=response.status_code)
                return False
            response.raise_for_status()
        except Exception as exc:
            logger.error("ElevenLabs auth failed: {exc}", exc=exc)
            return False
        self._tts_api_key = key
        self._authenticated = True
        logger.info("ElevenLabs TTS authenticated.")
        return True

    async def _authenticate_openai_tts(self) -> bool:
        """Validate OpenAI API key via GET /v1/models."""
        if self.guard_unconfigured_env("OPENAI_API_KEY"):
            return False
        key = os.environ["OPENAI_API_KEY"]
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{OPENAI_API_BASE}/models",
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=10.0,
                )
            if response.status_code == 401:
                logger.error("OpenAI API key rejected (401).")
                return False
            response.raise_for_status()
        except Exception as exc:
            logger.error("OpenAI TTS auth failed: {exc}", exc=exc)
            return False
        self._tts_api_key = key
        self._authenticated = True
        logger.info("OpenAI TTS authenticated.")
        return True

    async def _authenticate_google_tts(self) -> bool:
        """Validate Google TTS API key via GET /v1/voices."""
        if self.guard_unconfigured_env("GOOGLE_TTS_API_KEY"):
            return False
        key = os.environ["GOOGLE_TTS_API_KEY"]
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{GOOGLE_TTS_API_BASE}/voices",
                    params={"key": key},
                    timeout=10.0,
                )
            if response.status_code in (401, 403):
                logger.error("Google TTS API key rejected ({s}).", s=response.status_code)
                return False
            response.raise_for_status()
        except Exception as exc:
            logger.error("Google TTS auth failed: {exc}", exc=exc)
            return False
        self._tts_api_key = key
        self._authenticated = True
        logger.info("Google TTS authenticated.")
        return True

    @staticmethod
    def _extract_title(content: str) -> str:
        """Extract episode title from first heading or first line."""
        for line in content.strip().splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip()
            if stripped:
                return stripped[:100]
        return "Untitled Episode"

    @staticmethod
    def _xml_escape(text: str) -> str:
        """Escape XML special characters."""
        return (
            text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )


def _register() -> None:
    """Register PodcastRepurposer with the global RepurposerRegistry."""
    RepurposerRegistry.register("podcast", PodcastRepurposer)
    logger.debug("PodcastRepurposer registered.")


_register()
