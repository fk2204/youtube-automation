"""
Text-to-Speech Module using Edge-TTS (FREE!)

Edge-TTS uses Microsoft Edge's online TTS service.
No API key required - completely free with 300+ neural voices.

Usage:
    tts = TextToSpeech()
    await tts.generate("Hello world!", "output.mp3")

    # With custom voice
    await tts.generate("Hello!", "output.mp3", voice="en-US-JennyNeural")

    # Generate with subtitles
    await tts.generate_with_subtitles("Hello!", "output.mp3", "subtitles.vtt")
"""

import asyncio
import os
from pathlib import Path
from typing import Optional, List, Dict
from loguru import logger

try:
    import edge_tts
except ImportError:
    raise ImportError("Please install edge-tts: pip install edge-tts")


class TextToSpeech:
    """Text-to-Speech generator using Microsoft Edge TTS (FREE)"""

    # Popular voices for YouTube content
    RECOMMENDED_VOICES = {
        # US English
        "en-US-GuyNeural": "US Male - Professional, clear",
        "en-US-JennyNeural": "US Female - Friendly, warm",
        "en-US-AriaNeural": "US Female - Professional",
        "en-US-DavisNeural": "US Male - Casual, friendly",

        # UK English
        "en-GB-SoniaNeural": "UK Female - Professional",
        "en-GB-RyanNeural": "UK Male - Professional",

        # Australian English
        "en-AU-WilliamNeural": "AU Male - Friendly",
        "en-AU-NatashaNeural": "AU Female - Professional",
    }

    def __init__(self, default_voice: str = "en-US-GuyNeural"):
        """
        Initialize TTS with a default voice.

        Args:
            default_voice: Default voice to use (e.g., "en-US-GuyNeural")
        """
        self.default_voice = default_voice
        logger.info(f"TTS initialized with voice: {default_voice}")

    async def generate(
        self,
        text: str,
        output_file: str,
        voice: Optional[str] = None,
        rate: str = "+0%",
        pitch: str = "+0Hz",
        volume: str = "+0%"
    ) -> str:
        """
        Generate speech from text and save to file.

        Args:
            text: Text to convert to speech
            output_file: Output audio file path (mp3)
            voice: Voice to use (default: self.default_voice)
            rate: Speech rate adjustment (e.g., "+20%", "-10%")
            pitch: Pitch adjustment (e.g., "+5Hz", "-5Hz")
            volume: Volume adjustment (e.g., "+10%", "-10%")

        Returns:
            Path to the generated audio file
        """
        voice = voice or self.default_voice

        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating speech: {len(text)} chars -> {output_file}")
        logger.debug(f"Voice: {voice}, Rate: {rate}, Pitch: {pitch}")

        try:
            communicate = edge_tts.Communicate(
                text=text,
                voice=voice,
                rate=rate,
                pitch=pitch,
                volume=volume
            )
            await communicate.save(str(output_path))

            logger.success(f"Audio saved: {output_file}")
            return str(output_path)

        except (ConnectionError, TimeoutError, OSError, IOError) as e:
            logger.error(f"TTS generation failed: {e}")
            raise

    async def generate_with_subtitles(
        self,
        text: str,
        audio_file: str,
        subtitle_file: str,
        voice: Optional[str] = None,
        subtitle_format: str = "vtt"
    ) -> Dict[str, str]:
        """
        Generate speech with synchronized subtitles.

        Args:
            text: Text to convert to speech
            audio_file: Output audio file path
            subtitle_file: Output subtitle file path
            voice: Voice to use
            subtitle_format: "vtt" or "srt"

        Returns:
            Dict with paths to audio and subtitle files
        """
        voice = voice or self.default_voice

        # Ensure directories exist
        Path(audio_file).parent.mkdir(parents=True, exist_ok=True)
        Path(subtitle_file).parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating speech with subtitles...")

        try:
            communicate = edge_tts.Communicate(text=text, voice=voice)
            submaker = edge_tts.SubMaker()

            # Stream and collect subtitle data
            with open(audio_file, "wb") as audio_fp:
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_fp.write(chunk["data"])
                    elif chunk["type"] == "WordBoundary":
                        submaker.create_sub(
                            (chunk["offset"], chunk["duration"]),
                            chunk["text"]
                        )

            # Save subtitles
            subtitle_content = submaker.generate_subs()

            # Convert to SRT if needed
            if subtitle_format.lower() == "srt":
                subtitle_content = self._vtt_to_srt(subtitle_content)

            with open(subtitle_file, "w", encoding="utf-8") as sub_fp:
                sub_fp.write(subtitle_content)

            logger.success(f"Audio: {audio_file}, Subtitles: {subtitle_file}")

            return {
                "audio": audio_file,
                "subtitles": subtitle_file
            }

        except (ConnectionError, TimeoutError, OSError, IOError) as e:
            logger.error(f"TTS with subtitles failed: {e}")
            raise

    def _vtt_to_srt(self, vtt_content: str) -> str:
        """Convert VTT subtitle format to SRT format."""
        lines = vtt_content.split('\n')
        srt_lines = []
        counter = 1

        for i, line in enumerate(lines):
            # Skip VTT header
            if line.startswith('WEBVTT') or line.startswith('Kind:') or line.startswith('Language:'):
                continue
            # Convert timestamp format
            if '-->' in line:
                srt_lines.append(str(counter))
                counter += 1
                # VTT uses . for milliseconds, SRT uses ,
                line = line.replace('.', ',')
            srt_lines.append(line)

        return '\n'.join(srt_lines)

    @staticmethod
    async def list_voices(locale: Optional[str] = None) -> List[Dict]:
        """
        List all available voices.

        Args:
            locale: Filter by locale (e.g., "en-US", "en-GB")

        Returns:
            List of voice dictionaries with ShortName, Locale, Gender, etc.
        """
        voices = await edge_tts.list_voices()

        if locale:
            voices = [v for v in voices if v["Locale"].startswith(locale)]

        return voices

    @staticmethod
    async def print_voices(locale: str = "en"):
        """Print available voices for a locale (for easy reference)."""
        voices = await edge_tts.list_voices()

        print(f"\n{'='*60}")
        print(f"Available {locale} voices:")
        print(f"{'='*60}\n")

        for voice in voices:
            if voice["Locale"].startswith(locale):
                print(f"  {voice['ShortName']:30} | {voice['Gender']:6} | {voice['Locale']}")

        print(f"\n{'='*60}\n")


# Convenience function for quick generation
async def generate_speech(
    text: str,
    output_file: str,
    voice: str = "en-US-GuyNeural"
) -> str:
    """Quick function to generate speech."""
    tts = TextToSpeech(default_voice=voice)
    return await tts.generate(text, output_file)


# Example usage and testing
if __name__ == "__main__":
    async def main():
        # List available English voices
        await TextToSpeech.print_voices("en")

        # Generate sample audio
        tts = TextToSpeech()

        sample_text = """
        Welcome to this tutorial! Today we're going to learn
        how to automate YouTube video creation using Python.

        This is completely automated using AI and text-to-speech.
        Let's get started!
        """

        # Generate audio only
        await tts.generate(
            text=sample_text,
            output_file="output/test_audio.mp3",
            voice="en-US-GuyNeural",
            rate="+5%"  # Slightly faster
        )

        # Generate with subtitles
        await tts.generate_with_subtitles(
            text=sample_text,
            audio_file="output/test_with_subs.mp3",
            subtitle_file="output/test_subtitles.vtt"
        )

        print("Done! Check the output folder.")

    asyncio.run(main())
