"""
Text-to-Speech Module using Edge-TTS (FREE!) or Fish Audio (Premium)

Edge-TTS uses Microsoft Edge's online TTS service.
No API key required - completely free with 300+ neural voices.

Fish Audio provides higher quality TTS with premium voices.
Requires FISH_AUDIO_API_KEY environment variable.

Both providers support professional voice enhancement for broadcast-quality audio.

Usage:
    tts = TextToSpeech()
    await tts.generate("Hello world!", "output.mp3")

    # With custom voice
    await tts.generate("Hello!", "output.mp3", voice="en-US-JennyNeural")

    # Generate with subtitles
    await tts.generate_with_subtitles("Hello!", "output.mp3", "subtitles.vtt")

    # Generate with professional enhancement (broadcast quality)
    await tts.generate_enhanced("Hello!", "output.mp3", enhance=True)

    # Use Fish Audio provider
    from src.content.tts import get_tts_provider
    tts = get_tts_provider("fish")
    await tts.generate("Hello!", "output.mp3")
"""

import asyncio
import os
import re
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

    # Pause durations for SSML (in milliseconds)
    PAUSE_DURATIONS = {
        "short": 300,   # After comma
        "medium": 500,  # After period, semicolon
        "long": 800,    # After paragraph breaks, ellipsis
        "dramatic": 1200  # For dramatic effect
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

    def add_dramatic_pauses(self, text: str) -> str:
        """
        Add SSML pause tags at punctuation for dramatic storytelling effect.

        Inserts pauses after:
        - Periods: medium pause (500ms)
        - Commas: short pause (300ms)
        - Ellipsis (...): long pause (800ms)
        - Question marks: medium pause (500ms)
        - Exclamation marks: medium pause (500ms)
        - Paragraph breaks: long pause (800ms)

        Args:
            text: Plain text to add pauses to

        Returns:
            Text with SSML break tags inserted
        """
        # Replace ellipsis with long pause
        text = re.sub(
            r'\.{3}',
            f'<break time="{self.PAUSE_DURATIONS["long"]}ms"/>',
            text
        )

        # Replace double newlines (paragraph breaks) with long pause
        text = re.sub(
            r'\n\s*\n',
            f' <break time="{self.PAUSE_DURATIONS["long"]}ms"/> ',
            text
        )

        # Replace period followed by space with medium pause
        text = re.sub(
            r'\.\s+',
            f'. <break time="{self.PAUSE_DURATIONS["medium"]}ms"/> ',
            text
        )

        # Replace question mark followed by space with medium pause
        text = re.sub(
            r'\?\s+',
            f'? <break time="{self.PAUSE_DURATIONS["medium"]}ms"/> ',
            text
        )

        # Replace exclamation mark followed by space with medium pause
        text = re.sub(
            r'!\s+',
            f'! <break time="{self.PAUSE_DURATIONS["medium"]}ms"/> ',
            text
        )

        # Replace semicolon/colon followed by space with medium pause
        text = re.sub(
            r'[;:]\s+',
            f'; <break time="{self.PAUSE_DURATIONS["medium"]}ms"/> ',
            text
        )

        # Replace comma followed by space with short pause
        text = re.sub(
            r',\s+',
            f', <break time="{self.PAUSE_DURATIONS["short"]}ms"/> ',
            text
        )

        logger.debug("Added dramatic pauses to text with SSML breaks")
        return text

    def wrap_ssml(self, text: str, voice: str) -> str:
        """
        Wrap text in SSML speak tags for Edge-TTS.

        Args:
            text: Text (may contain SSML break tags)
            voice: Voice name for the SSML

        Returns:
            Complete SSML document
        """
        # Edge-TTS uses a simplified SSML format
        ssml = f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
    <voice name="{voice}">
        {text}
    </voice>
</speak>"""
        return ssml

    async def generate_with_ssml(
        self,
        text: str,
        output_file: str,
        voice: Optional[str] = None,
        rate: str = "+0%",
        pitch: str = "+0Hz",
        volume: str = "+0%",
        add_pauses: bool = False
    ) -> str:
        """
        Generate speech from text with SSML support.

        Args:
            text: Text to convert (can contain SSML tags or be plain text)
            output_file: Output audio file path (mp3)
            voice: Voice to use (default: self.default_voice)
            rate: Speech rate adjustment (e.g., "+20%", "-10%")
            pitch: Pitch adjustment (e.g., "+5Hz", "-5Hz")
            volume: Volume adjustment (e.g., "+10%", "-10%")
            add_pauses: If True, automatically add dramatic pauses at punctuation

        Returns:
            Path to the generated audio file
        """
        voice = voice or self.default_voice

        # Add dramatic pauses if requested
        if add_pauses:
            text = self.add_dramatic_pauses(text)
            logger.info("Added dramatic pauses for storytelling effect")

        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating SSML speech: {len(text)} chars -> {output_file}")
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

            logger.success(f"SSML audio saved: {output_file}")
            return str(output_path)

        except (ConnectionError, TimeoutError, OSError, IOError) as e:
            logger.error(f"SSML TTS generation failed: {e}")
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

    async def generate_enhanced(
        self,
        text: str,
        output_file: str,
        voice: Optional[str] = None,
        rate: str = "+0%",
        pitch: str = "+0Hz",
        volume: str = "+0%",
        enhance: bool = True,
        noise_reduction: bool = True,
        normalize_lufs: float = -14.0
    ) -> str:
        """
        Generate speech with professional broadcast-quality enhancement.

        This method generates speech and then applies professional audio processing:
        - FFT-based noise reduction
        - Voice presence EQ boost (2-4kHz)
        - De-essing for sibilant reduction
        - Dynamic compression
        - Loudness normalization to YouTube's -14 LUFS target

        Args:
            text: Text to convert to speech
            output_file: Output audio file path (mp3)
            voice: Voice to use (default: self.default_voice)
            rate: Speech rate adjustment (e.g., "+20%", "-10%")
            pitch: Pitch adjustment (e.g., "+5Hz", "-5Hz")
            volume: Volume adjustment (e.g., "+10%", "-10%")
            enhance: Apply professional voice enhancement (default: True)
            noise_reduction: Apply FFT noise reduction (default: True)
            normalize_lufs: Target loudness in LUFS (default: -14)

        Returns:
            Path to the generated and enhanced audio file
        """
        voice = voice or self.default_voice

        # Generate raw TTS first
        if enhance:
            # Generate to temp file, then enhance
            output_path = Path(output_file)
            temp_file = str(output_path.parent / f"_raw_{output_path.name}")
            raw_audio = await self.generate(text, temp_file, voice, rate, pitch, volume)

            try:
                # Import audio processor and enhance
                try:
                    from src.content.audio_processor import AudioProcessor
                except ImportError:
                    from .audio_processor import AudioProcessor

                processor = AudioProcessor()
                enhanced = processor.enhance_voice_professional(
                    input_file=raw_audio,
                    output_file=output_file,
                    noise_reduction=noise_reduction,
                    normalize_lufs=normalize_lufs
                )

                if enhanced:
                    # Cleanup temp file
                    try:
                        os.remove(temp_file)
                    except OSError:
                        pass
                    logger.success(f"Enhanced audio saved: {output_file}")
                    return enhanced
                else:
                    # Enhancement failed, keep raw audio
                    logger.warning("Enhancement failed, using raw TTS output")
                    os.rename(temp_file, output_file)
                    return output_file

            except ImportError as e:
                logger.warning(f"AudioProcessor not available: {e}. Using raw TTS output.")
                os.rename(temp_file, output_file)
                return output_file
        else:
            # No enhancement, just generate normally
            return await self.generate(text, output_file, voice, rate, pitch, volume)

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


def get_tts_provider(provider: str = "edge", **kwargs):
    """
    Factory function to get a TTS provider.

    Args:
        provider: TTS provider to use ("edge" or "fish")
        **kwargs: Additional arguments passed to the provider

    Returns:
        TTS provider instance (TextToSpeech or FishAudioTTS)

    Example:
        # Use Edge-TTS (default, free)
        tts = get_tts_provider("edge")

        # Use Fish Audio (premium quality)
        tts = get_tts_provider("fish")
        tts = get_tts_provider("fish", api_key="your_api_key")
    """
    provider = provider.lower().strip()

    if provider in ("edge", "edge-tts", "edgetts"):
        default_voice = kwargs.get("default_voice", "en-US-GuyNeural")
        logger.info(f"Using Edge-TTS provider with voice: {default_voice}")
        return TextToSpeech(default_voice=default_voice)

    elif provider in ("fish", "fish-audio", "fishaudio"):
        try:
            from src.content.tts_fish import FishAudioTTS
        except ImportError:
            # Handle relative import when running as module
            from .tts_fish import FishAudioTTS

        api_key = kwargs.get("api_key")
        logger.info("Using Fish Audio TTS provider")
        return FishAudioTTS(api_key=api_key)

    else:
        logger.warning(f"Unknown TTS provider '{provider}', falling back to Edge-TTS")
        default_voice = kwargs.get("default_voice", "en-US-GuyNeural")
        return TextToSpeech(default_voice=default_voice)


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
