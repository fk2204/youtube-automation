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

import random


class NaturalVoiceVariation:
    """
    Adds micro-variations to TTS output to reduce AI detection and make speech sound more natural.

    Uses SSML prosody tags to vary rate and pitch at sentence and phrase levels,
    mimicking natural human speech patterns where pace and tone fluctuate slightly.

    Usage:
        variation = NaturalVoiceVariation()
        ssml_text = variation.apply_natural_variation("Hello world. How are you today?")
    """

    # Default variation ranges
    DEFAULT_RATE_VARIATION_PERCENT = 8  # +/- 8% rate variation
    DEFAULT_PITCH_VARIATION_HZ = 5       # +/- 5Hz pitch variation

    def __init__(
        self,
        rate_variation_percent: int = None,
        pitch_variation_hz: int = None,
        seed: int = None
    ):
        """
        Initialize voice variation settings.

        Args:
            rate_variation_percent: Maximum rate variation percentage (+/-). Default: 8%
            pitch_variation_hz: Maximum pitch variation in Hz (+/-). Default: 5Hz
            seed: Random seed for reproducible variations (optional)
        """
        self.rate_variation = rate_variation_percent or self.DEFAULT_RATE_VARIATION_PERCENT
        self.pitch_variation = pitch_variation_hz or self.DEFAULT_PITCH_VARIATION_HZ

        if seed is not None:
            random.seed(seed)

        logger.debug(f"NaturalVoiceVariation initialized: rate=+/-{self.rate_variation}%, pitch=+/-{self.pitch_variation}Hz")

    def _parse_base_value(self, base_value: str, unit: str) -> float:
        """
        Parse a base value string like '+0%' or '+5Hz' into a float.

        Args:
            base_value: String like '+0%', '-5Hz', '+10%'
            unit: Expected unit ('%' or 'Hz')

        Returns:
            Float value extracted from string
        """
        try:
            # Remove unit and parse
            cleaned = base_value.replace(unit, '').replace('+', '').strip()
            return float(cleaned)
        except (ValueError, AttributeError):
            return 0.0

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences for rate variation.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Split on sentence-ending punctuation followed by space or end
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _split_into_phrases(self, text: str) -> List[str]:
        """
        Split text into phrases for pitch variation.

        Phrases are separated by commas, semicolons, colons, and dashes.

        Args:
            text: Input text

        Returns:
            List of phrases
        """
        # Split on phrase-separating punctuation
        phrase_pattern = r'(?<=[,;:\-])\s*'
        phrases = re.split(phrase_pattern, text.strip())
        return [p.strip() for p in phrases if p.strip()]

    def add_rate_variation(self, text: str, base_rate: str = "+0%") -> str:
        """
        Add rate variation to text using SSML prosody tags.

        Varies the speech rate by +/- the configured percentage per sentence,
        creating natural-sounding pace fluctuations.

        Args:
            text: Plain text or text with existing SSML
            base_rate: Base rate adjustment (e.g., "+0%", "+10%", "-5%")

        Returns:
            SSML text with rate variations applied per sentence
        """
        base_rate_value = self._parse_base_value(base_rate, '%')
        sentences = self._split_into_sentences(text)

        if not sentences:
            return text

        varied_parts = []
        for sentence in sentences:
            # Generate random variation within range
            variation = random.uniform(-self.rate_variation, self.rate_variation)
            final_rate = base_rate_value + variation

            # Format the rate value
            rate_str = f"+{final_rate:.1f}%" if final_rate >= 0 else f"{final_rate:.1f}%"

            # Wrap sentence in prosody tag
            varied_sentence = f'<prosody rate="{rate_str}">{sentence}</prosody>'
            varied_parts.append(varied_sentence)

        result = ' '.join(varied_parts)
        logger.debug(f"Added rate variation to {len(sentences)} sentences")
        return result

    def add_pitch_variation(self, text: str, base_pitch: str = "+0Hz") -> str:
        """
        Add pitch variation to text using SSML prosody tags.

        Varies the pitch by +/- the configured Hz value per phrase,
        creating natural intonation patterns.

        Args:
            text: Plain text or text with existing SSML
            base_pitch: Base pitch adjustment (e.g., "+0Hz", "+5Hz", "-3Hz")

        Returns:
            SSML text with pitch variations applied per phrase
        """
        base_pitch_value = self._parse_base_value(base_pitch, 'Hz')
        phrases = self._split_into_phrases(text)

        if not phrases:
            return text

        varied_parts = []
        for phrase in phrases:
            # Generate random variation within range
            variation = random.uniform(-self.pitch_variation, self.pitch_variation)
            final_pitch = base_pitch_value + variation

            # Format the pitch value
            pitch_str = f"+{final_pitch:.1f}Hz" if final_pitch >= 0 else f"{final_pitch:.1f}Hz"

            # Wrap phrase in prosody tag
            varied_phrase = f'<prosody pitch="{pitch_str}">{phrase}</prosody>'
            varied_parts.append(varied_phrase)

        result = ' '.join(varied_parts)
        logger.debug(f"Added pitch variation to {len(phrases)} phrases")
        return result

    def apply_natural_variation(
        self,
        text: str,
        base_rate: str = "+0%",
        base_pitch: str = "+0Hz"
    ) -> str:
        """
        Apply both rate and pitch variations for maximum naturalness.

        This method combines sentence-level rate variation with phrase-level
        pitch variation to create speech that sounds more human and less
        robotic/AI-generated.

        Args:
            text: Plain text to process
            base_rate: Base rate adjustment (e.g., "+0%", "+10%")
            base_pitch: Base pitch adjustment (e.g., "+0Hz", "+5Hz")

        Returns:
            SSML text with both rate and pitch variations
        """
        # First apply rate variation at sentence level
        rate_varied = self.add_rate_variation(text, base_rate)

        # Then apply pitch variation at phrase level
        # This creates nested prosody which is valid SSML
        fully_varied = self.add_pitch_variation(rate_varied, base_pitch)

        logger.info("Applied natural voice variation (rate + pitch)")
        return fully_varied

    def add_natural_pauses(self, text: str) -> str:
        """
        Add subtle breath pauses after long sentences for more natural pacing.

        Args:
            text: Input text

        Returns:
            Text with breath pause markers added
        """
        sentences = self._split_into_sentences(text)
        result_parts = []

        for sentence in sentences:
            word_count = len(sentence.split())
            result_parts.append(sentence)

            # Add a short breath pause after sentences > 15 words
            if word_count > 15:
                # Vary the pause duration slightly
                pause_ms = random.randint(150, 300)
                result_parts.append(f'<break time="{pause_ms}ms"/>')

        return ' '.join(result_parts)


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

    async def generate_natural(
        self,
        text: str,
        output_file: str,
        voice: Optional[str] = None,
        rate: str = "+0%",
        pitch: str = "+0Hz",
        volume: str = "+0%",
        rate_variation_percent: int = 8,
        pitch_variation_hz: int = 5,
        add_pauses: bool = True,
        add_breath_pauses: bool = True,
        enhance: bool = False,
        noise_reduction: bool = True,
        normalize_lufs: float = -14.0
    ) -> str:
        """
        Generate natural-sounding speech with AI authenticity features.

        This method applies micro-variations to rate and pitch to make TTS output
        sound more human and less AI-generated. It helps reduce AI detection by:
        - Varying speech rate +/- 8% per sentence (mimics natural pacing)
        - Varying pitch +/- 5Hz per phrase (mimics natural intonation)
        - Adding breath pauses after long sentences
        - Optionally adding dramatic pauses at punctuation

        Args:
            text: Text to convert to speech
            output_file: Output audio file path (mp3)
            voice: Voice to use (default: self.default_voice)
            rate: Base speech rate adjustment (e.g., "+0%", "+10%")
            pitch: Base pitch adjustment (e.g., "+0Hz", "+5Hz")
            volume: Volume adjustment (e.g., "+10%", "-10%")
            rate_variation_percent: Max rate variation percentage (+/-). Default: 8%
            pitch_variation_hz: Max pitch variation in Hz (+/-). Default: 5Hz
            add_pauses: Add dramatic pauses at punctuation. Default: True
            add_breath_pauses: Add breath pauses after long sentences. Default: True
            enhance: Apply professional audio enhancement. Default: False
            noise_reduction: Apply noise reduction (if enhance=True). Default: True
            normalize_lufs: Target loudness (if enhance=True). Default: -14 LUFS

        Returns:
            Path to the generated audio file

        Example:
            tts = TextToSpeech()
            # Generate natural-sounding speech for YouTube
            await tts.generate_natural(
                text="Hello everyone! Welcome to this tutorial.",
                output_file="output/natural_voice.mp3",
                rate_variation_percent=8,
                pitch_variation_hz=5
            )
        """
        voice = voice or self.default_voice

        # Apply natural voice variation
        variation = NaturalVoiceVariation(
            rate_variation_percent=rate_variation_percent,
            pitch_variation_hz=pitch_variation_hz
        )

        # Apply breath pauses first (for long sentences)
        processed_text = text
        if add_breath_pauses:
            processed_text = variation.add_natural_pauses(processed_text)
            logger.debug("Added breath pauses for natural pacing")

        # Apply rate and pitch variations
        processed_text = variation.apply_natural_variation(
            processed_text,
            base_rate=rate,
            base_pitch=pitch
        )

        # Add dramatic pauses at punctuation if requested
        if add_pauses:
            processed_text = self.add_dramatic_pauses(processed_text)
            logger.debug("Added dramatic pauses at punctuation")

        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating natural speech with AI authenticity features: {len(text)} chars -> {output_file}")
        logger.debug(f"Voice: {voice}, Rate variation: +/-{rate_variation_percent}%, Pitch variation: +/-{pitch_variation_hz}Hz")

        # Generate with or without enhancement
        if enhance:
            # Generate to temp file, then enhance
            temp_file = str(output_path.parent / f"_natural_raw_{output_path.name}")

            try:
                communicate = edge_tts.Communicate(
                    text=processed_text,
                    voice=voice,
                    volume=volume
                )
                await communicate.save(temp_file)

                # Import audio processor and enhance
                try:
                    from src.content.audio_processor import AudioProcessor
                except ImportError:
                    from .audio_processor import AudioProcessor

                processor = AudioProcessor()
                enhanced = processor.enhance_voice_professional(
                    input_file=temp_file,
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
                    logger.success(f"Natural enhanced audio saved: {output_file}")
                    return enhanced
                else:
                    # Enhancement failed, keep raw audio
                    logger.warning("Enhancement failed, using raw natural TTS output")
                    os.rename(temp_file, output_file)
                    return output_file

            except ImportError as e:
                logger.warning(f"AudioProcessor not available: {e}. Using raw natural TTS output.")
                try:
                    os.rename(temp_file, output_file)
                except OSError:
                    pass
                return output_file

        else:
            # Generate without enhancement
            try:
                communicate = edge_tts.Communicate(
                    text=processed_text,
                    voice=voice,
                    volume=volume
                )
                await communicate.save(str(output_path))

                logger.success(f"Natural audio saved: {output_file}")
                return str(output_path)

            except (ConnectionError, TimeoutError, OSError, IOError) as e:
                logger.error(f"Natural TTS generation failed: {e}")
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
