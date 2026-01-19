"""
Fish Audio TTS Provider
Higher quality alternative to Edge-TTS

Supports professional voice enhancement for broadcast-quality audio.

Usage:
    tts = FishAudioTTS()
    await tts.generate("Hello!", "output.mp3")

    # With professional enhancement
    await tts.generate_enhanced("Hello!", "output.mp3", enhance=True)
"""
from fish_audio_sdk import Session, TTSRequest
from pathlib import Path
from loguru import logger
import os


class FishAudioTTS:
    """Fish Audio Text-to-Speech provider with professional enhancement support"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("FISH_AUDIO_API_KEY")
        if not self.api_key:
            raise ValueError("FISH_AUDIO_API_KEY not set")
        self.session = Session(self.api_key)

        # High-quality voice presets
        self.voices = {
            "male_us": "your_voice_id",  # Replace with actual voice IDs
            "female_us": "your_voice_id",
            "male_uk": "your_voice_id",
        }

    async def generate(self, text: str, output_file: str, voice: str = "male_us") -> str:
        """Generate speech from text"""
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Use Fish Audio API
            with open(output_file, "wb") as f:
                for chunk in self.session.tts(TTSRequest(text=text)):
                    f.write(chunk)

            logger.info(f"Generated audio: {output_file}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Fish Audio TTS failed: {e}")
            raise

    async def generate_enhanced(
        self,
        text: str,
        output_file: str,
        voice: str = "male_us",
        enhance: bool = True,
        noise_reduction: bool = True,
        normalize_lufs: float = -14.0
    ) -> str:
        """
        Generate speech with professional broadcast-quality enhancement.

        This method generates speech using Fish Audio and then applies
        professional audio processing:
        - FFT-based noise reduction
        - Voice presence EQ boost (2-4kHz)
        - De-essing for sibilant reduction
        - Dynamic compression
        - Loudness normalization to YouTube's -14 LUFS target

        Args:
            text: Text to convert to speech
            output_file: Output audio file path (mp3)
            voice: Voice preset to use
            enhance: Apply professional voice enhancement (default: True)
            noise_reduction: Apply FFT noise reduction (default: True)
            normalize_lufs: Target loudness in LUFS (default: -14)

        Returns:
            Path to the generated and enhanced audio file
        """
        if enhance:
            # Generate to temp file, then enhance
            output_path = Path(output_file)
            temp_file = str(output_path.parent / f"_raw_{output_path.name}")
            raw_audio = await self.generate(text, temp_file, voice)

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
                    logger.warning("Enhancement failed, using raw Fish Audio output")
                    os.rename(temp_file, output_file)
                    return output_file

            except ImportError as e:
                logger.warning(f"AudioProcessor not available: {e}. Using raw Fish Audio output.")
                os.rename(temp_file, output_file)
                return output_file
        else:
            # No enhancement, just generate normally
            return await self.generate(text, output_file, voice)
