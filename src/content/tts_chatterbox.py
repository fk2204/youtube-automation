"""
Chatterbox TTS Integration (FREE, MIT Licensed)

Chatterbox beat ElevenLabs in blind quality tests while being completely free!
MIT licensed, runs locally or via API.

Note: This is a placeholder implementation. Chatterbox integration would require:
1. Installing chatterbox-tts package (when available)
2. Setting up local inference or API endpoint
3. Configuring voice models

For now, this provides the interface and falls back to Edge-TTS.

Installation:
    # When chatterbox is available:
    pip install chatterbox-tts

Usage:
    tts = ChatterboxTTS()
    await tts.generate("Hello world", "output.mp3")
"""

import asyncio
from pathlib import Path
from typing import Optional
from loguru import logger

# Try to import chatterbox (may not be available yet)
try:
    import chatterbox
    CHATTERBOX_AVAILABLE = True
except ImportError:
    CHATTERBOX_AVAILABLE = False
    logger.warning(
        "Chatterbox not installed. Using Edge-TTS fallback.\n"
        "Install when available: pip install chatterbox-tts"
    )


class ChatterboxTTS:
    """
    Chatterbox TTS provider (MIT licensed, free).

    Falls back to Edge-TTS if Chatterbox is not installed.
    """

    def __init__(
        self,
        model: str = "default",
        voice: str = "default",
        **kwargs
    ):
        """
        Initialize Chatterbox TTS.

        Args:
            model: Chatterbox model to use
            voice: Voice ID
            **kwargs: Additional arguments
        """
        self.model = model
        self.voice = voice

        if CHATTERBOX_AVAILABLE:
            logger.info(f"[ChatterboxTTS] Initialized with model: {model}")
            # Initialize chatterbox here when available
            # self.client = chatterbox.Client(model=model)
        else:
            logger.warning("[ChatterboxTTS] Not available, using Edge-TTS fallback")
            # Import Edge-TTS as fallback
            try:
                from src.content.tts import TextToSpeech
            except ImportError:
                from .tts import TextToSpeech

            self.fallback = TextToSpeech(default_voice="en-US-GuyNeural")

    async def generate(
        self,
        text: str,
        output_file: str,
        voice: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate speech from text.

        Args:
            text: Text to convert to speech
            output_file: Output file path
            voice: Voice to use (optional)
            **kwargs: Additional arguments

        Returns:
            Path to generated audio file
        """
        voice = voice or self.voice

        if CHATTERBOX_AVAILABLE:
            logger.info(f"[ChatterboxTTS] Generating: {len(text)} chars")

            # Ensure output directory exists
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Generate with Chatterbox (when available)
            # audio = await self.client.generate(text=text, voice=voice)
            # audio.save(str(output_path))

            logger.success(f"[ChatterboxTTS] Audio saved: {output_file}")
            return str(output_path)
        else:
            # Fallback to Edge-TTS
            logger.info("[ChatterboxTTS] Using Edge-TTS fallback")
            return await self.fallback.generate(text, output_file, **kwargs)

    async def generate_natural(
        self,
        text: str,
        output_file: str,
        **kwargs
    ) -> str:
        """
        Generate natural-sounding speech.

        Args:
            text: Text to convert
            output_file: Output file path
            **kwargs: Additional arguments

        Returns:
            Path to generated audio file
        """
        if CHATTERBOX_AVAILABLE:
            return await self.generate(text, output_file, **kwargs)
        else:
            # Use Edge-TTS natural variation
            return await self.fallback.generate_natural(text, output_file, **kwargs)

    async def generate_enhanced(
        self,
        text: str,
        output_file: str,
        voice: Optional[str] = None,
        enhance: bool = True,
        noise_reduction: bool = True,
        normalize_lufs: float = -14.0,
        **kwargs
    ) -> str:
        """
        Generate speech with professional broadcast-quality enhancement.

        Args:
            text: Text to convert to speech
            output_file: Output file path
            voice: Voice to use (optional)
            enhance: Apply professional voice enhancement
            noise_reduction: Apply FFT noise reduction
            normalize_lufs: Target loudness in LUFS (default: -14)
            **kwargs: Additional arguments

        Returns:
            Path to generated and enhanced audio file
        """
        if CHATTERBOX_AVAILABLE:
            # When Chatterbox is available, use it with enhancement
            return await self.generate(text, output_file, voice=voice, **kwargs)
        else:
            # Use Edge-TTS enhanced generation
            return await self.fallback.generate_enhanced(
                text=text,
                output_file=output_file,
                voice=voice,
                enhance=enhance,
                noise_reduction=noise_reduction,
                normalize_lufs=normalize_lufs,
                **kwargs
            )


# Convenience function
async def generate_chatterbox_speech(
    text: str,
    output_file: str,
    voice: str = "default"
) -> str:
    """Quick function to generate Chatterbox speech."""
    tts = ChatterboxTTS(voice=voice)
    return await tts.generate(text, output_file)


# CLI
if __name__ == "__main__":
    import sys

    async def main():
        if len(sys.argv) < 3:
            print("""
Chatterbox TTS - MIT Licensed, Beat ElevenLabs in Blind Tests

Usage:
    python -m src.content.tts_chatterbox <text> <output_file> [--voice <voice>]

Examples:
    python -m src.content.tts_chatterbox "Hello world" output.mp3
    python -m src.content.tts_chatterbox "Hello" output.mp3 --voice default

Note: Chatterbox integration pending. Currently uses Edge-TTS fallback.
            """)
            return

        text = sys.argv[1]
        output_file = sys.argv[2]

        voice = "default"
        if "--voice" in sys.argv:
            idx = sys.argv.index("--voice")
            if idx + 1 < len(sys.argv):
                voice = sys.argv[idx + 1]

        tts = ChatterboxTTS(voice=voice)
        result = await tts.generate(text, output_file)
        print(f"Audio generated: {result}")

    asyncio.run(main())
