"""
Whisper Local Captioning System (FREE!)

Generate accurate, word-level captions using OpenAI Whisper locally.
Replaces paid services like Kapwing ($16/mo savings).

Features:
- Local processing (no API calls, completely free)
- Word-level timestamps for animated captions
- SRT and VTT format support
- Multiple model sizes (tiny, base, small, medium, large)
- GPU acceleration support
- Language detection and translation

Installation:
    pip install openai-whisper

Usage:
    generator = WhisperCaptionGenerator(model_size="base")

    # Generate SRT captions
    await generator.generate_captions(
        audio_file="audio.mp3",
        output_file="captions.srt",
        format="srt"
    )

    # Word-level timestamps for kinetic typography
    word_data = await generator.generate_word_timestamps("audio.mp3")
    for word in word_data:
        print(f"{word['text']}: {word['start']}s - {word['end']}s")
"""

import asyncio
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning(
        "Whisper not installed. Install with: pip install openai-whisper\n"
        "For GPU support: pip install openai-whisper[cuda]"
    )

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class WordTimestamp:
    """Word-level timestamp data."""
    text: str
    start: float  # seconds
    end: float    # seconds
    confidence: float = 1.0

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence
        }


@dataclass
class CaptionSegment:
    """Caption segment with timing."""
    index: int
    start: float
    end: float
    text: str
    words: List[WordTimestamp] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "words": [w.to_dict() for w in self.words]
        }


class WhisperCaptionGenerator:
    """
    Generate captions using OpenAI Whisper locally (FREE!).

    Whisper is state-of-the-art speech recognition that runs locally.
    Perfect for YouTube captions with word-level timing.
    """

    # Model size tradeoffs
    MODEL_INFO = {
        "tiny": {
            "size": "~75 MB",
            "vram": "~1 GB",
            "speed": "~10x realtime",
            "accuracy": "Good for clear audio"
        },
        "base": {
            "size": "~142 MB",
            "vram": "~1 GB",
            "speed": "~7x realtime",
            "accuracy": "Better quality, still fast"
        },
        "small": {
            "size": "~466 MB",
            "vram": "~2 GB",
            "speed": "~4x realtime",
            "accuracy": "Production quality"
        },
        "medium": {
            "size": "~1.5 GB",
            "vram": "~5 GB",
            "speed": "~2x realtime",
            "accuracy": "Very high quality"
        },
        "large": {
            "size": "~2.9 GB",
            "vram": "~10 GB",
            "speed": "~1x realtime",
            "accuracy": "Best quality"
        }
    }

    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        cache_dir: str = "data/whisper_cache"
    ):
        """
        Initialize Whisper caption generator.

        Args:
            model_size: Model size ("tiny", "base", "small", "medium", "large")
            device: Device to use ("cuda", "cpu", or None for auto-detect)
            cache_dir: Directory to cache Whisper models
        """
        if not WHISPER_AVAILABLE:
            raise ImportError(
                "Whisper not installed. Install with: pip install openai-whisper"
            )

        self.model_size = model_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect device
        if device is None:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                device = "cuda"
                logger.info("GPU detected - using CUDA acceleration")
            else:
                device = "cpu"
                logger.info("Using CPU (GPU not available)")

        self.device = device

        # Load model
        logger.info(f"Loading Whisper model: {model_size} ({self.MODEL_INFO[model_size]['size']})")
        self.model = whisper.load_model(
            model_size,
            device=device,
            download_root=str(self.cache_dir)
        )
        logger.success(f"Whisper {model_size} loaded on {device}")

    async def generate_captions(
        self,
        audio_file: str,
        output_file: str,
        format: str = "srt",
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> str:
        """
        Generate captions from audio file.

        Args:
            audio_file: Path to audio file (mp3, wav, etc.)
            output_file: Path for output caption file
            format: Caption format ("srt", "vtt", "json")
            language: Language code (None for auto-detect)
            task: "transcribe" or "translate" (translate to English)

        Returns:
            Path to generated caption file
        """
        logger.info(f"Generating {format.upper()} captions from: {audio_file}")

        # Verify audio file exists
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        # Transcribe (run in thread to avoid blocking)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.model.transcribe(
                audio_file,
                language=language,
                task=task,
                word_timestamps=True,  # Get word-level timing
                verbose=False
            )
        )

        # Extract segments
        segments = []
        for i, segment in enumerate(result["segments"]):
            # Extract word timestamps if available
            words = []
            if "words" in segment:
                for word_data in segment["words"]:
                    words.append(WordTimestamp(
                        text=word_data.get("word", "").strip(),
                        start=word_data.get("start", 0.0),
                        end=word_data.get("end", 0.0),
                        confidence=word_data.get("probability", 1.0)
                    ))

            segments.append(CaptionSegment(
                index=i + 1,
                start=segment["start"],
                end=segment["end"],
                text=segment["text"].strip(),
                words=words
            ))

        # Save in requested format
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format.lower() == "srt":
            self._save_srt(segments, output_path)
        elif format.lower() == "vtt":
            self._save_vtt(segments, output_path)
        elif format.lower() == "json":
            self._save_json(segments, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'srt', 'vtt', or 'json'")

        detected_lang = result.get("language", "unknown")
        logger.success(
            f"Captions generated: {output_file} "
            f"(Language: {detected_lang}, {len(segments)} segments)"
        )

        return str(output_path)

    async def generate_word_timestamps(
        self,
        audio_file: str,
        language: Optional[str] = None
    ) -> List[Dict]:
        """
        Generate word-level timestamps for animated captions.

        Perfect for kinetic typography effects in videos.

        Args:
            audio_file: Path to audio file
            language: Language code (None for auto-detect)

        Returns:
            List of word dictionaries with timing data
        """
        logger.info(f"Extracting word-level timestamps from: {audio_file}")

        # Transcribe with word timestamps
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.model.transcribe(
                audio_file,
                language=language,
                word_timestamps=True,
                verbose=False
            )
        )

        # Collect all words
        all_words = []
        for segment in result["segments"]:
            if "words" in segment:
                for word_data in segment["words"]:
                    all_words.append({
                        "text": word_data.get("word", "").strip(),
                        "start": word_data.get("start", 0.0),
                        "end": word_data.get("end", 0.0),
                        "confidence": word_data.get("probability", 1.0),
                        "duration": word_data.get("end", 0.0) - word_data.get("start", 0.0)
                    })

        logger.success(f"Extracted {len(all_words)} word timestamps")
        return all_words

    def _save_srt(self, segments: List[CaptionSegment], output_path: Path):
        """Save captions in SRT format."""
        with open(output_path, "w", encoding="utf-8") as f:
            for seg in segments:
                # Write segment index
                f.write(f"{seg.index}\n")

                # Write timestamps in SRT format (HH:MM:SS,mmm --> HH:MM:SS,mmm)
                start_time = self._format_srt_time(seg.start)
                end_time = self._format_srt_time(seg.end)
                f.write(f"{start_time} --> {end_time}\n")

                # Write text
                f.write(f"{seg.text}\n\n")

    def _save_vtt(self, segments: List[CaptionSegment], output_path: Path):
        """Save captions in WebVTT format."""
        with open(output_path, "w", encoding="utf-8") as f:
            # VTT header
            f.write("WEBVTT\n\n")

            for seg in segments:
                # Write timestamps in VTT format (HH:MM:SS.mmm --> HH:MM:SS.mmm)
                start_time = self._format_vtt_time(seg.start)
                end_time = self._format_vtt_time(seg.end)
                f.write(f"{start_time} --> {end_time}\n")

                # Write text
                f.write(f"{seg.text}\n\n")

    def _save_json(self, segments: List[CaptionSegment], output_path: Path):
        """Save captions in JSON format with full data."""
        data = {
            "segments": [seg.to_dict() for seg in segments],
            "word_count": sum(len(seg.words) for seg in segments),
            "segment_count": len(segments),
            "total_duration": segments[-1].end if segments else 0
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _format_srt_time(self, seconds: float) -> str:
        """Format time for SRT (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _format_vtt_time(self, seconds: float) -> str:
        """Format time for VTT (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    async def detect_language(self, audio_file: str) -> Tuple[str, float]:
        """
        Detect language of audio file.

        Args:
            audio_file: Path to audio file

        Returns:
            Tuple of (language_code, confidence)
        """
        logger.info(f"Detecting language: {audio_file}")

        # Load audio and pad/trim it to 30 seconds for language detection
        loop = asyncio.get_event_loop()
        audio = await loop.run_in_executor(
            None,
            lambda: whisper.load_audio(audio_file)
        )
        audio = whisper.pad_or_trim(audio)

        # Make log-Mel spectrogram
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

        # Detect language
        _, probs = self.model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
        confidence = probs[detected_lang]

        logger.success(f"Language detected: {detected_lang} ({confidence:.1%} confidence)")
        return detected_lang, confidence

    @staticmethod
    def get_model_info(model_size: str = None) -> Dict:
        """Get information about Whisper models."""
        if model_size:
            return WhisperCaptionGenerator.MODEL_INFO.get(model_size, {})
        return WhisperCaptionGenerator.MODEL_INFO


# Convenience functions
async def generate_captions_from_audio(
    audio_file: str,
    output_file: str,
    model_size: str = "base",
    format: str = "srt"
) -> str:
    """Quick function to generate captions."""
    generator = WhisperCaptionGenerator(model_size=model_size)
    return await generator.generate_captions(audio_file, output_file, format=format)


async def get_word_timestamps(
    audio_file: str,
    model_size: str = "base"
) -> List[Dict]:
    """Quick function to get word-level timestamps."""
    generator = WhisperCaptionGenerator(model_size=model_size)
    return await generator.generate_word_timestamps(audio_file)


# CLI entry point
async def main():
    """CLI entry point for caption generation."""
    import sys

    if len(sys.argv) < 3:
        print("""
Whisper Caption Generator - FREE Local Captioning

Usage:
    python -m src.captions.whisper_generator <audio_file> <output_file> [options]

Options:
    --model <size>      Model size (tiny/base/small/medium/large) [default: base]
    --format <fmt>      Output format (srt/vtt/json) [default: srt]
    --language <code>   Language code (en, es, fr, etc.) [default: auto-detect]
    --translate         Translate to English
    --words-only        Output word timestamps as JSON only

Examples:
    # Generate SRT captions with base model
    python -m src.captions.whisper_generator audio.mp3 captions.srt

    # Use larger model for better quality
    python -m src.captions.whisper_generator audio.mp3 captions.srt --model small

    # Generate VTT format
    python -m src.captions.whisper_generator audio.mp3 captions.vtt --format vtt

    # Get word-level timestamps for animation
    python -m src.captions.whisper_generator audio.mp3 words.json --words-only

Model Info:
""")
        for model, info in WhisperCaptionGenerator.MODEL_INFO.items():
            print(f"  {model:8} - Size: {info['size']:10} Speed: {info['speed']:15} - {info['accuracy']}")
        return

    audio_file = sys.argv[1]
    output_file = sys.argv[2]

    # Parse options
    model_size = "base"
    format = "srt"
    language = None
    task = "transcribe"
    words_only = False

    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == "--model" and i + 1 < len(sys.argv):
            model_size = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--format" and i + 1 < len(sys.argv):
            format = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--language" and i + 1 < len(sys.argv):
            language = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--translate":
            task = "translate"
            i += 1
        elif sys.argv[i] == "--words-only":
            words_only = True
            format = "json"
            i += 1
        else:
            i += 1

    # Generate captions
    generator = WhisperCaptionGenerator(model_size=model_size)

    if words_only:
        # Just get word timestamps
        words = await generator.generate_word_timestamps(audio_file, language)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"words": words}, f, indent=2)
        print(f"Word timestamps saved to: {output_file}")
    else:
        # Generate full captions
        result = await generator.generate_captions(
            audio_file, output_file, format, language, task
        )
        print(f"Captions saved to: {result}")


if __name__ == "__main__":
    asyncio.run(main())
