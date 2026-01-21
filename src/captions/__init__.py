"""
Captions Module - Whisper-based captioning system.

Provides automatic captioning using OpenAI Whisper:
- WhisperCaptionGenerator: Generate SRT/VTT/JSON captions from audio
- Word-level timestamp support for karaoke-style subtitles
- Multi-language support with auto-detection

Integrated (2026-01-20) into:
- Pipeline Orchestrator for automatic caption generation
- run.py 'caption' command for standalone usage
"""

from .whisper_generator import (
    WhisperCaptionGenerator,
)

__all__ = [
    "WhisperCaptionGenerator",
]
