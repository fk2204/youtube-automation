# Content generation modules
from .tts import TextToSpeech
from .script_writer import ScriptWriter
from .audio_processor import AudioProcessor
from .script_validator import ScriptValidator, ValidationResult, clean_script, validate_script, improve_script
from .subtitles import (
    SubtitleGenerator,
    SubtitleTrack,
    SubtitleCue,
    SubtitlePosition,
    SUBTITLE_STYLES,
    NICHE_SUBTITLE_STYLES,
)
from .video_hooks import (
    VideoHookGenerator,
    HookTemplate,
    HookAnimationType,
    HookValidationResult,
    create_hook_generator,
)
from .thumbnail_generator import ThumbnailGenerator
from .stock_cache import StockCache, SmartPrefetcher
from .parallel_downloader import ParallelDownloader, DownloadResult, BatchDownloadResult

# Viral Hooks (Integrated 2026-01-20)
try:
    from .viral_hooks import (
        ViralHookGenerator,
        HookFormula,
        OpenLoop,
        PatternInterrupt,
        generate_viral_hook,
        enhance_retention,
    )
    VIRAL_HOOKS_AVAILABLE = True
except ImportError:
    VIRAL_HOOKS_AVAILABLE = False

# Chatterbox TTS (MIT Licensed, Integrated 2026-01-20)
try:
    from .tts_chatterbox import ChatterboxTTS, generate_chatterbox_speech
    CHATTERBOX_AVAILABLE = True
except ImportError:
    CHATTERBOX_AVAILABLE = False

__all__ = [
    "TextToSpeech",
    "ScriptWriter",
    "AudioProcessor",
    "ScriptValidator",
    "ValidationResult",
    "clean_script",
    "validate_script",
    "improve_script",
    "SubtitleGenerator",
    "SubtitleTrack",
    "SubtitleCue",
    "SubtitlePosition",
    "SUBTITLE_STYLES",
    "NICHE_SUBTITLE_STYLES",
    # Video Hooks
    "VideoHookGenerator",
    "HookTemplate",
    "HookAnimationType",
    "HookValidationResult",
    "create_hook_generator",
    # Viral Hooks (NEW)
    "ViralHookGenerator",
    "HookFormula",
    "OpenLoop",
    "PatternInterrupt",
    "generate_viral_hook",
    "enhance_retention",
    # Chatterbox TTS (NEW)
    "ChatterboxTTS",
    "generate_chatterbox_speech",
    # Thumbnails
    "ThumbnailGenerator",
    # Stock footage caching and prefetching
    "StockCache",
    "SmartPrefetcher",
    # Parallel downloading
    "ParallelDownloader",
    "DownloadResult",
    "BatchDownloadResult",
]
