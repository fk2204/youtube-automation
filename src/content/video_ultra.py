"""
Ultra Pro Video Generator

Creates broadcast-quality faceless YouTube videos with:
- Ken Burns effect (zoom/pan on footage)
- Dynamic visual changes every 3-5 seconds
- Animated text overlays and lower thirds
- Professional transitions (crossfade, zoom, glitch)
- Niche-specific color grading and styles
- Kinetic typography
- Ambient sound effects

Based on research of top faceless channels:
- Psych2Go (12.7M subs) - Animated characters, soft colors
- MrBallen (8.7M subs) - Dramatic pacing, sound design
- The Infographics Show (15M+ subs) - Motion graphics

Usage:
    from src.content.video_ultra import UltraVideoGenerator

    generator = UltraVideoGenerator()
    generator.create_video(
        audio_file="narration.mp3",
        script=script_object,
        output_file="video.mp4",
        niche="finance"
    )
"""

import os
import re
import math
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger
from dotenv import load_dotenv

# Load environment variables from relative paths (portable)
_env_paths = [
    Path(__file__).parent.parent.parent / "config" / ".env",  # src/content -> root/config
    Path.cwd() / "config" / ".env",  # Current working directory
]
for _env_path in _env_paths:
    if _env_path.exists():
        load_dotenv(_env_path)
        logger.debug(f"Loaded .env from: {_env_path}")
        break

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
except ImportError:
    raise ImportError("Please install pillow: pip install pillow")

# Import audio processor for normalization
try:
    from .audio_processor import AudioProcessor
    AUDIO_PROCESSOR_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSOR_AVAILABLE = False
    logger.debug("AudioProcessor not available - audio enhancement disabled")

# Import hook generator for catchy intros
try:
    from .video_hooks import VideoHookGenerator, HookValidationResult
    HOOKS_AVAILABLE = True
except ImportError:
    HOOKS_AVAILABLE = False
    logger.debug("VideoHookGenerator not available - hooks disabled")

# Import subtitle generator for burned-in captions (15-25% retention boost)
try:
    from .subtitles import SubtitleGenerator, SubtitleTrack
    SUBTITLES_AVAILABLE = True
except ImportError:
    SUBTITLES_AVAILABLE = False
    logger.debug("SubtitleGenerator not available - subtitles disabled")


# GPU encoder detection cache
_GPU_ENCODER_CACHE: Optional[Dict[str, str]] = None


def get_video_encoder(ffmpeg_path: str = None) -> Dict[str, str]:
    """
    Detect available GPU and return the best video encoder settings.

    Checks for hardware encoders in order of preference:
    1. NVIDIA NVENC (h264_nvenc) - Best quality/speed balance
    2. AMD AMF (h264_amf) - Good AMD GPU support
    3. Intel QuickSync (h264_qsv) - Intel integrated/discrete GPU
    4. CPU fallback (libx264) - Always available

    Returns:
        Dict with keys:
        - encoder: The FFmpeg encoder name (e.g., 'h264_nvenc', 'libx264')
        - preset: Encoder-specific preset for quality/speed balance
        - extra_args: List of additional FFmpeg arguments for the encoder
        - is_gpu: Boolean indicating if GPU acceleration is available
        - gpu_type: String indicating GPU type ('nvidia', 'amd', 'intel', 'cpu')
    """
    global _GPU_ENCODER_CACHE

    # Return cached result if available
    if _GPU_ENCODER_CACHE is not None:
        return _GPU_ENCODER_CACHE

    # Find FFmpeg if not provided
    if ffmpeg_path is None:
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            # Check common Windows locations
            common_paths = [
                os.path.expanduser("~\\AppData\\Local\\Microsoft\\WinGet\\Packages"),
                "C:\\ffmpeg\\bin",
                "C:\\Program Files\\ffmpeg\\bin",
            ]
            for base_path in common_paths:
                if os.path.exists(base_path):
                    for root, dirs, files in os.walk(base_path):
                        if "ffmpeg.exe" in files:
                            ffmpeg_path = os.path.join(root, "ffmpeg.exe")
                            break
                    if ffmpeg_path:
                        break

    if not ffmpeg_path:
        logger.warning("FFmpeg not found, defaulting to CPU encoder")
        _GPU_ENCODER_CACHE = {
            "encoder": "libx264",
            "preset": "fast",
            "extra_args": [],
            "is_gpu": False,
            "gpu_type": "cpu"
        }
        return _GPU_ENCODER_CACHE

    def check_encoder(encoder_name: str) -> bool:
        """Check if an encoder is available in FFmpeg."""
        try:
            result = subprocess.run(
                [ffmpeg_path, '-hide_banner', '-encoders'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return encoder_name in result.stdout
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
            return False

    def test_encoder(encoder_name: str, test_args: List[str] = None) -> bool:
        """Test if an encoder actually works (GPU may not be available)."""
        try:
            cmd = [
                ffmpeg_path, '-y', '-hide_banner', '-loglevel', 'error',
                '-f', 'lavfi', '-i', 'color=c=black:s=64x64:d=0.1',
                '-c:v', encoder_name
            ]
            if test_args:
                cmd.extend(test_args)
            cmd.extend(['-f', 'null', '-'])

            result = subprocess.run(cmd, capture_output=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
            return False

    # Check for NVIDIA NVENC (most common for content creators)
    if check_encoder('h264_nvenc'):
        # Test with actual encoding to ensure GPU is available
        if test_encoder('h264_nvenc', ['-preset', 'p4']):
            logger.info("GPU encoder detected: NVIDIA NVENC (h264_nvenc)")
            _GPU_ENCODER_CACHE = {
                "encoder": "h264_nvenc",
                "preset": "p4",  # p4 = balanced quality/speed (p1=fastest, p7=highest quality)
                "extra_args": [
                    "-rc", "vbr",           # Variable bitrate for better quality
                    "-cq", "23",            # Constant quality (similar to CRF 23)
                    "-b:v", "8M",           # Target bitrate 8 Mbps
                    "-maxrate", "12M",      # Max bitrate
                    "-bufsize", "16M",      # Buffer size
                    "-spatial-aq", "1",     # Spatial adaptive quantization
                    "-temporal-aq", "1",    # Temporal adaptive quantization
                ],
                "is_gpu": True,
                "gpu_type": "nvidia"
            }
            return _GPU_ENCODER_CACHE

    # Check for AMD AMF
    if check_encoder('h264_amf'):
        if test_encoder('h264_amf', ['-quality', 'balanced']):
            logger.info("GPU encoder detected: AMD AMF (h264_amf)")
            _GPU_ENCODER_CACHE = {
                "encoder": "h264_amf",
                "preset": "balanced",  # quality, balanced, speed
                "extra_args": [
                    "-rc", "vbr_lat",       # VBR latency mode
                    "-qp_i", "23",          # I-frame QP
                    "-qp_p", "23",          # P-frame QP
                    "-b:v", "8M",           # Target bitrate
                    "-maxrate", "12M",      # Max bitrate
                ],
                "is_gpu": True,
                "gpu_type": "amd"
            }
            return _GPU_ENCODER_CACHE

    # Check for Intel QuickSync
    if check_encoder('h264_qsv'):
        if test_encoder('h264_qsv', ['-preset', 'medium']):
            logger.info("GPU encoder detected: Intel QuickSync (h264_qsv)")
            _GPU_ENCODER_CACHE = {
                "encoder": "h264_qsv",
                "preset": "medium",  # veryfast, faster, fast, medium, slow, veryslow
                "extra_args": [
                    "-global_quality", "23",  # Quality level (similar to CRF)
                    "-b:v", "8M",             # Target bitrate
                    "-maxrate", "12M",        # Max bitrate
                    "-bufsize", "16M",        # Buffer size
                ],
                "is_gpu": True,
                "gpu_type": "intel"
            }
            return _GPU_ENCODER_CACHE

    # Fallback to CPU encoding (libx264)
    logger.info("No GPU encoder available, using CPU encoder (libx264)")
    _GPU_ENCODER_CACHE = {
        "encoder": "libx264",
        "preset": "fast",  # ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
        "extra_args": [
            "-crf", "23",       # Constant Rate Factor (18-28 recommended, lower = better)
            "-b:v", "8M",       # Target bitrate
            "-maxrate", "10M",  # Max bitrate
            "-bufsize", "16M",  # Buffer size
        ],
        "is_gpu": False,
        "gpu_type": "cpu"
    }
    return _GPU_ENCODER_CACHE


def get_encoder_args(encoder_info: Dict[str, str]) -> List[str]:
    """
    Build FFmpeg encoder arguments from encoder info dict.

    Args:
        encoder_info: Dict from get_video_encoder()

    Returns:
        List of FFmpeg arguments for video encoding
    """
    args = ["-c:v", encoder_info["encoder"]]

    # Add preset (different argument names for different encoders)
    if encoder_info["gpu_type"] == "nvidia":
        args.extend(["-preset", encoder_info["preset"]])
    elif encoder_info["gpu_type"] == "amd":
        args.extend(["-quality", encoder_info["preset"]])
    elif encoder_info["gpu_type"] == "intel":
        args.extend(["-preset", encoder_info["preset"]])
    else:  # CPU
        args.extend(["-preset", encoder_info["preset"]])

    # Add extra encoder-specific arguments
    args.extend(encoder_info["extra_args"])

    return args


@dataclass
class VisualSegment:
    """A visual segment in the video."""
    start_time: float
    end_time: float
    clip_path: Optional[str] = None
    text_overlay: Optional[str] = None
    effect: str = "ken_burns"  # ken_burns, zoom_in, zoom_out, pan_left, pan_right
    transition_in: str = "crossfade"  # crossfade, zoom, slide, glitch
    transition_out: str = "crossfade"


class EffectTracker:
    """
    Tracks which Ken Burns effects have been used recently to prevent repetition.

    Ensures visual variety by preventing the same effect from being used
    within a configurable number of segments (default: 3).

    Usage:
        tracker = EffectTracker(effects_list, history_size=3)
        effect = tracker.get_next_effect()  # Returns best available effect
    """

    def __init__(self, effects: List[Dict], history_size: int = 3):
        """
        Initialize the effect tracker.

        Args:
            effects: List of Ken Burns effect dictionaries
            history_size: Number of segments to prevent repetition within (default: 3)
        """
        self.effects = effects
        self.history_size = history_size
        self.recent_effects: List[int] = []  # Indices of recently used effects

    def get_next_effect(self) -> Dict:
        """
        Get the next effect that hasn't been used recently.

        Returns the best available effect that hasn't been used within
        the last `history_size` segments. If all effects have been used
        recently, returns the least recently used one.

        Returns:
            Dict: Ken Burns effect parameters
        """
        # Get indices of available effects (not in recent history)
        available_indices = [
            i for i in range(len(self.effects))
            if i not in self.recent_effects
        ]

        if available_indices:
            # Pick a random effect from available ones
            chosen_index = random.choice(available_indices)
        else:
            # All effects recently used - pick least recently used
            # (the first one in recent_effects is the oldest)
            chosen_index = self.recent_effects[0] if self.recent_effects else 0

        # Update history
        self.recent_effects.append(chosen_index)
        if len(self.recent_effects) > self.history_size:
            self.recent_effects.pop(0)

        return self.effects[chosen_index]

    def reset(self):
        """Reset the effect history."""
        self.recent_effects = []

    def get_effect_name(self, effect: Dict) -> str:
        """Get the name of an effect if available."""
        return effect.get("name", f"effect_{self.effects.index(effect) if effect in self.effects else 'unknown'}")

    def get_usage_stats(self) -> Dict[str, int]:
        """
        Get statistics on effect usage.

        Returns:
            Dict mapping effect names to usage counts
        """
        stats = {}
        for idx in self.recent_effects:
            if idx < len(self.effects):
                name = self.get_effect_name(self.effects[idx])
                stats[name] = stats.get(name, 0) + 1
        return stats


class DynamicSegmentController:
    """
    Adaptive visual pacing controller for content-aware segment timing.

    Manages segment durations based on content type to optimize viewer retention:
    - Hook segments: 1.5s (fast pacing to grab attention)
    - Content segments: 3.5s (standard pacing for information delivery)
    - Emphasis segments: 2.0s (medium pacing for key points)

    Target: 15+ visual changes per minute for optimal engagement.

    Usage:
        controller = DynamicSegmentController()
        duration = controller.get_segment_duration("hook")
        optimized = controller.optimize_segment_timing(script_sections)
    """

    # Segment duration presets (in seconds)
    SEGMENT_DURATIONS = {
        "hook": 1.5,      # Fast pacing - grab attention quickly
        "content": 3.5,   # Standard pacing - deliver information
        "emphasis": 2.0,  # Medium pacing - highlight key points
        "transition": 1.0, # Quick transition moments
        "climax": 2.5,    # Slightly longer for dramatic moments
        "default": 3.0    # Fallback duration
    }

    # Target visual changes per minute for optimal retention
    TARGET_VISUAL_CHANGES_PER_MINUTE = 15

    def __init__(self, custom_durations: Dict[str, float] = None):
        """
        Initialize the segment controller.

        Args:
            custom_durations: Optional dict to override default segment durations
        """
        self.durations = self.SEGMENT_DURATIONS.copy()
        if custom_durations:
            self.durations.update(custom_durations)

    def get_segment_duration(self, segment_type: str) -> float:
        """
        Get the duration for a specific segment type.

        Args:
            segment_type: Type of segment ("hook", "content", "emphasis", etc.)

        Returns:
            Duration in seconds
        """
        return self.durations.get(segment_type, self.durations["default"])

    def calculate_visual_changes_per_minute(
        self,
        total_duration: float,
        segments: List[Dict]
    ) -> int:
        """
        Calculate the number of visual changes per minute.

        Args:
            total_duration: Total video duration in seconds
            segments: List of segment dicts with 'duration' key

        Returns:
            Visual changes per minute (integer)
        """
        if total_duration <= 0:
            return 0

        num_segments = len(segments)
        minutes = total_duration / 60.0

        if minutes <= 0:
            return 0

        return int(num_segments / minutes)

    def optimize_segment_timing(
        self,
        script_sections: List,
        target_changes_per_minute: int = None
    ) -> List[Dict]:
        """
        Optimize segment timing based on script content.

        Analyzes script sections and assigns appropriate segment types
        and durations to achieve optimal visual pacing.

        Args:
            script_sections: List of script section objects
            target_changes_per_minute: Override target (default: 15)

        Returns:
            List of optimized segment dictionaries with:
            - segment_type: Type classification
            - duration: Optimized duration
            - section_index: Reference to original section
        """
        target = target_changes_per_minute or self.TARGET_VISUAL_CHANGES_PER_MINUTE
        optimized_segments = []

        for i, section in enumerate(script_sections):
            # Determine segment type based on section characteristics
            segment_type = self._classify_section(section, i, len(script_sections))
            duration = self.get_segment_duration(segment_type)

            # Check if section has keywords indicating emphasis
            if self._is_emphasis_section(section):
                segment_type = "emphasis"
                duration = self.get_segment_duration("emphasis")

            optimized_segments.append({
                "segment_type": segment_type,
                "duration": duration,
                "section_index": i,
                "section": section
            })

        # Adjust durations if we're far from target visual changes
        optimized_segments = self._balance_pacing(optimized_segments, target)

        return optimized_segments

    def _classify_section(
        self,
        section,
        index: int,
        total_sections: int
    ) -> str:
        """
        Classify a section based on its position and content.

        Args:
            section: Script section object
            index: Section index
            total_sections: Total number of sections

        Returns:
            Segment type string
        """
        # First section is always a hook
        if index == 0:
            return "hook"

        # Last section often has a climax/conclusion
        if index == total_sections - 1:
            return "climax"

        # Second section often continues the hook momentum
        if index == 1:
            return "emphasis"

        # Default to content for middle sections
        return "content"

    def _is_emphasis_section(self, section) -> bool:
        """
        Check if a section should be emphasized based on content.

        Args:
            section: Script section object

        Returns:
            True if section should be emphasized
        """
        # Check for emphasis indicators in section
        emphasis_keywords = [
            "important", "key", "crucial", "remember", "critical",
            "secret", "hack", "tip", "trick", "surprising"
        ]

        # Get section text content
        text = ""
        if hasattr(section, 'title'):
            text += str(section.title).lower()
        if hasattr(section, 'content'):
            text += str(section.content).lower()
        if hasattr(section, 'narration'):
            text += str(section.narration).lower()

        return any(keyword in text for keyword in emphasis_keywords)

    def _balance_pacing(
        self,
        segments: List[Dict],
        target_changes_per_minute: int
    ) -> List[Dict]:
        """
        Balance segment durations to achieve target visual changes per minute.

        Args:
            segments: List of segment dicts
            target_changes_per_minute: Target visual changes

        Returns:
            Adjusted segment list
        """
        if not segments:
            return segments

        # Calculate current total duration
        total_duration = sum(seg["duration"] for seg in segments)
        current_rate = self.calculate_visual_changes_per_minute(total_duration, segments)

        # If we're close enough to target, return as-is
        if abs(current_rate - target_changes_per_minute) <= 2:
            return segments

        # Calculate adjustment factor
        # More segments = shorter durations needed, fewer = longer
        if current_rate < target_changes_per_minute:
            # Need more visual changes - shorten durations slightly
            factor = 0.9
        else:
            # Too many changes - lengthen durations slightly
            factor = 1.1

        # Apply adjustment to content segments (preserve hooks and emphasis timing)
        for seg in segments:
            if seg["segment_type"] == "content":
                seg["duration"] = max(2.0, min(5.0, seg["duration"] * factor))

        return segments

    def get_pacing_summary(self, segments: List[Dict]) -> Dict:
        """
        Get a summary of pacing statistics.

        Args:
            segments: List of segment dicts

        Returns:
            Dict with pacing statistics
        """
        if not segments:
            return {"total_duration": 0, "segment_count": 0, "changes_per_minute": 0}

        total_duration = sum(seg["duration"] for seg in segments)
        segment_count = len(segments)

        type_counts = {}
        for seg in segments:
            seg_type = seg.get("segment_type", "unknown")
            type_counts[seg_type] = type_counts.get(seg_type, 0) + 1

        return {
            "total_duration": total_duration,
            "segment_count": segment_count,
            "changes_per_minute": self.calculate_visual_changes_per_minute(total_duration, segments),
            "type_distribution": type_counts,
            "average_duration": total_duration / segment_count if segment_count > 0 else 0
        }


class UltraVideoGenerator:
    """
    Ultra-professional video generator for faceless YouTube content.
    """

    # Visual segment duration (change visuals every X seconds)
    SEGMENT_DURATION = 4.0  # 3-5 seconds is optimal

    # Transition duration
    TRANSITION_DURATION = 0.5

    # Crossfade settings
    CROSSFADE_DURATION = 0.3  # Smooth blend between segments

    # Background music volume (relative to voice)
    MUSIC_VOLUME = 0.12  # 12% for main videos (lower than Shorts)

    # Niche-specific visual styles
    NICHE_STYLES = {
        "finance": {
            "primary_color": "#00d4aa",
            "secondary_color": "#1a1a2e",
            "accent_color": "#ffd700",
            "text_color": "#ffffff",
            "gradient_colors": ["#1a1a2e", "#0f3460", "#16213e"],
            "overlay_opacity": 0.4,
            "saturation": 0.9,
            "contrast": 1.1,
            "vignette": True,
            "color_grade": "teal_orange",
            "lower_third_style": "modern",
            "font_style": "bold"
        },
        "psychology": {
            "primary_color": "#9b59b6",
            "secondary_color": "#0f0f1a",
            "accent_color": "#e74c3c",
            "text_color": "#ffffff",
            "gradient_colors": ["#0f0f1a", "#1a0a2e", "#2d1b4e"],
            "overlay_opacity": 0.5,
            "saturation": 0.8,
            "contrast": 1.2,
            "vignette": True,
            "color_grade": "cool_dark",
            "lower_third_style": "minimal",
            "font_style": "clean"
        },
        "storytelling": {
            "primary_color": "#e74c3c",
            "secondary_color": "#0d0d0d",
            "accent_color": "#f39c12",
            "text_color": "#ffffff",
            "gradient_colors": ["#0d0d0d", "#1a0a0a", "#2d1515"],
            "overlay_opacity": 0.6,
            "saturation": 0.7,
            "contrast": 1.3,
            "vignette": True,
            "color_grade": "cinematic_dark",
            "lower_third_style": "dramatic",
            "font_style": "serif"
        },
        "default": {
            "primary_color": "#3498db",
            "secondary_color": "#1a1a2e",
            "accent_color": "#e74c3c",
            "text_color": "#ffffff",
            "gradient_colors": ["#1a1a2e", "#16213e", "#1a1a2e"],
            "overlay_opacity": 0.4,
            "saturation": 1.0,
            "contrast": 1.0,
            "vignette": False,
            "color_grade": "neutral",
            "lower_third_style": "modern",
            "font_style": "bold"
        }
    }

    # Ken Burns effect presets (expanded from 6 to 18 effects)
    KEN_BURNS_EFFECTS = [
        # Basic zooms
        {"start_scale": 1.0, "end_scale": 1.15, "start_x": 0, "end_x": 0, "start_y": 0, "end_y": 0, "name": "slow_zoom_in"},
        {"start_scale": 1.15, "end_scale": 1.0, "start_x": 0, "end_x": 0, "start_y": 0, "end_y": 0, "name": "slow_zoom_out"},
        {"start_scale": 1.0, "end_scale": 1.25, "start_x": 0, "end_x": 0, "start_y": 0, "end_y": 0, "name": "dramatic_zoom_in"},

        # Basic pans
        {"start_scale": 1.1, "end_scale": 1.1, "start_x": -50, "end_x": 50, "start_y": 0, "end_y": 0, "name": "pan_right"},
        {"start_scale": 1.1, "end_scale": 1.1, "start_x": 50, "end_x": -50, "start_y": 0, "end_y": 0, "name": "pan_left"},
        {"start_scale": 1.1, "end_scale": 1.1, "start_x": 0, "end_x": 0, "start_y": -40, "end_y": 40, "name": "pan_down"},
        {"start_scale": 1.1, "end_scale": 1.1, "start_x": 0, "end_x": 0, "start_y": 40, "end_y": -40, "name": "pan_up"},

        # Zoom + pan combinations
        {"start_scale": 1.0, "end_scale": 1.2, "start_x": -30, "end_x": 30, "start_y": -20, "end_y": 20, "name": "zoom_pan_diagonal_right"},
        {"start_scale": 1.2, "end_scale": 1.0, "start_x": 30, "end_x": -30, "start_y": 20, "end_y": -20, "name": "zoom_out_pan_diagonal_left"},

        # Diagonal movements (corner to corner)
        {"start_scale": 1.15, "end_scale": 1.15, "start_x": -60, "end_x": 60, "start_y": -40, "end_y": 40, "name": "diagonal_top_left_to_bottom_right"},
        {"start_scale": 1.15, "end_scale": 1.15, "start_x": 60, "end_x": -60, "start_y": -40, "end_y": 40, "name": "diagonal_top_right_to_bottom_left"},
        {"start_scale": 1.15, "end_scale": 1.15, "start_x": -60, "end_x": 60, "start_y": 40, "end_y": -40, "name": "diagonal_bottom_left_to_top_right"},
        {"start_scale": 1.15, "end_scale": 1.15, "start_x": 60, "end_x": -60, "start_y": 40, "end_y": -40, "name": "diagonal_bottom_right_to_top_left"},

        # Focus pull effects (zoom to specific regions)
        {"start_scale": 1.0, "end_scale": 1.3, "start_x": 0, "end_x": -40, "start_y": 0, "end_y": -30, "name": "focus_top_left"},
        {"start_scale": 1.0, "end_scale": 1.3, "start_x": 0, "end_x": 40, "start_y": 0, "end_y": -30, "name": "focus_top_right"},
        {"start_scale": 1.0, "end_scale": 1.3, "start_x": 0, "end_x": 0, "start_y": 0, "end_y": 30, "name": "focus_center_bottom"},

        # Multi-stage / spiral-like movements (simulated with strong diagonal + zoom)
        {"start_scale": 1.0, "end_scale": 1.2, "start_x": -40, "end_x": 40, "start_y": -30, "end_y": 30, "name": "spiral_zoom_clockwise"},
        {"start_scale": 1.2, "end_scale": 1.0, "start_x": 40, "end_x": -40, "start_y": 30, "end_y": -30, "name": "spiral_zoom_counter_clockwise"},
    ]

    def __init__(
        self,
        resolution: Tuple[int, int] = (1920, 1080),
        fps: int = 30
    ):
        self.resolution = resolution
        self.width, self.height = resolution
        self.fps = fps

        # Find FFmpeg
        self.ffmpeg = self._find_ffmpeg()
        self.ffprobe = self._find_ffprobe()

        if not self.ffmpeg:
            logger.error("FFmpeg not found! Install FFmpeg to use video generation.")

        # Initialize stock provider
        try:
            from .multi_stock import MultiStockProvider
            self.stock = MultiStockProvider()
        except ImportError as e:
            logger.warning(f"Multi-stock provider unavailable: {e}")
            try:
                from .stock_footage import StockFootage
                self.stock = StockFootage()
            except ImportError as e:
                logger.warning(f"Stock footage provider unavailable: {e}")
                self.stock = None

        # Temp directory
        self.temp_dir = Path(tempfile.gettempdir()) / "video_ultra"
        self.temp_dir.mkdir(exist_ok=True)

        # Load fonts
        self.fonts = self._load_fonts()

        # Initialize audio processor for normalization
        self.audio_processor = None
        if AUDIO_PROCESSOR_AVAILABLE:
            try:
                self.audio_processor = AudioProcessor()
                logger.debug("AudioProcessor initialized for audio enhancement")
            except Exception as e:
                logger.debug(f"AudioProcessor initialization failed: {e}")

        # Initialize hook generator for catchy intros
        self.hook_generator = None
        if HOOKS_AVAILABLE:
            try:
                self.hook_generator = VideoHookGenerator(
                    resolution=self.resolution,
                    fps=self.fps,
                    is_shorts=False  # Landscape format
                )
                logger.debug("VideoHookGenerator initialized for catchy intros")
            except Exception as e:
                logger.debug(f"VideoHookGenerator initialization failed: {e}")

        # Initialize subtitle generator for burned-in captions (15-25% retention boost)
        self.subtitle_generator = None
        if SUBTITLES_AVAILABLE:
            try:
                self.subtitle_generator = SubtitleGenerator()
                logger.debug("SubtitleGenerator initialized for burned-in captions")
            except Exception as e:
                logger.debug(f"SubtitleGenerator initialization failed: {e}")

        # Detect GPU encoder for hardware-accelerated encoding
        self.encoder_info = get_video_encoder(self.ffmpeg)
        if self.encoder_info["is_gpu"]:
            logger.info(f"GPU acceleration enabled: {self.encoder_info['gpu_type'].upper()} ({self.encoder_info['encoder']})")
        else:
            logger.info("Using CPU encoding (libx264)")

        # Initialize effect tracker for preventing Ken Burns effect repetition
        self.effect_tracker = EffectTracker(self.KEN_BURNS_EFFECTS, history_size=3)
        logger.debug("EffectTracker initialized (prevents effect repetition within 3 segments)")

        # Initialize dynamic segment controller for adaptive visual pacing
        self.segment_controller = DynamicSegmentController()
        logger.debug(f"DynamicSegmentController initialized (target: {self.segment_controller.TARGET_VISUAL_CHANGES_PER_MINUTE} changes/min)")

        logger.info(f"UltraVideoGenerator initialized ({self.width}x{self.height} @ {self.fps}fps)")

    def _find_ffmpeg(self) -> Optional[str]:
        """Find FFmpeg executable."""
        if shutil.which("ffmpeg"):
            return "ffmpeg"

        # Check common Windows locations
        common_paths = [
            os.path.expanduser("~\\AppData\\Local\\Microsoft\\WinGet\\Packages"),
            "C:\\ffmpeg\\bin",
            "C:\\Program Files\\ffmpeg\\bin",
        ]

        for base_path in common_paths:
            if os.path.exists(base_path):
                for root, dirs, files in os.walk(base_path):
                    if "ffmpeg.exe" in files:
                        return os.path.join(root, "ffmpeg.exe")

        return None

    def _find_ffprobe(self) -> Optional[str]:
        """Find FFprobe executable."""
        if shutil.which("ffprobe"):
            return "ffprobe"

        if self.ffmpeg:
            # Only replace the executable name, not directory names
            ffprobe = self.ffmpeg.replace("ffmpeg.exe", "ffprobe.exe")
            if ffprobe == self.ffmpeg:
                # No .exe extension (Linux/Mac)
                if self.ffmpeg.endswith("ffmpeg"):
                    ffprobe = self.ffmpeg[:-6] + "ffprobe"
            if os.path.exists(ffprobe):
                return ffprobe

        return None

    def _get_encoder_args(self, use_fast_preset: bool = False) -> List[str]:
        """
        Get FFmpeg encoder arguments for video encoding.

        Uses GPU acceleration if available, falls back to CPU.

        Args:
            use_fast_preset: If True, use fastest preset for intermediate encoding

        Returns:
            List of FFmpeg arguments for video encoding
        """
        if use_fast_preset:
            # For intermediate clips, use fastest settings
            if self.encoder_info["is_gpu"]:
                if self.encoder_info["gpu_type"] == "nvidia":
                    return ["-c:v", "h264_nvenc", "-preset", "p1"]  # Fastest NVENC preset
                elif self.encoder_info["gpu_type"] == "amd":
                    return ["-c:v", "h264_amf", "-quality", "speed"]
                elif self.encoder_info["gpu_type"] == "intel":
                    return ["-c:v", "h264_qsv", "-preset", "veryfast"]
            return ["-c:v", "libx264", "-preset", "ultrafast"]

        # For final output, use quality settings
        return get_encoder_args(self.encoder_info)

    def _load_fonts(self) -> Dict[str, str]:
        """Load available fonts."""
        fonts = {}
        font_paths = {
            "bold": ["C:\\Windows\\Fonts\\arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
            "regular": ["C:\\Windows\\Fonts\\arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
            "serif": ["C:\\Windows\\Fonts\\times.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"],
        }

        for name, paths in font_paths.items():
            for path in paths:
                if os.path.exists(path):
                    fonts[name] = path
                    break

        return fonts

    def get_audio_duration(self, audio_file: str) -> float:
        """Get audio file duration using FFprobe."""
        if not self.ffprobe or not os.path.exists(audio_file):
            # Fallback estimation
            file_size = os.path.getsize(audio_file)
            return file_size / 16000  # Rough estimate

        try:
            cmd = [
                self.ffprobe, '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                audio_file
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError, ValueError) as e:
            logger.warning(f"FFprobe failed: {e}")

        # Fallback
        return os.path.getsize(audio_file) / 16000

    def create_ken_burns_clip(
        self,
        input_path: str,
        output_path: str,
        duration: float,
        effect_preset: Dict = None,
        style: Dict = None
    ) -> Optional[str]:
        """
        Create a clip with Ken Burns effect (zoom/pan animation).

        Args:
            input_path: Input video or image
            output_path: Output video path
            duration: Target duration
            effect_preset: Ken Burns effect parameters
            style: Niche style settings

        Returns:
            Path to created clip or None
        """
        if not self.ffmpeg or not os.path.exists(input_path):
            return None

        try:
            # Use effect tracker if no preset specified (prevents repetition)
            effect = effect_preset or self.effect_tracker.get_next_effect()
            style = style or self.NICHE_STYLES["default"]

            # Get input info
            is_image = input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))

            # Build FFmpeg filter for Ken Burns
            start_scale = effect["start_scale"]
            end_scale = effect["end_scale"]
            start_x = effect["start_x"]
            end_x = effect["end_x"]
            start_y = effect["start_y"]
            end_y = effect["end_y"]

            # Calculate zoompan parameters
            # zoompan filter: z=zoom, x=pan_x, y=pan_y
            frames = int(duration * self.fps)

            # FFmpeg zoompan expression for smooth animation
            zoom_expr = f"zoom+({end_scale}-{start_scale})/{frames}"
            x_expr = f"(iw-iw/zoom)/2+{start_x}+({end_x}-{start_x})*on/{frames}"
            y_expr = f"(ih-ih/zoom)/2+{start_y}+({end_y}-{start_y})*on/{frames}"

            filters = []

            # Scale input to be larger than output for zoompan
            filters.append(f"scale=8000:-1")

            # Apply Ken Burns zoompan effect
            filters.append(
                f"zoompan=z='{start_scale}+({end_scale}-{start_scale})*on/{frames}':"
                f"x='iw/2-(iw/zoom/2)+{start_x}+({end_x}-{start_x})*on/{frames}':"
                f"y='ih/2-(ih/zoom/2)+{start_y}+({end_y}-{start_y})*on/{frames}':"
                f"d={frames}:s={self.width}x{self.height}:fps={self.fps}"
            )

            # Color grading based on niche
            if style.get("color_grade") == "teal_orange":
                filters.append("colorbalance=rs=0.1:gs=-0.05:bs=-0.1:rm=0.05:gm=0:bm=-0.05")
            elif style.get("color_grade") == "cool_dark":
                filters.append("colorbalance=rs=-0.1:gs=-0.05:bs=0.1")
                filters.append("eq=brightness=-0.05:saturation=0.8")
            elif style.get("color_grade") == "cinematic_dark":
                filters.append("colorbalance=rs=0.05:gs=-0.05:bs=-0.05")
                filters.append("eq=brightness=-0.1:contrast=1.2:saturation=0.7")

            # Saturation and contrast
            sat = style.get("saturation", 1.0)
            con = style.get("contrast", 1.0)
            if sat != 1.0 or con != 1.0:
                filters.append(f"eq=saturation={sat}:contrast={con}")

            # Vignette effect
            if style.get("vignette"):
                filters.append("vignette=PI/4")

            # Darken for text visibility
            overlay_opacity = style.get("overlay_opacity", 0.4)
            if overlay_opacity > 0:
                filters.append(f"eq=brightness=-{overlay_opacity * 0.5}")

            # Fade in/out
            fade_frames = int(self.TRANSITION_DURATION * self.fps)
            filters.append(f"fade=in:0:{fade_frames}")
            filters.append(f"fade=out:st={duration - self.TRANSITION_DURATION}:d={self.TRANSITION_DURATION}")

            filter_str = ",".join(filters)

            # Build command with GPU-accelerated encoding
            encoder_args = self._get_encoder_args(use_fast_preset=True)

            if is_image:
                cmd = [
                    self.ffmpeg, '-y',
                    '-loop', '1',
                    '-i', input_path,
                    '-t', str(duration),
                    '-vf', filter_str
                ] + encoder_args + [
                    '-pix_fmt', 'yuv420p',
                    '-an',
                    output_path
                ]
            else:
                # For video, need different approach
                cmd = [
                    self.ffmpeg, '-y',
                    '-i', input_path,
                    '-t', str(duration),
                    '-vf', f"scale={self.width}:{self.height}:force_original_aspect_ratio=increase,"
                           f"crop={self.width}:{self.height},"
                           f"eq=saturation={sat}:contrast={con}:brightness=-{overlay_opacity * 0.3},"
                           f"fade=in:0:{fade_frames},fade=out:st={duration - self.TRANSITION_DURATION}:d={self.TRANSITION_DURATION}"
                ] + encoder_args + [
                    '-pix_fmt', 'yuv420p',
                    '-an',
                    output_path
                ]

            result = subprocess.run(cmd, capture_output=True, timeout=120)

            if os.path.exists(output_path):
                return output_path

            # Fallback without zoompan
            logger.warning("Ken Burns failed, using simple scale")
            return self._create_simple_clip(input_path, output_path, duration, style)

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as e:
            logger.error(f"Ken Burns clip failed: {e}")
            return None

    def _create_simple_clip(
        self,
        input_path: str,
        output_path: str,
        duration: float,
        style: Dict
    ) -> Optional[str]:
        """Create a simple clip without Ken Burns (fallback)."""
        try:
            is_image = input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))

            filters = [
                f"scale={self.width}:{self.height}:force_original_aspect_ratio=increase",
                f"crop={self.width}:{self.height}",
            ]

            # Color adjustments
            sat = style.get("saturation", 1.0)
            con = style.get("contrast", 1.0)
            overlay = style.get("overlay_opacity", 0.4)
            filters.append(f"eq=saturation={sat}:contrast={con}:brightness=-{overlay * 0.3}")

            # Fade
            fade_frames = int(self.TRANSITION_DURATION * self.fps)
            filters.append(f"fade=in:0:{fade_frames}")
            filters.append(f"fade=out:st={duration - self.TRANSITION_DURATION}:d={self.TRANSITION_DURATION}")

            filter_str = ",".join(filters)

            # Use GPU-accelerated encoding
            encoder_args = self._get_encoder_args(use_fast_preset=True)

            if is_image:
                cmd = [
                    self.ffmpeg, '-y',
                    '-loop', '1', '-i', input_path,
                    '-t', str(duration),
                    '-vf', filter_str
                ] + encoder_args + ['-pix_fmt', 'yuv420p', '-an', output_path]
            else:
                cmd = [
                    self.ffmpeg, '-y',
                    '-stream_loop', '-1', '-i', input_path,
                    '-t', str(duration),
                    '-vf', filter_str
                ] + encoder_args + ['-pix_fmt', 'yuv420p', '-an', output_path]

            subprocess.run(cmd, capture_output=True, timeout=120)
            return output_path if os.path.exists(output_path) else None

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as e:
            logger.error(f"Simple clip failed: {e}")
            return None

    def create_lower_third(
        self,
        text: str,
        output_path: str,
        style: Dict,
        duration: float = 4.0
    ) -> Optional[str]:
        """
        Create an animated lower third graphic.

        Args:
            text: Text to display
            output_path: Output image path
            style: Niche style settings
            duration: Duration for animation
        """
        try:
            # Create transparent image
            img = Image.new('RGBA', self.resolution, (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)

            # Get colors
            primary = style.get("primary_color", "#3498db").lstrip('#')
            primary_rgb = tuple(int(primary[i:i+2], 16) for i in (0, 2, 4))
            secondary = style.get("secondary_color", "#1a1a2e").lstrip('#')
            secondary_rgb = tuple(int(secondary[i:i+2], 16) for i in (0, 2, 4))

            # Load font
            font_style = style.get("font_style", "bold")
            font_path = self.fonts.get(font_style, self.fonts.get("bold"))

            try:
                font = ImageFont.truetype(font_path, 42) if font_path else ImageFont.load_default()
            except (OSError, IOError) as e:
                logger.debug(f"Font loading failed, using default: {e}")
                font = ImageFont.load_default()

            # Calculate text size
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Lower third position (bottom left area)
            bar_height = text_height + 40
            bar_y = self.height - 150

            lower_third_style = style.get("lower_third_style", "modern")

            if lower_third_style == "modern":
                # Accent color bar
                draw.rectangle(
                    [0, bar_y, 8, bar_y + bar_height],
                    fill=primary_rgb + (255,)
                )
                # Background
                draw.rectangle(
                    [8, bar_y, text_width + 60, bar_y + bar_height],
                    fill=secondary_rgb + (220,)
                )

            elif lower_third_style == "minimal":
                # Just a thin line
                draw.rectangle(
                    [40, bar_y + bar_height - 4, text_width + 80, bar_y + bar_height],
                    fill=primary_rgb + (255,)
                )

            elif lower_third_style == "dramatic":
                # Full bar with gradient effect
                for i in range(bar_height):
                    alpha = int(255 * (1 - i / bar_height * 0.3))
                    draw.rectangle(
                        [0, bar_y + i, text_width + 80, bar_y + i + 1],
                        fill=secondary_rgb + (alpha,)
                    )
                # Accent line
                draw.rectangle(
                    [0, bar_y, text_width + 80, bar_y + 4],
                    fill=primary_rgb + (255,)
                )

            # Draw text
            text_x = 30
            text_y = bar_y + (bar_height - text_height) // 2

            # Shadow
            draw.text((text_x + 2, text_y + 2), text, font=font, fill=(0, 0, 0, 180))
            # Main text
            draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255, 255))

            # Save
            img.save(output_path)
            return output_path

        except (OSError, IOError, ValueError) as e:
            logger.error(f"Lower third creation failed: {e}")
            return None

    def create_title_card(
        self,
        title: str,
        output_path: str,
        style: Dict,
        subtitle: str = None
    ) -> Optional[str]:
        """
        Create a clean, professional title card.

        Design principles:
        - Minimal, clean gradient background (no particles/dots)
        - Clear typography hierarchy with generous spacing
        - Subtle vignette for depth
        - High contrast for text readability
        - Professional accent line
        """
        try:
            # Get colors
            primary = style.get("primary_color", "#3498db").lstrip('#')
            primary_rgb = tuple(int(primary[i:i+2], 16) for i in (0, 2, 4))
            secondary = style.get("secondary_color", "#1a1a2e").lstrip('#')
            secondary_rgb = tuple(int(secondary[i:i+2], 16) for i in (0, 2, 4))

            # Create base image
            img = Image.new('RGB', self.resolution, secondary_rgb)
            draw = ImageDraw.Draw(img)

            # Draw clean vertical gradient (dark to slightly lighter, subtle)
            for y in range(self.height):
                # Smooth gradient from top to bottom
                ratio = y / self.height
                # Use a subtle curve for more natural gradient
                curve = ratio * ratio  # Quadratic easing
                color = tuple(
                    int(secondary_rgb[i] + (primary_rgb[i] - secondary_rgb[i]) * curve * 0.08)
                    for i in range(3)
                )
                draw.line([(0, y), (self.width, y)], fill=color)

            # Add subtle radial vignette for depth (darkens edges)
            # Create vignette overlay
            vignette = Image.new('RGB', self.resolution, (0, 0, 0))
            vignette_draw = ImageDraw.Draw(vignette)
            center_x, center_y = self.width // 2, self.height // 2
            max_radius = int(math.sqrt(center_x**2 + center_y**2))

            for r in range(max_radius, 0, -4):
                # Vignette intensity increases toward edges
                intensity = int(255 * (1 - (r / max_radius) ** 1.5))
                vignette_draw.ellipse(
                    [center_x - r, center_y - r, center_x + r, center_y + r],
                    fill=(intensity, intensity, intensity)
                )

            # Blend vignette with base image
            img = Image.composite(img, Image.new('RGB', self.resolution, (0, 0, 0)), vignette)

            # Recreate draw object after composite
            draw = ImageDraw.Draw(img)

            # Load fonts with proper sizes for hierarchy
            try:
                title_font = ImageFont.truetype(self.fonts.get("bold", ""), 72)
                sub_font = ImageFont.truetype(self.fonts.get("regular", ""), 32)
            except (OSError, IOError) as e:
                logger.debug(f"Font loading failed, using default: {e}")
                title_font = ImageFont.load_default()
                sub_font = ImageFont.load_default()

            # Wrap title with generous margins
            max_width = self.width - 300  # More margin for cleaner look
            words = title.split()
            lines = []
            current_line = ""

            for word in words:
                test_line = f"{current_line} {word}".strip()
                bbox = draw.textbbox((0, 0), test_line, font=title_font)
                if bbox[2] - bbox[0] <= max_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)

            # Calculate layout with proper spacing
            line_height = 95  # Generous line spacing
            total_text_height = len(lines[:3]) * line_height
            if subtitle:
                total_text_height += 70  # Space for subtitle

            # Center vertically with slight upward offset for visual balance
            start_y = (self.height - total_text_height) // 2 - 20

            # Draw title lines with clean shadow for depth
            for i, line in enumerate(lines[:3]):
                bbox = draw.textbbox((0, 0), line, font=title_font)
                text_width = bbox[2] - bbox[0]
                x = (self.width - text_width) // 2
                y = start_y + i * line_height

                # Soft shadow for depth (offset and slightly transparent feel)
                shadow_offset = 3
                draw.text((x + shadow_offset, y + shadow_offset), line, font=title_font, fill=(0, 0, 0))

                # Main title in white for maximum readability
                draw.text((x, y), line, font=title_font, fill=(255, 255, 255))

            # Draw subtitle below title with proper spacing
            subtitle_y = start_y + len(lines[:3]) * line_height + 25
            if subtitle:
                sub_text = subtitle[:60]
                bbox = draw.textbbox((0, 0), sub_text, font=sub_font)
                x = (self.width - (bbox[2] - bbox[0])) // 2
                # Muted color for hierarchy
                draw.text((x, subtitle_y), sub_text, font=sub_font, fill=(180, 180, 180))

            # Add subtle accent line below content
            accent_y = subtitle_y + (50 if subtitle else 25)
            line_width = 120  # Shorter, more elegant line
            line_thickness = 3

            # Accent line with primary color
            draw.rectangle(
                [
                    (self.width - line_width) // 2,
                    accent_y,
                    (self.width + line_width) // 2,
                    accent_y + line_thickness
                ],
                fill=primary_rgb
            )

            img.save(output_path)
            return output_path

        except (OSError, IOError, ValueError) as e:
            logger.error(f"Title card creation failed: {e}")
            return None

    def create_gradient_background(
        self,
        output_path: str,
        style: Dict,
        duration: float,
        animated: bool = True
    ) -> Optional[str]:
        """
        Create an animated gradient background.

        Args:
            output_path: Output video path
            style: Niche style settings
            duration: Duration in seconds
            animated: If True, gradient slowly animates

        Returns:
            Path to created video or None
        """
        try:
            secondary = style.get("secondary_color", "#1a1a2e").lstrip('#')
            secondary_rgb = tuple(int(secondary[i:i+2], 16) for i in (0, 2, 4))
            primary = style.get("primary_color", "#3498db").lstrip('#')
            primary_rgb = tuple(int(primary[i:i+2], 16) for i in (0, 2, 4))
            accent = style.get("accent_color", "#e74c3c").lstrip('#')
            accent_rgb = tuple(int(accent[i:i+2], 16) for i in (0, 2, 4))

            if animated:
                # Create animated gradient using FFmpeg gradients filter
                # This creates a smooth shifting gradient effect
                fade_frames = int(self.TRANSITION_DURATION * self.fps)
                frames = int(duration * self.fps)

                # Build FFmpeg filter for animated gradient
                # Uses geq (generic equation) for animated colors
                encoder_args = self._get_encoder_args(use_fast_preset=True)
                cmd = [
                    self.ffmpeg, '-y',
                    '-f', 'lavfi',
                    '-i', f'color=c=black:s={self.width}x{self.height}:d={duration}:r={self.fps}',
                    '-vf',
                    f"geq="
                    f"r='clip({secondary_rgb[0]}+{primary_rgb[0]-secondary_rgb[0]}*sin(2*PI*N/{frames}+X/{self.width}*PI)*0.3+{accent_rgb[0]-secondary_rgb[0]}*(Y/{self.height})*0.2,0,255)':"
                    f"g='clip({secondary_rgb[1]}+{primary_rgb[1]-secondary_rgb[1]}*sin(2*PI*N/{frames}+X/{self.width}*PI+1)*0.3+{accent_rgb[1]-secondary_rgb[1]}*(Y/{self.height})*0.2,0,255)':"
                    f"b='clip({secondary_rgb[2]}+{primary_rgb[2]-secondary_rgb[2]}*sin(2*PI*N/{frames}+X/{self.width}*PI+2)*0.3+{accent_rgb[2]-secondary_rgb[2]}*(Y/{self.height})*0.2,0,255)',"
                    f"fade=in:0:{fade_frames},fade=out:st={duration - 0.5}:d=0.5"
                ] + encoder_args + [
                    '-pix_fmt', 'yuv420p',
                    '-an',
                    output_path
                ]

                result = subprocess.run(cmd, capture_output=True, timeout=120)

                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    return output_path

                # Fallback to static gradient if animated fails
                logger.warning("Animated gradient failed, using static")

            # Static gradient fallback
            img = Image.new('RGB', self.resolution, secondary_rgb)
            draw = ImageDraw.Draw(img)

            # Radial gradient with multiple layers
            center_x, center_y = self.width // 2, self.height // 2

            # Draw vertical gradient first
            for y in range(self.height):
                ratio = y / self.height
                color = tuple(
                    int(secondary_rgb[i] * (1 - ratio * 0.4) + accent_rgb[i] * ratio * 0.15)
                    for i in range(3)
                )
                draw.line([(0, y), (self.width, y)], fill=color)

            # Overlay radial gradient
            for r in range(0, max(self.width, self.height), 8):
                ratio = min(1.0, r / 900)
                alpha = int(255 * (1 - ratio) * 0.15)
                color = tuple(min(255, c + alpha) for c in primary_rgb)
                draw.ellipse(
                    [center_x - r, center_y - r, center_x + r, center_y + r],
                    outline=color
                )

            # Add subtle particles/bokeh effect
            for _ in range(30):
                x = random.randint(0, self.width)
                y = random.randint(0, self.height)
                size = random.randint(5, 25)
                alpha = random.randint(10, 40)
                color = tuple(min(255, primary_rgb[i] + alpha) for i in range(3))
                draw.ellipse([x, y, x + size, y + size], fill=color)

            # Save frame
            frame_path = str(Path(output_path).with_suffix('.png'))
            img.save(frame_path)

            # Create video with subtle zoom animation
            fade_frames = int(self.TRANSITION_DURATION * self.fps)
            frames = int(duration * self.fps)

            # Use GPU-accelerated encoding
            encoder_args = self._get_encoder_args(use_fast_preset=True)
            cmd = [
                self.ffmpeg, '-y',
                '-loop', '1', '-i', frame_path,
                '-t', str(duration),
                '-vf',
                f"scale=2048:-1,"
                f"zoompan=z='1.0+0.001*on':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={self.width}x{self.height}:fps={self.fps},"
                f"fade=in:0:{fade_frames},fade=out:st={duration - 0.5}:d=0.5"
            ] + encoder_args + ['-pix_fmt', 'yuv420p', '-an', output_path]
            subprocess.run(cmd, capture_output=True, timeout=120)

            # Cleanup
            if os.path.exists(frame_path):
                os.remove(frame_path)

            return output_path if os.path.exists(output_path) else None

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError, IOError) as e:
            logger.error(f"Gradient background failed: {e}")
            return None

    def _get_alternative_search_terms(self, title: str, niche: str) -> List[str]:
        """
        Generate alternative search terms when primary topic search returns nothing.

        Args:
            title: Original title/topic
            niche: Content niche

        Returns:
            List of alternative search terms to try
        """
        alternatives = []

        # Extract key words from title (words longer than 3 chars)
        words = [w.strip('.,!?:;') for w in title.split() if len(w) > 3]
        if words:
            # Try pairs of words
            for i in range(min(3, len(words))):
                alternatives.append(words[i])

            # Try combining first two meaningful words
            if len(words) >= 2:
                alternatives.append(f"{words[0]} {words[1]}")

        # Add niche-specific generic terms as fallback
        niche_fallbacks = {
            "finance": ["money", "business", "office", "success", "charts"],
            "psychology": ["person thinking", "conversation", "emotions", "mental health", "brain"],
            "storytelling": ["dramatic scene", "mystery", "night city", "documentary", "interview"],
            "technology": ["technology", "computer", "digital", "innovation", "future"],
            "motivation": ["success", "achievement", "sunrise", "running", "determination"],
            "health": ["wellness", "exercise", "healthy lifestyle", "medical", "nature"],
        }

        fallback_terms = niche_fallbacks.get(niche, ["abstract background", "cinematic", "nature", "city"])
        alternatives.extend(fallback_terms[:3])

        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for term in alternatives:
            term_lower = term.lower()
            if term_lower not in seen:
                seen.add(term_lower)
                unique.append(term)

        return unique

    def create_video(
        self,
        audio_file: str,
        script,
        output_file: str,
        niche: str = "default",
        background_music: str = None,
        music_volume: float = None,
        normalize_audio: bool = True,
        subtitles_enabled: bool = True,
        subtitle_style: str = "regular"
    ) -> Optional[str]:
        """
        Create a professional video with all effects.

        Args:
            audio_file: Path to narration audio
            script: VideoScript object or dict with title/sections
            output_file: Output video path
            niche: Content niche for styling
            background_music: Optional path to background music file
            music_volume: Optional music volume (0.0-1.0)
            normalize_audio: Normalize audio to YouTube's -14 LUFS standard (default: True)
            subtitles_enabled: Burn subtitles into video for 15-25% retention boost (default: True)
            subtitle_style: Subtitle style - "regular", "shorts", "minimal", "cinematic" (default: "regular")

        Returns:
            Path to created video or None
        """
        if not self.ffmpeg:
            logger.error("FFmpeg not available")
            return None

        if not os.path.exists(audio_file):
            logger.error(f"Audio file not found: {audio_file}")
            return None

        logger.info(f"Creating ultra-pro video: {output_file}")
        logger.info(f"Niche: {niche}")

        # Pre-process audio (normalization to -14 LUFS)
        processed_audio = audio_file
        temp_audio_file = None

        if normalize_audio and self.audio_processor:
            logger.info("Pre-processing audio: Normalizing to YouTube's -14 LUFS standard")
            temp_audio_file = str(self.temp_dir / "normalized_voice.mp3")
            result = self.audio_processor.normalize_audio(audio_file, temp_audio_file)
            if result:
                processed_audio = result
                logger.success("Audio normalized successfully")
            else:
                logger.warning("Audio normalization failed, using original audio")

        try:
            # Get style
            style = self.NICHE_STYLES.get(niche, self.NICHE_STYLES["default"])

            # Get audio duration
            audio_duration = self.get_audio_duration(processed_audio)
            logger.info(f"Audio duration: {audio_duration:.1f}s")

            # Calculate number of visual segments (change every 3-5 seconds)
            num_segments = int(audio_duration / self.SEGMENT_DURATION) + 1
            logger.info(f"Creating {num_segments} visual segments")

            # Get script info
            title = getattr(script, 'title', str(script)) if script else "Video"
            sections = getattr(script, 'sections', []) if script else []

            # PRE-FLIGHT CHECK: Verify stock provider is available and working
            stock_available = False
            if self.stock:
                if hasattr(self.stock, 'is_available') and self.stock.is_available():
                    stock_available = True
                    logger.info("Stock footage provider is available")
                elif hasattr(self.stock, 'clients') and self.stock.clients:
                    stock_available = True
                    logger.info(f"Stock footage provider has {len(self.stock.clients)} source(s)")
                else:
                    logger.warning(
                        "Stock footage provider has no configured API sources!\n"
                        "  Video will use GRADIENT BACKGROUNDS ONLY (small file size).\n"
                        "  To fix: Add PEXELS_API_KEY to config/.env"
                    )
            else:
                logger.warning(
                    "No stock footage provider available!\n"
                    "  Video will use GRADIENT BACKGROUNDS ONLY (small file size).\n"
                    "  Expected video size: ~0.1-0.5 MB (much smaller than normal)"
                )

            # Fetch stock footage
            stock_clips = []
            downloaded_paths = []
            gradient_fallback_count = 0

            if self.stock and stock_available:
                # Build search terms from script
                search_terms = [title]
                if sections:
                    for s in sections[:5]:
                        if hasattr(s, 'keywords') and s.keywords:
                            search_terms.extend(s.keywords[:2])
                        elif hasattr(s, 'title') and s.title:
                            search_terms.append(s.title)

                # Get clips
                logger.info("Fetching stock footage...")

                # Use multi-stock provider if available
                if hasattr(self.stock, 'get_clips_for_topic'):
                    clips = self.stock.get_clips_for_topic(
                        topic=title,
                        niche=niche,
                        count=num_segments + 5,
                        min_total_duration=int(audio_duration * 1.5)
                    )

                    # Try alternative search terms if primary search returns nothing
                    if not clips:
                        logger.warning(f"No clips found for primary topic: '{title}'. Trying alternatives...")
                        alternative_terms = self._get_alternative_search_terms(title, niche)
                        for alt_term in alternative_terms[:3]:
                            logger.info(f"Trying alternative search: '{alt_term}'")
                            clips = self.stock.get_clips_for_topic(
                                topic=alt_term,
                                niche=niche,
                                count=num_segments + 5,
                                min_total_duration=int(audio_duration * 1.5)
                            )
                            if clips:
                                logger.success(f"Found {len(clips)} clips with alternative term: '{alt_term}'")
                                break

                    for clip in clips:
                        path = self.stock.download_clip(clip)
                        if path:
                            downloaded_paths.append(path)
                else:
                    # Fallback to basic stock
                    for term in search_terms[:5]:
                        found = self.stock.search_videos(term, count=3)
                        for clip in found:
                            if hasattr(self.stock, 'download_video'):
                                path = self.temp_dir / f"clip_{len(downloaded_paths)}.mp4"
                                result = self.stock.download_video(clip, str(path))
                                if result:
                                    downloaded_paths.append(result)

                # Log warning if insufficient clips found
                if len(downloaded_paths) < num_segments:
                    shortfall = num_segments - len(downloaded_paths)
                    logger.warning(
                        f"Insufficient stock footage! Only {len(downloaded_paths)} clips for {num_segments} segments.\n"
                        f"  {shortfall} segment(s) will use GRADIENT FALLBACK.\n"
                        f"  This will result in smaller video file size."
                    )
                else:
                    logger.success(f"Downloaded {len(downloaded_paths)} stock clips (sufficient for {num_segments} segments)")

            # Also get some images for variety
            stock_images = []
            if self.stock and hasattr(self.stock, 'search_images'):
                images = self.stock.search_images(title, count=5)
                for img in images:
                    if hasattr(self.stock, 'download_image'):
                        path = self.stock.download_image(img)
                        if path:
                            stock_images.append(path)

            # Create video segments
            segment_files = []
            current_time = 0.0

            # 1. CATCHY ANIMATED HOOK (first 3-5 seconds)
            # Grab attention in first second - NEVER start with blank screen!
            hook_duration = min(4.0, max(3.0, audio_duration * 0.1))

            # Generate hook text based on niche
            hook_text = None
            if self.hook_generator:
                # Get suggested hooks for this topic
                suggested_hooks = self.hook_generator.get_suggested_hooks(title, niche, count=1)
                if suggested_hooks:
                    hook_text = suggested_hooks[0]
                    logger.info(f"Hook text: {hook_text}")

            # Use script hook if available
            if not hook_text and hasattr(script, 'hook') and script.hook:
                hook_text = script.hook
            elif not hook_text:
                # Default hook based on title
                hook_text = title[:60] if len(title) <= 60 else f"{title[:57]}..."

            # Try to create animated hook intro
            hook_video = None
            if self.hook_generator:
                logger.info("Creating animated hook intro (catchy first 3-4 seconds)...")
                animation_type = random.choice(["zoom_in", "fade_in", "slide_up"])
                hook_video = self.hook_generator.create_animated_intro(
                    hook_text=hook_text,
                    niche=niche,
                    duration=hook_duration,
                    animation_type=animation_type,
                    output_path=str(self.temp_dir / "hook_intro.mp4")
                )

            if hook_video and os.path.exists(hook_video):
                segment_files.append(hook_video)
                current_time += hook_duration
                logger.success(f"Animated hook created: {hook_duration}s")
            else:
                # Fallback to static title card with hook text (still better than blank!)
                logger.warning("Animated hook failed, using enhanced title card")
                title_duration = min(4.0, audio_duration * 0.1)
                title_frame = self.temp_dir / "title_frame.png"
                self.create_title_card(
                    title=hook_text[:60],  # Use hook text instead of plain title
                    output_path=str(title_frame),
                    style=style,
                    subtitle=getattr(script, 'description', '')[:50] if hasattr(script, 'description') else None
                )

                if title_frame.exists():
                    title_video = self.temp_dir / "title.mp4"
                    self._create_simple_clip(str(title_frame), str(title_video), title_duration, style)
                    if title_video.exists():
                        segment_files.append(str(title_video))
                        current_time += title_duration

            # 2. Main content segments with adaptive pacing
            media_index = 0
            all_media = downloaded_paths + stock_images  # Mix videos and images
            random.shuffle(all_media)  # Randomize for variety

            # Reset effect tracker for fresh video
            self.effect_tracker.reset()

            # Use DynamicSegmentController to optimize segment timing if sections available
            optimized_segments = []
            if sections:
                optimized_segments = self.segment_controller.optimize_segment_timing(sections)
                pacing_summary = self.segment_controller.get_pacing_summary(optimized_segments)
                logger.info(
                    f"Adaptive pacing enabled: {pacing_summary['changes_per_minute']} visual changes/min "
                    f"(target: {self.segment_controller.TARGET_VISUAL_CHANGES_PER_MINUTE})"
                )

            segment_num = 0
            optimized_index = 0  # Track position in optimized segments
            while current_time < audio_duration - 0.5:
                remaining = audio_duration - current_time

                # Use adaptive segment duration if available, otherwise use default
                if optimized_segments and optimized_index < len(optimized_segments):
                    seg_type = optimized_segments[optimized_index].get("segment_type", "content")
                    seg_duration = min(
                        self.segment_controller.get_segment_duration(seg_type),
                        remaining
                    )
                    optimized_index += 1
                else:
                    seg_duration = min(self.SEGMENT_DURATION, remaining)

                if seg_duration < 1:
                    break

                segment_path = self.temp_dir / f"segment_{segment_num}.mp4"

                # Use stock media if available
                if media_index < len(all_media):
                    media_path = all_media[media_index]
                    media_index += 1

                    # Apply Ken Burns effect using EffectTracker (prevents repetition)
                    effect = self.effect_tracker.get_next_effect()
                    effect_name = effect.get("name", f"effect_{segment_num}")
                    logger.debug(f"Segment {segment_num}: Using effect '{effect_name}' ({seg_duration:.1f}s)")

                    result = self.create_ken_burns_clip(
                        media_path,
                        str(segment_path),
                        seg_duration,
                        effect,
                        style
                    )

                    if result:
                        segment_files.append(result)
                        current_time += seg_duration
                        segment_num += 1
                        continue

                # Fallback: gradient background
                gradient_fallback_count += 1
                if gradient_fallback_count == 1:
                    logger.warning(
                        f"Using GRADIENT FALLBACK for segment {segment_num} "
                        f"(no stock footage available). Video file will be smaller."
                    )
                else:
                    logger.debug(f"Gradient fallback for segment {segment_num}")

                result = self.create_gradient_background(
                    str(segment_path),
                    style,
                    seg_duration
                )

                if result:
                    segment_files.append(result)

                current_time += seg_duration
                segment_num += 1

                # Reuse media if we run out
                if media_index >= len(all_media) and all_media:
                    media_index = 0
                    random.shuffle(all_media)

            # Log summary of gradient fallback usage
            if gradient_fallback_count > 0:
                logger.warning(
                    f"VIDEO QUALITY NOTICE: {gradient_fallback_count}/{segment_num} segments used gradient fallback.\n"
                    f"  This results in smaller video file size (~0.1-0.5 MB instead of 10-50 MB).\n"
                    f"  To fix: Ensure PEXELS_API_KEY is set correctly in config/.env"
                )

            if not segment_files:
                logger.error("No video segments created")
                return None

            # Log segment creation summary with effect usage stats
            effect_stats = self.effect_tracker.get_usage_stats()
            logger.info(f"Created {len(segment_files)} video segments ({segment_num - gradient_fallback_count} stock, {gradient_fallback_count} gradient)")
            if effect_stats:
                unique_effects = len(effect_stats)
                logger.debug(f"Effect variety: {unique_effects} unique effects used in recent segments")

            # 3. Concatenate all segments with crossfade transitions
            video_only = self.temp_dir / "video_only.mp4"

            # Use crossfade for smooth transitions (eliminates black frames)
            result = self.concatenate_with_crossfade(segment_files, str(video_only))

            if not video_only.exists():
                logger.error("Video concatenation failed")
                return None

            # 4. Combine with audio (using normalized audio if available)
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)

            final_cmd = [
                self.ffmpeg, '-y',
                '-i', str(video_only),
                '-i', processed_audio,  # Use normalized audio
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-b:a', '256k',  # Improved bitrate for better quality
                '-shortest',
                output_file
            ]

            result = subprocess.run(final_cmd, capture_output=True, timeout=300)

            if os.path.exists(output_file):
                # 5. Add background music if provided
                music_path = background_music or self.get_niche_music_path(niche)
                if music_path:
                    logger.info("Adding background music...")
                    video_with_music = self.temp_dir / "video_with_music.mp4"
                    music_result = self.add_background_music(
                        output_file,
                        music_path,
                        str(video_with_music),
                        music_volume
                    )
                    if music_result and str(video_with_music) == music_result:
                        # Replace output with music version
                        shutil.move(str(video_with_music), output_file)

                # 6. Burn subtitles into video (15-25% retention boost)
                if subtitles_enabled and self.subtitle_generator:
                    logger.info("Burning subtitles into video (15-25% retention boost)...")

                    # Get script text for subtitle generation
                    script_text = ""
                    if hasattr(script, 'get_full_narration'):
                        script_text = script.get_full_narration()
                    elif hasattr(script, 'narration'):
                        script_text = script.narration
                    elif hasattr(script, 'sections'):
                        # Build narration from sections
                        for section in script.sections:
                            if hasattr(section, 'narration'):
                                script_text += section.narration + " "
                            elif hasattr(section, 'content'):
                                script_text += section.content + " "
                    elif isinstance(script, str):
                        script_text = script
                    elif isinstance(script, dict):
                        script_text = script.get('narration', script.get('content', str(script)))

                    if script_text.strip():
                        try:
                            # Generate subtitles synced with audio
                            subtitle_track = self.subtitle_generator.sync_subtitles_with_audio(
                                script_text=script_text.strip(),
                                audio_path=processed_audio,
                                max_chars=50 if subtitle_style == "regular" else 30
                            )

                            if subtitle_track and subtitle_track.cues:
                                logger.info(f"Generated {len(subtitle_track.cues)} subtitle cues")

                                # Burn subtitles into video
                                video_with_subs = self.temp_dir / "video_with_subtitles.mp4"
                                sub_result = self.subtitle_generator.burn_subtitles(
                                    video_path=output_file,
                                    subtitle_track=subtitle_track,
                                    output_path=str(video_with_subs),
                                    style=subtitle_style,
                                    niche=niche
                                )

                                if sub_result and os.path.exists(str(video_with_subs)):
                                    # Replace output with subtitled version
                                    shutil.move(str(video_with_subs), output_file)
                                    logger.success(f"Subtitles burned successfully ({subtitle_style} style)")
                                else:
                                    logger.warning("Subtitle burning failed, continuing without subtitles")
                            else:
                                logger.warning("No subtitle cues generated, skipping subtitle burning")
                        except Exception as e:
                            logger.warning(f"Subtitle generation failed: {e}, continuing without subtitles")
                    else:
                        logger.warning("No script text available for subtitle generation")
                elif subtitles_enabled and not self.subtitle_generator:
                    logger.info("Subtitles requested but SubtitleGenerator not available")

                # 7. Validate hook (ensure first seconds are not blank)
                if self.hook_generator:
                    validation = self.hook_generator.validate_hook(output_file, hook_duration=4.0)
                    if not validation.is_valid:
                        logger.warning(f"Hook validation: {', '.join(validation.suggestions)}")
                    else:
                        logger.info("Hook validation passed: First seconds are visually engaging")

                file_size = os.path.getsize(output_file) / (1024 * 1024)
                logger.success(f"Ultra video created: {output_file} ({file_size:.1f} MB)")
                self._cleanup()
                return output_file
            else:
                logger.error(f"Final video creation failed: {result.stderr.decode()[:500] if hasattr(result, 'stderr') else 'unknown error'}")

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError, IOError) as e:
            logger.error(f"Video creation failed: {e}")
            import traceback
            traceback.print_exc()

        return None

    def _cleanup(self):
        """Clean up temporary files."""
        try:
            for f in self.temp_dir.glob("*"):
                if f.is_file():
                    f.unlink()
        except (OSError, IOError) as e:
            logger.debug(f"Cleanup failed: {e}")

    def _get_video_duration(self, video_path: str) -> Optional[float]:
        """
        Get the actual duration of a video file using ffprobe.

        Args:
            video_path: Path to video file

        Returns:
            Duration in seconds or None if failed
        """
        if not self.ffprobe or not os.path.exists(video_path):
            return None

        try:
            cmd = [
                self.ffprobe, '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                duration = float(result.stdout.strip())
                logger.debug(f"Video duration for {Path(video_path).name}: {duration:.3f}s")
                return duration
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError, ValueError) as e:
            logger.warning(f"Failed to get duration for {video_path}: {e}")

        return None

    def concatenate_with_crossfade(
        self,
        segment_files: List[str],
        output_path: str
    ) -> Optional[str]:
        """
        Concatenate video segments with crossfade transitions.

        Uses FFmpeg xfade filter for smooth blending between clips.
        Eliminates black frames between segments.

        For N segments with durations [d1, d2, d3, ...]:
        - Offset for xfade between segment i and i+1 = sum(d1..di) - (i * crossfade_duration)
        - This accounts for variable-length segments and overlapping transitions

        Args:
            segment_files: List of video segment paths
            output_path: Output video path

        Returns:
            Path to output or None
        """
        if not segment_files:
            logger.error("No segment files provided for concatenation")
            return None

        if len(segment_files) == 1:
            # Just copy the single file
            shutil.copy(segment_files[0], output_path)
            return output_path

        try:
            # Get actual durations for all segments using ffprobe
            durations = []
            for i, seg in enumerate(segment_files):
                duration = self._get_video_duration(seg)
                if duration is None:
                    # Fallback to assumed duration if ffprobe fails
                    duration = self.SEGMENT_DURATION
                    logger.warning(f"Segment {i} ({Path(seg).name}): using fallback duration {duration}s")
                else:
                    logger.debug(f"Segment {i} ({Path(seg).name}): actual duration {duration:.3f}s")
                durations.append(duration)

            logger.info(f"Concatenating {len(segment_files)} segments with crossfade")
            logger.info(f"Segment durations: {[f'{d:.2f}s' for d in durations]}")

            xfade_duration = self.CROSSFADE_DURATION

            # Validate durations - ensure segments are long enough for crossfade
            min_required_duration = xfade_duration + 0.1  # Need at least crossfade + small buffer
            for i, d in enumerate(durations):
                if d < min_required_duration:
                    logger.warning(
                        f"Segment {i} duration ({d:.2f}s) is too short for crossfade "
                        f"(min: {min_required_duration:.2f}s). Falling back to simple concat."
                    )
                    return self._simple_concat(segment_files, output_path)

            # Create input arguments
            inputs = []
            for seg in segment_files:
                inputs.extend(['-i', seg])

            # Build xfade filter chain
            # For xfade, offset is the time in the OUTPUT stream where transition starts
            # With overlapping transitions:
            # - First segment plays from 0 to offset1, then fades into second
            # - offset1 = d1 - xfade_duration
            # - offset2 = d1 + d2 - 2*xfade_duration (accounting for overlap)
            # General formula: offset_i = sum(d1..di) - i*xfade_duration

            if len(segment_files) == 2:
                # Simple case: two segments
                # Offset = first segment duration minus crossfade duration
                offset = durations[0] - xfade_duration
                if offset < 0.1:
                    logger.warning(f"First segment too short for crossfade (offset would be {offset:.2f}s)")
                    return self._simple_concat(segment_files, output_path)

                filter_str = f"[0:v][1:v]xfade=transition=fade:duration={xfade_duration}:offset={offset:.2f}[outv]"
                logger.debug(f"Two-segment xfade filter: offset={offset:.2f}s")
            else:
                # Multiple segments - chain xfades
                # Each xfade reduces total duration by xfade_duration
                parts = []
                current_label = "0:v"
                cumulative_duration = durations[0]

                for i in range(1, len(segment_files)):
                    next_input = f"{i}:v"
                    out_label = f"v{i}" if i < len(segment_files) - 1 else "outv"

                    # Offset = cumulative duration of all previous segments in OUTPUT
                    # minus xfade_duration (because we want transition to start before end)
                    # Since each previous xfade "ate" xfade_duration, we subtract (i-1)*xfade_duration
                    # from raw cumulative, then subtract one more xfade_duration for this transition
                    # Formula: offset_i = sum(d1..di) - i*xfade_duration
                    offset = cumulative_duration - (i * xfade_duration)

                    # Validate offset
                    if offset < 0.1:
                        logger.warning(
                            f"Invalid offset {offset:.2f}s at segment {i} "
                            f"(cumulative: {cumulative_duration:.2f}s, transitions: {i}). "
                            f"Segments may be too short for crossfade."
                        )
                        return self._simple_concat(segment_files, output_path)

                    parts.append(
                        f"[{current_label}][{next_input}]xfade=transition=fade:duration={xfade_duration}:offset={offset:.2f}[{out_label}]"
                    )
                    logger.debug(f"Xfade {i}: offset={offset:.2f}s (cumulative={cumulative_duration:.2f}s)")

                    # Add next segment duration to cumulative
                    cumulative_duration += durations[i]
                    current_label = out_label

                filter_str = ";".join(parts)

            # Calculate expected output duration for logging
            expected_duration = sum(durations) - (len(segment_files) - 1) * xfade_duration
            logger.info(f"Expected output duration: {expected_duration:.2f}s")

            # Build FFmpeg command with GPU-accelerated encoding
            encoder_args = self._get_encoder_args(use_fast_preset=True)
            cmd = [self.ffmpeg, '-y'] + inputs + [
                '-filter_complex', filter_str,
                '-map', '[outv]'
            ] + encoder_args + [
                '-pix_fmt', 'yuv420p',
                output_path
            ]

            logger.debug(f"FFmpeg xfade command: {' '.join(cmd[:10])}... [filter truncated]")
            result = subprocess.run(cmd, capture_output=True, timeout=600)

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                actual_duration = self._get_video_duration(output_path)
                logger.info(
                    f"Crossfade concatenation successful: {output_path} "
                    f"(duration: {actual_duration:.2f}s)" if actual_duration else
                    f"Crossfade concatenation successful: {output_path}"
                )
                return output_path

            # Log failure details
            stderr_output = result.stderr.decode('utf-8', errors='replace') if result.stderr else 'No stderr'
            logger.error(f"FFmpeg xfade failed (returncode={result.returncode})")
            logger.error(f"FFmpeg stderr: {stderr_output[:1000]}")

            # Fallback to simple concat if xfade fails
            logger.warning("Crossfade failed, falling back to simple concat")
            return self._simple_concat(segment_files, output_path)

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as e:
            logger.error(f"Crossfade concatenation failed with exception: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return self._simple_concat(segment_files, output_path)

    def _simple_concat(self, segment_files: List[str], output_path: str) -> Optional[str]:
        """Simple concatenation fallback."""
        concat_file = self.temp_dir / "concat_list.txt"
        with open(concat_file, 'w') as f:
            for seg in segment_files:
                safe_path = seg.replace("'", "'\\''")
                f.write(f"file '{safe_path}'\n")

        # Use GPU-accelerated encoding
        encoder_args = self._get_encoder_args(use_fast_preset=True)
        cmd = [
            self.ffmpeg, '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_file)
        ] + encoder_args + [
            '-pix_fmt', 'yuv420p',
            output_path
        ]

        subprocess.run(cmd, capture_output=True, timeout=600)
        return output_path if os.path.exists(output_path) else None

    def add_background_music(
        self,
        video_file: str,
        music_file: str,
        output_file: str,
        music_volume: float = None
    ) -> Optional[str]:
        """
        Add background music to a video.

        Args:
            video_file: Input video path
            music_file: Background music audio path
            output_file: Output video path
            music_volume: Volume level (0.0-1.0), defaults to MUSIC_VOLUME

        Returns:
            Path to output or None
        """
        if not os.path.exists(video_file):
            logger.error(f"Video not found: {video_file}")
            return None

        if not os.path.exists(music_file):
            logger.warning(f"Music file not found: {music_file}")
            return video_file  # Return original without music

        try:
            volume = music_volume or self.MUSIC_VOLUME

            # Get video duration
            video_duration = self.get_audio_duration(video_file)

            # FFmpeg command to mix audio tracks
            # Stream 0:a is original audio (voice), stream 1:a is music
            # Fade out music at the end
            cmd = [
                self.ffmpeg, '-y',
                '-i', video_file,
                '-stream_loop', '-1',  # Loop music if shorter than video
                '-i', music_file,
                '-filter_complex',
                f"[1:a]volume={volume},afade=t=out:st={video_duration - 2}:d=2[music];"
                f"[0:a][music]amix=inputs=2:duration=first[aout]",
                '-map', '0:v',
                '-map', '[aout]',
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-b:a', '192k',
                output_file
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=300)

            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                logger.info(f"Added background music at {volume*100:.0f}% volume")
                return output_file

            logger.warning("Music mixing failed, returning original")
            return video_file

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as e:
            logger.error(f"Background music failed: {e}")
            return video_file

    def get_niche_music_path(self, niche: str) -> Optional[str]:
        """
        Get path to background music file for a niche.

        Searches multiple locations for niche-specific or generic music files.
        Priority order:
        1. Niche-specific file (e.g., finance.mp3)
        2. Generic background.mp3
        3. Any .mp3 file in the music directory

        Args:
            niche: Content niche (finance, psychology, storytelling)

        Returns:
            Path to music file or None if not found
        """
        # Build list of directories to search (in priority order)
        project_root = Path(__file__).parent.parent.parent
        search_dirs = [
            project_root / "assets" / "music",           # Primary: assets/music/
            project_root / "music",                       # Alternative: music/
            Path.cwd() / "assets" / "music",             # CWD-relative
            Path.home() / "youtube-automation" / "assets" / "music",  # Home directory
        ]

        logger.debug(f"Searching for background music for niche: {niche}")

        for music_dir in search_dirs:
            if not music_dir.exists():
                continue

            logger.debug(f"Checking music directory: {music_dir}")

            # Try niche-specific first (e.g., finance.mp3)
            niche_music = music_dir / f"{niche}.mp3"
            if niche_music.exists():
                logger.info(f"Found niche-specific music: {niche_music}")
                return str(niche_music)

            # Try generic background.mp3
            generic_music = music_dir / "background.mp3"
            if generic_music.exists():
                logger.info(f"Found generic background music: {generic_music}")
                return str(generic_music)

            # Try any mp3 in the directory
            mp3_files = list(music_dir.glob("*.mp3"))
            if mp3_files:
                selected = mp3_files[0]
                logger.info(f"Found fallback music file: {selected}")
                return str(selected)

        # No music found - log helpful message
        primary_dir = project_root / "assets" / "music"
        logger.warning(
            f"No background music found for niche '{niche}'. "
            f"To add music, place MP3 files in: {primary_dir}"
        )
        logger.info(
            f"Expected files: {niche}.mp3, background.mp3, or any .mp3 file. "
            f"See assets/music/README.md for setup instructions."
        )

        return None

    def burn_captions(
        self,
        video_file: str,
        captions: List[Dict],
        output_file: str,
        style: Dict = None
    ) -> Optional[str]:
        """
        Burn captions/subtitles directly into video.

        Args:
            video_file: Input video path
            captions: List of caption dicts with 'start', 'end', 'text' keys
                     e.g. [{'start': 0.0, 'end': 2.5, 'text': 'Hello world'}]
            output_file: Output video path
            style: Optional niche style for caption colors

        Returns:
            Path to output or None
        """
        if not os.path.exists(video_file):
            logger.error(f"Video not found: {video_file}")
            return None

        if not captions:
            logger.warning("No captions provided")
            return video_file

        try:
            style = style or self.NICHE_STYLES["default"]

            # Create SRT subtitle file
            srt_path = self.temp_dir / "captions.srt"
            self._create_srt_file(captions, str(srt_path))

            if not srt_path.exists():
                logger.error("Failed to create SRT file")
                return video_file

            # Get font settings
            primary = style.get("primary_color", "#ffffff").lstrip('#')
            font_style = style.get("font_style", "bold")

            # Build subtitle filter
            # Use drawtext for more control over styling
            font_path = self.fonts.get(font_style, self.fonts.get("bold", ""))

            # Escape special characters in path for FFmpeg
            srt_escaped = str(srt_path).replace('\\', '/').replace(':', '\\:')

            # Subtitle filter with styling
            subtitle_filter = (
                f"subtitles='{srt_escaped}':"
                f"force_style='FontSize=24,FontName=Arial,PrimaryColour=&HFFFFFF&,"
                f"OutlineColour=&H000000&,Outline=2,Shadow=1,MarginV=50'"
            )

            # Use GPU-accelerated encoding
            encoder_args = self._get_encoder_args(use_fast_preset=False)  # Use quality preset for final output
            cmd = [
                self.ffmpeg, '-y',
                '-i', video_file,
                '-vf', subtitle_filter
            ] + encoder_args + [
                '-c:a', 'copy',
                output_file
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=600)

            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                logger.info("Captions burned into video")
                return output_file

            # Fallback: try with drawtext filter if subtitles fails
            logger.warning("Subtitle filter failed, trying drawtext")
            return self._burn_captions_drawtext(video_file, captions, output_file, style)

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as e:
            logger.error(f"Caption burning failed: {e}")
            return video_file

    def _create_srt_file(self, captions: List[Dict], output_path: str) -> bool:
        """Create SRT subtitle file from caption data."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, cap in enumerate(captions, 1):
                    start = self._seconds_to_srt_time(cap.get('start', 0))
                    end = self._seconds_to_srt_time(cap.get('end', cap.get('start', 0) + 2))
                    text = cap.get('text', '')

                    f.write(f"{i}\n")
                    f.write(f"{start} --> {end}\n")
                    f.write(f"{text}\n\n")

            return os.path.exists(output_path)

        except (OSError, IOError, KeyError) as e:
            logger.error(f"SRT creation failed: {e}")
            return False

    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _burn_captions_drawtext(
        self,
        video_file: str,
        captions: List[Dict],
        output_file: str,
        style: Dict
    ) -> Optional[str]:
        """Fallback caption burning using drawtext filter."""
        try:
            # Build drawtext filters for each caption
            filters = []
            font_path = self.fonts.get("bold", "")

            for cap in captions[:50]:  # Limit to prevent command line overflow
                start = cap.get('start', 0)
                end = cap.get('end', start + 2)
                text = cap.get('text', '').replace("'", "\\'").replace(":", "\\:")

                if not text:
                    continue

                filter_str = (
                    f"drawtext=text='{text}':"
                    f"fontsize=48:fontcolor=white:borderw=3:bordercolor=black:"
                    f"x=(w-text_w)/2:y=h-120:"
                    f"enable='between(t,{start},{end})'"
                )

                if font_path:
                    font_escaped = font_path.replace('\\', '/').replace(':', '\\:')
                    filter_str = filter_str.replace("fontsize=48", f"fontfile='{font_escaped}':fontsize=48")

                filters.append(filter_str)

            if not filters:
                return video_file

            # Combine all filters
            filter_chain = ",".join(filters)

            # Use GPU-accelerated encoding
            encoder_args = self._get_encoder_args(use_fast_preset=False)  # Use quality preset for final output
            cmd = [
                self.ffmpeg, '-y',
                '-i', video_file,
                '-vf', filter_chain
            ] + encoder_args + [
                '-c:a', 'copy',
                output_file
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=600)

            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                return output_file

            return video_file

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as e:
            logger.error(f"Drawtext caption fallback failed: {e}")
            return video_file

    def generate_captions_from_script(self, script, audio_duration: float) -> List[Dict]:
        """
        Generate caption timings from a script object.

        Args:
            script: VideoScript object with sections
            audio_duration: Total audio duration for timing

        Returns:
            List of caption dicts with start, end, text
        """
        captions = []

        sections = getattr(script, 'sections', []) if script else []
        if not sections:
            return captions

        # Estimate timing based on text length
        total_text = ""
        for section in sections:
            if hasattr(section, 'content'):
                total_text += section.content + " "
            elif hasattr(section, 'narration'):
                total_text += section.narration + " "

        total_words = len(total_text.split())
        if total_words == 0:
            return captions

        words_per_second = total_words / audio_duration
        current_time = 0.0

        for section in sections:
            content = ""
            if hasattr(section, 'content'):
                content = section.content
            elif hasattr(section, 'narration'):
                content = section.narration

            if not content:
                continue

            # Split into sentences for better caption timing
            sentences = re.split(r'(?<=[.!?])\s+', content)

            for sentence in sentences:
                if not sentence.strip():
                    continue

                # Calculate duration based on word count
                words = len(sentence.split())
                duration = words / words_per_second
                duration = max(1.5, min(duration, 6.0))  # 1.5-6 seconds per caption

                # Truncate long sentences
                display_text = sentence.strip()
                if len(display_text) > 80:
                    display_text = display_text[:77] + "..."

                captions.append({
                    'start': current_time,
                    'end': current_time + duration,
                    'text': display_text
                })

                current_time += duration

                # Don't exceed audio duration
                if current_time >= audio_duration:
                    break

            if current_time >= audio_duration:
                break

        return captions


# Test
if __name__ == "__main__":
    print("\n" + "="*60)
    print("ULTRA VIDEO GENERATOR TEST")
    print("="*60 + "\n")

    gen = UltraVideoGenerator()
    print(f"FFmpeg: {gen.ffmpeg}")
    print(f"FFprobe: {gen.ffprobe}")
    print(f"Stock provider: {type(gen.stock).__name__ if gen.stock else 'None'}")

    # Test title card
    style = gen.NICHE_STYLES["finance"]
    gen.create_title_card(
        "10 Passive Income Ideas",
        "output/test_title.png",
        style,
        "That Actually Work in 2025"
    )
    print("\nTitle card created: output/test_title.png")
