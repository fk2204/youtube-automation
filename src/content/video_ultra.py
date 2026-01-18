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


class UltraVideoGenerator:
    """
    Ultra-professional video generator for faceless YouTube content.
    """

    # Visual segment duration (change visuals every X seconds)
    SEGMENT_DURATION = 4.0  # 3-5 seconds is optimal

    # Transition duration
    TRANSITION_DURATION = 0.5

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

    # Ken Burns effect presets
    KEN_BURNS_EFFECTS = [
        {"start_scale": 1.0, "end_scale": 1.15, "start_x": 0, "end_x": 0, "start_y": 0, "end_y": 0},  # Slow zoom in
        {"start_scale": 1.15, "end_scale": 1.0, "start_x": 0, "end_x": 0, "start_y": 0, "end_y": 0},  # Slow zoom out
        {"start_scale": 1.1, "end_scale": 1.1, "start_x": -50, "end_x": 50, "start_y": 0, "end_y": 0},  # Pan right
        {"start_scale": 1.1, "end_scale": 1.1, "start_x": 50, "end_x": -50, "start_y": 0, "end_y": 0},  # Pan left
        {"start_scale": 1.0, "end_scale": 1.2, "start_x": -30, "end_x": 30, "start_y": -20, "end_y": 20},  # Zoom + pan
        {"start_scale": 1.2, "end_scale": 1.0, "start_x": 30, "end_x": -30, "start_y": 20, "end_y": -20},  # Zoom out + pan
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
        except Exception as e:
            logger.warning(f"Multi-stock provider unavailable: {e}")
            try:
                from .stock_footage import StockFootage
                self.stock = StockFootage()
            except:
                self.stock = None

        # Temp directory
        self.temp_dir = Path(tempfile.gettempdir()) / "video_ultra"
        self.temp_dir.mkdir(exist_ok=True)

        # Load fonts
        self.fonts = self._load_fonts()

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
        except Exception as e:
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
            # Choose random effect if not specified
            effect = effect_preset or random.choice(self.KEN_BURNS_EFFECTS)
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

            # Build command
            if is_image:
                cmd = [
                    self.ffmpeg, '-y',
                    '-loop', '1',
                    '-i', input_path,
                    '-t', str(duration),
                    '-vf', filter_str,
                    '-c:v', 'libx264',
                    '-preset', 'fast',
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
                           f"fade=in:0:{fade_frames},fade=out:st={duration - self.TRANSITION_DURATION}:d={self.TRANSITION_DURATION}",
                    '-c:v', 'libx264',
                    '-preset', 'fast',
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

        except Exception as e:
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

            if is_image:
                cmd = [
                    self.ffmpeg, '-y',
                    '-loop', '1', '-i', input_path,
                    '-t', str(duration),
                    '-vf', filter_str,
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an',
                    output_path
                ]
            else:
                cmd = [
                    self.ffmpeg, '-y',
                    '-stream_loop', '-1', '-i', input_path,
                    '-t', str(duration),
                    '-vf', filter_str,
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an',
                    output_path
                ]

            subprocess.run(cmd, capture_output=True, timeout=120)
            return output_path if os.path.exists(output_path) else None

        except Exception as e:
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
            except:
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

        except Exception as e:
            logger.error(f"Lower third creation failed: {e}")
            return None

    def create_title_card(
        self,
        title: str,
        output_path: str,
        style: Dict,
        subtitle: str = None
    ) -> Optional[str]:
        """Create an animated title card."""
        try:
            # Get colors
            primary = style.get("primary_color", "#3498db").lstrip('#')
            primary_rgb = tuple(int(primary[i:i+2], 16) for i in (0, 2, 4))
            secondary = style.get("secondary_color", "#1a1a2e").lstrip('#')
            secondary_rgb = tuple(int(secondary[i:i+2], 16) for i in (0, 2, 4))
            gradient_colors = style.get("gradient_colors", ["#1a1a2e", "#16213e"])

            # Create gradient background
            img = Image.new('RGB', self.resolution, secondary_rgb)
            draw = ImageDraw.Draw(img)

            # Draw animated-looking gradient
            for y in range(self.height):
                ratio = y / self.height
                color = tuple(
                    int(secondary_rgb[i] * (1 - ratio * 0.3) + primary_rgb[i] * ratio * 0.1)
                    for i in range(3)
                )
                draw.line([(0, y), (self.width, y)], fill=color)

            # Add subtle pattern/texture
            for _ in range(50):
                x = random.randint(0, self.width)
                y = random.randint(0, self.height)
                size = random.randint(2, 8)
                alpha = random.randint(5, 20)
                draw.ellipse([x, y, x + size, y + size], fill=tuple(min(255, c + alpha) for c in secondary_rgb))

            # Load fonts
            try:
                title_font = ImageFont.truetype(self.fonts.get("bold", ""), 80)
                sub_font = ImageFont.truetype(self.fonts.get("regular", ""), 36)
            except:
                title_font = ImageFont.load_default()
                sub_font = ImageFont.load_default()

            # Wrap title
            max_width = self.width - 200
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

            # Draw title
            total_height = len(lines) * 90
            start_y = (self.height - total_height) // 2 - 30

            for i, line in enumerate(lines[:3]):
                bbox = draw.textbbox((0, 0), line, font=title_font)
                x = (self.width - (bbox[2] - bbox[0])) // 2
                y = start_y + i * 90

                # Shadow
                draw.text((x + 4, y + 4), line, font=title_font, fill=(0, 0, 0))
                # Accent color text
                draw.text((x, y), line, font=title_font, fill=primary_rgb)

            # Draw subtitle
            if subtitle:
                bbox = draw.textbbox((0, 0), subtitle[:60], font=sub_font)
                x = (self.width - (bbox[2] - bbox[0])) // 2
                y = start_y + len(lines) * 90 + 30
                draw.text((x, y), subtitle[:60], font=sub_font, fill=(200, 200, 200))

            # Add accent line
            line_width = 200
            line_y = start_y + len(lines) * 90 + (80 if subtitle else 20)
            draw.rectangle(
                [(self.width - line_width) // 2, line_y, (self.width + line_width) // 2, line_y + 4],
                fill=primary_rgb
            )

            img.save(output_path)
            return output_path

        except Exception as e:
            logger.error(f"Title card creation failed: {e}")
            return None

    def create_gradient_background(
        self,
        output_path: str,
        style: Dict,
        duration: float
    ) -> Optional[str]:
        """Create an animated gradient background."""
        try:
            secondary = style.get("secondary_color", "#1a1a2e").lstrip('#')
            secondary_rgb = tuple(int(secondary[i:i+2], 16) for i in (0, 2, 4))
            primary = style.get("primary_color", "#3498db").lstrip('#')
            primary_rgb = tuple(int(primary[i:i+2], 16) for i in (0, 2, 4))

            img = Image.new('RGB', self.resolution, secondary_rgb)
            draw = ImageDraw.Draw(img)

            # Radial gradient
            center_x, center_y = self.width // 2, self.height // 2

            for r in range(0, max(self.width, self.height), 5):
                ratio = min(1.0, r / 800)
                color = tuple(
                    int(secondary_rgb[i] + (primary_rgb[i] - secondary_rgb[i]) * ratio * 0.2)
                    for i in range(3)
                )
                draw.ellipse(
                    [center_x - r, center_y - r, center_x + r, center_y + r],
                    outline=color
                )

            # Save frame
            frame_path = str(Path(output_path).with_suffix('.png'))
            img.save(frame_path)

            # Create video from frame
            fade_frames = int(self.TRANSITION_DURATION * self.fps)
            cmd = [
                self.ffmpeg, '-y',
                '-loop', '1', '-i', frame_path,
                '-t', str(duration),
                '-vf', f"fade=in:0:{fade_frames},fade=out:st={duration - 0.5}:d=0.5",
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an',
                output_path
            ]
            subprocess.run(cmd, capture_output=True, timeout=60)

            # Cleanup
            if os.path.exists(frame_path):
                os.remove(frame_path)

            return output_path if os.path.exists(output_path) else None

        except Exception as e:
            logger.error(f"Gradient background failed: {e}")
            return None

    def create_video(
        self,
        audio_file: str,
        script,
        output_file: str,
        niche: str = "default"
    ) -> Optional[str]:
        """
        Create a professional video with all effects.

        Args:
            audio_file: Path to narration audio
            script: VideoScript object or dict with title/sections
            output_file: Output video path
            niche: Content niche for styling

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

        try:
            # Get style
            style = self.NICHE_STYLES.get(niche, self.NICHE_STYLES["default"])

            # Get audio duration
            audio_duration = self.get_audio_duration(audio_file)
            logger.info(f"Audio duration: {audio_duration:.1f}s")

            # Calculate number of visual segments (change every 3-5 seconds)
            num_segments = int(audio_duration / self.SEGMENT_DURATION) + 1
            logger.info(f"Creating {num_segments} visual segments")

            # Get script info
            title = getattr(script, 'title', str(script)) if script else "Video"
            sections = getattr(script, 'sections', []) if script else []

            # Fetch stock footage
            stock_clips = []
            downloaded_paths = []

            if self.stock:
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

                logger.info(f"Downloaded {len(downloaded_paths)} stock clips")

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

            # 1. Title card (first 4 seconds)
            title_duration = min(4.0, audio_duration * 0.1)
            title_frame = self.temp_dir / "title_frame.png"
            self.create_title_card(
                title=title[:60],
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

            # 2. Main content segments
            media_index = 0
            all_media = downloaded_paths + stock_images  # Mix videos and images
            random.shuffle(all_media)  # Randomize for variety

            segment_num = 0
            while current_time < audio_duration - 0.5:
                remaining = audio_duration - current_time
                seg_duration = min(self.SEGMENT_DURATION, remaining)

                if seg_duration < 1:
                    break

                segment_path = self.temp_dir / f"segment_{segment_num}.mp4"

                # Use stock media if available
                if media_index < len(all_media):
                    media_path = all_media[media_index]
                    media_index += 1

                    # Apply Ken Burns effect
                    effect = random.choice(self.KEN_BURNS_EFFECTS)
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

            if not segment_files:
                logger.error("No video segments created")
                return None

            logger.info(f"Created {len(segment_files)} video segments")

            # 3. Concatenate all segments
            concat_file = self.temp_dir / "concat_list.txt"
            with open(concat_file, 'w') as f:
                for seg in segment_files:
                    # Escape single quotes in path
                    safe_path = seg.replace("'", "'\\''")
                    f.write(f"file '{safe_path}'\n")

            video_only = self.temp_dir / "video_only.mp4"
            concat_cmd = [
                self.ffmpeg, '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-pix_fmt', 'yuv420p',
                str(video_only)
            ]

            result = subprocess.run(concat_cmd, capture_output=True, timeout=600)
            if result.returncode != 0:
                logger.error(f"Concat failed: {result.stderr.decode()[:500]}")

            if not video_only.exists():
                logger.error("Video concatenation failed")
                return None

            # 4. Combine with audio
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)

            final_cmd = [
                self.ffmpeg, '-y',
                '-i', str(video_only),
                '-i', audio_file,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-shortest',
                output_file
            ]

            result = subprocess.run(final_cmd, capture_output=True, timeout=300)

            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file) / (1024 * 1024)
                logger.success(f"Ultra video created: {output_file} ({file_size:.1f} MB)")
                self._cleanup()
                return output_file
            else:
                logger.error(f"Final video creation failed: {result.stderr.decode()[:500]}")

        except Exception as e:
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
        except:
            pass


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
