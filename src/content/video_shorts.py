"""
YouTube Shorts Video Generator

Creates vertical videos optimized for YouTube Shorts format:
- Resolution: 1080x1920 (9:16 vertical aspect ratio)
- Duration: 15-60 seconds max
- Faster pacing (visual change every 2-3 seconds)
- Larger text overlays (readable on mobile)
- More aggressive Ken Burns effect
- Center-focused composition

Usage:
    from src.content.video_shorts import ShortsVideoGenerator

    generator = ShortsVideoGenerator()
    generator.create_short(
        audio_file="narration.mp3",
        script=script_object,
        output_file="short.mp4",
        niche="finance"
    )
"""

import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
from dotenv import load_dotenv

# Load environment variables from relative paths (portable)
_env_paths = [
    Path(__file__).parent.parent.parent / "config" / ".env",
    Path.cwd() / "config" / ".env",
]
for _env_path in _env_paths:
    if _env_path.exists():
        load_dotenv(_env_path)
        break

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
except ImportError:
    raise ImportError("Please install pillow: pip install pillow")


class ShortsVideoGenerator:
    """
    YouTube Shorts video generator for vertical faceless content.

    Key differences from regular videos:
    - Vertical aspect ratio (1080x1920)
    - Shorter duration (max 60s)
    - Bigger fonts (min 60px for readability)
    - Center-focused composition
    - More dynamic transitions
    - Faster visual pacing (2-3 seconds per segment)
    """

    # Vertical resolution for Shorts (9:16 aspect ratio)
    SHORTS_WIDTH = 1080
    SHORTS_HEIGHT = 1920

    # Duration settings (research: 20-35s optimal for engagement)
    MAX_DURATION = 60  # YouTube max
    OPTIMAL_DURATION = 30  # Research-backed sweet spot
    MIN_DURATION = 15  # seconds

    # Faster pacing for Shorts (visual change every 2-4 seconds)
    SEGMENT_DURATION = 2.5

    # Transition duration (shorter for faster pacing)
    TRANSITION_DURATION = 0.3

    # Hook timing (research: 50-60% drop off in first 3s)
    HOOK_DURATION = 1.5  # First visual must grab in 1.5s

    # Safe zones to avoid YouTube UI elements (research-backed)
    # These areas are covered by YouTube UI on mobile
    SAFE_ZONE = {
        "top": 288,      # Status bar, back button
        "bottom": 672,   # Like/comment/share buttons, description
        "left": 48,      # Small margin
        "right": 192,    # Follow button area
    }

    # Text settings (research: 40-56px optimal, min 24px)
    MIN_FONT_SIZE = 56  # Adjusted per research
    TITLE_FONT_SIZE = 72  # Slightly smaller for better fit
    SUBTITLE_FONT_SIZE = 44
    CAPTION_FONT_SIZE = 48  # For burned-in captions

    # Background music (research: 15% volume under voiceover)
    MUSIC_VOLUME = 0.15

    # More aggressive Ken Burns effect presets for Shorts
    KEN_BURNS_EFFECTS = [
        # More dramatic zoom in
        {"start_scale": 1.0, "end_scale": 1.25, "start_x": 0, "end_x": 0, "start_y": 0, "end_y": 0},
        # More dramatic zoom out
        {"start_scale": 1.25, "end_scale": 1.0, "start_x": 0, "end_x": 0, "start_y": 0, "end_y": 0},
        # Faster pan right with zoom
        {"start_scale": 1.15, "end_scale": 1.2, "start_x": -80, "end_x": 80, "start_y": 0, "end_y": 0},
        # Faster pan left with zoom
        {"start_scale": 1.2, "end_scale": 1.15, "start_x": 80, "end_x": -80, "start_y": 0, "end_y": 0},
        # Dramatic zoom + pan combination
        {"start_scale": 1.0, "end_scale": 1.3, "start_x": -50, "end_x": 50, "start_y": -30, "end_y": 30},
        # Pull back reveal
        {"start_scale": 1.35, "end_scale": 1.0, "start_x": 40, "end_x": -40, "start_y": 30, "end_y": -30},
        # Vertical pan (good for vertical format)
        {"start_scale": 1.2, "end_scale": 1.2, "start_x": 0, "end_x": 0, "start_y": -60, "end_y": 60},
        # Diagonal sweep
        {"start_scale": 1.1, "end_scale": 1.25, "start_x": -40, "end_x": 40, "start_y": -40, "end_y": 40},
    ]

    # Niche-specific visual styles (adapted for vertical format)
    NICHE_STYLES = {
        "finance": {
            "primary_color": "#00d4aa",
            "secondary_color": "#0a0a14",
            "accent_color": "#ffd700",
            "text_color": "#ffffff",
            "gradient_colors": ["#0a0a14", "#0f2027", "#0a0a14"],
            "overlay_opacity": 0.5,
            "saturation": 0.9,
            "contrast": 1.15,
            "vignette": True,
            "color_grade": "teal_orange",
            "font_style": "bold"
        },
        "psychology": {
            "primary_color": "#9b59b6",
            "secondary_color": "#050510",
            "accent_color": "#e74c3c",
            "text_color": "#ffffff",
            "gradient_colors": ["#050510", "#150520", "#050510"],
            "overlay_opacity": 0.55,
            "saturation": 0.75,
            "contrast": 1.25,
            "vignette": True,
            "color_grade": "cool_dark",
            "font_style": "bold"
        },
        "storytelling": {
            "primary_color": "#e74c3c",
            "secondary_color": "#080808",
            "accent_color": "#f39c12",
            "text_color": "#ffffff",
            "gradient_colors": ["#080808", "#150808", "#080808"],
            "overlay_opacity": 0.6,
            "saturation": 0.65,
            "contrast": 1.35,
            "vignette": True,
            "color_grade": "cinematic_dark",
            "font_style": "bold"
        },
        "default": {
            "primary_color": "#3498db",
            "secondary_color": "#0a0a14",
            "accent_color": "#e74c3c",
            "text_color": "#ffffff",
            "gradient_colors": ["#0a0a14", "#101020", "#0a0a14"],
            "overlay_opacity": 0.45,
            "saturation": 1.0,
            "contrast": 1.1,
            "vignette": True,
            "color_grade": "neutral",
            "font_style": "bold"
        }
    }

    def __init__(
        self,
        resolution: Tuple[int, int] = None,
        fps: int = 30
    ):
        """
        Initialize the Shorts video generator.

        Args:
            resolution: Override resolution (default: 1080x1920 vertical)
            fps: Frames per second (default: 30)
        """
        self.resolution = resolution or (self.SHORTS_WIDTH, self.SHORTS_HEIGHT)
        self.width, self.height = self.resolution
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
        self.temp_dir = Path(tempfile.gettempdir()) / "video_shorts"
        self.temp_dir.mkdir(exist_ok=True)

        # Load fonts
        self.fonts = self._load_fonts()

        logger.info(f"ShortsVideoGenerator initialized ({self.width}x{self.height} @ {self.fps}fps)")

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
            "impact": ["C:\\Windows\\Fonts\\impact.ttf", "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf"],
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
            file_size = os.path.getsize(audio_file)
            return file_size / 16000

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

        return os.path.getsize(audio_file) / 16000

    def crop_to_vertical(
        self,
        input_path: str,
        output_path: str,
        duration: float
    ) -> Optional[str]:
        """
        Crop horizontal footage to vertical format (center crop).

        Args:
            input_path: Input video or image
            output_path: Output video path
            duration: Target duration

        Returns:
            Path to cropped clip or None
        """
        if not self.ffmpeg or not os.path.exists(input_path):
            return None

        try:
            is_image = input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))

            # Center crop filter for vertical format
            # Crop to 9:16 aspect ratio from center
            crop_filter = f"scale=-1:{self.height*2},crop={self.width}:{self.height}"

            if is_image:
                cmd = [
                    self.ffmpeg, '-y',
                    '-loop', '1',
                    '-i', input_path,
                    '-t', str(duration),
                    '-vf', crop_filter,
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-an',
                    output_path
                ]
            else:
                cmd = [
                    self.ffmpeg, '-y',
                    '-i', input_path,
                    '-t', str(duration),
                    '-vf', crop_filter,
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-an',
                    output_path
                ]

            subprocess.run(cmd, capture_output=True, timeout=120)

            if os.path.exists(output_path):
                return output_path

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as e:
            logger.error(f"Crop to vertical failed: {e}")

        return None

    def create_ken_burns_clip(
        self,
        input_path: str,
        output_path: str,
        duration: float,
        effect_preset: Dict = None,
        style: Dict = None
    ) -> Optional[str]:
        """
        Create a clip with aggressive Ken Burns effect for Shorts.

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
            # Choose random effect if not specified (from more aggressive presets)
            effect = effect_preset or random.choice(self.KEN_BURNS_EFFECTS)
            style = style or self.NICHE_STYLES["default"]

            is_image = input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))

            start_scale = effect["start_scale"]
            end_scale = effect["end_scale"]
            start_x = effect["start_x"]
            end_x = effect["end_x"]
            start_y = effect["start_y"]
            end_y = effect["end_y"]

            frames = int(duration * self.fps)

            filters = []

            # Scale input larger for zoompan (need more resolution for cropping)
            filters.append(f"scale=4320:-1")

            # Apply Ken Burns zoompan effect (optimized for vertical)
            filters.append(
                f"zoompan=z='{start_scale}+({end_scale}-{start_scale})*on/{frames}':"
                f"x='iw/2-(iw/zoom/2)+{start_x}+({end_x}-{start_x})*on/{frames}':"
                f"y='ih/2-(ih/zoom/2)+{start_y}+({end_y}-{start_y})*on/{frames}':"
                f"d={frames}:s={self.width}x{self.height}:fps={self.fps}"
            )

            # Color grading based on niche (slightly more aggressive for Shorts)
            if style.get("color_grade") == "teal_orange":
                filters.append("colorbalance=rs=0.12:gs=-0.06:bs=-0.12:rm=0.06:gm=0:bm=-0.06")
            elif style.get("color_grade") == "cool_dark":
                filters.append("colorbalance=rs=-0.12:gs=-0.06:bs=0.12")
                filters.append("eq=brightness=-0.06:saturation=0.75")
            elif style.get("color_grade") == "cinematic_dark":
                filters.append("colorbalance=rs=0.06:gs=-0.06:bs=-0.06")
                filters.append("eq=brightness=-0.12:contrast=1.25:saturation=0.65")

            # Saturation and contrast
            sat = style.get("saturation", 1.0)
            con = style.get("contrast", 1.0)
            if sat != 1.0 or con != 1.0:
                filters.append(f"eq=saturation={sat}:contrast={con}")

            # Vignette effect (more pronounced for vertical)
            if style.get("vignette"):
                filters.append("vignette=PI/3.5")

            # Darken for text visibility
            overlay_opacity = style.get("overlay_opacity", 0.45)
            if overlay_opacity > 0:
                filters.append(f"eq=brightness=-{overlay_opacity * 0.55}")

            # Faster fade in/out for Shorts
            fade_frames = int(self.TRANSITION_DURATION * self.fps)
            filters.append(f"fade=in:0:{fade_frames}")
            filters.append(f"fade=out:st={duration - self.TRANSITION_DURATION}:d={self.TRANSITION_DURATION}")

            filter_str = ",".join(filters)

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
                # For video, use simpler crop approach
                cmd = [
                    self.ffmpeg, '-y',
                    '-i', input_path,
                    '-t', str(duration),
                    '-vf', f"scale=-1:{int(self.height*1.5)},"
                           f"crop={self.width}:{self.height},"
                           f"eq=saturation={sat}:contrast={con}:brightness=-{overlay_opacity * 0.35},"
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
            logger.warning("Ken Burns failed, using simple crop")
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
        """Create a simple cropped clip (fallback)."""
        try:
            is_image = input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))

            # Center crop for vertical format
            filters = [
                f"scale=-1:{int(self.height*1.5)}",
                f"crop={self.width}:{self.height}",
            ]

            # Color adjustments
            sat = style.get("saturation", 1.0)
            con = style.get("contrast", 1.0)
            overlay = style.get("overlay_opacity", 0.45)
            filters.append(f"eq=saturation={sat}:contrast={con}:brightness=-{overlay * 0.35}")

            # Faster fade for Shorts
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

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as e:
            logger.error(f"Simple clip failed: {e}")
            return None

    def create_title_card(
        self,
        title: str,
        output_path: str,
        style: Dict,
        subtitle: str = None
    ) -> Optional[str]:
        """
        Create a clean, professional title card optimized for vertical Shorts format.

        Design principles:
        - Minimal, clean gradient background (no particles/dots)
        - Large text for mobile readability
        - Respects safe zones (top 288px, bottom 672px)
        - Clear typography hierarchy with generous spacing
        - Subtle vignette for depth
        - High contrast for text readability
        """
        try:
            primary = style.get("primary_color", "#3498db").lstrip('#')
            primary_rgb = tuple(int(primary[i:i+2], 16) for i in (0, 2, 4))
            secondary = style.get("secondary_color", "#0a0a14").lstrip('#')
            secondary_rgb = tuple(int(secondary[i:i+2], 16) for i in (0, 2, 4))

            # Create base image
            img = Image.new('RGB', self.resolution, secondary_rgb)
            draw = ImageDraw.Draw(img)

            # Draw clean vertical gradient (subtle, dark to slightly lighter)
            for y in range(self.height):
                ratio = y / self.height
                # Subtle curve for natural gradient
                curve = ratio * ratio
                color = tuple(
                    int(secondary_rgb[i] + (primary_rgb[i] - secondary_rgb[i]) * curve * 0.06)
                    for i in range(3)
                )
                draw.line([(0, y), (self.width, y)], fill=color)

            # Add subtle radial vignette for depth
            center_x, center_y = self.width // 2, self.height // 2
            max_radius = int((self.width**2 + self.height**2) ** 0.5 / 2)

            # Draw vignette as concentric ellipses (darkens edges)
            for r in range(max_radius, 0, -6):
                intensity = int(255 * (1 - (r / max_radius) ** 1.8))
                draw.ellipse(
                    [center_x - r, center_y - int(r * 1.8), center_x + r, center_y + int(r * 1.8)],
                    outline=(intensity, intensity, intensity)
                )

            # Load fonts (larger for mobile Shorts viewing)
            try:
                title_font = ImageFont.truetype(self.fonts.get("bold", ""), self.TITLE_FONT_SIZE)
                sub_font = ImageFont.truetype(self.fonts.get("regular", ""), self.SUBTITLE_FONT_SIZE)
            except (OSError, IOError) as e:
                logger.debug(f"Font loading failed, using default: {e}")
                title_font = ImageFont.load_default()
                sub_font = ImageFont.load_default()

            # Calculate safe content area (respecting YouTube UI overlays)
            safe_top = self.SAFE_ZONE["top"]
            safe_bottom = self.height - self.SAFE_ZONE["bottom"]
            safe_center_y = (safe_top + safe_bottom) // 2

            # Wrap title with generous margins for vertical format
            max_width = self.width - 100  # Generous side margins
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
            line_height = self.TITLE_FONT_SIZE + 25  # Generous line spacing
            total_text_height = len(lines[:3]) * line_height
            if subtitle:
                total_text_height += 60  # Space for subtitle

            # Center vertically within safe zone
            start_y = safe_center_y - (total_text_height // 2) - 30

            # Draw title lines with clean shadow for depth
            for i, line in enumerate(lines[:3]):  # Max 3 lines for clean look
                bbox = draw.textbbox((0, 0), line, font=title_font)
                text_width = bbox[2] - bbox[0]
                x = (self.width - text_width) // 2
                y = start_y + i * line_height

                # Soft shadow for depth
                shadow_offset = 4
                draw.text((x + shadow_offset, y + shadow_offset), line, font=title_font, fill=(0, 0, 0))

                # Main title in white for maximum readability on mobile
                draw.text((x, y), line, font=title_font, fill=(255, 255, 255))

            # Draw subtitle with proper spacing
            subtitle_y = start_y + len(lines[:3]) * line_height + 20
            if subtitle:
                sub_text = subtitle[:45]  # Shorter for vertical format
                bbox = draw.textbbox((0, 0), sub_text, font=sub_font)
                x = (self.width - (bbox[2] - bbox[0])) // 2
                # Muted color for visual hierarchy
                draw.text((x, subtitle_y), sub_text, font=sub_font, fill=(170, 170, 170))

            # Add subtle accent line below content
            accent_y = subtitle_y + (55 if subtitle else 30)
            line_width = 100  # Shorter, elegant line for vertical format
            line_thickness = 4

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

    def create_text_overlay(
        self,
        text: str,
        output_path: str,
        style: Dict,
        position: str = "center"  # center, top, bottom
    ) -> Optional[str]:
        """
        Create a text overlay image for Shorts with safe zone awareness.

        Research-backed:
        - Uses 40-56px fonts for mobile readability
        - Respects YouTube UI safe zones
        - High contrast text with stroke/shadow
        """
        try:
            # Create transparent image
            img = Image.new('RGBA', self.resolution, (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)

            primary = style.get("primary_color", "#3498db").lstrip('#')
            primary_rgb = tuple(int(primary[i:i+2], 16) for i in (0, 2, 4))

            # Load font (research: 40-56px optimal)
            try:
                font = ImageFont.truetype(self.fonts.get("bold", ""), self.MIN_FONT_SIZE)
            except (OSError, IOError) as e:
                logger.debug(f"Font loading failed, using default: {e}")
                font = ImageFont.load_default()

            # Calculate safe area for text (avoiding YouTube UI)
            safe_left = self.SAFE_ZONE["left"]
            safe_right = self.width - self.SAFE_ZONE["right"]
            safe_top = self.SAFE_ZONE["top"]
            safe_bottom = self.height - self.SAFE_ZONE["bottom"]
            safe_width = safe_right - safe_left

            # Wrap text within safe zone
            max_width = safe_width - 60  # 30px padding on each side
            words = text.split()
            lines = []
            current_line = ""

            for word in words:
                test_line = f"{current_line} {word}".strip()
                bbox = draw.textbbox((0, 0), test_line, font=font)
                if bbox[2] - bbox[0] <= max_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)

            # Calculate position within safe zone
            line_height = self.MIN_FONT_SIZE + 15
            total_height = len(lines) * line_height

            if position == "top":
                start_y = safe_top + 50  # Padding from safe zone edge
            elif position == "bottom":
                start_y = safe_bottom - total_height - 50
            else:  # center
                center_safe = (safe_top + safe_bottom) // 2
                start_y = center_safe - (total_height // 2)

            # Draw text background (semi-transparent, within safe zone)
            padding = 25
            bg_left = safe_left + 20
            bg_right = safe_right - 20
            bg_top = start_y - padding
            bg_bottom = start_y + total_height + padding
            draw.rectangle(
                [bg_left, bg_top, bg_right, bg_bottom],
                fill=(0, 0, 0, 180)
            )

            # Draw text with high contrast (research: white + black stroke)
            for i, line in enumerate(lines[:4]):  # Max 4 lines for readability
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
                x = safe_left + (safe_width - text_width) // 2
                y = start_y + i * line_height

                # Black stroke for contrast (research: essential for readability)
                stroke_width = 3
                for dx in range(-stroke_width, stroke_width + 1):
                    for dy in range(-stroke_width, stroke_width + 1):
                        if dx != 0 or dy != 0:
                            draw.text((x + dx, y + dy), line, font=font, fill=(0, 0, 0, 255))

                # Main white text
                draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))

            img.save(output_path)
            return output_path

        except (OSError, IOError, ValueError) as e:
            logger.error(f"Text overlay creation failed: {e}")
            return None

    def create_gradient_background(
        self,
        output_path: str,
        style: Dict,
        duration: float
    ) -> Optional[str]:
        """Create a gradient background clip for Shorts."""
        try:
            secondary = style.get("secondary_color", "#0a0a14").lstrip('#')
            secondary_rgb = tuple(int(secondary[i:i+2], 16) for i in (0, 2, 4))
            primary = style.get("primary_color", "#3498db").lstrip('#')
            primary_rgb = tuple(int(primary[i:i+2], 16) for i in (0, 2, 4))

            img = Image.new('RGB', self.resolution, secondary_rgb)
            draw = ImageDraw.Draw(img)

            # Radial gradient (centered for vertical)
            center_x, center_y = self.width // 2, self.height // 2

            for r in range(0, max(self.width, self.height) + 200, 5):
                ratio = min(1.0, r / 1000)
                color = tuple(
                    int(secondary_rgb[i] + (primary_rgb[i] - secondary_rgb[i]) * ratio * 0.25)
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
                '-vf', f"fade=in:0:{fade_frames},fade=out:st={duration - 0.3}:d=0.3",
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an',
                output_path
            ]
            subprocess.run(cmd, capture_output=True, timeout=60)

            # Cleanup
            if os.path.exists(frame_path):
                os.remove(frame_path)

            return output_path if os.path.exists(output_path) else None

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError, IOError) as e:
            logger.error(f"Gradient background failed: {e}")
            return None

    def create_short(
        self,
        audio_file: str,
        script,
        output_file: str,
        niche: str = "default",
        background_music: str = None,
        music_volume: float = None
    ) -> Optional[str]:
        """
        Create a YouTube Short from audio and script.

        Args:
            audio_file: Path to narration audio (15-60 seconds)
            script: VideoScript object or dict with title/sections
            output_file: Output video path
            niche: Content niche for styling
            background_music: Optional path to background music file
            music_volume: Optional music volume (0.0-1.0), defaults to 0.15

        Returns:
            Path to created short video or None
        """
        if not self.ffmpeg:
            logger.error("FFmpeg not available")
            return None

        if not os.path.exists(audio_file):
            logger.error(f"Audio file not found: {audio_file}")
            return None

        logger.info(f"Creating YouTube Short: {output_file}")
        logger.info(f"Niche: {niche}")

        try:
            # Get style
            style = self.NICHE_STYLES.get(niche, self.NICHE_STYLES["default"])

            # Get audio duration
            audio_duration = self.get_audio_duration(audio_file)
            logger.info(f"Audio duration: {audio_duration:.1f}s")

            # Validate duration for Shorts
            if audio_duration > self.MAX_DURATION:
                logger.warning(f"Audio duration ({audio_duration:.1f}s) exceeds Shorts max ({self.MAX_DURATION}s). Trimming.")
                audio_duration = self.MAX_DURATION
            elif audio_duration < self.MIN_DURATION:
                logger.warning(f"Audio duration ({audio_duration:.1f}s) is below Shorts min ({self.MIN_DURATION}s).")

            # Research: 20-35 seconds is optimal for engagement
            if audio_duration > self.OPTIMAL_DURATION:
                logger.info(f"Tip: Research shows {self.OPTIMAL_DURATION}s is optimal for Shorts engagement. "
                           f"Current: {audio_duration:.1f}s")

            # Calculate number of visual segments (faster pacing for Shorts: 2-3 seconds)
            num_segments = int(audio_duration / self.SEGMENT_DURATION) + 1
            logger.info(f"Creating {num_segments} visual segments (fast pacing)")

            # Get script info
            title = getattr(script, 'title', str(script)) if script else "Short"
            sections = getattr(script, 'sections', []) if script else []

            # Fetch stock footage
            downloaded_paths = []

            if self.stock:
                # Build search terms from script
                search_terms = [title]
                if sections:
                    for s in sections[:3]:
                        if hasattr(s, 'keywords') and s.keywords:
                            search_terms.extend(s.keywords[:2])
                        elif hasattr(s, 'title') and s.title:
                            search_terms.append(s.title)

                logger.info("Fetching stock footage for Shorts...")

                # Use multi-stock provider if available
                if hasattr(self.stock, 'get_clips_for_topic'):
                    clips = self.stock.get_clips_for_topic(
                        topic=title,
                        niche=niche,
                        count=num_segments + 3,
                        min_total_duration=int(audio_duration * 1.2)
                    )
                    for clip in clips:
                        path = self.stock.download_clip(clip)
                        if path:
                            downloaded_paths.append(path)
                else:
                    # Fallback to basic stock
                    for term in search_terms[:3]:
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
                images = self.stock.search_images(title, count=3)
                for img in images:
                    if hasattr(self.stock, 'download_image'):
                        path = self.stock.download_image(img)
                        if path:
                            stock_images.append(path)

            # Create video segments
            segment_files = []
            current_time = 0.0

            # 1. Title card / Hook (research: must grab attention in 1.5s)
            title_duration = self.HOOK_DURATION  # Fast hook per research
            title_frame = self.temp_dir / "title_frame.png"
            self.create_title_card(
                title=title[:40],
                output_path=str(title_frame),
                style=style,
                subtitle=getattr(script, 'description', '')[:30] if hasattr(script, 'description') else None
            )

            if title_frame.exists():
                title_video = self.temp_dir / "title.mp4"
                self._create_simple_clip(str(title_frame), str(title_video), title_duration, style)
                if title_video.exists():
                    segment_files.append(str(title_video))
                    current_time += title_duration

            # 2. Main content segments (faster pacing)
            media_index = 0
            all_media = downloaded_paths + stock_images
            random.shuffle(all_media)

            segment_num = 0
            while current_time < audio_duration - 0.3:
                remaining = audio_duration - current_time
                seg_duration = min(self.SEGMENT_DURATION, remaining)

                if seg_duration < 0.5:
                    break

                segment_path = self.temp_dir / f"segment_{segment_num}.mp4"

                # Use stock media if available
                if media_index < len(all_media):
                    media_path = all_media[media_index]
                    media_index += 1

                    # Apply aggressive Ken Burns effect for Shorts
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

            result = subprocess.run(concat_cmd, capture_output=True, timeout=300)
            if result.returncode != 0:
                logger.error(f"Concat failed: {result.stderr.decode()[:500]}")

            if not video_only.exists():
                logger.error("Video concatenation failed")
                return None

            # 4. Combine with audio (trim to max duration if needed)
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)

            final_cmd = [
                self.ffmpeg, '-y',
                '-i', str(video_only),
                '-i', audio_file,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-t', str(min(audio_duration, self.MAX_DURATION)),
                '-shortest',
                output_file
            ]

            result = subprocess.run(final_cmd, capture_output=True, timeout=300)

            if os.path.exists(output_file):
                # Add background music if provided
                music_path = background_music or self.get_niche_music_path(niche)
                if music_path:
                    logger.info("Adding background music to Short...")
                    video_with_music = self.temp_dir / "short_with_music.mp4"
                    music_result = self.add_background_music(
                        output_file,
                        music_path,
                        str(video_with_music),
                        music_volume
                    )
                    if music_result and os.path.exists(str(video_with_music)):
                        # Replace output with music version
                        shutil.move(str(video_with_music), output_file)

                file_size = os.path.getsize(output_file) / (1024 * 1024)
                logger.success(f"YouTube Short created: {output_file} ({file_size:.1f} MB)")
                self._cleanup()
                return output_file
            else:
                logger.error(f"Final video creation failed: {result.stderr.decode()[:500]}")

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError, IOError) as e:
            logger.error(f"Short creation failed: {e}")
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

    def add_background_music(
        self,
        video_file: str,
        music_file: str,
        output_file: str,
        music_volume: float = None
    ) -> Optional[str]:
        """
        Add background music to a video at low volume.

        Research: Background music at 15% volume improves engagement
        without competing with voiceover.

        Args:
            video_file: Input video path
            music_file: Background music path (will be looped if needed)
            output_file: Output video path
            music_volume: Volume level (default: 0.15 per research)

        Returns:
            Path to output video or None
        """
        if not self.ffmpeg or not os.path.exists(video_file):
            return None

        if not os.path.exists(music_file):
            logger.warning(f"Music file not found: {music_file}")
            return None

        volume = music_volume or self.MUSIC_VOLUME

        try:
            # Get video duration
            duration = self.get_audio_duration(video_file)

            # Mix audio: original at full volume, music at 15%
            cmd = [
                self.ffmpeg, '-y',
                '-i', video_file,
                '-stream_loop', '-1',  # Loop music if needed
                '-i', music_file,
                '-t', str(duration),
                '-filter_complex',
                f'[0:a]volume=1.0[a1];'
                f'[1:a]volume={volume}[a2];'
                f'[a1][a2]amix=inputs=2:duration=first[aout]',
                '-map', '0:v',
                '-map', '[aout]',
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-b:a', '192k',
                output_file
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=300)

            if os.path.exists(output_file):
                logger.info(f"Added background music at {volume*100:.0f}% volume")
                return output_file
            else:
                logger.error(f"Failed to add music: {result.stderr.decode()[:200]}")

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as e:
            logger.error(f"Add music failed: {e}")

        return None

    def get_safe_text_area(self) -> Dict[str, int]:
        """
        Get the safe area for text placement.

        Returns coordinates where text won't be obscured by YouTube UI.

        Returns:
            Dict with top, bottom, left, right, width, height of safe area
        """
        safe_left = self.SAFE_ZONE["left"]
        safe_right = self.width - self.SAFE_ZONE["right"]
        safe_top = self.SAFE_ZONE["top"]
        safe_bottom = self.height - self.SAFE_ZONE["bottom"]

        return {
            "top": safe_top,
            "bottom": safe_bottom,
            "left": safe_left,
            "right": safe_right,
            "width": safe_right - safe_left,
            "height": safe_bottom - safe_top,
            "center_x": self.width // 2,
            "center_y": (safe_top + safe_bottom) // 2,
        }


# Test
if __name__ == "__main__":
    print("\n" + "="*60)
    print("YOUTUBE SHORTS GENERATOR TEST")
    print("="*60 + "\n")

    gen = ShortsVideoGenerator()
    print(f"FFmpeg: {gen.ffmpeg}")
    print(f"FFprobe: {gen.ffprobe}")
    print(f"Stock provider: {type(gen.stock).__name__ if gen.stock else 'None'}")
    print(f"Resolution: {gen.width}x{gen.height} (9:16 vertical)")
    print(f"Max duration: {gen.MAX_DURATION}s")
    print(f"Segment duration: {gen.SEGMENT_DURATION}s (fast pacing)")

    # Test title card
    style = gen.NICHE_STYLES["finance"]
    gen.create_title_card(
        "5 Money Secrets",
        "output/test_short_title.png",
        style,
        "Rich People Know"
    )
    print("\nShorts title card created: output/test_short_title.png")
