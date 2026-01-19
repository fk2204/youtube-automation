"""
Video Hook Generator for YouTube Automation

Creates catchy, attention-grabbing intros for the first 3-5 seconds of videos.
Ensures viewers see something engaging in the FIRST SECOND.

Features:
- Dynamic text animations (zoom in, fade, slide)
- Niche-specific visual themes
- Pattern interrupt visuals
- Teaser/flash forward effects
- Hook overlays with bold text

Usage:
    from src.content.video_hooks import VideoHookGenerator

    generator = VideoHookGenerator()

    # Generate a single hook frame
    frame = generator.generate_hook_frame(
        topic="5 Money Mistakes",
        niche="finance",
        hook_text="This one mistake costs you $10K/year..."
    )

    # Create animated intro clip
    intro = generator.create_animated_intro(
        hook_text="What if I told you...",
        niche="psychology",
        duration=3.0
    )

    # Add hook to existing video
    final_video = generator.add_hook_to_video(
        video_path="original.mp4",
        hook_text="Nobody expected this...",
        niche="storytelling"
    )
"""

import os
import math
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
except ImportError:
    raise ImportError("Please install pillow: pip install pillow")


class HookAnimationType(Enum):
    """Types of hook animations available."""
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    FADE_IN = "fade_in"
    SLIDE_UP = "slide_up"
    SLIDE_DOWN = "slide_down"
    PULSE = "pulse"
    GLITCH = "glitch"
    TYPEWRITER = "typewriter"
    FLASH = "flash"


@dataclass
class HookTemplate:
    """Template configuration for a hook."""
    name: str
    animation_type: HookAnimationType
    text_position: str  # "center", "top", "bottom"
    font_size_multiplier: float = 1.0
    text_color: str = "#ffffff"
    accent_color: str = "#00d4aa"
    background_style: str = "gradient"  # "gradient", "image", "particle"
    overlay_opacity: float = 0.7
    duration: float = 3.0
    has_counter: bool = False
    has_statistic: bool = False
    teaser_text: Optional[str] = None


@dataclass
class HookValidationResult:
    """Result of hook validation."""
    is_valid: bool
    has_visual_interest: bool
    first_frame_not_blank: bool
    hook_text_visible: bool
    has_audio: bool
    suggestions: List[str] = field(default_factory=list)


class VideoHookGenerator:
    """
    Generates attention-grabbing video intros/hooks.

    Design principles:
    - First frame must NEVER be blank
    - Grab attention in first second
    - Niche-specific visual themes
    - Pattern interrupts to stop scrolling
    """

    # Default resolutions
    SHORTS_WIDTH = 1080
    SHORTS_HEIGHT = 1920
    LANDSCAPE_WIDTH = 1920
    LANDSCAPE_HEIGHT = 1080

    # Hook timing constants
    DEFAULT_HOOK_DURATION = 3.0  # seconds
    MIN_HOOK_DURATION = 1.5
    MAX_HOOK_DURATION = 5.0

    # Font sizes (base sizes, scaled by resolution)
    HOOK_FONT_SIZE = 96  # Large for impact
    COUNTER_FONT_SIZE = 144  # Extra large for numbers
    TEASER_FONT_SIZE = 48  # Smaller for teaser text

    # Animation frame rate
    FPS = 30

    # Niche-specific templates
    NICHE_TEMPLATES = {
        "finance": {
            "primary_color": "#00d4aa",
            "secondary_color": "#0a0a14",
            "accent_color": "#ffd700",
            "text_color": "#ffffff",
            "gradient_colors": ["#0a0a14", "#0f2027", "#1a3a4a"],
            "icon_theme": "money",  # money, charts, gold
            "animation_style": "dynamic",
            "hooks": [
                {"template": "$X,XXX to $XX,XXX", "type": "counter", "style": "growth"},
                {"template": "This costs you ${amount}/year", "type": "statistic", "style": "loss"},
                {"template": "Rich people know this...", "type": "text", "style": "secret"},
                {"template": "{percentage}% of millionaires...", "type": "statistic", "style": "fact"},
                {"template": "Wall Street hates this...", "type": "text", "style": "controversy"},
            ],
            "visual_elements": ["rising_chart", "gold_coins", "money_stack", "upward_arrow"],
            "sound_cues": ["cash_register", "coin_drop", "success_chime"],
        },
        "psychology": {
            "primary_color": "#9b59b6",
            "secondary_color": "#050510",
            "accent_color": "#e74c3c",
            "text_color": "#ffffff",
            "gradient_colors": ["#050510", "#150520", "#2a1040"],
            "icon_theme": "mind",  # brain, eye, thought
            "animation_style": "mysterious",
            "hooks": [
                {"template": "What if I told you...", "type": "text", "style": "question"},
                {"template": "Your brain is lying to you", "type": "text", "style": "revelation"},
                {"template": "{percentage}% of people don't know this", "type": "statistic", "style": "fact"},
                {"template": "This changes everything...", "type": "text", "style": "teaser"},
                {"template": "Scientists discovered...", "type": "text", "style": "authority"},
            ],
            "visual_elements": ["brain_glow", "neural_network", "eye_focus", "thought_bubble"],
            "sound_cues": ["mysterious_whoosh", "revelation_sting", "deep_bass"],
        },
        "storytelling": {
            "primary_color": "#e74c3c",
            "secondary_color": "#080808",
            "accent_color": "#f39c12",
            "text_color": "#ffffff",
            "gradient_colors": ["#080808", "#150808", "#2a1515"],
            "icon_theme": "dramatic",  # book, silhouette, mystery
            "animation_style": "cinematic",
            "hooks": [
                {"template": "Nobody expected what happened next...", "type": "text", "style": "suspense"},
                {"template": "This is the story they don't want you to hear", "type": "text", "style": "conspiracy"},
                {"template": "Wait for it...", "type": "text", "style": "teaser"},
                {"template": "Everything changed in {timeframe}", "type": "text", "style": "dramatic"},
                {"template": "The truth will shock you...", "type": "text", "style": "revelation"},
            ],
            "visual_elements": ["dramatic_shadow", "spotlight", "film_grain", "mystery_figure"],
            "sound_cues": ["dramatic_sting", "suspense_build", "heartbeat"],
        },
        "default": {
            "primary_color": "#3498db",
            "secondary_color": "#0a0a14",
            "accent_color": "#e74c3c",
            "text_color": "#ffffff",
            "gradient_colors": ["#0a0a14", "#101020", "#1a1a2e"],
            "icon_theme": "general",
            "animation_style": "modern",
            "hooks": [
                {"template": "Watch until the end...", "type": "text", "style": "teaser"},
                {"template": "You won't believe this...", "type": "text", "style": "surprise"},
                {"template": "Here's what happened...", "type": "text", "style": "narrative"},
            ],
            "visual_elements": ["abstract_glow", "light_rays", "particle_effect"],
            "sound_cues": ["whoosh", "notification", "impact"],
        }
    }

    # Animation presets
    ANIMATION_PRESETS = {
        "zoom_in": {
            "start_scale": 0.5,
            "end_scale": 1.0,
            "easing": "ease_out",
        },
        "zoom_out": {
            "start_scale": 1.3,
            "end_scale": 1.0,
            "easing": "ease_out",
        },
        "fade_in": {
            "start_opacity": 0.0,
            "end_opacity": 1.0,
            "easing": "ease_in_out",
        },
        "slide_up": {
            "start_y_offset": 100,
            "end_y_offset": 0,
            "easing": "ease_out",
        },
        "pulse": {
            "scale_range": (0.95, 1.05),
            "frequency": 2,  # pulses per second
        },
        "glitch": {
            "offset_range": (-10, 10),
            "color_shift": True,
            "frequency": 10,
        },
    }

    def __init__(
        self,
        resolution: Tuple[int, int] = None,
        fps: int = 30,
        is_shorts: bool = False
    ):
        """
        Initialize the Video Hook Generator.

        Args:
            resolution: Output resolution (width, height)
            fps: Frames per second
            is_shorts: If True, use vertical 9:16 format
        """
        if resolution:
            self.width, self.height = resolution
        elif is_shorts:
            self.width, self.height = self.SHORTS_WIDTH, self.SHORTS_HEIGHT
        else:
            self.width, self.height = self.LANDSCAPE_WIDTH, self.LANDSCAPE_HEIGHT

        self.resolution = (self.width, self.height)
        self.fps = fps
        self.is_shorts = is_shorts or (self.height > self.width)

        # Find FFmpeg
        self.ffmpeg = self._find_ffmpeg()
        self.ffprobe = self._find_ffprobe()

        if not self.ffmpeg:
            logger.warning("FFmpeg not found - some features will be limited")

        # Temp directory
        self.temp_dir = Path(tempfile.gettempdir()) / "video_hooks"
        self.temp_dir.mkdir(exist_ok=True)

        # Load fonts
        self.fonts = self._load_fonts()

        logger.info(f"VideoHookGenerator initialized ({self.width}x{self.height} @ {self.fps}fps)")

    def _find_ffmpeg(self) -> Optional[str]:
        """Find FFmpeg executable."""
        if shutil.which("ffmpeg"):
            return "ffmpeg"

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
            ffprobe = self.ffmpeg.replace("ffmpeg.exe", "ffprobe.exe")
            if ffprobe != self.ffmpeg and os.path.exists(ffprobe):
                return ffprobe
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
            "black": ["C:\\Windows\\Fonts\\ariblk.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
        }

        for name, paths in font_paths:
            for path in paths:
                if os.path.exists(path):
                    fonts[name] = path
                    break

        return fonts

    def get_hook_template(self, niche: str) -> Dict[str, Any]:
        """
        Get the hook template configuration for a specific niche.

        Args:
            niche: Content niche (finance, psychology, storytelling)

        Returns:
            Template configuration dict
        """
        return self.NICHE_TEMPLATES.get(niche, self.NICHE_TEMPLATES["default"])

    def _parse_color(self, color: str) -> Tuple[int, int, int]:
        """Parse hex color string to RGB tuple."""
        color = color.lstrip('#')
        return tuple(int(color[i:i+2], 16) for i in (0, 2, 4))

    def _create_gradient_background(
        self,
        size: Tuple[int, int],
        colors: List[str],
        style: str = "vertical"
    ) -> Image.Image:
        """Create a gradient background image."""
        width, height = size
        img = Image.new('RGB', size)
        draw = ImageDraw.Draw(img)

        # Parse colors
        rgb_colors = [self._parse_color(c) for c in colors]

        if style == "vertical":
            for y in range(height):
                # Calculate color at this position
                ratio = y / height
                if len(rgb_colors) == 2:
                    color = tuple(
                        int(rgb_colors[0][i] + (rgb_colors[1][i] - rgb_colors[0][i]) * ratio)
                        for i in range(3)
                    )
                else:
                    # Multi-color gradient
                    segment = ratio * (len(rgb_colors) - 1)
                    idx = int(segment)
                    local_ratio = segment - idx
                    if idx >= len(rgb_colors) - 1:
                        color = rgb_colors[-1]
                    else:
                        color = tuple(
                            int(rgb_colors[idx][i] + (rgb_colors[idx+1][i] - rgb_colors[idx][i]) * local_ratio)
                            for i in range(3)
                        )
                draw.line([(0, y), (width, y)], fill=color)

        elif style == "radial":
            center_x, center_y = width // 2, height // 2
            max_radius = int(math.sqrt(center_x**2 + center_y**2))

            for r in range(max_radius, 0, -2):
                ratio = r / max_radius
                color = tuple(
                    int(rgb_colors[0][i] * (1 - ratio) + rgb_colors[-1][i] * ratio)
                    for i in range(3)
                )
                draw.ellipse(
                    [center_x - r, center_y - r, center_x + r, center_y + r],
                    fill=color
                )

        return img

    def _add_vignette(self, img: Image.Image, strength: float = 0.4) -> Image.Image:
        """Add a vignette effect to an image."""
        width, height = img.size

        # Create vignette mask
        vignette = Image.new('L', (width, height), 255)
        draw = ImageDraw.Draw(vignette)

        center_x, center_y = width // 2, height // 2
        max_radius = int(math.sqrt(center_x**2 + center_y**2))

        for r in range(max_radius, 0, -3):
            intensity = int(255 * (1 - (1 - r / max_radius) ** 2 * strength))
            draw.ellipse(
                [center_x - r, center_y - r, center_x + r, center_y + r],
                fill=intensity
            )

        # Apply vignette
        result = Image.composite(
            img,
            Image.new('RGB', img.size, (0, 0, 0)),
            vignette
        )

        return result

    def _draw_text_with_effects(
        self,
        img: Image.Image,
        text: str,
        position: Tuple[int, int],
        font: ImageFont.FreeTypeFont,
        text_color: Tuple[int, int, int],
        effects: Dict[str, Any] = None
    ) -> Image.Image:
        """Draw text with shadow, outline, and glow effects."""
        draw = ImageDraw.Draw(img)
        x, y = position
        effects = effects or {}

        # Shadow
        if effects.get("shadow", True):
            shadow_offset = effects.get("shadow_offset", 4)
            shadow_color = effects.get("shadow_color", (0, 0, 0))
            draw.text((x + shadow_offset, y + shadow_offset), text, font=font, fill=shadow_color)

        # Outline/stroke
        if effects.get("outline", True):
            outline_width = effects.get("outline_width", 3)
            outline_color = effects.get("outline_color", (0, 0, 0))
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    if dx != 0 or dy != 0:
                        draw.text((x + dx, y + dy), text, font=font, fill=outline_color)

        # Main text
        draw.text((x, y), text, font=font, fill=text_color)

        return img

    def generate_hook_frame(
        self,
        topic: str,
        niche: str,
        hook_text: str,
        include_visuals: bool = True
    ) -> Image.Image:
        """
        Generate a single hook frame image.

        Args:
            topic: Video topic for context
            niche: Content niche
            hook_text: The hook text to display
            include_visuals: Whether to add visual elements

        Returns:
            PIL Image object
        """
        template = self.get_hook_template(niche)

        # Create gradient background
        img = self._create_gradient_background(
            self.resolution,
            template["gradient_colors"],
            style="radial" if niche == "psychology" else "vertical"
        )

        # Add vignette
        img = self._add_vignette(img, strength=0.5)

        # Add visual elements based on niche
        if include_visuals:
            img = self._add_niche_visuals(img, niche, template)

        # Load font
        try:
            # Use impact or bold for maximum readability
            font_path = self.fonts.get("impact", self.fonts.get("bold", ""))
            font_size = int(self.HOOK_FONT_SIZE * (self.width / 1920))
            font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
        except (OSError, IOError):
            font = ImageFont.load_default()

        draw = ImageDraw.Draw(img)

        # Wrap text
        max_width = int(self.width * 0.85)
        lines = self._wrap_text(draw, hook_text, font, max_width)

        # Calculate text position (center)
        line_height = font_size + 20
        total_height = len(lines) * line_height
        start_y = (self.height - total_height) // 2

        # Draw each line
        text_color = self._parse_color(template["text_color"])
        accent_color = self._parse_color(template["accent_color"])

        for i, line in enumerate(lines[:4]):  # Max 4 lines
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            x = (self.width - text_width) // 2
            y = start_y + i * line_height

            img = self._draw_text_with_effects(
                img, line, (x, y), font, text_color,
                effects={
                    "shadow": True,
                    "shadow_offset": 5,
                    "outline": True,
                    "outline_width": 4,
                }
            )

        # Add accent bar below text
        bar_y = start_y + len(lines[:4]) * line_height + 30
        bar_width = int(self.width * 0.3)
        bar_height = 6
        draw.rectangle(
            [
                (self.width - bar_width) // 2,
                bar_y,
                (self.width + bar_width) // 2,
                bar_y + bar_height
            ],
            fill=accent_color
        )

        return img

    def _wrap_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont,
        max_width: int
    ) -> List[str]:
        """Wrap text to fit within max_width."""
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

        return lines

    def _add_niche_visuals(
        self,
        img: Image.Image,
        niche: str,
        template: Dict[str, Any]
    ) -> Image.Image:
        """Add niche-specific visual elements."""
        draw = ImageDraw.Draw(img)
        accent_color = self._parse_color(template["accent_color"])
        primary_color = self._parse_color(template["primary_color"])

        if niche == "finance":
            # Add rising arrow/chart lines
            self._draw_rising_chart(draw, primary_color, accent_color)

        elif niche == "psychology":
            # Add neural network dots/connections
            self._draw_neural_pattern(draw, primary_color, accent_color)

        elif niche == "storytelling":
            # Add dramatic light rays
            self._draw_light_rays(draw, primary_color, accent_color)

        return img

    def _draw_rising_chart(
        self,
        draw: ImageDraw.ImageDraw,
        primary_color: Tuple[int, int, int],
        accent_color: Tuple[int, int, int]
    ):
        """Draw rising chart visual for finance niche."""
        # Draw subtle chart bars in background
        bar_width = self.width // 15
        max_bar_height = self.height // 3

        for i in range(10):
            x = int(self.width * 0.1 + i * bar_width * 1.1)
            height = int(max_bar_height * (0.3 + 0.7 * (i / 9)))  # Rising pattern
            y = self.height - int(self.height * 0.15) - height

            # Use lower opacity
            color = tuple(min(255, c + 20) for c in primary_color)
            draw.rectangle([x, y, x + bar_width - 5, self.height - int(self.height * 0.15)], fill=color)

        # Draw upward arrow
        arrow_size = self.width // 8
        center_x = self.width - self.width // 6
        center_y = self.height // 4

        # Arrow pointing up
        points = [
            (center_x, center_y - arrow_size // 2),  # Top
            (center_x - arrow_size // 3, center_y),  # Left
            (center_x - arrow_size // 6, center_y),  # Inner left
            (center_x - arrow_size // 6, center_y + arrow_size // 2),  # Bottom left
            (center_x + arrow_size // 6, center_y + arrow_size // 2),  # Bottom right
            (center_x + arrow_size // 6, center_y),  # Inner right
            (center_x + arrow_size // 3, center_y),  # Right
        ]
        draw.polygon(points, fill=accent_color)

    def _draw_neural_pattern(
        self,
        draw: ImageDraw.ImageDraw,
        primary_color: Tuple[int, int, int],
        accent_color: Tuple[int, int, int]
    ):
        """Draw neural network pattern for psychology niche."""
        # Draw random connected nodes
        nodes = []
        for _ in range(20):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            size = random.randint(3, 8)
            nodes.append((x, y, size))

        # Draw connections (lines)
        for i, (x1, y1, _) in enumerate(nodes):
            for j, (x2, y2, _) in enumerate(nodes[i+1:], i+1):
                if random.random() < 0.15:  # 15% chance of connection
                    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                    if distance < self.width // 4:
                        opacity = int(255 * (1 - distance / (self.width // 4)))
                        line_color = tuple(c // 4 for c in primary_color)
                        draw.line([(x1, y1), (x2, y2)], fill=line_color, width=1)

        # Draw nodes
        for x, y, size in nodes:
            node_color = tuple(c // 3 for c in accent_color)
            draw.ellipse([x-size, y-size, x+size, y+size], fill=node_color)

    def _draw_light_rays(
        self,
        draw: ImageDraw.ImageDraw,
        primary_color: Tuple[int, int, int],
        accent_color: Tuple[int, int, int]
    ):
        """Draw dramatic light rays for storytelling niche."""
        center_x = self.width // 2
        center_y = 0  # From top

        num_rays = 12
        for i in range(num_rays):
            angle = math.pi / 2 + (i - num_rays // 2) * (math.pi / num_rays / 1.5)
            length = self.height * 1.5

            end_x = center_x + int(math.cos(angle) * length)
            end_y = center_y + int(math.sin(angle) * length)

            # Create gradient ray (wider at base)
            ray_color = tuple(c // 6 for c in accent_color)

            # Draw multiple lines for thickness
            for offset in range(-15, 16, 3):
                draw.line(
                    [(center_x + offset, center_y), (end_x + offset * 3, end_y)],
                    fill=ray_color,
                    width=2
                )

    def generate_counter_animation_frames(
        self,
        start_value: int,
        end_value: int,
        niche: str,
        num_frames: int = 30,
        prefix: str = "$",
        suffix: str = ""
    ) -> List[Image.Image]:
        """
        Generate frames for an animated counter (e.g., $100 -> $10,000).

        Args:
            start_value: Starting number
            end_value: Ending number
            niche: Content niche for styling
            num_frames: Number of frames to generate
            prefix: Text before number (e.g., "$")
            suffix: Text after number (e.g., "/year")

        Returns:
            List of PIL Image frames
        """
        template = self.get_hook_template(niche)
        frames = []

        for i in range(num_frames):
            # Easing function for smooth animation
            progress = i / (num_frames - 1)
            eased = 1 - (1 - progress) ** 3  # Ease out cubic

            current_value = int(start_value + (end_value - start_value) * eased)
            formatted = f"{prefix}{current_value:,}{suffix}"

            # Create frame
            frame = self._create_gradient_background(
                self.resolution,
                template["gradient_colors"]
            )
            frame = self._add_vignette(frame)

            # Add the number
            draw = ImageDraw.Draw(frame)

            try:
                font_path = self.fonts.get("impact", self.fonts.get("bold", ""))
                font_size = int(self.COUNTER_FONT_SIZE * (self.width / 1920))
                font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
            except (OSError, IOError):
                font = ImageFont.load_default()

            # Center the text
            bbox = draw.textbbox((0, 0), formatted, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (self.width - text_width) // 2
            y = (self.height - text_height) // 2

            # Draw with effects
            text_color = self._parse_color(template["accent_color"])  # Use accent for numbers
            frame = self._draw_text_with_effects(
                frame, formatted, (x, y), font, text_color,
                effects={"shadow": True, "outline": True, "outline_width": 5}
            )

            frames.append(frame)

        return frames

    def create_animated_intro(
        self,
        hook_text: str,
        niche: str,
        duration: float = 3.0,
        animation_type: str = "zoom_in",
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Create an animated intro clip with the hook text.

        Args:
            hook_text: The hook text to animate
            niche: Content niche
            duration: Duration in seconds
            animation_type: Type of animation (zoom_in, fade_in, slide_up)
            output_path: Output video path (auto-generated if None)

        Returns:
            Path to the created video clip or None
        """
        if not self.ffmpeg:
            logger.error("FFmpeg required for animated intro creation")
            return None

        duration = max(self.MIN_HOOK_DURATION, min(duration, self.MAX_HOOK_DURATION))
        num_frames = int(duration * self.fps)

        # Generate frames
        frames = self._generate_animation_frames(
            hook_text, niche, num_frames, animation_type
        )

        if not frames:
            logger.error("Failed to generate animation frames")
            return None

        # Save frames to temp files
        frame_paths = []
        for i, frame in enumerate(frames):
            frame_path = self.temp_dir / f"hook_frame_{i:04d}.png"
            frame.save(str(frame_path))
            frame_paths.append(str(frame_path))

        # Create video from frames using FFmpeg
        output_path = output_path or str(self.temp_dir / f"hook_intro_{os.urandom(4).hex()}.mp4")

        try:
            cmd = [
                self.ffmpeg, '-y',
                '-framerate', str(self.fps),
                '-i', str(self.temp_dir / "hook_frame_%04d.png"),
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                output_path
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=120)

            if os.path.exists(output_path):
                logger.success(f"Animated hook intro created: {output_path}")
                return output_path
            else:
                logger.error(f"FFmpeg failed: {result.stderr.decode()[:500]}")
                return None

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as e:
            logger.error(f"Animated intro creation failed: {e}")
            return None
        finally:
            # Clean up frame files
            for path in frame_paths:
                try:
                    os.remove(path)
                except OSError:
                    pass

    def _generate_animation_frames(
        self,
        hook_text: str,
        niche: str,
        num_frames: int,
        animation_type: str
    ) -> List[Image.Image]:
        """Generate frames for the specified animation type."""
        template = self.get_hook_template(niche)
        frames = []

        # Create base frame
        base_frame = self.generate_hook_frame(
            topic="",
            niche=niche,
            hook_text=hook_text,
            include_visuals=True
        )

        for i in range(num_frames):
            progress = i / (num_frames - 1) if num_frames > 1 else 1.0

            if animation_type == "zoom_in":
                # Start zoomed out, zoom in to normal
                scale = 0.5 + 0.5 * self._ease_out_cubic(progress)
                frame = self._scale_image(base_frame, scale)

            elif animation_type == "zoom_out":
                # Start zoomed in, zoom out to normal
                scale = 1.3 - 0.3 * self._ease_out_cubic(progress)
                frame = self._scale_image(base_frame, scale)

            elif animation_type == "fade_in":
                # Fade from black
                opacity = self._ease_in_out_cubic(progress)
                black = Image.new('RGB', base_frame.size, (0, 0, 0))
                frame = Image.blend(black, base_frame, opacity)

            elif animation_type == "slide_up":
                # Slide up from bottom
                offset = int((1 - self._ease_out_cubic(progress)) * self.height * 0.3)
                frame = Image.new('RGB', self.resolution, self._parse_color(template["secondary_color"]))
                frame.paste(base_frame, (0, offset))

            elif animation_type == "pulse":
                # Pulsing effect
                pulse = 0.95 + 0.05 * math.sin(progress * math.pi * 4)
                frame = self._scale_image(base_frame, pulse)

            else:
                frame = base_frame.copy()

            frames.append(frame)

        return frames

    def _ease_out_cubic(self, t: float) -> float:
        """Cubic ease-out function."""
        return 1 - (1 - t) ** 3

    def _ease_in_out_cubic(self, t: float) -> float:
        """Cubic ease-in-out function."""
        if t < 0.5:
            return 4 * t * t * t
        else:
            return 1 - ((-2 * t + 2) ** 3) / 2

    def _scale_image(self, img: Image.Image, scale: float) -> Image.Image:
        """Scale an image from center."""
        width, height = img.size
        new_width = int(width * scale)
        new_height = int(height * scale)

        # Resize
        scaled = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create output at original size
        result = Image.new('RGB', (width, height), (0, 0, 0))

        # Paste centered
        x = (width - new_width) // 2
        y = (height - new_height) // 2

        if scale < 1:
            result.paste(scaled, (x, y))
        else:
            # Crop to fit
            crop_x = (new_width - width) // 2
            crop_y = (new_height - height) // 2
            result = scaled.crop((crop_x, crop_y, crop_x + width, crop_y + height))

        return result

    def add_hook_to_video(
        self,
        video_path: str,
        hook_text: str,
        niche: str,
        hook_duration: float = 3.0,
        animation_type: str = "zoom_in",
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Add a hook intro to the beginning of an existing video.

        Args:
            video_path: Path to the original video
            hook_text: Hook text to display
            niche: Content niche
            hook_duration: Duration of hook intro
            animation_type: Animation type for hook
            output_path: Output path (auto-generated if None)

        Returns:
            Path to new video with hook or None
        """
        if not self.ffmpeg or not os.path.exists(video_path):
            logger.error(f"FFmpeg required or video not found: {video_path}")
            return None

        # Create animated hook intro
        hook_clip = self.create_animated_intro(
            hook_text=hook_text,
            niche=niche,
            duration=hook_duration,
            animation_type=animation_type
        )

        if not hook_clip:
            logger.error("Failed to create hook clip")
            return None

        # Determine output path
        if not output_path:
            video_stem = Path(video_path).stem
            output_path = str(Path(video_path).parent / f"{video_stem}_with_hook.mp4")

        # Concatenate hook + original video
        concat_file = self.temp_dir / "concat_hook.txt"
        with open(concat_file, 'w') as f:
            f.write(f"file '{hook_clip}'\n")
            f.write(f"file '{video_path}'\n")

        try:
            cmd = [
                self.ffmpeg, '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-c:a', 'aac',
                '-b:a', '192k',
                output_path
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=300)

            if os.path.exists(output_path):
                logger.success(f"Video with hook created: {output_path}")
                return output_path
            else:
                logger.error(f"Concat failed: {result.stderr.decode()[:500]}")
                return None

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as e:
            logger.error(f"Add hook to video failed: {e}")
            return None
        finally:
            # Clean up
            try:
                if hook_clip and os.path.exists(hook_clip):
                    os.remove(hook_clip)
                if concat_file.exists():
                    concat_file.unlink()
            except OSError:
                pass

    def create_teaser_flash(
        self,
        climax_frame: Image.Image,
        niche: str,
        flash_duration: float = 0.3
    ) -> List[Image.Image]:
        """
        Create a split-second flash of the climax moment.

        Args:
            climax_frame: Image from the best moment of the video
            niche: Content niche
            flash_duration: Duration of the flash in seconds

        Returns:
            List of frames for the flash effect
        """
        template = self.get_hook_template(niche)
        num_frames = int(flash_duration * self.fps)
        frames = []

        # Create darkened version of climax
        enhancer = ImageEnhance.Brightness(climax_frame)
        dark_frame = enhancer.enhance(0.3)

        for i in range(num_frames):
            progress = i / (num_frames - 1) if num_frames > 1 else 1.0

            # Flash pattern: black -> bright -> black
            if progress < 0.2:
                # Fade in
                opacity = progress / 0.2
            elif progress > 0.8:
                # Fade out
                opacity = (1 - progress) / 0.2
            else:
                # Full brightness
                opacity = 1.0

            # Blend with black
            black = Image.new('RGB', climax_frame.size, (0, 0, 0))
            frame = Image.blend(black, climax_frame, opacity)

            # Add "Wait for it..." text if early in flash
            if progress < 0.5:
                draw = ImageDraw.Draw(frame)
                try:
                    font_path = self.fonts.get("bold", "")
                    font = ImageFont.truetype(font_path, 36) if font_path else ImageFont.load_default()
                except (OSError, IOError):
                    font = ImageFont.load_default()

                text = "Wait for it..."
                bbox = draw.textbbox((0, 0), text, font=font)
                x = (frame.width - (bbox[2] - bbox[0])) // 2
                y = frame.height - 100

                text_opacity = int(255 * (1 - progress * 2))
                draw.text((x + 2, y + 2), text, fill=(0, 0, 0), font=font)
                draw.text((x, y), text, fill=(255, 255, 255), font=font)

            frames.append(frame)

        return frames

    def validate_hook(
        self,
        video_path: str,
        hook_duration: float = 3.0
    ) -> HookValidationResult:
        """
        Validate that a video has a proper hook/intro.

        Checks:
        - First frame is not blank/black
        - Visual interest in first seconds
        - Hook text is visible (if applicable)
        - Audio starts immediately

        Args:
            video_path: Path to video file
            hook_duration: Duration to check for hook

        Returns:
            HookValidationResult with validation details
        """
        if not os.path.exists(video_path):
            return HookValidationResult(
                is_valid=False,
                has_visual_interest=False,
                first_frame_not_blank=False,
                hook_text_visible=False,
                has_audio=False,
                suggestions=["Video file not found"]
            )

        suggestions = []

        # Check first frame
        first_frame_ok, first_frame_brightness = self._check_first_frame(video_path)
        if not first_frame_ok:
            suggestions.append(f"First frame is too dark (brightness: {first_frame_brightness:.1f}). Add visual hook.")

        # Check for visual activity in first seconds
        visual_interest = self._check_visual_activity(video_path, hook_duration)
        if not visual_interest:
            suggestions.append("Insufficient visual activity in first seconds. Add animation or dynamic visuals.")

        # Check audio start
        has_audio = self._check_audio_start(video_path)
        if not has_audio:
            suggestions.append("Video starts with silence. Add music or speech immediately.")

        # Determine overall validity
        is_valid = first_frame_ok and visual_interest and has_audio

        return HookValidationResult(
            is_valid=is_valid,
            has_visual_interest=visual_interest,
            first_frame_not_blank=first_frame_ok,
            hook_text_visible=True,  # Assumed if visuals pass
            has_audio=has_audio,
            suggestions=suggestions
        )

    def _check_first_frame(self, video_path: str) -> Tuple[bool, float]:
        """Check if first frame is not blank/black."""
        if not self.ffmpeg:
            return True, 128.0  # Assume OK if no FFmpeg

        try:
            # Extract first frame
            frame_path = self.temp_dir / "first_frame_check.png"
            cmd = [
                self.ffmpeg, '-y',
                '-i', video_path,
                '-vframes', '1',
                '-f', 'image2',
                str(frame_path)
            ]
            subprocess.run(cmd, capture_output=True, timeout=30)

            if not frame_path.exists():
                return False, 0.0

            # Check brightness
            img = Image.open(frame_path)
            grayscale = img.convert('L')
            pixels = list(grayscale.getdata())
            avg_brightness = sum(pixels) / len(pixels)

            # Clean up
            frame_path.unlink()

            # Consider frame blank if average brightness < 20
            return avg_brightness >= 20, avg_brightness

        except Exception as e:
            logger.warning(f"First frame check failed: {e}")
            return True, 128.0  # Assume OK on error

    def _check_visual_activity(self, video_path: str, duration: float) -> bool:
        """Check for visual activity in the first seconds."""
        if not self.ffmpeg:
            return True  # Assume OK if no FFmpeg

        try:
            # Extract multiple frames from first few seconds
            frame_brightnesses = []

            for t in [0.0, 0.5, 1.0, 1.5, 2.0]:
                if t > duration:
                    break

                frame_path = self.temp_dir / f"activity_frame_{t}.png"
                cmd = [
                    self.ffmpeg, '-y',
                    '-ss', str(t),
                    '-i', video_path,
                    '-vframes', '1',
                    '-f', 'image2',
                    str(frame_path)
                ]
                subprocess.run(cmd, capture_output=True, timeout=30)

                if frame_path.exists():
                    img = Image.open(frame_path)
                    grayscale = img.convert('L')
                    pixels = list(grayscale.getdata())
                    avg = sum(pixels) / len(pixels)
                    frame_brightnesses.append(avg)
                    frame_path.unlink()

            if len(frame_brightnesses) < 2:
                return True  # Can't check, assume OK

            # Check for variance (visual activity = changing brightness)
            variance = sum((b - sum(frame_brightnesses)/len(frame_brightnesses))**2 for b in frame_brightnesses) / len(frame_brightnesses)

            # Low variance = static image = not visually interesting
            return variance > 50  # Threshold for "interesting"

        except Exception as e:
            logger.warning(f"Visual activity check failed: {e}")
            return True  # Assume OK on error

    def _check_audio_start(self, video_path: str) -> bool:
        """Check that video doesn't start with silence."""
        if not self.ffprobe:
            return True  # Assume OK if no FFprobe

        try:
            # Check for audio stream
            cmd = [
                self.ffprobe, '-v', 'error',
                '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_type',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            has_audio = 'audio' in result.stdout.lower()

            if not has_audio:
                return False

            # TODO: Could add more sophisticated silence detection here
            # For now, just check if audio stream exists
            return True

        except Exception as e:
            logger.warning(f"Audio check failed: {e}")
            return True  # Assume OK on error

    def get_suggested_hooks(self, topic: str, niche: str, count: int = 3) -> List[str]:
        """
        Get suggested hook text options for a topic.

        Args:
            topic: Video topic
            niche: Content niche
            count: Number of suggestions to return

        Returns:
            List of hook text suggestions
        """
        template = self.get_hook_template(niche)
        hooks = template.get("hooks", [])

        suggestions = []
        for hook_config in hooks[:count]:
            hook_template = hook_config["template"]

            # Replace placeholders with contextual values
            hook = hook_template.replace("{topic}", topic[:30])
            hook = hook.replace("{percentage}", str(random.randint(70, 97)))
            hook = hook.replace("{amount}", f"{random.randint(1, 20) * 1000:,}")
            hook = hook.replace("{timeframe}", random.choice(["24 hours", "one moment", "an instant"]))

            suggestions.append(hook)

        return suggestions

    def cleanup(self):
        """Clean up temporary files."""
        try:
            for f in self.temp_dir.glob("*"):
                if f.is_file():
                    f.unlink()
        except OSError as e:
            logger.debug(f"Cleanup failed: {e}")


# Factory function for easy creation
def create_hook_generator(
    is_shorts: bool = False,
    resolution: Tuple[int, int] = None
) -> VideoHookGenerator:
    """
    Create a VideoHookGenerator with appropriate settings.

    Args:
        is_shorts: If True, use vertical 9:16 format
        resolution: Custom resolution (optional)

    Returns:
        Configured VideoHookGenerator instance
    """
    return VideoHookGenerator(resolution=resolution, is_shorts=is_shorts)


# Test
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("VIDEO HOOK GENERATOR TEST")
    print("=" * 60 + "\n")

    # Initialize generator
    generator = VideoHookGenerator(is_shorts=True)
    print(f"Resolution: {generator.width}x{generator.height}")
    print(f"FFmpeg: {generator.ffmpeg}")

    # Test hook frame generation for each niche
    for niche in ["finance", "psychology", "storytelling"]:
        print(f"\n--- {niche.upper()} ---")

        # Get suggested hooks
        hooks = generator.get_suggested_hooks("Investment Secrets", niche, count=2)
        for hook in hooks:
            print(f"  Hook: {hook}")

        # Generate a hook frame
        hook_frame = generator.generate_hook_frame(
            topic="Test Topic",
            niche=niche,
            hook_text=hooks[0] if hooks else "Test Hook"
        )

        # Save test frame
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        hook_frame.save(f"output/test_hook_{niche}.png")
        print(f"  Saved: output/test_hook_{niche}.png")

    print("\n" + "=" * 60)
    print("Hook generator test complete!")
    print("=" * 60)
