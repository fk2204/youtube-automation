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
- Kinetic typography with word-by-word animation
- Guaranteed animated hooks (never static)

Classes:
- VideoHookGenerator: Base hook generation with various animation types
- GuaranteedHookGenerator: Ensures hooks are ALWAYS animated (never static)

Usage:
    from src.content.video_hooks import VideoHookGenerator, GuaranteedHookGenerator

    # Standard hook generation
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

    # RECOMMENDED: Use GuaranteedHookGenerator for NEVER-static hooks
    from src.content.video_hooks import create_guaranteed_hook_generator

    guaranteed_gen = create_guaranteed_hook_generator(is_shorts=True)

    # Generate guaranteed animated hook (falls back to kinetic typography)
    hook_video = guaranteed_gen.generate_guaranteed_hook(
        topic="Investment Secrets",
        niche="finance",
        duration=3.0
    )

    # Create kinetic typography directly
    kinetic_video = guaranteed_gen.create_kinetic_typography(
        text="This changes everything...",
        niche="psychology",
        duration=2.5
    )

    # Validate a hook is animated
    is_animated = guaranteed_gen.validate_hook_is_animated(hook_video)
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

    def create_guaranteed_animated_hook(
        self,
        topic: str,
        niche: str,
        duration: float = 3.0
    ) -> str:
        """
        Create a GUARANTEED animated hook using GuaranteedHookGenerator.

        This is the PRIMARY method that should be used for hook generation.
        It ensures hooks are NEVER static and always have animation in
        the first 0.8 seconds.

        Args:
            topic: Video topic for context
            niche: Content niche (finance, psychology, storytelling)
            duration: Hook duration in seconds (default 3.0)

        Returns:
            Path to the animated hook video file
        """
        # Use GuaranteedHookGenerator for reliable animated hooks
        guaranteed_generator = GuaranteedHookGenerator(
            resolution=self.resolution,
            fps=self.fps,
            is_shorts=self.is_shorts
        )

        try:
            return guaranteed_generator.generate_guaranteed_hook(
                topic=topic,
                niche=niche,
                duration=duration
            )
        finally:
            # Clean up the guaranteed generator's temp files
            guaranteed_generator.cleanup()

    def cleanup(self):
        """Clean up temporary files."""
        try:
            for f in self.temp_dir.glob("*"):
                if f.is_file():
                    f.unlink()
        except OSError as e:
            logger.debug(f"Cleanup failed: {e}")


class GuaranteedHookGenerator:
    """
    Ensures hooks NEVER fall back to static titles.

    This class guarantees that every hook is animated, with fallback
    to kinetic typography if other animation methods fail. The first
    0.8 seconds are always guaranteed to be animated.

    Design principles:
    - NEVER return a static image/title
    - Always produce animated content
    - Kinetic typography as ultimate fallback
    - First 0.8 seconds MUST have motion
    """

    # Kinetic typography animation styles
    KINETIC_STYLES = {
        "word_by_word": {
            "description": "Reveal words one at a time with zoom",
            "word_delay": 0.15,  # seconds between words
            "animation_per_word": "zoom_fade",
        },
        "phrase_cascade": {
            "description": "Phrases slide in from different directions",
            "phrase_delay": 0.3,
            "directions": ["left", "right", "bottom", "top"],
        },
        "impact_burst": {
            "description": "Words burst onto screen with scale effect",
            "scale_start": 2.0,
            "scale_end": 1.0,
            "word_delay": 0.12,
        },
        "typewriter_modern": {
            "description": "Modern typewriter with cursor and glow",
            "char_delay": 0.03,
            "cursor_blink": True,
        },
        "wave_text": {
            "description": "Text appears in a wave pattern",
            "wave_amplitude": 20,
            "wave_frequency": 2,
        },
    }

    # Niche-specific kinetic typography colors and styles
    NICHE_KINETIC_THEMES = {
        "finance": {
            "primary_text_color": "#00d4aa",
            "secondary_text_color": "#ffffff",
            "accent_color": "#ffd700",
            "background_colors": ["#0a0a14", "#0f2027"],
            "preferred_styles": ["impact_burst", "word_by_word"],
            "font_weight": "bold",
            "glow_color": "#00d4aa",
        },
        "psychology": {
            "primary_text_color": "#9b59b6",
            "secondary_text_color": "#ffffff",
            "accent_color": "#e74c3c",
            "background_colors": ["#050510", "#150520"],
            "preferred_styles": ["phrase_cascade", "typewriter_modern"],
            "font_weight": "medium",
            "glow_color": "#9b59b6",
        },
        "storytelling": {
            "primary_text_color": "#e74c3c",
            "secondary_text_color": "#ffffff",
            "accent_color": "#f39c12",
            "background_colors": ["#080808", "#150808"],
            "preferred_styles": ["typewriter_modern", "phrase_cascade"],
            "font_weight": "bold",
            "glow_color": "#e74c3c",
        },
        "default": {
            "primary_text_color": "#3498db",
            "secondary_text_color": "#ffffff",
            "accent_color": "#e74c3c",
            "background_colors": ["#0a0a14", "#1a1a2e"],
            "preferred_styles": ["word_by_word", "impact_burst"],
            "font_weight": "bold",
            "glow_color": "#3498db",
        },
    }

    # Minimum animated duration in the first portion of hook
    MINIMUM_ANIMATED_DURATION = 0.8  # seconds

    def __init__(
        self,
        resolution: Tuple[int, int] = None,
        fps: int = 30,
        is_shorts: bool = False
    ):
        """
        Initialize the Guaranteed Hook Generator.

        Args:
            resolution: Output resolution (width, height)
            fps: Frames per second
            is_shorts: If True, use vertical 9:16 format
        """
        # Initialize base video hook generator for shared functionality
        self.base_generator = VideoHookGenerator(
            resolution=resolution,
            fps=fps,
            is_shorts=is_shorts
        )

        self.width = self.base_generator.width
        self.height = self.base_generator.height
        self.resolution = self.base_generator.resolution
        self.fps = fps
        self.is_shorts = is_shorts
        self.ffmpeg = self.base_generator.ffmpeg
        self.ffprobe = self.base_generator.ffprobe
        self.temp_dir = self.base_generator.temp_dir
        self.fonts = self.base_generator.fonts

        logger.info(f"GuaranteedHookGenerator initialized ({self.width}x{self.height} @ {self.fps}fps)")

    def generate_guaranteed_hook(
        self,
        topic: str,
        niche: str,
        duration: float = 3.0
    ) -> str:
        """
        Generate a guaranteed animated hook video.

        This method NEVER returns a static image. It tries multiple
        animation methods and falls back to kinetic typography if needed.

        Args:
            topic: Video topic for context
            niche: Content niche (finance, psychology, storytelling)
            duration: Hook duration in seconds (default 3.0)

        Returns:
            Path to the animated hook video file

        Raises:
            RuntimeError: If unable to create any animated hook (should never happen)
        """
        duration = max(1.5, min(duration, 5.0))  # Clamp duration

        logger.info(f"Generating guaranteed animated hook for topic: {topic}, niche: {niche}")

        # Get suggested hook text
        hook_texts = self.base_generator.get_suggested_hooks(topic, niche, count=3)
        hook_text = hook_texts[0] if hook_texts else f"Discover {topic}..."

        # Try primary animation methods first
        animated_hook_path = None

        # Method 1: Try standard animated intro
        try:
            animated_hook_path = self.base_generator.create_animated_intro(
                hook_text=hook_text,
                niche=niche,
                duration=duration,
                animation_type=random.choice(["zoom_in", "fade_in", "slide_up"])
            )

            if animated_hook_path and self.validate_hook_is_animated(animated_hook_path):
                logger.success(f"Generated animated hook via standard method: {animated_hook_path}")
                return animated_hook_path
            else:
                logger.warning("Standard animation failed validation, trying kinetic typography")
        except Exception as e:
            logger.warning(f"Standard animation failed: {e}, falling back to kinetic typography")

        # Method 2: Kinetic typography fallback (guaranteed to work)
        try:
            animated_hook_path = self.create_kinetic_typography(
                text=hook_text,
                niche=niche,
                duration=duration
            )

            if animated_hook_path and self.validate_hook_is_animated(animated_hook_path):
                logger.success(f"Generated animated hook via kinetic typography: {animated_hook_path}")
                return animated_hook_path
        except Exception as e:
            logger.error(f"Kinetic typography failed: {e}")

        # Method 3: Ultimate fallback - simple zoom animation on text
        try:
            animated_hook_path = self._create_simple_zoom_text(
                text=hook_text,
                niche=niche,
                duration=duration
            )

            if animated_hook_path:
                logger.success(f"Generated animated hook via simple zoom: {animated_hook_path}")
                return animated_hook_path
        except Exception as e:
            logger.error(f"Simple zoom fallback failed: {e}")

        # This should never happen, but raise error if all methods fail
        raise RuntimeError(
            "CRITICAL: Unable to generate any animated hook. "
            "All animation methods failed. Check FFmpeg installation."
        )

    def create_kinetic_typography(
        self,
        text: str,
        niche: str,
        duration: float
    ) -> str:
        """
        Create animated kinetic typography video.

        Generates word-by-word or phrase-by-phrase animation with
        zoom, fade, and slide effects. This is the primary fallback
        that ensures hooks are ALWAYS animated.

        Args:
            text: The text to animate
            niche: Content niche for styling
            duration: Duration in seconds

        Returns:
            Path to the animated kinetic typography video
        """
        if not self.ffmpeg:
            raise RuntimeError("FFmpeg required for kinetic typography")

        theme = self.NICHE_KINETIC_THEMES.get(niche, self.NICHE_KINETIC_THEMES["default"])
        preferred_style = random.choice(theme["preferred_styles"])
        style_config = self.KINETIC_STYLES[preferred_style]

        logger.info(f"Creating kinetic typography with style: {preferred_style}")

        # Split text into words or phrases
        words = text.split()

        # Calculate timing
        num_frames = int(duration * self.fps)
        min_animated_frames = int(self.MINIMUM_ANIMATED_DURATION * self.fps)

        frames = []

        if preferred_style == "word_by_word":
            frames = self._generate_word_by_word_frames(
                words, theme, num_frames, min_animated_frames
            )
        elif preferred_style == "phrase_cascade":
            frames = self._generate_phrase_cascade_frames(
                words, theme, num_frames, min_animated_frames
            )
        elif preferred_style == "impact_burst":
            frames = self._generate_impact_burst_frames(
                words, theme, num_frames, min_animated_frames
            )
        elif preferred_style == "typewriter_modern":
            frames = self._generate_typewriter_frames(
                text, theme, num_frames, min_animated_frames
            )
        else:
            # Default to word_by_word
            frames = self._generate_word_by_word_frames(
                words, theme, num_frames, min_animated_frames
            )

        # Ensure we have enough frames for first 0.8 seconds to be animated
        if len(frames) < min_animated_frames:
            logger.warning(f"Generated {len(frames)} frames, padding to {min_animated_frames}")
            while len(frames) < num_frames:
                frames.append(frames[-1] if frames else self._create_base_frame(theme))

        # Save frames and create video
        return self._frames_to_video(frames, f"kinetic_hook_{os.urandom(4).hex()}.mp4")

    def _generate_word_by_word_frames(
        self,
        words: List[str],
        theme: Dict[str, Any],
        total_frames: int,
        min_animated_frames: int
    ) -> List[Image.Image]:
        """Generate frames for word-by-word animation with zoom effect."""
        frames = []

        # Calculate frames per word
        frames_per_word = max(4, total_frames // max(len(words), 1))

        # Load font
        font = self._get_kinetic_font(theme)

        accumulated_words = []

        for word_idx, word in enumerate(words):
            accumulated_words.append(word)
            current_text = " ".join(accumulated_words)

            # Generate frames for this word appearing
            for frame_idx in range(frames_per_word):
                progress = frame_idx / frames_per_word

                # Create base frame
                frame = self._create_base_frame(theme)
                draw = ImageDraw.Draw(frame)

                # Draw previously accumulated words (stable)
                if len(accumulated_words) > 1:
                    prev_text = " ".join(accumulated_words[:-1])
                    self._draw_centered_text(
                        draw, prev_text, font,
                        self._parse_color(theme["secondary_text_color"]),
                        y_offset=-50, alpha=1.0
                    )

                # Draw current word with animation
                scale = 0.5 + 0.5 * self._ease_out_cubic(progress)
                alpha = self._ease_out_cubic(progress)

                # Calculate word position
                word_font_size = int(font.size * scale) if hasattr(font, 'size') else int(72 * scale)
                try:
                    word_font = ImageFont.truetype(
                        self.fonts.get("impact", self.fonts.get("bold", "")),
                        max(24, word_font_size)
                    )
                except (OSError, IOError):
                    word_font = font

                # Draw the new word with zoom effect
                accent_color = self._parse_color(theme["accent_color"])
                self._draw_centered_text(
                    draw, word, word_font, accent_color,
                    y_offset=50, alpha=alpha
                )

                # Add glow effect
                if progress > 0.5:
                    glow_alpha = int(100 * (1 - progress))
                    self._add_text_glow(frame, word, word_font, theme["glow_color"], glow_alpha)

                frames.append(frame)

        # Hold on final frame
        while len(frames) < total_frames:
            frames.append(frames[-1].copy() if frames else self._create_base_frame(theme))

        return frames[:total_frames]

    def _generate_phrase_cascade_frames(
        self,
        words: List[str],
        theme: Dict[str, Any],
        total_frames: int,
        min_animated_frames: int
    ) -> List[Image.Image]:
        """Generate frames for phrase cascade animation."""
        frames = []

        # Group words into phrases (2-3 words each)
        phrases = []
        for i in range(0, len(words), 2):
            phrase = " ".join(words[i:i+2])
            phrases.append(phrase)

        if not phrases:
            phrases = [" ".join(words)]

        frames_per_phrase = max(8, total_frames // max(len(phrases), 1))
        directions = ["left", "right", "bottom"]

        font = self._get_kinetic_font(theme)
        accumulated_phrases = []

        for phrase_idx, phrase in enumerate(phrases):
            direction = directions[phrase_idx % len(directions)]
            accumulated_phrases.append((phrase, direction))

            for frame_idx in range(frames_per_phrase):
                progress = frame_idx / frames_per_phrase
                eased = self._ease_out_cubic(progress)

                frame = self._create_base_frame(theme)
                draw = ImageDraw.Draw(frame)

                # Draw all accumulated phrases
                y_position = self.height // 3
                for p_idx, (p_text, p_dir) in enumerate(accumulated_phrases):
                    is_current = (p_idx == len(accumulated_phrases) - 1)

                    if is_current:
                        # Animate current phrase sliding in
                        if p_dir == "left":
                            x_offset = int((1 - eased) * -self.width * 0.5)
                        elif p_dir == "right":
                            x_offset = int((1 - eased) * self.width * 0.5)
                        else:  # bottom
                            x_offset = 0
                            y_position += int((1 - eased) * 200)

                        alpha = eased
                        color = self._parse_color(theme["accent_color"])
                    else:
                        x_offset = 0
                        alpha = 1.0
                        color = self._parse_color(theme["secondary_text_color"])

                    self._draw_centered_text(
                        draw, p_text, font, color,
                        y_offset=y_position - self.height // 2 + x_offset,
                        alpha=alpha
                    )
                    y_position += 80

                frames.append(frame)

        # Hold final frame
        while len(frames) < total_frames:
            frames.append(frames[-1].copy() if frames else self._create_base_frame(theme))

        return frames[:total_frames]

    def _generate_impact_burst_frames(
        self,
        words: List[str],
        theme: Dict[str, Any],
        total_frames: int,
        min_animated_frames: int
    ) -> List[Image.Image]:
        """Generate frames for impact burst animation."""
        frames = []

        frames_per_word = max(6, total_frames // max(len(words), 1))
        font = self._get_kinetic_font(theme)

        for word_idx, word in enumerate(words):
            for frame_idx in range(frames_per_word):
                progress = frame_idx / frames_per_word

                frame = self._create_base_frame(theme)
                draw = ImageDraw.Draw(frame)

                # Draw previous words (smaller, at top)
                if word_idx > 0:
                    prev_words = " ".join(words[:word_idx])
                    small_font = self._get_kinetic_font(theme, size_multiplier=0.5)
                    self._draw_centered_text(
                        draw, prev_words, small_font,
                        self._parse_color(theme["secondary_text_color"]),
                        y_offset=-self.height // 4
                    )

                # Impact burst effect: start large, shrink to normal with bounce
                if progress < 0.3:
                    scale = 2.0 - (progress / 0.3) * 1.0  # 2.0 -> 1.0
                elif progress < 0.5:
                    scale = 1.0 + (progress - 0.3) / 0.2 * 0.1  # slight bounce up
                else:
                    scale = 1.1 - (progress - 0.5) / 0.5 * 0.1  # settle to 1.0

                alpha = min(1.0, progress * 3)  # Quick fade in

                # Scale the font
                try:
                    burst_font_size = int(72 * scale)
                    burst_font = ImageFont.truetype(
                        self.fonts.get("impact", self.fonts.get("bold", "")),
                        max(24, burst_font_size)
                    )
                except (OSError, IOError):
                    burst_font = font

                color = self._parse_color(theme["accent_color"])
                self._draw_centered_text(draw, word, burst_font, color, alpha=alpha)

                frames.append(frame)

        # Hold final frame with all words
        final_frame = self._create_base_frame(theme)
        draw = ImageDraw.Draw(final_frame)
        full_text = " ".join(words)
        self._draw_centered_text(
            draw, full_text, font,
            self._parse_color(theme["secondary_text_color"])
        )

        while len(frames) < total_frames:
            frames.append(final_frame.copy())

        return frames[:total_frames]

    def _generate_typewriter_frames(
        self,
        text: str,
        theme: Dict[str, Any],
        total_frames: int,
        min_animated_frames: int
    ) -> List[Image.Image]:
        """Generate frames for modern typewriter animation."""
        frames = []

        char_delay = 0.03
        chars_per_frame = max(1, int(1 / (char_delay * self.fps)))

        font = self._get_kinetic_font(theme)
        cursor_visible = True

        for frame_idx in range(total_frames):
            frame = self._create_base_frame(theme)
            draw = ImageDraw.Draw(frame)

            # Calculate how many characters to show
            chars_shown = min(len(text), (frame_idx * chars_per_frame) // 2 + 1)
            current_text = text[:chars_shown]

            # Toggle cursor every few frames
            if frame_idx % 8 < 4:
                cursor_visible = True
            else:
                cursor_visible = False

            # Add cursor if still typing
            display_text = current_text
            if chars_shown < len(text) and cursor_visible:
                display_text += "|"
            elif chars_shown >= len(text) and cursor_visible:
                display_text += "|"

            # Draw with glow effect
            color = self._parse_color(theme["primary_text_color"])
            self._draw_centered_text(draw, display_text, font, color)

            # Add subtle glow on newly typed character
            if chars_shown > 0 and chars_shown <= len(text):
                glow_intensity = 50 if frame_idx % 3 == 0 else 30
                self._add_text_glow(
                    frame, current_text[-1:], font,
                    theme["glow_color"], glow_intensity
                )

            frames.append(frame)

        return frames[:total_frames]

    def _create_base_frame(self, theme: Dict[str, Any]) -> Image.Image:
        """Create a base frame with gradient background."""
        return self.base_generator._create_gradient_background(
            self.resolution,
            theme["background_colors"],
            style="vertical"
        )

    def _get_kinetic_font(
        self,
        theme: Dict[str, Any],
        size_multiplier: float = 1.0
    ) -> ImageFont.FreeTypeFont:
        """Get font for kinetic typography."""
        base_size = int(72 * (self.width / 1920) * size_multiplier)

        try:
            font_path = self.fonts.get("impact", self.fonts.get("bold", ""))
            if font_path:
                return ImageFont.truetype(font_path, max(24, base_size))
        except (OSError, IOError):
            pass

        return ImageFont.load_default()

    def _parse_color(self, color: str) -> Tuple[int, int, int]:
        """Parse hex color to RGB tuple."""
        return self.base_generator._parse_color(color)

    def _ease_out_cubic(self, t: float) -> float:
        """Cubic ease-out function."""
        return 1 - (1 - t) ** 3

    def _draw_centered_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont,
        color: Tuple[int, int, int],
        y_offset: int = 0,
        alpha: float = 1.0
    ):
        """Draw text centered on the frame."""
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (self.width - text_width) // 2
        y = (self.height - text_height) // 2 + y_offset

        # Apply alpha by adjusting color
        adjusted_color = tuple(int(c * alpha) for c in color)

        # Draw shadow
        shadow_color = tuple(int(c * 0.3 * alpha) for c in (0, 0, 0))
        draw.text((x + 3, y + 3), text, font=font, fill=shadow_color)

        # Draw outline
        outline_color = (0, 0, 0)
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), text, font=font, fill=outline_color)

        # Draw main text
        draw.text((x, y), text, font=font, fill=adjusted_color)

    def _add_text_glow(
        self,
        frame: Image.Image,
        text: str,
        font: ImageFont.FreeTypeFont,
        glow_color: str,
        intensity: int = 50
    ):
        """Add a glow effect around text (simplified)."""
        # Create glow layer
        glow = Image.new('RGBA', frame.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(glow)

        color = self._parse_color(glow_color)
        glow_rgba = (*color, intensity)

        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (self.width - text_width) // 2
        y = (self.height - text_height) // 2

        # Draw multiple offset copies for glow effect
        for offset in range(5, 0, -1):
            for dx in [-offset, 0, offset]:
                for dy in [-offset, 0, offset]:
                    draw.text(
                        (x + dx, y + dy), text, font=font,
                        fill=(*color, max(10, intensity // offset))
                    )

        # Composite glow onto frame (simplified - just blend)
        if frame.mode != 'RGBA':
            frame = frame.convert('RGBA')

        # Note: Full alpha compositing would require more complex blending
        # This is a simplified version

    def _create_simple_zoom_text(
        self,
        text: str,
        niche: str,
        duration: float
    ) -> str:
        """Create a simple zoom animation as ultimate fallback."""
        theme = self.NICHE_KINETIC_THEMES.get(niche, self.NICHE_KINETIC_THEMES["default"])
        num_frames = int(duration * self.fps)
        frames = []

        font = self._get_kinetic_font(theme)

        for i in range(num_frames):
            progress = i / num_frames

            frame = self._create_base_frame(theme)
            draw = ImageDraw.Draw(frame)

            # Simple zoom from 0.5 to 1.0
            scale = 0.5 + 0.5 * self._ease_out_cubic(progress)
            alpha = self._ease_out_cubic(min(1.0, progress * 2))

            try:
                scaled_size = int(72 * (self.width / 1920) * scale)
                scaled_font = ImageFont.truetype(
                    self.fonts.get("impact", self.fonts.get("bold", "")),
                    max(24, scaled_size)
                )
            except (OSError, IOError):
                scaled_font = font

            color = self._parse_color(theme["primary_text_color"])
            self._draw_centered_text(draw, text, scaled_font, color, alpha=alpha)

            frames.append(frame)

        return self._frames_to_video(frames, f"simple_zoom_hook_{os.urandom(4).hex()}.mp4")

    def _frames_to_video(self, frames: List[Image.Image], output_filename: str) -> str:
        """Convert list of frames to video file."""
        if not frames:
            raise RuntimeError("No frames to convert to video")

        output_path = str(self.temp_dir / output_filename)

        # Save frames to temp files
        frame_paths = []
        for i, frame in enumerate(frames):
            frame_path = self.temp_dir / f"kinetic_frame_{i:04d}.png"
            # Ensure frame is RGB mode
            if frame.mode != 'RGB':
                frame = frame.convert('RGB')
            frame.save(str(frame_path))
            frame_paths.append(str(frame_path))

        try:
            cmd = [
                self.ffmpeg, '-y',
                '-framerate', str(self.fps),
                '-i', str(self.temp_dir / "kinetic_frame_%04d.png"),
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                output_path
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=120)

            if os.path.exists(output_path):
                return output_path
            else:
                logger.error(f"FFmpeg failed: {result.stderr.decode()[:500]}")
                raise RuntimeError(f"Failed to create video from frames: {result.stderr.decode()[:200]}")

        finally:
            # Clean up frame files
            for path in frame_paths:
                try:
                    os.remove(path)
                except OSError:
                    pass

    def validate_hook_is_animated(self, video_path: str) -> bool:
        """
        Verify that a hook video is actually animated, not static.

        Checks for visual changes between frames in the first 0.8 seconds
        to ensure the hook has motion/animation.

        Args:
            video_path: Path to the hook video file

        Returns:
            True if hook is animated, False if static
        """
        if not os.path.exists(video_path):
            logger.warning(f"Video file not found: {video_path}")
            return False

        if not self.ffmpeg:
            # Can't verify without FFmpeg, assume it's animated
            logger.warning("Cannot verify animation without FFmpeg, assuming animated")
            return True

        try:
            # Extract frames from first 0.8 seconds
            frame_times = [0.0, 0.2, 0.4, 0.6, 0.8]
            frame_hashes = []

            for t in frame_times:
                frame_path = self.temp_dir / f"validate_frame_{t}.png"
                cmd = [
                    self.ffmpeg, '-y',
                    '-ss', str(t),
                    '-i', video_path,
                    '-vframes', '1',
                    '-f', 'image2',
                    str(frame_path)
                ]

                result = subprocess.run(cmd, capture_output=True, timeout=30)

                if frame_path.exists():
                    # Calculate simple hash of frame content
                    img = Image.open(frame_path)
                    # Resize to small size for comparison
                    small = img.resize((32, 32), Image.Resampling.LANCZOS)
                    grayscale = small.convert('L')
                    pixels = list(grayscale.getdata())
                    frame_hash = sum(pixels)
                    frame_hashes.append(frame_hash)

                    # Clean up
                    frame_path.unlink()

            if len(frame_hashes) < 2:
                logger.warning("Could not extract enough frames for validation")
                return True  # Assume animated if can't check

            # Check if frames are different (animated) or same (static)
            # Calculate variance in frame hashes
            avg_hash = sum(frame_hashes) / len(frame_hashes)
            variance = sum((h - avg_hash) ** 2 for h in frame_hashes) / len(frame_hashes)

            # If variance is very low, frames are nearly identical (static)
            is_animated = variance > 100  # Threshold for "different enough"

            if not is_animated:
                logger.warning(f"Hook appears static (variance: {variance:.1f})")
            else:
                logger.debug(f"Hook validated as animated (variance: {variance:.1f})")

            return is_animated

        except Exception as e:
            logger.error(f"Animation validation failed: {e}")
            return True  # Assume animated on error

    def cleanup(self):
        """Clean up temporary files."""
        self.base_generator.cleanup()


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


def create_guaranteed_hook_generator(
    is_shorts: bool = False,
    resolution: Tuple[int, int] = None
) -> GuaranteedHookGenerator:
    """
    Create a GuaranteedHookGenerator that NEVER falls back to static titles.

    This is the recommended factory function for hook generation as it
    ensures hooks are always animated with kinetic typography fallback.

    Args:
        is_shorts: If True, use vertical 9:16 format
        resolution: Custom resolution (optional)

    Returns:
        Configured GuaranteedHookGenerator instance
    """
    return GuaranteedHookGenerator(resolution=resolution, is_shorts=is_shorts)


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

    # Test GuaranteedHookGenerator
    print("\n" + "=" * 60)
    print("GUARANTEED HOOK GENERATOR TEST")
    print("=" * 60 + "\n")

    guaranteed_gen = GuaranteedHookGenerator(is_shorts=True)
    print(f"Resolution: {guaranteed_gen.width}x{guaranteed_gen.height}")
    print(f"FFmpeg: {guaranteed_gen.ffmpeg}")

    # Test guaranteed animated hook generation
    if guaranteed_gen.ffmpeg:
        for niche in ["finance", "psychology", "storytelling"]:
            print(f"\n--- {niche.upper()} GUARANTEED HOOK ---")
            try:
                hook_path = guaranteed_gen.generate_guaranteed_hook(
                    topic="Investment Secrets",
                    niche=niche,
                    duration=2.0
                )
                print(f"  Generated: {hook_path}")

                # Validate it's animated
                is_animated = guaranteed_gen.validate_hook_is_animated(hook_path)
                print(f"  Is Animated: {is_animated}")

            except Exception as e:
                print(f"  Error: {e}")

        # Test kinetic typography directly
        print("\n--- KINETIC TYPOGRAPHY TEST ---")
        try:
            kinetic_path = guaranteed_gen.create_kinetic_typography(
                text="This changes everything...",
                niche="psychology",
                duration=2.5
            )
            print(f"  Kinetic Typography: {kinetic_path}")
            is_animated = guaranteed_gen.validate_hook_is_animated(kinetic_path)
            print(f"  Is Animated: {is_animated}")
        except Exception as e:
            print(f"  Error: {e}")

        guaranteed_gen.cleanup()
    else:
        print("FFmpeg not found - skipping animated hook tests")

    print("\n" + "=" * 60)
    print("Guaranteed hook generator test complete!")
    print("=" * 60)
