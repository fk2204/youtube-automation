"""
Image content creation engine.

Generates platform-specific images:
- Pinterest pins (1000x1500, text overlay, branded)
- Quote images (1080x1080, overlay on photo/gradient)
- Infographics (1080x1350, data visualization)
- Carousel slides (1080x1080, numbered sequence)

Uses Pillow for image generation. Reuses channel_branding.py for colors.
"""

import os
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("Pillow not installed. Image generation disabled. pip install Pillow")


# Niche-specific color schemes
NICHE_COLORS = {
    "finance": {
        "primary": (16, 185, 129),     # Emerald green
        "secondary": (5, 150, 105),
        "accent": (252, 211, 77),       # Gold
        "bg": (15, 23, 42),             # Dark navy
        "text": (255, 255, 255),
    },
    "psychology": {
        "primary": (139, 92, 246),      # Purple
        "secondary": (109, 40, 217),
        "accent": (236, 72, 153),       # Pink
        "bg": (30, 10, 60),
        "text": (255, 255, 255),
    },
    "storytelling": {
        "primary": (239, 68, 68),       # Red
        "secondary": (185, 28, 28),
        "accent": (251, 146, 60),       # Orange
        "bg": (20, 10, 10),
        "text": (255, 255, 255),
    },
}
DEFAULT_COLORS = NICHE_COLORS["finance"]


@dataclass
class GeneratedImage:
    """A generated image file."""
    path: str
    style: str  # pin, quote, infographic, carousel_slide
    dimensions: Tuple[int, int]
    platform: str = ""


class ImageEngine:
    """Creates branded images for multi-platform distribution."""

    def __init__(self, font_path: Optional[str] = None):
        self._font_path = font_path
        self._output_dir = Path("output/images")
        self._output_dir.mkdir(parents=True, exist_ok=True)

    async def create_all(
        self,
        topic: str,
        niche: str = "general",
        channel: str = "",
        key_points: Optional[List[str]] = None,
        quotes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create all image types for a topic."""
        if not PIL_AVAILABLE:
            return {"status": "error", "images": [], "error": "Pillow not installed"}

        images = []
        colors = NICHE_COLORS.get(niche, DEFAULT_COLORS)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # Pinterest pin
        pin = self._create_pin(topic, niche, colors, timestamp)
        if pin:
            images.append(pin)

        # Quote image
        if quotes:
            for i, quote in enumerate(quotes[:3]):
                qi = self._create_quote_image(quote, niche, colors, f"{timestamp}_{i}")
                if qi:
                    images.append(qi)
        else:
            qi = self._create_quote_image(topic, niche, colors, timestamp)
            if qi:
                images.append(qi)

        # Carousel slides from key points
        if key_points:
            slides = self._create_carousel(topic, key_points, niche, colors, timestamp)
            images.extend(slides)

        return {
            "status": "success",
            "images": [{"path": img.path, "style": img.style, "dimensions": img.dimensions, "platform": img.platform} for img in images],
        }

    def _create_pin(
        self, topic: str, niche: str, colors: dict, timestamp: str
    ) -> Optional[GeneratedImage]:
        """Create a Pinterest pin (1000x1500)."""
        try:
            w, h = 1000, 1500
            img = Image.new("RGB", (w, h), colors["bg"])
            draw = ImageDraw.Draw(img)

            # Background gradient effect (top to bottom)
            for y in range(h):
                ratio = y / h
                r = int(colors["bg"][0] * (1 - ratio * 0.3))
                g = int(colors["bg"][1] * (1 - ratio * 0.3))
                b = int(colors["bg"][2] * (1 - ratio * 0.3))
                draw.line([(0, y), (w, y)], fill=(max(r, 0), max(g, 0), max(b, 0)))

            # Accent bar at top
            draw.rectangle([(0, 0), (w, 8)], fill=colors["accent"])

            # Title text
            font = self._get_font(48)
            font_small = self._get_font(28)

            wrapped = textwrap.fill(topic, width=25)
            y_pos = h // 3
            for line in wrapped.split("\n"):
                bbox = draw.textbbox((0, 0), line, font=font)
                text_w = bbox[2] - bbox[0]
                draw.text(((w - text_w) // 2, y_pos), line, fill=colors["text"], font=font)
                y_pos += 60

            # Niche label
            label = niche.upper()
            bbox = draw.textbbox((0, 0), label, font=font_small)
            label_w = bbox[2] - bbox[0]
            draw.text(((w - label_w) // 2, h - 120), label, fill=colors["accent"], font=font_small)

            # Bottom accent bar
            draw.rectangle([(0, h - 8), (w, h)], fill=colors["primary"])

            path = str(self._output_dir / f"pin_{timestamp}.png")
            img.save(path, "PNG", quality=95)
            return GeneratedImage(path=path, style="pin", dimensions=(w, h), platform="pinterest")

        except Exception as e:
            logger.error(f"Pin creation failed: {e}")
            return None

    def _create_quote_image(
        self, quote: str, niche: str, colors: dict, timestamp: str
    ) -> Optional[GeneratedImage]:
        """Create a quote image (1080x1080)."""
        try:
            w, h = 1080, 1080
            img = Image.new("RGB", (w, h), colors["bg"])
            draw = ImageDraw.Draw(img)

            # Border
            border = 40
            draw.rectangle(
                [(border, border), (w - border, h - border)],
                outline=colors["primary"],
                width=3,
            )

            # Quote marks
            font_quote = self._get_font(72)
            draw.text((80, 100), "\u201C", fill=colors["accent"], font=font_quote)

            # Quote text
            font = self._get_font(36)
            wrapped = textwrap.fill(quote, width=28)
            y_pos = 200
            for line in wrapped.split("\n"):
                bbox = draw.textbbox((0, 0), line, font=font)
                text_w = bbox[2] - bbox[0]
                draw.text(((w - text_w) // 2, y_pos), line, fill=colors["text"], font=font)
                y_pos += 50

            # Niche watermark
            font_small = self._get_font(22)
            draw.text((80, h - 100), f"@{niche}_insights", fill=colors["secondary"], font=font_small)

            path = str(self._output_dir / f"quote_{timestamp}.png")
            img.save(path, "PNG", quality=95)
            return GeneratedImage(path=path, style="quote", dimensions=(w, h), platform="instagram")

        except Exception as e:
            logger.error(f"Quote image creation failed: {e}")
            return None

    def _create_carousel(
        self, topic: str, points: List[str], niche: str, colors: dict, timestamp: str
    ) -> List[GeneratedImage]:
        """Create carousel slides (1080x1080 each)."""
        slides = []
        w, h = 1080, 1080

        # Slide 1: Title card
        try:
            img = Image.new("RGB", (w, h), colors["bg"])
            draw = ImageDraw.Draw(img)
            font = self._get_font(44)
            font_small = self._get_font(24)

            draw.rectangle([(0, 0), (w, 6)], fill=colors["accent"])

            wrapped = textwrap.fill(topic, width=22)
            y_pos = h // 3
            for line in wrapped.split("\n"):
                bbox = draw.textbbox((0, 0), line, font=font)
                text_w = bbox[2] - bbox[0]
                draw.text(((w - text_w) // 2, y_pos), line, fill=colors["text"], font=font)
                y_pos += 55

            draw.text((80, h - 100), "Swipe for insights \u2192", fill=colors["accent"], font=font_small)

            path = str(self._output_dir / f"carousel_{timestamp}_00.png")
            img.save(path, "PNG", quality=95)
            slides.append(GeneratedImage(path=path, style="carousel_slide", dimensions=(w, h), platform="instagram"))
        except Exception as e:
            logger.error(f"Carousel title slide failed: {e}")

        # Content slides
        for i, point in enumerate(points[:8], 1):
            try:
                img = Image.new("RGB", (w, h), colors["bg"])
                draw = ImageDraw.Draw(img)
                font = self._get_font(36)
                font_num = self._get_font(72)
                font_small = self._get_font(22)

                # Slide number
                draw.text((80, 80), str(i), fill=colors["accent"], font=font_num)

                # Point text
                wrapped = textwrap.fill(point, width=30)
                y_pos = 250
                for line in wrapped.split("\n"):
                    draw.text((80, y_pos), line, fill=colors["text"], font=font)
                    y_pos += 48

                # Progress dots
                total = min(len(points), 8) + 2  # +2 for title and CTA
                dot_y = h - 60
                dot_start = (w - total * 20) // 2
                for d in range(total):
                    color = colors["accent"] if d == i else colors["secondary"]
                    draw.ellipse(
                        [(dot_start + d * 20, dot_y), (dot_start + d * 20 + 10, dot_y + 10)],
                        fill=color,
                    )

                path = str(self._output_dir / f"carousel_{timestamp}_{i:02d}.png")
                img.save(path, "PNG", quality=95)
                slides.append(GeneratedImage(path=path, style="carousel_slide", dimensions=(w, h), platform="instagram"))
            except Exception as e:
                logger.error(f"Carousel slide {i} failed: {e}")

        return slides

    def _get_font(self, size: int):
        """Get a font, falling back to default if custom not available."""
        if self._font_path:
            try:
                return ImageFont.truetype(self._font_path, size)
            except Exception:
                pass

        # Try common system fonts
        for font_name in ["DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttf", "FreeSans.ttf"]:
            try:
                return ImageFont.truetype(font_name, size)
            except Exception:
                continue

        # Try finding fonts in common locations
        for font_dir in ["/usr/share/fonts", "/usr/local/share/fonts", "C:/Windows/Fonts"]:
            try:
                for root, dirs, files in os.walk(font_dir):
                    for f in files:
                        if f.endswith(".ttf"):
                            return ImageFont.truetype(os.path.join(root, f), size)
            except Exception:
                continue

        return ImageFont.load_default()
