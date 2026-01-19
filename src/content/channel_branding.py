"""
Channel Branding Generator - Profile Pictures and Visual Assets

Generates professional profile pictures and branding assets for YouTube channels
using PIL/Pillow.

Usage:
    from src.content.channel_branding import ChannelBrandingGenerator

    generator = ChannelBrandingGenerator()

    # Generate profile picture
    path = generator.generate_profile_picture("money_blueprints", "finance")

    # Get brand colors for a niche
    colors = generator.get_brand_colors("finance")
"""

import os
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from loguru import logger

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
except ImportError:
    logger.warning("PIL/Pillow not installed. Install with: pip install pillow")
    Image = None


@dataclass
class BrandColors:
    """Color scheme for a brand."""
    primary: str
    secondary: str
    accent: str
    background: str

    def to_rgb(self, color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        color = color.lstrip('#')
        return tuple(int(color[i:i+2], 16) for i in (0, 2, 4))

    @property
    def primary_rgb(self) -> Tuple[int, int, int]:
        return self.to_rgb(self.primary)

    @property
    def secondary_rgb(self) -> Tuple[int, int, int]:
        return self.to_rgb(self.secondary)

    @property
    def accent_rgb(self) -> Tuple[int, int, int]:
        return self.to_rgb(self.accent)

    @property
    def background_rgb(self) -> Tuple[int, int, int]:
        return self.to_rgb(self.background)


# Brand color schemes for each niche
BRAND_COLORS = {
    "finance": BrandColors(
        primary="#1a1a2e",      # Dark blue
        secondary="#FFD700",    # Gold
        accent="#00d4aa",       # Green/teal
        background="#0a0a14"    # Very dark blue
    ),
    "psychology": BrandColors(
        primary="#0f0f1a",      # Deep blue/black
        secondary="#9b59b6",    # Purple
        accent="#E0E0FF",       # Light purple/white
        background="#050510"    # Almost black
    ),
    "storytelling": BrandColors(
        primary="#0d0d0d",      # Black
        secondary="#e74c3c",    # Dark red
        accent="#FFD700",       # Gold
        background="#050505"    # Very dark
    ),
    "default": BrandColors(
        primary="#1a1a2e",
        secondary="#3498db",
        accent="#ffffff",
        background="#0a0a14"
    )
}

# Recommended fonts for each niche
BRAND_FONTS = {
    "finance": ["Arial Bold", "Helvetica Bold", "Impact", "Bebas Neue"],
    "psychology": ["Georgia", "Playfair Display", "Lora", "Times New Roman"],
    "storytelling": ["Cinzel", "Trajan Pro", "Oswald", "Impact"],
    "default": ["Arial Bold", "Helvetica", "Verdana"]
}


class ChannelBrandingGenerator:
    """
    Generate professional branding assets for YouTube channels.

    Creates profile pictures, banners, and other visual assets
    using PIL/Pillow with professional gradients and effects.
    """

    def __init__(self, output_dir: str = None):
        """
        Initialize the branding generator.

        Args:
            output_dir: Directory for output files (default: assets/branding/)
        """
        if Image is None:
            raise ImportError("PIL/Pillow is required. Install with: pip install pillow")

        self.output_dir = Path(output_dir or "assets/branding")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"ChannelBrandingGenerator initialized. Output: {self.output_dir}")

    def get_brand_colors(self, niche: str) -> BrandColors:
        """
        Get the brand color scheme for a niche.

        Args:
            niche: Content niche (finance, psychology, storytelling)

        Returns:
            BrandColors dataclass with color values
        """
        return BRAND_COLORS.get(niche, BRAND_COLORS["default"])

    def get_brand_fonts(self, niche: str) -> List[str]:
        """
        Get recommended fonts for a niche.

        Args:
            niche: Content niche

        Returns:
            List of font names in preference order
        """
        return BRAND_FONTS.get(niche, BRAND_FONTS["default"])

    def _create_radial_gradient(
        self,
        size: Tuple[int, int],
        center_color: Tuple[int, int, int],
        edge_color: Tuple[int, int, int],
        center: Tuple[float, float] = (0.5, 0.5)
    ) -> Image.Image:
        """Create a radial gradient image."""
        width, height = size
        img = Image.new('RGB', size)
        pixels = img.load()

        cx = int(width * center[0])
        cy = int(height * center[1])
        max_dist = math.sqrt(cx**2 + cy**2)

        for y in range(height):
            for x in range(width):
                dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                ratio = min(dist / max_dist, 1.0)

                r = int(center_color[0] * (1 - ratio) + edge_color[0] * ratio)
                g = int(center_color[1] * (1 - ratio) + edge_color[1] * ratio)
                b = int(center_color[2] * (1 - ratio) + edge_color[2] * ratio)

                pixels[x, y] = (r, g, b)

        return img

    def _draw_circle(
        self,
        draw: ImageDraw.Draw,
        center: Tuple[int, int],
        radius: int,
        fill: Tuple[int, int, int] = None,
        outline: Tuple[int, int, int] = None,
        width: int = 1
    ):
        """Draw a circle with optional fill and outline."""
        x, y = center
        bbox = [x - radius, y - radius, x + radius, y + radius]
        if fill:
            draw.ellipse(bbox, fill=fill, outline=outline, width=width)
        elif outline:
            draw.ellipse(bbox, outline=outline, width=width)

    def _get_font(self, size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
        """Get a font, falling back to default if needed."""
        font_names = [
            "arial.ttf", "Arial.ttf",
            "arialbd.ttf", "Arial Bold.ttf",
            "impact.ttf", "Impact.ttf",
            "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"
        ]

        for font_name in font_names:
            try:
                return ImageFont.truetype(font_name, size)
            except (OSError, IOError):
                continue

        # Fall back to default
        return ImageFont.load_default()

    def _generate_finance_profile(self, size: int = 800) -> Image.Image:
        """
        Generate profile picture for finance channel.

        Design: Dark blue gradient background with gold dollar sign
        and upward trending green arrow/chart line.
        """
        colors = self.get_brand_colors("finance")

        # Create radial gradient background
        img = self._create_radial_gradient(
            (size, size),
            colors.primary_rgb,
            colors.background_rgb,
            center=(0.5, 0.4)
        )

        draw = ImageDraw.Draw(img)
        center = size // 2

        # Draw outer glow circle
        for i in range(5, 0, -1):
            alpha = 30 + (5 - i) * 10
            glow_color = (
                min(255, colors.secondary_rgb[0]),
                min(255, colors.secondary_rgb[1]),
                min(255, colors.secondary_rgb[2])
            )
            self._draw_circle(
                draw, (center, center),
                int(size * 0.42) + i * 3,
                outline=glow_color,
                width=2
            )

        # Draw main circle border
        self._draw_circle(
            draw, (center, center),
            int(size * 0.42),
            outline=colors.secondary_rgb,
            width=8
        )

        # Draw dollar sign
        font_size = int(size * 0.5)
        font = self._get_font(font_size, bold=True)

        text = "$"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        text_x = center - text_width // 2
        text_y = center - text_height // 2 - int(size * 0.05)

        # Draw gold dollar sign with shadow
        draw.text((text_x + 4, text_y + 4), text, fill=(50, 50, 50), font=font)
        draw.text((text_x, text_y), text, fill=colors.secondary_rgb, font=font)

        # Draw upward trending arrow/line
        arrow_start = (int(size * 0.25), int(size * 0.75))
        arrow_mid = (int(size * 0.5), int(size * 0.55))
        arrow_end = (int(size * 0.75), int(size * 0.65))

        # Draw trend line
        draw.line([arrow_start, arrow_mid, arrow_end], fill=colors.accent_rgb, width=6)

        # Draw arrow head
        arrow_tip = arrow_end
        draw.polygon([
            arrow_tip,
            (arrow_tip[0] - 15, arrow_tip[1] + 15),
            (arrow_tip[0] - 15, arrow_tip[1] - 15)
        ], fill=colors.accent_rgb)

        return img

    def _generate_psychology_profile(self, size: int = 800) -> Image.Image:
        """
        Generate profile picture for psychology channel.

        Design: Deep purple/blue gradient with stylized brain outline
        and key/unlock symbol. Glowing ethereal effect.
        """
        colors = self.get_brand_colors("psychology")

        # Create radial gradient background
        img = self._create_radial_gradient(
            (size, size),
            colors.secondary_rgb,
            colors.background_rgb,
            center=(0.5, 0.5)
        )

        # Apply slight blur for ethereal effect
        img = img.filter(ImageFilter.GaussianBlur(radius=2))

        draw = ImageDraw.Draw(img)
        center = size // 2

        # Draw outer glow rings
        for i in range(8, 0, -1):
            glow_intensity = int(80 + (8 - i) * 15)
            glow_color = (
                min(255, glow_intensity),
                min(255, int(glow_intensity * 0.6)),
                min(255, glow_intensity)
            )
            self._draw_circle(
                draw, (center, center),
                int(size * 0.38) + i * 5,
                outline=glow_color,
                width=1
            )

        # Draw main circle
        self._draw_circle(
            draw, (center, center),
            int(size * 0.38),
            outline=colors.accent_rgb,
            width=4
        )

        # Draw stylized brain shape using curves
        brain_color = colors.accent_rgb

        # Left hemisphere
        left_center = (center - int(size * 0.1), center - int(size * 0.05))
        draw.arc(
            [left_center[0] - int(size * 0.18), left_center[1] - int(size * 0.2),
             left_center[0] + int(size * 0.12), left_center[1] + int(size * 0.2)],
            start=180, end=360,
            fill=brain_color, width=4
        )

        # Right hemisphere
        right_center = (center + int(size * 0.1), center - int(size * 0.05))
        draw.arc(
            [right_center[0] - int(size * 0.12), right_center[1] - int(size * 0.2),
             right_center[0] + int(size * 0.18), right_center[1] + int(size * 0.2)],
            start=180, end=360,
            fill=brain_color, width=4
        )

        # Brain stem
        stem_top = (center, center + int(size * 0.1))
        stem_bottom = (center, center + int(size * 0.25))
        draw.line([stem_top, stem_bottom], fill=brain_color, width=4)

        # Draw key symbol at bottom
        key_y = center + int(size * 0.2)
        key_x = center

        # Key circle (lock part)
        key_radius = int(size * 0.08)
        self._draw_circle(
            draw, (key_x, key_y),
            key_radius,
            outline=colors.secondary_rgb,
            width=4
        )

        # Key shaft
        shaft_start = (key_x, key_y + key_radius)
        shaft_end = (key_x, key_y + key_radius + int(size * 0.1))
        draw.line([shaft_start, shaft_end], fill=colors.secondary_rgb, width=4)

        # Key teeth
        teeth_y = key_y + key_radius + int(size * 0.06)
        draw.line([(key_x, teeth_y), (key_x + int(size * 0.04), teeth_y)],
                  fill=colors.secondary_rgb, width=4)

        return img

    def _generate_storytelling_profile(self, size: int = 800) -> Image.Image:
        """
        Generate profile picture for storytelling channel.

        Design: Black background with dramatic red lighting,
        open book or film reel silhouette with gold accents.
        """
        colors = self.get_brand_colors("storytelling")

        # Create dramatic gradient with red tint
        img = self._create_radial_gradient(
            (size, size),
            (40, 10, 10),  # Subtle dark red center
            colors.background_rgb,
            center=(0.5, 0.3)
        )

        draw = ImageDraw.Draw(img)
        center = size // 2

        # Draw red atmospheric glow
        for i in range(10, 0, -1):
            glow_color = (
                min(255, int(colors.secondary_rgb[0] * (i / 15))),
                min(255, int(colors.secondary_rgb[1] * (i / 30))),
                min(255, int(colors.secondary_rgb[2] * (i / 30)))
            )
            self._draw_circle(
                draw, (center, int(center * 0.8)),
                int(size * 0.3) + i * 8,
                outline=glow_color,
                width=2
            )

        # Draw open book shape
        book_center = center
        book_y = center - int(size * 0.05)
        book_width = int(size * 0.5)
        book_height = int(size * 0.35)

        # Left page
        left_points = [
            (book_center - int(book_width * 0.05), book_y - int(book_height * 0.5)),
            (book_center - book_width // 2, book_y - int(book_height * 0.4)),
            (book_center - book_width // 2, book_y + int(book_height * 0.4)),
            (book_center - int(book_width * 0.05), book_y + int(book_height * 0.5)),
        ]
        draw.polygon(left_points, outline=colors.accent_rgb, width=3)

        # Right page
        right_points = [
            (book_center + int(book_width * 0.05), book_y - int(book_height * 0.5)),
            (book_center + book_width // 2, book_y - int(book_height * 0.4)),
            (book_center + book_width // 2, book_y + int(book_height * 0.4)),
            (book_center + int(book_width * 0.05), book_y + int(book_height * 0.5)),
        ]
        draw.polygon(right_points, outline=colors.accent_rgb, width=3)

        # Book spine
        spine_top = (book_center, book_y - int(book_height * 0.5))
        spine_bottom = (book_center, book_y + int(book_height * 0.5))
        draw.line([spine_top, spine_bottom], fill=colors.accent_rgb, width=4)

        # Page lines (left)
        for i in range(3):
            line_y = book_y - int(book_height * 0.2) + i * int(book_height * 0.15)
            line_start = (book_center - book_width // 2 + 20, line_y)
            line_end = (book_center - 15, line_y)
            draw.line([line_start, line_end], fill=colors.secondary_rgb, width=2)

        # Page lines (right)
        for i in range(3):
            line_y = book_y - int(book_height * 0.2) + i * int(book_height * 0.15)
            line_start = (book_center + 15, line_y)
            line_end = (book_center + book_width // 2 - 20, line_y)
            draw.line([line_start, line_end], fill=colors.secondary_rgb, width=2)

        # Draw film reel elements on sides
        reel_y = center + int(size * 0.25)

        # Left reel
        self._draw_circle(
            draw, (int(size * 0.2), reel_y),
            int(size * 0.08),
            outline=colors.secondary_rgb,
            width=3
        )
        self._draw_circle(
            draw, (int(size * 0.2), reel_y),
            int(size * 0.03),
            fill=colors.secondary_rgb
        )

        # Right reel
        self._draw_circle(
            draw, (int(size * 0.8), reel_y),
            int(size * 0.08),
            outline=colors.secondary_rgb,
            width=3
        )
        self._draw_circle(
            draw, (int(size * 0.8), reel_y),
            int(size * 0.03),
            fill=colors.secondary_rgb
        )

        # Film strip connecting reels
        strip_y = reel_y
        draw.line(
            [(int(size * 0.28), strip_y), (int(size * 0.72), strip_y)],
            fill=colors.secondary_rgb,
            width=12
        )

        # Film perforations
        for i in range(5):
            perf_x = int(size * 0.32) + i * int(size * 0.1)
            draw.rectangle(
                [perf_x - 4, strip_y - 8, perf_x + 4, strip_y + 8],
                fill=colors.background_rgb
            )

        # Draw outer circle border
        self._draw_circle(
            draw, (center, center),
            int(size * 0.45),
            outline=colors.secondary_rgb,
            width=6
        )

        return img

    def generate_profile_picture(
        self,
        channel_id: str,
        niche: str,
        style: str = "default",
        size: int = 800
    ) -> str:
        """
        Generate a profile picture for a channel.

        Args:
            channel_id: Channel identifier (e.g., "money_blueprints")
            niche: Content niche (finance, psychology, storytelling)
            style: Style variant (default only for now)
            size: Image size in pixels (default 800x800)

        Returns:
            Path to the generated image file
        """
        logger.info(f"Generating profile picture for {channel_id} ({niche})")

        # Generate based on niche
        if niche == "finance":
            img = self._generate_finance_profile(size)
        elif niche == "psychology":
            img = self._generate_psychology_profile(size)
        elif niche == "storytelling":
            img = self._generate_storytelling_profile(size)
        else:
            # Default to finance style
            img = self._generate_finance_profile(size)

        # Save image
        output_path = self.output_dir / f"{channel_id}_profile.png"
        img.save(output_path, "PNG", quality=95)

        logger.success(f"Profile picture saved: {output_path}")
        return str(output_path)

    def generate_banner(
        self,
        channel_id: str,
        niche: str,
        style: str = "default"
    ) -> str:
        """
        Generate a channel banner.

        Args:
            channel_id: Channel identifier
            niche: Content niche
            style: Style variant

        Returns:
            Path to the generated banner file
        """
        logger.info(f"Generating banner for {channel_id} ({niche})")

        # YouTube banner dimensions: 2560 x 1440
        width, height = 2560, 1440
        colors = self.get_brand_colors(niche)

        # Create gradient background
        img = self._create_radial_gradient(
            (width, height),
            colors.primary_rgb,
            colors.background_rgb,
            center=(0.5, 0.5)
        )

        draw = ImageDraw.Draw(img)

        # Add accent lines
        for i in range(5):
            y = 200 + i * 250
            draw.line(
                [(0, y), (width, y)],
                fill=(*colors.secondary_rgb, 30),
                width=2
            )

        # Save
        output_path = self.output_dir / f"{channel_id}_banner.png"
        img.save(output_path, "PNG", quality=95)

        logger.success(f"Banner saved: {output_path}")
        return str(output_path)

    def generate_all_assets(self, channel_id: str, niche: str) -> Dict[str, str]:
        """
        Generate all branding assets for a channel.

        Args:
            channel_id: Channel identifier
            niche: Content niche

        Returns:
            Dict with paths to all generated assets
        """
        return {
            "profile_picture": self.generate_profile_picture(channel_id, niche),
            "banner": self.generate_banner(channel_id, niche),
            "brand_colors": self.get_brand_colors(niche).__dict__,
            "brand_fonts": self.get_brand_fonts(niche)
        }


def main():
    """CLI entry point for generating branding assets."""
    import sys

    if len(sys.argv) < 2:
        print("""
Channel Branding Generator
==========================

Generate professional profile pictures and banners for YouTube channels.

Usage:
    python -m src.content.channel_branding <channel_id> <niche>
    python -m src.content.channel_branding all

Examples:
    python -m src.content.channel_branding money_blueprints finance
    python -m src.content.channel_branding mind_unlocked psychology
    python -m src.content.channel_branding untold_stories storytelling
    python -m src.content.channel_branding all

Niches: finance, psychology, storytelling
        """)
        return

    generator = ChannelBrandingGenerator()

    if sys.argv[1] == "all":
        # Generate for all channels
        channels = [
            ("money_blueprints", "finance"),
            ("mind_unlocked", "psychology"),
            ("untold_stories", "storytelling")
        ]

        for channel_id, niche in channels:
            print(f"\nGenerating assets for {channel_id}...")
            assets = generator.generate_all_assets(channel_id, niche)
            print(f"  Profile: {assets['profile_picture']}")
            print(f"  Banner: {assets['banner']}")

        print(f"\nAll assets generated in: {generator.output_dir}")

    else:
        channel_id = sys.argv[1]
        niche = sys.argv[2] if len(sys.argv) > 2 else "default"

        assets = generator.generate_all_assets(channel_id, niche)
        print(f"\nAssets generated for {channel_id}:")
        print(f"  Profile: {assets['profile_picture']}")
        print(f"  Banner: {assets['banner']}")
        print(f"  Colors: {assets['brand_colors']}")


if __name__ == "__main__":
    main()
