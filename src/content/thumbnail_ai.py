"""
AI-Powered Thumbnail Generator - Generate Thumbnails with AI Faces

Uses Replicate API (Flux model) to generate realistic AI faces with
emotions tailored to specific niches for higher CTR thumbnails.

Features:
- Niche-specific emotion prompts (finance, psychology, storytelling)
- AI face generation using Replicate's Flux model
- A/B test variant generation
- Fallback to standard text-only thumbnails on failure
- Compositing AI faces with branded backgrounds

Usage:
    from src.content.thumbnail_ai import AIThumbnailGenerator

    generator = AIThumbnailGenerator()
    path = generator.generate_with_ai_face(
        title="How I Made $10,000",
        niche="finance",
        emotion="confident",
        output_path="output/thumb.png"
    )

CLI:
    python run.py thumbnail "My Title" --niche finance --ai
    python run.py thumbnail "My Title" --niche finance --variants 3
"""

import os
import re
import io
import time
import requests
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from loguru import logger

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
except ImportError:
    logger.warning("PIL/Pillow not installed. Install with: pip install pillow")
    Image = None

try:
    import replicate
except ImportError:
    logger.warning("Replicate not installed. Install with: pip install replicate")
    replicate = None

from .thumbnail_generator import ThumbnailGenerator
from .channel_branding import BRAND_COLORS, BrandColors


# Niche-specific emotion mappings with detailed prompts for AI face generation
NICHE_EMOTIONS = {
    "finance": {
        "confident": {
            "prompt": "professional confident businessman in suit, direct eye contact, slight smile, studio lighting, dark background, high quality portrait, 4k, sharp focus",
            "expression": "confident",
            "description": "Confident businessman"
        },
        "surprised": {
            "prompt": "surprised person looking at smartphone screen with wide eyes and open mouth, shocked expression, studio lighting, dark background, high quality portrait, 4k",
            "expression": "surprised",
            "description": "Surprised at phone"
        },
        "thoughtful": {
            "prompt": "thoughtful professional person with hand on chin, contemplating, serious expression, studio lighting, dark background, high quality portrait, 4k",
            "expression": "thoughtful",
            "description": "Thoughtful pose"
        },
        "excited": {
            "prompt": "excited person celebrating success with raised fist, big smile, energetic expression, studio lighting, dark background, high quality portrait, 4k",
            "expression": "excited",
            "description": "Celebrating success"
        },
        "serious": {
            "prompt": "serious businessman with stern look, focused intense gaze, professional attire, studio lighting, dark background, high quality portrait, 4k",
            "expression": "serious",
            "description": "Serious businessman"
        }
    },
    "psychology": {
        "secretive": {
            "prompt": "mysterious person with finger on lips in shh gesture, knowing look, slight smile, dramatic lighting, dark background, high quality portrait, 4k",
            "expression": "secretive",
            "description": "Shh gesture"
        },
        "shocked": {
            "prompt": "person with wide-eyed shocked expression, eyebrows raised, mouth slightly open, dramatic lighting, dark background, high quality portrait, 4k",
            "expression": "shocked",
            "description": "Wide-eyed shocked"
        },
        "knowing": {
            "prompt": "person with knowing smile, one eyebrow slightly raised, wise mysterious expression, dramatic lighting, dark background, high quality portrait, 4k",
            "expression": "knowing",
            "description": "Knowing smile"
        },
        "intrigued": {
            "prompt": "person with intrigued curious expression, head tilted slightly, eyebrows furrowed, dramatic lighting, dark background, high quality portrait, 4k",
            "expression": "intrigued",
            "description": "Intrigued expression"
        },
        "contemplative": {
            "prompt": "person in deep thought with eyes looking up, contemplative philosophical expression, dramatic lighting, dark background, high quality portrait, 4k",
            "expression": "contemplative",
            "description": "Deep in thought"
        }
    },
    "storytelling": {
        "horrified": {
            "prompt": "person with horrified terrified expression, wide eyes full of fear, mouth open in shock, dramatic cinematic lighting, dark background, high quality portrait, 4k",
            "expression": "horrified",
            "description": "Horrified expression"
        },
        "curious": {
            "prompt": "person leaning forward with intense curious expression, eyes wide with interest, dramatic lighting, dark background, high quality portrait, 4k",
            "expression": "curious",
            "description": "Intense curious lean"
        },
        "surprised": {
            "prompt": "person with hand over mouth in surprise, wide eyes, shocked gasping expression, dramatic lighting, dark background, high quality portrait, 4k",
            "expression": "surprised",
            "description": "Hand over mouth surprised"
        },
        "mysterious": {
            "prompt": "person with mysterious enigmatic expression, half face in shadow, intense gaze, dramatic cinematic lighting, dark background, high quality portrait, 4k",
            "expression": "mysterious",
            "description": "Mysterious half-shadow"
        },
        "dramatic": {
            "prompt": "person with dramatic intense expression, strong emotions, powerful gaze, cinematic lighting with red tones, dark background, high quality portrait, 4k",
            "expression": "dramatic",
            "description": "Dramatic intensity"
        }
    }
}

# Default emotions for each niche (first one used if not specified)
DEFAULT_EMOTIONS = {
    "finance": "confident",
    "psychology": "secretive",
    "storytelling": "horrified"
}


class AIThumbnailGenerator:
    """
    AI-powered thumbnail generator with face generation capabilities.

    Uses Replicate API (Flux model) to generate realistic faces with
    emotions tailored to specific niches, then composites them with
    branded backgrounds and text overlays.
    """

    # Face positioning on thumbnail (from right edge)
    FACE_POSITION_RIGHT = 0.35  # Face takes up right 35% of thumbnail
    FACE_SCALE = 0.9  # Scale factor for face height relative to thumbnail

    def __init__(self, output_dir: str = None):
        """
        Initialize the AI thumbnail generator.

        Args:
            output_dir: Directory for output files (default: assets/thumbnails/)
        """
        if Image is None:
            raise ImportError("PIL/Pillow is required. Install with: pip install pillow")

        self.output_dir = Path(output_dir or "assets/thumbnails")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize base thumbnail generator for fallback and utilities
        self.base_generator = ThumbnailGenerator(output_dir=str(self.output_dir))

        # Get Replicate API key from environment
        self.replicate_api_key = os.getenv("REPLICATE_API_KEY")
        if not self.replicate_api_key:
            logger.warning("REPLICATE_API_KEY not set. AI face generation will not be available.")

        logger.info(f"AIThumbnailGenerator initialized. Output: {self.output_dir}")

    def _get_emotion_prompt(self, emotion: str, niche: str) -> Dict:
        """
        Get the emotion prompt configuration for a niche.

        Args:
            emotion: Emotion name (e.g., "confident", "surprised")
            niche: Content niche

        Returns:
            Dict with prompt, expression, and description
        """
        niche_emotions = NICHE_EMOTIONS.get(niche, NICHE_EMOTIONS["finance"])

        if emotion in niche_emotions:
            return niche_emotions[emotion]

        # Return default emotion for niche
        default_emotion = DEFAULT_EMOTIONS.get(niche, "confident")
        return niche_emotions.get(default_emotion, list(niche_emotions.values())[0])

    def _generate_ai_face(self, emotion_prompt: str, niche: str) -> Optional[Image.Image]:
        """
        Generate an AI face using Replicate API (Flux model).

        Args:
            emotion_prompt: Full prompt for face generation
            niche: Content niche for styling context

        Returns:
            PIL Image of generated face, or None if generation fails
        """
        if not self.replicate_api_key:
            logger.error("REPLICATE_API_KEY not set. Cannot generate AI face.")
            return None

        if replicate is None:
            logger.error("Replicate library not installed. Install with: pip install replicate")
            return None

        try:
            logger.info(f"Generating AI face with prompt: {emotion_prompt[:50]}...")

            # Set the API token
            os.environ["REPLICATE_API_TOKEN"] = self.replicate_api_key

            # Use Flux Schnell model for fast generation
            # Alternative: "black-forest-labs/flux-dev" for higher quality
            output = replicate.run(
                "black-forest-labs/flux-schnell",
                input={
                    "prompt": emotion_prompt,
                    "num_outputs": 1,
                    "aspect_ratio": "3:4",  # Portrait orientation
                    "output_format": "png",
                    "output_quality": 90
                }
            )

            # Output is a list of URLs or file objects
            if output and len(output) > 0:
                image_url = output[0]

                # Handle FileOutput object from replicate
                if hasattr(image_url, 'url'):
                    image_url = image_url.url
                elif hasattr(image_url, 'read'):
                    # It's a file-like object
                    img = Image.open(image_url)
                    logger.success("AI face generated successfully")
                    return img

                # Download the image
                response = requests.get(str(image_url), timeout=30)
                response.raise_for_status()

                img = Image.open(io.BytesIO(response.content))
                logger.success("AI face generated successfully")
                return img

            logger.error("No output received from Replicate API")
            return None

        except replicate.exceptions.ReplicateError as e:
            logger.error(f"Replicate API error: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download generated image: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error generating AI face: {e}")
            return None

    def _composite_face_on_thumbnail(
        self,
        background: Image.Image,
        face: Image.Image,
        position: str = "right"
    ) -> Image.Image:
        """
        Composite an AI-generated face onto the thumbnail background.

        Args:
            background: Base thumbnail background image
            face: AI-generated face image
            position: Face position ("right", "left", "center")

        Returns:
            Composited thumbnail image
        """
        result = background.copy()
        thumb_width, thumb_height = result.size

        # Calculate face size (scale to fit height with some padding)
        face_height = int(thumb_height * self.FACE_SCALE)
        face_aspect = face.width / face.height
        face_width = int(face_height * face_aspect)

        # Resize face
        face_resized = face.resize((face_width, face_height), Image.Resampling.LANCZOS)

        # Calculate position
        if position == "right":
            x = thumb_width - face_width - int(thumb_width * 0.02)  # 2% padding from right
        elif position == "left":
            x = int(thumb_width * 0.02)  # 2% padding from left
        else:  # center
            x = (thumb_width - face_width) // 2

        y = (thumb_height - face_height) // 2

        # Create gradient mask for smooth blending
        mask = Image.new('L', (face_width, face_height), 255)
        mask_draw = ImageDraw.Draw(mask)

        # Fade on the inner edge (towards text)
        fade_width = int(face_width * 0.3)
        if position == "right":
            # Fade on left side of face
            for i in range(fade_width):
                alpha = int(255 * (i / fade_width))
                mask_draw.line([(i, 0), (i, face_height)], fill=alpha)
        elif position == "left":
            # Fade on right side of face
            for i in range(fade_width):
                alpha = int(255 * (i / fade_width))
                mask_draw.line([(face_width - i - 1, 0), (face_width - i - 1, face_height)], fill=alpha)

        # Apply slight blur to mask for smoother blending
        mask = mask.filter(ImageFilter.GaussianBlur(radius=10))

        # Convert face to RGBA if needed
        if face_resized.mode != 'RGBA':
            face_resized = face_resized.convert('RGBA')

        # Paste face with mask
        result.paste(face_resized, (x, y), mask)

        return result

    def _add_text_to_ai_thumbnail(
        self,
        image: Image.Image,
        text: str,
        niche: str,
        face_position: str = "right"
    ) -> Image.Image:
        """
        Add text overlay to AI thumbnail, positioning opposite to face.

        Args:
            image: Thumbnail image with face
            text: Text to overlay
            niche: Content niche for styling
            face_position: Where the face is positioned

        Returns:
            Image with text overlay
        """
        result = image.copy()
        draw = ImageDraw.Draw(result)
        colors = BRAND_COLORS.get(niche, BRAND_COLORS["default"])

        width, height = result.size

        # Text goes on opposite side of face
        if face_position == "right":
            text_area_width = int(width * 0.6)  # Left 60%
            text_x_center = text_area_width // 2
        else:
            text_area_width = int(width * 0.6)
            text_x_center = width - text_area_width // 2

        # Calculate font size to fit text area
        max_width = int(text_area_width * 0.9)
        font_size = self.base_generator._calculate_text_size(text, max_width)
        font = self.base_generator._get_font(font_size)

        # Calculate text dimensions
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Position text
        x = text_x_center - text_width // 2
        y = (height - text_height) // 2

        # Draw shadow
        shadow_offset = max(4, font_size // 30)
        for dx in range(-shadow_offset, shadow_offset + 1):
            for dy in range(-shadow_offset, shadow_offset + 1):
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), text, fill=(0, 0, 0), font=font)

        # Draw outline
        outline_width = max(3, font_size // 40)
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if abs(dx) + abs(dy) <= outline_width * 2:
                    draw.text((x + dx, y + dy), text, fill=(0, 0, 0), font=font)

        # Draw main text
        draw.text((x, y), text, fill=colors.secondary_rgb, font=font)

        return result

    def generate_with_ai_face(
        self,
        title: str,
        niche: str,
        emotion: str = None,
        output_path: str = None,
        face_position: str = "right"
    ) -> str:
        """
        Generate a thumbnail with an AI-generated face.

        Args:
            title: Video title (key words will be extracted)
            niche: Content niche (finance, psychology, storytelling)
            emotion: Emotion for face (uses niche default if not specified)
            output_path: Output file path
            face_position: Where to place face ("right", "left")

        Returns:
            Path to generated thumbnail
        """
        logger.info(f"Generating AI thumbnail for: {title} ({niche}, {emotion or 'default'})")

        # Use default emotion if not specified
        if not emotion:
            emotion = DEFAULT_EMOTIONS.get(niche, "confident")

        # Get emotion prompt
        emotion_config = self._get_emotion_prompt(emotion, niche)
        prompt = emotion_config["prompt"]

        # Generate AI face
        face_image = self._generate_ai_face(prompt, niche)

        if face_image is None:
            logger.warning("AI face generation failed. Falling back to text-only thumbnail.")
            return self.base_generator.generate(
                title=title,
                niche=niche,
                output_path=output_path
            )

        # Create background using base generator's method
        colors = BRAND_COLORS.get(niche, BRAND_COLORS["default"])
        background = self.base_generator._create_gradient_background(colors, "radial")
        background = self.base_generator._apply_vignette(background, strength=0.5)

        # Composite face onto background
        thumbnail = self._composite_face_on_thumbnail(background, face_image, face_position)

        # Extract key words and add text
        key_words = self.base_generator._extract_key_words(title)
        logger.info(f"Extracted key words: {key_words}")

        thumbnail = self._add_text_to_ai_thumbnail(thumbnail, key_words, niche, face_position)

        # Add accent elements (subdued)
        thumbnail = self.base_generator._add_accent_elements(thumbnail, niche)

        # Generate output path if not provided
        if not output_path:
            safe_title = re.sub(r'[^\w\s-]', '', title)[:30].strip().replace(' ', '_')
            output_path = str(self.output_dir / f"thumb_ai_{safe_title}_{niche}_{emotion}.png")

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save thumbnail
        thumbnail.save(output_path, "PNG", quality=95)
        logger.success(f"AI thumbnail saved: {output_path}")

        return output_path

    def generate_ab_test_variants(
        self,
        title: str,
        niche: str,
        count: int = 3
    ) -> List[str]:
        """
        Generate multiple thumbnail variants for A/B testing.

        Creates variants with different emotions, face positions,
        and styling for testing which performs best.

        Args:
            title: Video title
            niche: Content niche
            count: Number of variants to generate (default: 3)

        Returns:
            List of paths to generated thumbnails
        """
        logger.info(f"Generating {count} A/B test variants for: {title}")

        variants = []
        niche_emotions = NICHE_EMOTIONS.get(niche, NICHE_EMOTIONS["finance"])
        emotions = list(niche_emotions.keys())
        positions = ["right", "left"]

        for i in range(count):
            # Rotate through emotions and positions
            emotion = emotions[i % len(emotions)]
            position = positions[i % len(positions)]

            safe_title = re.sub(r'[^\w\s-]', '', title)[:20].strip().replace(' ', '_')
            output_path = str(self.output_dir / f"thumb_ai_{safe_title}_{niche}_v{i+1}_{emotion}.png")

            try:
                path = self.generate_with_ai_face(
                    title=title,
                    niche=niche,
                    emotion=emotion,
                    output_path=output_path,
                    face_position=position
                )
                variants.append(path)

                # Small delay between API calls to avoid rate limiting
                if i < count - 1:
                    time.sleep(1)

            except Exception as e:
                logger.error(f"Failed to generate variant {i+1}: {e}")
                # Try fallback to text-only variant
                try:
                    fallback_path = self.base_generator.generate(
                        title=title,
                        niche=niche,
                        output_path=output_path.replace("_ai_", "_fallback_"),
                        text_position=["center", "top", "bottom"][i % 3]
                    )
                    variants.append(fallback_path)
                except Exception as e2:
                    logger.error(f"Fallback also failed for variant {i+1}: {e2}")

        logger.info(f"Generated {len(variants)} variants")
        return variants

    def get_available_emotions(self, niche: str) -> List[Dict]:
        """
        Get list of available emotions for a niche.

        Args:
            niche: Content niche

        Returns:
            List of emotion configurations
        """
        niche_emotions = NICHE_EMOTIONS.get(niche, NICHE_EMOTIONS["finance"])
        return [
            {"name": name, **config}
            for name, config in niche_emotions.items()
        ]


def main():
    """CLI entry point for AI thumbnail generation."""
    import sys
    import argparse

    if len(sys.argv) < 2:
        print("""
AI-Powered Thumbnail Generator
==============================

Generate viral thumbnails with AI-generated faces for higher CTR.

Usage:
    python -m src.content.thumbnail_ai "<title>" --niche <niche> [options]

Examples:
    python -m src.content.thumbnail_ai "How I Made $10,000" --niche finance
    python -m src.content.thumbnail_ai "Dark Psychology Secrets" --niche psychology --emotion secretive
    python -m src.content.thumbnail_ai "The Untold Story" --niche storytelling --variants 3

Options:
    --niche <niche>      Content niche: finance, psychology, storytelling (required)
    --emotion <emotion>  Face emotion (niche-specific, see below)
    --output <path>      Output file path (optional)
    --variants <n>       Generate N A/B test variants
    --position <pos>     Face position: right, left (default: right)
    --list-emotions      List available emotions for the niche

Emotions by Niche:
    finance:      confident, surprised, thoughtful, excited, serious
    psychology:   secretive, shocked, knowing, intrigued, contemplative
    storytelling: horrified, curious, surprised, mysterious, dramatic

Environment Variables:
    REPLICATE_API_KEY    Required for AI face generation
        """)
        return

    parser = argparse.ArgumentParser(description="Generate AI-powered thumbnails")
    parser.add_argument("title", nargs="?", help="Video title")
    parser.add_argument("--niche", required=True,
                       choices=["finance", "psychology", "storytelling", "default"],
                       help="Content niche")
    parser.add_argument("--emotion", help="Face emotion")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--variants", type=int, help="Generate N A/B test variants")
    parser.add_argument("--position", default="right",
                       choices=["right", "left"],
                       help="Face position")
    parser.add_argument("--list-emotions", action="store_true",
                       help="List available emotions for niche")

    args = parser.parse_args()

    generator = AIThumbnailGenerator()

    if args.list_emotions:
        emotions = generator.get_available_emotions(args.niche)
        print(f"\nAvailable emotions for {args.niche}:")
        for e in emotions:
            print(f"  - {e['name']}: {e['description']}")
        return

    if not args.title:
        print("Error: Please provide a video title")
        return

    if args.variants:
        print(f"\nGenerating {args.variants} A/B test variants...")
        paths = generator.generate_ab_test_variants(
            title=args.title,
            niche=args.niche,
            count=args.variants
        )
        print(f"\nGenerated {len(paths)} variants:")
        for path in paths:
            print(f"  - {path}")
    else:
        path = generator.generate_with_ai_face(
            title=args.title,
            niche=args.niche,
            emotion=args.emotion,
            output_path=args.output,
            face_position=args.position
        )
        print(f"\nAI Thumbnail generated: {path}")


if __name__ == "__main__":
    main()
