"""
Thumbnail Agent - Standalone Thumbnail Generation Agent

A production agent for generating high-CTR YouTube thumbnails with AI faces,
text overlays, and niche-specific color schemes.

Features:
- AI face generation using Replicate API (when available)
- High CTR text overlay patterns
- Niche-specific color schemes (finance=green/gold, psychology=blue/purple)
- A/B test variant generation (2-3 per video)
- Mobile-optimized display (1280x720 with large text)
- Predicted CTR scoring

Usage:
    from src.agents.thumbnail_agent import ThumbnailAgent, ThumbnailResult

    agent = ThumbnailAgent()
    result = agent.run(
        title="5 Money Mistakes Costing You $1000",
        niche="finance",
        generate_variants=True,
        use_ai_face=True
    )

    if result.success:
        print(f"Thumbnail: {result.data['thumbnail_file']}")
        print(f"Variants: {result.data['variants']}")
        print(f"Predicted CTR: {result.data['predicted_ctr']}")

CLI:
    python -m src.agents.thumbnail_agent "My Video Title" --niche finance --variants 3
"""

import os
import re
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from loguru import logger

from .base_agent import BaseAgent, AgentResult


# Niche-specific color schemes for high CTR
NICHE_COLOR_SCHEMES = {
    "finance": {
        "primary": "#1a5f1a",    # Deep green (money)
        "secondary": "#FFD700",  # Gold (wealth)
        "accent": "#2E8B57",     # Sea green
        "background": "#0a1f0a",  # Dark green
        "text": "#FFFFFF",       # White text
        "highlight": "#00FF00",  # Bright green
        "ctr_boost_colors": ["#FFD700", "#00FF00", "#32CD32"],
    },
    "psychology": {
        "primary": "#4B0082",    # Indigo (deep/mysterious)
        "secondary": "#9B59B6",  # Purple (psychology)
        "accent": "#E040FB",     # Pink/magenta
        "background": "#0f0f1a", # Dark blue-black
        "text": "#FFFFFF",       # White text
        "highlight": "#00BFFF",  # Deep sky blue
        "ctr_boost_colors": ["#9B59B6", "#E040FB", "#FF69B4"],
    },
    "storytelling": {
        "primary": "#8B0000",    # Dark red (dramatic)
        "secondary": "#FF4500",  # Orange-red
        "accent": "#DC143C",     # Crimson
        "background": "#0d0d0d", # Near black
        "text": "#FFFFFF",       # White text
        "highlight": "#FF6347",  # Tomato red
        "ctr_boost_colors": ["#FF4500", "#DC143C", "#FF0000"],
    },
    "default": {
        "primary": "#1E90FF",    # Dodger blue
        "secondary": "#FFFFFF",  # White
        "accent": "#FFD700",     # Gold
        "background": "#0a0a1a", # Dark
        "text": "#FFFFFF",       # White text
        "highlight": "#00FFFF",  # Cyan
        "ctr_boost_colors": ["#FFD700", "#00FFFF", "#FF69B4"],
    }
}

# High CTR text patterns for thumbnails
HIGH_CTR_PATTERNS = {
    "finance": [
        "{keyword} $$$",
        "{keyword}!",
        "STOP {keyword}",
        "{keyword} NOW",
        "HOW TO {keyword}",
        "{keyword} (SECRET)",
        "{keyword} EXPOSED",
    ],
    "psychology": [
        "DARK {keyword}",
        "{keyword}?!",
        "THEY {keyword}",
        "{keyword} (HIDDEN)",
        "WHY {keyword}",
        "{keyword} TRICKS",
        "MIND {keyword}",
    ],
    "storytelling": [
        "TRUE {keyword}",
        "{keyword}...",
        "UNTOLD {keyword}",
        "{keyword} EXPOSED",
        "WHAT {keyword}",
        "{keyword} REVEALED",
        "THE REAL {keyword}",
    ],
}

# Emotion-to-expression mapping for AI faces
NICHE_EMOTIONS = {
    "finance": ["confident", "surprised", "thoughtful", "excited", "serious"],
    "psychology": ["secretive", "shocked", "knowing", "intrigued", "contemplative"],
    "storytelling": ["horrified", "curious", "surprised", "mysterious", "dramatic"],
}


@dataclass
class ThumbnailResult:
    """
    Result from thumbnail generation.

    Attributes:
        thumbnail_file: Path to the primary thumbnail
        variants: List of variant thumbnail paths for A/B testing
        predicted_ctr: Predicted click-through rate (0.0-1.0)
        color_scheme: Color scheme used
        text_overlay: Text that was overlaid
        ai_face_used: Whether AI face was generated
        emotion: Emotion used for AI face (if applicable)
        file_size_kb: File size in KB
        dimensions: Tuple of (width, height)
        niche: Content niche
        mobile_optimized: Whether thumbnail is mobile-optimized
    """
    thumbnail_file: str
    variants: List[str] = field(default_factory=list)
    predicted_ctr: float = 0.0
    color_scheme: str = "default"
    text_overlay: str = ""
    ai_face_used: bool = False
    emotion: Optional[str] = None
    file_size_kb: float = 0.0
    dimensions: Tuple[int, int] = (1280, 720)
    niche: str = "default"
    mobile_optimized: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["dimensions"] = list(self.dimensions)
        return result


class ThumbnailAgent(BaseAgent):
    """
    Production agent for generating high-CTR YouTube thumbnails.

    Wraps existing thumbnail_ai.py and thumbnail_generator.py functionality
    to provide a standardized agent interface with:
    - AI face generation via Replicate API
    - Niche-specific color schemes
    - A/B test variant generation
    - Mobile optimization
    - CTR prediction scoring
    """

    # YouTube thumbnail specifications
    THUMBNAIL_WIDTH = 1280
    THUMBNAIL_HEIGHT = 720
    MAX_FILE_SIZE_KB = 2048  # 2MB max for YouTube
    MIN_FILE_SIZE_KB = 50    # Sanity check

    # CTR prediction weights
    CTR_WEIGHTS = {
        "has_face": 0.15,
        "has_emotion": 0.10,
        "has_numbers": 0.08,
        "has_power_words": 0.12,
        "niche_colors": 0.08,
        "text_size_good": 0.07,
        "contrast_good": 0.10,
        "mobile_optimized": 0.10,
        "ai_face": 0.05,
        "base_score": 0.15,
    }

    def __init__(self, provider: str = "groq", api_key: str = None, output_dir: str = None):
        """
        Initialize the ThumbnailAgent.

        Args:
            provider: AI provider (for potential text optimization)
            api_key: API key for provider
            output_dir: Directory for output thumbnails
        """
        super().__init__(provider=provider, api_key=api_key)
        self.name = "ThumbnailAgent"

        self.output_dir = Path(output_dir or "assets/thumbnails")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-load thumbnail generators
        self._ai_generator = None
        self._basic_generator = None

        # Check for Replicate API key
        self.replicate_available = bool(os.getenv("REPLICATE_API_KEY"))
        if not self.replicate_available:
            logger.warning("REPLICATE_API_KEY not set. AI face generation will use fallback.")

        logger.info(f"ThumbnailAgent initialized. Output: {self.output_dir}")

    def _get_ai_generator(self):
        """Lazy-load AI thumbnail generator."""
        if self._ai_generator is None:
            try:
                from ..content.thumbnail_ai import AIThumbnailGenerator
                self._ai_generator = AIThumbnailGenerator(output_dir=str(self.output_dir))
            except ImportError as e:
                logger.warning(f"Could not import AIThumbnailGenerator: {e}")
        return self._ai_generator

    def _get_basic_generator(self):
        """Lazy-load basic thumbnail generator."""
        if self._basic_generator is None:
            try:
                from ..content.thumbnail_generator import ThumbnailGenerator
                self._basic_generator = ThumbnailGenerator(output_dir=str(self.output_dir))
            except ImportError as e:
                logger.warning(f"Could not import ThumbnailGenerator: {e}")
        return self._basic_generator

    def run(
        self,
        title: str,
        niche: str = "default",
        use_ai_face: bool = True,
        generate_variants: bool = True,
        variant_count: int = 3,
        emotion: str = None,
        face_position: str = "right",
        output_path: str = None,
        **kwargs
    ) -> AgentResult:
        """
        Generate a thumbnail with optional variants for A/B testing.

        Args:
            title: Video title (key words will be extracted)
            niche: Content niche (finance, psychology, storytelling)
            use_ai_face: Whether to use AI face generation (requires Replicate API)
            generate_variants: Whether to generate A/B test variants
            variant_count: Number of variants (2-3 recommended)
            emotion: Specific emotion for AI face (uses niche default if not specified)
            face_position: Position for AI face ("right", "left")
            output_path: Custom output path for primary thumbnail
            **kwargs: Additional parameters passed to generators

        Returns:
            AgentResult containing ThumbnailResult data
        """
        logger.info(f"[ThumbnailAgent] Generating thumbnail for: {title}")
        logger.info(f"[ThumbnailAgent] Niche: {niche}, AI Face: {use_ai_face}, Variants: {variant_count if generate_variants else 0}")

        warnings = []
        thumbnail_file = None
        variants = []
        ai_face_used = False
        used_emotion = None

        # Normalize niche
        niche = niche.lower() if niche else "default"
        if niche not in NICHE_COLOR_SCHEMES:
            niche = "default"
            warnings.append(f"Unknown niche, using default color scheme")

        # Get color scheme
        color_scheme = NICHE_COLOR_SCHEMES.get(niche, NICHE_COLOR_SCHEMES["default"])

        # Select emotion if not specified
        if not emotion:
            niche_emotions = NICHE_EMOTIONS.get(niche, NICHE_EMOTIONS.get("finance"))
            emotion = niche_emotions[0] if niche_emotions else "confident"
        used_emotion = emotion

        # Extract text overlay from title
        text_overlay = self._extract_thumbnail_text(title, niche)

        try:
            # Try AI face generation first if enabled
            if use_ai_face and self.replicate_available:
                ai_gen = self._get_ai_generator()
                if ai_gen:
                    try:
                        thumbnail_file = ai_gen.generate_with_ai_face(
                            title=title,
                            niche=niche,
                            emotion=emotion,
                            output_path=output_path,
                            face_position=face_position
                        )
                        ai_face_used = True
                        logger.success(f"[ThumbnailAgent] AI face thumbnail generated: {thumbnail_file}")
                    except Exception as e:
                        logger.warning(f"[ThumbnailAgent] AI face generation failed: {e}")
                        warnings.append(f"AI face generation failed: {str(e)[:50]}")

            # Fallback to basic generator
            if not thumbnail_file:
                basic_gen = self._get_basic_generator()
                if basic_gen:
                    thumbnail_file = basic_gen.generate(
                        title=title,
                        niche=niche,
                        output_path=output_path,
                        text_position="center"
                    )
                    logger.info(f"[ThumbnailAgent] Basic thumbnail generated: {thumbnail_file}")
                else:
                    return AgentResult(
                        success=False,
                        error="No thumbnail generator available",
                        data={"title": title, "niche": niche}
                    )

            # Generate A/B test variants
            if generate_variants and variant_count > 1:
                variants = self._generate_variants(
                    title=title,
                    niche=niche,
                    count=variant_count,
                    use_ai_face=use_ai_face and self.replicate_available
                )
                logger.info(f"[ThumbnailAgent] Generated {len(variants)} variants")

            # Calculate predicted CTR
            predicted_ctr = self._predict_ctr(
                title=title,
                niche=niche,
                has_ai_face=ai_face_used,
                text_overlay=text_overlay,
                emotion=used_emotion
            )

            # Get file info
            file_size_kb = 0
            dimensions = (self.THUMBNAIL_WIDTH, self.THUMBNAIL_HEIGHT)
            if thumbnail_file and os.path.exists(thumbnail_file):
                file_size_kb = os.path.getsize(thumbnail_file) / 1024
                try:
                    from PIL import Image
                    with Image.open(thumbnail_file) as img:
                        dimensions = img.size
                except:
                    pass

            # Check file size constraints
            if file_size_kb > self.MAX_FILE_SIZE_KB:
                warnings.append(f"File size ({file_size_kb:.0f}KB) exceeds 2MB limit")
            elif file_size_kb < self.MIN_FILE_SIZE_KB:
                warnings.append(f"File size unusually small ({file_size_kb:.0f}KB)")

            # Create thumbnail result
            thumb_result = ThumbnailResult(
                thumbnail_file=thumbnail_file,
                variants=variants,
                predicted_ctr=predicted_ctr,
                color_scheme=niche,
                text_overlay=text_overlay,
                ai_face_used=ai_face_used,
                emotion=used_emotion if ai_face_used else None,
                file_size_kb=file_size_kb,
                dimensions=dimensions,
                niche=niche,
                mobile_optimized=dimensions == (1280, 720)
            )

            # Log operation
            self.log_operation("generate_thumbnail", tokens=0, cost=0.0)

            return AgentResult(
                success=True,
                data=thumb_result.to_dict(),
                tokens_used=0,
                cost=0.0,
                metadata={
                    "title": title,
                    "niche": niche,
                    "warnings": warnings
                }
            )

        except Exception as e:
            logger.error(f"[ThumbnailAgent] Thumbnail generation failed: {e}")
            return AgentResult(
                success=False,
                error=str(e),
                data={"title": title, "niche": niche}
            )

    def _extract_thumbnail_text(self, title: str, niche: str) -> str:
        """
        Extract and format text for thumbnail overlay.

        Uses high-CTR patterns and extracts key words from title.

        Args:
            title: Full video title
            niche: Content niche

        Returns:
            Formatted text for thumbnail (max 4 words, uppercase)
        """
        # Filter words to remove
        filter_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into", "through",
            "and", "but", "or", "if", "this", "that", "these", "those", "i", "you"
        }

        # Extract words
        words = re.sub(r'[^\w\s$]', '', title).split()
        key_words = []

        for word in words:
            word_lower = word.lower()
            if word_lower not in filter_words and len(word) > 2:
                # Score priority
                score = 0
                if '$' in word or word.replace(',', '').isdigit():
                    score = 100  # Money/numbers highest priority
                elif any(c.isdigit() for c in word):
                    score = 50   # Numbers
                elif word_lower in ["secret", "truth", "hidden", "shocking", "real", "dark"]:
                    score = 40   # Power words
                else:
                    score = 10

                key_words.append((word, score))

        # Sort by score and take top 3-4
        key_words.sort(key=lambda x: x[1], reverse=True)
        selected = [w[0] for w in key_words[:4]]

        if not selected:
            selected = words[:3]

        # Apply high-CTR pattern if available
        patterns = HIGH_CTR_PATTERNS.get(niche, HIGH_CTR_PATTERNS.get("default", ["{keyword}"]))
        if patterns and selected:
            keyword = ' '.join(selected[:2])
            pattern = random.choice(patterns)
            text = pattern.replace("{keyword}", keyword)
        else:
            text = ' '.join(selected)

        return text.upper()[:30]  # Limit length

    def _generate_variants(
        self,
        title: str,
        niche: str,
        count: int,
        use_ai_face: bool
    ) -> List[str]:
        """
        Generate A/B test thumbnail variants.

        Creates variants with different:
        - Emotions (for AI faces)
        - Face positions (left/right)
        - Text positions (top/center/bottom)
        - Color emphasis

        Args:
            title: Video title
            niche: Content niche
            count: Number of variants to generate
            use_ai_face: Whether to use AI faces

        Returns:
            List of variant file paths
        """
        variants = []
        emotions = NICHE_EMOTIONS.get(niche, NICHE_EMOTIONS.get("finance"))
        positions = ["right", "left"]
        text_positions = ["center", "top", "bottom"]

        if use_ai_face and self.replicate_available:
            ai_gen = self._get_ai_generator()
            if ai_gen:
                try:
                    # Use built-in variant generation
                    variants = ai_gen.generate_ab_test_variants(
                        title=title,
                        niche=niche,
                        count=min(count, 3)
                    )
                    return variants
                except Exception as e:
                    logger.warning(f"AI variant generation failed: {e}")

        # Fallback to basic variants
        basic_gen = self._get_basic_generator()
        if basic_gen:
            try:
                variants = basic_gen.generate_variants(
                    title=title,
                    niche=niche,
                    count=min(count, 3)
                )
            except Exception as e:
                logger.warning(f"Basic variant generation failed: {e}")

        return variants

    def _predict_ctr(
        self,
        title: str,
        niche: str,
        has_ai_face: bool,
        text_overlay: str,
        emotion: str
    ) -> float:
        """
        Predict click-through rate based on thumbnail elements.

        Uses weighted scoring based on known CTR factors.

        Args:
            title: Video title
            niche: Content niche
            has_ai_face: Whether AI face was used
            text_overlay: Text on thumbnail
            emotion: Emotion expression (if AI face)

        Returns:
            Predicted CTR as float (0.0 to 1.0, typically 0.03-0.15)
        """
        score = self.CTR_WEIGHTS["base_score"]

        # Face presence (human faces increase CTR)
        if has_ai_face:
            score += self.CTR_WEIGHTS["has_face"]
            score += self.CTR_WEIGHTS["ai_face"]

        # Emotional expression
        high_ctr_emotions = ["surprised", "shocked", "excited", "horrified", "secretive"]
        if emotion and emotion.lower() in high_ctr_emotions:
            score += self.CTR_WEIGHTS["has_emotion"]

        # Numbers in text (increases CTR)
        if any(c.isdigit() for c in text_overlay):
            score += self.CTR_WEIGHTS["has_numbers"]

        # Power words
        power_words = ["secret", "truth", "hidden", "shocking", "exposed", "revealed", "dark", "stop"]
        if any(word.lower() in text_overlay.lower() for word in power_words):
            score += self.CTR_WEIGHTS["has_power_words"]

        # Niche-specific colors (assume we used them correctly)
        if niche in NICHE_COLOR_SCHEMES:
            score += self.CTR_WEIGHTS["niche_colors"]

        # Assume mobile optimization if we're following specs
        score += self.CTR_WEIGHTS["mobile_optimized"]

        # Good text size (assume yes for our generator)
        score += self.CTR_WEIGHTS["text_size_good"]

        # High contrast (assume yes)
        score += self.CTR_WEIGHTS["contrast_good"]

        # Add some randomness for realism (CTR varies)
        import random
        variance = random.uniform(-0.02, 0.02)
        score += variance

        # Clamp to realistic CTR range (3% to 15%)
        return max(0.03, min(0.15, score))

    def generate_for_video(
        self,
        video_data: Dict[str, Any],
        variant_count: int = 2
    ) -> AgentResult:
        """
        Convenience method to generate thumbnails from video data dict.

        Args:
            video_data: Dict with 'title', 'niche', 'topic' keys
            variant_count: Number of variants

        Returns:
            AgentResult with thumbnail data
        """
        title = video_data.get("title", video_data.get("topic", "Video"))
        niche = video_data.get("niche", "default")

        return self.run(
            title=title,
            niche=niche,
            generate_variants=variant_count > 1,
            variant_count=variant_count
        )


# CLI entry point
def main():
    """CLI entry point for thumbnail agent."""
    import sys
    import argparse

    if len(sys.argv) < 2:
        print("""
Thumbnail Agent - High CTR Thumbnail Generation

Usage:
    python -m src.agents.thumbnail_agent "<title>" [options]

Examples:
    python -m src.agents.thumbnail_agent "5 Money Mistakes" --niche finance
    python -m src.agents.thumbnail_agent "Dark Psychology" --niche psychology --variants 3
    python -m src.agents.thumbnail_agent "Untold Story" --niche storytelling --no-ai

Options:
    --niche <niche>     Content niche (finance, psychology, storytelling)
    --variants <n>      Number of A/B test variants (default: 2)
    --no-ai             Skip AI face generation
    --emotion <emotion> Specific emotion for AI face
    --output <path>     Custom output path
    --position <pos>    Face position (right, left)
        """)
        return

    parser = argparse.ArgumentParser(description="Generate YouTube thumbnails")
    parser.add_argument("title", help="Video title")
    parser.add_argument("--niche", default="default", help="Content niche")
    parser.add_argument("--variants", type=int, default=2, help="Number of variants")
    parser.add_argument("--no-ai", action="store_true", help="Skip AI face")
    parser.add_argument("--emotion", help="Emotion for AI face")
    parser.add_argument("--output", "-o", help="Output path")
    parser.add_argument("--position", default="right", help="Face position")

    args = parser.parse_args()

    # Run agent
    agent = ThumbnailAgent()
    result = agent.run(
        title=args.title,
        niche=args.niche,
        use_ai_face=not args.no_ai,
        generate_variants=args.variants > 1,
        variant_count=args.variants,
        emotion=args.emotion,
        face_position=args.position,
        output_path=args.output
    )

    # Print result
    print("\n" + "=" * 60)
    print("THUMBNAIL AGENT RESULT")
    print("=" * 60)
    print(f"Success: {result.success}")

    if result.success:
        data = result.data
        print(f"\nThumbnail: {data.get('thumbnail_file')}")
        print(f"Predicted CTR: {data.get('predicted_ctr', 0):.1%}")
        print(f"AI Face Used: {data.get('ai_face_used', False)}")
        print(f"Emotion: {data.get('emotion', 'N/A')}")
        print(f"File Size: {data.get('file_size_kb', 0):.0f} KB")
        print(f"Dimensions: {data.get('dimensions')}")

        variants = data.get('variants', [])
        if variants:
            print(f"\nVariants ({len(variants)}):")
            for v in variants:
                print(f"  - {v}")
    else:
        print(f"\nError: {result.error}")

    if result.metadata.get("warnings"):
        print(f"\nWarnings:")
        for w in result.metadata["warnings"]:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
