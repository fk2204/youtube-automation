"""
Pika Labs Video Generator

AI-powered video generation using Pika Labs API via fal.ai.
Supports text-to-video and image-to-video generation for:
- YouTube Shorts (primary use case)
- B-roll clips for long-form videos

API Reference: https://fal.ai/models/fal-ai/pika/v2.2/text-to-video
Pricing: $0.20/5s at 720p, $0.45/5s at 1080p

Usage:
    from src.content.video_pika import PikaVideoGenerator

    async def main():
        generator = PikaVideoGenerator()

        # Text-to-video
        result = await generator.generate_from_text(
            prompt="A serene sunset over the ocean with gentle waves",
            output_file="output/pika_video.mp4",
            duration=5,
            resolution="720p"
        )

        # Image-to-video
        result = await generator.generate_from_image(
            image_path="input/image.png",
            prompt="The scene slowly comes to life with gentle movement",
            output_file="output/animated.mp4"
        )

    asyncio.run(main())
"""

import os
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal
from dataclasses import dataclass
from enum import Enum
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

# Try to import fal_client
try:
    import fal_client
    FAL_CLIENT_AVAILABLE = True
except ImportError:
    FAL_CLIENT_AVAILABLE = False
    logger.warning("fal-client not installed. Install with: pip install fal-client")

# Try to import httpx for downloading
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    try:
        import aiohttp
        AIOHTTP_AVAILABLE = True
    except ImportError:
        AIOHTTP_AVAILABLE = False


class PikaResolution(str, Enum):
    """Available resolution options for Pika video generation."""
    HD_720 = "720p"
    FHD_1080 = "1080p"


class PikaAspectRatio(str, Enum):
    """Available aspect ratios for Pika video generation."""
    LANDSCAPE_16_9 = "16:9"
    PORTRAIT_9_16 = "9:16"
    SQUARE_1_1 = "1:1"


@dataclass
class PikaVideoResult:
    """Result from Pika video generation."""
    success: bool
    video_url: Optional[str] = None
    local_path: Optional[str] = None
    duration: Optional[float] = None
    resolution: Optional[str] = None
    error: Optional[str] = None
    request_id: Optional[str] = None
    cost_estimate: Optional[float] = None

    def __bool__(self) -> bool:
        return self.success


class PikaVideoGenerator:
    """
    Pika Labs video generator using fal.ai API.

    Supports:
    - Text-to-video generation (Pika 2.2)
    - Image-to-video animation (Pika 2.2)
    - Multiple resolutions (720p, 1080p)
    - Multiple aspect ratios (16:9, 9:16, 1:1)

    Pricing (fal.ai):
    - 720p 5-second: $0.20
    - 1080p 5-second: $0.45
    """

    # fal.ai model endpoints
    TEXT_TO_VIDEO_MODEL = "fal-ai/pika/v2.2/text-to-video"
    IMAGE_TO_VIDEO_MODEL = "fal-ai/pika/v2.2/image-to-video"
    IMAGE_TO_VIDEO_TURBO_MODEL = "fal-ai/pika/v2/turbo/image-to-video"

    # Pricing per 5-second video (USD)
    PRICING = {
        "720p": 0.20,
        "1080p": 0.45,
    }

    # Default settings
    DEFAULT_DURATION = 5  # seconds
    DEFAULT_RESOLUTION = PikaResolution.HD_720
    DEFAULT_ASPECT_RATIO = PikaAspectRatio.LANDSCAPE_16_9

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Pika video generator.

        Args:
            api_key: fal.ai API key. If not provided, uses FAL_KEY or PIKA_API_KEY
                     environment variable.
        """
        self.api_key = api_key or os.getenv("FAL_KEY") or os.getenv("PIKA_API_KEY")

        if not FAL_CLIENT_AVAILABLE:
            logger.error(
                "fal-client package not installed. "
                "Install with: pip install fal-client"
            )
            raise ImportError("fal-client package required for Pika video generation")

        if not self.api_key:
            logger.warning(
                "PIKA_API_KEY/FAL_KEY not set. "
                "Get your API key from https://fal.ai and set FAL_KEY environment variable."
            )
        else:
            # Set the API key for fal_client
            os.environ["FAL_KEY"] = self.api_key
            logger.info("PikaVideoGenerator initialized with API key")

        # Temp directory for downloads
        self.temp_dir = Path(tempfile.gettempdir()) / "pika_videos"
        self.temp_dir.mkdir(exist_ok=True)

    def _validate_api_key(self) -> bool:
        """Check if API key is configured."""
        if not self.api_key:
            logger.error(
                "Pika API key not configured. "
                "Set PIKA_API_KEY or FAL_KEY environment variable."
            )
            return False
        return True

    def _estimate_cost(
        self,
        duration: int,
        resolution: str
    ) -> float:
        """
        Estimate cost for video generation.

        Args:
            duration: Video duration in seconds
            resolution: Resolution string ("720p" or "1080p")

        Returns:
            Estimated cost in USD
        """
        base_price = self.PRICING.get(resolution, self.PRICING["720p"])
        # Pricing is per 5 seconds
        cost = base_price * (duration / 5)
        return round(cost, 2)

    async def _download_video(
        self,
        url: str,
        output_path: str
    ) -> Optional[str]:
        """
        Download video from URL to local file.

        Args:
            url: Video URL to download
            output_path: Local path to save the video

        Returns:
            Local file path or None on failure
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if HTTPX_AVAILABLE:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.get(url)
                    response.raise_for_status()

                    with open(output_path, "wb") as f:
                        f.write(response.content)

                    logger.info(f"Downloaded video to: {output_path}")
                    return str(output_path)

            elif AIOHTTP_AVAILABLE:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as response:
                        response.raise_for_status()
                        content = await response.read()

                        with open(output_path, "wb") as f:
                            f.write(content)

                        logger.info(f"Downloaded video to: {output_path}")
                        return str(output_path)
            else:
                # Fallback to synchronous download
                import urllib.request
                urllib.request.urlretrieve(url, str(output_path))
                logger.info(f"Downloaded video to: {output_path}")
                return str(output_path)

        except Exception as e:
            logger.error(f"Failed to download video: {e}")
            return None

    async def generate_from_text(
        self,
        prompt: str,
        output_file: Optional[str] = None,
        duration: int = 5,
        resolution: str = "720p",
        aspect_ratio: str = "16:9",
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None
    ) -> PikaVideoResult:
        """
        Generate video from text prompt.

        Args:
            prompt: Text description of the video to generate.
                    Best practice: Include subject, style, motion, camera direction.
                    Example: "A cinematic 16:9 shot of waves crashing on rocks at sunset,
                             slow motion, golden hour lighting"
            output_file: Path to save the generated video. If None, saves to temp dir.
            duration: Video duration in seconds (5-10 supported)
            resolution: Video resolution ("720p" or "1080p")
            aspect_ratio: Aspect ratio ("16:9", "9:16", or "1:1")
            negative_prompt: Things to avoid in the video
            seed: Random seed for reproducibility

        Returns:
            PikaVideoResult with video URL and local path
        """
        if not self._validate_api_key():
            return PikaVideoResult(
                success=False,
                error="API key not configured"
            )

        # Validate parameters
        duration = max(5, min(10, duration))  # Clamp to 5-10 seconds
        resolution = resolution if resolution in ["720p", "1080p"] else "720p"

        logger.info(f"Generating text-to-video with Pika 2.2")
        logger.info(f"  Prompt: {prompt[:100]}...")
        logger.info(f"  Duration: {duration}s, Resolution: {resolution}")

        cost_estimate = self._estimate_cost(duration, resolution)
        logger.info(f"  Estimated cost: ${cost_estimate:.2f}")

        try:
            # Build request arguments
            arguments: Dict[str, Any] = {
                "prompt": prompt,
                "resolution": resolution,
                "duration": duration,
                "aspect_ratio": aspect_ratio,
            }

            if negative_prompt:
                arguments["negative_prompt"] = negative_prompt
            if seed is not None:
                arguments["seed"] = seed

            # Call fal.ai API with subscription pattern
            logger.info("Submitting to Pika API (this may take 1-3 minutes)...")

            # Use subscribe for async queue handling
            result = await asyncio.to_thread(
                fal_client.subscribe,
                self.TEXT_TO_VIDEO_MODEL,
                arguments=arguments,
                with_logs=True,
                on_queue_update=self._on_queue_update
            )

            # Extract video URL from result
            video_url = None
            if isinstance(result, dict):
                if "video" in result and isinstance(result["video"], dict):
                    video_url = result["video"].get("url")
                elif "video_url" in result:
                    video_url = result["video_url"]
                elif "url" in result:
                    video_url = result["url"]

            if not video_url:
                logger.error(f"No video URL in response: {result}")
                return PikaVideoResult(
                    success=False,
                    error="No video URL in API response"
                )

            logger.success(f"Video generated: {video_url}")

            # Download to local file
            if not output_file:
                output_file = str(self.temp_dir / f"pika_t2v_{os.urandom(4).hex()}.mp4")

            local_path = await self._download_video(video_url, output_file)

            return PikaVideoResult(
                success=True,
                video_url=video_url,
                local_path=local_path,
                duration=duration,
                resolution=resolution,
                cost_estimate=cost_estimate,
                request_id=result.get("request_id") if isinstance(result, dict) else None
            )

        except Exception as e:
            logger.error(f"Pika text-to-video failed: {e}")
            return PikaVideoResult(
                success=False,
                error=str(e)
            )

    async def generate_from_image(
        self,
        image_path: str,
        prompt: str,
        output_file: Optional[str] = None,
        duration: int = 5,
        resolution: str = "720p",
        use_turbo: bool = False,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None
    ) -> PikaVideoResult:
        """
        Generate video from an image (animate the image).

        Args:
            image_path: Path to the input image (PNG, JPG, WebP)
            prompt: Description of the motion/animation to apply.
                    Example: "The camera slowly pushes in as leaves fall gently"
            output_file: Path to save the generated video
            duration: Video duration in seconds (5-10 supported)
            resolution: Video resolution ("720p" or "1080p")
            use_turbo: Use turbo model (faster but lower quality)
            negative_prompt: Things to avoid in the video
            seed: Random seed for reproducibility

        Returns:
            PikaVideoResult with video URL and local path
        """
        if not self._validate_api_key():
            return PikaVideoResult(
                success=False,
                error="API key not configured"
            )

        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return PikaVideoResult(
                success=False,
                error=f"Image file not found: {image_path}"
            )

        # Validate parameters
        duration = max(5, min(10, duration))
        resolution = resolution if resolution in ["720p", "1080p"] else "720p"

        logger.info(f"Generating image-to-video with Pika 2.2")
        logger.info(f"  Image: {image_path}")
        logger.info(f"  Prompt: {prompt[:100]}...")
        logger.info(f"  Duration: {duration}s, Resolution: {resolution}")

        cost_estimate = self._estimate_cost(duration, resolution)
        logger.info(f"  Estimated cost: ${cost_estimate:.2f}")

        try:
            # Upload image to fal.ai
            logger.info("Uploading image to fal.ai...")
            image_url = fal_client.upload_file(image_path)
            logger.debug(f"Image uploaded: {image_url}")

            # Build request arguments
            arguments: Dict[str, Any] = {
                "image_url": image_url,
                "prompt": prompt,
                "resolution": resolution,
                "duration": duration,
            }

            if negative_prompt:
                arguments["negative_prompt"] = negative_prompt
            if seed is not None:
                arguments["seed"] = seed

            # Select model
            model = self.IMAGE_TO_VIDEO_TURBO_MODEL if use_turbo else self.IMAGE_TO_VIDEO_MODEL

            # Call fal.ai API
            logger.info("Submitting to Pika API (this may take 1-3 minutes)...")

            result = await asyncio.to_thread(
                fal_client.subscribe,
                model,
                arguments=arguments,
                with_logs=True,
                on_queue_update=self._on_queue_update
            )

            # Extract video URL
            video_url = None
            if isinstance(result, dict):
                if "video" in result and isinstance(result["video"], dict):
                    video_url = result["video"].get("url")
                elif "video_url" in result:
                    video_url = result["video_url"]
                elif "url" in result:
                    video_url = result["url"]

            if not video_url:
                logger.error(f"No video URL in response: {result}")
                return PikaVideoResult(
                    success=False,
                    error="No video URL in API response"
                )

            logger.success(f"Video generated: {video_url}")

            # Download to local file
            if not output_file:
                output_file = str(self.temp_dir / f"pika_i2v_{os.urandom(4).hex()}.mp4")

            local_path = await self._download_video(video_url, output_file)

            return PikaVideoResult(
                success=True,
                video_url=video_url,
                local_path=local_path,
                duration=duration,
                resolution=resolution,
                cost_estimate=cost_estimate,
                request_id=result.get("request_id") if isinstance(result, dict) else None
            )

        except Exception as e:
            logger.error(f"Pika image-to-video failed: {e}")
            return PikaVideoResult(
                success=False,
                error=str(e)
            )

    def _on_queue_update(self, update: Any) -> None:
        """Handle queue status updates from fal.ai."""
        try:
            if hasattr(update, "status"):
                status = update.status
                if status == "IN_QUEUE":
                    position = getattr(update, "queue_position", "?")
                    logger.debug(f"In queue, position: {position}")
                elif status == "IN_PROGRESS":
                    if hasattr(update, "logs"):
                        for log in update.logs:
                            if hasattr(log, "message"):
                                logger.debug(f"Pika: {log.message}")
                elif status == "COMPLETED":
                    logger.debug("Generation completed")
        except Exception as e:
            logger.debug(f"Queue update error: {e}")

    async def generate_short_clip(
        self,
        prompt: str,
        output_file: Optional[str] = None,
        duration: int = 5,
        aspect_ratio: str = "9:16"
    ) -> PikaVideoResult:
        """
        Generate a vertical clip optimized for YouTube Shorts.

        Args:
            prompt: Text description of the video
            output_file: Path to save the video
            duration: Duration in seconds (5-10)
            aspect_ratio: Aspect ratio (default "9:16" for vertical)

        Returns:
            PikaVideoResult
        """
        # Optimize prompt for Shorts
        enhanced_prompt = f"{prompt}, vertical format, mobile-friendly composition, centered subject"

        return await self.generate_from_text(
            prompt=enhanced_prompt,
            output_file=output_file,
            duration=duration,
            resolution="720p",  # Use 720p for cost efficiency
            aspect_ratio=aspect_ratio
        )

    async def generate_broll_clip(
        self,
        topic: str,
        output_file: Optional[str] = None,
        style: str = "cinematic",
        duration: int = 5
    ) -> PikaVideoResult:
        """
        Generate a B-roll clip for long-form videos.

        Args:
            topic: Topic/subject for the B-roll
            output_file: Path to save the video
            style: Visual style ("cinematic", "documentary", "corporate", "abstract")
            duration: Duration in seconds

        Returns:
            PikaVideoResult
        """
        # Build style-specific prompt enhancements
        style_prompts = {
            "cinematic": "cinematic shot, shallow depth of field, dramatic lighting, film grain",
            "documentary": "documentary style, natural lighting, realistic, observational",
            "corporate": "clean, professional, modern, well-lit, business setting",
            "abstract": "abstract visuals, artistic, creative transitions, symbolic imagery",
        }

        style_addition = style_prompts.get(style, style_prompts["cinematic"])
        enhanced_prompt = f"{topic}, {style_addition}, 16:9 widescreen, high quality b-roll footage"

        return await self.generate_from_text(
            prompt=enhanced_prompt,
            output_file=output_file,
            duration=duration,
            resolution="720p",
            aspect_ratio="16:9"
        )

    async def generate_multiple_clips(
        self,
        prompts: List[str],
        output_dir: str,
        duration: int = 5,
        resolution: str = "720p",
        max_concurrent: int = 3
    ) -> List[PikaVideoResult]:
        """
        Generate multiple video clips concurrently.

        Args:
            prompts: List of text prompts
            output_dir: Directory to save videos
            duration: Duration for each clip
            resolution: Resolution for all clips
            max_concurrent: Maximum concurrent generations

        Returns:
            List of PikaVideoResult objects
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def generate_with_limit(idx: int, prompt: str) -> PikaVideoResult:
            async with semaphore:
                output_file = str(output_path / f"clip_{idx:03d}.mp4")
                return await self.generate_from_text(
                    prompt=prompt,
                    output_file=output_file,
                    duration=duration,
                    resolution=resolution
                )

        tasks = [
            generate_with_limit(i, prompt)
            for i, prompt in enumerate(prompts)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(PikaVideoResult(
                    success=False,
                    error=str(result)
                ))
            else:
                processed_results.append(result)

        # Log summary
        successful = sum(1 for r in processed_results if r.success)
        total_cost = sum(r.cost_estimate or 0 for r in processed_results if r.success)
        logger.info(f"Generated {successful}/{len(prompts)} clips, total cost: ${total_cost:.2f}")

        return processed_results

    def cleanup_temp_files(self) -> None:
        """Remove temporary downloaded files."""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.temp_dir.mkdir(exist_ok=True)
                logger.debug("Cleaned up Pika temp files")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp files: {e}")


# Convenience function for quick access
def get_pika_generator(api_key: Optional[str] = None) -> Optional[PikaVideoGenerator]:
    """
    Get a PikaVideoGenerator instance.

    Args:
        api_key: Optional API key (uses environment variable if not provided)

    Returns:
        PikaVideoGenerator instance or None if not available
    """
    try:
        return PikaVideoGenerator(api_key=api_key)
    except ImportError as e:
        logger.error(f"Cannot create PikaVideoGenerator: {e}")
        return None


# Example usage and testing
if __name__ == "__main__":
    async def test_pika():
        print("\n" + "=" * 60)
        print("PIKA LABS VIDEO GENERATOR TEST")
        print("=" * 60 + "\n")

        try:
            generator = PikaVideoGenerator()
            print(f"API Key configured: {bool(generator.api_key)}")
            print(f"Temp directory: {generator.temp_dir}")
            print(f"fal-client available: {FAL_CLIENT_AVAILABLE}")

            if not generator.api_key:
                print("\nNo API key configured. Set FAL_KEY or PIKA_API_KEY to test.")
                print("Get your API key from: https://fal.ai")
                return

            # Test text-to-video
            print("\nTesting text-to-video generation...")
            result = await generator.generate_from_text(
                prompt="A serene forest scene with sunlight filtering through trees, peaceful atmosphere",
                duration=5,
                resolution="720p"
            )

            if result.success:
                print(f"Success! Video saved to: {result.local_path}")
                print(f"Cost estimate: ${result.cost_estimate:.2f}")
            else:
                print(f"Failed: {result.error}")

        except ImportError as e:
            print(f"Import error: {e}")
            print("Install required packages: pip install fal-client httpx")
        except Exception as e:
            print(f"Error: {e}")

    asyncio.run(test_pika())
