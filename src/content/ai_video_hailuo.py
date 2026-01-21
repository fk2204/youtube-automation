"""
HailuoAI (MiniMax) Video Generation via fal.ai

Budget-friendly AI video generation using HailuoAI (MiniMax) through fal.ai.
- Cost: ~$0.28 per 5-second video (significantly cheaper than Pika)
- Quality: High quality output comparable to other AI video providers
- Use case: High-volume production, budget-conscious video creation

API Reference: https://fal.ai/models/fal-ai/minimax/video-01
Pricing: ~$0.28 per 5-second video at default resolution

Usage:
    from src.content.ai_video_hailuo import HailuoVideoGenerator

    async def main():
        generator = HailuoVideoGenerator()

        # Text-to-video
        result = await generator.generate_video(
            prompt="A serene sunset over the ocean with gentle waves",
            output_file="output/hailuo_video.mp4"
        )

        # Image-to-video
        result = await generator.generate_from_image(
            image_url="https://example.com/image.jpg",
            prompt="The scene slowly comes to life with gentle movement"
        )

    asyncio.run(main())
"""

import os
import asyncio
import tempfile
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
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


@dataclass
class HailuoVideoResult:
    """Result from HailuoAI video generation."""
    success: bool
    video_url: Optional[str] = None
    local_path: Optional[str] = None
    duration: Optional[float] = None
    error: Optional[str] = None
    request_id: Optional[str] = None
    cost_estimate: Optional[float] = None
    generation_time: Optional[float] = None  # Time taken to generate in seconds

    def __bool__(self) -> bool:
        return self.success

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "video_url": self.video_url,
            "local_path": self.local_path,
            "duration": self.duration,
            "error": self.error,
            "request_id": self.request_id,
            "cost_estimate": self.cost_estimate,
            "generation_time": self.generation_time
        }


@dataclass
class BatchResult:
    """Result from batch video generation."""
    total: int
    successful: int
    failed: int
    results: List[HailuoVideoResult] = field(default_factory=list)
    total_cost: float = 0.0
    total_time: float = 0.0

    def __bool__(self) -> bool:
        return self.successful > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "total": self.total,
            "successful": self.successful,
            "failed": self.failed,
            "results": [r.to_dict() for r in self.results],
            "total_cost": self.total_cost,
            "total_time": self.total_time
        }


class HailuoVideoGenerator:
    """
    HailuoAI (MiniMax) video generator using fal.ai API.

    Supports:
    - Text-to-video generation (MiniMax video-01)
    - Image-to-video generation (animate images)
    - Async batch processing with concurrency control
    - Cost tracking integration with token_manager
    - Automatic retries with exponential backoff

    Pricing (fal.ai):
    - ~$0.28 per 5-second video

    This is significantly cheaper than Pika Labs ($0.20-0.45) and most other
    AI video generation services, making it ideal for high-volume production.
    """

    # fal.ai model endpoint for HailuoAI/MiniMax
    TEXT_TO_VIDEO_MODEL = "fal-ai/minimax/video-01"
    IMAGE_TO_VIDEO_MODEL = "fal-ai/minimax/video-01-live"  # Image-to-video variant

    # Pricing per video (USD)
    COST_PER_VIDEO = 0.28  # ~$0.28 per 5-second video

    # Default settings
    DEFAULT_PROMPT_OPTIMIZER = True  # MiniMax has a built-in prompt optimizer

    def __init__(
        self,
        api_key: Optional[str] = None,
        track_costs: bool = True,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        """
        Initialize HailuoAI video generator.

        Args:
            api_key: fal.ai API key. If not provided, uses FAL_KEY environment variable.
            track_costs: Whether to track costs with token_manager.
            max_retries: Maximum number of retries on failure.
            retry_delay: Base delay between retries in seconds (uses exponential backoff).
        """
        self.api_key = api_key or os.getenv("FAL_KEY")
        self.track_costs = track_costs
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if not FAL_CLIENT_AVAILABLE:
            logger.error(
                "fal-client package not installed. "
                "Install with: pip install fal-client"
            )
            raise ImportError("fal-client package required for HailuoAI video generation")

        if not self.api_key:
            logger.warning(
                "FAL_KEY not set. "
                "Get your API key from https://fal.ai and set FAL_KEY environment variable."
            )
        else:
            # Set the API key for fal_client
            os.environ["FAL_KEY"] = self.api_key
            logger.info("HailuoVideoGenerator initialized with API key")

        # Temp directory for downloads
        self.temp_dir = Path(tempfile.gettempdir()) / "hailuo_videos"
        self.temp_dir.mkdir(exist_ok=True)

        # Cost tracking
        self._total_cost = 0.0
        self._total_generations = 0

        # Initialize token tracker if tracking enabled
        self._token_tracker = None
        if self.track_costs:
            try:
                from src.utils.token_manager import get_token_manager
                self._token_tracker = get_token_manager()
            except ImportError:
                logger.debug("Token manager not available, cost tracking disabled")

    def _validate_api_key(self) -> bool:
        """Check if API key is configured."""
        if not self.api_key:
            logger.error(
                "FAL_KEY not configured. "
                "Set FAL_KEY environment variable."
            )
            return False
        return True

    def _record_cost(self, cost: float, operation: str = "hailuo_video_generation"):
        """Record cost to token_manager if available."""
        self._total_cost += cost
        self._total_generations += 1

        if self._token_tracker:
            try:
                # Record as a "video generation" operation
                # We use 0 tokens since this is a flat-rate video service
                self._token_tracker.record_usage(
                    provider="hailuo",
                    input_tokens=0,
                    output_tokens=0,
                    operation=operation
                )
            except Exception as e:
                logger.debug(f"Failed to record cost to token_manager: {e}")

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
                async with httpx.AsyncClient(timeout=180.0) as client:
                    response = await client.get(url)
                    response.raise_for_status()

                    with open(output_path, "wb") as f:
                        f.write(response.content)

                    logger.info(f"Downloaded video to: {output_path}")
                    return str(output_path)

            elif AIOHTTP_AVAILABLE:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=180)) as response:
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
                                logger.debug(f"HailuoAI: {log.message}")
                elif status == "COMPLETED":
                    logger.debug("Generation completed")
        except Exception as e:
            logger.debug(f"Queue update error: {e}")

    async def _generate_with_retry(
        self,
        model: str,
        arguments: Dict[str, Any],
        operation: str = "video_generation"
    ) -> Dict[str, Any]:
        """
        Execute fal.ai API call with retry logic.

        Args:
            model: Model endpoint
            arguments: API arguments
            operation: Operation name for logging

        Returns:
            API response dict

        Raises:
            Exception: If all retries fail
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                result = await asyncio.to_thread(
                    fal_client.subscribe,
                    model,
                    arguments=arguments,
                    with_logs=True,
                    on_queue_update=self._on_queue_update
                )
                return result
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"{operation} failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"{operation} failed after {self.max_retries} attempts: {e}")

        raise last_error

    async def generate_video(
        self,
        prompt: str,
        output_file: Optional[str] = None,
        prompt_optimizer: bool = True,
        **kwargs
    ) -> HailuoVideoResult:
        """
        Generate video from text prompt using HailuoAI (MiniMax).

        Args:
            prompt: Text description of the video to generate.
                    Best practice: Include subject, style, motion, camera direction.
                    Example: "A cinematic shot of waves crashing on rocks at sunset,
                             slow motion, golden hour lighting"
            output_file: Path to save the generated video. If None, saves to temp dir.
            prompt_optimizer: Whether to use MiniMax's built-in prompt optimizer.
            **kwargs: Additional arguments passed to the API.

        Returns:
            HailuoVideoResult with video URL and local path
        """
        if not self._validate_api_key():
            return HailuoVideoResult(
                success=False,
                error="API key not configured"
            )

        logger.info(f"Generating video with HailuoAI (MiniMax)")
        logger.info(f"  Prompt: {prompt[:100]}...")
        logger.info(f"  Estimated cost: ${self.COST_PER_VIDEO:.2f}")

        start_time = time.time()

        try:
            # Build request arguments
            arguments: Dict[str, Any] = {
                "prompt": prompt,
                "prompt_optimizer": prompt_optimizer,
            }

            # Add any additional kwargs
            arguments.update(kwargs)

            # Call fal.ai API with retry
            logger.info("Submitting to HailuoAI API (this may take 2-5 minutes)...")

            result = await self._generate_with_retry(
                self.TEXT_TO_VIDEO_MODEL,
                arguments,
                "HailuoAI text-to-video"
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
                return HailuoVideoResult(
                    success=False,
                    error="No video URL in API response"
                )

            generation_time = time.time() - start_time
            logger.success(f"Video generated in {generation_time:.1f}s: {video_url}")

            # Download to local file
            if not output_file:
                output_file = str(self.temp_dir / f"hailuo_{os.urandom(4).hex()}.mp4")

            local_path = await self._download_video(video_url, output_file)

            # Record cost
            self._record_cost(self.COST_PER_VIDEO, "hailuo_text_to_video")

            return HailuoVideoResult(
                success=True,
                video_url=video_url,
                local_path=local_path,
                duration=5.0,  # Default duration
                cost_estimate=self.COST_PER_VIDEO,
                generation_time=generation_time,
                request_id=result.get("request_id") if isinstance(result, dict) else None
            )

        except Exception as e:
            logger.error(f"HailuoAI video generation failed: {e}")
            return HailuoVideoResult(
                success=False,
                error=str(e),
                generation_time=time.time() - start_time
            )

    async def generate_from_image(
        self,
        image_url: str,
        prompt: str,
        output_file: Optional[str] = None,
        prompt_optimizer: bool = True,
        **kwargs
    ) -> HailuoVideoResult:
        """
        Generate video from an image (animate the image).

        Args:
            image_url: URL of the input image (must be publicly accessible)
            prompt: Description of the motion/animation to apply.
                    Example: "The camera slowly pushes in as leaves fall gently"
            output_file: Path to save the generated video
            prompt_optimizer: Whether to use MiniMax's built-in prompt optimizer.
            **kwargs: Additional arguments passed to the API.

        Returns:
            HailuoVideoResult with video URL and local path
        """
        if not self._validate_api_key():
            return HailuoVideoResult(
                success=False,
                error="API key not configured"
            )

        logger.info(f"Generating image-to-video with HailuoAI (MiniMax)")
        logger.info(f"  Image URL: {image_url[:80]}...")
        logger.info(f"  Prompt: {prompt[:100]}...")
        logger.info(f"  Estimated cost: ${self.COST_PER_VIDEO:.2f}")

        start_time = time.time()

        try:
            # Build request arguments
            arguments: Dict[str, Any] = {
                "prompt": prompt,
                "image_url": image_url,
                "prompt_optimizer": prompt_optimizer,
            }

            # Add any additional kwargs
            arguments.update(kwargs)

            # Call fal.ai API with retry
            logger.info("Submitting to HailuoAI API (this may take 2-5 minutes)...")

            result = await self._generate_with_retry(
                self.IMAGE_TO_VIDEO_MODEL,
                arguments,
                "HailuoAI image-to-video"
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
                return HailuoVideoResult(
                    success=False,
                    error="No video URL in API response"
                )

            generation_time = time.time() - start_time
            logger.success(f"Video generated in {generation_time:.1f}s: {video_url}")

            # Download to local file
            if not output_file:
                output_file = str(self.temp_dir / f"hailuo_i2v_{os.urandom(4).hex()}.mp4")

            local_path = await self._download_video(video_url, output_file)

            # Record cost
            self._record_cost(self.COST_PER_VIDEO, "hailuo_image_to_video")

            return HailuoVideoResult(
                success=True,
                video_url=video_url,
                local_path=local_path,
                duration=5.0,
                cost_estimate=self.COST_PER_VIDEO,
                generation_time=generation_time,
                request_id=result.get("request_id") if isinstance(result, dict) else None
            )

        except Exception as e:
            logger.error(f"HailuoAI image-to-video failed: {e}")
            return HailuoVideoResult(
                success=False,
                error=str(e),
                generation_time=time.time() - start_time
            )

    async def generate_batch(
        self,
        prompts: List[str],
        output_dir: Optional[str] = None,
        max_concurrent: int = 2,
        on_progress: Optional[Callable[[int, int, HailuoVideoResult], None]] = None,
        **kwargs
    ) -> BatchResult:
        """
        Generate multiple videos concurrently with batching.

        Args:
            prompts: List of text prompts
            output_dir: Directory to save videos (uses temp if None)
            max_concurrent: Maximum concurrent generations (default 2 to avoid rate limits)
            on_progress: Callback function called after each video completes.
                        Signature: (completed_count, total_count, result)
            **kwargs: Additional arguments passed to generate_video

        Returns:
            BatchResult with all results and statistics
        """
        if not prompts:
            return BatchResult(total=0, successful=0, failed=0)

        output_path = Path(output_dir) if output_dir else self.temp_dir / "batch"
        output_path.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        results: List[HailuoVideoResult] = []
        semaphore = asyncio.Semaphore(max_concurrent)
        completed = 0

        async def generate_with_limit(idx: int, prompt: str) -> HailuoVideoResult:
            nonlocal completed
            async with semaphore:
                output_file = str(output_path / f"video_{idx:03d}.mp4")
                result = await self.generate_video(
                    prompt=prompt,
                    output_file=output_file,
                    **kwargs
                )
                completed += 1
                if on_progress:
                    try:
                        on_progress(completed, len(prompts), result)
                    except Exception as e:
                        logger.debug(f"Progress callback error: {e}")
                return result

        logger.info(f"Starting batch generation of {len(prompts)} videos (max concurrent: {max_concurrent})")

        tasks = [
            generate_with_limit(i, prompt)
            for i, prompt in enumerate(prompts)
        ]

        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for i, result in enumerate(task_results):
            if isinstance(result, Exception):
                results.append(HailuoVideoResult(
                    success=False,
                    error=str(result)
                ))
            else:
                results.append(result)

        # Calculate statistics
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        total_cost = sum(r.cost_estimate or 0 for r in results if r.success)
        total_time = time.time() - start_time

        logger.info(
            f"Batch complete: {successful}/{len(prompts)} successful, "
            f"${total_cost:.2f} total cost, {total_time:.1f}s total time"
        )

        return BatchResult(
            total=len(prompts),
            successful=successful,
            failed=failed,
            results=results,
            total_cost=total_cost,
            total_time=total_time
        )

    def get_cost_summary(self) -> Dict[str, Any]:
        """
        Get cost tracking summary.

        Returns:
            Dict with cost statistics
        """
        return {
            "total_cost": self._total_cost,
            "total_generations": self._total_generations,
            "average_cost_per_video": (
                self._total_cost / self._total_generations
                if self._total_generations > 0 else 0
            ),
            "cost_per_video": self.COST_PER_VIDEO,
            "provider": "hailuo"
        }

    def cleanup_temp_files(self) -> None:
        """Remove temporary downloaded files."""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.temp_dir.mkdir(exist_ok=True)
                logger.debug("Cleaned up HailuoAI temp files")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp files: {e}")


# Convenience function for quick access
def get_hailuo_generator(api_key: Optional[str] = None) -> Optional[HailuoVideoGenerator]:
    """
    Get a HailuoVideoGenerator instance.

    Args:
        api_key: Optional API key (uses environment variable if not provided)

    Returns:
        HailuoVideoGenerator instance or None if not available
    """
    try:
        return HailuoVideoGenerator(api_key=api_key)
    except ImportError as e:
        logger.error(f"Cannot create HailuoVideoGenerator: {e}")
        return None


# Example usage and testing
if __name__ == "__main__":
    async def test_hailuo():
        print("\n" + "=" * 60)
        print("HAILUOAI (MINIMAX) VIDEO GENERATOR TEST")
        print("=" * 60 + "\n")

        try:
            generator = HailuoVideoGenerator()
            print(f"API Key configured: {bool(generator.api_key)}")
            print(f"Temp directory: {generator.temp_dir}")
            print(f"fal-client available: {FAL_CLIENT_AVAILABLE}")
            print(f"Cost per video: ${generator.COST_PER_VIDEO:.2f}")

            if not generator.api_key:
                print("\nNo API key configured. Set FAL_KEY to test.")
                print("Get your API key from: https://fal.ai")
                return

            # Test text-to-video
            print("\nTesting text-to-video generation...")
            result = await generator.generate_video(
                prompt="A serene forest scene with sunlight filtering through trees, peaceful atmosphere, cinematic"
            )

            if result.success:
                print(f"Success! Video saved to: {result.local_path}")
                print(f"Cost: ${result.cost_estimate:.2f}")
                print(f"Generation time: {result.generation_time:.1f}s")
            else:
                print(f"Failed: {result.error}")

            # Print cost summary
            print("\nCost Summary:")
            summary = generator.get_cost_summary()
            print(f"  Total cost: ${summary['total_cost']:.2f}")
            print(f"  Total generations: {summary['total_generations']}")

        except ImportError as e:
            print(f"Import error: {e}")
            print("Install required packages: pip install fal-client httpx")
        except Exception as e:
            print(f"Error: {e}")

    asyncio.run(test_hailuo())
