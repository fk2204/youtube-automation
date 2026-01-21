"""
Runway Gen-3/Gen-4 AI Video Generation

AI-powered video generation using Runway ML API.
Supports text-to-video, image-to-video, and batch processing
for generating B-roll clips and YouTube Shorts.

API Reference: https://docs.runwayml.com/
SDK: runwayml

Usage:
    from src.content.ai_video_runway import RunwayVideoGenerator

    async def main():
        generator = RunwayVideoGenerator()

        # Text-to-video
        result = await generator.generate_from_text(
            prompt="A cinematic sunset over the ocean with gentle waves",
            duration=5,
            aspect_ratio="16:9"
        )

        # Image-to-video
        result = await generator.generate_from_image(
            image_path="input/image.png",
            motion_prompt="Camera slowly zooms in as clouds drift by"
        )

        # Generate B-roll for script segment
        result = await generator.generate_broll(
            script_segment="The stock market crashed in 2008",
            style="cinematic"
        )

    asyncio.run(main())
"""

import os
import asyncio
import tempfile
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal, Union
from dataclasses import dataclass, field
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

# Try to import runwayml SDK
try:
    from runwayml import RunwayML
    RUNWAY_SDK_AVAILABLE = True
except ImportError:
    RUNWAY_SDK_AVAILABLE = False
    RunwayML = None
    logger.warning("runwayml SDK not installed. Install with: pip install runwayml")

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

# Import token manager for cost tracking
try:
    from ..utils.token_manager import get_token_manager, BudgetGuard
    TOKEN_MANAGER_AVAILABLE = True
except ImportError:
    TOKEN_MANAGER_AVAILABLE = False
    get_token_manager = None
    BudgetGuard = None


class RunwayModel(str, Enum):
    """Available Runway models for video generation."""
    GEN3A_TURBO = "gen3a_turbo"  # Faster, cheaper
    GEN3A_ALPHA = "gen3a_alpha"  # Higher quality


class RunwayAspectRatio(str, Enum):
    """Available aspect ratios for Runway video generation."""
    LANDSCAPE_16_9 = "16:9"
    PORTRAIT_9_16 = "9:16"
    WIDESCREEN_21_9 = "21:9"


# Pricing per second (USD) - approximate based on Runway pricing
RUNWAY_PRICING = {
    "gen3a_turbo": 0.05,  # $0.05 per second
    "gen3a_alpha": 0.10,  # $0.10 per second (higher quality)
}


@dataclass
class RunwayVideoResult:
    """Result from Runway video generation."""
    success: bool
    video_url: Optional[str] = None
    local_path: Optional[str] = None
    duration: Optional[float] = None
    aspect_ratio: Optional[str] = None
    model: Optional[str] = None
    error: Optional[str] = None
    task_id: Optional[str] = None
    cost_estimate: Optional[float] = None
    generation_time: Optional[float] = None

    def __bool__(self) -> bool:
        return self.success

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "video_url": self.video_url,
            "local_path": self.local_path,
            "duration": self.duration,
            "aspect_ratio": self.aspect_ratio,
            "model": self.model,
            "error": self.error,
            "task_id": self.task_id,
            "cost_estimate": self.cost_estimate,
            "generation_time": self.generation_time,
        }


@dataclass
class BrollGenerationResult:
    """Result from B-roll generation for a script segment."""
    success: bool
    segment_text: str
    video_results: List[RunwayVideoResult] = field(default_factory=list)
    total_cost: float = 0.0
    error: Optional[str] = None

    def __bool__(self) -> bool:
        return self.success


class RunwayVideoGenerator:
    """
    Runway ML video generator for text-to-video and image-to-video.

    Supports:
    - Text-to-video generation (Gen-3 Alpha/Turbo)
    - Image-to-video animation
    - Multiple aspect ratios (16:9, 9:16, 21:9)
    - Quality presets (turbo for speed, alpha for quality)
    - Cost tracking via token_manager
    - Retry logic with exponential backoff

    Pricing (approximate):
    - Gen-3 Turbo: ~$0.05/second
    - Gen-3 Alpha: ~$0.10/second
    """

    DEFAULT_MODEL = RunwayModel.GEN3A_TURBO
    DEFAULT_DURATION = 5  # seconds
    DEFAULT_ASPECT_RATIO = RunwayAspectRatio.LANDSCAPE_16_9
    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 2.0  # seconds

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        max_retries: int = 3
    ):
        """
        Initialize Runway video generator.

        Args:
            api_key: Runway API key. If not provided, uses RUNWAYML_API_SECRET
                     or RUNWAY_API_KEY environment variable.
            default_model: Default model to use (gen3a_turbo or gen3a_alpha)
            max_retries: Maximum retry attempts for failed requests
        """
        self.api_key = (
            api_key or
            os.getenv("RUNWAYML_API_SECRET") or
            os.getenv("RUNWAY_API_KEY")
        )

        if not RUNWAY_SDK_AVAILABLE:
            logger.error(
                "runwayml SDK not installed. "
                "Install with: pip install runwayml"
            )
            raise ImportError("runwayml SDK required for Runway video generation")

        if not self.api_key:
            logger.warning(
                "RUNWAYML_API_SECRET not set. "
                "Get your API key from https://app.runwayml.com/settings/api-keys "
                "and set RUNWAYML_API_SECRET environment variable."
            )
        else:
            logger.info("RunwayVideoGenerator initialized with API key")

        # Initialize SDK client
        self.client = RunwayML(api_key=self.api_key) if self.api_key else None

        # Default settings
        self.default_model = default_model or self.DEFAULT_MODEL
        self.max_retries = max_retries

        # Temp directory for downloads
        self.temp_dir = Path(tempfile.gettempdir()) / "runway_videos"
        self.temp_dir.mkdir(exist_ok=True)

        # Token tracker for cost management
        self.token_tracker = get_token_manager() if TOKEN_MANAGER_AVAILABLE else None

    def _validate_api_key(self) -> bool:
        """Check if API key is configured."""
        if not self.api_key:
            logger.error(
                "Runway API key not configured. "
                "Set RUNWAYML_API_SECRET environment variable."
            )
            return False
        return True

    def _estimate_cost(self, duration: int, model: str) -> float:
        """
        Estimate cost for video generation.

        Args:
            duration: Video duration in seconds
            model: Model name

        Returns:
            Estimated cost in USD
        """
        price_per_second = RUNWAY_PRICING.get(model, RUNWAY_PRICING["gen3a_turbo"])
        cost = price_per_second * duration
        return round(cost, 4)

    def _record_cost(self, cost: float, operation: str = "ai_video_runway") -> None:
        """Record cost to token tracker."""
        if self.token_tracker:
            # Record as a custom cost entry
            # Using tokens as a proxy (1 token = $0.0001 for tracking)
            tokens = int(cost * 10000)
            self.token_tracker.record_usage(
                provider="runway",
                input_tokens=tokens,
                output_tokens=0,
                operation=operation
            )
            logger.debug(f"Recorded Runway cost: ${cost:.4f}")

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
                await asyncio.to_thread(urllib.request.urlretrieve, url, str(output_path))
                logger.info(f"Downloaded video to: {output_path}")
                return str(output_path)

        except Exception as e:
            logger.error(f"Failed to download video: {e}")
            return None

    async def _wait_for_task(
        self,
        task_id: str,
        timeout: int = 300,
        poll_interval: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        Wait for a Runway task to complete.

        Args:
            task_id: Task ID to poll
            timeout: Maximum time to wait in seconds
            poll_interval: Time between polls in seconds

        Returns:
            Task result or None on timeout/failure
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                task = await asyncio.to_thread(
                    self.client.tasks.retrieve,
                    task_id
                )

                status = task.status if hasattr(task, 'status') else task.get('status')

                if status == "SUCCEEDED":
                    return task
                elif status == "FAILED":
                    error = task.failure if hasattr(task, 'failure') else task.get('failure')
                    logger.error(f"Task failed: {error}")
                    return None
                elif status in ["PENDING", "RUNNING"]:
                    logger.debug(f"Task {task_id} status: {status}")
                    await asyncio.sleep(poll_interval)
                else:
                    logger.warning(f"Unknown task status: {status}")
                    await asyncio.sleep(poll_interval)

            except Exception as e:
                logger.error(f"Error polling task: {e}")
                await asyncio.sleep(poll_interval)

        logger.error(f"Task {task_id} timed out after {timeout} seconds")
        return None

    async def _retry_with_backoff(
        self,
        func,
        *args,
        max_retries: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Execute function with exponential backoff retry.

        Args:
            func: Async function to execute
            *args: Positional arguments
            max_retries: Override max retries
            **kwargs: Keyword arguments

        Returns:
            Function result or raises last exception
        """
        retries = max_retries if max_retries is not None else self.max_retries
        last_exception = None

        for attempt in range(retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < retries - 1:
                    delay = self.RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {retries} attempts failed")

        raise last_exception

    async def generate_from_text(
        self,
        prompt: str,
        output_file: Optional[str] = None,
        duration: int = 5,
        aspect_ratio: str = "16:9",
        model: Optional[str] = None,
        seed: Optional[int] = None
    ) -> RunwayVideoResult:
        """
        Generate video from text prompt.

        Args:
            prompt: Text description of the video to generate.
                    Best practice: Be specific about subject, motion, style.
                    Example: "Cinematic aerial shot of a city at sunset,
                             golden hour lighting, slow camera pan right"
            output_file: Path to save the generated video. If None, saves to temp dir.
            duration: Video duration in seconds (5 or 10)
            aspect_ratio: Aspect ratio ("16:9", "9:16", or "21:9")
            model: Model to use (gen3a_turbo or gen3a_alpha)
            seed: Random seed for reproducibility

        Returns:
            RunwayVideoResult with video URL and local path
        """
        if not self._validate_api_key():
            return RunwayVideoResult(
                success=False,
                error="API key not configured"
            )

        # Validate parameters
        duration = max(5, min(10, duration))
        model = model or self.default_model
        if isinstance(model, RunwayModel):
            model = model.value

        # Map aspect ratio to Runway format
        ratio_map = {
            "16:9": "1280:720",
            "9:16": "720:1280",
            "21:9": "1680:720",
            "1280:720": "1280:720",
            "720:1280": "720:1280",
        }
        runway_ratio = ratio_map.get(aspect_ratio, "1280:720")

        logger.info(f"Generating text-to-video with Runway {model}")
        logger.info(f"  Prompt: {prompt[:100]}...")
        logger.info(f"  Duration: {duration}s, Aspect Ratio: {aspect_ratio}")

        cost_estimate = self._estimate_cost(duration, model)
        logger.info(f"  Estimated cost: ${cost_estimate:.2f}")

        # Check budget if available
        if TOKEN_MANAGER_AVAILABLE and BudgetGuard:
            try:
                with BudgetGuard(estimated_cost=cost_estimate) as guard:
                    if not guard.can_afford(cost_estimate):
                        logger.warning(f"Budget too low for generation (need ${cost_estimate:.4f})")
                        return RunwayVideoResult(
                            success=False,
                            error=f"Insufficient budget. Need ${cost_estimate:.4f}, have ${guard.remaining:.4f}"
                        )
            except Exception as e:
                logger.warning(f"Budget check failed: {e}")

        start_time = time.time()

        try:
            # Create the generation task
            async def _create_task():
                task = await asyncio.to_thread(
                    self.client.image_to_video.create,
                    model=model,
                    prompt_text=prompt,
                    duration=duration,
                    ratio=runway_ratio,
                    seed=seed
                )
                return task

            task = await self._retry_with_backoff(_create_task)

            task_id = task.id if hasattr(task, 'id') else task.get('id')
            logger.info(f"Task created: {task_id}")

            # Wait for completion
            result = await self._wait_for_task(task_id)

            if not result:
                return RunwayVideoResult(
                    success=False,
                    task_id=task_id,
                    error="Task failed or timed out"
                )

            # Extract video URL
            video_url = None
            if hasattr(result, 'output'):
                output = result.output
                if isinstance(output, list) and len(output) > 0:
                    video_url = output[0]
                elif isinstance(output, str):
                    video_url = output
            elif isinstance(result, dict):
                output = result.get('output', [])
                if isinstance(output, list) and len(output) > 0:
                    video_url = output[0]
                elif isinstance(output, str):
                    video_url = output

            if not video_url:
                logger.error(f"No video URL in result: {result}")
                return RunwayVideoResult(
                    success=False,
                    task_id=task_id,
                    error="No video URL in response"
                )

            generation_time = time.time() - start_time
            logger.success(f"Video generated in {generation_time:.1f}s: {video_url}")

            # Download to local file
            if not output_file:
                output_file = str(self.temp_dir / f"runway_t2v_{os.urandom(4).hex()}.mp4")

            local_path = await self._download_video(video_url, output_file)

            # Record cost
            self._record_cost(cost_estimate, "ai_video_text_to_video")

            return RunwayVideoResult(
                success=True,
                video_url=video_url,
                local_path=local_path,
                duration=duration,
                aspect_ratio=aspect_ratio,
                model=model,
                task_id=task_id,
                cost_estimate=cost_estimate,
                generation_time=generation_time
            )

        except Exception as e:
            logger.error(f"Runway text-to-video failed: {e}")
            return RunwayVideoResult(
                success=False,
                error=str(e)
            )

    async def generate_from_image(
        self,
        image_path: str,
        motion_prompt: str,
        output_file: Optional[str] = None,
        duration: int = 5,
        model: Optional[str] = None,
        seed: Optional[int] = None
    ) -> RunwayVideoResult:
        """
        Generate video from an image (animate the image).

        Args:
            image_path: Path to the input image (PNG, JPG, WebP)
            motion_prompt: Description of the motion/animation to apply.
                          Example: "Camera slowly zooms in, clouds drift across sky"
            output_file: Path to save the generated video
            duration: Video duration in seconds (5 or 10)
            model: Model to use (gen3a_turbo or gen3a_alpha)
            seed: Random seed for reproducibility

        Returns:
            RunwayVideoResult with video URL and local path
        """
        if not self._validate_api_key():
            return RunwayVideoResult(
                success=False,
                error="API key not configured"
            )

        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return RunwayVideoResult(
                success=False,
                error=f"Image file not found: {image_path}"
            )

        # Validate parameters
        duration = max(5, min(10, duration))
        model = model or self.default_model
        if isinstance(model, RunwayModel):
            model = model.value

        logger.info(f"Generating image-to-video with Runway {model}")
        logger.info(f"  Image: {image_path}")
        logger.info(f"  Motion: {motion_prompt[:100]}...")
        logger.info(f"  Duration: {duration}s")

        cost_estimate = self._estimate_cost(duration, model)
        logger.info(f"  Estimated cost: ${cost_estimate:.2f}")

        start_time = time.time()

        try:
            # Read image and encode as base64 or use file path
            # Runway SDK typically handles file uploads
            async def _create_task():
                # Open image file for upload
                with open(image_path, 'rb') as img_file:
                    task = await asyncio.to_thread(
                        self.client.image_to_video.create,
                        model=model,
                        prompt_image=img_file,
                        prompt_text=motion_prompt,
                        duration=duration,
                        seed=seed
                    )
                return task

            task = await self._retry_with_backoff(_create_task)

            task_id = task.id if hasattr(task, 'id') else task.get('id')
            logger.info(f"Task created: {task_id}")

            # Wait for completion
            result = await self._wait_for_task(task_id)

            if not result:
                return RunwayVideoResult(
                    success=False,
                    task_id=task_id,
                    error="Task failed or timed out"
                )

            # Extract video URL
            video_url = None
            if hasattr(result, 'output'):
                output = result.output
                if isinstance(output, list) and len(output) > 0:
                    video_url = output[0]
                elif isinstance(output, str):
                    video_url = output
            elif isinstance(result, dict):
                output = result.get('output', [])
                if isinstance(output, list) and len(output) > 0:
                    video_url = output[0]
                elif isinstance(output, str):
                    video_url = output

            if not video_url:
                logger.error(f"No video URL in result: {result}")
                return RunwayVideoResult(
                    success=False,
                    task_id=task_id,
                    error="No video URL in response"
                )

            generation_time = time.time() - start_time
            logger.success(f"Video generated in {generation_time:.1f}s: {video_url}")

            # Download to local file
            if not output_file:
                output_file = str(self.temp_dir / f"runway_i2v_{os.urandom(4).hex()}.mp4")

            local_path = await self._download_video(video_url, output_file)

            # Record cost
            self._record_cost(cost_estimate, "ai_video_image_to_video")

            return RunwayVideoResult(
                success=True,
                video_url=video_url,
                local_path=local_path,
                duration=duration,
                model=model,
                task_id=task_id,
                cost_estimate=cost_estimate,
                generation_time=generation_time
            )

        except Exception as e:
            logger.error(f"Runway image-to-video failed: {e}")
            return RunwayVideoResult(
                success=False,
                error=str(e)
            )

    async def generate_broll(
        self,
        script_segment: str,
        style: str = "cinematic",
        output_file: Optional[str] = None,
        duration: int = 5,
        model: Optional[str] = None,
        niche: str = "default"
    ) -> RunwayVideoResult:
        """
        Generate B-roll footage for a script segment.

        Analyzes the script segment to create an appropriate visual prompt.

        Args:
            script_segment: Text from the video script to visualize
            style: Visual style ("cinematic", "documentary", "corporate", "abstract")
            output_file: Path to save the generated video
            duration: Duration in seconds
            model: Model to use
            niche: Content niche for context (finance, psychology, storytelling)

        Returns:
            RunwayVideoResult
        """
        # Build style-specific prompt enhancements
        style_prompts = {
            "cinematic": "cinematic shot, shallow depth of field, dramatic lighting, film grain, professional cinematography",
            "documentary": "documentary style, natural lighting, realistic, observational, handheld camera feel",
            "corporate": "clean, professional, modern, well-lit, business setting, polished",
            "abstract": "abstract visuals, artistic, creative, symbolic imagery, visual metaphor",
            "dramatic": "dramatic lighting, high contrast, intense mood, tension, impactful",
            "calm": "serene atmosphere, soft lighting, peaceful, gentle motion, relaxed",
        }

        # Niche-specific visual enhancements
        niche_visuals = {
            "finance": "money, charts, business, wealth, investment, professional setting",
            "psychology": "human emotion, brain, thinking, relationships, abstract mind",
            "storytelling": "dramatic scene, narrative moment, atmospheric, mysterious",
            "technology": "futuristic, digital, tech devices, innovation, modern",
            "health": "wellness, medical, fitness, healthy lifestyle, nature",
        }

        style_addition = style_prompts.get(style, style_prompts["cinematic"])
        niche_addition = niche_visuals.get(niche, "")

        # Create visual prompt from script segment
        # Extract key concepts and create a visual description
        enhanced_prompt = (
            f"B-roll footage visualizing: {script_segment}. "
            f"Style: {style_addition}. "
            f"{niche_addition}. "
            f"16:9 widescreen, high quality, smooth motion."
        )

        logger.info(f"Generating B-roll for: {script_segment[:50]}...")

        return await self.generate_from_text(
            prompt=enhanced_prompt,
            output_file=output_file,
            duration=duration,
            aspect_ratio="16:9",
            model=model
        )

    async def batch_generate(
        self,
        prompts: List[str],
        output_dir: str,
        duration: int = 5,
        aspect_ratio: str = "16:9",
        model: Optional[str] = None,
        max_concurrent: int = 2
    ) -> List[RunwayVideoResult]:
        """
        Generate multiple video clips with controlled concurrency.

        Args:
            prompts: List of text prompts
            output_dir: Directory to save videos
            duration: Duration for each clip
            aspect_ratio: Aspect ratio for all clips
            model: Model to use
            max_concurrent: Maximum concurrent generations (be careful with API limits)

        Returns:
            List of RunwayVideoResult objects
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Estimate total cost
        model_name = model or self.default_model
        if isinstance(model_name, RunwayModel):
            model_name = model_name.value
        total_cost_estimate = self._estimate_cost(duration, model_name) * len(prompts)

        logger.info(f"Batch generation: {len(prompts)} videos")
        logger.info(f"  Total estimated cost: ${total_cost_estimate:.2f}")

        # Check budget
        if TOKEN_MANAGER_AVAILABLE and BudgetGuard:
            try:
                with BudgetGuard(estimated_cost=total_cost_estimate) as guard:
                    if not guard.can_afford(total_cost_estimate):
                        logger.error(f"Insufficient budget for batch (need ${total_cost_estimate:.4f})")
                        return [
                            RunwayVideoResult(
                                success=False,
                                error=f"Insufficient budget for batch generation"
                            )
                        ]
            except Exception as e:
                logger.warning(f"Budget check failed: {e}")

        results = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def generate_with_limit(idx: int, prompt: str) -> RunwayVideoResult:
            async with semaphore:
                output_file = str(output_path / f"clip_{idx:03d}.mp4")
                return await self.generate_from_text(
                    prompt=prompt,
                    output_file=output_file,
                    duration=duration,
                    aspect_ratio=aspect_ratio,
                    model=model
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
                processed_results.append(RunwayVideoResult(
                    success=False,
                    error=str(result)
                ))
            else:
                processed_results.append(result)

        # Log summary
        successful = sum(1 for r in processed_results if r.success)
        total_cost = sum(r.cost_estimate or 0 for r in processed_results if r.success)
        logger.info(f"Batch complete: {successful}/{len(prompts)} videos, total cost: ${total_cost:.2f}")

        return processed_results

    async def generate_vertical_short(
        self,
        prompt: str,
        output_file: Optional[str] = None,
        duration: int = 5,
        model: Optional[str] = None
    ) -> RunwayVideoResult:
        """
        Generate a vertical video optimized for YouTube Shorts.

        Args:
            prompt: Text description of the video
            output_file: Path to save the video
            duration: Duration in seconds (5-10)
            model: Model to use

        Returns:
            RunwayVideoResult
        """
        # Optimize prompt for vertical format
        enhanced_prompt = (
            f"{prompt}. "
            f"Vertical composition, mobile-friendly, centered subject, "
            f"9:16 aspect ratio, engaging visual, eye-catching."
        )

        return await self.generate_from_text(
            prompt=enhanced_prompt,
            output_file=output_file,
            duration=duration,
            aspect_ratio="9:16",
            model=model
        )

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary for Runway usage."""
        if not self.token_tracker:
            return {"error": "Token tracker not available"}

        # Get usage filtered by runway provider
        daily = self.token_tracker.get_daily_usage()
        # Note: This gets total usage, you might want to filter by provider

        return {
            "provider": "runway",
            "daily_cost": daily.get("cost", 0),
            "pricing": RUNWAY_PRICING,
            "note": "Cost tracking uses token_manager with runway provider"
        }


# Convenience function for quick access
def get_runway_generator(api_key: Optional[str] = None) -> Optional[RunwayVideoGenerator]:
    """
    Get a RunwayVideoGenerator instance.

    Args:
        api_key: Optional API key (uses environment variable if not provided)

    Returns:
        RunwayVideoGenerator instance or None if not available
    """
    try:
        return RunwayVideoGenerator(api_key=api_key)
    except ImportError as e:
        logger.error(f"Cannot create RunwayVideoGenerator: {e}")
        return None


# Add Runway to PROVIDER_COSTS in token_manager
def _extend_provider_costs():
    """Extend token_manager PROVIDER_COSTS with Runway pricing."""
    try:
        from ..utils.token_manager import PROVIDER_COSTS
        # Add runway as a provider (using per-token approximation)
        # 1 second of video ~ 500 "tokens" for cost tracking
        PROVIDER_COSTS["runway"] = {
            "input": 0.10,  # $0.05/sec = $0.10 per 1000 "tokens" (5s video)
            "output": 0.0
        }
    except Exception:
        pass


_extend_provider_costs()


# Example usage and testing
if __name__ == "__main__":
    async def test_runway():
        print("\n" + "=" * 60)
        print("RUNWAY VIDEO GENERATOR TEST")
        print("=" * 60 + "\n")

        try:
            generator = RunwayVideoGenerator()
            print(f"API Key configured: {bool(generator.api_key)}")
            print(f"Temp directory: {generator.temp_dir}")
            print(f"SDK available: {RUNWAY_SDK_AVAILABLE}")
            print(f"Default model: {generator.default_model}")

            if not generator.api_key:
                print("\nNo API key configured. Set RUNWAYML_API_SECRET to test.")
                print("Get your API key from: https://app.runwayml.com/settings/api-keys")
                return

            # Test text-to-video
            print("\nTesting text-to-video generation...")
            print("(This may take 1-3 minutes)")

            result = await generator.generate_from_text(
                prompt="A serene forest scene with sunlight filtering through trees, peaceful atmosphere, gentle camera movement",
                duration=5,
                model="gen3a_turbo"
            )

            if result.success:
                print(f"Success! Video saved to: {result.local_path}")
                print(f"Cost estimate: ${result.cost_estimate:.2f}")
                print(f"Generation time: {result.generation_time:.1f}s")
            else:
                print(f"Failed: {result.error}")

        except ImportError as e:
            print(f"Import error: {e}")
            print("Install required packages: pip install runwayml httpx")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(test_runway())
