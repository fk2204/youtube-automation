"""
AI Video Provider Abstraction

Unified interface for multiple AI video generation providers.
Supports easy switching between Runway, Pika, HailuoAI, and other providers.
Includes cost comparison and smart routing for optimal provider selection.

Usage:
    from src.content.ai_video_providers import (
        get_ai_video_provider,
        AIVideoProviderRouter,
        AIVideoProviderType
    )

    # Get a specific provider
    provider = get_ai_video_provider("runway")
    result = await provider.generate_video("A sunset over the ocean")

    # Use smart routing (auto-selects based on cost/quality)
    router = AIVideoProviderRouter()
    result = await router.generate_video(
        prompt="A sunset over the ocean",
        prefer_quality=True  # Use higher quality provider
    )
"""

import os
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Literal
from loguru import logger

# Load environment variables
from dotenv import load_dotenv
_env_paths = [
    Path(__file__).parent.parent.parent / "config" / ".env",
    Path.cwd() / "config" / ".env",
]
for _env_path in _env_paths:
    if _env_path.exists():
        load_dotenv(_env_path)
        break

# Import available providers
try:
    from .ai_video_runway import RunwayVideoGenerator, RunwayVideoResult
    RUNWAY_AVAILABLE = True
except ImportError:
    RUNWAY_AVAILABLE = False
    RunwayVideoGenerator = None
    RunwayVideoResult = None

try:
    from .video_pika import PikaVideoGenerator, PikaVideoResult
    PIKA_AVAILABLE = True
except ImportError:
    PIKA_AVAILABLE = False
    PikaVideoGenerator = None
    PikaVideoResult = None

# Import token manager for cost tracking
try:
    from ..utils.token_manager import get_token_manager, BudgetGuard
    TOKEN_MANAGER_AVAILABLE = True
except ImportError:
    TOKEN_MANAGER_AVAILABLE = False
    get_token_manager = None
    BudgetGuard = None


class AIVideoProviderType(str, Enum):
    """Supported AI video providers."""
    RUNWAY = "runway"
    PIKA = "pika"
    HAILUO = "hailuo"  # Future support
    KLING = "kling"    # Future support


@dataclass
class AIVideoResult:
    """Unified result from any AI video provider."""
    success: bool
    provider: str
    video_url: Optional[str] = None
    local_path: Optional[str] = None
    duration: Optional[float] = None
    aspect_ratio: Optional[str] = None
    error: Optional[str] = None
    task_id: Optional[str] = None
    cost_estimate: Optional[float] = None
    generation_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.success

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "provider": self.provider,
            "video_url": self.video_url,
            "local_path": self.local_path,
            "duration": self.duration,
            "aspect_ratio": self.aspect_ratio,
            "error": self.error,
            "task_id": self.task_id,
            "cost_estimate": self.cost_estimate,
            "generation_time": self.generation_time,
            "metadata": self.metadata,
        }


class AIVideoProvider(ABC):
    """
    Abstract base class for AI video providers.

    All AI video providers should inherit from this class and implement
    the required methods.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available (SDK installed, API key set)."""
        pass

    @abstractmethod
    async def generate_video(
        self,
        prompt: str,
        output_file: Optional[str] = None,
        duration: int = 5,
        aspect_ratio: str = "16:9",
        **kwargs
    ) -> AIVideoResult:
        """
        Generate video from text prompt.

        Args:
            prompt: Text description of the video
            output_file: Path to save the video
            duration: Video duration in seconds
            aspect_ratio: Aspect ratio (16:9, 9:16, etc.)
            **kwargs: Provider-specific options

        Returns:
            AIVideoResult with generation results
        """
        pass

    @abstractmethod
    async def generate_from_image(
        self,
        image_path: str,
        motion_prompt: str,
        output_file: Optional[str] = None,
        duration: int = 5,
        **kwargs
    ) -> AIVideoResult:
        """
        Generate video from image with motion.

        Args:
            image_path: Path to input image
            motion_prompt: Description of motion/animation
            output_file: Path to save the video
            duration: Video duration in seconds
            **kwargs: Provider-specific options

        Returns:
            AIVideoResult with generation results
        """
        pass

    @abstractmethod
    def get_cost_per_second(self) -> float:
        """
        Get cost per second of video generation.

        Returns:
            Cost in USD per second
        """
        pass

    def estimate_cost(self, duration: int) -> float:
        """
        Estimate cost for generating a video.

        Args:
            duration: Video duration in seconds

        Returns:
            Estimated cost in USD
        """
        return self.get_cost_per_second() * duration

    def get_provider_name(self) -> str:
        """Get provider name (alias for name property)."""
        return self.name

    def get_cost_per_video(self, duration: int = 5) -> float:
        """Get cost for a standard video (default 5 seconds)."""
        return self.estimate_cost(duration)


class RunwayProvider(AIVideoProvider):
    """Runway ML video provider implementation."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gen3a_turbo"):
        self._api_key = api_key or os.getenv("RUNWAYML_API_SECRET") or os.getenv("RUNWAY_API_KEY")
        self._model = model
        self._generator = None

        if RUNWAY_AVAILABLE and self._api_key:
            try:
                self._generator = RunwayVideoGenerator(api_key=self._api_key, default_model=model)
            except Exception as e:
                logger.warning(f"Failed to initialize Runway: {e}")

    @property
    def name(self) -> str:
        return "runway"

    @property
    def is_available(self) -> bool:
        return RUNWAY_AVAILABLE and self._generator is not None

    async def generate_video(
        self,
        prompt: str,
        output_file: Optional[str] = None,
        duration: int = 5,
        aspect_ratio: str = "16:9",
        **kwargs
    ) -> AIVideoResult:
        if not self.is_available:
            return AIVideoResult(
                success=False,
                provider=self.name,
                error="Runway provider not available"
            )

        result = await self._generator.generate_from_text(
            prompt=prompt,
            output_file=output_file,
            duration=duration,
            aspect_ratio=aspect_ratio,
            model=kwargs.get("model", self._model),
            seed=kwargs.get("seed")
        )

        return AIVideoResult(
            success=result.success,
            provider=self.name,
            video_url=result.video_url,
            local_path=result.local_path,
            duration=result.duration,
            aspect_ratio=result.aspect_ratio,
            error=result.error,
            task_id=result.task_id,
            cost_estimate=result.cost_estimate,
            generation_time=result.generation_time,
            metadata={"model": result.model}
        )

    async def generate_from_image(
        self,
        image_path: str,
        motion_prompt: str,
        output_file: Optional[str] = None,
        duration: int = 5,
        **kwargs
    ) -> AIVideoResult:
        if not self.is_available:
            return AIVideoResult(
                success=False,
                provider=self.name,
                error="Runway provider not available"
            )

        result = await self._generator.generate_from_image(
            image_path=image_path,
            motion_prompt=motion_prompt,
            output_file=output_file,
            duration=duration,
            model=kwargs.get("model", self._model),
            seed=kwargs.get("seed")
        )

        return AIVideoResult(
            success=result.success,
            provider=self.name,
            video_url=result.video_url,
            local_path=result.local_path,
            duration=result.duration,
            error=result.error,
            task_id=result.task_id,
            cost_estimate=result.cost_estimate,
            generation_time=result.generation_time,
            metadata={"model": result.model}
        )

    def get_cost_per_second(self) -> float:
        # Pricing varies by model
        pricing = {
            "gen3a_turbo": 0.05,
            "gen3a_alpha": 0.10,
        }
        return pricing.get(self._model, 0.05)


class PikaProvider(AIVideoProvider):
    """Pika Labs video provider implementation."""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("FAL_KEY") or os.getenv("PIKA_API_KEY")
        self._generator = None

        if PIKA_AVAILABLE and self._api_key:
            try:
                self._generator = PikaVideoGenerator(api_key=self._api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Pika: {e}")

    @property
    def name(self) -> str:
        return "pika"

    @property
    def is_available(self) -> bool:
        return PIKA_AVAILABLE and self._generator is not None

    async def generate_video(
        self,
        prompt: str,
        output_file: Optional[str] = None,
        duration: int = 5,
        aspect_ratio: str = "16:9",
        **kwargs
    ) -> AIVideoResult:
        if not self.is_available:
            return AIVideoResult(
                success=False,
                provider=self.name,
                error="Pika provider not available"
            )

        resolution = kwargs.get("resolution", "720p")

        result = await self._generator.generate_from_text(
            prompt=prompt,
            output_file=output_file,
            duration=duration,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            negative_prompt=kwargs.get("negative_prompt"),
            seed=kwargs.get("seed")
        )

        return AIVideoResult(
            success=result.success,
            provider=self.name,
            video_url=result.video_url,
            local_path=result.local_path,
            duration=result.duration,
            aspect_ratio=aspect_ratio,
            error=result.error,
            task_id=result.request_id,
            cost_estimate=result.cost_estimate,
            metadata={"resolution": resolution}
        )

    async def generate_from_image(
        self,
        image_path: str,
        motion_prompt: str,
        output_file: Optional[str] = None,
        duration: int = 5,
        **kwargs
    ) -> AIVideoResult:
        if not self.is_available:
            return AIVideoResult(
                success=False,
                provider=self.name,
                error="Pika provider not available"
            )

        resolution = kwargs.get("resolution", "720p")

        result = await self._generator.generate_from_image(
            image_path=image_path,
            prompt=motion_prompt,
            output_file=output_file,
            duration=duration,
            resolution=resolution,
            use_turbo=kwargs.get("use_turbo", False),
            negative_prompt=kwargs.get("negative_prompt"),
            seed=kwargs.get("seed")
        )

        return AIVideoResult(
            success=result.success,
            provider=self.name,
            video_url=result.video_url,
            local_path=result.local_path,
            duration=result.duration,
            error=result.error,
            task_id=result.request_id,
            cost_estimate=result.cost_estimate,
            metadata={"resolution": resolution}
        )

    def get_cost_per_second(self) -> float:
        # Pika pricing is per 5 seconds
        # $0.20/5s at 720p = $0.04/sec
        return 0.04


# Provider registry
_PROVIDER_CLASSES: Dict[str, type] = {
    "runway": RunwayProvider,
    "pika": PikaProvider,
}

# Provider cost comparison (cost per second in USD)
PROVIDER_COSTS = {
    "runway_turbo": 0.05,
    "runway_alpha": 0.10,
    "pika_720p": 0.04,
    "pika_1080p": 0.09,
}

# Provider quality rankings (1-10, higher is better)
PROVIDER_QUALITY = {
    "runway_alpha": 10,
    "runway_turbo": 8,
    "pika_1080p": 7,
    "pika_720p": 6,
}


def get_ai_video_provider(
    name: str,
    api_key: Optional[str] = None,
    **kwargs
) -> Optional[AIVideoProvider]:
    """
    Factory function to get an AI video provider.

    Args:
        name: Provider name ("runway", "pika", etc.)
        api_key: Optional API key (uses environment if not provided)
        **kwargs: Provider-specific configuration

    Returns:
        AIVideoProvider instance or None if not available
    """
    name = name.lower()

    if name not in _PROVIDER_CLASSES:
        logger.error(f"Unknown provider: {name}. Available: {list(_PROVIDER_CLASSES.keys())}")
        return None

    provider_class = _PROVIDER_CLASSES[name]

    try:
        return provider_class(api_key=api_key, **kwargs)
    except Exception as e:
        logger.error(f"Failed to create provider {name}: {e}")
        return None


def get_available_providers() -> List[str]:
    """Get list of available (configured) providers."""
    available = []

    for name in _PROVIDER_CLASSES:
        provider = get_ai_video_provider(name)
        if provider and provider.is_available:
            available.append(name)

    return available


class AIVideoProviderRouter:
    """
    Smart router for AI video generation.

    Automatically selects the best provider based on:
    - Cost constraints
    - Quality preferences
    - Provider availability
    - Budget limits
    """

    def __init__(
        self,
        preferred_providers: Optional[List[str]] = None,
        budget_limit: Optional[float] = None
    ):
        """
        Initialize the router.

        Args:
            preferred_providers: List of preferred providers in order
            budget_limit: Maximum cost per video in USD
        """
        self.preferred_providers = preferred_providers or ["runway", "pika"]
        self.budget_limit = budget_limit

        # Initialize available providers
        self.providers: Dict[str, AIVideoProvider] = {}
        for name in self.preferred_providers:
            provider = get_ai_video_provider(name)
            if provider and provider.is_available:
                self.providers[name] = provider

        logger.info(f"Router initialized with providers: {list(self.providers.keys())}")

    def select_provider(
        self,
        duration: int = 5,
        prefer_quality: bool = False,
        prefer_speed: bool = False,
        max_cost: Optional[float] = None
    ) -> Optional[AIVideoProvider]:
        """
        Select the best provider based on requirements.

        Args:
            duration: Video duration in seconds
            prefer_quality: Prefer higher quality over cost
            prefer_speed: Prefer faster generation
            max_cost: Maximum cost for this video

        Returns:
            Selected provider or None if none available
        """
        if not self.providers:
            logger.error("No providers available")
            return None

        max_cost = max_cost or self.budget_limit

        # Score each provider
        candidates = []
        for name, provider in self.providers.items():
            cost = provider.estimate_cost(duration)

            # Skip if over budget
            if max_cost and cost > max_cost:
                logger.debug(f"Skipping {name}: cost ${cost:.2f} > max ${max_cost:.2f}")
                continue

            # Calculate score
            score = 0

            # Quality score (0-10 points)
            quality_key = f"{name}_turbo" if not prefer_quality else f"{name}_alpha"
            quality = PROVIDER_QUALITY.get(quality_key, 5)
            score += quality * (2 if prefer_quality else 1)

            # Cost score (inverse - cheaper is better)
            cost_key = f"{name}_turbo"
            base_cost = PROVIDER_COSTS.get(cost_key, 0.05)
            cost_score = 10 - (base_cost * 100)  # Lower cost = higher score
            score += cost_score * (0.5 if prefer_quality else 1)

            # Speed bonus for turbo models
            if prefer_speed and "turbo" in name.lower():
                score += 3

            candidates.append((name, provider, score, cost))

        if not candidates:
            logger.warning("No providers within budget")
            return None

        # Sort by score (highest first)
        candidates.sort(key=lambda x: x[2], reverse=True)

        selected_name, selected_provider, score, cost = candidates[0]
        logger.info(f"Selected provider: {selected_name} (score: {score:.1f}, cost: ${cost:.2f})")

        return selected_provider

    async def generate_video(
        self,
        prompt: str,
        output_file: Optional[str] = None,
        duration: int = 5,
        aspect_ratio: str = "16:9",
        prefer_quality: bool = False,
        prefer_speed: bool = False,
        max_cost: Optional[float] = None,
        fallback: bool = True,
        **kwargs
    ) -> AIVideoResult:
        """
        Generate video using the best available provider.

        Args:
            prompt: Text description of the video
            output_file: Path to save the video
            duration: Video duration in seconds
            aspect_ratio: Aspect ratio
            prefer_quality: Prefer higher quality
            prefer_speed: Prefer faster generation
            max_cost: Maximum cost for this video
            fallback: Try other providers if selected one fails
            **kwargs: Additional provider-specific options

        Returns:
            AIVideoResult from the selected provider
        """
        provider = self.select_provider(
            duration=duration,
            prefer_quality=prefer_quality,
            prefer_speed=prefer_speed,
            max_cost=max_cost
        )

        if not provider:
            return AIVideoResult(
                success=False,
                provider="none",
                error="No suitable provider available"
            )

        # Try selected provider
        result = await provider.generate_video(
            prompt=prompt,
            output_file=output_file,
            duration=duration,
            aspect_ratio=aspect_ratio,
            **kwargs
        )

        # Try fallback providers if failed
        if not result.success and fallback:
            for name, fallback_provider in self.providers.items():
                if fallback_provider == provider:
                    continue

                logger.warning(f"Primary provider failed, trying {name}...")

                result = await fallback_provider.generate_video(
                    prompt=prompt,
                    output_file=output_file,
                    duration=duration,
                    aspect_ratio=aspect_ratio,
                    **kwargs
                )

                if result.success:
                    break

        return result

    async def generate_from_image(
        self,
        image_path: str,
        motion_prompt: str,
        output_file: Optional[str] = None,
        duration: int = 5,
        prefer_quality: bool = False,
        max_cost: Optional[float] = None,
        **kwargs
    ) -> AIVideoResult:
        """
        Generate video from image using the best available provider.

        Args:
            image_path: Path to input image
            motion_prompt: Description of motion/animation
            output_file: Path to save the video
            duration: Video duration in seconds
            prefer_quality: Prefer higher quality
            max_cost: Maximum cost for this video
            **kwargs: Additional provider-specific options

        Returns:
            AIVideoResult from the selected provider
        """
        provider = self.select_provider(
            duration=duration,
            prefer_quality=prefer_quality,
            max_cost=max_cost
        )

        if not provider:
            return AIVideoResult(
                success=False,
                provider="none",
                error="No suitable provider available"
            )

        return await provider.generate_from_image(
            image_path=image_path,
            motion_prompt=motion_prompt,
            output_file=output_file,
            duration=duration,
            **kwargs
        )

    async def generate_broll(
        self,
        script_segment: str,
        style: str = "cinematic",
        output_file: Optional[str] = None,
        duration: int = 5,
        niche: str = "default",
        **kwargs
    ) -> AIVideoResult:
        """
        Generate B-roll footage for a script segment.

        Args:
            script_segment: Text from the script to visualize
            style: Visual style
            output_file: Path to save the video
            duration: Duration in seconds
            niche: Content niche for context
            **kwargs: Additional options

        Returns:
            AIVideoResult
        """
        # Build enhanced prompt based on style and niche
        style_prompts = {
            "cinematic": "cinematic shot, dramatic lighting, film grain, professional",
            "documentary": "documentary style, natural lighting, realistic",
            "corporate": "clean, professional, modern, well-lit",
            "abstract": "abstract visuals, artistic, symbolic",
        }

        niche_context = {
            "finance": "business, money, investment, professional setting",
            "psychology": "human emotion, mind, thinking, relationships",
            "storytelling": "dramatic, narrative, atmospheric",
            "technology": "futuristic, digital, tech, innovation",
        }

        style_addition = style_prompts.get(style, style_prompts["cinematic"])
        niche_addition = niche_context.get(niche, "")

        enhanced_prompt = (
            f"B-roll footage: {script_segment}. "
            f"{style_addition}. {niche_addition}. "
            f"16:9 widescreen, high quality."
        )

        return await self.generate_video(
            prompt=enhanced_prompt,
            output_file=output_file,
            duration=duration,
            aspect_ratio="16:9",
            **kwargs
        )

    def compare_costs(self, duration: int = 5) -> Dict[str, float]:
        """
        Compare costs across all available providers.

        Args:
            duration: Video duration in seconds

        Returns:
            Dict mapping provider name to estimated cost
        """
        costs = {}
        for name, provider in self.providers.items():
            costs[name] = provider.estimate_cost(duration)
        return costs

    def get_status(self) -> Dict[str, Any]:
        """Get router status and provider information."""
        status = {
            "available_providers": list(self.providers.keys()),
            "preferred_order": self.preferred_providers,
            "budget_limit": self.budget_limit,
            "provider_details": {}
        }

        for name, provider in self.providers.items():
            status["provider_details"][name] = {
                "available": provider.is_available,
                "cost_per_second": provider.get_cost_per_second(),
                "cost_5s": provider.estimate_cost(5),
                "cost_10s": provider.estimate_cost(10),
            }

        return status

    def get_cheapest_provider(self) -> Optional[AIVideoProvider]:
        """
        Get the cheapest available provider.

        Returns:
            Cheapest AIVideoProvider or None if none available
        """
        if not self.providers:
            return None

        cheapest = None
        lowest_cost = float('inf')

        for name, provider in self.providers.items():
            cost = provider.get_cost_per_second()
            if cost < lowest_cost:
                lowest_cost = cost
                cheapest = provider

        return cheapest

    def estimate_batch_cost(
        self,
        clip_count: int,
        provider_name: Optional[str] = None,
        duration: int = 5
    ) -> Dict[str, Any]:
        """
        Estimate cost for generating multiple clips.

        Args:
            clip_count: Number of clips to generate
            provider_name: Specific provider to estimate for, or None for all
            duration: Duration per clip in seconds

        Returns:
            Dict with cost estimates
        """
        if provider_name and provider_name in self.providers:
            provider = self.providers[provider_name]
            cost_per_clip = provider.estimate_cost(duration)
            return {
                "clip_count": clip_count,
                "provider": provider_name,
                "cost_per_clip": cost_per_clip,
                "total_cost": cost_per_clip * clip_count,
            }

        # Estimate for all providers
        estimates = []
        for name, provider in self.providers.items():
            cost_per_clip = provider.estimate_cost(duration)
            quality = PROVIDER_QUALITY.get(f"{name}_turbo", 5) / 10.0
            estimates.append({
                "provider": name,
                "cost_per_clip": cost_per_clip,
                "total_cost": cost_per_clip * clip_count,
                "quality": quality,
            })

        # Sort by cost
        estimates.sort(key=lambda x: x["cost_per_clip"])

        # Calculate savings
        savings = 0
        if len(estimates) >= 2:
            savings = (estimates[-1]["total_cost"] - estimates[0]["total_cost"])

        return {
            "clip_count": clip_count,
            "duration_per_clip": duration,
            "estimates": estimates,
            "cheapest": estimates[0]["provider"] if estimates else None,
            "savings_vs_pika": savings,
        }


# =============================================================================
# Compatibility functions for existing CLI commands in run.py
# =============================================================================

def get_smart_router(budget_limit: Optional[float] = None) -> AIVideoProviderRouter:
    """
    Get a smart router instance for AI video generation.

    This is a convenience function used by run.py CLI commands.

    Args:
        budget_limit: Optional budget limit per video

    Returns:
        AIVideoProviderRouter instance
    """
    return AIVideoProviderRouter(budget_limit=budget_limit)


def list_providers() -> List[Dict[str, Any]]:
    """
    List all registered AI video providers with their status.

    This function is used by run.py CLI for `ai-video-providers` command.

    Returns:
        List of provider info dicts with name, cost, quality, availability
    """
    providers_info = []

    for name in _PROVIDER_CLASSES:
        provider = get_ai_video_provider(name)

        info = {
            "name": name,
            "available": provider.is_available if provider else False,
            "cost_per_video": provider.estimate_cost(5) if (provider and provider.is_available) else None,
            "cost_per_second": provider.get_cost_per_second() if (provider and provider.is_available) else None,
            "quality_score": PROVIDER_QUALITY.get(f"{name}_turbo", 0.5) / 10.0 if provider else None,
        }

        providers_info.append(info)

    return providers_info


def print_provider_comparison():
    """Print a comparison of all AI video providers."""
    print("\n" + "=" * 70)
    print("AI VIDEO PROVIDER COMPARISON")
    print("=" * 70)

    router = AIVideoProviderRouter()
    status = router.get_status()

    print(f"\nAvailable Providers: {status['available_providers']}")
    print(f"Budget Limit: ${status['budget_limit'] or 'None'}")

    print(f"\n{'Provider':<15} {'Available':<12} {'$/sec':<10} {'5s Cost':<10} {'10s Cost':<10}")
    print("-" * 60)

    for name, details in status["provider_details"].items():
        available = "Yes" if details["available"] else "No"
        print(
            f"{name:<15} {available:<12} "
            f"${details['cost_per_second']:<9.2f} "
            f"${details['cost_5s']:<9.2f} "
            f"${details['cost_10s']:<9.2f}"
        )

    # Show all registered providers (including unavailable)
    print("\n\nAll Registered Providers:")
    print("-" * 40)
    for name in _PROVIDER_CLASSES:
        provider = get_ai_video_provider(name)
        if provider:
            status_str = "Ready" if provider.is_available else "Not configured"
        else:
            status_str = "SDK not installed"
        print(f"  {name}: {status_str}")

    print("\n" + "=" * 70)


# Example usage and testing
if __name__ == "__main__":
    async def test_providers():
        print("\n" + "=" * 60)
        print("AI VIDEO PROVIDERS TEST")
        print("=" * 60 + "\n")

        # Show provider comparison
        print_provider_comparison()

        # Test router
        router = AIVideoProviderRouter()

        print("\n\nCost comparison for 5-second video:")
        costs = router.compare_costs(5)
        for provider, cost in costs.items():
            print(f"  {provider}: ${cost:.2f}")

        print("\nCost comparison for 10-second video:")
        costs = router.compare_costs(10)
        for provider, cost in costs.items():
            print(f"  {provider}: ${cost:.2f}")

        # Select provider
        print("\n\nProvider selection tests:")

        # Cost-optimized
        provider = router.select_provider(duration=5, prefer_quality=False)
        if provider:
            print(f"  Cost-optimized: {provider.name}")

        # Quality-optimized
        provider = router.select_provider(duration=5, prefer_quality=True)
        if provider:
            print(f"  Quality-optimized: {provider.name}")

        # Budget-constrained
        provider = router.select_provider(duration=5, max_cost=0.20)
        if provider:
            print(f"  Budget ($0.20 max): {provider.name}")

    asyncio.run(test_providers())
