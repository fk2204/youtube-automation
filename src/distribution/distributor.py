"""
Multi-platform content distribution orchestrator.

Takes created content (video, image, text) and distributes it across
all configured platforms with:
- Platform-specific format adaptation (VideoResizer)
- Metadata transformation (hashtags, descriptions, tags)
- Staggered upload timing
- First-hour social media boost
- Result tracking in database
- Error recovery and retry

This is the central hub that the API and daily automation call.
"""

import asyncio
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from src.social.platform_adapter import (
    ContentMetadata,
    ContentPlatform,
    PlatformRegistry,
    UploadResult,
    UploadStatus,
    get_platform_registry,
)


@dataclass
class DistributionResult:
    """Result of distributing content across platforms."""
    content_id: str = ""
    total_platforms: int = 0
    successful: int = 0
    failed: int = 0
    results: List[Dict[str, Any]] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["started_at"] = self.started_at.isoformat() if self.started_at else None
        d["completed_at"] = self.completed_at.isoformat() if self.completed_at else None
        return d


@dataclass
class DistributionConfig:
    """Configuration for distribution behavior."""
    stagger_delay_seconds: int = 30  # Delay between platform uploads
    retry_count: int = 2
    retry_delay_seconds: int = 10
    enable_social_boost: bool = True  # Trigger first-hour social posts
    enable_format_adaptation: bool = True  # Auto-resize videos per platform
    screenshot_on_error: bool = True


class ContentDistributor:
    """Orchestrates content distribution across all configured platforms.

    Usage:
        distributor = ContentDistributor()

        # Distribute a video
        result = await distributor.distribute(
            content_path="output/videos/my_video.mp4",
            content_type="video_long",
            title="My Video",
            niche="finance",
            channel="money_blueprints",
            platforms=["youtube", "tiktok"],
        )

        # Distribute with full metadata
        metadata = ContentMetadata(
            title="My Video",
            description="Full description...",
            tags=["finance", "investing"],
            niche="finance",
            channel_id="money_blueprints",
            video_path="output/videos/my_video.mp4",
        )
        result = await distributor.distribute_with_metadata(metadata)
    """

    def __init__(self, config: Optional[DistributionConfig] = None):
        self._config = config or DistributionConfig()
        self._registry = get_platform_registry()
        self._resizer = None

    async def distribute(
        self,
        content_path: str,
        content_type: str,
        title: str,
        niche: str,
        channel: str,
        description: str = "",
        tags: List[str] = None,
        platforms: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Distribute content to platforms.

        Args:
            content_path: Path to the content file
            content_type: Type of content (video_long, video_short, image, etc.)
            title: Content title
            niche: Content niche (finance, psychology, etc.)
            channel: Channel ID
            description: Content description
            tags: Content tags
            platforms: Specific platforms (None = all enabled)

        Returns:
            Dict with distribution results per platform
        """
        metadata = ContentMetadata(
            title=title,
            description=description,
            tags=tags or [],
            niche=niche,
            channel_id=channel,
            video_path=content_path if "video" in content_type else None,
            image_paths=[content_path] if content_type == "image" else [],
            hashtags=self._generate_hashtags(niche, tags or []),
        )

        return await self.distribute_with_metadata(metadata, platforms)

    async def distribute_with_metadata(
        self,
        metadata: ContentMetadata,
        platforms: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Distribute content with full metadata control.

        Args:
            metadata: Universal content metadata
            platforms: Specific platforms to target (None = all enabled)

        Returns:
            Dict with results per platform
        """
        dist = DistributionResult(started_at=datetime.utcnow())

        # Get target platforms
        if platforms:
            target_platforms = [
                (name, p) for name, p in self._registry.get_enabled()
                if name in platforms
            ]
        else:
            target_platforms = self._registry.get_enabled()

        dist.total_platforms = len(target_platforms)

        if not target_platforms:
            logger.warning("No platforms configured for distribution")
            dist.completed_at = datetime.utcnow()
            return dist.to_dict()

        logger.info(f"Distributing to {len(target_platforms)} platforms: {[n for n, _ in target_platforms]}")

        # Distribute to each platform with staggering
        for i, (name, platform) in enumerate(target_platforms):
            if i > 0 and self._config.stagger_delay_seconds > 0:
                logger.info(f"Stagger delay: {self._config.stagger_delay_seconds}s before {name}")
                await asyncio.sleep(self._config.stagger_delay_seconds)

            result = await self._upload_to_platform(name, platform, metadata)
            dist.results.append(result.to_dict())

            if result.status == UploadStatus.SUCCESS:
                dist.successful += 1
            else:
                dist.failed += 1

        dist.completed_at = datetime.utcnow()

        # Trigger first-hour social boost
        if self._config.enable_social_boost:
            await self._trigger_social_boost(metadata, dist.results)

        # Persist results
        await self._persist_results(metadata, dist)

        logger.info(
            f"Distribution complete: {dist.successful}/{dist.total_platforms} successful"
        )
        return dist.to_dict()

    async def _upload_to_platform(
        self, name: str, platform: ContentPlatform, metadata: ContentMetadata
    ) -> UploadResult:
        """Upload to a single platform with retry logic."""
        adapted = platform.adapt_metadata(metadata)

        for attempt in range(self._config.retry_count + 1):
            try:
                if metadata.video_path:
                    # Adapt video format if needed
                    video_path = metadata.video_path
                    if self._config.enable_format_adaptation:
                        video_path = await self._adapt_video_format(
                            metadata.video_path, name, platform
                        )

                    result = await platform.upload_video(video_path, adapted)
                elif metadata.image_paths:
                    result = await platform.upload_image(metadata.image_paths[0], adapted)
                elif metadata.blog_content:
                    result = await platform.post_text(metadata.blog_content, adapted)
                else:
                    result = await platform.post_text(metadata.description, adapted)

                if result.status == UploadStatus.SUCCESS:
                    logger.info(f"Successfully uploaded to {name}: {result.url or 'OK'}")
                    return result

                logger.warning(f"Upload to {name} returned {result.status.value}: {result.error}")

                if result.status == UploadStatus.AUTH_REQUIRED:
                    return result  # Don't retry auth failures

            except Exception as e:
                logger.error(f"Upload to {name} attempt {attempt + 1} failed: {e}")
                result = UploadResult(
                    platform=name,
                    status=UploadStatus.FAILED,
                    error=str(e),
                )

            if attempt < self._config.retry_count:
                delay = self._config.retry_delay_seconds * (attempt + 1)
                logger.info(f"Retrying {name} in {delay}s (attempt {attempt + 2})")
                await asyncio.sleep(delay)

        return result

    async def _adapt_video_format(
        self, video_path: str, platform_name: str, platform: ContentPlatform
    ) -> str:
        """Adapt video format for the target platform using VideoResizer."""
        specs = platform.get_platform_specs()
        if not specs:
            return video_path

        try:
            if self._resizer is None:
                from src.social.multi_platform import VideoResizer
                self._resizer = VideoResizer()

            # Check if video needs resizing
            adapted_path = str(
                Path("output/adapted") / f"{platform_name}_{Path(video_path).name}"
            )
            Path("output/adapted").mkdir(parents=True, exist_ok=True)

            # Only resize if aspect ratio differs
            if hasattr(specs, "aspect_ratio") and specs.aspect_ratio.value == "9:16":
                if hasattr(self._resizer, "resize_for_platform"):
                    self._resizer.resize_for_platform(video_path, adapted_path, specs)
                    if Path(adapted_path).exists():
                        return adapted_path

        except (ImportError, Exception) as e:
            logger.debug(f"Video adaptation skipped for {platform_name}: {e}")

        return video_path

    async def _trigger_social_boost(
        self, metadata: ContentMetadata, results: List[Dict]
    ) -> None:
        """Trigger first-hour social media boost for algorithm lift."""
        # Find YouTube URL from results
        youtube_url = None
        for r in results:
            if r.get("platform") == "youtube" and r.get("url"):
                youtube_url = r["url"]
                break

        if not youtube_url:
            return

        try:
            from src.social.social_poster import SocialMediaManager
            mgr = SocialMediaManager()
            if hasattr(mgr, "schedule_first_hour_posts"):
                mgr.schedule_first_hour_posts(
                    video_title=metadata.title,
                    video_url=youtube_url,
                    niche=metadata.niche,
                )
                logger.info("First-hour social boost scheduled")
        except (ImportError, Exception) as e:
            logger.debug(f"Social boost unavailable: {e}")

    async def _persist_results(
        self, metadata: ContentMetadata, dist: DistributionResult
    ) -> None:
        """Save distribution results to database."""
        try:
            from src.database.db import get_session
            from src.database.models import Upload, UploadStatus as DBStatus

            session = get_session()
            for r in dist.results:
                if r.get("status") == "success" and r.get("url"):
                    upload = Upload(
                        youtube_url=r["url"] if r["platform"] == "youtube" else None,
                        youtube_id=r.get("post_id"),
                        privacy=metadata.privacy,
                        status=DBStatus.COMPLETED,
                        uploaded_at=datetime.utcnow(),
                    )
                    session.add(upload)
            session.commit()
        except (ImportError, Exception) as e:
            logger.debug(f"Result persistence skipped: {e}")

    def _generate_hashtags(self, niche: str, tags: List[str]) -> List[str]:
        """Generate platform-appropriate hashtags from niche and tags."""
        niche_hashtags = {
            "finance": ["money", "investing", "financialfreedom", "personalfinance", "wealth"],
            "psychology": ["psychology", "mindset", "selfimprovement", "motivation", "mentalhealth"],
            "storytelling": ["stories", "truecrime", "mystery", "history", "storytelling"],
        }

        hashtags = list(tags[:5])
        hashtags.extend(niche_hashtags.get(niche, [niche])[:5])
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for h in hashtags:
            h_lower = h.lower().replace("#", "").replace(" ", "")
            if h_lower not in seen:
                seen.add(h_lower)
                unique.append(h_lower)
        return unique[:15]


# Singleton
_instance: Optional[ContentDistributor] = None


def get_distributor() -> ContentDistributor:
    global _instance
    if _instance is None:
        _instance = ContentDistributor()
    return _instance
