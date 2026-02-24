"""
Full daily content empire automation.

Orchestrates the complete daily pipeline:
1. Research trending topics per niche
2. Create full content suite (video + shorts + blog + images + social)
3. Distribute to all configured platforms
4. Run analytics and optimization

Called by:
- API endpoint: POST /api/v1/schedule/daily
- CLI: python run.py empire-daily
- Scheduler: daily cron job
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class EmpireResult:
    """Result of a daily empire run."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    channels_processed: int = 0
    videos_created: int = 0
    shorts_created: int = 0
    blogs_created: int = 0
    images_created: int = 0
    social_posts_created: int = 0
    distributions: int = 0
    errors: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.videos_created > 0 or self.blogs_created > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "channels_processed": self.channels_processed,
            "videos_created": self.videos_created,
            "shorts_created": self.shorts_created,
            "blogs_created": self.blogs_created,
            "images_created": self.images_created,
            "social_posts_created": self.social_posts_created,
            "distributions": self.distributions,
            "errors": self.errors,
            "success": self.success,
        }


async def run_daily_empire(
    channels: Optional[List[str]] = None,
    content_types: Optional[List[str]] = None,
    distribute: bool = True,
) -> Dict[str, Any]:
    """Execute the full daily content empire pipeline.

    Args:
        channels: Channels to process (None = all configured)
        content_types: Content types to create (None = video + short)
        distribute: Whether to distribute to other platforms

    Returns:
        Dict with results per channel and content type.
    """
    result = EmpireResult()
    content_types = content_types or ["video_long", "video_short"]

    # Load channels
    if not channels:
        channels = _get_all_channels()

    logger.info(f"Empire daily run starting: {len(channels)} channels, types={content_types}")

    for channel in channels:
        niche = _get_channel_niche(channel)
        result.channels_processed += 1

        try:
            # Step 1: Research topics
            topics = await _research_topics(niche)
            if not topics:
                result.errors.append(f"No topics found for {channel}")
                continue

            topic = topics[0]  # Best topic
            logger.info(f"[{channel}] Topic: {topic}")

            # Step 2: Create content
            if "video_long" in content_types:
                video_result = await _create_video(channel, niche, topic, short=False)
                if video_result.get("status") == "success":
                    result.videos_created += 1

                    # Distribute video
                    if distribute:
                        dist = await _distribute_content(
                            content_path=video_result.get("video_path", ""),
                            content_type="video_long",
                            title=video_result.get("title", topic),
                            niche=niche,
                            channel=channel,
                        )
                        result.distributions += dist

            if "video_short" in content_types:
                short_result = await _create_video(channel, niche, topic, short=True)
                if short_result.get("status") == "success":
                    result.shorts_created += 1

                    if distribute:
                        dist = await _distribute_content(
                            content_path=short_result.get("video_path", ""),
                            content_type="video_short",
                            title=short_result.get("title", topic),
                            niche=niche,
                            channel=channel,
                        )
                        result.distributions += dist

            if "blog" in content_types:
                blog_result = await _create_blog(topic, niche, channel)
                if blog_result:
                    result.blogs_created += 1

            if "image" in content_types:
                img_result = await _create_images(topic, niche, channel)
                result.images_created += img_result

        except Exception as e:
            error_msg = f"[{channel}] Error: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)

    # Step 3: Run optimization
    try:
        from src.analytics.content_optimizer import get_content_optimizer
        optimizer = get_content_optimizer()
        insights = await optimizer.daily_analysis()
        result.details["optimization_insights"] = len(insights)
    except Exception as e:
        logger.debug(f"Optimization skipped: {e}")

    result.completed_at = datetime.utcnow()
    duration = (result.completed_at - result.started_at).total_seconds()

    logger.info(
        f"Empire daily complete: {result.videos_created} videos, "
        f"{result.shorts_created} shorts, {result.blogs_created} blogs, "
        f"{result.distributions} distributions in {duration:.1f}s"
    )

    return result.to_dict()


# --- Helper functions ---

async def _research_topics(niche: str, count: int = 3) -> List[str]:
    """Research trending topics for a niche."""
    try:
        from src.research.idea_generator import IdeaGenerator
        gen = IdeaGenerator(provider="groq")
        ideas = gen.generate_ideas(niche=niche, count=count)
        return [idea.topic if hasattr(idea, "topic") else str(idea) for idea in ideas]
    except Exception as e:
        logger.warning(f"Topic research failed: {e}")
        # Fallback topics
        fallbacks = {
            "finance": ["passive income strategies", "investing mistakes to avoid", "budgeting tips"],
            "psychology": ["dark psychology tricks", "body language secrets", "manipulation tactics"],
            "storytelling": ["unsolved mysteries", "bizarre historical events", "true crime cases"],
        }
        return fallbacks.get(niche, ["trending topics"])


async def _create_video(channel: str, niche: str, topic: str, short: bool) -> Dict[str, Any]:
    """Create a video using the existing pipeline."""
    try:
        from src.automation.unified_launcher import UnifiedLauncher
        launcher = UnifiedLauncher()
        result = await launcher.launch_full_pipeline(channel, "short" if short else "video")
        if result.success:
            return {
                "status": "success",
                "video_path": result.details.get("video_path", ""),
                "title": result.details.get("title", topic),
                **result.details,
            }
        return {"status": "failed", "errors": result.errors}
    except Exception as e:
        logger.error(f"Video creation failed: {e}")
        return {"status": "failed", "errors": [str(e)]}


async def _create_blog(topic: str, niche: str, channel: str) -> bool:
    """Create a blog article."""
    try:
        from src.content.blog_engine import BlogEngine
        engine = BlogEngine()
        article = await engine.create_article(topic=topic, niche=niche, channel=channel)
        return article is not None
    except Exception as e:
        logger.debug(f"Blog creation skipped: {e}")
        return False


async def _create_images(topic: str, niche: str, channel: str) -> int:
    """Create images and return count."""
    try:
        from src.content.image_engine import ImageEngine
        engine = ImageEngine()
        result = await engine.create_all(topic=topic, niche=niche, channel=channel)
        return len(result.get("images", []))
    except Exception as e:
        logger.debug(f"Image creation skipped: {e}")
        return 0


async def _distribute_content(
    content_path: str, content_type: str, title: str, niche: str, channel: str
) -> int:
    """Distribute content and return number of successful distributions."""
    if not content_path:
        return 0

    try:
        from src.distribution.distributor import ContentDistributor
        distributor = ContentDistributor()
        result = await distributor.distribute(
            content_path=content_path,
            content_type=content_type,
            title=title,
            niche=niche,
            channel=channel,
        )
        return result.get("successful", 0)
    except Exception as e:
        logger.warning(f"Distribution failed: {e}")
        return 0


def _get_all_channels() -> List[str]:
    """Get all configured channel IDs."""
    try:
        import yaml
        with open("config/channels.yaml") as f:
            data = yaml.safe_load(f)
        return list(data.get("channels", {}).keys())
    except Exception:
        return ["money_blueprints", "mind_unlocked", "untold_stories"]


def _get_channel_niche(channel: str) -> str:
    """Get niche for a channel."""
    niches = {
        "money_blueprints": "finance",
        "mind_unlocked": "psychology",
        "untold_stories": "storytelling",
    }
    return niches.get(channel, "general")
