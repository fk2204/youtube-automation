"""
Intelligent content type router.

Detects or accepts content type, then routes to the appropriate
creation pipeline and target platforms. Configuration-driven via
config/platforms.yaml routing section.
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from loguru import logger


# Default routing table (overridden by config/platforms.yaml)
DEFAULT_ROUTES: Dict[str, List[str]] = {
    "video_long": ["youtube", "tiktok", "instagram_reels", "pinterest"],
    "video_short": ["tiktok", "youtube_shorts", "instagram_reels"],
    "blog": ["medium", "linkedin", "quora"],
    "image": ["pinterest", "instagram_post", "twitter"],
    "carousel": ["instagram_post", "linkedin"],
    "text_post": ["twitter", "reddit", "linkedin", "facebook", "discord"],
}

# Content type detection by file extension
EXTENSION_MAP = {
    ".mp4": "video_long",
    ".mov": "video_long",
    ".avi": "video_long",
    ".webm": "video_long",
    ".md": "blog",
    ".html": "blog",
    ".txt": "text_post",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
}


@dataclass
class RouteDecision:
    """Result of routing a content piece."""
    content_type: str
    target_platforms: List[str]
    creation_pipeline: str  # Which pipeline to run
    adaptations: Dict[str, str] = field(default_factory=dict)  # platform -> format


class ContentRouter:
    """Routes content to appropriate pipelines and platforms.

    Loads routing rules from config/platforms.yaml. Falls back to
    DEFAULT_ROUTES if config unavailable.
    """

    def __init__(self, config_path: str = "config/platforms.yaml"):
        self._routes = DEFAULT_ROUTES.copy()
        self._load_config(config_path)

    def _load_config(self, path: str) -> None:
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            routing = data.get("routing", {})
            if routing:
                self._routes.update(routing)
        except FileNotFoundError:
            logger.debug(f"No routing config at {path}, using defaults")
        except Exception as e:
            logger.warning(f"Failed to load routing config: {e}")

    def route(
        self,
        content_type: Optional[str] = None,
        file_path: Optional[str] = None,
        platforms: Optional[List[str]] = None,
    ) -> RouteDecision:
        """Determine target platforms and pipeline for content.

        Args:
            content_type: Explicit type (video_long, blog, image, etc.)
            file_path: Path to content file (for auto-detection)
            platforms: Override target platforms

        Returns:
            RouteDecision with targets and pipeline.
        """
        if not content_type and file_path:
            content_type = self._detect_type(file_path)
        if not content_type:
            content_type = "text_post"

        target = platforms or self._routes.get(content_type, ["youtube"])
        pipeline = self._get_pipeline(content_type)
        adaptations = self._get_adaptations(content_type, target)

        return RouteDecision(
            content_type=content_type,
            target_platforms=target,
            creation_pipeline=pipeline,
            adaptations=adaptations,
        )

    def _detect_type(self, file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        detected = EXTENSION_MAP.get(ext, "text_post")

        # Distinguish short vs long video by duration or filename
        if detected == "video_long" and ("short" in file_path.lower() or "reel" in file_path.lower()):
            detected = "video_short"

        return detected

    def _get_pipeline(self, content_type: str) -> str:
        pipelines = {
            "video_long": "full_video",
            "video_short": "short_video",
            "blog": "blog_creation",
            "image": "image_creation",
            "carousel": "carousel_creation",
            "text_post": "text_creation",
        }
        return pipelines.get(content_type, "text_creation")

    def _get_adaptations(self, content_type: str, platforms: List[str]) -> Dict[str, str]:
        """Determine format adaptations needed per platform."""
        adaptations = {}
        if content_type in ("video_long", "video_short"):
            for p in platforms:
                if p in ("tiktok", "instagram_reels", "youtube_shorts"):
                    adaptations[p] = "portrait_9_16"
                elif p == "pinterest":
                    adaptations[p] = "portrait_2_3"
                else:
                    adaptations[p] = "landscape_16_9"
        elif content_type == "image":
            for p in platforms:
                if p == "pinterest":
                    adaptations[p] = "pin_1000x1500"
                elif p == "instagram_post":
                    adaptations[p] = "square_1080x1080"
                else:
                    adaptations[p] = "landscape_1200x675"
        return adaptations

    def get_all_routes(self) -> Dict[str, List[str]]:
        """Return the full routing table."""
        return self._routes.copy()


# Singleton
_router: Optional[ContentRouter] = None


def get_content_router() -> ContentRouter:
    global _router
    if _router is None:
        _router = ContentRouter()
    return _router
