"""
Unified ContentPlatform adapter interface.

All content distribution platforms implement this ABC. The PlatformRegistry
discovers and manages configured platforms. The ContentDistributor uses
the registry to route content to the right uploaders.

Existing integrations:
- YouTube: Official API via src/youtube/uploader.py
- Twitter, Reddit, Discord, LinkedIn, Facebook: Text/image via src/social/social_poster.py
- TikTok, Instagram: Video export via src/social/multi_platform.py (no upload)

This module provides the unified interface that adds video upload capabilities
to all platforms, plus a registry for dynamic platform management.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from loguru import logger

# Reuse existing platform specs
try:
    from src.social.multi_platform import PlatformSpec, PLATFORM_SPECS, AspectRatio
except ImportError:
    PlatformSpec = None
    PLATFORM_SPECS = {}
    AspectRatio = None


class UploadStatus(str, Enum):
    """Upload result status."""
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"
    RATE_LIMITED = "rate_limited"
    AUTH_REQUIRED = "auth_required"


@dataclass
class ContentMetadata:
    """Universal content metadata — platform-agnostic.

    The ContentDistributor creates one of these per content piece.
    Each platform adapter then transforms it via adapt_metadata().
    """
    title: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    category: str = "Education"
    niche: str = "general"
    channel_id: str = ""
    language: str = "en"
    privacy: str = "public"
    video_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    audio_path: Optional[str] = None
    image_paths: List[str] = field(default_factory=list)
    blog_content: Optional[str] = None
    hashtags: List[str] = field(default_factory=list)
    scheduled_time: Optional[datetime] = None
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["scheduled_time"] = self.scheduled_time.isoformat() if self.scheduled_time else None
        return d


@dataclass
class PlatformMetadata:
    """Platform-specific adapted metadata.

    Created by ContentPlatform.adapt_metadata() from universal ContentMetadata.
    Each field is already formatted for the target platform's limits.
    """
    title: str
    description: str
    hashtags: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    category: str = ""
    privacy: str = "public"
    language: str = "en"
    scheduled_time: Optional[datetime] = None
    platform_specific: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UploadResult:
    """Result of an upload/post operation."""
    platform: str
    status: UploadStatus
    url: Optional[str] = None
    post_id: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    uploaded_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["uploaded_at"] = self.uploaded_at.isoformat() if self.uploaded_at else None
        return d


class ContentPlatform(ABC):
    """Abstract base class for all content distribution platforms.

    Implement this interface to add a new platform. Register the
    implementation with PlatformRegistry.

    Each platform handles:
    1. Video upload (primary use case)
    2. Image upload (Pinterest pins, Instagram posts)
    3. Text posting (Twitter, Reddit, LinkedIn)
    4. Metadata adaptation (format metadata for platform constraints)
    """

    @abstractmethod
    async def upload_video(self, video_path: str, metadata: PlatformMetadata) -> UploadResult:
        """Upload a video file to the platform."""
        ...

    @abstractmethod
    async def upload_image(self, image_path: str, metadata: PlatformMetadata) -> UploadResult:
        """Upload an image to the platform."""
        ...

    @abstractmethod
    async def post_text(self, content: str, metadata: PlatformMetadata) -> UploadResult:
        """Post text content to the platform."""
        ...

    @abstractmethod
    def adapt_metadata(self, base: ContentMetadata) -> PlatformMetadata:
        """Transform universal metadata to platform-specific format.

        Must respect platform constraints:
        - Character limits for title/description
        - Hashtag limits
        - Tag formatting rules
        """
        ...

    @abstractmethod
    def get_platform_name(self) -> str:
        """Return the platform identifier (e.g. 'youtube', 'tiktok')."""
        ...

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if platform has valid credentials/config."""
        ...

    def get_platform_specs(self) -> Optional[Any]:
        """Return PlatformSpec if available (video platforms)."""
        return None

    async def health_check(self) -> bool:
        """Check if platform is reachable. Override for custom logic."""
        return self.is_configured()


class PlatformRegistry:
    """Registry of all configured content platforms.

    Supports dynamic registration, priority-ordered iteration,
    and platform lookup by name.
    """

    def __init__(self):
        self._platforms: Dict[str, ContentPlatform] = {}
        self._priorities: Dict[str, int] = {}
        self._auto_discover()

    def register(self, name: str, platform: ContentPlatform, priority: int = 50) -> None:
        """Register a platform adapter.

        Args:
            name: Platform identifier (e.g. 'youtube', 'tiktok')
            platform: ContentPlatform instance
            priority: Upload order (lower = first). YouTube=10, TikTok=20, etc.
        """
        self._platforms[name] = platform
        self._priorities[name] = priority
        logger.debug(f"Registered platform: {name} (priority={priority})")

    def get(self, name: str) -> Optional[ContentPlatform]:
        """Get platform by name."""
        return self._platforms.get(name)

    def get_enabled(self) -> List[tuple]:
        """Get all enabled platforms sorted by priority.

        Returns: List of (name, platform) tuples, sorted by priority.
        """
        enabled = [
            (name, p) for name, p in self._platforms.items()
            if p.is_configured()
        ]
        enabled.sort(key=lambda x: self._priorities.get(x[0], 50))
        return enabled

    def list_platforms(self) -> List[Dict[str, Any]]:
        """List all registered platforms with status."""
        result = []
        for name, p in self._platforms.items():
            result.append({
                "name": name,
                "configured": p.is_configured(),
                "priority": self._priorities.get(name, 50),
                "type": type(p).__name__,
            })
        result.sort(key=lambda x: x["priority"])
        return result

    def _auto_discover(self) -> None:
        """Auto-discover and register available platforms."""
        # YouTube (always available)
        try:
            from src.social.youtube_platform import YouTubePlatform
            yt = YouTubePlatform()
            self.register("youtube", yt, priority=10)
        except ImportError:
            logger.debug("YouTubePlatform not available")

        # TikTok (browser automation)
        try:
            from src.social.tiktok_uploader import TikTokPlatform
            tt = TikTokPlatform()
            self.register("tiktok", tt, priority=20)
        except ImportError:
            logger.debug("TikTokPlatform not available")

        # Instagram (Phase 2)
        try:
            from src.social.instagram_uploader import InstagramPlatform
            ig = InstagramPlatform()
            self.register("instagram", ig, priority=30)
        except ImportError:
            logger.debug("InstagramPlatform not available (Phase 2)")

        # Pinterest (Phase 2)
        try:
            from src.social.pinterest_uploader import PinterestPlatform
            pin = PinterestPlatform()
            self.register("pinterest", pin, priority=40)
        except ImportError:
            logger.debug("PinterestPlatform not available (Phase 2)")

        # Legacy social platforms (text/image only, wrapped)
        try:
            from src.social.social_poster import SocialMediaManager
            mgr = SocialMediaManager()
            for name in ["twitter", "reddit", "linkedin", "facebook", "discord"]:
                poster = mgr.get_platform(name) if hasattr(mgr, "get_platform") else None
                if poster:
                    wrapped = LegacySocialPlatform(name, poster)
                    self.register(name, wrapped, priority=60)
        except (ImportError, Exception):
            logger.debug("Legacy social platforms not available")


class LegacySocialPlatform(ContentPlatform):
    """Wraps existing SocialPlatform (text/image) as ContentPlatform.

    The existing social_poster.py platforms only support text and image posting.
    This wrapper implements the ContentPlatform interface with video upload
    returning NOT_SUPPORTED.
    """

    def __init__(self, name: str, legacy_platform):
        self._name = name
        self._legacy = legacy_platform

    async def upload_video(self, video_path: str, metadata: PlatformMetadata) -> UploadResult:
        return UploadResult(
            platform=self._name,
            status=UploadStatus.FAILED,
            error=f"{self._name} does not support direct video upload via legacy adapter",
        )

    async def upload_image(self, image_path: str, metadata: PlatformMetadata) -> UploadResult:
        try:
            result = self._legacy.post(
                content=metadata.description,
                image=image_path,
            )
            return UploadResult(
                platform=self._name,
                status=UploadStatus.SUCCESS,
                post_id=result.get("post_id"),
                url=result.get("url"),
                uploaded_at=datetime.utcnow(),
            )
        except Exception as e:
            return UploadResult(
                platform=self._name,
                status=UploadStatus.FAILED,
                error=str(e),
            )

    async def post_text(self, content: str, metadata: PlatformMetadata) -> UploadResult:
        try:
            result = self._legacy.post(content=content)
            return UploadResult(
                platform=self._name,
                status=UploadStatus.SUCCESS,
                post_id=result.get("post_id"),
                url=result.get("url"),
                uploaded_at=datetime.utcnow(),
            )
        except Exception as e:
            return UploadResult(
                platform=self._name,
                status=UploadStatus.FAILED,
                error=str(e),
            )

    def adapt_metadata(self, base: ContentMetadata) -> PlatformMetadata:
        limits = {
            "twitter": {"desc": 280, "hashtags": 5},
            "reddit": {"desc": 40000, "hashtags": 0},
            "linkedin": {"desc": 3000, "hashtags": 10},
            "facebook": {"desc": 63206, "hashtags": 10},
            "discord": {"desc": 2000, "hashtags": 0},
        }
        lim = limits.get(self._name, {"desc": 500, "hashtags": 5})

        desc = base.description[:lim["desc"]]
        hashtags = base.hashtags[:lim["hashtags"]]

        return PlatformMetadata(
            title=base.title[:200],
            description=desc,
            hashtags=hashtags,
            tags=base.tags[:10],
        )

    def get_platform_name(self) -> str:
        return self._name

    def is_configured(self) -> bool:
        return self._legacy.is_configured() if hasattr(self._legacy, "is_configured") else False


# Singleton registry
_registry: Optional[PlatformRegistry] = None


def get_platform_registry() -> PlatformRegistry:
    global _registry
    if _registry is None:
        _registry = PlatformRegistry()
    return _registry
