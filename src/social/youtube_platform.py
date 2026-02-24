"""
YouTube as a ContentPlatform adapter.

Thin wrapper around the existing YouTubeUploader (src/youtube/uploader.py)
that implements the unified ContentPlatform interface.
"""

from datetime import datetime
from typing import Any, Dict, Optional

from loguru import logger

from src.social.platform_adapter import (
    ContentPlatform,
    ContentMetadata,
    PlatformMetadata,
    UploadResult,
    UploadStatus,
)


class YouTubePlatform(ContentPlatform):
    """YouTube platform adapter using the official YouTube Data API."""

    def __init__(self):
        self._uploader = None
        self._auth = None

    def _get_uploader(self):
        if self._uploader is None:
            try:
                from src.youtube.uploader import YouTubeUploader
                self._uploader = YouTubeUploader()
            except Exception as e:
                logger.error(f"Failed to initialize YouTubeUploader: {e}")
        return self._uploader

    async def upload_video(self, video_path: str, metadata: PlatformMetadata) -> UploadResult:
        """Upload video to YouTube using the official API."""
        uploader = self._get_uploader()
        if not uploader:
            return UploadResult(
                platform="youtube",
                status=UploadStatus.FAILED,
                error="YouTubeUploader not initialized",
            )

        try:
            # Map to existing uploader interface
            upload_kwargs = {
                "video_file": video_path,
                "title": metadata.title,
                "description": metadata.description,
                "tags": metadata.tags,
                "category": metadata.category or "27",  # 27 = Education
                "privacy": metadata.privacy,
            }

            # Add thumbnail if available
            thumbnail = metadata.platform_specific.get("thumbnail_path")
            if thumbnail:
                upload_kwargs["thumbnail_file"] = thumbnail

            result = uploader.upload_video(**upload_kwargs)

            if result and hasattr(result, "video_url"):
                return UploadResult(
                    platform="youtube",
                    status=UploadStatus.SUCCESS,
                    url=result.video_url,
                    post_id=result.video_id if hasattr(result, "video_id") else None,
                    uploaded_at=datetime.utcnow(),
                    metadata={"channel": metadata.platform_specific.get("channel_id", "")},
                )
            elif isinstance(result, dict):
                return UploadResult(
                    platform="youtube",
                    status=UploadStatus.SUCCESS,
                    url=result.get("url") or result.get("video_url"),
                    post_id=result.get("id") or result.get("video_id"),
                    uploaded_at=datetime.utcnow(),
                )
            else:
                return UploadResult(
                    platform="youtube",
                    status=UploadStatus.SUCCESS,
                    uploaded_at=datetime.utcnow(),
                    metadata={"raw_result": str(result)},
                )

        except Exception as e:
            logger.error(f"YouTube upload failed: {e}")
            return UploadResult(
                platform="youtube",
                status=UploadStatus.FAILED,
                error=str(e),
            )

    async def upload_image(self, image_path: str, metadata: PlatformMetadata) -> UploadResult:
        """YouTube doesn't support standalone image uploads."""
        return UploadResult(
            platform="youtube",
            status=UploadStatus.FAILED,
            error="YouTube does not support standalone image uploads. Use upload_video with thumbnail.",
        )

    async def post_text(self, content: str, metadata: PlatformMetadata) -> UploadResult:
        """YouTube doesn't support text-only posts via API."""
        return UploadResult(
            platform="youtube",
            status=UploadStatus.FAILED,
            error="YouTube does not support text-only posts via Data API.",
        )

    def adapt_metadata(self, base: ContentMetadata) -> PlatformMetadata:
        """Adapt metadata for YouTube's constraints."""
        # YouTube limits: title 100 chars, description 5000, tags 500 total chars
        title = base.title[:100]
        description = base.description[:5000]
        tags = base.tags[:30]  # Max ~30 tags practical
        hashtags = base.hashtags[:15]

        # Add hashtags to description (YouTube supports #tags in description)
        if hashtags and not any(f"#{h}" in description for h in hashtags[:3]):
            hashtag_str = " ".join(f"#{h}" for h in hashtags[:3])
            description = f"{description}\n\n{hashtag_str}"

        return PlatformMetadata(
            title=title,
            description=description,
            hashtags=hashtags,
            tags=tags,
            category=base.category or "27",
            privacy=base.privacy,
            language=base.language,
            scheduled_time=base.scheduled_time,
            platform_specific={
                "channel_id": base.channel_id,
                "thumbnail_path": base.thumbnail_path,
            },
        )

    def get_platform_name(self) -> str:
        return "youtube"

    def is_configured(self) -> bool:
        """Check if YouTube OAuth credentials exist."""
        try:
            from src.youtube.auth import YouTubeAuth
            auth = YouTubeAuth()
            return auth.get_authenticated_service() is not None
        except Exception:
            return False

    def get_platform_specs(self):
        """Return YouTube video specs."""
        try:
            from src.social.multi_platform import PLATFORM_SPECS, Platform
            return PLATFORM_SPECS.get(Platform.YOUTUBE_LONG)
        except ImportError:
            return None
