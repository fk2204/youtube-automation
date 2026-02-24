"""
Meta Threads API integration.

Uses Meta's Threads API (launched 2024) for posting.
Requires THREADS_ACCESS_TOKEN from Meta Business Suite.
"""

import os
from datetime import datetime
from typing import Optional

import aiohttp
from loguru import logger

from src.social.platform_adapter import (
    ContentPlatform,
    ContentMetadata,
    PlatformMetadata,
    UploadResult,
    UploadStatus,
)

THREADS_API = "https://graph.threads.net/v1.0"


class ThreadsPlatform(ContentPlatform):
    """Meta Threads posting via API."""

    def __init__(self):
        self._token = os.environ.get("THREADS_ACCESS_TOKEN", "")
        self._user_id = os.environ.get("THREADS_USER_ID", "me")

    async def upload_video(self, video_path: str, metadata: PlatformMetadata) -> UploadResult:
        """Create a video thread post."""
        # Threads API requires video to be publicly accessible URL
        return UploadResult(
            platform="threads",
            status=UploadStatus.FAILED,
            error="Threads video upload requires public URL. Upload to CDN first.",
        )

    async def upload_image(self, image_path: str, metadata: PlatformMetadata) -> UploadResult:
        return UploadResult(platform="threads", status=UploadStatus.FAILED, error="Image upload requires public URL")

    async def post_text(self, content: str, metadata: PlatformMetadata) -> UploadResult:
        """Create a text thread."""
        if not self._token:
            return UploadResult(platform="threads", status=UploadStatus.AUTH_REQUIRED, error="THREADS_ACCESS_TOKEN not set")

        try:
            # Step 1: Create media container
            async with aiohttp.ClientSession() as session:
                create_resp = await session.post(
                    f"{THREADS_API}/{self._user_id}/threads",
                    params={
                        "media_type": "TEXT",
                        "text": content[:500],
                        "access_token": self._token,
                    },
                )
                if create_resp.status != 200:
                    return UploadResult(platform="threads", status=UploadStatus.FAILED, error=f"Create failed: {await create_resp.text()}")

                create_data = await create_resp.json()
                creation_id = create_data.get("id")

                # Step 2: Publish
                pub_resp = await session.post(
                    f"{THREADS_API}/{self._user_id}/threads_publish",
                    params={
                        "creation_id": creation_id,
                        "access_token": self._token,
                    },
                )
                if pub_resp.status == 200:
                    pub_data = await pub_resp.json()
                    return UploadResult(
                        platform="threads",
                        status=UploadStatus.SUCCESS,
                        post_id=pub_data.get("id"),
                        uploaded_at=datetime.utcnow(),
                    )
                else:
                    return UploadResult(platform="threads", status=UploadStatus.FAILED, error=f"Publish failed: {await pub_resp.text()}")

        except Exception as e:
            return UploadResult(platform="threads", status=UploadStatus.FAILED, error=str(e))

    def adapt_metadata(self, base: ContentMetadata) -> PlatformMetadata:
        return PlatformMetadata(
            title=base.title[:100],
            description=base.description[:500],
            hashtags=base.hashtags[:5],
        )

    def get_platform_name(self) -> str:
        return "threads"

    def is_configured(self) -> bool:
        return bool(self._token)
