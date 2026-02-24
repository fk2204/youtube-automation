"""
Medium article publishing via API.

Uses Medium's official REST API for article creation.
Requires MEDIUM_TOKEN from https://medium.com/me/settings/security
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

MEDIUM_API = "https://api.medium.com/v1"


class MediumPlatform(ContentPlatform):
    """Medium article publishing via REST API."""

    def __init__(self):
        self._token = os.environ.get("MEDIUM_TOKEN", "")
        self._user_id: Optional[str] = None

    async def _get_user_id(self) -> Optional[str]:
        if self._user_id:
            return self._user_id
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{MEDIUM_API}/me",
                    headers={"Authorization": f"Bearer {self._token}"},
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self._user_id = data["data"]["id"]
                        return self._user_id
        except Exception as e:
            logger.error(f"Failed to get Medium user: {e}")
        return None

    async def upload_video(self, video_path: str, metadata: PlatformMetadata) -> UploadResult:
        return UploadResult(platform="medium", status=UploadStatus.FAILED, error="Medium does not support video uploads")

    async def upload_image(self, image_path: str, metadata: PlatformMetadata) -> UploadResult:
        return UploadResult(platform="medium", status=UploadStatus.FAILED, error="Use post_text with embedded images")

    async def post_text(self, content: str, metadata: PlatformMetadata) -> UploadResult:
        """Publish an article to Medium."""
        user_id = await self._get_user_id()
        if not user_id:
            return UploadResult(platform="medium", status=UploadStatus.AUTH_REQUIRED, error="Could not authenticate with Medium")

        payload = {
            "title": metadata.title,
            "contentFormat": "markdown",
            "content": content,
            "tags": metadata.tags[:5],  # Medium max 5 tags
            "publishStatus": "public",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{MEDIUM_API}/users/{user_id}/posts",
                    headers={
                        "Authorization": f"Bearer {self._token}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                ) as resp:
                    if resp.status in (200, 201):
                        data = await resp.json()
                        post_data = data.get("data", {})
                        return UploadResult(
                            platform="medium",
                            status=UploadStatus.SUCCESS,
                            url=post_data.get("url"),
                            post_id=post_data.get("id"),
                            uploaded_at=datetime.utcnow(),
                        )
                    else:
                        error_text = await resp.text()
                        return UploadResult(platform="medium", status=UploadStatus.FAILED, error=f"HTTP {resp.status}: {error_text}")

        except Exception as e:
            return UploadResult(platform="medium", status=UploadStatus.FAILED, error=str(e))

    def adapt_metadata(self, base: ContentMetadata) -> PlatformMetadata:
        return PlatformMetadata(
            title=base.title[:100],
            description=base.description[:500],
            tags=[base.niche] + base.tags[:4],
        )

    def get_platform_name(self) -> str:
        return "medium"

    def is_configured(self) -> bool:
        return bool(self._token)
