"""
Platform status and health check endpoints.
"""

import os
from typing import List

from fastapi import APIRouter, Depends
from loguru import logger

from src.api.auth import verify_api_key
from src.api.models import (
    PlatformName,
    PlatformStatus,
    PlatformStatusResponse,
    PlatformListResponse,
)

router = APIRouter(prefix="/api/v1/platforms", tags=["platforms"])


def _check_youtube_status() -> PlatformStatusResponse:
    """Check YouTube API connection."""
    try:
        from src.youtube.auth import YouTubeAuth
        auth = YouTubeAuth()
        service = auth.get_authenticated_service()
        return PlatformStatusResponse(
            platform=PlatformName.YOUTUBE,
            status=PlatformStatus.CONNECTED if service else PlatformStatus.ERROR,
            enabled=True,
            upload_type="api",
        )
    except Exception as e:
        return PlatformStatusResponse(
            platform=PlatformName.YOUTUBE,
            status=PlatformStatus.ERROR,
            enabled=True,
            upload_type="api",
            error=str(e),
        )


def _check_tiktok_status() -> PlatformStatusResponse:
    """Check TikTok browser session status."""
    session_dir = "config/browser_sessions/tiktok"
    has_session = os.path.exists(session_dir) and os.listdir(session_dir) if os.path.exists(session_dir) else False

    return PlatformStatusResponse(
        platform=PlatformName.TIKTOK,
        status=PlatformStatus.CONNECTED if has_session else PlatformStatus.NOT_CONFIGURED,
        enabled=True,
        upload_type="browser",
        error=None if has_session else "No browser session found. Run initial login first.",
    )


def _check_social_platform(platform: PlatformName, env_keys: List[str], upload_type: str = "api") -> PlatformStatusResponse:
    """Generic check for social platforms that use env-based API keys."""
    configured = any(os.environ.get(key) for key in env_keys)
    return PlatformStatusResponse(
        platform=platform,
        status=PlatformStatus.CONNECTED if configured else PlatformStatus.NOT_CONFIGURED,
        enabled=configured,
        upload_type=upload_type,
        error=None if configured else f"Missing env vars: {', '.join(env_keys)}",
    )


def _check_browser_platform(platform: PlatformName) -> PlatformStatusResponse:
    """Check for platforms that use browser automation."""
    session_dir = f"config/browser_sessions/{platform.value}"
    has_session = os.path.exists(session_dir) and os.listdir(session_dir) if os.path.exists(session_dir) else False

    return PlatformStatusResponse(
        platform=platform,
        status=PlatformStatus.CONNECTED if has_session else PlatformStatus.NOT_CONFIGURED,
        enabled=False,  # Disabled until Phase 2
        upload_type="browser",
        error=None if has_session else "Browser session not configured (Phase 2)",
    )


@router.get("/status", response_model=PlatformListResponse)
async def get_platform_status(_: str = Depends(verify_api_key)):
    """Get connection status of all configured platforms."""
    platforms = []

    # YouTube (official API)
    platforms.append(_check_youtube_status())

    # TikTok (browser automation)
    platforms.append(_check_tiktok_status())

    # Instagram (browser — Phase 2)
    platforms.append(_check_browser_platform(PlatformName.INSTAGRAM_REELS))

    # Pinterest (browser — Phase 2)
    platforms.append(_check_browser_platform(PlatformName.PINTEREST))

    # Twitter/X
    platforms.append(_check_social_platform(
        PlatformName.TWITTER,
        ["TWITTER_BEARER_TOKEN", "TWITTER_API_KEY"],
    ))

    # Reddit
    platforms.append(_check_social_platform(
        PlatformName.REDDIT,
        ["REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET"],
    ))

    # LinkedIn
    platforms.append(_check_social_platform(
        PlatformName.LINKEDIN,
        ["LINKEDIN_ACCESS_TOKEN"],
    ))

    # Facebook
    platforms.append(_check_social_platform(
        PlatformName.FACEBOOK,
        ["FACEBOOK_PAGE_ACCESS_TOKEN"],
    ))

    # Discord
    platforms.append(_check_social_platform(
        PlatformName.DISCORD,
        ["DISCORD_WEBHOOK_URL"],
        upload_type="webhook",
    ))

    # Medium (Phase 2)
    platforms.append(_check_social_platform(
        PlatformName.MEDIUM,
        ["MEDIUM_TOKEN"],
    ))

    # Threads (Phase 2)
    platforms.append(_check_social_platform(
        PlatformName.THREADS,
        ["THREADS_ACCESS_TOKEN"],
    ))

    total_enabled = sum(1 for p in platforms if p.enabled)
    total_connected = sum(1 for p in platforms if p.status == PlatformStatus.CONNECTED)

    return PlatformListResponse(
        platforms=platforms,
        total_enabled=total_enabled,
        total_connected=total_connected,
    )
