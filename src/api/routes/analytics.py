"""
Analytics and reporting endpoints.

Cross-platform performance data aggregation.
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query
from loguru import logger

from src.api.auth import verify_api_key
from src.api.models import AnalyticsReport

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


@router.get("/report", response_model=AnalyticsReport)
async def get_analytics_report(
    period: str = Query("last_7_days", regex="^(last_7_days|last_30_days|last_90_days|all_time)$"),
    channel: Optional[str] = Query(None),
    _: str = Depends(verify_api_key),
):
    """Get cross-platform analytics report.

    Aggregates performance data from YouTube Analytics API and
    browser-scraped metrics from TikTok/Instagram/Pinterest.
    """
    channels_data = {}
    platforms_data = {}
    top_performers = []
    recommendations = []

    # Try to load YouTube analytics
    try:
        from src.youtube.analytics_api import YouTubeAnalytics
        yt = YouTubeAnalytics()
        # Get channel analytics if available
        yt_data = yt.get_channel_stats() if hasattr(yt, "get_channel_stats") else {}
        if yt_data:
            platforms_data["youtube"] = yt_data
    except (ImportError, Exception) as e:
        logger.debug(f"YouTube analytics unavailable: {e}")
        platforms_data["youtube"] = {"status": "not_available", "reason": str(e)}

    # Try to load success tracker data
    try:
        from src.analytics.success_tracker import get_success_tracker
        tracker = get_success_tracker()
        dashboard = tracker.get_dashboard_data() if hasattr(tracker, "get_dashboard_data") else {}
        if dashboard:
            for ch_id, ch_data in dashboard.get("channels", {}).items():
                if channel and ch_id != channel:
                    continue
                channels_data[ch_id] = ch_data
    except (ImportError, Exception) as e:
        logger.debug(f"Success tracker unavailable: {e}")

    # Try to load cross-platform tracker (Phase 3)
    try:
        from src.analytics.cross_platform_tracker import CrossPlatformTracker
        tracker = CrossPlatformTracker()
        cross_data = await tracker.get_report(period=period, channel=channel)
        platforms_data.update(cross_data.get("platforms", {}))
        top_performers = cross_data.get("top_performers", [])
        recommendations = cross_data.get("recommendations", [])
    except ImportError:
        recommendations.append("Cross-platform analytics not yet available (Phase 3)")
    except Exception as e:
        logger.debug(f"Cross-platform tracker error: {e}")

    # Default recommendations if none generated
    if not recommendations:
        recommendations = [
            "Upload consistently to build algorithm trust",
            "Test different content types on TikTok for fastest feedback",
            "Use Pinterest for evergreen traffic (pins have 6+ month lifespan)",
        ]

    return AnalyticsReport(
        period=period,
        channels=channels_data,
        platforms=platforms_data,
        top_performers=top_performers,
        recommendations=recommendations,
    )


@router.get("/cost")
async def get_cost_report(_: str = Depends(verify_api_key)):
    """Get API cost and token usage report."""
    try:
        from src.utils.token_manager import TokenManager
        mgr = TokenManager()
        return mgr.get_report() if hasattr(mgr, "get_report") else {"status": "no_data"}
    except ImportError:
        return {"status": "token_manager_not_available"}
    except Exception as e:
        return {"status": "error", "error": str(e)}
