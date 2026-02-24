"""
Cross-platform performance tracker.

Unified analytics aggregation across YouTube, TikTok, Instagram,
Pinterest, Twitter, Reddit, and all other platforms.

Data sources:
- YouTube: analytics_api.py (official API)
- TikTok/Instagram/Pinterest: Browser scraping via Playwright
- Twitter/Reddit/LinkedIn: API-based metrics
- Internal: SQLite tracking of all uploads
"""

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


DB_PATH = Path("data/cross_platform_analytics.db")


@dataclass
class PlatformMetric:
    """Performance metrics for a single post on a platform."""
    platform: str
    post_id: str
    content_id: str
    url: Optional[str] = None
    views: int = 0
    likes: int = 0
    comments: int = 0
    shares: int = 0
    saves: int = 0
    click_through_rate: float = 0.0
    avg_watch_time_seconds: float = 0.0
    retention_pct: float = 0.0
    impressions: int = 0
    engagement_rate: float = 0.0
    scraped_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ContentPerformance:
    """Aggregated performance across all platforms for one content piece."""
    content_id: str
    topic: str
    niche: str
    total_views: int = 0
    total_likes: int = 0
    total_comments: int = 0
    total_shares: int = 0
    best_platform: str = ""
    best_platform_views: int = 0
    worst_platform: str = ""
    worst_platform_views: int = 0
    platforms: Dict[str, PlatformMetric] = field(default_factory=dict)


class CrossPlatformTracker:
    """Tracks and aggregates performance across all platforms."""

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = db_path or DB_PATH
        self._init_db()

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS platform_posts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_id TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    post_id TEXT,
                    url TEXT,
                    title TEXT,
                    niche TEXT,
                    channel TEXT,
                    posted_at TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS platform_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    post_row_id INTEGER REFERENCES platform_posts(id),
                    views INTEGER DEFAULT 0,
                    likes INTEGER DEFAULT 0,
                    comments INTEGER DEFAULT 0,
                    shares INTEGER DEFAULT 0,
                    saves INTEGER DEFAULT 0,
                    impressions INTEGER DEFAULT 0,
                    click_through_rate REAL DEFAULT 0.0,
                    avg_watch_time REAL DEFAULT 0.0,
                    retention_pct REAL DEFAULT 0.0,
                    engagement_rate REAL DEFAULT 0.0,
                    scraped_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS ix_posts_content ON platform_posts(content_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS ix_posts_platform ON platform_posts(platform)")
            conn.execute("CREATE INDEX IF NOT EXISTS ix_metrics_post ON platform_metrics(post_row_id)")

    def record_post(
        self, content_id: str, platform: str, post_id: str = "",
        url: str = "", title: str = "", niche: str = "", channel: str = "",
    ) -> int:
        """Record a new post across platforms. Returns row ID."""
        with sqlite3.connect(str(self._db_path)) as conn:
            cursor = conn.execute(
                "INSERT INTO platform_posts (content_id, platform, post_id, url, title, niche, channel, posted_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (content_id, platform, post_id, url, title, niche, channel, datetime.utcnow().isoformat()),
            )
            return cursor.lastrowid

    def record_metrics(self, post_row_id: int, metrics: PlatformMetric) -> None:
        """Record metrics for a post."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                """INSERT INTO platform_metrics
                (post_row_id, views, likes, comments, shares, saves, impressions,
                 click_through_rate, avg_watch_time, retention_pct, engagement_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (post_row_id, metrics.views, metrics.likes, metrics.comments,
                 metrics.shares, metrics.saves, metrics.impressions,
                 metrics.click_through_rate, metrics.avg_watch_time_seconds,
                 metrics.retention_pct, metrics.engagement_rate),
            )

    async def get_report(
        self, period: str = "last_7_days", channel: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate cross-platform analytics report."""
        days = {"last_7_days": 7, "last_30_days": 30, "last_90_days": 90, "all_time": 3650}
        since = datetime.utcnow() - timedelta(days=days.get(period, 7))

        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row

            # Posts per platform
            query = """
                SELECT p.platform, COUNT(*) as posts,
                       COALESCE(SUM(m.views), 0) as total_views,
                       COALESCE(SUM(m.likes), 0) as total_likes,
                       COALESCE(AVG(m.engagement_rate), 0) as avg_engagement
                FROM platform_posts p
                LEFT JOIN platform_metrics m ON m.post_row_id = p.id
                WHERE p.posted_at >= ?
            """
            params = [since.isoformat()]
            if channel:
                query += " AND p.channel = ?"
                params.append(channel)
            query += " GROUP BY p.platform"

            rows = conn.execute(query, params).fetchall()

            platforms_data = {}
            for row in rows:
                platforms_data[row["platform"]] = {
                    "posts": row["posts"],
                    "total_views": row["total_views"],
                    "total_likes": row["total_likes"],
                    "avg_engagement": round(row["avg_engagement"], 2),
                }

            # Top performers
            top_query = """
                SELECT p.title, p.platform, p.url, m.views, m.likes, m.engagement_rate
                FROM platform_posts p
                JOIN platform_metrics m ON m.post_row_id = p.id
                WHERE p.posted_at >= ?
                ORDER BY m.views DESC LIMIT 10
            """
            top_rows = conn.execute(top_query, [since.isoformat()]).fetchall()
            top_performers = [
                {
                    "title": r["title"],
                    "platform": r["platform"],
                    "url": r["url"],
                    "views": r["views"],
                    "likes": r["likes"],
                    "engagement_rate": r["engagement_rate"],
                }
                for r in top_rows
            ]

        recommendations = self._generate_recommendations(platforms_data, top_performers)

        return {
            "period": period,
            "platforms": platforms_data,
            "top_performers": top_performers,
            "recommendations": recommendations,
        }

    def _generate_recommendations(
        self, platforms: Dict, top_performers: List
    ) -> List[str]:
        """Generate actionable recommendations from performance data."""
        recs = []

        if not platforms:
            recs.append("No data yet. Start posting to see analytics.")
            return recs

        # Find best/worst performing platform
        sorted_platforms = sorted(
            platforms.items(),
            key=lambda x: x[1].get("total_views", 0),
            reverse=True,
        )

        if len(sorted_platforms) >= 2:
            best = sorted_platforms[0]
            worst = sorted_platforms[-1]
            recs.append(
                f"Best performing platform: {best[0]} ({best[1]['total_views']} views). "
                f"Consider increasing posting frequency here."
            )
            if worst[1].get("total_views", 0) == 0:
                recs.append(f"{worst[0]} has zero views. Check if content format is optimized for this platform.")

        # Engagement insights
        for name, data in platforms.items():
            if data.get("avg_engagement", 0) > 5:
                recs.append(f"High engagement on {name} ({data['avg_engagement']}%). This audience is highly active.")

        return recs


# Singleton
_tracker: Optional[CrossPlatformTracker] = None


def get_cross_platform_tracker() -> CrossPlatformTracker:
    global _tracker
    if _tracker is None:
        _tracker = CrossPlatformTracker()
    return _tracker
