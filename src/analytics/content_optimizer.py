"""
AI-driven content optimization feedback loop.

Analyzes performance data to automatically:
- Identify winning content patterns (hooks, titles, topics)
- Adjust posting times per platform
- Update viral hook weights
- Recommend content mix changes
- Generate weekly optimization reports

This is the learning brain that makes the system improve over time.
"""

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class OptimizationInsight:
    """A single optimization insight."""
    category: str  # content_type, platform, timing, hook_type, topic
    insight: str
    confidence: float  # 0.0 to 1.0
    action: str  # What to change
    expected_impact: str  # e.g. "+15% CTR"
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WeeklyReport:
    """Weekly optimization report."""
    period_start: datetime
    period_end: datetime
    total_posts: int = 0
    total_views: int = 0
    avg_engagement_rate: float = 0.0
    best_content: Dict[str, Any] = field(default_factory=dict)
    worst_content: Dict[str, Any] = field(default_factory=dict)
    insights: List[OptimizationInsight] = field(default_factory=list)
    applied_changes: List[str] = field(default_factory=list)


class ContentOptimizer:
    """AI-driven feedback loop for content optimization.

    Usage:
        optimizer = ContentOptimizer()

        # Run daily analysis
        insights = await optimizer.daily_analysis()

        # Run weekly report
        report = await optimizer.weekly_report()

        # Apply automatic optimizations
        changes = await optimizer.auto_optimize()
    """

    def __init__(self, db_path: str = "data/content_optimizer.db"):
        self._db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS optimization_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    insight TEXT NOT NULL,
                    action TEXT,
                    applied BOOLEAN DEFAULT FALSE,
                    result TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS content_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_id TEXT NOT NULL,
                    platform TEXT,
                    hook_type TEXT,
                    title_pattern TEXT,
                    topic_category TEXT,
                    niche TEXT,
                    posting_hour INTEGER,
                    posting_day TEXT,
                    views INTEGER DEFAULT 0,
                    engagement_rate REAL DEFAULT 0.0,
                    retention_pct REAL DEFAULT 0.0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

    async def daily_analysis(self) -> List[OptimizationInsight]:
        """Run daily performance analysis and generate insights."""
        insights = []

        # Analyze by platform
        platform_insights = await self._analyze_platforms()
        insights.extend(platform_insights)

        # Analyze by content type
        content_insights = await self._analyze_content_patterns()
        insights.extend(content_insights)

        # Analyze by timing
        timing_insights = await self._analyze_timing()
        insights.extend(timing_insights)

        # Persist insights
        self._save_insights(insights)

        logger.info(f"Daily analysis complete: {len(insights)} insights generated")
        return insights

    async def weekly_report(self) -> WeeklyReport:
        """Generate weekly optimization report."""
        now = datetime.utcnow()
        week_ago = now - timedelta(days=7)

        report = WeeklyReport(period_start=week_ago, period_end=now)

        try:
            from src.analytics.cross_platform_tracker import get_cross_platform_tracker
            tracker = get_cross_platform_tracker()
            data = await tracker.get_report(period="last_7_days")

            # Aggregate totals
            for platform, metrics in data.get("platforms", {}).items():
                report.total_views += metrics.get("total_views", 0)
                report.total_posts += metrics.get("posts", 0)

            if report.total_posts > 0:
                report.avg_engagement_rate = sum(
                    m.get("avg_engagement", 0)
                    for m in data.get("platforms", {}).values()
                ) / len(data.get("platforms", {}))

            # Best/worst performers
            top = data.get("top_performers", [])
            if top:
                report.best_content = top[0]
                report.worst_content = top[-1] if len(top) > 1 else {}

            report.insights = await self.daily_analysis()

        except Exception as e:
            logger.warning(f"Weekly report data collection failed: {e}")

        return report

    async def auto_optimize(self) -> List[str]:
        """Apply automatic optimizations based on accumulated insights."""
        changes = []

        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                # Get unapplied high-confidence insights
                rows = conn.execute(
                    "SELECT * FROM optimization_log WHERE applied = FALSE ORDER BY created_at DESC LIMIT 10"
                ).fetchall()

                for row in rows:
                    action = row[3]  # action column
                    if action:
                        applied = await self._apply_action(action)
                        if applied:
                            conn.execute(
                                "UPDATE optimization_log SET applied = TRUE, result = ? WHERE id = ?",
                                (applied, row[0]),
                            )
                            changes.append(applied)

        except Exception as e:
            logger.error(f"Auto-optimization failed: {e}")

        return changes

    async def _analyze_platforms(self) -> List[OptimizationInsight]:
        """Analyze performance differences across platforms."""
        insights = []

        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute("""
                    SELECT platform, AVG(views) as avg_views, AVG(engagement_rate) as avg_eng,
                           COUNT(*) as posts
                    FROM content_scores
                    WHERE created_at >= datetime('now', '-7 days')
                    GROUP BY platform
                    HAVING posts >= 3
                    ORDER BY avg_views DESC
                """).fetchall()

                if len(rows) >= 2:
                    best = rows[0]
                    worst = rows[-1]

                    if best["avg_views"] > worst["avg_views"] * 2:
                        insights.append(OptimizationInsight(
                            category="platform",
                            insight=f"{best['platform']} outperforms {worst['platform']} by {best['avg_views']/max(worst['avg_views'],1):.1f}x in views",
                            confidence=min(best["posts"] / 10, 1.0),
                            action=f"increase_posting_frequency:{best['platform']}",
                            expected_impact=f"+{int((best['avg_views']/max(worst['avg_views'],1)-1)*100)}% total views",
                            evidence={"best_avg": best["avg_views"], "worst_avg": worst["avg_views"]},
                        ))

        except Exception as e:
            logger.debug(f"Platform analysis failed: {e}")

        return insights

    async def _analyze_content_patterns(self) -> List[OptimizationInsight]:
        """Analyze which content patterns perform best."""
        insights = []

        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.row_factory = sqlite3.Row

                # By hook type
                rows = conn.execute("""
                    SELECT hook_type, AVG(views) as avg_views, AVG(engagement_rate) as avg_eng,
                           COUNT(*) as count
                    FROM content_scores
                    WHERE hook_type IS NOT NULL AND created_at >= datetime('now', '-30 days')
                    GROUP BY hook_type
                    HAVING count >= 3
                    ORDER BY avg_eng DESC
                """).fetchall()

                if rows:
                    best_hook = rows[0]
                    insights.append(OptimizationInsight(
                        category="hook_type",
                        insight=f"Hook type '{best_hook['hook_type']}' has highest engagement ({best_hook['avg_eng']:.1f}%)",
                        confidence=min(best_hook["count"] / 10, 1.0),
                        action=f"prioritize_hook:{best_hook['hook_type']}",
                        expected_impact="+10-20% engagement",
                    ))

        except Exception as e:
            logger.debug(f"Content pattern analysis failed: {e}")

        return insights

    async def _analyze_timing(self) -> List[OptimizationInsight]:
        """Analyze optimal posting times."""
        insights = []

        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.row_factory = sqlite3.Row

                rows = conn.execute("""
                    SELECT posting_hour, posting_day, AVG(views) as avg_views,
                           COUNT(*) as count
                    FROM content_scores
                    WHERE posting_hour IS NOT NULL AND created_at >= datetime('now', '-30 days')
                    GROUP BY posting_hour, posting_day
                    HAVING count >= 2
                    ORDER BY avg_views DESC
                    LIMIT 5
                """).fetchall()

                if rows:
                    best = rows[0]
                    insights.append(OptimizationInsight(
                        category="timing",
                        insight=f"Best posting time: {best['posting_day']} at {best['posting_hour']}:00 UTC ({best['avg_views']:.0f} avg views)",
                        confidence=min(best["count"] / 5, 1.0),
                        action=f"adjust_schedule:{best['posting_day']}:{best['posting_hour']}",
                        expected_impact="+5-15% views from timing optimization",
                    ))

        except Exception as e:
            logger.debug(f"Timing analysis failed: {e}")

        return insights

    async def _apply_action(self, action: str) -> Optional[str]:
        """Apply an optimization action."""
        parts = action.split(":")
        action_type = parts[0]

        if action_type == "increase_posting_frequency":
            platform = parts[1] if len(parts) > 1 else ""
            logger.info(f"Recommendation: Increase posting frequency on {platform}")
            return f"Recommended: increase {platform} posting frequency"

        elif action_type == "prioritize_hook":
            hook_type = parts[1] if len(parts) > 1 else ""
            logger.info(f"Recommendation: Prioritize {hook_type} hooks")
            return f"Recommended: prioritize {hook_type} hook formula"

        elif action_type == "adjust_schedule":
            day = parts[1] if len(parts) > 1 else ""
            hour = parts[2] if len(parts) > 2 else ""
            logger.info(f"Recommendation: Post on {day} at {hour}:00 UTC")
            return f"Recommended: schedule posts for {day} {hour}:00 UTC"

        return None

    def _save_insights(self, insights: List[OptimizationInsight]) -> None:
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                for insight in insights:
                    conn.execute(
                        "INSERT INTO optimization_log (category, insight, action) VALUES (?, ?, ?)",
                        (insight.category, insight.insight, insight.action),
                    )
        except Exception as e:
            logger.error(f"Failed to save insights: {e}")

    def record_content_score(
        self, content_id: str, platform: str, views: int = 0,
        engagement_rate: float = 0.0, hook_type: str = "",
        niche: str = "", posting_hour: int = 0, posting_day: str = "",
    ) -> None:
        """Record content performance for optimization learning."""
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute(
                    """INSERT INTO content_scores
                    (content_id, platform, hook_type, niche, posting_hour, posting_day, views, engagement_rate)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (content_id, platform, hook_type, niche, posting_hour, posting_day, views, engagement_rate),
                )
        except Exception as e:
            logger.error(f"Failed to record content score: {e}")


# Singleton
_optimizer: Optional[ContentOptimizer] = None


def get_content_optimizer() -> ContentOptimizer:
    global _optimizer
    if _optimizer is None:
        _optimizer = ContentOptimizer()
    return _optimizer
