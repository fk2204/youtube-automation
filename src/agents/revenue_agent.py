"""
Revenue Agent - Revenue Tracking and Optimization

A token-efficient agent specialized in tracking AdSense revenue,
analyzing CPM by niche, and identifying high-revenue topics.

Usage:
    from src.agents.revenue_agent import RevenueAgent

    agent = RevenueAgent()

    # Track revenue for a channel
    result = agent.track_revenue("money_blueprints", period="30d")

    # Get revenue per video
    result = agent.analyze_video_revenue("VIDEO_ID")

    # Find high-revenue topics
    result = agent.identify_high_revenue_topics("finance")
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from loguru import logger

from ..utils.token_manager import (
    get_token_manager,
    get_cost_optimizer,
    get_prompt_cache
)
from ..utils.best_practices import get_niche_metrics


# CPM ranges by niche ($ per 1000 views)
NICHE_CPM_RANGES = {
    "finance": (10.0, 22.0),
    "psychology": (3.0, 6.0),
    "storytelling": (4.0, 15.0),
    "default": (2.0, 8.0),
}


@dataclass
class VideoRevenue:
    """Revenue data for a single video."""
    video_id: str
    title: str
    views: int
    estimated_cpm: float
    estimated_revenue: float
    niche: str
    upload_date: Optional[str] = None
    rpm: float = 0.0  # Revenue per mille (actual)


@dataclass
class RevenueResult:
    """Result from revenue agent operations."""
    success: bool
    operation: str
    total_revenue: float = 0.0
    revenue_by_video: Dict[str, float] = field(default_factory=dict)
    revenue_by_channel: Dict[str, float] = field(default_factory=dict)
    avg_cpm: float = 0.0
    best_topics: List[Dict[str, Any]] = field(default_factory=list)
    revenue_trend: str = "stable"  # up, down, stable
    videos: List[VideoRevenue] = field(default_factory=list)
    projections: Dict[str, float] = field(default_factory=dict)
    tokens_used: int = 0
    cost: float = 0.0
    provider: str = ""
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["videos"] = [asdict(v) for v in self.videos]
        return result

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Revenue Report",
            f"==============",
            "",
            f"Total Revenue: ${self.total_revenue:.2f}",
            f"Average CPM: ${self.avg_cpm:.2f}",
            f"Revenue Trend: {self.revenue_trend.upper()}",
            ""
        ]

        if self.revenue_by_channel:
            lines.append("Revenue by Channel:")
            for channel, revenue in sorted(
                self.revenue_by_channel.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                lines.append(f"  {channel}: ${revenue:.2f}")
            lines.append("")

        if self.best_topics:
            lines.append("Best Revenue Topics:")
            for i, topic in enumerate(self.best_topics[:5], 1):
                lines.append(f"  {i}. {topic.get('topic', 'Unknown')} - ${topic.get('avg_revenue', 0):.2f}/video")
            lines.append("")

        if self.projections:
            lines.append("Projections:")
            for period, amount in self.projections.items():
                lines.append(f"  {period}: ${amount:.2f}")

        return "\n".join(lines)

    def save(self, path: str = None):
        """Save result to JSON file."""
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"data/revenue/report_{timestamp}.json"

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Revenue report saved to {path}")


class RevenueAgent:
    """
    Revenue Agent for tracking and optimizing YouTube revenue.

    Token-efficient design:
    - Uses local database for revenue tracking
    - Estimates revenue based on CPM * views/1000
    - No AI tokens needed for basic tracking
    - AI only used for optimization recommendations
    """

    # Channel to niche mapping
    CHANNEL_NICHE_MAP = {
        "money_blueprints": "finance",
        "mind_unlocked": "psychology",
        "untold_stories": "storytelling",
    }

    def __init__(self, provider: str = None, api_key: str = None):
        """
        Initialize the revenue agent.

        Args:
            provider: AI provider (for optimization recommendations)
            api_key: API key for cloud providers
        """
        self.tracker = get_token_manager()
        self.optimizer = get_cost_optimizer()
        self.cache = get_prompt_cache()

        # Select provider for AI analysis
        if provider is None:
            provider = self.optimizer.select_provider("idea_generation")

        self.provider = provider
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")

        # Database paths
        self.revenue_db = Path("data/revenue/revenue.db")
        self.performance_db = Path("data/video_performance.db")

        self._init_db()

        logger.info(f"RevenueAgent initialized with provider: {provider}")

    def _init_db(self):
        """Initialize revenue database."""
        self.revenue_db.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.revenue_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS video_revenue (
                    video_id TEXT PRIMARY KEY,
                    channel TEXT,
                    title TEXT,
                    niche TEXT,
                    views INTEGER DEFAULT 0,
                    estimated_cpm REAL DEFAULT 0,
                    estimated_revenue REAL DEFAULT 0,
                    actual_revenue REAL,
                    rpm REAL DEFAULT 0,
                    upload_date TEXT,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_revenue (
                    date TEXT,
                    channel TEXT,
                    revenue REAL DEFAULT 0,
                    views INTEGER DEFAULT 0,
                    avg_cpm REAL DEFAULT 0,
                    PRIMARY KEY (date, channel)
                )
            """)

            # Indexes for performance
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_revenue_channel
                ON video_revenue(channel)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_revenue_niche
                ON video_revenue(niche)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_revenue_date
                ON video_revenue(upload_date)
            """)

    def _get_estimated_cpm(self, niche: str, video_data: Dict = None) -> float:
        """
        Get estimated CPM for a niche.

        Uses niche-specific ranges with variance based on video characteristics.

        Args:
            niche: Content niche
            video_data: Optional video data for more accurate estimation

        Returns:
            Estimated CPM value
        """
        cpm_range = NICHE_CPM_RANGES.get(niche, NICHE_CPM_RANGES["default"])

        # Base CPM is midpoint of range
        base_cpm = (cpm_range[0] + cpm_range[1]) / 2

        # Adjust based on video characteristics if available
        if video_data:
            # Higher retention = higher CPM (advertisers pay more for engaged viewers)
            retention = video_data.get("retention", 40)
            if retention > 50:
                base_cpm *= 1.1
            elif retention < 30:
                base_cpm *= 0.9

            # Longer videos can have more ad placements
            duration_minutes = video_data.get("duration", 8)
            if duration_minutes > 10:
                base_cpm *= 1.15  # Mid-roll ads
            elif duration_minutes < 3:
                base_cpm *= 0.85  # Less ad inventory

        return min(base_cpm, cpm_range[1])  # Cap at max

    def record_video_revenue(
        self,
        video_id: str,
        channel: str,
        title: str,
        views: int,
        niche: str = None,
        actual_revenue: float = None,
        upload_date: str = None
    ):
        """
        Record video revenue data.

        Args:
            video_id: YouTube video ID
            channel: Channel ID
            title: Video title
            views: View count
            niche: Content niche (auto-detected if not provided)
            actual_revenue: Actual AdSense revenue if available
            upload_date: Video upload date
        """
        if niche is None:
            niche = self.CHANNEL_NICHE_MAP.get(channel, "default")

        estimated_cpm = self._get_estimated_cpm(niche)
        estimated_revenue = (views / 1000) * estimated_cpm

        # Calculate RPM if actual revenue provided
        rpm = (actual_revenue / views * 1000) if actual_revenue and views > 0 else 0

        with sqlite3.connect(self.revenue_db) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO video_revenue
                (video_id, channel, title, niche, views, estimated_cpm,
                 estimated_revenue, actual_revenue, rpm, upload_date, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (video_id, channel, title, niche, views, estimated_cpm,
                  estimated_revenue, actual_revenue, rpm, upload_date))

        logger.info(f"Recorded revenue for video: {video_id} - ${estimated_revenue:.2f}")

    def track_revenue(
        self,
        channel: str = None,
        period: str = "30d"
    ) -> RevenueResult:
        """
        Track revenue for a channel or all channels.

        Token cost: ZERO (database queries only)

        Args:
            channel: Channel ID (all channels if None)
            period: Analysis period (7d, 30d, 90d, all)

        Returns:
            RevenueResult with revenue data
        """
        operation = f"track_revenue_{channel or 'all'}_{period}"
        logger.info(f"[RevenueAgent] Tracking revenue for: {channel or 'all channels'} ({period})")

        # Parse period
        days = {"7d": 7, "30d": 30, "90d": 90, "all": 365 * 10}.get(period, 30)
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        try:
            with sqlite3.connect(self.revenue_db) as conn:
                # Build query
                if channel:
                    query = """
                        SELECT video_id, channel, title, niche, views,
                               estimated_cpm, estimated_revenue, actual_revenue, rpm, upload_date
                        FROM video_revenue
                        WHERE channel = ? AND (upload_date >= ? OR upload_date IS NULL)
                        ORDER BY estimated_revenue DESC
                    """
                    rows = conn.execute(query, (channel, cutoff)).fetchall()
                else:
                    query = """
                        SELECT video_id, channel, title, niche, views,
                               estimated_cpm, estimated_revenue, actual_revenue, rpm, upload_date
                        FROM video_revenue
                        WHERE upload_date >= ? OR upload_date IS NULL
                        ORDER BY estimated_revenue DESC
                    """
                    rows = conn.execute(query, (cutoff,)).fetchall()

            if not rows:
                # Try to load from performance database
                rows = self._load_from_performance_db(channel, cutoff)

            if not rows:
                return RevenueResult(
                    success=True,
                    operation=operation,
                    total_revenue=0.0,
                    avg_cpm=0.0,
                    provider="database",
                    error="No revenue data found. Record video data first."
                )

            # Process results
            videos = []
            revenue_by_video = {}
            revenue_by_channel = {}
            total_revenue = 0.0
            total_views = 0
            cpm_sum = 0.0

            for row in rows:
                video_id, ch, title, niche, views, cpm, est_rev, actual_rev, rpm, upload_date = row

                # Use actual revenue if available, otherwise estimated
                revenue = actual_rev if actual_rev else est_rev

                videos.append(VideoRevenue(
                    video_id=video_id,
                    title=title or "Unknown",
                    views=views or 0,
                    estimated_cpm=cpm or 0,
                    estimated_revenue=revenue or 0,
                    niche=niche or "default",
                    upload_date=upload_date,
                    rpm=rpm or 0
                ))

                revenue_by_video[video_id] = revenue or 0
                revenue_by_channel[ch] = revenue_by_channel.get(ch, 0) + (revenue or 0)
                total_revenue += revenue or 0
                total_views += views or 0
                cpm_sum += cpm or 0

            avg_cpm = cpm_sum / len(rows) if rows else 0

            # Calculate revenue trend
            revenue_trend = self._calculate_trend(channel, period)

            # Identify best topics
            best_topics = self._identify_best_topics(rows)

            # Generate projections
            projections = self._generate_projections(total_revenue, len(rows), days)

            result = RevenueResult(
                success=True,
                operation=operation,
                total_revenue=total_revenue,
                revenue_by_video=revenue_by_video,
                revenue_by_channel=revenue_by_channel,
                avg_cpm=avg_cpm,
                best_topics=best_topics,
                revenue_trend=revenue_trend,
                videos=videos,
                projections=projections,
                tokens_used=0,
                cost=0.0,
                provider="database"
            )

            logger.success(f"[RevenueAgent] Revenue tracked: ${total_revenue:.2f} total")
            return result

        except Exception as e:
            logger.error(f"[RevenueAgent] Error tracking revenue: {e}")
            return RevenueResult(
                success=False,
                operation=operation,
                error=str(e),
                provider="database"
            )

    def _load_from_performance_db(
        self,
        channel: str = None,
        cutoff: str = None
    ) -> List[tuple]:
        """Load and estimate revenue from performance database."""
        if not self.performance_db.exists():
            return []

        try:
            with sqlite3.connect(self.performance_db) as conn:
                if channel:
                    query = """
                        SELECT video_id, channel, title, niche, views,
                               0 as cpm, 0 as est_rev, NULL as actual_rev, 0 as rpm, uploaded_at
                        FROM video_performance
                        WHERE channel = ? AND (uploaded_at >= ? OR uploaded_at IS NULL)
                        ORDER BY views DESC
                    """
                    rows = conn.execute(query, (channel, cutoff)).fetchall()
                else:
                    query = """
                        SELECT video_id, channel, title, niche, views,
                               0 as cpm, 0 as est_rev, NULL as actual_rev, 0 as rpm, uploaded_at
                        FROM video_performance
                        WHERE uploaded_at >= ? OR uploaded_at IS NULL
                        ORDER BY views DESC
                    """
                    rows = conn.execute(query, (cutoff,)).fetchall()

            # Estimate revenue for each video
            result = []
            for row in rows:
                video_id, ch, title, niche, views, _, _, _, _, upload_date = row
                niche = niche or self.CHANNEL_NICHE_MAP.get(ch, "default")
                cpm = self._get_estimated_cpm(niche)
                est_rev = (views / 1000) * cpm if views else 0
                result.append((video_id, ch, title, niche, views, cpm, est_rev, None, 0, upload_date))

            return result

        except Exception as e:
            logger.warning(f"Could not load from performance DB: {e}")
            return []

    def _calculate_trend(self, channel: str = None, period: str = "30d") -> str:
        """Calculate revenue trend (up, down, stable)."""
        days = {"7d": 7, "30d": 30, "90d": 90}.get(period, 30)

        # Compare current period to previous period
        current_end = datetime.now()
        current_start = current_end - timedelta(days=days)
        previous_end = current_start
        previous_start = previous_end - timedelta(days=days)

        try:
            with sqlite3.connect(self.revenue_db) as conn:
                def get_period_revenue(start: datetime, end: datetime) -> float:
                    if channel:
                        query = """
                            SELECT COALESCE(SUM(estimated_revenue), 0)
                            FROM video_revenue
                            WHERE channel = ? AND upload_date BETWEEN ? AND ?
                        """
                        row = conn.execute(query, (
                            channel,
                            start.strftime("%Y-%m-%d"),
                            end.strftime("%Y-%m-%d")
                        )).fetchone()
                    else:
                        query = """
                            SELECT COALESCE(SUM(estimated_revenue), 0)
                            FROM video_revenue
                            WHERE upload_date BETWEEN ? AND ?
                        """
                        row = conn.execute(query, (
                            start.strftime("%Y-%m-%d"),
                            end.strftime("%Y-%m-%d")
                        )).fetchone()
                    return row[0] if row else 0

                current_rev = get_period_revenue(current_start, current_end)
                previous_rev = get_period_revenue(previous_start, previous_end)

                if previous_rev == 0:
                    return "stable"

                change_pct = (current_rev - previous_rev) / previous_rev * 100

                if change_pct > 10:
                    return "up"
                elif change_pct < -10:
                    return "down"
                else:
                    return "stable"

        except Exception:
            return "stable"

    def _identify_best_topics(self, rows: List[tuple]) -> List[Dict[str, Any]]:
        """Identify highest revenue topics from video data."""
        # Group by title keywords
        topic_revenue = {}

        for row in rows:
            _, _, title, niche, views, _, est_rev, actual_rev, _, _ = row
            revenue = actual_rev if actual_rev else est_rev

            # Extract key topics from title (simple keyword extraction)
            if title:
                words = title.lower().split()
                for word in words:
                    # Skip common words
                    if len(word) > 4 and word not in [
                        "video", "about", "first", "every", "never", "always",
                        "what", "when", "where", "which", "these", "those"
                    ]:
                        if word not in topic_revenue:
                            topic_revenue[word] = {"count": 0, "total_revenue": 0, "niche": niche}
                        topic_revenue[word]["count"] += 1
                        topic_revenue[word]["total_revenue"] += revenue or 0

        # Calculate average revenue per topic
        best_topics = []
        for topic, data in topic_revenue.items():
            if data["count"] >= 2:  # At least 2 videos with this topic
                avg_revenue = data["total_revenue"] / data["count"]
                best_topics.append({
                    "topic": topic,
                    "video_count": data["count"],
                    "total_revenue": data["total_revenue"],
                    "avg_revenue": avg_revenue,
                    "niche": data["niche"]
                })

        # Sort by average revenue
        best_topics.sort(key=lambda x: x["avg_revenue"], reverse=True)
        return best_topics[:10]

    def _generate_projections(
        self,
        total_revenue: float,
        video_count: int,
        days: int
    ) -> Dict[str, float]:
        """Generate revenue projections."""
        if days == 0 or video_count == 0:
            return {}

        daily_revenue = total_revenue / days
        videos_per_day = video_count / days

        return {
            "weekly": daily_revenue * 7,
            "monthly": daily_revenue * 30,
            "quarterly": daily_revenue * 90,
            "yearly": daily_revenue * 365,
            "per_video_avg": total_revenue / video_count if video_count > 0 else 0
        }

    def analyze_cpm_by_niche(self) -> RevenueResult:
        """
        Analyze CPM performance by niche.

        Token cost: ZERO (database queries only)

        Returns:
            RevenueResult with CPM analysis by niche
        """
        operation = "analyze_cpm_by_niche"
        logger.info("[RevenueAgent] Analyzing CPM by niche")

        try:
            with sqlite3.connect(self.revenue_db) as conn:
                rows = conn.execute("""
                    SELECT niche,
                           AVG(estimated_cpm) as avg_cpm,
                           AVG(rpm) as avg_rpm,
                           SUM(estimated_revenue) as total_revenue,
                           COUNT(*) as video_count
                    FROM video_revenue
                    WHERE niche IS NOT NULL
                    GROUP BY niche
                    ORDER BY avg_cpm DESC
                """).fetchall()

            if not rows:
                # Return theoretical CPM ranges
                best_topics = []
                for niche, cpm_range in NICHE_CPM_RANGES.items():
                    best_topics.append({
                        "topic": niche,
                        "avg_revenue": (cpm_range[0] + cpm_range[1]) / 2,
                        "cpm_range": f"${cpm_range[0]}-${cpm_range[1]}",
                        "video_count": 0
                    })

                return RevenueResult(
                    success=True,
                    operation=operation,
                    best_topics=best_topics,
                    provider="theoretical",
                    error="No actual data - showing theoretical CPM ranges"
                )

            best_topics = []
            total_revenue = 0
            total_cpm = 0

            for row in rows:
                niche, avg_cpm, avg_rpm, niche_revenue, video_count = row
                cpm_range = NICHE_CPM_RANGES.get(niche, NICHE_CPM_RANGES["default"])

                best_topics.append({
                    "topic": niche,
                    "avg_cpm": avg_cpm,
                    "avg_rpm": avg_rpm or 0,
                    "total_revenue": niche_revenue,
                    "video_count": video_count,
                    "cpm_range": f"${cpm_range[0]}-${cpm_range[1]}"
                })

                total_revenue += niche_revenue or 0
                total_cpm += avg_cpm or 0

            result = RevenueResult(
                success=True,
                operation=operation,
                total_revenue=total_revenue,
                avg_cpm=total_cpm / len(rows) if rows else 0,
                best_topics=best_topics,
                tokens_used=0,
                cost=0.0,
                provider="database"
            )

            logger.success(f"[RevenueAgent] CPM analysis complete: {len(rows)} niches")
            return result

        except Exception as e:
            logger.error(f"[RevenueAgent] Error analyzing CPM: {e}")
            return RevenueResult(
                success=False,
                operation=operation,
                error=str(e),
                provider="database"
            )

    def optimize_for_rpm(
        self,
        channel: str = None,
        use_ai: bool = False
    ) -> RevenueResult:
        """
        Generate recommendations to optimize RPM.

        Args:
            channel: Channel to optimize
            use_ai: Use AI for recommendations (costs tokens)

        Returns:
            RevenueResult with optimization recommendations
        """
        operation = f"optimize_rpm_{channel or 'all'}"
        logger.info(f"[RevenueAgent] Generating RPM optimization for: {channel or 'all'}")

        # Get current revenue data
        current = self.track_revenue(channel, "30d")

        recommendations = []

        # Rule-based recommendations
        niche = self.CHANNEL_NICHE_MAP.get(channel, "default") if channel else "default"
        niche_metrics = get_niche_metrics(niche)
        expected_cpm = NICHE_CPM_RANGES.get(niche, NICHE_CPM_RANGES["default"])

        if current.avg_cpm < expected_cpm[0]:
            recommendations.append(
                f"CPM (${current.avg_cpm:.2f}) is below expected range "
                f"(${expected_cpm[0]}-${expected_cpm[1]}). Consider improving content quality."
            )

        # Video length optimization
        optimal_length = niche_metrics.get("optimal_video_length", (8, 15))
        recommendations.append(
            f"Target video length: {optimal_length[0]}-{optimal_length[1]} minutes "
            f"for optimal mid-roll ad placement (8+ min required)."
        )

        # Topic recommendations based on best performers
        if current.best_topics:
            top_topic = current.best_topics[0]
            recommendations.append(
                f"Top revenue topic: '{top_topic.get('topic')}' with "
                f"${top_topic.get('avg_revenue', 0):.2f} average per video. Create more content like this."
            )

        # High-CPM niche suggestions
        if niche != "finance":
            finance_cpm = NICHE_CPM_RANGES["finance"]
            recommendations.append(
                f"Finance content has highest CPM (${finance_cpm[0]}-${finance_cpm[1]}). "
                f"Consider adding finance-related topics to increase revenue."
            )

        # Posting frequency
        if current.projections.get("per_video_avg", 0) > 0:
            rec_videos = 10.0 / current.projections["per_video_avg"]
            recommendations.append(
                f"At ${current.projections['per_video_avg']:.2f}/video, "
                f"you need ~{rec_videos:.0f} videos to reach $10/day."
            )

        # AI-powered recommendations
        if use_ai:
            ai_recs = self._get_ai_recommendations(current, niche)
            if ai_recs:
                recommendations.extend(ai_recs)
                current.tokens_used = 800
                current.cost = self.tracker.record_usage(
                    provider=self.provider,
                    input_tokens=500,
                    output_tokens=300,
                    operation="revenue_ai_optimize"
                )
                current.provider = self.provider

        result = RevenueResult(
            success=True,
            operation=operation,
            total_revenue=current.total_revenue,
            avg_cpm=current.avg_cpm,
            best_topics=[{"topic": r, "type": "recommendation"} for r in recommendations],
            projections=current.projections,
            tokens_used=current.tokens_used,
            cost=current.cost,
            provider=current.provider or "rule_based"
        )

        return result

    def _get_ai_recommendations(
        self,
        current: RevenueResult,
        niche: str
    ) -> List[str]:
        """Get AI-powered revenue optimization recommendations."""
        try:
            from ..content.script_writer import get_provider
            ai = get_provider(self.provider, self.api_key)

            prompt = f"""Analyze this YouTube channel revenue data and provide 3 specific recommendations:

Channel niche: {niche}
Total revenue (30d): ${current.total_revenue:.2f}
Average CPM: ${current.avg_cpm:.2f}
Video count: {len(current.videos)}
Revenue trend: {current.revenue_trend}

Top topics by revenue:
{json.dumps(current.best_topics[:3], indent=2) if current.best_topics else 'No data'}

Provide 3 actionable recommendations to increase revenue. Be specific and practical.

Respond with ONLY a JSON array of strings:
["recommendation 1", "recommendation 2", "recommendation 3"]"""

            response = ai.generate(prompt, max_tokens=300)

            # Parse response
            if "[" in response:
                start = response.find("[")
                end = response.rfind("]") + 1
                return json.loads(response[start:end])

        except Exception as e:
            logger.warning(f"AI recommendations failed: {e}")

        return []

    def run(self, command: str = None, **kwargs) -> RevenueResult:
        """
        Main entry point for CLI usage.

        Args:
            command: Command string
            **kwargs: Parameters

        Returns:
            RevenueResult
        """
        channel = kwargs.get("channel")
        period = kwargs.get("period", "30d")
        optimize = kwargs.get("optimize", False)
        cpm_analysis = kwargs.get("cpm_analysis", False)

        if cpm_analysis:
            return self.analyze_cpm_by_niche()

        if optimize:
            return self.optimize_for_rpm(channel, use_ai=kwargs.get("ai", False))

        return self.track_revenue(channel, period)


# CLI entry point
def main():
    """CLI entry point for revenue agent."""
    import sys

    if len(sys.argv) < 2:
        print("""
Revenue Agent - Revenue Tracking and Optimization

Usage:
    python -m src.agents.revenue_agent --channel money_blueprints
    python -m src.agents.revenue_agent --cpm-analysis
    python -m src.agents.revenue_agent --optimize --channel money_blueprints
    python -m src.agents.revenue_agent --optimize --ai

Options:
    --channel <id>      Analyze specific channel
    --period <period>   Analysis period (7d, 30d, 90d)
    --cpm-analysis      Analyze CPM by niche
    --optimize          Generate optimization recommendations
    --ai                Use AI for recommendations (uses tokens)
    --save              Save report
    --json              Output as JSON

Examples:
    python -m src.agents.revenue_agent --channel money_blueprints --period 30d
    python -m src.agents.revenue_agent --cpm-analysis
    python -m src.agents.revenue_agent --optimize --ai --save
        """)
        return

    # Parse arguments
    kwargs = {}
    i = 1
    output_json = False
    save_report = False

    while i < len(sys.argv):
        if sys.argv[i] == "--channel" and i + 1 < len(sys.argv):
            kwargs["channel"] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--period" and i + 1 < len(sys.argv):
            kwargs["period"] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--cpm-analysis":
            kwargs["cpm_analysis"] = True
            i += 1
        elif sys.argv[i] == "--optimize":
            kwargs["optimize"] = True
            i += 1
        elif sys.argv[i] == "--ai":
            kwargs["ai"] = True
            i += 1
        elif sys.argv[i] == "--save":
            save_report = True
            i += 1
        elif sys.argv[i] == "--json":
            output_json = True
            i += 1
        else:
            i += 1

    # Run agent
    agent = RevenueAgent()
    result = agent.run(**kwargs)

    # Output
    if output_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("\n" + "=" * 60)
        print("REVENUE AGENT RESULT")
        print("=" * 60)
        print(result.summary())

    # Save if requested
    if save_report:
        result.save()


if __name__ == "__main__":
    main()
