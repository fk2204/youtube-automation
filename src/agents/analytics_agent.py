"""
Analytics Agent - Video Performance Analysis

A token-efficient agent specialized in analyzing video performance,
identifying patterns, and suggesting content strategy improvements.

Usage:
    from src.agents.analytics_agent import AnalyticsAgent

    agent = AnalyticsAgent()

    # Analyze channel performance
    result = agent.analyze_channel("money_blueprints", period="30d")

    # Find patterns in successful videos
    result = agent.find_patterns("finance")

    # Generate content strategy
    result = agent.generate_strategy("psychology")
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from loguru import logger

from ..utils.token_manager import (
    get_token_manager,
    get_cost_optimizer,
    get_prompt_cache
)
from ..utils.best_practices import get_best_practices, get_niche_metrics


@dataclass
class PerformanceMetrics:
    """Performance metrics for a video or channel."""
    total_views: int = 0
    avg_views: float = 0.0
    avg_retention: float = 0.0
    avg_ctr: float = 0.0
    total_likes: int = 0
    total_comments: int = 0
    video_count: int = 0


@dataclass
class AnalyticsResult:
    """Result from analytics agent operations."""
    success: bool
    operation: str
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    patterns: Dict[str, Any] = field(default_factory=dict)
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    top_performers: List[Dict] = field(default_factory=list)
    underperformers: List[Dict] = field(default_factory=list)
    tokens_used: int = 0
    cost: float = 0.0
    provider: str = ""
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["metrics"] = asdict(self.metrics)
        return result

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Analytics Report",
            f"================",
            "",
            f"Videos Analyzed: {self.metrics.video_count}",
            f"Total Views: {self.metrics.total_views:,}",
            f"Avg Views: {self.metrics.avg_views:,.0f}",
            f"Avg Retention: {self.metrics.avg_retention:.1f}%",
            f"Avg CTR: {self.metrics.avg_ctr:.1f}%",
            ""
        ]

        if self.insights:
            lines.append("Key Insights:")
            for insight in self.insights[:5]:
                lines.append(f"  - {insight}")
            lines.append("")

        if self.recommendations:
            lines.append("Recommendations:")
            for rec in self.recommendations[:5]:
                lines.append(f"  - {rec}")

        return "\n".join(lines)

    def save(self, path: str = None):
        """Save result to JSON file."""
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"data/analytics_reports/report_{timestamp}.json"

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Analytics report saved to {path}")


class AnalyticsAgent:
    """
    Analytics Agent for video performance analysis.

    Token-efficient design:
    - Uses local database for historical data
    - Rule-based pattern detection (0 tokens)
    - AI insights only when specifically requested
    """

    # Channel to niche mapping
    CHANNEL_NICHE_MAP = {
        "money_blueprints": "finance",
        "mind_unlocked": "psychology",
        "untold_stories": "storytelling",
    }

    def __init__(self, provider: str = None, api_key: str = None):
        """
        Initialize the analytics agent.

        Args:
            provider: AI provider (for AI-powered insights)
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
        self.performance_db = Path("data/video_performance.db")
        self.upload_history = Path("data/upload_history.json")

        self._init_db()

        logger.info(f"AnalyticsAgent initialized with provider: {provider}")

    def _init_db(self):
        """Initialize performance database with optimized indexes."""
        self.performance_db.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.performance_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS video_performance (
                    video_id TEXT PRIMARY KEY,
                    channel TEXT,
                    title TEXT,
                    niche TEXT,
                    views INTEGER DEFAULT 0,
                    likes INTEGER DEFAULT 0,
                    comments INTEGER DEFAULT 0,
                    retention REAL DEFAULT 0,
                    ctr REAL DEFAULT 0,
                    uploaded_at DATETIME,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for performance optimization
            # These indexes speed up common queries by channel, niche, date, and views
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_channel ON video_performance(channel)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_niche ON video_performance(niche)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_uploaded_at ON video_performance(uploaded_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_views ON video_performance(views)
            """)

    def record_video(
        self,
        video_id: str,
        channel: str,
        title: str,
        niche: str = None,
        views: int = 0,
        likes: int = 0,
        comments: int = 0,
        retention: float = 0.0,
        ctr: float = 0.0
    ):
        """
        Record video performance data.

        Args:
            video_id: YouTube video ID
            channel: Channel ID
            title: Video title
            niche: Content niche
            views: View count
            likes: Like count
            comments: Comment count
            retention: Average retention %
            ctr: Click-through rate %
        """
        if niche is None:
            niche = self.CHANNEL_NICHE_MAP.get(channel, "default")

        with sqlite3.connect(self.performance_db) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO video_performance
                (video_id, channel, title, niche, views, likes, comments, retention, ctr, uploaded_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """, (video_id, channel, title, niche, views, likes, comments, retention, ctr))

        logger.info(f"Recorded performance for video: {video_id}")

    def analyze_channel(
        self,
        channel: str,
        period: str = "30d"
    ) -> AnalyticsResult:
        """
        Analyze channel performance.

        Token cost: ZERO (database queries only)

        Args:
            channel: Channel ID
            period: Analysis period (7d, 30d, 90d, all)

        Returns:
            AnalyticsResult with channel analytics
        """
        operation = f"analyze_channel_{channel}_{period}"
        logger.info(f"[AnalyticsAgent] Analyzing channel: {channel} ({period})")

        # Parse period
        days = {"7d": 7, "30d": 30, "90d": 90, "all": 365*10}.get(period, 30)
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        # Query database
        with sqlite3.connect(self.performance_db) as conn:
            # Get video stats
            rows = conn.execute("""
                SELECT title, views, likes, comments, retention, ctr, uploaded_at
                FROM video_performance
                WHERE channel = ? AND uploaded_at > ?
                ORDER BY views DESC
            """, (channel, cutoff)).fetchall()

        if not rows:
            # No data - use estimated metrics based on niche
            niche = self.CHANNEL_NICHE_MAP.get(channel, "default")
            niche_metrics = get_niche_metrics(niche)

            return AnalyticsResult(
                success=True,
                operation=operation,
                metrics=PerformanceMetrics(video_count=0),
                insights=["No video data found for this period"],
                recommendations=[
                    f"Start uploading {niche} content",
                    f"Target CPM: ${niche_metrics['cpm_range'][0]}-${niche_metrics['cpm_range'][1]}",
                    f"Optimal video length: {niche_metrics['optimal_video_length'][0]}-{niche_metrics['optimal_video_length'][1]} minutes"
                ],
                tokens_used=0,
                cost=0.0,
                provider="database"
            )

        # Calculate metrics
        total_views = sum(r[1] for r in rows)
        total_likes = sum(r[2] for r in rows)
        total_comments = sum(r[3] for r in rows)
        avg_retention = sum(r[4] for r in rows) / len(rows) if rows else 0
        avg_ctr = sum(r[5] for r in rows) / len(rows) if rows else 0

        metrics = PerformanceMetrics(
            total_views=total_views,
            avg_views=total_views / len(rows) if rows else 0,
            avg_retention=avg_retention,
            avg_ctr=avg_ctr,
            total_likes=total_likes,
            total_comments=total_comments,
            video_count=len(rows)
        )

        # Identify top and under performers
        top_performers = [
            {"title": r[0], "views": r[1], "retention": r[4]}
            for r in rows[:3]
        ]

        underperformers = [
            {"title": r[0], "views": r[1], "retention": r[4]}
            for r in sorted(rows, key=lambda x: x[1])[:3]
        ]

        # Generate insights
        insights = self._generate_insights(rows, channel)
        recommendations = self._generate_recommendations(metrics, channel)

        result = AnalyticsResult(
            success=True,
            operation=operation,
            metrics=metrics,
            insights=insights,
            recommendations=recommendations,
            top_performers=top_performers,
            underperformers=underperformers,
            tokens_used=0,
            cost=0.0,
            provider="database"
        )

        logger.success(f"[AnalyticsAgent] Channel analysis complete: {len(rows)} videos")
        return result

    def find_patterns(
        self,
        niche: str = None,
        channel: str = None
    ) -> AnalyticsResult:
        """
        Find patterns in successful videos.

        Token cost: ZERO (pattern matching only)

        Args:
            niche: Content niche to analyze
            channel: Specific channel to analyze

        Returns:
            AnalyticsResult with identified patterns
        """
        operation = f"find_patterns_{niche or channel}"
        logger.info(f"[AnalyticsAgent] Finding patterns for: {niche or channel}")

        # Query videos
        with sqlite3.connect(self.performance_db) as conn:
            if channel:
                rows = conn.execute("""
                    SELECT title, views, retention, ctr, niche
                    FROM video_performance
                    WHERE channel = ?
                    ORDER BY views DESC
                """, (channel,)).fetchall()
            elif niche:
                rows = conn.execute("""
                    SELECT title, views, retention, ctr, niche
                    FROM video_performance
                    WHERE niche = ?
                    ORDER BY views DESC
                """, (niche,)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT title, views, retention, ctr, niche
                    FROM video_performance
                    ORDER BY views DESC
                """).fetchall()

        patterns = {
            "title_patterns": [],
            "timing_patterns": [],
            "content_patterns": []
        }

        if not rows:
            # Use best practices as fallback patterns
            practices = get_best_practices(niche or "finance")
            patterns["title_patterns"] = practices.get("viral_title_patterns", [])[:5]
            patterns["content_patterns"] = ["Use proven viral templates"]

            return AnalyticsResult(
                success=True,
                operation=operation,
                patterns=patterns,
                insights=["Using best practices data (no historical data available)"],
                recommendations=["Start uploading videos to build analytics data"],
                tokens_used=0,
                cost=0.0,
                provider="best_practices"
            )

        # Analyze titles of top performers
        top_titles = [r[0] for r in rows[:10]]
        title_patterns = self._analyze_title_patterns(top_titles)
        patterns["title_patterns"] = title_patterns

        # Identify content patterns
        insights = []
        recommendations = []

        # Check for number patterns
        numbers_in_titles = sum(1 for t in top_titles if any(c.isdigit() for c in t))
        if numbers_in_titles > len(top_titles) * 0.5:
            patterns["content_patterns"].append("Numbers in titles perform well")
            insights.append(f"{numbers_in_titles}/{len(top_titles)} top videos have numbers in titles")

        # Check for question marks
        questions = sum(1 for t in top_titles if "?" in t)
        if questions > len(top_titles) * 0.3:
            patterns["content_patterns"].append("Question-based titles get engagement")
            insights.append(f"{questions}/{len(top_titles)} top videos use questions")

        result = AnalyticsResult(
            success=True,
            operation=operation,
            patterns=patterns,
            insights=insights,
            recommendations=recommendations,
            tokens_used=0,
            cost=0.0,
            provider="pattern_analysis"
        )

        return result

    def generate_strategy(
        self,
        niche: str,
        use_ai: bool = False
    ) -> AnalyticsResult:
        """
        Generate content strategy recommendations.

        Args:
            niche: Content niche
            use_ai: Use AI for strategic insights

        Returns:
            AnalyticsResult with strategy recommendations
        """
        operation = f"generate_strategy_{niche}"
        logger.info(f"[AnalyticsAgent] Generating strategy for: {niche}")

        # Get best practices
        practices = get_best_practices(niche)
        metrics = practices.get("metrics", {})

        recommendations = [
            f"Target CPM: ${metrics.get('cpm_range', (5, 10))[0]}-${metrics.get('cpm_range', (5, 10))[1]}",
            f"Optimal video length: {metrics.get('optimal_video_length', (8, 15))[0]}-{metrics.get('optimal_video_length', (8, 15))[1]} minutes",
            f"Posting frequency: {metrics.get('posting_frequency', '2-3 videos per week')}",
            f"Best posting days: {', '.join(metrics.get('best_days', ['Mon', 'Wed', 'Fri']))}",
        ]

        # Add hook formula suggestions
        hooks = practices.get("hook_formulas", [])[:3]
        if hooks:
            recommendations.append(f"Use hooks like: {hooks[0][:50]}...")

        # Add title pattern suggestions
        title_patterns = practices.get("viral_title_patterns", [])[:3]
        if title_patterns:
            recommendations.append(f"Title patterns that work: {title_patterns[0]}")

        insights = [
            f"{niche.title()} niche characteristics analyzed",
            f"Competitor best practices incorporated",
            "Recommendations based on top performer analysis"
        ]

        # AI enhancement if requested
        tokens_used = 0
        cost = 0.0
        provider = "best_practices"

        if use_ai:
            ai_result = self._ai_generate_strategy(niche, practices)
            if ai_result:
                recommendations.extend(ai_result.get("recommendations", []))
                insights.extend(ai_result.get("insights", []))
                tokens_used = 1000
                cost = self.tracker.record_usage(
                    provider=self.provider,
                    input_tokens=700,
                    output_tokens=300,
                    operation="analytics_ai_strategy"
                )
                provider = self.provider

        result = AnalyticsResult(
            success=True,
            operation=operation,
            insights=insights,
            recommendations=recommendations,
            tokens_used=tokens_used,
            cost=cost,
            provider=provider
        )

        return result

    def get_cost_analysis(self) -> AnalyticsResult:
        """
        Analyze token costs and ROI.

        Token cost: ZERO (reads from token tracker)

        Returns:
            AnalyticsResult with cost analysis
        """
        operation = "cost_analysis"
        logger.info(f"[AnalyticsAgent] Analyzing costs")

        # Get token usage data
        daily = self.tracker.get_daily_usage()
        weekly = self.tracker.get_weekly_usage()
        monthly = self.tracker.get_monthly_usage()
        by_provider = self.tracker.get_usage_by_provider()
        by_operation = self.tracker.get_usage_by_operation()

        insights = [
            f"Daily cost: ${daily['cost']:.4f}",
            f"Weekly cost: ${weekly['cost']:.4f}",
            f"Monthly cost: ${monthly['cost']:.4f}",
        ]

        # Find most expensive operations
        if by_operation:
            expensive_ops = sorted(by_operation, key=lambda x: x['cost'], reverse=True)[:3]
            for op in expensive_ops:
                insights.append(f"{op['operation']}: ${op['cost']:.4f} ({op['count']} calls)")

        recommendations = []

        # Provider optimization
        for p in by_provider:
            if p['provider'] in ['claude', 'openai'] and p['cost'] > 1.0:
                recommendations.append(f"Consider using Groq for {p['provider']} tasks - save ${p['cost']:.2f}")

        # Budget check
        budget = self.tracker.check_budget()
        if budget['warning']:
            recommendations.append("WARNING: Daily budget 80% used")
        if budget['exceeded']:
            recommendations.append("ALERT: Daily budget exceeded!")

        result = AnalyticsResult(
            success=True,
            operation=operation,
            insights=insights,
            recommendations=recommendations,
            tokens_used=0,
            cost=0.0,
            provider="token_tracker"
        )

        return result

    def _generate_insights(
        self,
        rows: List,
        channel: str
    ) -> List[str]:
        """Generate insights from video data."""
        insights = []

        if not rows:
            return ["No data to analyze"]

        # Calculate averages
        views = [r[1] for r in rows]
        avg_views = sum(views) / len(views)

        # Top performer insight
        top_video = rows[0]
        insights.append(f"Top performer: '{top_video[0][:40]}...' with {top_video[1]:,} views")

        # Retention insight
        retentions = [r[4] for r in rows if r[4] > 0]
        if retentions:
            avg_ret = sum(retentions) / len(retentions)
            if avg_ret > 50:
                insights.append(f"Good average retention: {avg_ret:.1f}%")
            elif avg_ret < 30:
                insights.append(f"Retention needs improvement: {avg_ret:.1f}%")

        # CTR insight
        ctrs = [r[5] for r in rows if r[5] > 0]
        if ctrs:
            avg_ctr = sum(ctrs) / len(ctrs)
            if avg_ctr > 8:
                insights.append(f"Strong CTR: {avg_ctr:.1f}% (above YouTube average)")
            elif avg_ctr < 4:
                insights.append(f"CTR below average: {avg_ctr:.1f}% - improve thumbnails/titles")

        return insights

    def _generate_recommendations(
        self,
        metrics: PerformanceMetrics,
        channel: str
    ) -> List[str]:
        """Generate recommendations based on metrics."""
        recommendations = []
        niche = self.CHANNEL_NICHE_MAP.get(channel, "default")
        niche_metrics = get_niche_metrics(niche)

        # Compare to niche benchmarks
        optimal_length = niche_metrics.get("optimal_video_length", (8, 15))
        recommendations.append(
            f"Target video length: {optimal_length[0]}-{optimal_length[1]} minutes for {niche}"
        )

        # Posting frequency
        recommendations.append(
            f"Recommended posting: {niche_metrics.get('posting_frequency', '2-3 videos per week')}"
        )

        # Best days
        best_days = niche_metrics.get("best_days", ["Mon", "Wed", "Fri"])
        recommendations.append(f"Best posting days: {', '.join(best_days)}")

        # Retention improvement
        if metrics.avg_retention < 40:
            recommendations.append("Focus on hook quality - first 30 seconds are critical")
            recommendations.append("Add pattern interrupts every 45-60 seconds")

        # CTR improvement
        if metrics.avg_ctr < 5:
            recommendations.append("A/B test thumbnails with faces and contrast")
            recommendations.append("Use numbers and power words in titles")

        return recommendations

    def _analyze_title_patterns(self, titles: List[str]) -> List[str]:
        """Analyze patterns in successful titles."""
        patterns = []

        # Check for common words
        word_counts = {}
        for title in titles:
            words = title.lower().split()
            for word in words:
                if len(word) > 3:
                    word_counts[word] = word_counts.get(word, 0) + 1

        # Top words
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        if top_words:
            patterns.append(f"Common words: {', '.join(w[0] for w in top_words)}")

        # Check for structural patterns
        has_numbers = sum(1 for t in titles if any(c.isdigit() for c in t))
        has_questions = sum(1 for t in titles if "?" in t)
        has_how = sum(1 for t in titles if t.lower().startswith("how"))

        if has_numbers > len(titles) // 2:
            patterns.append("Numbers in titles correlate with success")
        if has_questions > len(titles) // 3:
            patterns.append("Question-based titles perform well")
        if has_how > len(titles) // 3:
            patterns.append("'How to' format is effective")

        return patterns

    def _ai_generate_strategy(
        self,
        niche: str,
        practices: Dict
    ) -> Optional[Dict]:
        """Use AI to generate strategic insights."""
        try:
            from ..content.script_writer import get_provider
            ai = get_provider(self.provider, self.api_key)

            prompt = f"""Generate content strategy insights for a {niche} YouTube channel.

Best practices data:
- CPM range: ${practices['metrics']['cpm_range'][0]}-${practices['metrics']['cpm_range'][1]}
- Optimal length: {practices['metrics']['optimal_video_length']} minutes
- Posting frequency: {practices['metrics']['posting_frequency']}

Provide 3 strategic recommendations and 2 unique insights.

Respond with ONLY a JSON object:
{{
    "recommendations": ["rec1", "rec2", "rec3"],
    "insights": ["insight1", "insight2"]
}}"""

            response = ai.generate(prompt, max_tokens=400)

            # Parse response
            if "{" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                return json.loads(response[start:end])

        except Exception as e:
            logger.warning(f"AI strategy generation failed: {e}")

        return None

    def run(self, command: str = None, **kwargs) -> AnalyticsResult:
        """
        Main entry point for CLI usage.

        Args:
            command: Command string
            **kwargs: Parameters

        Returns:
            AnalyticsResult
        """
        channel = kwargs.get("channel")
        niche = kwargs.get("niche")
        period = kwargs.get("period", "30d")
        strategy = kwargs.get("strategy", False)
        cost = kwargs.get("cost", False)

        if cost:
            return self.get_cost_analysis()

        if strategy and niche:
            return self.generate_strategy(niche, use_ai=kwargs.get("ai", False))

        if channel:
            return self.analyze_channel(channel, period)

        if niche:
            return self.find_patterns(niche=niche)

        return AnalyticsResult(
            success=False,
            operation="unknown",
            error="Specify --channel, --niche, --strategy, or --cost"
        )


# CLI entry point
def main():
    """CLI entry point for analytics agent."""
    import sys

    if len(sys.argv) < 2:
        print("""
Analytics Agent - Video Performance Analysis

Usage:
    python -m src.agents.analytics_agent --channel money_blueprints
    python -m src.agents.analytics_agent --niche finance --strategy
    python -m src.agents.analytics_agent --cost

Options:
    --channel <id>      Analyze specific channel
    --niche <niche>     Analyze by niche
    --period <period>   Analysis period (7d, 30d, 90d)
    --strategy          Generate content strategy
    --cost              Analyze token costs
    --ai                Use AI for insights (uses tokens)
    --save              Save report
    --json              Output as JSON

Examples:
    python -m src.agents.analytics_agent --channel money_blueprints --period 7d
    python -m src.agents.analytics_agent --niche psychology --strategy
    python -m src.agents.analytics_agent --cost
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
        elif sys.argv[i] == "--niche" and i + 1 < len(sys.argv):
            kwargs["niche"] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--period" and i + 1 < len(sys.argv):
            kwargs["period"] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--strategy":
            kwargs["strategy"] = True
            i += 1
        elif sys.argv[i] == "--cost":
            kwargs["cost"] = True
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
    agent = AnalyticsAgent()
    result = agent.run(**kwargs)

    # Output
    if output_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("\n" + "=" * 60)
        print("ANALYTICS AGENT RESULT")
        print("=" * 60)
        print(result.summary())

    # Save if requested
    if save_report:
        result.save()


if __name__ == "__main__":
    main()
