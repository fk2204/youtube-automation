"""
Content Strategy Agent - Long-term Content Planning

A token-efficient agent specialized in analyzing content portfolio,
generating content calendars, and balancing risk/reward in topic selection.

Usage:
    from src.agents.content_strategy_agent import ContentStrategyAgent

    agent = ContentStrategyAgent()

    # Generate content calendar
    result = agent.generate_calendar("money_blueprints", weeks=4)

    # Analyze content portfolio
    result = agent.analyze_portfolio("money_blueprints")

    # Get recommended topics
    result = agent.recommend_topics("finance", count=10)
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from loguru import logger
import random

from ..utils.token_manager import (
    get_token_manager,
    get_cost_optimizer,
    get_prompt_cache
)
from ..utils.best_practices import get_best_practices, get_niche_metrics


# Content type classifications
CONTENT_TYPES = {
    "evergreen": {
        "description": "Timeless content that remains relevant",
        "keywords": ["how to", "guide", "explained", "basics", "fundamentals", "tips", "mistakes"],
        "weight": 0.6  # 60% of content should be evergreen
    },
    "trending": {
        "description": "Timely content on current topics",
        "keywords": ["2026", "new", "latest", "update", "news", "breaking", "just announced"],
        "weight": 0.25  # 25% trending
    },
    "viral": {
        "description": "High-risk, high-reward content",
        "keywords": ["secret", "truth", "exposed", "shocking", "nobody", "never", "dark side"],
        "weight": 0.15  # 15% viral attempts
    }
}


@dataclass
class ContentPlan:
    """Single content plan item for calendar."""
    date: str
    day_of_week: str
    topic: str
    content_type: str  # evergreen, trending, viral
    niche: str
    priority: int = 1  # 1 = high, 2 = medium, 3 = low
    notes: str = ""
    estimated_cpm: float = 0.0


@dataclass
class PortfolioBalance:
    """Content portfolio balance metrics."""
    evergreen_percent: float = 0.0
    trending_percent: float = 0.0
    viral_percent: float = 0.0
    target_evergreen: float = 60.0
    target_trending: float = 25.0
    target_viral: float = 15.0
    is_balanced: bool = False
    adjustments_needed: List[str] = field(default_factory=list)


@dataclass
class GrowthProjection:
    """Growth projection data."""
    period: str
    current_videos: int
    projected_videos: int
    current_views: int
    projected_views: int
    confidence: float = 0.0
    assumptions: List[str] = field(default_factory=list)


@dataclass
class StrategyResult:
    """Result from content strategy agent operations."""
    success: bool
    operation: str
    content_calendar: List[ContentPlan] = field(default_factory=list)
    recommended_topics: List[Dict[str, Any]] = field(default_factory=list)
    portfolio_balance: Optional[PortfolioBalance] = None
    growth_projections: Dict[str, GrowthProjection] = field(default_factory=dict)
    strategy_score: int = 0  # 0-100
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    tokens_used: int = 0
    cost: float = 0.0
    provider: str = ""
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["content_calendar"] = [asdict(p) for p in self.content_calendar]
        if self.portfolio_balance:
            result["portfolio_balance"] = asdict(self.portfolio_balance)
        result["growth_projections"] = {
            k: asdict(v) for k, v in self.growth_projections.items()
        }
        return result

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Content Strategy Report",
            f"=======================",
            "",
            f"Strategy Score: {self.strategy_score}/100",
            ""
        ]

        if self.portfolio_balance:
            lines.append("Portfolio Balance:")
            lines.append(f"  Evergreen: {self.portfolio_balance.evergreen_percent:.0f}% "
                        f"(target: {self.portfolio_balance.target_evergreen:.0f}%)")
            lines.append(f"  Trending: {self.portfolio_balance.trending_percent:.0f}% "
                        f"(target: {self.portfolio_balance.target_trending:.0f}%)")
            lines.append(f"  Viral: {self.portfolio_balance.viral_percent:.0f}% "
                        f"(target: {self.portfolio_balance.target_viral:.0f}%)")
            lines.append(f"  Balanced: {'Yes' if self.portfolio_balance.is_balanced else 'No'}")
            lines.append("")

        if self.content_calendar:
            lines.append(f"Content Calendar ({len(self.content_calendar)} items):")
            for plan in self.content_calendar[:7]:
                lines.append(f"  {plan.date} ({plan.day_of_week}): {plan.topic[:50]}...")
                lines.append(f"    Type: {plan.content_type} | Priority: {plan.priority}")
            lines.append("")

        if self.recommended_topics:
            lines.append("Recommended Topics:")
            for i, topic in enumerate(self.recommended_topics[:5], 1):
                lines.append(f"  {i}. {topic.get('topic', 'Unknown')}")
                lines.append(f"     Type: {topic.get('content_type', 'unknown')} | "
                           f"Risk: {topic.get('risk_level', 'medium')}")
            lines.append("")

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
            path = f"data/strategy/report_{timestamp}.json"

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Strategy report saved to {path}")


class ContentStrategyAgent:
    """
    Content Strategy Agent for long-term content planning.

    Token-efficient design:
    - Uses local database for portfolio analysis
    - Rule-based content classification
    - AI only for advanced strategy recommendations
    - Caches strategy results (7-day TTL)
    """

    # Channel to niche mapping
    CHANNEL_NICHE_MAP = {
        "money_blueprints": "finance",
        "mind_unlocked": "psychology",
        "untold_stories": "storytelling",
    }

    # Posting schedule by channel
    POSTING_SCHEDULES = {
        "money_blueprints": [0, 2, 4],  # Mon, Wed, Fri
        "mind_unlocked": [1, 3, 5],      # Tue, Thu, Sat
        "untold_stories": [0, 1, 2, 3, 4, 5, 6],  # Daily
    }

    DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    def __init__(self, provider: str = None, api_key: str = None):
        """
        Initialize the content strategy agent.

        Args:
            provider: AI provider (for advanced strategy)
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
        self.strategy_db = Path("data/strategy/strategy.db")
        self.performance_db = Path("data/video_performance.db")

        self._init_db()

        logger.info(f"ContentStrategyAgent initialized with provider: {provider}")

    def _init_db(self):
        """Initialize strategy database."""
        self.strategy_db.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.strategy_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS content_plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel TEXT,
                    date TEXT,
                    topic TEXT,
                    content_type TEXT,
                    priority INTEGER DEFAULT 1,
                    status TEXT DEFAULT 'planned',
                    video_id TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_goals (
                    channel TEXT PRIMARY KEY,
                    weekly_video_target INTEGER DEFAULT 3,
                    evergreen_ratio REAL DEFAULT 0.6,
                    trending_ratio REAL DEFAULT 0.25,
                    viral_ratio REAL DEFAULT 0.15,
                    monthly_view_target INTEGER DEFAULT 10000,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_plans_channel
                ON content_plans(channel)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_plans_date
                ON content_plans(date)
            """)

    def _classify_content_type(self, title: str, topic: str = None) -> str:
        """
        Classify content as evergreen, trending, or viral.

        Args:
            title: Video title
            topic: Topic if different from title

        Returns:
            Content type classification
        """
        text = (title + " " + (topic or "")).lower()

        # Check for trending indicators
        for keyword in CONTENT_TYPES["trending"]["keywords"]:
            if keyword in text:
                return "trending"

        # Check for viral indicators
        for keyword in CONTENT_TYPES["viral"]["keywords"]:
            if keyword in text:
                return "viral"

        # Default to evergreen
        return "evergreen"

    def analyze_portfolio(self, channel: str, period: str = "90d") -> StrategyResult:
        """
        Analyze content portfolio balance.

        Token cost: ZERO (database queries only)

        Args:
            channel: Channel ID
            period: Analysis period (30d, 90d, all)

        Returns:
            StrategyResult with portfolio analysis
        """
        operation = f"analyze_portfolio_{channel}_{period}"
        logger.info(f"[ContentStrategyAgent] Analyzing portfolio for: {channel}")

        days = {"30d": 30, "90d": 90, "all": 365 * 10}.get(period, 90)
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        niche = self.CHANNEL_NICHE_MAP.get(channel, "default")

        try:
            # Get video data from performance database
            videos = []
            if self.performance_db.exists():
                with sqlite3.connect(self.performance_db) as conn:
                    rows = conn.execute("""
                        SELECT title, views, retention, ctr, uploaded_at
                        FROM video_performance
                        WHERE channel = ? AND (uploaded_at >= ? OR uploaded_at IS NULL)
                        ORDER BY uploaded_at DESC
                    """, (channel, cutoff)).fetchall()

                    for row in rows:
                        title, views, retention, ctr, upload_date = row
                        content_type = self._classify_content_type(title)
                        videos.append({
                            "title": title,
                            "views": views or 0,
                            "retention": retention or 0,
                            "ctr": ctr or 0,
                            "content_type": content_type,
                            "upload_date": upload_date
                        })

            if not videos:
                # No data - provide theoretical analysis
                return StrategyResult(
                    success=True,
                    operation=operation,
                    portfolio_balance=PortfolioBalance(
                        evergreen_percent=0,
                        trending_percent=0,
                        viral_percent=0,
                        is_balanced=False,
                        adjustments_needed=["No video data available. Start uploading content."]
                    ),
                    insights=["No video data found for analysis"],
                    recommendations=[
                        f"Start with evergreen content (60% of uploads)",
                        f"Add trending content on current events (25%)",
                        f"Experiment with viral hooks (15%)"
                    ],
                    provider="database"
                )

            # Calculate portfolio balance
            total = len(videos)
            evergreen = sum(1 for v in videos if v["content_type"] == "evergreen")
            trending = sum(1 for v in videos if v["content_type"] == "trending")
            viral = sum(1 for v in videos if v["content_type"] == "viral")

            evergreen_pct = (evergreen / total * 100) if total > 0 else 0
            trending_pct = (trending / total * 100) if total > 0 else 0
            viral_pct = (viral / total * 100) if total > 0 else 0

            adjustments = []
            if evergreen_pct < 50:
                adjustments.append(f"Increase evergreen content: {evergreen_pct:.0f}% -> 60%")
            if trending_pct > 35:
                adjustments.append(f"Reduce trending content: {trending_pct:.0f}% -> 25%")
            if viral_pct > 25:
                adjustments.append(f"Reduce viral attempts: {viral_pct:.0f}% -> 15%")

            is_balanced = (
                45 <= evergreen_pct <= 75 and
                10 <= trending_pct <= 40 and
                viral_pct <= 30
            )

            portfolio = PortfolioBalance(
                evergreen_percent=evergreen_pct,
                trending_percent=trending_pct,
                viral_percent=viral_pct,
                is_balanced=is_balanced,
                adjustments_needed=adjustments
            )

            # Generate insights
            insights = []

            # Analyze performance by content type
            evergreen_videos = [v for v in videos if v["content_type"] == "evergreen"]
            trending_videos = [v for v in videos if v["content_type"] == "trending"]
            viral_videos = [v for v in videos if v["content_type"] == "viral"]

            if evergreen_videos:
                avg_views = sum(v["views"] for v in evergreen_videos) / len(evergreen_videos)
                insights.append(f"Evergreen content averages {avg_views:,.0f} views")

            if trending_videos:
                avg_views = sum(v["views"] for v in trending_videos) / len(trending_videos)
                insights.append(f"Trending content averages {avg_views:,.0f} views")

            if viral_videos:
                avg_views = sum(v["views"] for v in viral_videos) / len(viral_videos)
                insights.append(f"Viral attempts average {avg_views:,.0f} views")

            # Calculate strategy score
            strategy_score = self._calculate_strategy_score(portfolio, videos)

            # Generate recommendations
            recommendations = self._generate_portfolio_recommendations(portfolio, videos, niche)

            result = StrategyResult(
                success=True,
                operation=operation,
                portfolio_balance=portfolio,
                strategy_score=strategy_score,
                insights=insights,
                recommendations=recommendations,
                tokens_used=0,
                cost=0.0,
                provider="database"
            )

            logger.success(f"[ContentStrategyAgent] Portfolio analysis complete: {strategy_score}/100")
            return result

        except Exception as e:
            logger.error(f"[ContentStrategyAgent] Error analyzing portfolio: {e}")
            return StrategyResult(
                success=False,
                operation=operation,
                error=str(e),
                provider="database"
            )

    def generate_calendar(
        self,
        channel: str,
        weeks: int = 4,
        use_ai: bool = False
    ) -> StrategyResult:
        """
        Generate content calendar suggestions.

        Args:
            channel: Channel ID
            weeks: Number of weeks to plan
            use_ai: Use AI for topic generation (costs tokens)

        Returns:
            StrategyResult with content calendar
        """
        operation = f"generate_calendar_{channel}_{weeks}w"
        logger.info(f"[ContentStrategyAgent] Generating {weeks}-week calendar for: {channel}")

        niche = self.CHANNEL_NICHE_MAP.get(channel, "default")
        posting_days = self.POSTING_SCHEDULES.get(channel, [0, 2, 4])
        practices = get_best_practices(niche)
        niche_metrics = get_niche_metrics(niche)

        # Get topic templates from practices
        viral_patterns = practices.get("viral_title_patterns", [])
        hook_formulas = practices.get("hook_formulas", [])

        calendar = []
        start_date = datetime.now()

        # Track content type distribution
        type_counts = {"evergreen": 0, "trending": 0, "viral": 0}
        total_slots = 0

        for week in range(weeks):
            for day_offset in range(7):
                current_date = start_date + timedelta(days=week * 7 + day_offset)
                day_of_week = current_date.weekday()

                if day_of_week not in posting_days:
                    continue

                total_slots += 1

                # Determine content type based on target ratio
                # Use weighted random selection to match targets
                evergreen_weight = max(0, 0.6 - type_counts["evergreen"] / max(total_slots, 1))
                trending_weight = max(0, 0.25 - type_counts["trending"] / max(total_slots, 1))
                viral_weight = max(0, 0.15 - type_counts["viral"] / max(total_slots, 1))

                weights = [evergreen_weight, trending_weight, viral_weight]
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w / total_weight for w in weights]
                else:
                    weights = [0.6, 0.25, 0.15]

                content_type = random.choices(
                    ["evergreen", "trending", "viral"],
                    weights=weights
                )[0]

                type_counts[content_type] += 1

                # Generate topic based on content type
                topic = self._generate_topic(niche, content_type, viral_patterns)

                # Set priority based on content type
                priority = 1 if content_type == "evergreen" else (2 if content_type == "trending" else 3)

                calendar.append(ContentPlan(
                    date=current_date.strftime("%Y-%m-%d"),
                    day_of_week=self.DAY_NAMES[day_of_week],
                    topic=topic,
                    content_type=content_type,
                    niche=niche,
                    priority=priority,
                    estimated_cpm=niche_metrics.get("cpm_range", (5, 10))[0]
                ))

        # Use AI to enhance topics if requested
        if use_ai and calendar:
            calendar = self._enhance_calendar_with_ai(calendar, niche)

        # Calculate portfolio balance for the calendar
        total = len(calendar)
        portfolio = PortfolioBalance(
            evergreen_percent=(type_counts["evergreen"] / total * 100) if total > 0 else 0,
            trending_percent=(type_counts["trending"] / total * 100) if total > 0 else 0,
            viral_percent=(type_counts["viral"] / total * 100) if total > 0 else 0,
            is_balanced=True
        )

        # Generate growth projections
        projections = self._generate_growth_projections(channel, len(calendar), weeks)

        result = StrategyResult(
            success=True,
            operation=operation,
            content_calendar=calendar,
            portfolio_balance=portfolio,
            growth_projections=projections,
            insights=[
                f"Generated {len(calendar)} content items for {weeks} weeks",
                f"Content mix: {type_counts['evergreen']} evergreen, "
                f"{type_counts['trending']} trending, {type_counts['viral']} viral"
            ],
            recommendations=[
                "Adjust topics based on current trends before production",
                "Monitor performance and shift ratios as needed"
            ],
            tokens_used=0 if not use_ai else 1000,
            cost=0.0,
            provider="rule_based" if not use_ai else self.provider
        )

        logger.success(f"[ContentStrategyAgent] Calendar generated: {len(calendar)} items")
        return result

    def _generate_topic(
        self,
        niche: str,
        content_type: str,
        viral_patterns: List[str]
    ) -> str:
        """Generate a topic based on niche and content type."""
        templates = {
            "finance": {
                "evergreen": [
                    "How to Build Wealth Starting from Zero",
                    "5 Money Mistakes That Keep You Poor",
                    "The Complete Guide to Passive Income",
                    "Investment Basics Explained Simply",
                    "How to Budget Like a Millionaire",
                    "The Psychology of Saving Money",
                    "Compound Interest: Your Path to Wealth"
                ],
                "trending": [
                    "The 2026 Stock Market: What's Coming",
                    "New Tax Laws You Need to Know",
                    "Is This the Next Big Investment?",
                    "Breaking: Market Analysis Update",
                    "2026 Side Hustles That Actually Work"
                ],
                "viral": [
                    "The Money Secret They Don't Want You to Know",
                    "Why 90% of People Will Never Be Rich",
                    "The Dark Truth About Financial Advice",
                    "This Investment Trick Changed Everything",
                    "What Rich People Never Tell You"
                ]
            },
            "psychology": {
                "evergreen": [
                    "5 Signs of High Emotional Intelligence",
                    "How to Read Body Language Like a Pro",
                    "The Science of Building Good Habits",
                    "Understanding Cognitive Biases",
                    "How Your Brain Makes Decisions"
                ],
                "trending": [
                    "The Psychology Behind Social Media in 2026",
                    "New Research on Mental Health",
                    "Why Everyone is Talking About This Study"
                ],
                "viral": [
                    "Dark Psychology Tricks Used on You Daily",
                    "Signs Someone is Secretly a Narcissist",
                    "The Manipulation Tactics You Don't See",
                    "Why Intelligent People Do This"
                ]
            },
            "storytelling": {
                "evergreen": [
                    "The Untold Story of [Company Name]",
                    "How [Person] Built a Billion Dollar Empire",
                    "The Rise and Fall of [Brand]",
                    "What Really Happened at [Company]"
                ],
                "trending": [
                    "The 2026 [Industry] Revolution",
                    "Breaking Down the Latest [Event]",
                    "What's Really Going On With [Topic]"
                ],
                "viral": [
                    "The Dark Side of [Company/Person]",
                    "The Truth They Don't Want You to Know",
                    "This Changed Everything We Knew",
                    "The Scandal That Shocked Everyone"
                ]
            }
        }

        niche_templates = templates.get(niche, templates["finance"])
        type_templates = niche_templates.get(content_type, niche_templates["evergreen"])

        # Add viral patterns if available
        if content_type == "viral" and viral_patterns:
            type_templates = type_templates + viral_patterns[:3]

        return random.choice(type_templates)

    def _enhance_calendar_with_ai(
        self,
        calendar: List[ContentPlan],
        niche: str
    ) -> List[ContentPlan]:
        """Use AI to enhance calendar topics."""
        try:
            from ..content.script_writer import get_provider
            ai = get_provider(self.provider, self.api_key)

            # Only enhance first few items to save tokens
            topics_to_enhance = [p.topic for p in calendar[:5]]

            prompt = f"""Improve these YouTube video topics for a {niche} channel.
Make them more engaging and specific.

Original topics:
{json.dumps(topics_to_enhance, indent=2)}

Respond with ONLY a JSON array of improved topic strings:
["improved topic 1", "improved topic 2", ...]"""

            response = ai.generate(prompt, max_tokens=400)

            if "[" in response:
                start = response.find("[")
                end = response.rfind("]") + 1
                improved = json.loads(response[start:end])

                for i, topic in enumerate(improved):
                    if i < len(calendar):
                        calendar[i].topic = topic

        except Exception as e:
            logger.warning(f"AI enhancement failed: {e}")

        return calendar

    def recommend_topics(
        self,
        niche: str = None,
        channel: str = None,
        count: int = 10
    ) -> StrategyResult:
        """
        Generate recommended topics with risk/reward analysis.

        Args:
            niche: Content niche
            channel: Channel ID (for automatic niche detection)
            count: Number of topics to recommend

        Returns:
            StrategyResult with recommended topics
        """
        if channel and not niche:
            niche = self.CHANNEL_NICHE_MAP.get(channel, "default")
        niche = niche or "finance"

        operation = f"recommend_topics_{niche}_{count}"
        logger.info(f"[ContentStrategyAgent] Recommending {count} topics for: {niche}")

        practices = get_best_practices(niche)
        niche_metrics = get_niche_metrics(niche)

        # Generate topics from all content types
        topics = []

        # 60% evergreen
        evergreen_count = int(count * 0.6)
        for _ in range(evergreen_count):
            topic = self._generate_topic(niche, "evergreen", [])
            topics.append({
                "topic": topic,
                "content_type": "evergreen",
                "risk_level": "low",
                "reward_potential": "medium",
                "recommended_length": niche_metrics.get("optimal_video_length", (8, 15)),
                "expected_cpm": niche_metrics.get("cpm_range", (5, 10))[0]
            })

        # 25% trending
        trending_count = int(count * 0.25)
        for _ in range(trending_count):
            topic = self._generate_topic(niche, "trending", [])
            topics.append({
                "topic": topic,
                "content_type": "trending",
                "risk_level": "medium",
                "reward_potential": "high",
                "recommended_length": niche_metrics.get("optimal_video_length", (8, 15)),
                "expected_cpm": niche_metrics.get("cpm_range", (5, 10))[1]
            })

        # 15% viral
        viral_count = count - evergreen_count - trending_count
        viral_patterns = practices.get("viral_title_patterns", [])
        for _ in range(viral_count):
            topic = self._generate_topic(niche, "viral", viral_patterns)
            topics.append({
                "topic": topic,
                "content_type": "viral",
                "risk_level": "high",
                "reward_potential": "very high",
                "recommended_length": niche_metrics.get("optimal_video_length", (8, 15)),
                "expected_cpm": niche_metrics.get("cpm_range", (5, 10))[1] * 1.2
            })

        # Shuffle to mix types
        random.shuffle(topics)

        result = StrategyResult(
            success=True,
            operation=operation,
            recommended_topics=topics,
            insights=[
                f"Generated {len(topics)} topics following 60/25/15 content mix",
                f"Niche CPM range: ${niche_metrics.get('cpm_range', (5, 10))[0]}-${niche_metrics.get('cpm_range', (5, 10))[1]}"
            ],
            recommendations=[
                "Prioritize evergreen content for stable growth",
                "Use trending content to capture timely opportunities",
                "Limit viral attempts to avoid burnout"
            ],
            tokens_used=0,
            cost=0.0,
            provider="rule_based"
        )

        return result

    def track_strategy_execution(
        self,
        channel: str
    ) -> StrategyResult:
        """
        Track how well strategy is being executed against goals.

        Args:
            channel: Channel ID

        Returns:
            StrategyResult with execution tracking
        """
        operation = f"track_execution_{channel}"
        logger.info(f"[ContentStrategyAgent] Tracking strategy execution for: {channel}")

        try:
            # Get planned content
            with sqlite3.connect(self.strategy_db) as conn:
                planned = conn.execute("""
                    SELECT COUNT(*), status
                    FROM content_plans
                    WHERE channel = ? AND date >= date('now', '-30 days')
                    GROUP BY status
                """, (channel,)).fetchall()

            # Get actual uploads
            actual_uploads = 0
            if self.performance_db.exists():
                with sqlite3.connect(self.performance_db) as conn:
                    row = conn.execute("""
                        SELECT COUNT(*)
                        FROM video_performance
                        WHERE channel = ? AND uploaded_at >= date('now', '-30 days')
                    """, (channel,)).fetchone()
                    actual_uploads = row[0] if row else 0

            # Calculate execution rate
            total_planned = sum(p[0] for p in planned) if planned else 0
            completed = sum(p[0] for p in planned if p[1] == "completed") if planned else 0

            execution_rate = (completed / total_planned * 100) if total_planned > 0 else 0

            # Get goals
            with sqlite3.connect(self.strategy_db) as conn:
                goals_row = conn.execute("""
                    SELECT weekly_video_target, monthly_view_target
                    FROM strategy_goals
                    WHERE channel = ?
                """, (channel,)).fetchone()

            weekly_target = goals_row[0] if goals_row else 3
            monthly_view_target = goals_row[1] if goals_row else 10000

            insights = [
                f"Planned content: {total_planned} items",
                f"Completed: {completed} items ({execution_rate:.0f}%)",
                f"Actual uploads (30d): {actual_uploads}",
                f"Weekly target: {weekly_target} videos"
            ]

            recommendations = []
            if execution_rate < 50:
                recommendations.append("Execution rate is low. Simplify content workflow.")
            if actual_uploads < weekly_target * 4:
                recommendations.append(f"Below target. Aim for {weekly_target} videos/week.")

            strategy_score = min(100, int(execution_rate))

            result = StrategyResult(
                success=True,
                operation=operation,
                strategy_score=strategy_score,
                insights=insights,
                recommendations=recommendations,
                tokens_used=0,
                cost=0.0,
                provider="database"
            )

            return result

        except Exception as e:
            logger.error(f"[ContentStrategyAgent] Error tracking execution: {e}")
            return StrategyResult(
                success=False,
                operation=operation,
                error=str(e),
                provider="database"
            )

    def _calculate_strategy_score(
        self,
        portfolio: PortfolioBalance,
        videos: List[Dict]
    ) -> int:
        """Calculate overall strategy score (0-100)."""
        score = 0

        # Portfolio balance (40 points)
        if portfolio.is_balanced:
            score += 40
        else:
            # Partial credit based on how close to balanced
            evergreen_diff = abs(portfolio.evergreen_percent - 60)
            trending_diff = abs(portfolio.trending_percent - 25)
            viral_diff = abs(portfolio.viral_percent - 15)
            total_diff = evergreen_diff + trending_diff + viral_diff
            score += max(0, 40 - int(total_diff / 2))

        # Content volume (20 points)
        if len(videos) >= 12:  # 3/week for a month
            score += 20
        else:
            score += int(len(videos) / 12 * 20)

        # Consistency (20 points) - based on views
        if videos:
            views = [v["views"] for v in videos]
            avg_views = sum(views) / len(views)
            if avg_views > 5000:
                score += 20
            elif avg_views > 1000:
                score += 15
            elif avg_views > 500:
                score += 10
            else:
                score += 5

        # Engagement (20 points)
        if videos:
            retentions = [v.get("retention", 0) for v in videos]
            avg_retention = sum(retentions) / len(retentions) if retentions else 0
            if avg_retention > 50:
                score += 20
            elif avg_retention > 40:
                score += 15
            elif avg_retention > 30:
                score += 10
            else:
                score += 5

        return min(100, score)

    def _generate_portfolio_recommendations(
        self,
        portfolio: PortfolioBalance,
        videos: List[Dict],
        niche: str
    ) -> List[str]:
        """Generate recommendations based on portfolio analysis."""
        recommendations = []

        if portfolio.evergreen_percent < 50:
            recommendations.append(
                f"Increase evergreen content from {portfolio.evergreen_percent:.0f}% to 60%. "
                "These videos provide stable long-term views."
            )

        if portfolio.trending_percent > 35:
            recommendations.append(
                f"Reduce trending content from {portfolio.trending_percent:.0f}% to 25%. "
                "Too much trending content leads to inconsistent performance."
            )

        if portfolio.viral_percent > 25:
            recommendations.append(
                f"Reduce viral attempts from {portfolio.viral_percent:.0f}% to 15%. "
                "Focus on sustainable growth over viral hits."
            )

        niche_metrics = get_niche_metrics(niche)
        optimal_length = niche_metrics.get("optimal_video_length", (8, 15))
        recommendations.append(
            f"Target video length: {optimal_length[0]}-{optimal_length[1]} minutes "
            f"for optimal {niche} content performance."
        )

        return recommendations

    def _generate_growth_projections(
        self,
        channel: str,
        planned_videos: int,
        weeks: int
    ) -> Dict[str, GrowthProjection]:
        """Generate growth projections based on planned content."""
        niche = self.CHANNEL_NICHE_MAP.get(channel, "default")
        niche_metrics = get_niche_metrics(niche)

        # Estimate views per video based on niche
        base_views = 1000  # Conservative estimate
        cpm_range = niche_metrics.get("cpm_range", (5, 10))
        avg_cpm = (cpm_range[0] + cpm_range[1]) / 2

        projections = {}

        # Weekly projection
        weekly_videos = planned_videos / weeks if weeks > 0 else 3
        projections["weekly"] = GrowthProjection(
            period="weekly",
            current_videos=0,
            projected_videos=int(weekly_videos),
            current_views=0,
            projected_views=int(weekly_videos * base_views),
            confidence=0.7,
            assumptions=["Based on niche averages", "Assumes consistent posting"]
        )

        # Monthly projection
        monthly_videos = weekly_videos * 4
        projections["monthly"] = GrowthProjection(
            period="monthly",
            current_videos=0,
            projected_videos=int(monthly_videos),
            current_views=0,
            projected_views=int(monthly_videos * base_views * 1.2),  # Compounding effect
            confidence=0.6,
            assumptions=["Assumes growing subscriber base", "Based on niche averages"]
        )

        # Quarterly projection
        quarterly_videos = monthly_videos * 3
        projections["quarterly"] = GrowthProjection(
            period="quarterly",
            current_videos=0,
            projected_videos=int(quarterly_videos),
            current_views=0,
            projected_views=int(quarterly_videos * base_views * 1.5),
            confidence=0.5,
            assumptions=["High uncertainty", "Depends on algorithm favor"]
        )

        return projections

    def run(self, command: str = None, **kwargs) -> StrategyResult:
        """
        Main entry point for CLI usage.

        Args:
            command: Command string
            **kwargs: Parameters

        Returns:
            StrategyResult
        """
        channel = kwargs.get("channel")
        niche = kwargs.get("niche")
        weeks = kwargs.get("weeks", 4)
        calendar = kwargs.get("calendar", False)
        portfolio = kwargs.get("portfolio", False)
        topics = kwargs.get("topics", False)
        track = kwargs.get("track", False)

        if calendar:
            return self.generate_calendar(channel, weeks, use_ai=kwargs.get("ai", False))

        if portfolio and channel:
            return self.analyze_portfolio(channel, kwargs.get("period", "90d"))

        if track and channel:
            return self.track_strategy_execution(channel)

        if topics:
            return self.recommend_topics(niche=niche, channel=channel, count=kwargs.get("count", 10))

        # Default: portfolio analysis if channel provided, otherwise recommend topics
        if channel:
            return self.analyze_portfolio(channel)
        else:
            return self.recommend_topics(niche=niche or "finance")


# CLI entry point
def main():
    """CLI entry point for content strategy agent."""
    import sys

    if len(sys.argv) < 2:
        print("""
Content Strategy Agent - Long-term Content Planning

Usage:
    python -m src.agents.content_strategy_agent --channel money_blueprints --portfolio
    python -m src.agents.content_strategy_agent --channel money_blueprints --calendar --weeks 4
    python -m src.agents.content_strategy_agent --topics --niche finance --count 10
    python -m src.agents.content_strategy_agent --track --channel money_blueprints

Options:
    --channel <id>      Channel ID
    --niche <niche>     Content niche (finance, psychology, storytelling)
    --portfolio         Analyze content portfolio
    --calendar          Generate content calendar
    --weeks <n>         Weeks to plan (default: 4)
    --topics            Get recommended topics
    --count <n>         Number of topics (default: 10)
    --track             Track strategy execution
    --ai                Use AI enhancement (uses tokens)
    --save              Save report
    --json              Output as JSON

Examples:
    python -m src.agents.content_strategy_agent --channel money_blueprints --portfolio
    python -m src.agents.content_strategy_agent --calendar --channel mind_unlocked --weeks 4
    python -m src.agents.content_strategy_agent --topics --niche psychology --count 15
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
        elif sys.argv[i] == "--weeks" and i + 1 < len(sys.argv):
            kwargs["weeks"] = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--count" and i + 1 < len(sys.argv):
            kwargs["count"] = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--period" and i + 1 < len(sys.argv):
            kwargs["period"] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--portfolio":
            kwargs["portfolio"] = True
            i += 1
        elif sys.argv[i] == "--calendar":
            kwargs["calendar"] = True
            i += 1
        elif sys.argv[i] == "--topics":
            kwargs["topics"] = True
            i += 1
        elif sys.argv[i] == "--track":
            kwargs["track"] = True
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
    agent = ContentStrategyAgent()
    result = agent.run(**kwargs)

    # Output
    if output_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("\n" + "=" * 60)
        print("CONTENT STRATEGY AGENT RESULT")
        print("=" * 60)
        print(result.summary())

    # Save if requested
    if save_report:
        result.save()


if __name__ == "__main__":
    main()
