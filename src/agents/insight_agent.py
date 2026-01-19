"""
Insight Agent - AI-powered Analytics Insights

A token-efficient agent specialized in identifying successful patterns,
generating strategic recommendations, and producing executive summaries.

Uses Groq for AI analysis (free tier, fast inference).

Usage:
    from src.agents.insight_agent import InsightAgent

    agent = InsightAgent()

    # Identify patterns from top performers
    result = agent.identify_patterns("money_blueprints")

    # Generate strategic recommendations
    result = agent.generate_recommendations("finance")

    # Produce executive summary
    result = agent.executive_summary("money_blueprints", period="30d")
"""

import os
import json
import sqlite3
import hashlib
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
from ..utils.best_practices import get_best_practices, get_niche_metrics


# Cache TTL for insights (7 days in hours)
INSIGHT_CACHE_TTL_HOURS = 7 * 24


@dataclass
class PatternAnalysis:
    """Analysis of successful content patterns."""
    pattern_type: str  # title, hook, length, timing, etc.
    pattern: str
    occurrences: int
    avg_performance: float
    confidence: float
    examples: List[str] = field(default_factory=list)


@dataclass
class CompetitorInsight:
    """Insights from competitor analysis."""
    competitor: str
    strength: str
    weakness: str
    opportunity: str
    threat: str


@dataclass
class InsightResult:
    """Result from insight agent operations."""
    success: bool
    operation: str
    patterns: Dict[str, List[PatternAnalysis]] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    top_performers_analysis: str = ""
    underperformers_analysis: str = ""
    competitor_comparison: List[CompetitorInsight] = field(default_factory=list)
    executive_summary: str = ""
    predicted_trends: List[str] = field(default_factory=list)
    action_items: List[Dict[str, Any]] = field(default_factory=list)
    tokens_used: int = 0
    cost: float = 0.0
    provider: str = ""
    cache_hit: bool = False
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["patterns"] = {
            k: [asdict(p) for p in v]
            for k, v in self.patterns.items()
        }
        result["competitor_comparison"] = [
            asdict(c) for c in self.competitor_comparison
        ]
        return result

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Insights Report",
            f"===============",
            f"Provider: {self.provider} {'(cached)' if self.cache_hit else ''}",
            f"Tokens used: {self.tokens_used}",
            ""
        ]

        if self.executive_summary:
            lines.append("Executive Summary:")
            lines.append(self.executive_summary)
            lines.append("")

        if self.patterns:
            lines.append("Identified Patterns:")
            for pattern_type, patterns in self.patterns.items():
                lines.append(f"\n  {pattern_type.title()} Patterns:")
                for p in patterns[:3]:
                    lines.append(f"    - {p.pattern}")
                    lines.append(f"      Occurrences: {p.occurrences}, "
                               f"Confidence: {p.confidence:.0%}")
            lines.append("")

        if self.top_performers_analysis:
            lines.append("Top Performers Analysis:")
            lines.append(self.top_performers_analysis)
            lines.append("")

        if self.recommendations:
            lines.append("Strategic Recommendations:")
            for i, rec in enumerate(self.recommendations[:5], 1):
                lines.append(f"  {i}. {rec}")
            lines.append("")

        if self.action_items:
            lines.append("Action Items:")
            for item in self.action_items[:5]:
                lines.append(f"  [{item.get('priority', 'medium').upper()}] {item.get('action', '')}")
                lines.append(f"    Impact: {item.get('impact', 'Unknown')}")
            lines.append("")

        if self.predicted_trends:
            lines.append("Predicted Trends:")
            for trend in self.predicted_trends[:3]:
                lines.append(f"  - {trend}")

        return "\n".join(lines)

    def save(self, path: str = None):
        """Save result to JSON file."""
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"data/analytics_reports/insight_{timestamp}.json"

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Insight report saved to {path}")


class InsightAgent:
    """
    Insight Agent for AI-powered analytics insights.

    Token-efficient design:
    - Uses Groq (free tier) for AI analysis
    - Caches insights for 7 days
    - Rule-based pattern detection before AI
    - Minimal token prompts for efficiency
    """

    # Channel to niche mapping
    CHANNEL_NICHE_MAP = {
        "money_blueprints": "finance",
        "mind_unlocked": "psychology",
        "untold_stories": "storytelling",
    }

    def __init__(self, provider: str = None, api_key: str = None):
        """
        Initialize the insight agent.

        Args:
            provider: AI provider (defaults to Groq for free inference)
            api_key: API key for cloud providers
        """
        self.tracker = get_token_manager()
        self.optimizer = get_cost_optimizer()
        self.cache = get_prompt_cache()

        # Use Groq by default (free tier, fast)
        if provider is None:
            provider = "groq"

        self.provider = provider
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")

        # Database paths
        self.performance_db = Path("data/video_performance.db")
        self.insights_db = Path("data/analytics_reports/insights.db")

        self._init_db()

        logger.info(f"InsightAgent initialized with provider: {provider}")

    def _init_db(self):
        """Initialize insights database."""
        self.insights_db.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.insights_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS insight_cache (
                    cache_key TEXT PRIMARY KEY,
                    insight_data TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expires_at DATETIME
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires
                ON insight_cache(expires_at)
            """)

    def _get_cached_insight(self, cache_key: str) -> Optional[Dict]:
        """Get cached insight if not expired."""
        try:
            with sqlite3.connect(self.insights_db) as conn:
                row = conn.execute("""
                    SELECT insight_data FROM insight_cache
                    WHERE cache_key = ? AND expires_at > datetime('now')
                """, (cache_key,)).fetchone()

                if row:
                    return json.loads(row[0])
        except Exception as e:
            logger.warning(f"Cache read error: {e}")

        return None

    def _cache_insight(self, cache_key: str, data: Dict, ttl_hours: int = INSIGHT_CACHE_TTL_HOURS):
        """Cache insight data."""
        try:
            expires_at = (datetime.now() + timedelta(hours=ttl_hours)).isoformat()

            with sqlite3.connect(self.insights_db) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO insight_cache (cache_key, insight_data, expires_at)
                    VALUES (?, ?, ?)
                """, (cache_key, json.dumps(data), expires_at))

        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def _generate_cache_key(self, operation: str, **kwargs) -> str:
        """Generate cache key from operation and parameters."""
        key_data = f"{operation}_{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]

    def identify_patterns(
        self,
        channel: str = None,
        niche: str = None,
        period: str = "90d"
    ) -> InsightResult:
        """
        Identify successful patterns from top performers.

        Combines rule-based pattern detection with AI analysis.

        Args:
            channel: Channel ID
            niche: Content niche (auto-detected from channel)
            period: Analysis period

        Returns:
            InsightResult with identified patterns
        """
        if channel and not niche:
            niche = self.CHANNEL_NICHE_MAP.get(channel, "default")
        niche = niche or "default"

        operation = f"identify_patterns_{channel or niche}_{period}"
        cache_key = self._generate_cache_key(operation, channel=channel, niche=niche)

        logger.info(f"[InsightAgent] Identifying patterns for: {channel or niche}")

        # Check cache
        cached = self._get_cached_insight(cache_key)
        if cached:
            logger.info("[InsightAgent] Using cached patterns")
            return InsightResult(
                success=True,
                operation=operation,
                patterns=cached.get("patterns", {}),
                recommendations=cached.get("recommendations", []),
                top_performers_analysis=cached.get("top_performers_analysis", ""),
                cache_hit=True,
                tokens_used=0,
                cost=0.0,
                provider="cache"
            )

        # Get video data
        days = {"7d": 7, "30d": 30, "90d": 90, "all": 365 * 10}.get(period, 90)
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        videos = self._load_video_data(channel, cutoff)

        if not videos:
            return InsightResult(
                success=True,
                operation=operation,
                patterns={},
                recommendations=["No video data available. Upload content first."],
                provider="database"
            )

        # Rule-based pattern detection
        patterns = self._detect_patterns_rule_based(videos)

        # AI-enhanced pattern analysis
        ai_analysis = self._analyze_patterns_with_ai(videos, niche)

        if ai_analysis:
            patterns.update(ai_analysis.get("patterns", {}))
            recommendations = ai_analysis.get("recommendations", [])
            top_analysis = ai_analysis.get("top_performers_analysis", "")
            tokens_used = ai_analysis.get("tokens_used", 0)
        else:
            recommendations = self._generate_rule_based_recommendations(patterns, niche)
            top_analysis = self._generate_top_performers_summary(videos)
            tokens_used = 0

        result = InsightResult(
            success=True,
            operation=operation,
            patterns=patterns,
            recommendations=recommendations,
            top_performers_analysis=top_analysis,
            tokens_used=tokens_used,
            cost=self.tracker.record_usage(
                provider=self.provider,
                input_tokens=tokens_used // 2,
                output_tokens=tokens_used // 2,
                operation="insight_patterns"
            ) if tokens_used > 0 else 0.0,
            provider=self.provider if tokens_used > 0 else "rule_based"
        )

        # Cache the result
        cache_data = {
            "patterns": {k: [asdict(p) for p in v] for k, v in patterns.items()},
            "recommendations": recommendations,
            "top_performers_analysis": top_analysis
        }
        self._cache_insight(cache_key, cache_data)

        logger.success(f"[InsightAgent] Pattern identification complete")
        return result

    def _load_video_data(
        self,
        channel: str = None,
        cutoff: str = None
    ) -> List[Dict]:
        """Load video data from performance database."""
        if not self.performance_db.exists():
            return []

        try:
            with sqlite3.connect(self.performance_db) as conn:
                if channel:
                    query = """
                        SELECT video_id, title, views, likes, comments, retention, ctr, uploaded_at
                        FROM video_performance
                        WHERE channel = ? AND (uploaded_at >= ? OR uploaded_at IS NULL)
                        ORDER BY views DESC
                    """
                    rows = conn.execute(query, (channel, cutoff)).fetchall()
                else:
                    query = """
                        SELECT video_id, title, views, likes, comments, retention, ctr, uploaded_at
                        FROM video_performance
                        WHERE uploaded_at >= ? OR uploaded_at IS NULL
                        ORDER BY views DESC
                    """
                    rows = conn.execute(query, (cutoff,)).fetchall()

            videos = []
            for row in rows:
                videos.append({
                    "video_id": row[0],
                    "title": row[1] or "",
                    "views": row[2] or 0,
                    "likes": row[3] or 0,
                    "comments": row[4] or 0,
                    "retention": row[5] or 0,
                    "ctr": row[6] or 0,
                    "uploaded_at": row[7]
                })

            return videos

        except Exception as e:
            logger.warning(f"Error loading video data: {e}")
            return []

    def _detect_patterns_rule_based(
        self,
        videos: List[Dict]
    ) -> Dict[str, List[PatternAnalysis]]:
        """Detect patterns using rule-based analysis."""
        patterns = {
            "title": [],
            "length": [],
            "timing": []
        }

        if not videos:
            return patterns

        # Sort by views for top performer analysis
        sorted_videos = sorted(videos, key=lambda x: x["views"], reverse=True)
        top_videos = sorted_videos[:max(len(sorted_videos) // 4, 3)]

        # Title pattern analysis
        title_keywords = {}
        for video in top_videos:
            title = video.get("title", "").lower()
            words = title.split()
            for word in words:
                if len(word) > 3:
                    if word not in title_keywords:
                        title_keywords[word] = {"count": 0, "total_views": 0, "examples": []}
                    title_keywords[word]["count"] += 1
                    title_keywords[word]["total_views"] += video["views"]
                    if len(title_keywords[word]["examples"]) < 3:
                        title_keywords[word]["examples"].append(video["title"][:50])

        # Find significant keywords
        for word, data in sorted(title_keywords.items(), key=lambda x: x[1]["count"], reverse=True)[:5]:
            if data["count"] >= 2:
                avg_views = data["total_views"] / data["count"]
                patterns["title"].append(PatternAnalysis(
                    pattern_type="title",
                    pattern=f"Title contains '{word}'",
                    occurrences=data["count"],
                    avg_performance=avg_views,
                    confidence=min(data["count"] / len(top_videos), 1.0),
                    examples=data["examples"]
                ))

        # Title structure patterns
        structure_patterns = {
            "numbers": lambda t: any(c.isdigit() for c in t),
            "questions": lambda t: "?" in t,
            "how_to": lambda t: t.lower().startswith("how"),
            "why": lambda t: t.lower().startswith("why")
        }

        for pattern_name, check_func in structure_patterns.items():
            matching = [v for v in top_videos if check_func(v.get("title", ""))]
            if matching and len(matching) >= 2:
                avg_views = sum(v["views"] for v in matching) / len(matching)
                patterns["title"].append(PatternAnalysis(
                    pattern_type="title",
                    pattern=f"Title structure: {pattern_name.replace('_', ' ')}",
                    occurrences=len(matching),
                    avg_performance=avg_views,
                    confidence=len(matching) / len(top_videos),
                    examples=[v["title"][:50] for v in matching[:3]]
                ))

        return patterns

    def _analyze_patterns_with_ai(
        self,
        videos: List[Dict],
        niche: str
    ) -> Optional[Dict]:
        """Use AI to analyze patterns (Groq for efficiency)."""
        if not videos:
            return None

        try:
            from ..content.script_writer import get_provider
            ai = get_provider(self.provider, self.api_key)

            # Prepare condensed data (keep tokens low)
            top_videos = sorted(videos, key=lambda x: x["views"], reverse=True)[:5]
            bottom_videos = sorted(videos, key=lambda x: x["views"])[:3]

            video_summary = {
                "top_performers": [
                    {"title": v["title"][:50], "views": v["views"], "retention": v["retention"]}
                    for v in top_videos
                ],
                "underperformers": [
                    {"title": v["title"][:50], "views": v["views"], "retention": v["retention"]}
                    for v in bottom_videos
                ]
            }

            # Keep prompt concise for token efficiency
            prompt = f"""Analyze these {niche} YouTube video patterns briefly.

TOP PERFORMERS:
{json.dumps(video_summary["top_performers"], indent=2)}

UNDERPERFORMERS:
{json.dumps(video_summary["underperformers"], indent=2)}

Respond with ONLY JSON:
{{
    "patterns": {{
        "content": [
            {{"pattern": "pattern description", "confidence": 0.8}}
        ]
    }},
    "recommendations": ["rec1", "rec2", "rec3"],
    "top_performers_analysis": "Brief 1-2 sentence analysis"
}}"""

            response = ai.generate(prompt, max_tokens=500)

            # Parse response
            if "{" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                result = json.loads(response[start:end])

                # Convert patterns to PatternAnalysis objects
                if "patterns" in result and "content" in result["patterns"]:
                    result["patterns"]["content"] = [
                        PatternAnalysis(
                            pattern_type="content",
                            pattern=p.get("pattern", ""),
                            occurrences=0,
                            avg_performance=0,
                            confidence=p.get("confidence", 0.5)
                        )
                        for p in result["patterns"]["content"]
                    ]

                result["tokens_used"] = 800  # Estimate
                return result

        except Exception as e:
            logger.warning(f"AI pattern analysis failed: {e}")

        return None

    def _generate_rule_based_recommendations(
        self,
        patterns: Dict[str, List[PatternAnalysis]],
        niche: str
    ) -> List[str]:
        """Generate recommendations based on detected patterns."""
        recommendations = []
        practices = get_best_practices(niche)

        # From title patterns
        if patterns.get("title"):
            top_pattern = patterns["title"][0]
            recommendations.append(
                f"Use more titles with '{top_pattern.pattern}' - "
                f"associated with {top_pattern.avg_performance:,.0f} avg views"
            )

        # From best practices
        if practices.get("viral_title_patterns"):
            recommendations.append(
                f"Try viral pattern: {practices['viral_title_patterns'][0]}"
            )

        # Generic niche recommendations
        niche_metrics = get_niche_metrics(niche)
        optimal_length = niche_metrics.get("optimal_video_length", (8, 15))
        recommendations.append(
            f"Target {optimal_length[0]}-{optimal_length[1]} min videos for optimal {niche} performance"
        )

        return recommendations

    def _generate_top_performers_summary(self, videos: List[Dict]) -> str:
        """Generate summary of top performers."""
        if not videos:
            return "No video data available."

        top_videos = sorted(videos, key=lambda x: x["views"], reverse=True)[:3]

        summary_parts = []
        for v in top_videos:
            summary_parts.append(f"'{v['title'][:40]}...' ({v['views']:,} views)")

        return f"Top performers: {'; '.join(summary_parts)}"

    def generate_recommendations(
        self,
        channel: str = None,
        niche: str = None,
        context: Dict = None
    ) -> InsightResult:
        """
        Generate strategic recommendations using AI.

        Args:
            channel: Channel ID
            niche: Content niche
            context: Additional context data

        Returns:
            InsightResult with recommendations
        """
        if channel and not niche:
            niche = self.CHANNEL_NICHE_MAP.get(channel, "default")
        niche = niche or "default"

        operation = f"generate_recommendations_{channel or niche}"
        cache_key = self._generate_cache_key(operation, channel=channel, niche=niche)

        logger.info(f"[InsightAgent] Generating recommendations for: {channel or niche}")

        # Check cache
        cached = self._get_cached_insight(cache_key)
        if cached:
            return InsightResult(
                success=True,
                operation=operation,
                recommendations=cached.get("recommendations", []),
                action_items=cached.get("action_items", []),
                cache_hit=True,
                provider="cache"
            )

        # Get context data
        practices = get_best_practices(niche)
        metrics = get_niche_metrics(niche)

        # Generate AI recommendations
        try:
            from ..content.script_writer import get_provider
            ai = get_provider(self.provider, self.api_key)

            prompt = f"""Generate 5 actionable recommendations for a {niche} YouTube channel.

Context:
- CPM range: ${metrics.get('cpm_range', (5, 10))[0]}-${metrics.get('cpm_range', (5, 10))[1]}
- Optimal length: {metrics.get('optimal_video_length', (8, 15))} min
- Best days: {metrics.get('best_days', ['Mon', 'Wed', 'Fri'])}

Respond with ONLY JSON:
{{
    "recommendations": ["specific rec 1", "specific rec 2", ...],
    "action_items": [
        {{"action": "specific action", "priority": "high/medium/low", "impact": "expected result"}}
    ]
}}"""

            response = ai.generate(prompt, max_tokens=600)
            tokens_used = 800

            recommendations = []
            action_items = []

            if "{" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                data = json.loads(response[start:end])
                recommendations = data.get("recommendations", [])
                action_items = data.get("action_items", [])

        except Exception as e:
            logger.warning(f"AI recommendations failed: {e}")
            recommendations = [
                f"Target {metrics.get('optimal_video_length', (8, 15))[0]}-{metrics.get('optimal_video_length', (8, 15))[1]} min videos",
                f"Post on {', '.join(metrics.get('best_days', ['Mon', 'Wed', 'Fri']))}",
                "Use numbers and power words in titles",
                "Hook viewers in first 5 seconds",
                "Add pattern interrupts every 30-60 seconds"
            ]
            action_items = [
                {"action": "Review top performing titles", "priority": "high", "impact": "Improve CTR"},
                {"action": "Analyze retention graphs", "priority": "medium", "impact": "Identify drop-off points"}
            ]
            tokens_used = 0

        result = InsightResult(
            success=True,
            operation=operation,
            recommendations=recommendations,
            action_items=action_items,
            tokens_used=tokens_used,
            cost=self.tracker.record_usage(
                provider=self.provider,
                input_tokens=tokens_used // 2,
                output_tokens=tokens_used // 2,
                operation="insight_recommendations"
            ) if tokens_used > 0 else 0.0,
            provider=self.provider if tokens_used > 0 else "rule_based"
        )

        # Cache result
        self._cache_insight(cache_key, {
            "recommendations": recommendations,
            "action_items": action_items
        })

        return result

    def compare_competitors(
        self,
        niche: str
    ) -> InsightResult:
        """
        Compare against competitors in the niche.

        Uses pre-compiled competitor data (no tokens).

        Args:
            niche: Content niche

        Returns:
            InsightResult with competitor comparison
        """
        operation = f"compare_competitors_{niche}"
        logger.info(f"[InsightAgent] Comparing competitors for: {niche}")

        practices = get_best_practices(niche)

        # Pre-defined competitor data by niche
        competitor_data = {
            "finance": [
                CompetitorInsight(
                    competitor="The Swedish Investor",
                    strength="Deep business analysis, professional presentation",
                    weakness="Slow upload frequency",
                    opportunity="More trending content on recent events",
                    threat="Highly polished content raises viewer expectations"
                ),
                CompetitorInsight(
                    competitor="Practical Wisdom",
                    strength="Consistent uploads, clear explanations",
                    weakness="Less unique visual style",
                    opportunity="Differentiate with storytelling approach",
                    threat="Similar target audience"
                )
            ],
            "psychology": [
                CompetitorInsight(
                    competitor="Psych2Go",
                    strength="12.7M subscribers, strong brand recognition",
                    weakness="Animation style can feel repetitive",
                    opportunity="Deeper analysis content",
                    threat="Dominant market presence"
                ),
                CompetitorInsight(
                    competitor="Brainy Dose",
                    strength="Consistent viral content",
                    weakness="Surface-level analysis",
                    opportunity="Provide more scientific depth",
                    threat="Similar topic selection"
                )
            ],
            "storytelling": [
                CompetitorInsight(
                    competitor="JCS Criminal Psychology",
                    strength="5M+ subs, unique interrogation analysis",
                    weakness="Limited topic range",
                    opportunity="Broader storytelling topics",
                    threat="Sets high production standard"
                ),
                CompetitorInsight(
                    competitor="Lazy Masquerade",
                    strength="1.7M subs, mystery content",
                    weakness="Niche focus limits growth",
                    opportunity="Expand to business/tech stories",
                    threat="Strong mystery/horror audience"
                )
            ]
        }

        competitors = competitor_data.get(niche, [
            CompetitorInsight(
                competitor="Generic Competitor",
                strength="Established audience",
                weakness="Inconsistent quality",
                opportunity="Content gaps to fill",
                threat="Market saturation"
            )
        ])

        # Generate insights
        recommendations = [
            f"Differentiate from {competitors[0].competitor}: {competitors[0].opportunity}",
            f"Avoid weakness of competitors: {competitors[0].weakness}",
            "Focus on consistent upload schedule to build audience",
            "Monitor competitor titles and thumbnails for trends"
        ]

        result = InsightResult(
            success=True,
            operation=operation,
            competitor_comparison=competitors,
            recommendations=recommendations,
            tokens_used=0,
            cost=0.0,
            provider="competitor_database"
        )

        return result

    def executive_summary(
        self,
        channel: str = None,
        period: str = "30d"
    ) -> InsightResult:
        """
        Produce executive summary of channel performance.

        Args:
            channel: Channel ID
            period: Analysis period

        Returns:
            InsightResult with executive summary
        """
        operation = f"executive_summary_{channel}_{period}"
        cache_key = self._generate_cache_key(operation, channel=channel)

        logger.info(f"[InsightAgent] Generating executive summary for: {channel}")

        # Check cache
        cached = self._get_cached_insight(cache_key)
        if cached:
            return InsightResult(
                success=True,
                operation=operation,
                executive_summary=cached.get("executive_summary", ""),
                recommendations=cached.get("recommendations", []),
                predicted_trends=cached.get("predicted_trends", []),
                cache_hit=True,
                provider="cache"
            )

        # Get video data
        days = {"7d": 7, "30d": 30, "90d": 90}.get(period, 30)
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        videos = self._load_video_data(channel, cutoff)
        niche = self.CHANNEL_NICHE_MAP.get(channel, "default")

        # Calculate metrics
        total_videos = len(videos)
        total_views = sum(v["views"] for v in videos)
        avg_views = total_views / total_videos if total_videos > 0 else 0
        avg_retention = sum(v["retention"] for v in videos) / total_videos if total_videos > 0 else 0
        avg_ctr = sum(v["ctr"] for v in videos) / total_videos if total_videos > 0 else 0

        # Generate summary with AI
        try:
            from ..content.script_writer import get_provider
            ai = get_provider(self.provider, self.api_key)

            prompt = f"""Write a brief executive summary for this {niche} YouTube channel ({period} data):

Metrics:
- Videos: {total_videos}
- Total views: {total_views:,}
- Avg views/video: {avg_views:,.0f}
- Avg retention: {avg_retention:.1f}%
- Avg CTR: {avg_ctr:.1f}%

Respond with ONLY JSON:
{{
    "executive_summary": "2-3 sentence summary",
    "predicted_trends": ["trend1", "trend2"],
    "recommendations": ["rec1", "rec2", "rec3"]
}}"""

            response = ai.generate(prompt, max_tokens=400)
            tokens_used = 600

            if "{" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                data = json.loads(response[start:end])

                summary = data.get("executive_summary", "")
                trends = data.get("predicted_trends", [])
                recommendations = data.get("recommendations", [])
            else:
                raise ValueError("Invalid AI response")

        except Exception as e:
            logger.warning(f"AI summary failed: {e}")
            summary = (
                f"The {niche} channel uploaded {total_videos} videos in the past {period}, "
                f"generating {total_views:,} total views with an average of {avg_views:,.0f} views per video. "
                f"Retention rate is {avg_retention:.1f}% and CTR is {avg_ctr:.1f}%."
            )
            trends = ["Continue current strategy", "Monitor algorithm changes"]
            recommendations = ["Maintain upload consistency", "Focus on retention optimization"]
            tokens_used = 0

        result = InsightResult(
            success=True,
            operation=operation,
            executive_summary=summary,
            predicted_trends=trends,
            recommendations=recommendations,
            tokens_used=tokens_used,
            cost=self.tracker.record_usage(
                provider=self.provider,
                input_tokens=tokens_used // 2,
                output_tokens=tokens_used // 2,
                operation="insight_executive_summary"
            ) if tokens_used > 0 else 0.0,
            provider=self.provider if tokens_used > 0 else "rule_based"
        )

        # Cache result
        self._cache_insight(cache_key, {
            "executive_summary": summary,
            "predicted_trends": trends,
            "recommendations": recommendations
        })

        return result

    def run(self, command: str = None, **kwargs) -> InsightResult:
        """
        Main entry point for CLI usage.

        Args:
            command: Command string
            **kwargs: Parameters

        Returns:
            InsightResult
        """
        channel = kwargs.get("channel")
        niche = kwargs.get("niche")
        period = kwargs.get("period", "30d")
        patterns = kwargs.get("patterns", False)
        recommendations = kwargs.get("recommendations", False)
        competitors = kwargs.get("competitors", False)
        summary = kwargs.get("summary", False)

        if patterns:
            return self.identify_patterns(channel, niche, period)

        if recommendations:
            return self.generate_recommendations(channel, niche)

        if competitors:
            return self.compare_competitors(niche or self.CHANNEL_NICHE_MAP.get(channel, "finance"))

        if summary:
            return self.executive_summary(channel, period)

        # Default: executive summary if channel provided, otherwise patterns
        if channel:
            return self.executive_summary(channel, period)
        else:
            return self.identify_patterns(channel, niche, period)


# CLI entry point
def main():
    """CLI entry point for insight agent."""
    import sys

    if len(sys.argv) < 2:
        print("""
Insight Agent - AI-powered Analytics Insights

Usage:
    python -m src.agents.insight_agent --channel money_blueprints --summary
    python -m src.agents.insight_agent --patterns --channel money_blueprints
    python -m src.agents.insight_agent --recommendations --niche finance
    python -m src.agents.insight_agent --competitors --niche psychology

Options:
    --channel <id>      Channel ID
    --niche <niche>     Content niche (finance, psychology, storytelling)
    --period <period>   Analysis period (7d, 30d, 90d)
    --patterns          Identify successful patterns
    --recommendations   Generate strategic recommendations
    --competitors       Compare against competitors
    --summary           Generate executive summary
    --save              Save report
    --json              Output as JSON

Examples:
    python -m src.agents.insight_agent --channel money_blueprints --summary
    python -m src.agents.insight_agent --patterns --channel mind_unlocked --period 90d
    python -m src.agents.insight_agent --competitors --niche finance
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
        elif sys.argv[i] == "--patterns":
            kwargs["patterns"] = True
            i += 1
        elif sys.argv[i] == "--recommendations":
            kwargs["recommendations"] = True
            i += 1
        elif sys.argv[i] == "--competitors":
            kwargs["competitors"] = True
            i += 1
        elif sys.argv[i] == "--summary":
            kwargs["summary"] = True
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
    agent = InsightAgent()
    result = agent.run(**kwargs)

    # Output
    if output_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("\n" + "=" * 60)
        print("INSIGHT AGENT RESULT")
        print("=" * 60)
        print(result.summary())

    # Save if requested
    if save_report:
        result.save()


if __name__ == "__main__":
    main()
