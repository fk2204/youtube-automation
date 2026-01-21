"""
Predictive Topic Selection Module for YouTube Automation

Combines multiple signals to predict viral potential and recommend
optimal posting timing for maximum reach.

Data Sources:
- Google Trends velocity
- Reddit engagement growth
- YouTube search suggest changes
- Seasonal patterns
- Historical performance data

Features:
- Score topics by predicted viral potential
- Recommend optimal posting timing
- Detect emerging trends before peak
- Identify seasonal opportunities
- Cross-reference multiple data sources

Usage:
    from src.research.trend_predictor import TrendPredictor

    predictor = TrendPredictor()

    # Get viral potential score for a topic
    score = await predictor.score_topic("passive income", niche="finance")
    print(f"Viral Score: {score.viral_score}")
    print(f"Best Time: {score.optimal_posting_time}")

    # Predict emerging trends
    trends = await predictor.predict_emerging_trends("finance")

    # Get seasonal opportunities
    opportunities = await predictor.get_seasonal_opportunities(months_ahead=2)
"""

import os
import json
import asyncio
import sqlite3
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from enum import Enum
from collections import Counter
from loguru import logger

try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    logger.warning("pytrends not installed. Install with: pip install pytrends")
    PYTRENDS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    logger.warning("numpy not installed. Some features may be limited.")
    NUMPY_AVAILABLE = False


class TrendDirection(Enum):
    """Direction of trend movement."""
    RISING = "rising"
    STABLE = "stable"
    FALLING = "falling"
    SEASONAL_PEAK = "seasonal_peak"
    BREAKOUT = "breakout"


class SignalSource(Enum):
    """Data source for trend signals."""
    GOOGLE_TRENDS = "google_trends"
    REDDIT = "reddit"
    YOUTUBE_SUGGEST = "youtube_suggest"
    SEASONAL = "seasonal"
    HISTORICAL = "historical"


@dataclass
class TrendSignal:
    """A signal from a single data source."""
    source: SignalSource
    topic: str
    score: float  # 0-100
    direction: TrendDirection
    velocity: float  # Rate of change (negative = falling)
    confidence: float  # 0-1
    data_points: int
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["source"] = self.source.value
        result["direction"] = self.direction.value
        result["timestamp"] = self.timestamp.isoformat()
        return result


@dataclass
class TopicScore:
    """Viral potential score for a topic."""
    topic: str
    niche: str
    viral_score: float  # 0-100 overall score
    trend_score: float  # Current trending momentum
    competition_score: float  # Lower = less competition
    seasonality_score: float  # How seasonal is this topic
    freshness_score: float  # Is it new/emerging
    signals: List[TrendSignal]
    optimal_posting_time: Optional[datetime]
    posting_urgency: str  # immediate, this_week, flexible
    confidence: float
    analysis_time: datetime = field(default_factory=datetime.now)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["signals"] = [s.to_dict() for s in self.signals]
        result["optimal_posting_time"] = self.optimal_posting_time.isoformat() if self.optimal_posting_time else None
        result["analysis_time"] = self.analysis_time.isoformat()
        return result


@dataclass
class SeasonalEvent:
    """A seasonal event that affects content performance."""
    name: str
    start_date: str  # MM-DD format
    end_date: str  # MM-DD format
    peak_date: str  # MM-DD format
    boost_niches: List[str]
    boost_topics: List[str]
    historical_lift: float  # Multiplier (1.5 = 50% boost)

    def is_active(self, date: datetime = None) -> bool:
        date = date or datetime.now()
        start = datetime(date.year, int(self.start_date[:2]), int(self.start_date[3:]))
        end = datetime(date.year, int(self.end_date[:2]), int(self.end_date[3:]))
        return start <= date <= end

    def days_until_peak(self, date: datetime = None) -> int:
        date = date or datetime.now()
        peak = datetime(date.year, int(self.peak_date[:2]), int(self.peak_date[3:]))
        if peak < date:
            peak = datetime(date.year + 1, int(self.peak_date[:2]), int(self.peak_date[3:]))
        return (peak - date).days


# Major seasonal events that affect content performance
SEASONAL_EVENTS: List[SeasonalEvent] = [
    SeasonalEvent(
        name="New Year / Resolution Season",
        start_date="12-26",
        end_date="01-31",
        peak_date="01-01",
        boost_niches=["finance", "psychology", "self-improvement"],
        boost_topics=["goals", "habits", "money", "budgeting", "productivity", "motivation"],
        historical_lift=1.8
    ),
    SeasonalEvent(
        name="Valentine's Day",
        start_date="02-01",
        end_date="02-14",
        peak_date="02-14",
        boost_niches=["psychology", "relationships"],
        boost_topics=["love", "dating", "relationships", "psychology", "attraction"],
        historical_lift=1.4
    ),
    SeasonalEvent(
        name="Tax Season",
        start_date="02-15",
        end_date="04-15",
        peak_date="04-01",
        boost_niches=["finance"],
        boost_topics=["taxes", "deductions", "tax tips", "filing", "refund"],
        historical_lift=2.2
    ),
    SeasonalEvent(
        name="Summer Planning",
        start_date="04-15",
        end_date="06-01",
        peak_date="05-15",
        boost_niches=["finance", "lifestyle"],
        boost_topics=["travel", "vacation", "budget", "saving"],
        historical_lift=1.3
    ),
    SeasonalEvent(
        name="Back to School",
        start_date="07-15",
        end_date="09-15",
        peak_date="08-15",
        boost_niches=["finance", "psychology", "education"],
        boost_topics=["budgeting", "student", "learning", "productivity", "focus"],
        historical_lift=1.5
    ),
    SeasonalEvent(
        name="Halloween",
        start_date="10-01",
        end_date="10-31",
        peak_date="10-31",
        boost_niches=["storytelling", "psychology"],
        boost_topics=["scary", "horror", "dark", "mystery", "creepy", "true crime"],
        historical_lift=1.6
    ),
    SeasonalEvent(
        name="Holiday Shopping",
        start_date="11-01",
        end_date="12-31",
        peak_date="11-29",
        boost_niches=["finance"],
        boost_topics=["shopping", "deals", "black friday", "gifts", "budget", "saving"],
        historical_lift=1.9
    ),
    SeasonalEvent(
        name="Year End Review",
        start_date="12-01",
        end_date="12-31",
        peak_date="12-20",
        boost_niches=["finance", "psychology", "storytelling"],
        boost_topics=["year review", "best of", "predictions", "lessons learned"],
        historical_lift=1.4
    ),
]


class GoogleTrendsAnalyzer:
    """Analyze Google Trends data for topic prediction."""

    def __init__(self):
        if PYTRENDS_AVAILABLE:
            self.pytrends = TrendReq(hl='en-US', tz=360)
        else:
            self.pytrends = None

    async def get_trend_data(
        self,
        topic: str,
        timeframe: str = "today 3-m"
    ) -> Optional[TrendSignal]:
        """Get trend data from Google Trends."""
        if not self.pytrends:
            return None

        try:
            # Build payload
            self.pytrends.build_payload([topic], timeframe=timeframe)

            # Get interest over time
            data = await asyncio.get_event_loop().run_in_executor(
                None, self.pytrends.interest_over_time
            )

            if data.empty:
                return None

            values = data[topic].values.tolist()

            # Calculate trend metrics
            current = values[-1] if values else 0
            avg = sum(values) / len(values) if values else 0

            # Calculate velocity (rate of change)
            if len(values) >= 4:
                recent_avg = sum(values[-4:]) / 4
                older_avg = sum(values[:4]) / 4
                velocity = (recent_avg - older_avg) / max(older_avg, 1) * 100
            else:
                velocity = 0

            # Determine direction
            if velocity > 20:
                direction = TrendDirection.BREAKOUT
            elif velocity > 5:
                direction = TrendDirection.RISING
            elif velocity < -10:
                direction = TrendDirection.FALLING
            else:
                direction = TrendDirection.STABLE

            return TrendSignal(
                source=SignalSource.GOOGLE_TRENDS,
                topic=topic,
                score=min(100, current * 1.0),  # Normalize
                direction=direction,
                velocity=velocity,
                confidence=0.8 if len(values) >= 10 else 0.5,
                data_points=len(values),
                metadata={"current": current, "average": avg, "max": max(values) if values else 0}
            )

        except Exception as e:
            logger.warning(f"Google Trends fetch failed: {e}")
            return None

    async def get_related_queries(self, topic: str) -> List[str]:
        """Get related queries for topic expansion."""
        if not self.pytrends:
            return []

        try:
            self.pytrends.build_payload([topic], timeframe="today 3-m")
            related = await asyncio.get_event_loop().run_in_executor(
                None, self.pytrends.related_queries
            )

            if topic in related and related[topic].get("rising") is not None:
                rising = related[topic]["rising"]
                if not rising.empty:
                    return rising["query"].tolist()[:10]

            return []
        except Exception as e:
            logger.debug(f"Related queries fetch failed: {e}")
            return []

    async def compare_topics(
        self,
        topics: List[str],
        timeframe: str = "today 3-m"
    ) -> Dict[str, float]:
        """Compare multiple topics and rank by interest."""
        if not self.pytrends or len(topics) > 5:
            return {}

        try:
            self.pytrends.build_payload(topics[:5], timeframe=timeframe)
            data = await asyncio.get_event_loop().run_in_executor(
                None, self.pytrends.interest_over_time
            )

            if data.empty:
                return {}

            # Calculate average interest for each topic
            results = {}
            for topic in topics[:5]:
                if topic in data.columns:
                    values = data[topic].values.tolist()
                    results[topic] = sum(values) / len(values) if values else 0

            return results

        except Exception as e:
            logger.warning(f"Topic comparison failed: {e}")
            return {}


class RedditSignalAnalyzer:
    """Analyze Reddit for engagement signals."""

    def __init__(self):
        self._reddit = None

    async def _get_reddit(self):
        """Lazy load Reddit client."""
        if self._reddit is None:
            try:
                from src.research.reddit import RedditResearcher
                researcher = RedditResearcher()
                if researcher.reddit:
                    self._reddit = researcher
            except Exception:
                pass
        return self._reddit

    async def get_engagement_signal(
        self,
        topic: str,
        subreddits: Optional[List[str]] = None
    ) -> Optional[TrendSignal]:
        """Get engagement signal from Reddit."""
        reddit = await self._get_reddit()
        if not reddit:
            return None

        try:
            # Search for posts about topic
            posts = reddit.search_posts(topic, subreddits=subreddits, limit=50)

            if not posts:
                return TrendSignal(
                    source=SignalSource.REDDIT,
                    topic=topic,
                    score=0,
                    direction=TrendDirection.STABLE,
                    velocity=0,
                    confidence=0.3,
                    data_points=0
                )

            # Calculate engagement metrics
            total_score = sum(p.score for p in posts)
            total_comments = sum(p.num_comments for p in posts)
            avg_score = total_score / len(posts)

            # Calculate velocity (posts from last 24h vs older)
            now = datetime.now()
            recent_posts = [p for p in posts if (now - p.created_utc).days < 1]
            older_posts = [p for p in posts if (now - p.created_utc).days >= 1]

            if older_posts:
                recent_avg = sum(p.score for p in recent_posts) / max(len(recent_posts), 1)
                older_avg = sum(p.score for p in older_posts) / len(older_posts)
                velocity = (recent_avg - older_avg) / max(older_avg, 1) * 100
            else:
                velocity = 0

            # Determine direction
            if len(recent_posts) > len(posts) * 0.3 and velocity > 20:
                direction = TrendDirection.BREAKOUT
            elif velocity > 10:
                direction = TrendDirection.RISING
            elif velocity < -20:
                direction = TrendDirection.FALLING
            else:
                direction = TrendDirection.STABLE

            # Calculate score (0-100)
            score = min(100, (avg_score / 100) * 50 + (total_comments / 1000) * 50)

            return TrendSignal(
                source=SignalSource.REDDIT,
                topic=topic,
                score=score,
                direction=direction,
                velocity=velocity,
                confidence=0.6 if len(posts) >= 10 else 0.4,
                data_points=len(posts),
                metadata={
                    "total_score": total_score,
                    "total_comments": total_comments,
                    "avg_score": avg_score,
                    "recent_posts": len(recent_posts)
                }
            )

        except Exception as e:
            logger.warning(f"Reddit signal fetch failed: {e}")
            return None


class YouTubeSuggestAnalyzer:
    """Analyze YouTube search suggestions for trend signals."""

    def __init__(self):
        self._suggestion_cache: Dict[str, Tuple[List[str], datetime]] = {}
        self._cache_duration = timedelta(hours=6)

    async def get_suggestions(self, query: str) -> List[str]:
        """Get YouTube search suggestions for a query."""
        # Check cache
        if query in self._suggestion_cache:
            suggestions, cached_time = self._suggestion_cache[query]
            if datetime.now() - cached_time < self._cache_duration:
                return suggestions

        try:
            import requests

            # YouTube suggestion API
            url = "https://suggestqueries.google.com/complete/search"
            params = {
                "client": "youtube",
                "q": query,
                "hl": "en"
            }

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.get(url, params=params, timeout=10)
            )

            if response.status_code == 200:
                # Parse JSONP response
                text = response.text
                if text.startswith("window.google.ac.h("):
                    text = text[19:-1]  # Remove wrapper

                data = json.loads(text)
                if len(data) > 1:
                    suggestions = [item[0] for item in data[1]]
                    self._suggestion_cache[query] = (suggestions, datetime.now())
                    return suggestions

            return []

        except Exception as e:
            logger.debug(f"YouTube suggestions fetch failed: {e}")
            return []

    async def get_suggestion_signal(self, topic: str) -> Optional[TrendSignal]:
        """Analyze YouTube suggestions for trend signals."""
        suggestions = await self.get_suggestions(topic)

        if not suggestions:
            return None

        # Analyze suggestion patterns
        suggestion_keywords = []
        for s in suggestions:
            words = s.lower().split()
            suggestion_keywords.extend(words)

        # Count common words (excluding the topic itself)
        word_counts = Counter(suggestion_keywords)
        topic_words = set(topic.lower().split())
        for tw in topic_words:
            if tw in word_counts:
                del word_counts[tw]

        # Check for trending indicators
        trending_words = ["2024", "2025", "2026", "new", "best", "latest", "top", "how to"]
        trending_count = sum(1 for w in word_counts if any(tw in w for tw in trending_words))

        # Calculate score
        score = min(100, len(suggestions) * 10 + trending_count * 15)

        # Check for velocity by looking at suggestions
        direction = TrendDirection.STABLE
        if any("new" in s.lower() or "2026" in s.lower() for s in suggestions):
            direction = TrendDirection.RISING

        return TrendSignal(
            source=SignalSource.YOUTUBE_SUGGEST,
            topic=topic,
            score=score,
            direction=direction,
            velocity=0,  # Hard to calculate for suggestions
            confidence=0.5,
            data_points=len(suggestions),
            metadata={
                "suggestions": suggestions[:10],
                "top_keywords": word_counts.most_common(5)
            }
        )


class SeasonalAnalyzer:
    """Analyze seasonal patterns and opportunities."""

    def __init__(self):
        self.events = SEASONAL_EVENTS

    def get_active_events(self, date: datetime = None) -> List[SeasonalEvent]:
        """Get currently active seasonal events."""
        date = date or datetime.now()
        return [e for e in self.events if e.is_active(date)]

    def get_upcoming_events(self, days_ahead: int = 30) -> List[Tuple[SeasonalEvent, int]]:
        """Get upcoming events within N days."""
        now = datetime.now()
        upcoming = []

        for event in self.events:
            days_until = event.days_until_peak(now)
            if 0 < days_until <= days_ahead:
                upcoming.append((event, days_until))

        upcoming.sort(key=lambda x: x[1])
        return upcoming

    def get_seasonal_signal(self, topic: str, niche: str) -> TrendSignal:
        """Get seasonal signal for a topic."""
        now = datetime.now()
        topic_lower = topic.lower()
        niche_lower = niche.lower()

        # Find matching events
        matching_events = []
        for event in self.events:
            if niche_lower in [n.lower() for n in event.boost_niches]:
                matching_events.append(event)
            elif any(t in topic_lower for t in event.boost_topics):
                matching_events.append(event)

        if not matching_events:
            return TrendSignal(
                source=SignalSource.SEASONAL,
                topic=topic,
                score=50,  # Neutral
                direction=TrendDirection.STABLE,
                velocity=0,
                confidence=0.7,
                data_points=len(self.events)
            )

        # Calculate seasonal score
        active_events = [e for e in matching_events if e.is_active(now)]
        upcoming_events = [(e, e.days_until_peak(now)) for e in matching_events
                         if not e.is_active(now) and e.days_until_peak(now) <= 45]

        if active_events:
            # During an active event
            best_event = max(active_events, key=lambda e: e.historical_lift)
            score = min(100, 60 + best_event.historical_lift * 20)
            direction = TrendDirection.SEASONAL_PEAK
            velocity = 30
        elif upcoming_events:
            # Event coming up
            best_event, days_until = min(upcoming_events, key=lambda x: x[1])
            proximity_bonus = max(0, 30 - days_until)  # More bonus closer to event
            score = min(100, 40 + proximity_bonus + best_event.historical_lift * 10)
            direction = TrendDirection.RISING
            velocity = proximity_bonus
        else:
            # Off-season
            score = 30
            direction = TrendDirection.STABLE
            velocity = 0
            best_event = matching_events[0] if matching_events else None

        return TrendSignal(
            source=SignalSource.SEASONAL,
            topic=topic,
            score=score,
            direction=direction,
            velocity=velocity,
            confidence=0.85,
            data_points=len(matching_events),
            metadata={
                "matching_events": [e.name for e in matching_events],
                "active_events": [e.name for e in active_events],
                "upcoming_events": [(e.name, d) for e, d in upcoming_events],
                "best_event_lift": best_event.historical_lift if best_event else 1.0
            }
        )


class TrendPredictor:
    """
    Main class for predictive topic selection.

    Combines multiple signals to score topics by viral potential
    and recommend optimal posting timing.
    """

    def __init__(self, db_path: str = "data/trend_predictor.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize analyzers
        self.google_trends = GoogleTrendsAnalyzer()
        self.reddit = RedditSignalAnalyzer()
        self.youtube = YouTubeSuggestAnalyzer()
        self.seasonal = SeasonalAnalyzer()

        # Signal weights for combining scores
        self.signal_weights = {
            SignalSource.GOOGLE_TRENDS: 0.35,
            SignalSource.REDDIT: 0.25,
            SignalSource.YOUTUBE_SUGGEST: 0.25,
            SignalSource.SEASONAL: 0.15,
        }

        self._init_database()
        logger.info("TrendPredictor initialized")

    def _init_database(self):
        """Initialize database for caching and history."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS topic_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    niche TEXT,
                    viral_score REAL,
                    trend_score REAL,
                    competition_score REAL,
                    seasonality_score REAL,
                    freshness_score REAL,
                    confidence REAL,
                    analyzed_at TEXT,
                    signals TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_topic_scores_topic
                ON topic_scores(topic, niche)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_topic_scores_score
                ON topic_scores(viral_score DESC)
            """)

    async def score_topic(
        self,
        topic: str,
        niche: str = "general",
        use_cache: bool = True,
        cache_hours: int = 6
    ) -> TopicScore:
        """
        Calculate viral potential score for a topic.

        Args:
            topic: Topic to analyze
            niche: Content niche for context
            use_cache: Whether to use cached results
            cache_hours: Cache validity in hours

        Returns:
            TopicScore with viral potential analysis
        """
        logger.info(f"Scoring topic: {topic} (niche: {niche})")

        # Check cache
        if use_cache:
            cached = self._get_cached_score(topic, niche, cache_hours)
            if cached:
                return cached

        # Gather signals from all sources
        signals = []

        # Google Trends
        gt_signal = await self.google_trends.get_trend_data(topic)
        if gt_signal:
            signals.append(gt_signal)

        # Reddit
        reddit_signal = await self.reddit.get_engagement_signal(topic)
        if reddit_signal:
            signals.append(reddit_signal)

        # YouTube Suggestions
        yt_signal = await self.youtube.get_suggestion_signal(topic)
        if yt_signal:
            signals.append(yt_signal)

        # Seasonal
        seasonal_signal = self.seasonal.get_seasonal_signal(topic, niche)
        signals.append(seasonal_signal)

        # Calculate component scores
        trend_score = self._calculate_trend_score(signals)
        competition_score = self._calculate_competition_score(topic, signals)
        seasonality_score = self._calculate_seasonality_score(signals)
        freshness_score = self._calculate_freshness_score(signals)

        # Calculate viral score (weighted average)
        viral_score = (
            trend_score * 0.35 +
            (100 - competition_score) * 0.25 +  # Invert: lower competition = higher score
            seasonality_score * 0.20 +
            freshness_score * 0.20
        )

        # Calculate confidence based on signals
        confidence = sum(s.confidence for s in signals) / len(signals) if signals else 0.3

        # Determine optimal posting time
        optimal_time, urgency = self._calculate_optimal_posting(signals, seasonal_signal)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            topic, niche, signals, viral_score, competition_score
        )

        score = TopicScore(
            topic=topic,
            niche=niche,
            viral_score=viral_score,
            trend_score=trend_score,
            competition_score=competition_score,
            seasonality_score=seasonality_score,
            freshness_score=freshness_score,
            signals=signals,
            optimal_posting_time=optimal_time,
            posting_urgency=urgency,
            confidence=confidence,
            recommendations=recommendations
        )

        # Cache result
        self._cache_score(score)

        return score

    def _calculate_trend_score(self, signals: List[TrendSignal]) -> float:
        """Calculate overall trend score from signals."""
        if not signals:
            return 50.0

        weighted_sum = 0
        weight_total = 0

        for signal in signals:
            weight = self.signal_weights.get(signal.source, 0.1)

            # Boost for rising/breakout trends
            direction_multiplier = {
                TrendDirection.BREAKOUT: 1.5,
                TrendDirection.RISING: 1.2,
                TrendDirection.SEASONAL_PEAK: 1.3,
                TrendDirection.STABLE: 1.0,
                TrendDirection.FALLING: 0.7,
            }.get(signal.direction, 1.0)

            adjusted_score = signal.score * direction_multiplier * signal.confidence
            weighted_sum += adjusted_score * weight
            weight_total += weight

        return min(100, weighted_sum / max(weight_total, 0.1))

    def _calculate_competition_score(
        self,
        topic: str,
        signals: List[TrendSignal]
    ) -> float:
        """
        Calculate competition score (higher = more competition).

        Based on:
        - Number of YouTube suggestions (more = saturated)
        - Reddit activity (high activity = competitive)
        """
        score = 50.0  # Baseline

        for signal in signals:
            if signal.source == SignalSource.YOUTUBE_SUGGEST:
                suggestions = signal.metadata.get("suggestions", [])
                # More suggestions = more competition
                score += len(suggestions) * 3

            elif signal.source == SignalSource.GOOGLE_TRENDS:
                # Very high interest = competitive
                if signal.metadata.get("current", 0) > 75:
                    score += 20

            elif signal.source == SignalSource.REDDIT:
                # High engagement = competitive
                if signal.metadata.get("avg_score", 0) > 1000:
                    score += 15

        return min(100, max(0, score))

    def _calculate_seasonality_score(self, signals: List[TrendSignal]) -> float:
        """Calculate seasonality bonus/penalty."""
        for signal in signals:
            if signal.source == SignalSource.SEASONAL:
                return signal.score
        return 50.0

    def _calculate_freshness_score(self, signals: List[TrendSignal]) -> float:
        """Calculate how 'fresh' or emerging a topic is."""
        freshness = 50.0

        for signal in signals:
            if signal.direction == TrendDirection.BREAKOUT:
                freshness += 30
            elif signal.direction == TrendDirection.RISING:
                freshness += 15

            # Check velocity
            if signal.velocity > 30:
                freshness += 20
            elif signal.velocity > 10:
                freshness += 10

        return min(100, freshness)

    def _calculate_optimal_posting(
        self,
        signals: List[TrendSignal],
        seasonal_signal: TrendSignal
    ) -> Tuple[Optional[datetime], str]:
        """Calculate optimal posting time and urgency."""
        now = datetime.now()

        # Check for breakout opportunity
        for signal in signals:
            if signal.direction == TrendDirection.BREAKOUT:
                # Post immediately for breakout trends
                return now + timedelta(hours=2), "immediate"

        # Check seasonal timing
        if seasonal_signal.direction == TrendDirection.SEASONAL_PEAK:
            return now + timedelta(days=1), "immediate"
        elif seasonal_signal.direction == TrendDirection.RISING:
            upcoming = seasonal_signal.metadata.get("upcoming_events", [])
            if upcoming:
                days_until = upcoming[0][1]
                if days_until <= 7:
                    return now + timedelta(days=1), "this_week"
                return now + timedelta(days=days_until - 3), "this_week"

        # Check general trend direction
        rising_count = sum(1 for s in signals if s.direction in
                         [TrendDirection.RISING, TrendDirection.BREAKOUT])

        if rising_count >= 2:
            return now + timedelta(days=2), "this_week"

        return now + timedelta(days=7), "flexible"

    def _generate_recommendations(
        self,
        topic: str,
        niche: str,
        signals: List[TrendSignal],
        viral_score: float,
        competition_score: float
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Score-based recommendations
        if viral_score >= 75:
            recommendations.append("HIGH POTENTIAL: Prioritize this topic for immediate production")
        elif viral_score >= 50:
            recommendations.append("GOOD POTENTIAL: Add to content calendar within 1-2 weeks")
        else:
            recommendations.append("LOWER PRIORITY: Consider combining with trending elements")

        # Competition recommendations
        if competition_score >= 70:
            recommendations.append("High competition - focus on unique angle or niche-down")
        elif competition_score <= 30:
            recommendations.append("Low competition - opportunity to establish authority")

        # Trend-specific recommendations
        for signal in signals:
            if signal.direction == TrendDirection.BREAKOUT:
                recommendations.append(f"BREAKING: {signal.source.value} shows breakout trend - act fast")
            elif signal.direction == TrendDirection.RISING:
                recommendations.append(f"Rising on {signal.source.value} - good timing window")

        # Seasonal recommendations
        seasonal = next((s for s in signals if s.source == SignalSource.SEASONAL), None)
        if seasonal:
            events = seasonal.metadata.get("upcoming_events", [])
            if events:
                recommendations.append(f"Seasonal opportunity: {events[0][0]} in {events[0][1]} days")

        return recommendations

    def _get_cached_score(
        self,
        topic: str,
        niche: str,
        cache_hours: int
    ) -> Optional[TopicScore]:
        """Get cached score if valid."""
        cutoff = datetime.now() - timedelta(hours=cache_hours)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("""
                SELECT * FROM topic_scores
                WHERE topic = ? AND niche = ? AND analyzed_at > ?
                ORDER BY analyzed_at DESC LIMIT 1
            """, (topic, niche, cutoff.isoformat())).fetchone()

            if row:
                # Reconstruct TopicScore (simplified - signals not fully reconstructed)
                return TopicScore(
                    topic=row["topic"],
                    niche=row["niche"],
                    viral_score=row["viral_score"],
                    trend_score=row["trend_score"],
                    competition_score=row["competition_score"],
                    seasonality_score=row["seasonality_score"],
                    freshness_score=row["freshness_score"],
                    signals=[],  # Signals not reconstructed from cache
                    optimal_posting_time=None,
                    posting_urgency="flexible",
                    confidence=row["confidence"],
                    analysis_time=datetime.fromisoformat(row["analyzed_at"])
                )

        return None

    def _cache_score(self, score: TopicScore):
        """Cache a topic score."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO topic_scores
                (topic, niche, viral_score, trend_score, competition_score,
                 seasonality_score, freshness_score, confidence, analyzed_at, signals)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                score.topic, score.niche, score.viral_score, score.trend_score,
                score.competition_score, score.seasonality_score, score.freshness_score,
                score.confidence, score.analysis_time.isoformat(),
                json.dumps([s.to_dict() for s in score.signals])
            ))

    async def predict_emerging_trends(
        self,
        niche: str,
        seed_topics: Optional[List[str]] = None,
        count: int = 10
    ) -> List[TopicScore]:
        """
        Predict emerging trends in a niche.

        Args:
            niche: Content niche
            seed_topics: Optional seed topics to expand from
            count: Number of trend predictions to return

        Returns:
            List of TopicScore objects sorted by potential
        """
        logger.info(f"Predicting emerging trends for niche: {niche}")

        # Default seed topics by niche
        default_seeds = {
            "finance": ["passive income", "investing", "side hustle", "budgeting", "retirement"],
            "psychology": ["manipulation", "persuasion", "habits", "mindset", "emotions"],
            "storytelling": ["true crime", "mystery", "history", "documentary", "conspiracy"],
        }

        topics_to_analyze = seed_topics or default_seeds.get(niche, ["trending"])

        # Expand topics with related queries
        expanded_topics = set(topics_to_analyze)
        for topic in topics_to_analyze[:3]:  # Limit API calls
            related = await self.google_trends.get_related_queries(topic)
            expanded_topics.update(related[:5])

        # Score all topics
        scored_topics = []
        for topic in list(expanded_topics)[:count * 2]:  # Analyze more than needed
            try:
                score = await self.score_topic(topic, niche)
                scored_topics.append(score)
            except Exception as e:
                logger.debug(f"Failed to score topic '{topic}': {e}")

            # Rate limiting
            await asyncio.sleep(0.5)

        # Sort by viral score and return top N
        scored_topics.sort(key=lambda s: s.viral_score, reverse=True)
        return scored_topics[:count]

    def get_seasonal_opportunities(
        self,
        months_ahead: int = 2,
        niches: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get upcoming seasonal opportunities.

        Args:
            months_ahead: How far ahead to look
            niches: Optional list of niches to filter for

        Returns:
            List of opportunity dicts with event info and recommendations
        """
        days_ahead = months_ahead * 30
        upcoming = self.seasonal.get_upcoming_events(days_ahead)

        opportunities = []
        for event, days_until in upcoming:
            # Filter by niche if specified
            if niches:
                if not any(n.lower() in [en.lower() for en in event.boost_niches] for n in niches):
                    continue

            opportunity = {
                "event": event.name,
                "days_until_peak": days_until,
                "peak_date": event.peak_date,
                "boost_niches": event.boost_niches,
                "boost_topics": event.boost_topics,
                "expected_lift": f"{(event.historical_lift - 1) * 100:.0f}%",
                "start_planning_by": (datetime.now() + timedelta(days=max(0, days_until - 14))).strftime("%Y-%m-%d"),
                "recommended_publish_date": (datetime.now() + timedelta(days=max(0, days_until - 3))).strftime("%Y-%m-%d"),
            }
            opportunities.append(opportunity)

        return opportunities

    async def rank_topics(
        self,
        topics: List[str],
        niche: str
    ) -> List[TopicScore]:
        """
        Rank multiple topics by viral potential.

        Args:
            topics: List of topics to rank
            niche: Content niche

        Returns:
            List of TopicScore objects sorted by score
        """
        scored = []
        for topic in topics:
            score = await self.score_topic(topic, niche)
            scored.append(score)
            await asyncio.sleep(0.3)  # Rate limiting

        scored.sort(key=lambda s: s.viral_score, reverse=True)
        return scored

    def get_trend_report(self, days: int = 7) -> Dict[str, Any]:
        """Get a trend analysis report from recent data."""
        cutoff = datetime.now() - timedelta(days=days)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Top scoring topics
            top_topics = conn.execute("""
                SELECT topic, niche, viral_score, analyzed_at
                FROM topic_scores
                WHERE analyzed_at > ?
                ORDER BY viral_score DESC
                LIMIT 20
            """, (cutoff.isoformat(),)).fetchall()

            # Score distribution
            score_dist = conn.execute("""
                SELECT
                    CASE
                        WHEN viral_score >= 75 THEN 'high'
                        WHEN viral_score >= 50 THEN 'medium'
                        ELSE 'low'
                    END as category,
                    COUNT(*) as count
                FROM topic_scores
                WHERE analyzed_at > ?
                GROUP BY category
            """, (cutoff.isoformat(),)).fetchall()

        return {
            "period_days": days,
            "top_topics": [dict(row) for row in top_topics],
            "score_distribution": {row["category"]: row["count"] for row in score_dist},
            "upcoming_seasonal": self.get_seasonal_opportunities(1),
            "generated_at": datetime.now().isoformat(),
        }


# Convenience functions
async def score_topic(topic: str, niche: str = "general") -> TopicScore:
    """Quick function to score a topic."""
    predictor = TrendPredictor()
    return await predictor.score_topic(topic, niche)


async def get_emerging_trends(niche: str) -> List[TopicScore]:
    """Quick function to get emerging trends."""
    predictor = TrendPredictor()
    return await predictor.predict_emerging_trends(niche)


if __name__ == "__main__":
    import sys

    async def main():
        print("\n" + "=" * 60)
        print("TREND PREDICTOR")
        print("=" * 60 + "\n")

        predictor = TrendPredictor()

        if len(sys.argv) > 1:
            command = sys.argv[1]

            if command == "score" and len(sys.argv) >= 3:
                topic = sys.argv[2]
                niche = sys.argv[3] if len(sys.argv) > 3 else "general"

                print(f"Scoring topic: {topic} ({niche})\n")
                score = await predictor.score_topic(topic, niche)

                print(f"VIRAL SCORE: {score.viral_score:.1f}/100")
                print(f"  Trend Score: {score.trend_score:.1f}")
                print(f"  Competition: {score.competition_score:.1f} ({'High' if score.competition_score > 60 else 'Low'})")
                print(f"  Seasonality: {score.seasonality_score:.1f}")
                print(f"  Freshness: {score.freshness_score:.1f}")
                print(f"  Confidence: {score.confidence:.0%}")
                print(f"\nPosting Urgency: {score.posting_urgency.upper()}")
                if score.optimal_posting_time:
                    print(f"Optimal Time: {score.optimal_posting_time.strftime('%Y-%m-%d')}")
                print("\nRecommendations:")
                for rec in score.recommendations:
                    print(f"  - {rec}")

            elif command == "trends" and len(sys.argv) >= 3:
                niche = sys.argv[2]
                print(f"Predicting emerging trends for: {niche}\n")

                trends = await predictor.predict_emerging_trends(niche, count=5)
                for i, t in enumerate(trends, 1):
                    print(f"{i}. {t.topic}")
                    print(f"   Score: {t.viral_score:.1f} | Urgency: {t.posting_urgency}")

            elif command == "seasonal":
                print("Upcoming Seasonal Opportunities:\n")
                opportunities = predictor.get_seasonal_opportunities(3)
                for opp in opportunities:
                    print(f"{opp['event']}")
                    print(f"  Days until peak: {opp['days_until_peak']}")
                    print(f"  Expected lift: {opp['expected_lift']}")
                    print(f"  Topics: {', '.join(opp['boost_topics'][:5])}")
                    print(f"  Start planning by: {opp['start_planning_by']}")
                    print()

            elif command == "report":
                print("Trend Report (Last 7 Days):\n")
                report = predictor.get_trend_report(7)
                print(f"Score Distribution: {report['score_distribution']}")
                print("\nTop Topics:")
                for t in report['top_topics'][:10]:
                    print(f"  {t['topic']} ({t['niche']}): {t['viral_score']:.1f}")
            else:
                print("Unknown command")
        else:
            print("Usage:")
            print("  python -m src.research.trend_predictor score <topic> [niche]")
            print("  python -m src.research.trend_predictor trends <niche>")
            print("  python -m src.research.trend_predictor seasonal")
            print("  python -m src.research.trend_predictor report")

    asyncio.run(main())
