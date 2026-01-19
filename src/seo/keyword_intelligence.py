"""
World-Class Keyword Intelligence System

Advanced keyword research, trend prediction, and competitive analysis
for YouTube SEO optimization.

Features:
- KeywordResearcher: Find low-competition, high-volume keywords
- TrendPredictor: Identify rising topics before they peak using statistical analysis
- CompetitorAnalyzer: Analyze top performers in niche with pattern recognition
- SearchIntentClassifier: Match content to search intent with ML-like scoring
- LongTailGenerator: Generate long-tail keyword variations algorithmically
- SeasonalityDetector: Identify cyclical trends using time series analysis

Usage:
    from src.seo.keyword_intelligence import KeywordIntelligence

    ki = KeywordIntelligence()

    # Research a keyword with full analysis
    result = ki.full_analysis("passive income", niche="finance")

    # Predict trending topics
    trends = ki.predict_trends("cryptocurrency", days_ahead=14)

    # Generate long-tail variations
    longtails = ki.generate_longtails("investing", count=50)
"""

import hashlib
import json
import math
import re
import sqlite3
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

from loguru import logger

try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False
    logger.warning("pytrends not installed. Install with: pip install pytrends")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not installed. Install with: pip install requests")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class KeywordMetrics:
    """Comprehensive keyword metrics."""
    keyword: str
    search_volume_score: float = 0.0  # 0-100 normalized score
    competition_score: float = 0.0  # 0-100 (lower is better for ranking)
    opportunity_score: float = 0.0  # 0-100 composite score
    trend_direction: str = "stable"  # rising, stable, declining
    trend_velocity: float = 0.0  # Rate of change
    seasonality_index: float = 0.0  # 0-1 seasonality strength
    search_intent: str = "informational"
    intent_confidence: float = 0.0
    difficulty_level: str = "medium"  # easy, medium, hard, very_hard
    estimated_monthly_searches: int = 0
    cpc_estimate: float = 0.0  # Cost per click indicator

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrendPrediction:
    """Trend prediction result."""
    keyword: str
    current_interest: float  # Current relative interest 0-100
    predicted_interest: float  # Predicted interest
    prediction_date: str
    confidence: float  # 0-1
    trend_type: str  # "breakout", "rising", "stable", "declining", "seasonal_peak"
    peak_timing: Optional[str] = None  # Estimated peak date if applicable
    supporting_signals: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CompetitorInsight:
    """Competitor analysis insight."""
    competitor_name: str
    estimated_views: int = 0
    title_pattern: str = ""
    keywords_used: List[str] = field(default_factory=list)
    posting_frequency: str = ""
    avg_video_length: str = ""
    engagement_indicators: Dict[str, float] = field(default_factory=dict)
    content_gaps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LongTailKeyword:
    """Long-tail keyword variation."""
    keyword: str
    parent_keyword: str
    word_count: int
    estimated_difficulty: float  # 0-100
    estimated_volume: str  # "high", "medium", "low", "very_low"
    intent_match: float  # 0-1 how well it matches search intent
    variation_type: str  # "question", "modifier", "location", "comparison", "how_to"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SeasonalPattern:
    """Seasonality analysis result."""
    keyword: str
    is_seasonal: bool
    seasonality_strength: float  # 0-1
    peak_months: List[int]  # 1-12
    trough_months: List[int]
    current_phase: str  # "peak", "rising", "trough", "declining"
    days_to_next_peak: int
    historical_pattern: List[float]  # 12 months of normalized interest

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# KEYWORD RESEARCHER
# ============================================================

class KeywordResearcher:
    """
    Find low-competition, high-volume keywords using multiple data sources.

    Combines YouTube autocomplete, Google Trends, and heuristic analysis
    to identify keyword opportunities.
    """

    # Keyword difficulty modifiers
    DIFFICULTY_MODIFIERS = {
        "high_competition_words": [
            "best", "top", "review", "tutorial", "how to", "guide",
            "tips", "2026", "2025", "free", "easy"
        ],
        "low_competition_signals": [
            "for beginners", "step by step", "complete guide",
            "explained simply", "without", "alternative"
        ],
        "high_value_intents": [
            "buy", "purchase", "price", "cost", "worth it",
            "vs", "comparison", "review"
        ]
    }

    def __init__(self, cache_dir: str = "data/seo_cache"):
        """Initialize keyword researcher with caching."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "keyword_research.db"
        self._init_database()

    def _init_database(self):
        """Initialize SQLite cache database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS keyword_cache (
                    keyword_hash TEXT PRIMARY KEY,
                    keyword TEXT,
                    metrics TEXT,
                    autocomplete TEXT,
                    created_at TEXT,
                    expires_at TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_keyword ON keyword_cache(keyword)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires ON keyword_cache(expires_at)
            """)

    def research(
        self,
        keyword: str,
        niche: str = "default",
        include_related: bool = True
    ) -> Tuple[KeywordMetrics, List[str]]:
        """
        Research a keyword and return metrics with related keywords.

        Args:
            keyword: Primary keyword to research
            niche: Content niche for context
            include_related: Whether to fetch related keywords

        Returns:
            Tuple of (KeywordMetrics, related_keywords)
        """
        logger.info(f"[KeywordResearcher] Researching: {keyword}")

        # Check cache first
        cached = self._get_cached(keyword, niche)
        if cached:
            logger.info(f"[KeywordResearcher] Using cached data for: {keyword}")
            return cached

        # Get autocomplete suggestions
        autocomplete = self._get_youtube_autocomplete(keyword)

        # Get trends data if available
        trends_data = self._get_trends_data(keyword) if PYTRENDS_AVAILABLE else {}

        # Calculate metrics
        metrics = self._calculate_metrics(keyword, niche, autocomplete, trends_data)

        # Get related keywords
        related = []
        if include_related:
            related = self._extract_related_keywords(keyword, autocomplete, trends_data)

        # Cache results
        self._cache_results(keyword, niche, metrics, autocomplete)

        logger.success(f"[KeywordResearcher] Research complete: {keyword}")
        return metrics, related

    def bulk_research(
        self,
        keywords: List[str],
        niche: str = "default"
    ) -> List[Tuple[KeywordMetrics, List[str]]]:
        """Research multiple keywords efficiently."""
        results = []
        for keyword in keywords:
            try:
                result = self.research(keyword, niche, include_related=False)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to research '{keyword}': {e}")
                # Return default metrics on failure
                results.append((KeywordMetrics(keyword=keyword), []))
        return results

    def find_opportunities(
        self,
        seed_keyword: str,
        niche: str = "default",
        min_opportunity_score: float = 60.0,
        max_results: int = 20
    ) -> List[KeywordMetrics]:
        """
        Find keyword opportunities starting from a seed keyword.

        Args:
            seed_keyword: Starting keyword
            niche: Content niche
            min_opportunity_score: Minimum opportunity score threshold
            max_results: Maximum number of results

        Returns:
            List of KeywordMetrics sorted by opportunity score
        """
        logger.info(f"[KeywordResearcher] Finding opportunities for: {seed_keyword}")

        opportunities = []

        # Research seed keyword
        seed_metrics, related = self.research(seed_keyword, niche)
        if seed_metrics.opportunity_score >= min_opportunity_score:
            opportunities.append(seed_metrics)

        # Research related keywords
        for related_kw in related[:30]:  # Limit to avoid rate limiting
            try:
                metrics, _ = self.research(related_kw, niche, include_related=False)
                if metrics.opportunity_score >= min_opportunity_score:
                    opportunities.append(metrics)
            except Exception as e:
                logger.debug(f"Skipping '{related_kw}': {e}")

        # Sort by opportunity score
        opportunities.sort(key=lambda m: m.opportunity_score, reverse=True)

        return opportunities[:max_results]

    def _get_youtube_autocomplete(self, keyword: str) -> List[str]:
        """Get YouTube search suggestions."""
        if not REQUESTS_AVAILABLE:
            return []

        suggestions = []

        try:
            # Primary autocomplete request
            url = "https://suggestqueries.google.com/complete/search"
            params = {
                "client": "youtube",
                "q": keyword,
                "ds": "yt",
            }

            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                text = response.text
                start = text.find("(") + 1
                end = text.rfind(")")
                if start > 0 and end > start:
                    data = json.loads(text[start:end])
                    if len(data) > 1 and isinstance(data[1], list):
                        suggestions = [item[0] for item in data[1] if isinstance(item, list)]

            # Add alphabetic expansions for more suggestions
            for letter in "abcdefghij":
                try:
                    params["q"] = f"{keyword} {letter}"
                    response = requests.get(url, params=params, timeout=3)
                    if response.status_code == 200:
                        text = response.text
                        start = text.find("(") + 1
                        end = text.rfind(")")
                        if start > 0 and end > start:
                            data = json.loads(text[start:end])
                            if len(data) > 1 and isinstance(data[1], list):
                                for item in data[1]:
                                    if isinstance(item, list) and item[0] not in suggestions:
                                        suggestions.append(item[0])
                except:
                    pass

        except Exception as e:
            logger.warning(f"Autocomplete fetch failed: {e}")

        return suggestions[:50]

    def _get_trends_data(self, keyword: str) -> Dict[str, Any]:
        """Get Google Trends data."""
        if not PYTRENDS_AVAILABLE:
            return {}

        try:
            pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25))
            pytrends.build_payload([keyword], cat=0, timeframe='today 12-m')

            # Get interest over time
            interest = pytrends.interest_over_time()
            trend_data = []
            if not interest.empty and keyword in interest.columns:
                trend_data = interest[keyword].tolist()

            # Get related queries
            related = pytrends.related_queries()
            related_list = []
            rising_list = []

            if keyword in related and related[keyword]:
                top = related[keyword].get('top')
                if top is not None and not top.empty:
                    related_list = top['query'].tolist()[:15]

                rising = related[keyword].get('rising')
                if rising is not None and not rising.empty:
                    rising_list = rising['query'].tolist()[:15]

            return {
                "trend": trend_data,
                "related": related_list,
                "rising": rising_list
            }

        except Exception as e:
            logger.debug(f"Trends API error for '{keyword}': {e}")
            return {}

    def _calculate_metrics(
        self,
        keyword: str,
        niche: str,
        autocomplete: List[str],
        trends_data: Dict[str, Any]
    ) -> KeywordMetrics:
        """Calculate comprehensive keyword metrics."""

        metrics = KeywordMetrics(keyword=keyword)

        # Word count analysis
        word_count = len(keyword.split())

        # Competition score (0-100, lower is better)
        competition = 50.0

        # Adjust for word count (longer = less competition)
        if word_count >= 5:
            competition -= 20
        elif word_count >= 4:
            competition -= 15
        elif word_count >= 3:
            competition -= 10
        elif word_count == 1:
            competition += 25

        # Check for high competition signals
        keyword_lower = keyword.lower()
        for word in self.DIFFICULTY_MODIFIERS["high_competition_words"]:
            if word in keyword_lower:
                competition += 8

        # Check for low competition signals
        for phrase in self.DIFFICULTY_MODIFIERS["low_competition_signals"]:
            if phrase in keyword_lower:
                competition -= 12

        # Adjust by autocomplete volume (more suggestions = more interest but more competition)
        autocomplete_count = len(autocomplete)
        if autocomplete_count > 40:
            competition += 15
            metrics.search_volume_score = 85
        elif autocomplete_count > 25:
            competition += 10
            metrics.search_volume_score = 70
        elif autocomplete_count > 15:
            competition += 5
            metrics.search_volume_score = 55
        elif autocomplete_count > 5:
            metrics.search_volume_score = 40
        else:
            competition -= 10
            metrics.search_volume_score = 25

        metrics.competition_score = max(0, min(100, competition))

        # Trend analysis
        trend = trends_data.get("trend", [])
        if trend and len(trend) >= 4:
            # Calculate trend direction and velocity
            recent = trend[-4:]  # Last month
            earlier = trend[:4]  # First month

            recent_avg = sum(recent) / len(recent)
            earlier_avg = sum(earlier) / len(earlier) if earlier else recent_avg

            if earlier_avg > 0:
                change_pct = ((recent_avg - earlier_avg) / earlier_avg) * 100
            else:
                change_pct = 0

            if change_pct > 30:
                metrics.trend_direction = "rising"
                metrics.trend_velocity = min(change_pct / 100, 1.0)
            elif change_pct < -30:
                metrics.trend_direction = "declining"
                metrics.trend_velocity = max(change_pct / 100, -1.0)
            else:
                metrics.trend_direction = "stable"
                metrics.trend_velocity = change_pct / 100

            # Estimate monthly searches from trend data
            if recent_avg > 0:
                # Scale based on trend interest (rough estimate)
                metrics.estimated_monthly_searches = int(recent_avg * 1000)

        # Calculate opportunity score
        # High volume + low competition + rising trend = high opportunity
        volume_factor = metrics.search_volume_score / 100
        competition_factor = (100 - metrics.competition_score) / 100
        trend_factor = 0.5 + (metrics.trend_velocity * 0.5) if metrics.trend_direction == "rising" else 0.5

        metrics.opportunity_score = (
            volume_factor * 35 +  # 35% weight on volume
            competition_factor * 45 +  # 45% weight on low competition
            trend_factor * 20  # 20% weight on trend
        )

        # Classify difficulty
        if metrics.competition_score < 30:
            metrics.difficulty_level = "easy"
        elif metrics.competition_score < 50:
            metrics.difficulty_level = "medium"
        elif metrics.competition_score < 70:
            metrics.difficulty_level = "hard"
        else:
            metrics.difficulty_level = "very_hard"

        # CPC estimate based on niche and intent
        niche_cpc = {
            "finance": 2.50,
            "psychology": 0.80,
            "storytelling": 0.60,
            "technology": 1.50,
            "health": 1.80,
            "default": 1.00
        }
        base_cpc = niche_cpc.get(niche, niche_cpc["default"])

        # Adjust CPC for commercial intent
        if any(word in keyword_lower for word in self.DIFFICULTY_MODIFIERS["high_value_intents"]):
            base_cpc *= 1.5

        metrics.cpc_estimate = round(base_cpc, 2)

        return metrics

    def _extract_related_keywords(
        self,
        keyword: str,
        autocomplete: List[str],
        trends_data: Dict[str, Any]
    ) -> List[str]:
        """Extract and deduplicate related keywords."""
        related = set()

        # Add autocomplete suggestions
        for suggestion in autocomplete:
            if suggestion.lower() != keyword.lower():
                related.add(suggestion)

        # Add trends related queries
        for query in trends_data.get("related", []):
            related.add(query)

        # Add rising queries (high potential)
        for query in trends_data.get("rising", []):
            related.add(query)

        return list(related)

    def _get_cached(
        self,
        keyword: str,
        niche: str
    ) -> Optional[Tuple[KeywordMetrics, List[str]]]:
        """Get cached research results if valid."""
        key_hash = hashlib.md5(f"{keyword}:{niche}".encode()).hexdigest()

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT metrics, autocomplete, expires_at FROM keyword_cache WHERE keyword_hash = ?",
                (key_hash,)
            ).fetchone()

        if row:
            expires_at = datetime.fromisoformat(row[2])
            if datetime.now() < expires_at:
                metrics_dict = json.loads(row[0])
                autocomplete = json.loads(row[1])
                return KeywordMetrics(**metrics_dict), autocomplete

        return None

    def _cache_results(
        self,
        keyword: str,
        niche: str,
        metrics: KeywordMetrics,
        autocomplete: List[str],
        ttl_days: int = 7
    ):
        """Cache research results."""
        key_hash = hashlib.md5(f"{keyword}:{niche}".encode()).hexdigest()
        expires_at = datetime.now() + timedelta(days=ttl_days)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO keyword_cache
                (keyword_hash, keyword, metrics, autocomplete, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                key_hash,
                keyword,
                json.dumps(metrics.to_dict()),
                json.dumps(autocomplete),
                datetime.now().isoformat(),
                expires_at.isoformat()
            ))


# ============================================================
# TREND PREDICTOR
# ============================================================

class TrendPredictor:
    """
    Identify rising topics before they peak using statistical analysis.

    Uses trend velocity, acceleration, and pattern matching to predict
    which topics are likely to gain popularity.
    """

    # Breakout detection thresholds
    BREAKOUT_THRESHOLD = 50  # % increase considered breakout
    RISING_THRESHOLD = 20   # % increase considered rising

    def __init__(self):
        """Initialize trend predictor."""
        self.pytrends = None
        if PYTRENDS_AVAILABLE:
            self.pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25))

    def predict(
        self,
        keyword: str,
        days_ahead: int = 14
    ) -> TrendPrediction:
        """
        Predict trend trajectory for a keyword.

        Args:
            keyword: Keyword to analyze
            days_ahead: Days ahead to predict

        Returns:
            TrendPrediction with analysis
        """
        logger.info(f"[TrendPredictor] Predicting trend for: {keyword}")

        prediction = TrendPrediction(
            keyword=keyword,
            current_interest=0,
            predicted_interest=0,
            prediction_date=(datetime.now() + timedelta(days=days_ahead)).isoformat(),
            confidence=0.5,
            trend_type="stable"
        )

        if not self.pytrends:
            logger.warning("pytrends not available for trend prediction")
            return prediction

        try:
            # Get historical trend data
            self.pytrends.build_payload([keyword], cat=0, timeframe='today 3-m')
            interest = self.pytrends.interest_over_time()

            if interest.empty or keyword not in interest.columns:
                return prediction

            trend_data = interest[keyword].tolist()

            if len(trend_data) < 10:
                return prediction

            # Calculate current metrics
            current_interest = trend_data[-1]
            prediction.current_interest = current_interest

            # Calculate trend velocity (first derivative)
            velocity = self._calculate_velocity(trend_data)

            # Calculate acceleration (second derivative)
            acceleration = self._calculate_acceleration(trend_data)

            # Predict future interest
            predicted = self._extrapolate(
                current_interest,
                velocity,
                acceleration,
                days_ahead
            )
            prediction.predicted_interest = max(0, min(100, predicted))

            # Classify trend type
            prediction.trend_type = self._classify_trend(
                current_interest,
                velocity,
                acceleration,
                trend_data
            )

            # Calculate confidence
            prediction.confidence = self._calculate_confidence(
                trend_data,
                velocity,
                acceleration
            )

            # Add supporting signals
            prediction.supporting_signals = self._get_supporting_signals(
                keyword,
                velocity,
                acceleration,
                trend_data
            )

            # Estimate peak timing for rising trends
            if prediction.trend_type in ["breakout", "rising"]:
                prediction.peak_timing = self._estimate_peak(
                    velocity,
                    acceleration,
                    days_ahead
                )

            logger.success(f"[TrendPredictor] Prediction: {prediction.trend_type}")

        except Exception as e:
            logger.warning(f"Trend prediction failed: {e}")

        return prediction

    def find_breakout_topics(
        self,
        seed_keywords: List[str],
        niche: str = "default"
    ) -> List[TrendPrediction]:
        """
        Find topics showing breakout potential.

        Args:
            seed_keywords: List of seed keywords to analyze
            niche: Content niche

        Returns:
            List of TrendPredictions for breakout topics
        """
        logger.info(f"[TrendPredictor] Scanning for breakout topics in {niche}")

        breakouts = []

        for keyword in seed_keywords:
            prediction = self.predict(keyword)
            if prediction.trend_type in ["breakout", "rising"]:
                breakouts.append(prediction)

        # Sort by predicted growth
        breakouts.sort(
            key=lambda p: p.predicted_interest - p.current_interest,
            reverse=True
        )

        return breakouts

    def compare_trends(
        self,
        keywords: List[str]
    ) -> Dict[str, TrendPrediction]:
        """Compare trend trajectories for multiple keywords."""
        results = {}
        for keyword in keywords[:5]:  # Limit to avoid rate limiting
            results[keyword] = self.predict(keyword)
        return results

    def _calculate_velocity(self, data: List[float]) -> float:
        """Calculate trend velocity (rate of change)."""
        if len(data) < 2:
            return 0.0

        # Use weighted average of recent changes
        recent = data[-7:]  # Last week
        if len(recent) < 2:
            return 0.0

        changes = [recent[i] - recent[i-1] for i in range(1, len(recent))]

        # Weight more recent changes higher
        weights = list(range(1, len(changes) + 1))
        weighted_sum = sum(c * w for c, w in zip(changes, weights))
        weight_total = sum(weights)

        return weighted_sum / weight_total if weight_total > 0 else 0.0

    def _calculate_acceleration(self, data: List[float]) -> float:
        """Calculate trend acceleration (change in velocity)."""
        if len(data) < 14:
            return 0.0

        # Compare recent velocity to earlier velocity
        recent = data[-7:]
        earlier = data[-14:-7]

        recent_velocity = self._calculate_velocity(recent)
        earlier_velocity = self._calculate_velocity(earlier)

        return recent_velocity - earlier_velocity

    def _extrapolate(
        self,
        current: float,
        velocity: float,
        acceleration: float,
        days: int
    ) -> float:
        """Extrapolate future interest using physics-like model."""
        # Use kinematic equation: x = x0 + v*t + 0.5*a*t^2
        # But dampen acceleration effect over time
        damping = 0.5  # Reduce acceleration impact

        predicted = (
            current +
            velocity * days +
            0.5 * acceleration * damping * (days ** 0.8)  # Sublinear acceleration
        )

        return predicted

    def _classify_trend(
        self,
        current: float,
        velocity: float,
        acceleration: float,
        data: List[float]
    ) -> str:
        """Classify the trend type."""
        # Calculate percentage change over recent period
        if len(data) >= 7:
            week_ago = data[-7]
            if week_ago > 0:
                week_change = ((current - week_ago) / week_ago) * 100
            else:
                week_change = 0
        else:
            week_change = 0

        # Check for breakout (rapid recent growth)
        if week_change >= self.BREAKOUT_THRESHOLD and acceleration > 0:
            return "breakout"

        # Check for rising trend
        if velocity > 1 and week_change >= self.RISING_THRESHOLD:
            return "rising"

        # Check for declining trend
        if velocity < -1 and week_change < -10:
            return "declining"

        # Check for seasonal peak pattern
        if current > 70 and acceleration < 0:
            return "seasonal_peak"

        return "stable"

    def _calculate_confidence(
        self,
        data: List[float],
        velocity: float,
        acceleration: float
    ) -> float:
        """Calculate prediction confidence (0-1)."""
        if len(data) < 10:
            return 0.3

        # Factors that increase confidence
        confidence = 0.5

        # More data = higher confidence
        if len(data) >= 52:  # Full year
            confidence += 0.15
        elif len(data) >= 26:  # 6 months
            confidence += 0.10

        # Consistent trend direction
        recent = data[-14:]
        if len(recent) >= 7:
            increasing = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i-1])
            consistency = increasing / (len(recent) - 1)

            if consistency > 0.7 or consistency < 0.3:  # Consistent direction
                confidence += 0.15

        # Low volatility
        if len(data) >= 7:
            try:
                std_dev = statistics.stdev(data[-14:])
                mean_val = statistics.mean(data[-14:])
                if mean_val > 0:
                    cv = std_dev / mean_val  # Coefficient of variation
                    if cv < 0.3:
                        confidence += 0.1
            except:
                pass

        return min(0.95, confidence)

    def _get_supporting_signals(
        self,
        keyword: str,
        velocity: float,
        acceleration: float,
        data: List[float]
    ) -> List[str]:
        """Get supporting signals for the prediction."""
        signals = []

        if velocity > 2:
            signals.append("Strong upward momentum")
        elif velocity > 0.5:
            signals.append("Moderate growth trend")
        elif velocity < -2:
            signals.append("Strong downward pressure")

        if acceleration > 1:
            signals.append("Accelerating growth")
        elif acceleration < -1:
            signals.append("Decelerating/reversing")

        # Check for recent spike
        if len(data) >= 7:
            recent_max = max(data[-7:])
            recent_avg = sum(data[-7:]) / 7
            if recent_max > recent_avg * 1.5:
                signals.append("Recent interest spike detected")

        # Check historical highs
        if len(data) >= 30:
            all_time_max = max(data)
            current = data[-1]
            if current > all_time_max * 0.9:
                signals.append("Near all-time high interest")

        return signals

    def _estimate_peak(
        self,
        velocity: float,
        acceleration: float,
        max_days: int
    ) -> str:
        """Estimate when the trend will peak."""
        if acceleration >= 0:
            # Still accelerating, peak not imminent
            return f">{max_days} days (still accelerating)"

        # Using v = v0 + at, solve for t when v = 0
        if acceleration != 0:
            days_to_peak = -velocity / acceleration
            if 0 < days_to_peak <= max_days:
                peak_date = datetime.now() + timedelta(days=int(days_to_peak))
                return peak_date.strftime("%Y-%m-%d")

        return "Unknown"


# ============================================================
# COMPETITOR ANALYZER
# ============================================================

class CompetitorAnalyzer:
    """
    Analyze top performers in niche with pattern recognition.

    Identifies successful content patterns, title structures,
    and content gaps for competitive advantage.
    """

    # Known successful patterns by niche
    NICHE_PATTERNS = {
        "finance": {
            "title_patterns": [
                "{number} {topic} That Will {benefit}",
                "How {person} Made ${amount} {timeframe}",
                "The {adjective} {topic} Nobody Talks About",
                "Why {percent}% of People Fail at {topic}",
                "I Tried {topic} for {time} - Here's What Happened"
            ],
            "content_signals": ["case study", "numbers", "proof", "results"],
            "avg_length_minutes": (10, 15),
            "top_creators": ["Graham Stephan", "Andrei Jikh", "Mark Tilbury"]
        },
        "psychology": {
            "title_patterns": [
                "{number} Signs of {condition}",
                "The Psychology Behind {behavior}",
                "Why Your Brain {action}",
                "Dark {topic} Tactics You Need to Know",
                "How to {skill} Using Psychology"
            ],
            "content_signals": ["research", "studies", "experts", "science"],
            "avg_length_minutes": (8, 12),
            "top_creators": ["Psych2Go", "Practical Psychology", "Charisma on Command"]
        },
        "storytelling": {
            "title_patterns": [
                "The Untold Story of {subject}",
                "What Really Happened to {subject}",
                "The {adjective} Case of {subject}",
                "How {subject} {achievement}",
                "The Rise and Fall of {subject}"
            ],
            "content_signals": ["timeline", "narrative", "exclusive", "investigation"],
            "avg_length_minutes": (15, 30),
            "top_creators": ["Veritasium", "Johnny Harris", "Wendover Productions"]
        }
    }

    def __init__(self, data_dir: str = "data/competitor_data"):
        """Initialize competitor analyzer."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def analyze(
        self,
        keyword: str,
        niche: str = "default"
    ) -> CompetitorInsight:
        """
        Analyze competition for a keyword.

        Args:
            keyword: Keyword to analyze
            niche: Content niche

        Returns:
            CompetitorInsight with analysis
        """
        logger.info(f"[CompetitorAnalyzer] Analyzing competition for: {keyword}")

        niche_data = self.NICHE_PATTERNS.get(niche, self.NICHE_PATTERNS.get("finance", {}))

        insight = CompetitorInsight(
            competitor_name=f"Top {niche} creators",
            title_pattern=niche_data.get("title_patterns", [""])[0],
            keywords_used=self._extract_niche_keywords(keyword, niche),
            posting_frequency=self._estimate_posting_frequency(niche),
            avg_video_length=f"{niche_data.get('avg_length_minutes', (10, 15))[0]}-{niche_data.get('avg_length_minutes', (10, 15))[1]} minutes",
            content_gaps=self._identify_content_gaps(keyword, niche),
            engagement_indicators=self._estimate_engagement_metrics(niche)
        )

        logger.success(f"[CompetitorAnalyzer] Analysis complete for: {keyword}")
        return insight

    def get_title_templates(self, niche: str = "default") -> List[str]:
        """Get proven title templates for a niche."""
        niche_data = self.NICHE_PATTERNS.get(niche, self.NICHE_PATTERNS.get("finance", {}))
        return niche_data.get("title_patterns", [])

    def identify_gaps(
        self,
        keyword: str,
        niche: str = "default",
        count: int = 10
    ) -> List[str]:
        """Identify content gaps for a keyword."""
        return self._identify_content_gaps(keyword, niche)[:count]

    def get_top_creators(self, niche: str = "default") -> List[str]:
        """Get top creators in a niche."""
        niche_data = self.NICHE_PATTERNS.get(niche, {})
        return niche_data.get("top_creators", [])

    def _extract_niche_keywords(self, keyword: str, niche: str) -> List[str]:
        """Extract relevant keywords for the niche."""
        keywords = [keyword]

        # Add niche-specific modifiers
        modifiers = {
            "finance": ["money", "investing", "wealth", "income", "budget", "savings"],
            "psychology": ["mind", "brain", "behavior", "personality", "emotions", "mental"],
            "storytelling": ["story", "history", "investigation", "documentary", "case"]
        }

        niche_mods = modifiers.get(niche, [])
        for mod in niche_mods[:5]:
            keywords.append(f"{keyword} {mod}")

        return keywords

    def _estimate_posting_frequency(self, niche: str) -> str:
        """Estimate optimal posting frequency for niche."""
        frequencies = {
            "finance": "2-3 videos per week",
            "psychology": "2-3 videos per week",
            "storytelling": "1-2 videos per week (longer content)"
        }
        return frequencies.get(niche, "2-3 videos per week")

    def _identify_content_gaps(self, keyword: str, niche: str) -> List[str]:
        """Identify content gaps and opportunities."""
        gaps = []

        # Universal gaps
        gaps.extend([
            f"Beginner's complete guide to {keyword}",
            f"Advanced {keyword} strategies for 2026",
            f"Common {keyword} mistakes and how to avoid them",
            f"{keyword} case studies with real results",
            f"{keyword} vs alternatives comparison"
        ])

        # Niche-specific gaps
        niche_gaps = {
            "finance": [
                f"Low-risk {keyword} for beginners",
                f"{keyword} during recession/inflation",
                f"{keyword} tax implications explained"
            ],
            "psychology": [
                f"Scientific research on {keyword}",
                f"Practical {keyword} exercises",
                f"{keyword} in relationships"
            ],
            "storytelling": [
                f"The untold story of {keyword}",
                f"Latest developments in {keyword}",
                f"{keyword} documentary deep dive"
            ]
        }

        gaps.extend(niche_gaps.get(niche, []))

        return gaps

    def _estimate_engagement_metrics(self, niche: str) -> Dict[str, float]:
        """Estimate typical engagement metrics for niche."""
        metrics = {
            "finance": {
                "avg_ctr": 5.5,
                "avg_retention": 45.0,
                "avg_like_ratio": 4.5,
                "comment_rate": 0.5
            },
            "psychology": {
                "avg_ctr": 6.0,
                "avg_retention": 48.0,
                "avg_like_ratio": 5.0,
                "comment_rate": 0.6
            },
            "storytelling": {
                "avg_ctr": 5.0,
                "avg_retention": 55.0,
                "avg_like_ratio": 4.0,
                "comment_rate": 0.4
            }
        }
        return metrics.get(niche, metrics["finance"])


# ============================================================
# SEARCH INTENT CLASSIFIER
# ============================================================

class SearchIntentClassifier:
    """
    Match content to search intent with ML-like scoring.

    Classifies queries into intent categories and provides
    content recommendations for each intent type.
    """

    # Intent classification patterns
    INTENT_SIGNALS = {
        "informational": {
            "keywords": [
                "how to", "what is", "why", "when", "where", "who",
                "guide", "tutorial", "learn", "understand", "explained",
                "definition", "meaning", "ways to", "tips", "steps"
            ],
            "weight": 1.0,
            "content_format": "educational_tutorial",
            "optimal_length": (8, 15),
            "cta_timing": [0.3, 0.6, 0.95]
        },
        "navigational": {
            "keywords": [
                "login", "sign in", "website", "official", "download",
                "app", "channel", "account", "page"
            ],
            "weight": 0.8,
            "content_format": "direct_resource",
            "optimal_length": (3, 8),
            "cta_timing": [0.1, 0.5]
        },
        "transactional": {
            "keywords": [
                "buy", "purchase", "order", "deal", "discount", "coupon",
                "cheap", "price", "cost", "free", "sale", "offer"
            ],
            "weight": 1.2,
            "content_format": "product_showcase",
            "optimal_length": (5, 12),
            "cta_timing": [0.2, 0.5, 0.8, 0.95]
        },
        "commercial": {
            "keywords": [
                "best", "top", "review", "comparison", "vs", "versus",
                "alternative", "which", "should i", "worth it", "honest"
            ],
            "weight": 1.1,
            "content_format": "comparison_review",
            "optimal_length": (10, 20),
            "cta_timing": [0.3, 0.7, 0.95]
        }
    }

    # Content recommendations by intent
    CONTENT_RECOMMENDATIONS = {
        "informational": [
            "Use step-by-step structure with clear progression",
            "Include timestamps for easy navigation",
            "Start with the problem, end with the solution",
            "Add practical examples and demonstrations",
            "Include a quick summary at the end"
        ],
        "navigational": [
            "Get to the point quickly",
            "Include direct links in description",
            "Show the exact process/location",
            "Keep it concise and focused"
        ],
        "transactional": [
            "Lead with benefits and results",
            "Include social proof and testimonials",
            "Show before/after transformations",
            "Add clear call-to-action",
            "Include pricing/value discussion"
        ],
        "commercial": [
            "Present objective comparison criteria",
            "Show real-world testing results",
            "Discuss pros AND cons honestly",
            "Give a clear recommendation",
            "Include affiliate disclosure if applicable"
        ]
    }

    def classify(self, query: str) -> Tuple[str, float, Dict[str, Any]]:
        """
        Classify search intent for a query.

        Args:
            query: Search query to classify

        Returns:
            Tuple of (intent_type, confidence, metadata)
        """
        logger.info(f"[SearchIntentClassifier] Classifying: {query}")

        query_lower = query.lower()
        scores = defaultdict(float)

        # Score each intent type
        for intent, config in self.INTENT_SIGNALS.items():
            weight = config["weight"]

            for keyword in config["keywords"]:
                if keyword in query_lower:
                    # Base score
                    base_score = 10 * weight

                    # Bonus for keyword at start
                    if query_lower.startswith(keyword):
                        base_score *= 1.5

                    # Bonus for exact phrase match
                    if f" {keyword} " in f" {query_lower} ":
                        base_score *= 1.2

                    scores[intent] += base_score

        # Default to informational if no strong signal
        if not scores or max(scores.values()) < 5:
            primary_intent = "informational"
            confidence = 0.5
        else:
            primary_intent = max(scores, key=scores.get)
            total_score = sum(scores.values())
            confidence = min(0.95, scores[primary_intent] / max(total_score, 1) + 0.3)

        # Build metadata
        intent_config = self.INTENT_SIGNALS[primary_intent]
        metadata = {
            "content_format": intent_config["content_format"],
            "optimal_length_minutes": intent_config["optimal_length"],
            "cta_timing": intent_config["cta_timing"],
            "recommendations": self.CONTENT_RECOMMENDATIONS[primary_intent],
            "all_scores": dict(scores)
        }

        logger.success(f"[SearchIntentClassifier] Intent: {primary_intent} ({confidence:.0%})")
        return primary_intent, confidence, metadata

    def get_content_strategy(self, intent: str) -> Dict[str, Any]:
        """Get content strategy for an intent type."""
        config = self.INTENT_SIGNALS.get(intent, self.INTENT_SIGNALS["informational"])

        return {
            "intent": intent,
            "content_format": config["content_format"],
            "optimal_length": config["optimal_length"],
            "cta_placement": config["cta_timing"],
            "recommendations": self.CONTENT_RECOMMENDATIONS.get(intent, []),
            "title_strategy": self._get_title_strategy(intent)
        }

    def _get_title_strategy(self, intent: str) -> List[str]:
        """Get title strategy for intent type."""
        strategies = {
            "informational": [
                "How to {topic} (Complete Guide)",
                "{topic} Explained for Beginners",
                "{number} Steps to {achieve goal}",
                "The Ultimate {topic} Tutorial"
            ],
            "navigational": [
                "{brand} {feature} Tutorial",
                "How to Use {brand} in {year}",
                "{brand} Complete Walkthrough"
            ],
            "transactional": [
                "Is {product} Worth It? (Honest Review)",
                "{product} Review After {time}",
                "Why I Bought {product}"
            ],
            "commercial": [
                "Best {category} in {year}",
                "{product} vs {competitor}: Which Wins?",
                "Top {number} {category} Compared"
            ]
        }
        return strategies.get(intent, strategies["informational"])


# ============================================================
# LONG TAIL GENERATOR
# ============================================================

class LongTailGenerator:
    """
    Generate long-tail keyword variations algorithmically.

    Creates targeted, specific keyword variations with lower
    competition and higher conversion potential.
    """

    # Modifiers for generating variations
    MODIFIERS = {
        "question": [
            "how to", "what is", "why does", "when to", "where to",
            "who can", "which", "can you", "should i", "is it"
        ],
        "intent": [
            "for beginners", "for professionals", "for students",
            "for business", "for personal use", "at home", "online"
        ],
        "time": [
            "in 2026", "this year", "quickly", "in 5 minutes",
            "overnight", "step by step", "fast", "easily"
        ],
        "comparison": [
            "vs", "versus", "or", "compared to", "alternative to",
            "better than", "instead of", "like"
        ],
        "qualifier": [
            "best", "top", "ultimate", "complete", "proven",
            "simple", "advanced", "free", "cheap", "affordable"
        ],
        "problem": [
            "without", "even if", "despite", "when", "while",
            "before", "after", "during"
        ],
        "result": [
            "to make money", "to save time", "to grow",
            "to succeed", "to improve", "to master", "to learn"
        ]
    }

    def generate(
        self,
        seed_keyword: str,
        count: int = 50,
        include_questions: bool = True
    ) -> List[LongTailKeyword]:
        """
        Generate long-tail keyword variations.

        Args:
            seed_keyword: Base keyword
            count: Number of variations to generate
            include_questions: Include question-based variations

        Returns:
            List of LongTailKeyword objects
        """
        logger.info(f"[LongTailGenerator] Generating variations for: {seed_keyword}")

        variations = []
        seen = set()

        # Generate variations using each modifier type
        for var_type, modifiers in self.MODIFIERS.items():
            if var_type == "question" and not include_questions:
                continue

            for modifier in modifiers:
                # Prefix variation
                prefix_var = f"{modifier} {seed_keyword}"
                if prefix_var.lower() not in seen:
                    seen.add(prefix_var.lower())
                    variations.append(self._create_variation(
                        prefix_var,
                        seed_keyword,
                        var_type
                    ))

                # Suffix variation
                suffix_var = f"{seed_keyword} {modifier}"
                if suffix_var.lower() not in seen:
                    seen.add(suffix_var.lower())
                    variations.append(self._create_variation(
                        suffix_var,
                        seed_keyword,
                        var_type
                    ))

                if len(variations) >= count * 2:
                    break

        # Generate compound variations
        compound_variations = self._generate_compound_variations(
            seed_keyword,
            count // 4
        )
        for var in compound_variations:
            if var.keyword.lower() not in seen:
                seen.add(var.keyword.lower())
                variations.append(var)

        # Score and sort by estimated difficulty (easier first)
        variations.sort(key=lambda v: v.estimated_difficulty)

        logger.success(f"[LongTailGenerator] Generated {len(variations[:count])} variations")
        return variations[:count]

    def generate_questions(
        self,
        seed_keyword: str,
        count: int = 20
    ) -> List[LongTailKeyword]:
        """Generate question-based long-tail keywords."""
        questions = []

        question_templates = [
            f"how to {seed_keyword}",
            f"what is {seed_keyword}",
            f"why is {seed_keyword} important",
            f"when should i {seed_keyword}",
            f"where can i {seed_keyword}",
            f"who should {seed_keyword}",
            f"which {seed_keyword} is best",
            f"can you {seed_keyword}",
            f"should i {seed_keyword}",
            f"is {seed_keyword} worth it",
            f"does {seed_keyword} work",
            f"how long does {seed_keyword} take",
            f"how much does {seed_keyword} cost",
            f"what are the benefits of {seed_keyword}",
            f"what are the risks of {seed_keyword}",
            f"how do i start {seed_keyword}",
            f"what is the best way to {seed_keyword}",
            f"why should i {seed_keyword}",
            f"how often should i {seed_keyword}",
            f"what happens if i {seed_keyword}"
        ]

        for template in question_templates[:count]:
            questions.append(self._create_variation(
                template,
                seed_keyword,
                "question"
            ))

        return questions

    def _create_variation(
        self,
        keyword: str,
        parent: str,
        var_type: str
    ) -> LongTailKeyword:
        """Create a LongTailKeyword object."""
        word_count = len(keyword.split())

        # Estimate difficulty (longer = easier)
        base_difficulty = 60
        if word_count >= 5:
            base_difficulty -= 25
        elif word_count >= 4:
            base_difficulty -= 15
        elif word_count >= 3:
            base_difficulty -= 5

        # Questions are often easier
        if var_type == "question":
            base_difficulty -= 10

        # Estimate volume based on word count
        if word_count <= 2:
            volume = "medium"
        elif word_count <= 4:
            volume = "low"
        else:
            volume = "very_low"

        # Calculate intent match
        intent_scores = {
            "question": 0.9,
            "intent": 0.85,
            "time": 0.75,
            "comparison": 0.8,
            "qualifier": 0.7,
            "problem": 0.85,
            "result": 0.9
        }

        return LongTailKeyword(
            keyword=keyword,
            parent_keyword=parent,
            word_count=word_count,
            estimated_difficulty=max(10, min(90, base_difficulty)),
            estimated_volume=volume,
            intent_match=intent_scores.get(var_type, 0.7),
            variation_type=var_type
        )

    def _generate_compound_variations(
        self,
        seed: str,
        count: int
    ) -> List[LongTailKeyword]:
        """Generate compound variations combining multiple modifiers."""
        compounds = []

        # Combine question + intent
        for q in self.MODIFIERS["question"][:3]:
            for i in self.MODIFIERS["intent"][:3]:
                var = f"{q} {seed} {i}"
                compounds.append(self._create_variation(var, seed, "question"))

        # Combine qualifier + time
        for qual in self.MODIFIERS["qualifier"][:3]:
            for time in self.MODIFIERS["time"][:3]:
                var = f"{qual} {seed} {time}"
                compounds.append(self._create_variation(var, seed, "qualifier"))

        return compounds[:count]


# ============================================================
# SEASONALITY DETECTOR
# ============================================================

class SeasonalityDetector:
    """
    Identify cyclical trends using time series analysis.

    Detects seasonal patterns in search interest to optimize
    content timing and capitalize on predictable interest spikes.
    """

    # Known seasonal topics
    KNOWN_SEASONAL = {
        "tax": {"peak_months": [1, 2, 3, 4], "category": "annual"},
        "christmas": {"peak_months": [11, 12], "category": "holiday"},
        "halloween": {"peak_months": [9, 10], "category": "holiday"},
        "summer": {"peak_months": [5, 6, 7, 8], "category": "seasonal"},
        "winter": {"peak_months": [11, 12, 1, 2], "category": "seasonal"},
        "new year": {"peak_months": [12, 1], "category": "holiday"},
        "black friday": {"peak_months": [11], "category": "shopping"},
        "back to school": {"peak_months": [7, 8, 9], "category": "annual"},
        "valentine": {"peak_months": [1, 2], "category": "holiday"},
        "mother's day": {"peak_months": [4, 5], "category": "holiday"},
        "father's day": {"peak_months": [5, 6], "category": "holiday"}
    }

    def __init__(self):
        """Initialize seasonality detector."""
        self.pytrends = None
        if PYTRENDS_AVAILABLE:
            self.pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25))

    def detect(self, keyword: str) -> SeasonalPattern:
        """
        Detect seasonality pattern for a keyword.

        Args:
            keyword: Keyword to analyze

        Returns:
            SeasonalPattern with analysis
        """
        logger.info(f"[SeasonalityDetector] Analyzing seasonality for: {keyword}")

        # Check for known seasonal patterns first
        keyword_lower = keyword.lower()
        for seasonal_term, config in self.KNOWN_SEASONAL.items():
            if seasonal_term in keyword_lower:
                return self._create_known_pattern(keyword, config)

        # Analyze using trends data
        if not self.pytrends:
            return SeasonalPattern(
                keyword=keyword,
                is_seasonal=False,
                seasonality_strength=0.0,
                peak_months=[],
                trough_months=[],
                current_phase="unknown",
                days_to_next_peak=0,
                historical_pattern=[]
            )

        try:
            # Get 5 years of data for seasonality detection
            self.pytrends.build_payload([keyword], cat=0, timeframe='today 5-y')
            interest = self.pytrends.interest_over_time()

            if interest.empty or keyword not in interest.columns:
                return self._create_default_pattern(keyword)

            data = interest[keyword].tolist()

            # Analyze monthly patterns
            monthly_data = self._aggregate_monthly(data)

            # Detect seasonality
            is_seasonal, strength = self._calculate_seasonality_strength(monthly_data)

            # Find peaks and troughs
            peak_months, trough_months = self._find_peaks_troughs(monthly_data)

            # Determine current phase
            current_month = datetime.now().month
            current_phase = self._determine_phase(current_month, peak_months, trough_months)

            # Calculate days to next peak
            days_to_peak = self._calculate_days_to_peak(current_month, peak_months)

            pattern = SeasonalPattern(
                keyword=keyword,
                is_seasonal=is_seasonal,
                seasonality_strength=strength,
                peak_months=peak_months,
                trough_months=trough_months,
                current_phase=current_phase,
                days_to_next_peak=days_to_peak,
                historical_pattern=monthly_data
            )

            logger.success(f"[SeasonalityDetector] Seasonal: {is_seasonal}, Strength: {strength:.2f}")
            return pattern

        except Exception as e:
            logger.warning(f"Seasonality detection failed: {e}")
            return self._create_default_pattern(keyword)

    def get_optimal_timing(self, keyword: str) -> Dict[str, Any]:
        """Get optimal content timing based on seasonality."""
        pattern = self.detect(keyword)

        recommendations = []

        if pattern.is_seasonal:
            # Recommend publishing before peak
            if pattern.peak_months:
                pre_peak_month = pattern.peak_months[0] - 1
                if pre_peak_month < 1:
                    pre_peak_month = 12
                recommendations.append(
                    f"Publish {4-6} weeks before peak (Month {pre_peak_month})"
                )

            if pattern.current_phase == "rising":
                recommendations.append("Now is a good time - interest is rising")
            elif pattern.current_phase == "peak":
                recommendations.append("Currently at peak - capitalize quickly")
            elif pattern.current_phase == "declining":
                recommendations.append("Interest declining - wait for next cycle")
            elif pattern.current_phase == "trough":
                recommendations.append("Low interest period - plan for upcoming peak")
        else:
            recommendations.append("No strong seasonality - publish any time")
            recommendations.append("Focus on trend momentum instead")

        return {
            "pattern": pattern.to_dict(),
            "recommendations": recommendations,
            "days_to_optimal": pattern.days_to_next_peak if pattern.is_seasonal else 0
        }

    def _aggregate_monthly(self, weekly_data: List[float]) -> List[float]:
        """Aggregate weekly data into monthly averages."""
        if len(weekly_data) < 52:
            return [50.0] * 12  # Not enough data

        # Assume ~4.33 weeks per month
        monthly = []
        weeks_per_month = len(weekly_data) / 12

        for month in range(12):
            start_idx = int(month * weeks_per_month)
            end_idx = int((month + 1) * weeks_per_month)
            month_data = weekly_data[start_idx:end_idx]
            if month_data:
                monthly.append(sum(month_data) / len(month_data))
            else:
                monthly.append(50.0)

        return monthly

    def _calculate_seasonality_strength(
        self,
        monthly_data: List[float]
    ) -> Tuple[bool, float]:
        """Calculate seasonality strength (0-1)."""
        if not monthly_data or len(monthly_data) < 12:
            return False, 0.0

        try:
            mean_val = statistics.mean(monthly_data)
            std_val = statistics.stdev(monthly_data)

            if mean_val == 0:
                return False, 0.0

            # Coefficient of variation
            cv = std_val / mean_val

            # Normalize to 0-1 scale
            strength = min(1.0, cv * 2)  # cv of 0.5 = strength of 1.0

            # Consider seasonal if strength > 0.3
            is_seasonal = strength > 0.3

            return is_seasonal, strength

        except:
            return False, 0.0

    def _find_peaks_troughs(
        self,
        monthly_data: List[float]
    ) -> Tuple[List[int], List[int]]:
        """Find peak and trough months."""
        if not monthly_data:
            return [], []

        mean_val = statistics.mean(monthly_data)

        peaks = []
        troughs = []

        for i, value in enumerate(monthly_data):
            month = i + 1  # 1-indexed months

            if value > mean_val * 1.2:  # 20% above average
                peaks.append(month)
            elif value < mean_val * 0.8:  # 20% below average
                troughs.append(month)

        return peaks, troughs

    def _determine_phase(
        self,
        current_month: int,
        peaks: List[int],
        troughs: List[int]
    ) -> str:
        """Determine current phase in seasonal cycle."""
        if current_month in peaks:
            return "peak"
        if current_month in troughs:
            return "trough"

        # Check if approaching peak
        for peak in peaks:
            if peak > current_month and peak - current_month <= 2:
                return "rising"
            if peak < current_month and current_month - peak <= 2:
                return "declining"

        return "stable"

    def _calculate_days_to_peak(
        self,
        current_month: int,
        peaks: List[int]
    ) -> int:
        """Calculate days until next peak."""
        if not peaks:
            return 0

        # Find next peak
        future_peaks = [p for p in peaks if p > current_month]
        if not future_peaks:
            # Wrap to next year
            next_peak = peaks[0] + 12
        else:
            next_peak = future_peaks[0]

        months_to_peak = next_peak - current_month
        if months_to_peak < 0:
            months_to_peak += 12

        return months_to_peak * 30  # Approximate days

    def _create_known_pattern(
        self,
        keyword: str,
        config: Dict
    ) -> SeasonalPattern:
        """Create pattern for known seasonal topic."""
        peaks = config["peak_months"]
        current_month = datetime.now().month

        # Calculate troughs (opposite of peaks)
        all_months = set(range(1, 13))
        peak_set = set(peaks)
        troughs = list(all_months - peak_set)[:len(peaks)]

        return SeasonalPattern(
            keyword=keyword,
            is_seasonal=True,
            seasonality_strength=0.85,
            peak_months=peaks,
            trough_months=troughs,
            current_phase=self._determine_phase(current_month, peaks, troughs),
            days_to_next_peak=self._calculate_days_to_peak(current_month, peaks),
            historical_pattern=self._generate_pattern_from_peaks(peaks)
        )

    def _create_default_pattern(self, keyword: str) -> SeasonalPattern:
        """Create default non-seasonal pattern."""
        return SeasonalPattern(
            keyword=keyword,
            is_seasonal=False,
            seasonality_strength=0.0,
            peak_months=[],
            trough_months=[],
            current_phase="stable",
            days_to_next_peak=0,
            historical_pattern=[50.0] * 12
        )

    def _generate_pattern_from_peaks(self, peaks: List[int]) -> List[float]:
        """Generate normalized pattern from known peaks."""
        pattern = [30.0] * 12  # Base level

        for peak in peaks:
            idx = peak - 1  # 0-indexed
            pattern[idx] = 100.0

            # Add gradual increase/decrease around peak
            if idx > 0:
                pattern[idx - 1] = 70.0
            if idx < 11:
                pattern[idx + 1] = 70.0

        return pattern


# ============================================================
# MAIN KEYWORD INTELLIGENCE CLASS
# ============================================================

class KeywordIntelligence:
    """
    Main interface for keyword intelligence system.

    Coordinates all sub-components for comprehensive keyword analysis.
    """

    def __init__(self, cache_dir: str = "data/seo_cache"):
        """Initialize keyword intelligence system."""
        self.researcher = KeywordResearcher(cache_dir)
        self.trend_predictor = TrendPredictor()
        self.competitor_analyzer = CompetitorAnalyzer()
        self.intent_classifier = SearchIntentClassifier()
        self.longtail_generator = LongTailGenerator()
        self.seasonality_detector = SeasonalityDetector()

        logger.info("[KeywordIntelligence] System initialized")

    def full_analysis(
        self,
        keyword: str,
        niche: str = "default"
    ) -> Dict[str, Any]:
        """
        Perform full keyword analysis.

        Args:
            keyword: Keyword to analyze
            niche: Content niche

        Returns:
            Comprehensive analysis results
        """
        logger.info(f"[KeywordIntelligence] Full analysis for: {keyword}")

        # Research keyword
        metrics, related = self.researcher.research(keyword, niche)

        # Predict trend
        trend = self.trend_predictor.predict(keyword)

        # Classify intent
        intent, confidence, intent_meta = self.intent_classifier.classify(keyword)

        # Detect seasonality
        seasonality = self.seasonality_detector.detect(keyword)

        # Analyze competition
        competition = self.competitor_analyzer.analyze(keyword, niche)

        # Generate long-tails
        longtails = self.longtail_generator.generate(keyword, count=10)

        # Update metrics with additional data
        metrics.search_intent = intent
        metrics.intent_confidence = confidence
        metrics.seasonality_index = seasonality.seasonality_strength

        return {
            "keyword": keyword,
            "niche": niche,
            "metrics": metrics.to_dict(),
            "trend_prediction": trend.to_dict(),
            "seasonality": seasonality.to_dict(),
            "competition": competition.to_dict(),
            "intent": {
                "type": intent,
                "confidence": confidence,
                "recommendations": intent_meta["recommendations"]
            },
            "related_keywords": related[:20],
            "longtail_opportunities": [lt.to_dict() for lt in longtails],
            "analysis_timestamp": datetime.now().isoformat()
        }

    def predict_trends(
        self,
        keyword: str,
        days_ahead: int = 14
    ) -> TrendPrediction:
        """Predict trend trajectory."""
        return self.trend_predictor.predict(keyword, days_ahead)

    def find_opportunities(
        self,
        seed_keyword: str,
        niche: str = "default",
        count: int = 20
    ) -> List[KeywordMetrics]:
        """Find keyword opportunities."""
        return self.researcher.find_opportunities(
            seed_keyword, niche,
            min_opportunity_score=50.0,
            max_results=count
        )

    def generate_longtails(
        self,
        keyword: str,
        count: int = 50
    ) -> List[LongTailKeyword]:
        """Generate long-tail variations."""
        return self.longtail_generator.generate(keyword, count)

    def get_content_timing(self, keyword: str) -> Dict[str, Any]:
        """Get optimal content timing based on seasonality."""
        return self.seasonality_detector.get_optimal_timing(keyword)

    def compare_keywords(
        self,
        keywords: List[str],
        niche: str = "default"
    ) -> List[Dict[str, Any]]:
        """Compare multiple keywords."""
        results = []

        for keyword in keywords:
            metrics, _ = self.researcher.research(keyword, niche, include_related=False)
            trend = self.trend_predictor.predict(keyword)

            results.append({
                "keyword": keyword,
                "opportunity_score": metrics.opportunity_score,
                "competition_score": metrics.competition_score,
                "trend_type": trend.trend_type,
                "predicted_growth": trend.predicted_interest - trend.current_interest
            })

        # Sort by opportunity
        results.sort(key=lambda x: x["opportunity_score"], reverse=True)

        return results


# CLI entry point
def main():
    """CLI entry point for keyword intelligence."""
    import sys

    if len(sys.argv) < 2:
        print("""
Keyword Intelligence System - World-Class SEO Research

Commands:
    research <keyword> [--niche <niche>]
        Full keyword research with metrics

    trend <keyword> [--days <days>]
        Predict trend trajectory

    opportunities <seed> [--niche <niche>] [--count <n>]
        Find keyword opportunities

    longtails <keyword> [--count <n>]
        Generate long-tail variations

    seasonality <keyword>
        Detect seasonal patterns

    compare <kw1> <kw2> [<kw3>...] [--niche <niche>]
        Compare multiple keywords

Examples:
    python -m src.seo.keyword_intelligence research "passive income" --niche finance
    python -m src.seo.keyword_intelligence trend "cryptocurrency" --days 30
    python -m src.seo.keyword_intelligence opportunities "investing" --count 20
    python -m src.seo.keyword_intelligence longtails "make money" --count 50
        """)
        return

    ki = KeywordIntelligence()
    cmd = sys.argv[1]

    # Parse arguments
    args = sys.argv[2:]
    kwargs = {}
    positional = []

    i = 0
    while i < len(args):
        if args[i].startswith("--"):
            key = args[i][2:]
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                value = args[i + 1]
                try:
                    value = int(value)
                except ValueError:
                    pass
                kwargs[key] = value
                i += 2
            else:
                kwargs[key] = True
                i += 1
        else:
            positional.append(args[i])
            i += 1

    if cmd == "research" and positional:
        result = ki.full_analysis(positional[0], kwargs.get("niche", "default"))
        print(json.dumps(result, indent=2))

    elif cmd == "trend" and positional:
        result = ki.predict_trends(positional[0], kwargs.get("days", 14))
        print(json.dumps(result.to_dict(), indent=2))

    elif cmd == "opportunities" and positional:
        results = ki.find_opportunities(
            positional[0],
            kwargs.get("niche", "default"),
            kwargs.get("count", 20)
        )
        for r in results:
            print(f"{r.keyword}: Opportunity={r.opportunity_score:.0f}, Competition={r.competition_score:.0f}")

    elif cmd == "longtails" and positional:
        results = ki.generate_longtails(positional[0], kwargs.get("count", 50))
        for r in results[:20]:
            print(f"{r.keyword} (Difficulty: {r.estimated_difficulty:.0f}, Type: {r.variation_type})")

    elif cmd == "seasonality" and positional:
        result = ki.seasonality_detector.detect(positional[0])
        print(json.dumps(result.to_dict(), indent=2))

    elif cmd == "compare" and positional:
        results = ki.compare_keywords(positional, kwargs.get("niche", "default"))
        for r in results:
            print(f"{r['keyword']}: Opp={r['opportunity_score']:.0f}, Trend={r['trend_type']}")

    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
