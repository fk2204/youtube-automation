"""
World-Class SEO Strategist Agent

A comprehensive SEO strategist that acts as a content architect and
search-intent analyst for YouTube optimization.

Features:
- Keyword research using pytrends + YouTube autocomplete
- Search intent classification and optimization
- Competitor intelligence and gap analysis
- CTR and retention prediction
- A/B testing framework for titles
- SEO-driven content calendar

Usage:
    from src.agents.seo_strategist import SEOStrategist

    strategist = SEOStrategist()

    # Research a keyword
    research = strategist.research_keyword("passive income", niche="finance")

    # Full optimization before upload
    result = strategist.full_optimization(title, description, tags, niche)

    # Get content strategy
    strategy = strategist.content_strategy("psychology", topics=10)
"""

import os
import json
import re
import sqlite3
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from urllib.parse import quote_plus
import random
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


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class KeywordData:
    """Data structure for keyword research results."""
    keyword: str
    search_volume_trend: List[float] = field(default_factory=list)
    related_queries: List[str] = field(default_factory=list)
    rising_queries: List[str] = field(default_factory=list)
    difficulty_score: float = 0.0
    opportunity_score: float = 0.0
    competition_level: str = "medium"  # low, medium, high
    suggested_keywords: List[str] = field(default_factory=list)
    youtube_autocomplete: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SearchIntent:
    """Search intent classification result."""
    query: str
    primary_intent: str  # informational, navigational, transactional, commercial
    confidence: float
    secondary_intent: Optional[str] = None
    user_goal: str = ""
    content_recommendations: List[str] = field(default_factory=list)
    title_templates: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CompetitorVideo:
    """Data about a competitor video."""
    title: str
    channel: str = ""
    views: int = 0
    likes: int = 0
    duration: str = ""
    published: str = ""
    description_preview: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CompetitorReport:
    """Report from competitor analysis."""
    keyword: str
    top_videos: List[CompetitorVideo] = field(default_factory=list)
    common_title_patterns: List[str] = field(default_factory=list)
    avg_title_length: float = 0.0
    common_keywords: List[str] = field(default_factory=list)
    content_gaps: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CTRPrediction:
    """Click-through rate prediction."""
    title: str
    predicted_ctr: float  # 0-100 scale
    confidence: float  # 0-1 scale
    factors: Dict[str, float] = field(default_factory=dict)
    improvements: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RetentionPrediction:
    """Audience retention prediction."""
    estimated_avg_view_duration: float  # percentage
    hook_strength: float  # 0-100
    pacing_score: float  # 0-100
    engagement_points: List[Tuple[int, str]] = field(default_factory=list)
    drop_off_risks: List[Tuple[int, str]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TitleVariant:
    """A/B test title variant."""
    title: str
    variant_type: str  # curiosity, urgency, how-to, listicle, etc.
    predicted_ctr: float
    score: float
    rationale: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TopicSuggestion:
    """Content topic suggestion."""
    topic: str
    keyword: str
    opportunity_score: float
    search_volume: str  # high, medium, low
    competition: str  # high, medium, low
    content_angle: str = ""
    title_suggestion: str = ""
    rationale: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SEOStrategyResult:
    """Result from SEO strategy operations."""
    success: bool
    operation: str
    data: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    tokens_used: int = 0
    cached: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return asdict(self)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"SEO Strategy: {self.operation}",
            f"Status: {'Success' if self.success else 'Failed'}",
            ""
        ]
        if self.recommendations:
            lines.append("Recommendations:")
            for rec in self.recommendations[:5]:
                lines.append(f"  - {rec}")
        return "\n".join(lines)


# ============================================================
# KEYWORD RESEARCHER
# ============================================================

class KeywordResearcher:
    """
    Research keywords using pytrends + YouTube autocomplete.

    Token cost: ZERO - Uses free APIs only
    """

    def __init__(self, cache_db: str = None):
        """Initialize keyword researcher with optional cache."""
        self.cache_db = cache_db or "src/agents/seo_data/keyword_cache.db"
        self._init_cache()

    def _init_cache(self):
        """Initialize SQLite cache for keyword data."""
        cache_path = Path(self.cache_db)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(cache_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS keyword_cache (
                keyword_hash TEXT PRIMARY KEY,
                keyword TEXT,
                data TEXT,
                created_at TEXT,
                expires_at TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _get_cached(self, keyword: str, niche: str) -> Optional[KeywordData]:
        """Get cached keyword data if not expired."""
        key_hash = hashlib.md5(f"{keyword}:{niche}".encode()).hexdigest()

        conn = sqlite3.connect(str(self.cache_db))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT data, expires_at FROM keyword_cache WHERE keyword_hash = ?",
            (key_hash,)
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            expires_at = datetime.fromisoformat(row[1])
            if datetime.now() < expires_at:
                data = json.loads(row[0])
                return KeywordData(**data)

        return None

    def _set_cached(self, keyword: str, niche: str, data: KeywordData, ttl_days: int = 7):
        """Cache keyword data."""
        key_hash = hashlib.md5(f"{keyword}:{niche}".encode()).hexdigest()
        expires_at = datetime.now() + timedelta(days=ttl_days)

        conn = sqlite3.connect(str(self.cache_db))
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO keyword_cache (keyword_hash, keyword, data, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?)
        """, (key_hash, keyword, json.dumps(data.to_dict()), datetime.now().isoformat(), expires_at.isoformat()))
        conn.commit()
        conn.close()

    def research_keyword(self, keyword: str, niche: str = "default") -> KeywordData:
        """
        Research a keyword using pytrends and YouTube autocomplete.

        Args:
            keyword: Keyword to research
            niche: Content niche for context

        Returns:
            KeywordData with research results
        """
        logger.info(f"[KeywordResearcher] Researching: {keyword}")

        # Check cache first
        cached = self._get_cached(keyword, niche)
        if cached:
            logger.info(f"[KeywordResearcher] Using cached data for: {keyword}")
            return cached

        data = KeywordData(keyword=keyword)

        # Get YouTube autocomplete suggestions
        autocomplete = self.get_youtube_autocomplete(keyword)
        data.youtube_autocomplete = autocomplete
        data.suggested_keywords = autocomplete[:10]

        # Get trends data if pytrends available
        if PYTRENDS_AVAILABLE:
            try:
                trends_data = self._get_trends_data(keyword)
                data.search_volume_trend = trends_data.get("trend", [])
                data.related_queries = trends_data.get("related", [])
                data.rising_queries = trends_data.get("rising", [])
            except Exception as e:
                logger.warning(f"Failed to get trends data: {e}")

        # Calculate difficulty and opportunity scores
        data.difficulty_score = self.calculate_keyword_difficulty(keyword, data)
        data.opportunity_score = self._calculate_opportunity(data)
        data.competition_level = self._classify_competition(data.difficulty_score)

        # Cache the result
        self._set_cached(keyword, niche, data)

        logger.success(f"[KeywordResearcher] Research complete for: {keyword}")
        return data

    def get_youtube_autocomplete(self, seed: str) -> List[str]:
        """
        Get YouTube search suggestions for keyword expansion.

        Args:
            seed: Seed keyword

        Returns:
            List of autocomplete suggestions
        """
        if not REQUESTS_AVAILABLE:
            return []

        suggestions = []

        try:
            # YouTube autocomplete API
            url = f"https://suggestqueries.google.com/complete/search"
            params = {
                "client": "youtube",
                "q": seed,
                "ds": "yt",
            }

            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                # Parse JSONP response
                text = response.text
                start = text.find("(") + 1
                end = text.rfind(")")
                if start > 0 and end > start:
                    data = json.loads(text[start:end])
                    if len(data) > 1 and isinstance(data[1], list):
                        suggestions = [item[0] for item in data[1] if isinstance(item, list)]

        except Exception as e:
            logger.warning(f"YouTube autocomplete failed: {e}")

        # Also try with common prefixes
        prefixes = ["how to", "why", "what is", "best", "top"]
        for prefix in prefixes[:2]:  # Limit to avoid too many requests
            try:
                url = f"https://suggestqueries.google.com/complete/search"
                params = {
                    "client": "youtube",
                    "q": f"{prefix} {seed}",
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
                            for item in data[1]:
                                if isinstance(item, list) and item[0] not in suggestions:
                                    suggestions.append(item[0])
            except:
                pass

        return suggestions[:20]

    def _get_trends_data(self, keyword: str) -> Dict[str, Any]:
        """Get Google Trends data for keyword."""
        if not PYTRENDS_AVAILABLE:
            return {}

        try:
            pytrends = TrendReq(hl='en-US', tz=360)
            pytrends.build_payload([keyword], cat=0, timeframe='today 3-m')

            # Get interest over time
            interest = pytrends.interest_over_time()
            trend = []
            if not interest.empty and keyword in interest.columns:
                trend = interest[keyword].tolist()

            # Get related queries
            related = pytrends.related_queries()
            related_list = []
            rising_list = []

            if keyword in related and related[keyword]:
                top = related[keyword].get('top')
                if top is not None and not top.empty:
                    related_list = top['query'].tolist()[:10]

                rising = related[keyword].get('rising')
                if rising is not None and not rising.empty:
                    rising_list = rising['query'].tolist()[:10]

            return {
                "trend": trend,
                "related": related_list,
                "rising": rising_list
            }

        except Exception as e:
            logger.warning(f"Trends API error: {e}")
            return {}

    def calculate_keyword_difficulty(self, keyword: str, data: KeywordData = None) -> float:
        """
        Calculate keyword difficulty score (0-100).

        Factors:
        - Word count (longer = easier)
        - Competition indicators
        - Trend direction

        Args:
            keyword: The keyword
            data: Optional KeywordData for additional signals

        Returns:
            Difficulty score 0-100
        """
        score = 50.0  # Base score

        # Longer keywords are typically easier to rank for
        word_count = len(keyword.split())
        if word_count >= 4:
            score -= 15
        elif word_count >= 3:
            score -= 10
        elif word_count <= 1:
            score += 20

        # Check for high-competition indicators
        high_comp_words = ["best", "top", "review", "tutorial", "how to"]
        for word in high_comp_words:
            if word in keyword.lower():
                score += 10
                break

        # Low competition indicators (niche terms)
        if data and data.youtube_autocomplete:
            # More autocomplete = more competition
            score += min(len(data.youtube_autocomplete) * 2, 20)

        # Trend direction (if available)
        if data and data.search_volume_trend:
            trend = data.search_volume_trend
            if len(trend) >= 2:
                # Rising trend = more competition coming
                if trend[-1] > trend[0]:
                    score += 5

        return max(0, min(100, score))

    def _calculate_opportunity(self, data: KeywordData) -> float:
        """Calculate opportunity score (0-100)."""
        # High opportunity = high interest, low difficulty
        base = 100 - data.difficulty_score

        # Boost for rising queries
        if data.rising_queries:
            base += 10

        # Boost for many autocomplete suggestions (interest exists)
        if len(data.youtube_autocomplete) > 10:
            base += 5

        return max(0, min(100, base))

    def _classify_competition(self, difficulty: float) -> str:
        """Classify competition level."""
        if difficulty < 40:
            return "low"
        elif difficulty < 70:
            return "medium"
        else:
            return "high"


# ============================================================
# SEARCH INTENT ANALYZER
# ============================================================

class SearchIntentAnalyzer:
    """
    Classify and optimize for search intent.

    Token cost: ZERO - Rule-based classification
    """

    INTENT_TYPES = ["informational", "navigational", "transactional", "commercial"]

    # Intent patterns
    INTENT_PATTERNS = {
        "informational": {
            "keywords": ["how to", "what is", "why", "guide", "tutorial", "learn", "explained",
                        "tips", "ways to", "understanding", "definition", "meaning"],
            "questions": ["what", "how", "why", "when", "who", "where"],
        },
        "navigational": {
            "keywords": ["login", "sign in", "website", "official", "download", "app"],
            "patterns": [r"\b(youtube|netflix|amazon|google)\b"],
        },
        "transactional": {
            "keywords": ["buy", "purchase", "order", "deal", "discount", "coupon",
                        "cheap", "price", "cost", "subscription", "free trial"],
        },
        "commercial": {
            "keywords": ["best", "top", "review", "comparison", "vs", "versus",
                        "alternative", "which", "should i", "worth it"],
        }
    }

    # Content recommendations by intent
    CONTENT_RECOMMENDATIONS = {
        "informational": [
            "Use step-by-step structure",
            "Include visual demonstrations",
            "Define key terms early",
            "Add timestamps for navigation",
            "Include practical examples"
        ],
        "navigational": [
            "Clear brand mention in title",
            "Direct link in description",
            "Official branding elements"
        ],
        "transactional": [
            "Include pricing information",
            "Show before/after results",
            "Add call-to-action",
            "Include testimonials"
        ],
        "commercial": [
            "Compare multiple options",
            "Use pros/cons format",
            "Include personal recommendation",
            "Show real-world testing"
        ]
    }

    # Title templates by intent
    TITLE_TEMPLATES = {
        "informational": [
            "How to {topic} (Complete Guide {year})",
            "{topic} Explained: Everything You Need to Know",
            "{number} Ways to {topic} (Step by Step)",
            "The Ultimate Guide to {topic}",
            "{topic} Tutorial for Beginners"
        ],
        "navigational": [
            "{brand} Official Tutorial",
            "How to Use {brand} ({year})",
            "{brand} Complete Walkthrough"
        ],
        "transactional": [
            "Is {topic} Worth It? (Honest Review)",
            "{topic} Review: My Experience After {time}",
            "Why I {action} {topic} (And You Should Too)"
        ],
        "commercial": [
            "Best {topic} in {year} (Top {number} Compared)",
            "{topic} vs {alternative}: Which is Better?",
            "{number} Best {topic} for {audience}",
            "The Truth About {topic} (Honest Review)"
        ]
    }

    def classify_intent(self, query: str) -> SearchIntent:
        """
        Determine search intent from query.

        Args:
            query: Search query to classify

        Returns:
            SearchIntent with classification and recommendations
        """
        logger.info(f"[SearchIntentAnalyzer] Classifying: {query}")

        query_lower = query.lower()
        scores = {intent: 0 for intent in self.INTENT_TYPES}

        # Score each intent type
        for intent, patterns in self.INTENT_PATTERNS.items():
            # Check keywords
            for kw in patterns.get("keywords", []):
                if kw in query_lower:
                    scores[intent] += 2

            # Check question words
            for qw in patterns.get("questions", []):
                if query_lower.startswith(qw) or f" {qw} " in query_lower:
                    scores[intent] += 1

            # Check regex patterns
            for pattern in patterns.get("patterns", []):
                if re.search(pattern, query_lower):
                    scores[intent] += 2

        # Determine primary intent
        max_score = max(scores.values())
        if max_score == 0:
            primary_intent = "informational"  # Default
            confidence = 0.5
        else:
            primary_intent = max(scores, key=scores.get)
            confidence = min(max_score / 6, 1.0)

        # Find secondary intent
        sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        secondary_intent = None
        if len(sorted_intents) > 1 and sorted_intents[1][1] > 0:
            secondary_intent = sorted_intents[1][0]

        # Determine user goal
        user_goal = self._determine_user_goal(query, primary_intent)

        # Get recommendations
        recommendations = self.CONTENT_RECOMMENDATIONS.get(primary_intent, [])
        templates = self.TITLE_TEMPLATES.get(primary_intent, [])

        intent = SearchIntent(
            query=query,
            primary_intent=primary_intent,
            confidence=confidence,
            secondary_intent=secondary_intent,
            user_goal=user_goal,
            content_recommendations=recommendations,
            title_templates=templates
        )

        logger.success(f"[SearchIntentAnalyzer] Intent: {primary_intent} ({confidence:.0%})")
        return intent

    def _determine_user_goal(self, query: str, intent: str) -> str:
        """Determine what the user wants to achieve."""
        goals = {
            "informational": "Learn or understand something",
            "navigational": "Find a specific page or resource",
            "transactional": "Complete a purchase or action",
            "commercial": "Research options before deciding"
        }
        return goals.get(intent, "Unknown")

    def get_intent_optimizations(self, intent: SearchIntent) -> List[str]:
        """
        Get specific optimizations for an intent type.

        Args:
            intent: SearchIntent object

        Returns:
            List of optimization recommendations
        """
        optimizations = []

        # Base recommendations
        optimizations.extend(self.CONTENT_RECOMMENDATIONS.get(intent.primary_intent, []))

        # Intent-specific additions
        if intent.primary_intent == "informational":
            optimizations.extend([
                "Front-load key information in first 30 seconds",
                "Use numbered lists for clarity",
                "Include a clear conclusion/summary"
            ])
        elif intent.primary_intent == "commercial":
            optimizations.extend([
                "Show actual usage footage",
                "Be transparent about affiliate relationships",
                "Include pricing comparison"
            ])

        return optimizations


# ============================================================
# COMPETITOR ANALYZER
# ============================================================

class CompetitorAnalyzer:
    """
    Monitor and learn from competitor content.

    Token cost: ZERO - Web scraping only
    """

    def __init__(self, data_file: str = None):
        """Initialize with optional data storage file."""
        self.data_file = data_file or "src/agents/seo_data/competitor_data.json"
        Path(self.data_file).parent.mkdir(parents=True, exist_ok=True)

    def analyze_top_results(self, keyword: str, count: int = 10) -> CompetitorReport:
        """
        Analyze top YouTube search results for a keyword.

        Note: This uses simulated data as actual YouTube scraping
        requires API access. In production, integrate YouTube Data API.

        Args:
            keyword: Keyword to search
            count: Number of results to analyze

        Returns:
            CompetitorReport with analysis
        """
        logger.info(f"[CompetitorAnalyzer] Analyzing top {count} results for: {keyword}")

        # In a real implementation, this would use YouTube Data API
        # For now, we generate pattern-based insights

        report = CompetitorReport(keyword=keyword)

        # Analyze title patterns based on niche detection
        patterns = self._detect_title_patterns(keyword)
        report.common_title_patterns = patterns

        # Generate content gap analysis
        gaps = self._identify_content_gaps(keyword)
        report.content_gaps = gaps

        # Generate recommendations
        report.recommendations = self._generate_recommendations(keyword, patterns, gaps)

        # Estimate average title length for this niche
        report.avg_title_length = self._estimate_title_length(keyword)

        # Common keywords
        report.common_keywords = self._extract_common_keywords(keyword)

        logger.success(f"[CompetitorAnalyzer] Analysis complete for: {keyword}")
        return report

    def _detect_title_patterns(self, keyword: str) -> List[str]:
        """Detect common title patterns for a keyword."""
        keyword_lower = keyword.lower()

        patterns = []

        # Finance patterns
        if any(w in keyword_lower for w in ["money", "invest", "income", "wealth", "stock"]):
            patterns = [
                "[Number] [Keyword] That [Benefit]",
                "How [Person] Made $[Amount] [Timeframe]",
                "The [Adjective] [Keyword] Nobody Talks About",
                "Why [Percentage]% of People [Fail/Succeed] at [Keyword]"
            ]

        # Psychology patterns
        elif any(w in keyword_lower for w in ["psychology", "mind", "brain", "behavior", "manipulation"]):
            patterns = [
                "[Number] Signs of [Condition/Type]",
                "The Psychology of [Topic]",
                "Why Your Brain [Action]",
                "Dark [Keyword] Tactics [Group] Uses"
            ]

        # Storytelling patterns
        elif any(w in keyword_lower for w in ["story", "true", "mystery", "crime", "untold"]):
            patterns = [
                "The Untold Story of [Subject]",
                "What Really Happened to [Subject]",
                "The [Adjective] Case of [Subject]",
                "How [Subject] [Achievement/Failure]"
            ]

        # Generic patterns
        else:
            patterns = [
                "How to [Keyword] ([Year])",
                "[Number] Best [Keyword] for [Audience]",
                "The Complete Guide to [Keyword]",
                "[Keyword] Explained in [Time]"
            ]

        return patterns

    def _identify_content_gaps(self, keyword: str) -> List[str]:
        """Identify content gaps competitors haven't covered well."""
        keyword_lower = keyword.lower()

        gaps = []

        # Universal gaps
        gaps.append(f"Beginner's guide to {keyword}")
        gaps.append(f"Common {keyword} mistakes and how to avoid them")
        gaps.append(f"{keyword.title()} case studies from 2026")

        # Niche-specific gaps
        if any(w in keyword_lower for w in ["money", "finance", "invest"]):
            gaps.extend([
                "Low-risk strategies for beginners",
                "Step-by-step tutorials with real examples",
                "Comparison with traditional alternatives"
            ])
        elif any(w in keyword_lower for w in ["psychology", "mind"]):
            gaps.extend([
                "Scientific research summaries",
                "Practical application exercises",
                "Real-life case examples"
            ])

        return gaps[:6]

    def _generate_recommendations(self, keyword: str, patterns: List[str], gaps: List[str]) -> List[str]:
        """Generate strategic recommendations."""
        recommendations = [
            f"Use proven patterns: {patterns[0] if patterns else 'Number + Benefit format'}",
            "Include specific numbers and data points in titles",
            "Address content gaps for differentiation",
            "Focus on unique angles competitors haven't explored",
            "Combine multiple successful patterns creatively"
        ]

        if gaps:
            recommendations.append(f"Content gap opportunity: {gaps[0]}")

        return recommendations

    def _estimate_title_length(self, keyword: str) -> float:
        """Estimate optimal title length for niche."""
        keyword_lower = keyword.lower()

        if any(w in keyword_lower for w in ["tutorial", "guide", "how to"]):
            return 55.0
        elif any(w in keyword_lower for w in ["story", "mystery", "crime"]):
            return 48.0
        else:
            return 52.0

    def _extract_common_keywords(self, keyword: str) -> List[str]:
        """Extract common keywords that should be included."""
        words = keyword.lower().split()
        common = words.copy()

        # Add year
        common.append("2026")

        # Add intent modifiers
        common.extend(["how", "why", "best", "top"])

        # Add power words
        common.extend(["secret", "proven", "ultimate"])

        return list(set(common))[:10]

    def extract_patterns(self, videos: List[CompetitorVideo]) -> Dict[str, Any]:
        """
        Find common elements in successful videos.

        Args:
            videos: List of competitor videos

        Returns:
            Patterns found in successful content
        """
        if not videos:
            return {}

        patterns = {
            "title_structures": [],
            "common_words": {},
            "avg_title_length": 0,
            "number_usage": 0,
            "question_usage": 0
        }

        total_length = 0
        word_counts = {}

        for video in videos:
            title = video.title

            # Track title length
            total_length += len(title)

            # Track number usage
            if re.search(r'\d+', title):
                patterns["number_usage"] += 1

            # Track question usage
            if "?" in title or any(title.lower().startswith(q) for q in ["how", "what", "why"]):
                patterns["question_usage"] += 1

            # Count words
            for word in title.lower().split():
                if len(word) > 3:
                    word_counts[word] = word_counts.get(word, 0) + 1

        patterns["avg_title_length"] = total_length / len(videos) if videos else 0
        patterns["common_words"] = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        patterns["number_usage"] = patterns["number_usage"] / len(videos) if videos else 0
        patterns["question_usage"] = patterns["question_usage"] / len(videos) if videos else 0

        return patterns


# ============================================================
# PERFORMANCE PREDICTOR
# ============================================================

class PerformancePredictor:
    """
    Predict CTR and retention based on metadata patterns.

    Token cost: ZERO - Uses learned patterns
    """

    # CTR impact factors and weights
    CTR_FACTORS = {
        "has_number": {"weight": 1.15, "description": "Contains specific number"},
        "has_brackets": {"weight": 1.10, "description": "Uses brackets/parentheses"},
        "has_year": {"weight": 1.08, "description": "Includes current year"},
        "power_word": {"weight": 1.12, "description": "Contains power word"},
        "optimal_length": {"weight": 1.05, "description": "Title length 40-60 chars"},
        "curiosity_gap": {"weight": 1.18, "description": "Creates curiosity"},
        "starts_with_number": {"weight": 1.10, "description": "Starts with number"},
        "question_format": {"weight": 1.08, "description": "Question format"}
    }

    POWER_WORDS = [
        "secret", "proven", "ultimate", "shocking", "hidden",
        "revealed", "truth", "exclusive", "insider", "genius",
        "legendary", "untold", "massive", "insane", "unbelievable"
    ]

    CURIOSITY_TRIGGERS = [
        "nobody tells you", "you didn't know", "they don't want",
        "what happens when", "the truth about", "here's why",
        "you won't believe", "this is why"
    ]

    def __init__(self, patterns_file: str = None):
        """Initialize with optional patterns file."""
        self.patterns_file = patterns_file or "src/agents/seo_data/patterns.json"
        self.learned_patterns = self._load_patterns()

    def _load_patterns(self) -> Dict:
        """Load learned patterns from file."""
        path = Path(self.patterns_file)
        if path.exists():
            try:
                return json.loads(path.read_text())
            except:
                pass
        return {}

    def predict_ctr(self, title: str, niche: str = "default") -> CTRPrediction:
        """
        Predict CTR score for a title.

        Args:
            title: Video title to analyze
            niche: Content niche

        Returns:
            CTRPrediction with score and improvements
        """
        logger.info(f"[PerformancePredictor] Predicting CTR for: {title[:40]}...")

        base_ctr = 4.0  # Average YouTube CTR
        factors = {}
        improvements = []
        title_lower = title.lower()

        # Check each factor
        if re.search(r'\d+', title):
            factors["has_number"] = self.CTR_FACTORS["has_number"]["weight"]
        else:
            improvements.append("Add a specific number (e.g., '5 Ways', '10 Tips')")

        if "(" in title or "[" in title:
            factors["has_brackets"] = self.CTR_FACTORS["has_brackets"]["weight"]
        else:
            improvements.append("Add parentheses with context (e.g., '(2026 Guide)')")

        current_year = str(datetime.now().year)
        if current_year in title:
            factors["has_year"] = self.CTR_FACTORS["has_year"]["weight"]
        else:
            improvements.append(f"Add current year ({current_year}) for freshness")

        if any(pw in title_lower for pw in self.POWER_WORDS):
            factors["power_word"] = self.CTR_FACTORS["power_word"]["weight"]
        else:
            improvements.append("Add a power word (secret, proven, ultimate)")

        if 40 <= len(title) <= 60:
            factors["optimal_length"] = self.CTR_FACTORS["optimal_length"]["weight"]
        elif len(title) > 60:
            improvements.append("Shorten title to under 60 characters")
        else:
            improvements.append("Expand title to 40+ characters")

        if any(trigger in title_lower for trigger in self.CURIOSITY_TRIGGERS):
            factors["curiosity_gap"] = self.CTR_FACTORS["curiosity_gap"]["weight"]
        else:
            improvements.append("Add curiosity trigger (e.g., 'nobody tells you')")

        if re.match(r'^\d+\s', title):
            factors["starts_with_number"] = self.CTR_FACTORS["starts_with_number"]["weight"]

        if "?" in title:
            factors["question_format"] = self.CTR_FACTORS["question_format"]["weight"]

        # Calculate predicted CTR
        multiplier = 1.0
        for factor_weight in factors.values():
            multiplier *= factor_weight

        predicted_ctr = min(base_ctr * multiplier, 15.0)  # Cap at 15%

        # Convert to 0-100 scale for consistency
        ctr_score = min(predicted_ctr * 6.67, 100)  # 15% CTR = 100 score

        # Calculate confidence based on factors found
        confidence = len(factors) / len(self.CTR_FACTORS)

        prediction = CTRPrediction(
            title=title,
            predicted_ctr=ctr_score,
            confidence=confidence,
            factors={k: v for k, v in factors.items()},
            improvements=improvements[:5]
        )

        logger.success(f"[PerformancePredictor] CTR Score: {ctr_score:.1f}")
        return prediction

    def predict_retention(self, script: str) -> RetentionPrediction:
        """
        Predict audience retention based on script.

        Args:
            script: Video script text

        Returns:
            RetentionPrediction with analysis
        """
        logger.info("[PerformancePredictor] Predicting retention...")

        # Analyze hook (first 100 chars)
        hook = script[:100] if len(script) > 100 else script
        hook_strength = self._analyze_hook(hook)

        # Analyze pacing
        pacing_score = self._analyze_pacing(script)

        # Find engagement points
        engagement_points = self._find_engagement_points(script)

        # Find potential drop-off risks
        drop_off_risks = self._find_drop_off_risks(script)

        # Estimate retention
        base_retention = 40.0  # Average YouTube retention
        retention_boost = (hook_strength + pacing_score) / 200 * 20
        estimated_retention = min(base_retention + retention_boost, 70.0)

        prediction = RetentionPrediction(
            estimated_avg_view_duration=estimated_retention,
            hook_strength=hook_strength,
            pacing_score=pacing_score,
            engagement_points=engagement_points,
            drop_off_risks=drop_off_risks
        )

        logger.success(f"[PerformancePredictor] Retention: {estimated_retention:.1f}%")
        return prediction

    def _analyze_hook(self, hook: str) -> float:
        """Analyze hook strength (0-100)."""
        score = 50.0

        # Strong hook indicators
        strong_hooks = [
            "what if", "imagine", "here's the truth", "nobody tells",
            "i'm going to show", "by the end", "the secret"
        ]

        for phrase in strong_hooks:
            if phrase in hook.lower():
                score += 15
                break

        # Questions engage
        if "?" in hook:
            score += 10

        # Specific numbers/data
        if re.search(r'\$?\d+[,.]?\d*', hook):
            score += 10

        return min(score, 100)

    def _analyze_pacing(self, script: str) -> float:
        """Analyze script pacing (0-100)."""
        score = 60.0

        # Count sentences
        sentences = len(re.findall(r'[.!?]+', script))
        words = len(script.split())

        if sentences > 0:
            avg_sentence_length = words / sentences

            # Optimal sentence length for video: 10-20 words
            if 10 <= avg_sentence_length <= 20:
                score += 20
            elif avg_sentence_length > 30:
                score -= 10

        # Check for pattern interrupts (questions, exclamations)
        questions = len(re.findall(r'\?', script))
        exclamations = len(re.findall(r'!', script))

        if questions > 5:
            score += 10
        if exclamations > 3:
            score += 5

        return min(max(score, 0), 100)

    def _find_engagement_points(self, script: str) -> List[Tuple[int, str]]:
        """Find strong engagement points in script."""
        points = []

        engagement_phrases = [
            "but here's the thing",
            "now here's where it gets interesting",
            "what most people don't realize",
            "let me show you",
            "and this is important"
        ]

        script_lower = script.lower()
        for phrase in engagement_phrases:
            idx = script_lower.find(phrase)
            if idx != -1:
                # Approximate timestamp (assuming 150 words/min)
                words_before = len(script[:idx].split())
                timestamp_sec = int(words_before / 2.5)  # ~150 words/min
                points.append((timestamp_sec, phrase))

        return sorted(points, key=lambda x: x[0])[:5]

    def _find_drop_off_risks(self, script: str) -> List[Tuple[int, str]]:
        """Find potential drop-off points."""
        risks = []

        # Long paragraphs without breaks
        paragraphs = script.split('\n\n')
        for i, para in enumerate(paragraphs):
            words = len(para.split())
            if words > 200:
                # Approximate position
                words_before = sum(len(p.split()) for p in paragraphs[:i])
                timestamp_sec = int(words_before / 2.5)
                risks.append((timestamp_sec, "Long unbroken section"))

        return risks[:3]

    def suggest_improvements(self, metadata: Dict[str, Any]) -> List[str]:
        """
        Suggest improvements for video metadata.

        Args:
            metadata: Dict with title, description, tags

        Returns:
            List of improvement suggestions
        """
        suggestions = []

        title = metadata.get("title", "")
        description = metadata.get("description", "")
        tags = metadata.get("tags", [])

        # Title suggestions
        ctr_pred = self.predict_ctr(title)
        suggestions.extend(ctr_pred.improvements)

        # Description suggestions
        if len(description) < 200:
            suggestions.append("Expand description to 200+ characters")

        if "http" not in description:
            suggestions.append("Add relevant links in description")

        # Tag suggestions
        if len(tags) < 10:
            suggestions.append(f"Add more tags (currently {len(tags)}, aim for 10+)")

        return suggestions[:8]


# ============================================================
# A/B TEST MANAGER
# ============================================================

class ABTestManager:
    """
    Manage title and thumbnail A/B testing.

    Token cost: LOW - AI only for variant generation
    """

    VARIANT_TYPES = {
        "curiosity": {
            "patterns": [
                "The Truth About {topic}",
                "Why Nobody Tells You About {topic}",
                "What {experts} Don't Want You to Know About {topic}"
            ],
            "description": "Creates information gap"
        },
        "urgency": {
            "patterns": [
                "{topic} Before It's Too Late",
                "Stop {bad_action} RIGHT NOW",
                "The {topic} Mistake You're Making Today"
            ],
            "description": "Creates time pressure"
        },
        "how_to": {
            "patterns": [
                "How to {topic} (Step by Step)",
                "The Complete Guide to {topic}",
                "How I {achievement} with {topic}"
            ],
            "description": "Educational format"
        },
        "listicle": {
            "patterns": [
                "{number} {topic} That Will {benefit}",
                "Top {number} {topic} in {year}",
                "{number} {topic} Every {audience} Needs"
            ],
            "description": "Numbered list format"
        },
        "story": {
            "patterns": [
                "How {person} {achievement} with {topic}",
                "The {adjective} Story of {topic}",
                "What Happened When I {action}"
            ],
            "description": "Narrative format"
        },
        "controversy": {
            "patterns": [
                "Why {topic} is a {opinion}",
                "The Problem With {topic}",
                "Why I Quit {topic}"
            ],
            "description": "Creates debate"
        }
    }

    def __init__(self, results_file: str = None):
        """Initialize with results storage."""
        self.results_file = results_file or "src/agents/seo_data/test_results.json"
        self.predictor = PerformancePredictor()
        self._ensure_results_file()

    def _ensure_results_file(self):
        """Ensure results file exists."""
        path = Path(self.results_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("{}")

    def generate_variants(self, title: str, count: int = 5, niche: str = "default") -> List[TitleVariant]:
        """
        Create title variations using different psychological triggers.

        Args:
            title: Original title
            count: Number of variants to generate
            niche: Content niche

        Returns:
            List of title variants
        """
        logger.info(f"[ABTestManager] Generating {count} variants for: {title[:40]}...")

        variants = []

        # Extract topic from title
        topic = self._extract_topic(title)

        # Generate one variant per type
        for variant_type, config in self.VARIANT_TYPES.items():
            if len(variants) >= count:
                break

            pattern = random.choice(config["patterns"])
            new_title = self._apply_pattern(pattern, topic, title, niche)

            # Score the variant
            ctr_pred = self.predictor.predict_ctr(new_title, niche)

            variant = TitleVariant(
                title=new_title,
                variant_type=variant_type,
                predicted_ctr=ctr_pred.predicted_ctr,
                score=ctr_pred.predicted_ctr,
                rationale=config["description"]
            )
            variants.append(variant)

        # Sort by predicted CTR
        variants.sort(key=lambda v: v.score, reverse=True)

        logger.success(f"[ABTestManager] Generated {len(variants)} variants")
        return variants

    def _extract_topic(self, title: str) -> str:
        """Extract main topic from title."""
        # Remove common prefixes
        prefixes = ["how to", "why", "what", "the", "a", "an"]
        topic = title.lower()

        for prefix in prefixes:
            if topic.startswith(prefix + " "):
                topic = topic[len(prefix) + 1:]
                break

        # Remove year
        topic = re.sub(r'\s*\(?\d{4}\)?\s*', '', topic)

        # Remove brackets content
        topic = re.sub(r'\s*[\[\(].*?[\]\)]\s*', '', topic)

        return topic.strip().title()

    def _apply_pattern(self, pattern: str, topic: str, original: str, niche: str) -> str:
        """Apply a pattern to generate new title."""
        year = str(datetime.now().year)

        # Variable replacements
        replacements = {
            "{topic}": topic,
            "{year}": year,
            "{number}": str(random.choice([3, 5, 7, 10])),
            "{person}": random.choice(["Warren Buffett", "Elon Musk", "I", "This Expert"]),
            "{experts}": random.choice(["Experts", "Gurus", "They", "Professionals"]),
            "{achievement}": random.choice(["succeeded", "made $10,000", "changed my life", "achieved this"]),
            "{adjective}": random.choice(["Untold", "Hidden", "Secret", "Shocking"]),
            "{audience}": random.choice(["Beginner", "Professional", "Smart Person", "Investor"]),
            "{benefit}": random.choice(["Change Your Life", "Save You Money", "Make You Rich", "Blow Your Mind"]),
            "{action}": random.choice(["Tried This", "Did This", "Made This Change"]),
            "{bad_action}": random.choice(["Wasting Money", "Making This Mistake", "Doing This Wrong"]),
            "{opinion}": random.choice(["Lie", "Scam", "Mistake", "Myth"])
        }

        result = pattern
        for key, value in replacements.items():
            result = result.replace(key, value)

        return result

    def score_variants(self, variants: List[TitleVariant], niche: str = "default") -> List[TitleVariant]:
        """
        Score and rank all variants.

        Args:
            variants: List of variants to score
            niche: Content niche

        Returns:
            Ranked list of variants
        """
        for variant in variants:
            ctr_pred = self.predictor.predict_ctr(variant.title, niche)
            variant.predicted_ctr = ctr_pred.predicted_ctr
            variant.score = ctr_pred.predicted_ctr

        variants.sort(key=lambda v: v.score, reverse=True)
        return variants

    def track_test_results(self, video_id: str, variant: str, metrics: Dict[str, Any]):
        """
        Record performance for learning.

        Args:
            video_id: YouTube video ID
            variant: Title variant used
            metrics: Performance metrics (views, CTR, retention)
        """
        try:
            path = Path(self.results_file)
            results = json.loads(path.read_text())

            results[video_id] = {
                "variant": variant,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat()
            }

            path.write_text(json.dumps(results, indent=2))
            logger.info(f"[ABTestManager] Tracked results for video: {video_id}")

        except Exception as e:
            logger.warning(f"Failed to track results: {e}")


# ============================================================
# CONTENT CALENDAR
# ============================================================

class ContentCalendar:
    """
    SEO-driven content planning.

    Token cost: ZERO to LOW
    """

    # Optimal posting times by niche (UTC)
    POSTING_TIMES = {
        "finance": {
            "best_days": ["Monday", "Wednesday", "Friday"],
            "best_hours": [15, 19, 21],
            "avoid_days": ["Saturday"],
            "rationale": "Finance audience active during work week"
        },
        "psychology": {
            "best_days": ["Tuesday", "Thursday", "Saturday"],
            "best_hours": [16, 19, 21],
            "avoid_days": [],
            "rationale": "Self-improvement seekers browse evenings"
        },
        "storytelling": {
            "best_days": ["Daily"],
            "best_hours": [17, 20, 22],
            "avoid_days": [],
            "rationale": "Entertainment works any day"
        }
    }

    def __init__(self):
        """Initialize content calendar."""
        self.keyword_researcher = KeywordResearcher()
        self.intent_analyzer = SearchIntentAnalyzer()
        self.competitor_analyzer = CompetitorAnalyzer()

    def suggest_topics(self, niche: str, count: int = 10) -> List[TopicSuggestion]:
        """
        Suggest topics based on trending keywords and content gaps.

        Args:
            niche: Content niche
            count: Number of suggestions

        Returns:
            List of topic suggestions
        """
        logger.info(f"[ContentCalendar] Generating {count} topic suggestions for {niche}")

        suggestions = []

        # Define seed keywords by niche
        seeds = self._get_niche_seeds(niche)

        for seed in seeds[:count]:
            # Research the keyword
            research = self.keyword_researcher.research_keyword(seed, niche)

            # Analyze intent
            intent = self.intent_analyzer.classify_intent(seed)

            # Create suggestion
            suggestion = TopicSuggestion(
                topic=seed.title(),
                keyword=seed,
                opportunity_score=research.opportunity_score,
                search_volume=self._classify_volume(research.search_volume_trend),
                competition=research.competition_level,
                content_angle=self._suggest_angle(seed, intent),
                title_suggestion=self._generate_title(seed, niche, intent),
                rationale=f"Opportunity score: {research.opportunity_score:.0f}/100"
            )
            suggestions.append(suggestion)

        # Sort by opportunity
        suggestions.sort(key=lambda s: s.opportunity_score, reverse=True)

        logger.success(f"[ContentCalendar] Generated {len(suggestions)} suggestions")
        return suggestions[:count]

    def _get_niche_seeds(self, niche: str) -> List[str]:
        """Get seed keywords for a niche."""
        seeds = {
            "finance": [
                "passive income ideas", "how to invest", "money mistakes",
                "stock market beginner", "side hustle", "wealth building",
                "financial freedom", "investing strategy", "save money tips",
                "compound interest"
            ],
            "psychology": [
                "dark psychology", "manipulation tactics", "body language signs",
                "cognitive biases", "narcissist signs", "emotional intelligence",
                "subconscious mind", "psychology tricks", "human behavior",
                "mental models"
            ],
            "storytelling": [
                "true crime stories", "unsolved mysteries", "company failures",
                "billion dollar mistakes", "rise and fall", "untold stories",
                "what happened to", "business scandals", "documentary",
                "case study"
            ]
        }
        return seeds.get(niche, seeds["finance"])

    def _classify_volume(self, trend: List[float]) -> str:
        """Classify search volume from trend data."""
        if not trend:
            return "medium"

        avg = sum(trend) / len(trend) if trend else 0
        if avg > 70:
            return "high"
        elif avg > 30:
            return "medium"
        else:
            return "low"

    def _suggest_angle(self, keyword: str, intent: SearchIntent) -> str:
        """Suggest content angle based on intent."""
        angles = {
            "informational": "Step-by-step tutorial with practical examples",
            "commercial": "Honest comparison with pros and cons",
            "transactional": "Results-focused case study",
            "navigational": "Complete guide with resources"
        }
        return angles.get(intent.primary_intent, angles["informational"])

    def _generate_title(self, keyword: str, niche: str, intent: SearchIntent) -> str:
        """Generate a title suggestion."""
        if intent.title_templates:
            template = intent.title_templates[0]
            return template.replace("{topic}", keyword.title()).replace("{year}", "2026").replace("{number}", "5")

        return f"The Complete Guide to {keyword.title()} (2026)"

    def optimal_posting_schedule(self, niche: str) -> Dict[str, Any]:
        """
        Get optimal posting schedule based on niche.

        Args:
            niche: Content niche

        Returns:
            Schedule recommendations
        """
        schedule = self.POSTING_TIMES.get(niche, self.POSTING_TIMES["finance"])

        return {
            "niche": niche,
            **schedule,
            "recommendation": f"Post on {', '.join(schedule['best_days'][:3])} at {schedule['best_hours'][0]}:00 UTC"
        }

    def keyword_opportunity_score(self, keyword: str, niche: str = "default") -> float:
        """
        Calculate opportunity score for a keyword.

        Balance of search volume vs competition.

        Args:
            keyword: Keyword to evaluate
            niche: Content niche

        Returns:
            Opportunity score 0-100
        """
        research = self.keyword_researcher.research_keyword(keyword, niche)
        return research.opportunity_score

    def generate_content_plan(self, niche: str, weeks: int = 4) -> Dict[str, Any]:
        """
        Generate a content plan for specified weeks.

        Args:
            niche: Content niche
            weeks: Number of weeks to plan

        Returns:
            Content plan with topics and schedule
        """
        logger.info(f"[ContentCalendar] Generating {weeks}-week content plan for {niche}")

        schedule = self.optimal_posting_schedule(niche)
        topics = self.suggest_topics(niche, count=weeks * 3)

        plan = {
            "niche": niche,
            "weeks": weeks,
            "schedule": schedule,
            "topics": [t.to_dict() for t in topics],
            "weekly_plan": []
        }

        # Distribute topics across weeks
        topic_idx = 0
        for week in range(1, weeks + 1):
            week_topics = []
            for _ in range(3):  # 3 videos per week
                if topic_idx < len(topics):
                    week_topics.append(topics[topic_idx].topic)
                    topic_idx += 1

            plan["weekly_plan"].append({
                "week": week,
                "topics": week_topics
            })

        logger.success(f"[ContentCalendar] Generated {weeks}-week plan")
        return plan


# ============================================================
# MAIN SEO STRATEGIST
# ============================================================

class SEOStrategist:
    """
    World-class SEO strategist that coordinates all components.

    Main entry point for SEO optimization and strategy.
    """

    def __init__(self):
        """Initialize all sub-components."""
        self.keyword_researcher = KeywordResearcher()
        self.intent_analyzer = SearchIntentAnalyzer()
        self.competitor_analyzer = CompetitorAnalyzer()
        self.predictor = PerformancePredictor()
        self.ab_manager = ABTestManager()
        self.calendar = ContentCalendar()

        logger.info("[SEOStrategist] Initialized with all components")

    def research_keyword(self, keyword: str, niche: str = "default") -> SEOStrategyResult:
        """
        Comprehensive keyword research.

        Args:
            keyword: Keyword to research
            niche: Content niche

        Returns:
            SEOStrategyResult with research data
        """
        logger.info(f"[SEOStrategist] Full research for: {keyword}")

        # Get keyword data
        kw_data = self.keyword_researcher.research_keyword(keyword, niche)

        # Analyze intent
        intent = self.intent_analyzer.classify_intent(keyword)

        # Get competitor insights
        competitors = self.competitor_analyzer.analyze_top_results(keyword)

        recommendations = [
            f"Search intent: {intent.primary_intent} ({intent.confidence:.0%} confidence)",
            f"Competition level: {kw_data.competition_level}",
            f"Opportunity score: {kw_data.opportunity_score:.0f}/100",
        ]

        if kw_data.rising_queries:
            recommendations.append(f"Rising queries: {', '.join(kw_data.rising_queries[:3])}")

        recommendations.extend(intent.content_recommendations[:3])
        recommendations.extend(competitors.recommendations[:2])

        return SEOStrategyResult(
            success=True,
            operation=f"research:{keyword}",
            data={
                "keyword_data": kw_data.to_dict(),
                "intent": intent.to_dict(),
                "competitor_report": competitors.to_dict()
            },
            recommendations=recommendations
        )

    def full_optimization(
        self,
        title: str,
        description: str,
        tags: List[str],
        niche: str = "default"
    ) -> SEOStrategyResult:
        """
        Full pre-upload optimization.

        Args:
            title: Video title
            description: Video description
            tags: Video tags
            niche: Content niche

        Returns:
            SEOStrategyResult with optimizations and variants
        """
        logger.info(f"[SEOStrategist] Full optimization for: {title[:40]}...")

        # Predict CTR
        ctr_pred = self.predictor.predict_ctr(title, niche)

        # Generate A/B variants
        variants = self.ab_manager.generate_variants(title, count=5, niche=niche)

        # Get intent
        intent = self.intent_analyzer.classify_intent(title)

        recommendations = []
        recommendations.extend(ctr_pred.improvements)
        recommendations.extend(intent.content_recommendations[:2])

        if variants:
            best_variant = variants[0]
            if best_variant.score > ctr_pred.predicted_ctr:
                recommendations.insert(0, f"Consider alternative: '{best_variant.title}' (score: {best_variant.score:.0f})")

        return SEOStrategyResult(
            success=True,
            operation="full_optimization",
            data={
                "original": {"title": title, "description": description, "tags": tags},
                "ctr_prediction": ctr_pred.to_dict(),
                "variants": [v.to_dict() for v in variants],
                "intent": intent.to_dict()
            },
            recommendations=recommendations
        )

    def content_strategy(self, niche: str, topics: int = 10, weeks: int = 4) -> SEOStrategyResult:
        """
        Generate full content strategy.

        Args:
            niche: Content niche
            topics: Number of topics to suggest
            weeks: Weeks to plan for

        Returns:
            SEOStrategyResult with strategy
        """
        logger.info(f"[SEOStrategist] Generating strategy for {niche}")

        # Get topic suggestions
        topic_suggestions = self.calendar.suggest_topics(niche, topics)

        # Get content plan
        content_plan = self.calendar.generate_content_plan(niche, weeks)

        # Get schedule
        schedule = self.calendar.optimal_posting_schedule(niche)

        recommendations = [
            f"Focus on {schedule['best_days'][0]}s for best engagement",
            f"Post around {schedule['best_hours'][0]}:00 UTC",
        ]

        if topic_suggestions:
            best_topic = topic_suggestions[0]
            recommendations.append(f"Top opportunity: '{best_topic.topic}' (score: {best_topic.opportunity_score:.0f})")

        return SEOStrategyResult(
            success=True,
            operation=f"strategy:{niche}",
            data={
                "topics": [t.to_dict() for t in topic_suggestions],
                "content_plan": content_plan,
                "schedule": schedule
            },
            recommendations=recommendations
        )

    def analyze_competitors(self, keyword: str, count: int = 10) -> SEOStrategyResult:
        """
        Analyze competitors for a keyword.

        Args:
            keyword: Keyword to analyze
            count: Number of competitors

        Returns:
            SEOStrategyResult with competitor analysis
        """
        report = self.competitor_analyzer.analyze_top_results(keyword, count)

        return SEOStrategyResult(
            success=True,
            operation=f"competitors:{keyword}",
            data={"report": report.to_dict()},
            recommendations=report.recommendations
        )

    def generate_ab_variants(self, title: str, count: int = 5, niche: str = "default") -> SEOStrategyResult:
        """
        Generate A/B test variants for a title.

        Args:
            title: Original title
            count: Number of variants
            niche: Content niche

        Returns:
            SEOStrategyResult with variants
        """
        variants = self.ab_manager.generate_variants(title, count, niche)

        recommendations = []
        if variants:
            for i, v in enumerate(variants[:3], 1):
                recommendations.append(f"#{i}: {v.title} (CTR: {v.predicted_ctr:.0f})")

        return SEOStrategyResult(
            success=True,
            operation="ab_variants",
            data={
                "original": title,
                "variants": [v.to_dict() for v in variants]
            },
            recommendations=recommendations
        )

    def run(self, command: str = None, **kwargs) -> SEOStrategyResult:
        """
        CLI entry point.

        Args:
            command: Command (research, optimize, strategy, competitors, ab-test)
            **kwargs: Command arguments

        Returns:
            SEOStrategyResult
        """
        niche = kwargs.get("niche", "default")

        if command == "research":
            keyword = kwargs.get("keyword", "")
            if keyword:
                return self.research_keyword(keyword, niche)

        elif command == "optimize":
            file_path = kwargs.get("file")
            if file_path:
                with open(file_path) as f:
                    data = json.load(f)
                return self.full_optimization(
                    title=data.get("title", ""),
                    description=data.get("description", ""),
                    tags=data.get("tags", []),
                    niche=niche
                )

        elif command == "strategy":
            topics = kwargs.get("topics", 10)
            weeks = kwargs.get("weeks", 4)
            return self.content_strategy(niche, topics, weeks)

        elif command == "competitors":
            keyword = kwargs.get("keyword", "")
            count = kwargs.get("top", 10)
            if keyword:
                return self.analyze_competitors(keyword, count)

        elif command == "ab-test":
            title = kwargs.get("title", "")
            count = kwargs.get("variants", 5)
            if title:
                return self.generate_ab_variants(title, count, niche)

        elif command == "calendar":
            weeks = kwargs.get("weeks", 4)
            return self.content_strategy(niche, topics=weeks * 3, weeks=weeks)

        return SEOStrategyResult(
            success=False,
            operation="unknown",
            recommendations=["Unknown command. Use: research, optimize, strategy, competitors, ab-test, calendar"]
        )


# CLI Entry Point
def main():
    """CLI entry point."""
    import sys

    if len(sys.argv) < 2:
        print("""
SEO Strategist - World-Class YouTube SEO Agent
===============================================

Commands:
    research <keyword> [--niche <niche>]
        Full keyword research with trends and competition

    optimize --file <path> [--niche <niche>]
        Optimize metadata from JSON file

    strategy --niche <niche> [--topics N] [--weeks N]
        Generate content strategy

    competitors <keyword> [--top N]
        Analyze top competitors

    ab-test "<title>" [--variants N] [--niche <niche>]
        Generate A/B test title variants

    calendar --niche <niche> [--weeks N]
        Generate content calendar

Examples:
    python -m src.agents.seo_strategist research "passive income" --niche finance
    python -m src.agents.seo_strategist strategy --niche psychology --topics 10
    python -m src.agents.seo_strategist ab-test "How to Make Money Online" --variants 5
    python -m src.agents.seo_strategist competitors "money mistakes" --top 10

Niches: finance, psychology, storytelling
        """)
        return

    strategist = SEOStrategist()
    cmd = sys.argv[1]

    # Parse arguments
    kwargs = {}
    i = 2
    positional = None

    while i < len(sys.argv):
        arg = sys.argv[i]

        if arg.startswith("--"):
            key = arg[2:]
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
                value = sys.argv[i + 1]
                # Try to convert to int
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
            positional = arg
            i += 1

    # Map positional arg to command-specific key
    if positional:
        if cmd in ["research", "competitors"]:
            kwargs["keyword"] = positional
        elif cmd == "ab-test":
            kwargs["title"] = positional

    # Run command
    result = strategist.run(cmd, **kwargs)

    # Output
    if kwargs.get("json"):
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("\n" + "=" * 60)
        print(f"SEO STRATEGIST: {cmd.upper()}")
        print("=" * 60)
        print(result.summary())

        if result.data:
            print("\nData:")
            for key in list(result.data.keys())[:3]:
                print(f"  {key}: ...")


if __name__ == "__main__":
    main()
