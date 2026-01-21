"""
Free Keyword Research System

YouTube Search Suggest API scraping and Google Trends integration (FREE!).
Find low-competition keywords (1K-20K searches) without VidIQ/TubeBuddy.

This module extends the existing keyword_intelligence.py with additional
FREE data sources and simplified workflows.

Features:
- YouTube autocomplete scraping
- Google Trends API (free)
- Keyword clustering
- Competition analysis
- Long-tail generation
- Search volume estimation

Usage:
    researcher = FreeKeywordResearch()

    # Find keywords
    keywords = researcher.find_keywords("passive income", count=50)

    # Analyze competition
    analysis = researcher.analyze_competition("make money online")

    # Get trending topics
    trending = researcher.get_trending_topics(niche="finance")
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from loguru import logger

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False


@dataclass
class KeywordResult:
    """Keyword research result."""
    keyword: str
    search_volume_estimate: str  # "very_high", "high", "medium", "low", "very_low"
    competition: str  # "very_high", "high", "medium", "low", "very_low"
    opportunity_score: float  # 0-100
    trend_direction: str  # "rising", "stable", "declining"
    suggestions_count: int = 0  # Number of autocomplete suggestions
    is_longtail: bool = False

    def to_dict(self) -> Dict:
        return {
            "keyword": self.keyword,
            "search_volume_estimate": self.search_volume_estimate,
            "competition": self.competition,
            "opportunity_score": self.opportunity_score,
            "trend_direction": self.trend_direction,
            "suggestions_count": self.suggestions_count,
            "is_longtail": self.is_longtail
        }


class FreeKeywordResearch:
    """
    FREE keyword research using YouTube autocomplete and Google Trends.

    No paid API required! (Saves $20-50/month vs VidIQ/TubeBuddy)
    """

    # Alphabet for autocomplete expansion
    ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"

    # Question words for long-tail generation
    QUESTION_WORDS = ["how", "what", "why", "when", "where", "who", "which", "can", "should", "is"]

    # Modifier words for variations
    MODIFIERS = ["best", "top", "free", "easy", "simple", "ultimate", "complete", "beginner"]

    def __init__(self):
        """Initialize free keyword research system."""
        self.pytrends = None
        if PYTRENDS_AVAILABLE:
            self.pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25))

        logger.info("[FreeKeywordResearch] Initialized")

    def find_keywords(
        self,
        seed_keyword: str,
        count: int = 50,
        include_longtail: bool = True
    ) -> List[KeywordResult]:
        """
        Find keyword opportunities from seed keyword.

        Args:
            seed_keyword: Starting keyword
            count: Number of keywords to return
            include_longtail: Include long-tail variations

        Returns:
            List of KeywordResult objects sorted by opportunity
        """
        logger.info(f"[FreeKeywordResearch] Finding keywords for: {seed_keyword}")

        all_keywords = set()

        # 1. Get YouTube autocomplete suggestions
        autocomplete = self._get_youtube_suggestions(seed_keyword)
        all_keywords.update(autocomplete)

        # 2. Expand with alphabet
        for letter in self.ALPHABET[:10]:  # Limit to avoid rate limiting
            expanded = self._get_youtube_suggestions(f"{seed_keyword} {letter}")
            all_keywords.update(expanded)

        # 3. Add question-based keywords
        if include_longtail:
            for question in self.QUESTION_WORDS[:5]:
                question_kws = self._get_youtube_suggestions(f"{question} {seed_keyword}")
                all_keywords.update(question_kws)

        # 4. Add modifier-based keywords
        for modifier in self.MODIFIERS[:5]:
            modified = self._get_youtube_suggestions(f"{modifier} {seed_keyword}")
            all_keywords.update(modified)

        # Remove seed keyword
        all_keywords.discard(seed_keyword)

        # Analyze each keyword
        results = []
        for kw in list(all_keywords)[:count * 2]:  # Analyze more than needed
            try:
                result = self._analyze_keyword(kw)
                results.append(result)
                time.sleep(0.2)  # Rate limiting
            except Exception as e:
                logger.debug(f"Failed to analyze '{kw}': {e}")

        # Sort by opportunity score
        results.sort(key=lambda r: r.opportunity_score, reverse=True)

        logger.success(f"[FreeKeywordResearch] Found {len(results[:count])} keywords")
        return results[:count]

    def _get_youtube_suggestions(self, query: str) -> List[str]:
        """Get YouTube autocomplete suggestions."""
        if not REQUESTS_AVAILABLE:
            return []

        try:
            url = "https://suggestqueries.google.com/complete/search"
            params = {
                "client": "youtube",
                "q": query,
                "ds": "yt"
            }

            response = requests.get(url, params=params, timeout=5)
            if response.status_code != 200:
                return []

            # Parse response
            text = response.text
            start = text.find("(") + 1
            end = text.rfind(")")

            if start > 0 and end > start:
                data = json.loads(text[start:end])
                if len(data) > 1 and isinstance(data[1], list):
                    return [item[0] for item in data[1] if isinstance(item, list)]

        except Exception as e:
            logger.debug(f"YouTube autocomplete failed for '{query}': {e}")

        return []

    def _analyze_keyword(self, keyword: str) -> KeywordResult:
        """Analyze a single keyword."""
        # Get suggestion count
        suggestions = self._get_youtube_suggestions(keyword)
        suggestion_count = len(suggestions)

        # Estimate search volume based on suggestion count
        if suggestion_count > 30:
            volume = "very_high"
            comp = "very_high"
        elif suggestion_count > 20:
            volume = "high"
            comp = "high"
        elif suggestion_count > 10:
            volume = "medium"
            comp = "medium"
        elif suggestion_count > 5:
            volume = "low"
            comp = "low"
        else:
            volume = "very_low"
            comp = "very_low"

        # Adjust competition based on word count (longer = less competition)
        word_count = len(keyword.split())
        if word_count >= 4:
            comp_levels = ["very_high", "high", "medium", "low", "very_low"]
            current_idx = comp_levels.index(comp)
            new_idx = min(len(comp_levels) - 1, current_idx + 2)
            comp = comp_levels[new_idx]

        # Get trend direction
        trend = "stable"
        if self.pytrends:
            try:
                self.pytrends.build_payload([keyword], timeframe='today 3-m')
                interest = self.pytrends.interest_over_time()

                if not interest.empty and keyword in interest.columns:
                    data = interest[keyword].tolist()
                    if len(data) >= 2:
                        recent_avg = sum(data[-4:]) / 4
                        earlier_avg = sum(data[:4]) / 4

                        if recent_avg > earlier_avg * 1.2:
                            trend = "rising"
                        elif recent_avg < earlier_avg * 0.8:
                            trend = "declining"
            except:
                pass

        # Calculate opportunity score
        opportunity = self._calculate_opportunity(volume, comp, trend, word_count)

        # Determine if long-tail
        is_longtail = word_count >= 3

        return KeywordResult(
            keyword=keyword,
            search_volume_estimate=volume,
            competition=comp,
            opportunity_score=opportunity,
            trend_direction=trend,
            suggestions_count=suggestion_count,
            is_longtail=is_longtail
        )

    def _calculate_opportunity(
        self,
        volume: str,
        competition: str,
        trend: str,
        word_count: int
    ) -> float:
        """Calculate opportunity score (0-100)."""
        score = 0.0

        # Volume score (0-30)
        volume_scores = {"very_high": 30, "high": 25, "medium": 20, "low": 15, "very_low": 5}
        score += volume_scores.get(volume, 10)

        # Competition score (0-40, lower competition = higher score)
        comp_scores = {"very_low": 40, "low": 30, "medium": 20, "high": 10, "very_high": 5}
        score += comp_scores.get(competition, 15)

        # Trend score (0-20)
        trend_scores = {"rising": 20, "stable": 10, "declining": 0}
        score += trend_scores.get(trend, 10)

        # Long-tail bonus (0-10)
        if word_count >= 4:
            score += 10
        elif word_count >= 3:
            score += 5

        return min(100.0, score)

    def analyze_competition(self, keyword: str) -> Dict:
        """
        Analyze competition for a keyword.

        Args:
            keyword: Keyword to analyze

        Returns:
            Competition analysis dictionary
        """
        result = self._analyze_keyword(keyword)

        # Get related keywords
        related = self._get_youtube_suggestions(keyword)

        # Get trend data
        trend_data = []
        if self.pytrends:
            try:
                self.pytrends.build_payload([keyword], timeframe='today 12-m')
                interest = self.pytrends.interest_over_time()

                if not interest.empty and keyword in interest.columns:
                    trend_data = interest[keyword].tolist()
            except:
                pass

        return {
            "keyword": keyword,
            "competition_level": result.competition,
            "opportunity_score": result.opportunity_score,
            "search_volume_estimate": result.search_volume_estimate,
            "trend_direction": result.trend_direction,
            "suggestion_count": result.suggestions_count,
            "related_keywords": related[:10],
            "trend_data": trend_data,
            "recommendation": self._get_recommendation(result)
        }

    def _get_recommendation(self, result: KeywordResult) -> str:
        """Get actionable recommendation."""
        if result.opportunity_score >= 70:
            return "HIGH OPPORTUNITY - Great keyword to target!"
        elif result.opportunity_score >= 50:
            return "GOOD OPPORTUNITY - Worth creating content for"
        elif result.opportunity_score >= 30:
            return "MODERATE - Consider as supporting keyword"
        else:
            return "LOW OPPORTUNITY - High competition or low volume"

    def get_trending_topics(
        self,
        niche: str = "",
        region: str = "US",
        count: int = 10
    ) -> List[Dict]:
        """
        Get trending topics from Google Trends.

        Args:
            niche: Niche category
            region: Region code (US, GB, etc.)
            count: Number of topics to return

        Returns:
            List of trending topic dictionaries
        """
        if not self.pytrends:
            logger.warning("pytrends not available")
            return []

        try:
            trending = self.pytrends.trending_searches(pn=region.lower())
            topics = []

            for i, topic in enumerate(trending[0].tolist()[:count]):
                topics.append({
                    "rank": i + 1,
                    "topic": topic,
                    "region": region
                })

            logger.success(f"[FreeKeywordResearch] Found {len(topics)} trending topics")
            return topics

        except Exception as e:
            logger.error(f"Failed to get trending topics: {e}")
            return []


# CLI
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("""
Free Keyword Research - No Paid Tools Required!

Commands:
    find <seed_keyword> [--count <n>]
        Find keyword opportunities

    analyze <keyword>
        Detailed competition analysis

    trending [--region <code>]
        Get trending topics

Examples:
    python -m src.seo.free_keyword_research find "passive income" --count 50
    python -m src.seo.free_keyword_research analyze "make money online"
    python -m src.seo.free_keyword_research trending --region US
        """)
    else:
        researcher = FreeKeywordResearch()
        cmd = sys.argv[1]

        if cmd == "find" and len(sys.argv) >= 3:
            seed = sys.argv[2]
            count = 50

            if "--count" in sys.argv:
                idx = sys.argv.index("--count")
                if idx + 1 < len(sys.argv):
                    count = int(sys.argv[idx + 1])

            results = researcher.find_keywords(seed, count)

            print(f"\nTop {len(results)} Keyword Opportunities for '{seed}':\n")
            for i, result in enumerate(results[:20], 1):
                print(
                    f"{i}. {result.keyword} "
                    f"(Score: {result.opportunity_score:.0f}/100, "
                    f"Vol: {result.search_volume_estimate}, "
                    f"Comp: {result.competition})"
                )

        elif cmd == "analyze" and len(sys.argv) >= 3:
            keyword = sys.argv[2]
            analysis = researcher.analyze_competition(keyword)

            print(f"\nCompetition Analysis: {keyword}\n")
            print(f"Opportunity Score: {analysis['opportunity_score']:.0f}/100")
            print(f"Competition: {analysis['competition_level'].upper()}")
            print(f"Volume: {analysis['search_volume_estimate'].upper()}")
            print(f"Trend: {analysis['trend_direction'].upper()}")
            print(f"\n{analysis['recommendation']}\n")
            print(f"Related Keywords ({len(analysis['related_keywords'])}):")
            for kw in analysis['related_keywords']:
                print(f"  - {kw}")

        elif cmd == "trending":
            region = "US"
            if "--region" in sys.argv:
                idx = sys.argv.index("--region")
                if idx + 1 < len(sys.argv):
                    region = sys.argv[idx + 1]

            topics = researcher.get_trending_topics(region=region)

            print(f"\nTrending Topics in {region}:\n")
            for topic in topics:
                print(f"{topic['rank']}. {topic['topic']}")
