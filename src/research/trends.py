"""
Google Trends Research Module

Finds trending topics using Google Trends API (pytrends).
No API key required!

Usage:
    researcher = TrendResearcher()
    trends = researcher.get_trending_topics("python programming")
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

try:
    from pytrends.request import TrendReq
except ImportError:
    raise ImportError("Please install pytrends: pip install pytrends")


@dataclass
class TrendTopic:
    """Represents a trending topic."""
    keyword: str
    interest_score: int      # 0-100 relative interest
    trend_direction: str     # rising, stable, declining
    related_queries: List[str]
    related_topics: List[str]


class TrendResearcher:
    """Research trending topics using Google Trends."""

    def __init__(self, geo: str = "US", language: str = "en-US"):
        """
        Initialize the trend researcher.

        Args:
            geo: Geographic location (US, GB, etc.)
            language: Language code
        """
        self.geo = geo
        self.language = language
        self.pytrends = TrendReq(hl=language, tz=360)
        logger.info(f"TrendResearcher initialized for {geo}")

    def get_trending_topics(
        self,
        seed_keyword: str,
        category: int = 0,
        timeframe: str = "today 3-m"
    ) -> List[TrendTopic]:
        """
        Get trending topics related to a seed keyword.

        Args:
            seed_keyword: Starting keyword to research
            category: Google category ID (0=all, 5=computers, 13=programming)
            timeframe: Time range (today 3-m, today 12-m, now 7-d)

        Returns:
            List of TrendTopic objects
        """
        logger.info(f"Researching trends for: {seed_keyword}")

        try:
            # Build payload
            self.pytrends.build_payload(
                kw_list=[seed_keyword],
                cat=category,
                timeframe=timeframe,
                geo=self.geo
            )

            # Get related queries with safe access
            rising_queries = []
            top_queries = []
            related_topics = []

            try:
                related = self.pytrends.related_queries()
                if related and seed_keyword in related:
                    rising_data = related[seed_keyword].get("rising")
                    top_data = related[seed_keyword].get("top")

                    if rising_data is not None and not rising_data.empty:
                        rising_queries = rising_data["query"].tolist()[:10]
                    if top_data is not None and not top_data.empty:
                        top_queries = top_data["query"].tolist()[:10]
            except Exception:
                pass  # Continue without related queries

            # Get related topics with safe access
            try:
                topics = self.pytrends.related_topics()
                if topics and seed_keyword in topics:
                    rising_topics = topics[seed_keyword].get("rising")
                    if rising_topics is not None and not rising_topics.empty:
                        related_topics = rising_topics["topic_title"].tolist()[:10]
            except Exception:
                pass  # Continue without related topics

            # Get interest over time to determine trend direction
            interest = self.pytrends.interest_over_time()
            trend_direction = "stable"
            interest_score = 50

            if not interest.empty and seed_keyword in interest.columns:
                values = interest[seed_keyword].values
                if len(values) >= 2:
                    recent = values[-4:].mean() if len(values) >= 4 else values[-1]
                    older = values[:4].mean() if len(values) >= 4 else values[0]
                    interest_score = int(recent)

                    if recent > older * 1.2:
                        trend_direction = "rising"
                    elif recent < older * 0.8:
                        trend_direction = "declining"

            # Create trend topics from rising queries
            trend_topics = []

            # Add the seed keyword
            trend_topics.append(TrendTopic(
                keyword=seed_keyword,
                interest_score=interest_score,
                trend_direction=trend_direction,
                related_queries=rising_queries + top_queries,
                related_topics=related_topics
            ))

            # Add rising queries as separate topics
            for query in rising_queries[:5]:
                trend_topics.append(TrendTopic(
                    keyword=query,
                    interest_score=80,  # Rising = high interest
                    trend_direction="rising",
                    related_queries=[],
                    related_topics=[]
                ))

            logger.success(f"Found {len(trend_topics)} trending topics")
            return trend_topics

        except Exception as e:
            logger.error(f"Trend research failed: {e}")
            return []

    def get_realtime_trends(self, category: str = "all") -> List[str]:
        """
        Get current realtime trending searches.

        Args:
            category: Category filter (all, business, entertainment, etc.)

        Returns:
            List of trending search terms
        """
        logger.info("Fetching realtime trends...")

        try:
            # Get trending searches
            trends = self.pytrends.trending_searches(pn=self.geo.lower())

            if trends is not None and not trends.empty:
                return trends[0].tolist()[:20]
            return []

        except Exception as e:
            logger.error(f"Realtime trends failed: {e}")
            return []

    def compare_keywords(
        self,
        keywords: List[str],
        timeframe: str = "today 3-m"
    ) -> Dict[str, int]:
        """
        Compare interest levels between multiple keywords.

        Args:
            keywords: List of keywords to compare (max 5)
            timeframe: Time range for comparison

        Returns:
            Dict mapping keyword to relative interest score
        """
        if len(keywords) > 5:
            keywords = keywords[:5]
            logger.warning("Truncated to 5 keywords (Google Trends limit)")

        logger.info(f"Comparing {len(keywords)} keywords...")

        try:
            self.pytrends.build_payload(
                kw_list=keywords,
                timeframe=timeframe,
                geo=self.geo
            )

            interest = self.pytrends.interest_over_time()

            if interest.empty:
                return {k: 0 for k in keywords}

            # Get average interest for each keyword
            scores = {}
            for keyword in keywords:
                if keyword in interest.columns:
                    scores[keyword] = int(interest[keyword].mean())
                else:
                    scores[keyword] = 0

            return scores

        except Exception as e:
            logger.error(f"Keyword comparison failed: {e}")
            return {k: 0 for k in keywords}

    def get_seasonal_trends(
        self,
        keyword: str,
        years: int = 5
    ) -> Dict[str, float]:
        """
        Analyze seasonal patterns for a keyword.

        Args:
            keyword: Keyword to analyze
            years: Number of years to analyze

        Returns:
            Dict mapping month names to average interest
        """
        logger.info(f"Analyzing seasonal trends for: {keyword}")

        try:
            self.pytrends.build_payload(
                kw_list=[keyword],
                timeframe=f"today {years*12}-m",
                geo=self.geo
            )

            interest = self.pytrends.interest_over_time()

            if interest.empty:
                return {}

            # Group by month
            monthly = interest.groupby(interest.index.month)[keyword].mean()

            month_names = [
                "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
            ]

            return {
                month_names[i-1]: round(monthly.get(i, 0), 1)
                for i in range(1, 13)
            }

        except Exception as e:
            logger.error(f"Seasonal analysis failed: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    researcher = TrendResearcher()

    # Get trending topics
    print("\n" + "="*60)
    print("TRENDING TOPICS FOR 'Python Programming'")
    print("="*60 + "\n")

    trends = researcher.get_trending_topics("python programming")
    for trend in trends:
        print(f"  {trend.keyword}")
        print(f"    Interest: {trend.interest_score}/100 ({trend.trend_direction})")
        if trend.related_queries:
            print(f"    Related: {', '.join(trend.related_queries[:3])}")
        print()

    # Compare keywords
    print("\n" + "="*60)
    print("KEYWORD COMPARISON")
    print("="*60 + "\n")

    comparison = researcher.compare_keywords([
        "python tutorial",
        "javascript tutorial",
        "react tutorial"
    ])

    for keyword, score in sorted(comparison.items(), key=lambda x: -x[1]):
        print(f"  {keyword}: {score}/100")
