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
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests
import time

try:
    from pytrends.request import TrendReq
    from pytrends.exceptions import ResponseError
except ImportError:
    raise ImportError("Please install pytrends: pip install pytrends")


# Fallback trending topics when Google Trends API fails
FALLBACK_TOPICS: Dict[str, List[Dict[str, any]]] = {
    "python": [
        {"keyword": "python for beginners", "interest": 85, "direction": "rising"},
        {"keyword": "python automation scripts", "interest": 78, "direction": "rising"},
        {"keyword": "python data analysis", "interest": 82, "direction": "stable"},
        {"keyword": "python web scraping tutorial", "interest": 75, "direction": "rising"},
        {"keyword": "python API tutorial", "interest": 80, "direction": "rising"},
        {"keyword": "python machine learning", "interest": 88, "direction": "stable"},
        {"keyword": "python discord bot", "interest": 72, "direction": "rising"},
        {"keyword": "python game development", "interest": 70, "direction": "stable"},
    ],
    "programming": [
        {"keyword": "learn to code 2024", "interest": 80, "direction": "rising"},
        {"keyword": "best programming language", "interest": 75, "direction": "stable"},
        {"keyword": "coding interview tips", "interest": 82, "direction": "rising"},
        {"keyword": "web development tutorial", "interest": 85, "direction": "stable"},
        {"keyword": "software engineering career", "interest": 78, "direction": "rising"},
        {"keyword": "full stack developer roadmap", "interest": 80, "direction": "rising"},
        {"keyword": "coding projects for beginners", "interest": 83, "direction": "rising"},
    ],
    "javascript": [
        {"keyword": "javascript tutorial", "interest": 85, "direction": "stable"},
        {"keyword": "react js tutorial", "interest": 88, "direction": "rising"},
        {"keyword": "node js backend", "interest": 80, "direction": "stable"},
        {"keyword": "javascript frameworks 2024", "interest": 75, "direction": "rising"},
        {"keyword": "typescript tutorial", "interest": 82, "direction": "rising"},
        {"keyword": "next js tutorial", "interest": 85, "direction": "rising"},
    ],
    "ai": [
        {"keyword": "chatgpt tutorial", "interest": 90, "direction": "rising"},
        {"keyword": "ai tools for productivity", "interest": 88, "direction": "rising"},
        {"keyword": "machine learning basics", "interest": 82, "direction": "stable"},
        {"keyword": "ai image generator", "interest": 85, "direction": "rising"},
        {"keyword": "llm tutorial", "interest": 80, "direction": "rising"},
        {"keyword": "ai coding assistant", "interest": 87, "direction": "rising"},
    ],
    "tech": [
        {"keyword": "best laptops 2024", "interest": 80, "direction": "rising"},
        {"keyword": "tech tips and tricks", "interest": 75, "direction": "stable"},
        {"keyword": "smartphone comparison", "interest": 78, "direction": "stable"},
        {"keyword": "gadget reviews", "interest": 72, "direction": "stable"},
        {"keyword": "home automation setup", "interest": 76, "direction": "rising"},
    ],
    "gaming": [
        {"keyword": "gaming setup tour", "interest": 80, "direction": "stable"},
        {"keyword": "best gaming pc build", "interest": 82, "direction": "rising"},
        {"keyword": "game reviews 2024", "interest": 78, "direction": "stable"},
        {"keyword": "gaming tips and tricks", "interest": 75, "direction": "stable"},
        {"keyword": "esports highlights", "interest": 70, "direction": "stable"},
    ],
    "fitness": [
        {"keyword": "home workout routine", "interest": 85, "direction": "rising"},
        {"keyword": "weight loss tips", "interest": 88, "direction": "stable"},
        {"keyword": "muscle building for beginners", "interest": 80, "direction": "stable"},
        {"keyword": "healthy meal prep", "interest": 82, "direction": "rising"},
        {"keyword": "morning exercise routine", "interest": 78, "direction": "rising"},
    ],
    "finance": [
        {"keyword": "investing for beginners", "interest": 85, "direction": "rising"},
        {"keyword": "passive income ideas", "interest": 88, "direction": "rising"},
        {"keyword": "stock market basics", "interest": 80, "direction": "stable"},
        {"keyword": "cryptocurrency tutorial", "interest": 75, "direction": "stable"},
        {"keyword": "budgeting tips", "interest": 82, "direction": "rising"},
    ],
    "default": [
        {"keyword": "tutorial for beginners", "interest": 75, "direction": "stable"},
        {"keyword": "how to get started", "interest": 70, "direction": "stable"},
        {"keyword": "tips and tricks", "interest": 72, "direction": "stable"},
        {"keyword": "complete guide", "interest": 78, "direction": "stable"},
        {"keyword": "best practices", "interest": 74, "direction": "stable"},
    ],
}


class RateLimitError(Exception):
    """Custom exception for rate limiting."""
    pass


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
        self._last_request_time: float = 0
        self._min_request_interval: float = 1.0  # Minimum seconds between requests
        logger.info(f"TrendResearcher initialized for {geo}")

    def _rate_limit_delay(self) -> None:
        """Ensure minimum delay between API requests to avoid rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _get_fallback_topics(self, seed_keyword: str) -> List['TrendTopic']:
        """
        Get fallback topics when Google Trends API fails.

        Args:
            seed_keyword: The keyword to find fallback topics for

        Returns:
            List of TrendTopic objects from predefined fallbacks
        """
        seed_lower = seed_keyword.lower()

        # Find matching fallback category
        fallback_data = None
        for category, topics in FALLBACK_TOPICS.items():
            if category in seed_lower or seed_lower in category:
                fallback_data = topics
                break

        # Use default if no match found
        if fallback_data is None:
            fallback_data = FALLBACK_TOPICS["default"]
            logger.debug(f"Using default fallback topics for: {seed_keyword}")
        else:
            logger.debug(f"Using category-specific fallback topics for: {seed_keyword}")

        # Convert to TrendTopic objects
        trend_topics = []

        # Add seed keyword first
        trend_topics.append(TrendTopic(
            keyword=seed_keyword,
            interest_score=70,
            trend_direction="stable",
            related_queries=[t["keyword"] for t in fallback_data[:5]],
            related_topics=[]
        ))

        # Add fallback topics
        for topic_data in fallback_data[:5]:
            trend_topics.append(TrendTopic(
                keyword=topic_data["keyword"],
                interest_score=topic_data["interest"],
                trend_direction=topic_data["direction"],
                related_queries=[],
                related_topics=[]
            ))

        logger.info(f"Returning {len(trend_topics)} fallback topics for: {seed_keyword}")
        return trend_topics

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if an exception is a rate limit error."""
        error_str = str(error).lower()
        if hasattr(error, 'response') and hasattr(error.response, 'status_code'):
            if error.response.status_code == 429:
                return True
        return '429' in error_str or 'rate' in error_str or 'too many' in error_str

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type((requests.RequestException, ConnectionError, TimeoutError, RateLimitError))
    )
    def _build_payload_with_retry(self, kw_list: List[str], cat: int, timeframe: str, geo: str):
        """Build pytrends payload with retry logic for rate limiting."""
        self._rate_limit_delay()
        try:
            self.pytrends.build_payload(kw_list=kw_list, cat=cat, timeframe=timeframe, geo=geo)
        except Exception as e:
            if self._is_rate_limit_error(e):
                logger.warning(f"Rate limited on build_payload, will retry: {e}")
                raise RateLimitError(str(e))
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type((requests.RequestException, ConnectionError, TimeoutError, RateLimitError))
    )
    def _get_related_queries_with_retry(self):
        """Get related queries with retry logic."""
        self._rate_limit_delay()
        try:
            return self.pytrends.related_queries()
        except Exception as e:
            if self._is_rate_limit_error(e):
                logger.warning(f"Rate limited on related_queries, will retry: {e}")
                raise RateLimitError(str(e))
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type((requests.RequestException, ConnectionError, TimeoutError, RateLimitError))
    )
    def _get_related_topics_with_retry(self):
        """Get related topics with retry logic."""
        self._rate_limit_delay()
        try:
            return self.pytrends.related_topics()
        except Exception as e:
            if self._is_rate_limit_error(e):
                logger.warning(f"Rate limited on related_topics, will retry: {e}")
                raise RateLimitError(str(e))
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type((requests.RequestException, ConnectionError, TimeoutError, RateLimitError))
    )
    def _get_interest_over_time_with_retry(self):
        """Get interest over time with retry logic."""
        self._rate_limit_delay()
        try:
            return self.pytrends.interest_over_time()
        except Exception as e:
            if self._is_rate_limit_error(e):
                logger.warning(f"Rate limited on interest_over_time, will retry: {e}")
                raise RateLimitError(str(e))
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type((requests.RequestException, ConnectionError, TimeoutError, RateLimitError))
    )
    def _get_trending_searches_with_retry(self, pn: str):
        """Get trending searches with retry logic."""
        self._rate_limit_delay()
        try:
            return self.pytrends.trending_searches(pn=pn)
        except Exception as e:
            if self._is_rate_limit_error(e):
                logger.warning(f"Rate limited on trending_searches, will retry: {e}")
                raise RateLimitError(str(e))
            raise

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
            List of TrendTopic objects (never empty - falls back to predefined topics)
        """
        if not seed_keyword or not isinstance(seed_keyword, str):
            logger.warning("Invalid seed_keyword provided, using fallback topics")
            return self._get_fallback_topics("general")

        seed_keyword = seed_keyword.strip()
        if not seed_keyword:
            logger.warning("Empty seed_keyword provided, using fallback topics")
            return self._get_fallback_topics("general")

        logger.info(f"Researching trends for: {seed_keyword}")

        try:
            # Build payload with retry
            self._build_payload_with_retry(
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
                related = self._get_related_queries_with_retry()
                # Defensive checks for nested dict access
                if related is not None and isinstance(related, dict) and seed_keyword in related:
                    keyword_data = related[seed_keyword]
                    if isinstance(keyword_data, dict):
                        rising_data = keyword_data.get("rising")
                        top_data = keyword_data.get("top")

                        # Safe DataFrame access
                        if rising_data is not None and hasattr(rising_data, 'empty') and not rising_data.empty:
                            if "query" in rising_data.columns and len(rising_data) > 0:
                                rising_queries = rising_data["query"].tolist()[:10]
                        if top_data is not None and hasattr(top_data, 'empty') and not top_data.empty:
                            if "query" in top_data.columns and len(top_data) > 0:
                                top_queries = top_data["query"].tolist()[:10]
            except (KeyError, AttributeError, TypeError, IndexError) as e:
                logger.debug(f"Related queries unavailable: {e}")  # Continue without related queries

            # Get related topics with safe access
            try:
                topics = self._get_related_topics_with_retry()
                # Defensive checks for nested dict access
                if topics is not None and isinstance(topics, dict) and seed_keyword in topics:
                    keyword_topics = topics[seed_keyword]
                    if isinstance(keyword_topics, dict):
                        rising_topics_data = keyword_topics.get("rising")
                        # Safe DataFrame access
                        if rising_topics_data is not None and hasattr(rising_topics_data, 'empty') and not rising_topics_data.empty:
                            if "topic_title" in rising_topics_data.columns and len(rising_topics_data) > 0:
                                related_topics = rising_topics_data["topic_title"].tolist()[:10]
            except (KeyError, AttributeError, TypeError, IndexError) as e:
                logger.debug(f"Related topics unavailable: {e}")  # Continue without related topics

            # Get interest over time to determine trend direction
            trend_direction = "stable"
            interest_score = 50

            try:
                interest = self._get_interest_over_time_with_retry()
                # Defensive checks for DataFrame access
                if interest is not None and hasattr(interest, 'empty') and not interest.empty:
                    if seed_keyword in interest.columns:
                        values = interest[seed_keyword].values
                        if values is not None and len(values) >= 2:
                            recent = values[-4:].mean() if len(values) >= 4 else values[-1]
                            older = values[:4].mean() if len(values) >= 4 else values[0]
                            interest_score = int(recent) if recent is not None else 50

                            if recent > older * 1.2:
                                trend_direction = "rising"
                            elif recent < older * 0.8:
                                trend_direction = "declining"
            except (KeyError, AttributeError, TypeError, IndexError) as e:
                logger.debug(f"Interest over time unavailable: {e}")  # Continue with defaults

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

            # Add rising queries as separate topics (safe iteration)
            for query in rising_queries[:5] if rising_queries else []:
                if query and isinstance(query, str):
                    trend_topics.append(TrendTopic(
                        keyword=query,
                        interest_score=80,  # Rising = high interest
                        trend_direction="rising",
                        related_queries=[],
                        related_topics=[]
                    ))

            logger.success(f"Found {len(trend_topics)} trending topics")
            return trend_topics

        except (requests.RequestException, ConnectionError, TimeoutError, RateLimitError) as e:
            logger.warning(f"Trend research failed (network/rate limit): {e}")
            logger.info("Using fallback topics due to API failure")
            return self._get_fallback_topics(seed_keyword)
        except (KeyError, ValueError, IndexError, AttributeError, TypeError) as e:
            logger.warning(f"Trend research failed (data parsing): {e}")
            logger.info("Using fallback topics due to parsing error")
            return self._get_fallback_topics(seed_keyword)
        except Exception as e:
            logger.error(f"Unexpected error in trend research: {e}")
            logger.info("Using fallback topics due to unexpected error")
            return self._get_fallback_topics(seed_keyword)

    def get_realtime_trends(self, category: str = "all") -> List[str]:
        """
        Get current realtime trending searches.

        Args:
            category: Category filter (all, business, entertainment, etc.)

        Returns:
            List of trending search terms (may return fallback topics if API fails)
        """
        logger.info("Fetching realtime trends...")

        try:
            # Get trending searches with retry
            trends = self._get_trending_searches_with_retry(pn=self.geo.lower())

            # Defensive check for DataFrame structure
            if trends is not None and hasattr(trends, 'empty') and not trends.empty:
                # Check if column 0 exists before accessing
                if len(trends.columns) > 0 and 0 in trends.columns:
                    result = trends[0].tolist()
                    if result and len(result) > 0:
                        return result[:20]
                # Try first column by index if 0 not available
                elif len(trends.columns) > 0:
                    first_col = trends.columns[0]
                    result = trends[first_col].tolist()
                    if result and len(result) > 0:
                        return result[:20]

            logger.warning("No realtime trends data available")
            return []

        except (requests.RequestException, ConnectionError, TimeoutError, RateLimitError) as e:
            logger.warning(f"Realtime trends failed (network/rate limit): {e}")
            return []
        except (KeyError, ValueError, IndexError, AttributeError, TypeError) as e:
            logger.warning(f"Realtime trends failed (data parsing): {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in realtime trends: {e}")
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
        # Input validation
        if not keywords or not isinstance(keywords, list):
            logger.warning("Invalid keywords list provided")
            return {}

        # Filter out invalid keywords
        valid_keywords = [k for k in keywords if k and isinstance(k, str) and k.strip()]
        if not valid_keywords:
            logger.warning("No valid keywords provided")
            return {}

        if len(valid_keywords) > 5:
            valid_keywords = valid_keywords[:5]
            logger.warning("Truncated to 5 keywords (Google Trends limit)")

        logger.info(f"Comparing {len(valid_keywords)} keywords...")

        try:
            self._build_payload_with_retry(
                kw_list=valid_keywords,
                cat=0,
                timeframe=timeframe,
                geo=self.geo
            )

            interest = self._get_interest_over_time_with_retry()

            # Defensive check for DataFrame
            if interest is None or not hasattr(interest, 'empty') or interest.empty:
                return {k: 0 for k in valid_keywords}

            # Get average interest for each keyword with safe access
            scores = {}
            for keyword in valid_keywords:
                try:
                    if keyword in interest.columns:
                        mean_val = interest[keyword].mean()
                        scores[keyword] = int(mean_val) if mean_val is not None else 0
                    else:
                        scores[keyword] = 0
                except (KeyError, TypeError, ValueError) as e:
                    logger.debug(f"Could not get score for '{keyword}': {e}")
                    scores[keyword] = 0

            return scores

        except (requests.RequestException, ConnectionError, TimeoutError, RateLimitError) as e:
            logger.warning(f"Keyword comparison failed (network/rate limit): {e}")
            return {k: 0 for k in valid_keywords}
        except (KeyError, ValueError, IndexError, AttributeError, TypeError) as e:
            logger.warning(f"Keyword comparison failed (data parsing): {e}")
            return {k: 0 for k in valid_keywords}
        except Exception as e:
            logger.error(f"Unexpected error in keyword comparison: {e}")
            return {k: 0 for k in valid_keywords}

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
        # Input validation
        if not keyword or not isinstance(keyword, str) or not keyword.strip():
            logger.warning("Invalid keyword provided for seasonal trends")
            return {}

        keyword = keyword.strip()
        logger.info(f"Analyzing seasonal trends for: {keyword}")

        try:
            self._build_payload_with_retry(
                kw_list=[keyword],
                cat=0,
                timeframe=f"today {years*12}-m",
                geo=self.geo
            )

            interest = self._get_interest_over_time_with_retry()

            # Defensive check for DataFrame
            if interest is None or not hasattr(interest, 'empty') or interest.empty:
                return {}

            # Check if keyword column exists
            if keyword not in interest.columns:
                logger.warning(f"Keyword '{keyword}' not found in interest data")
                return {}

            # Defensive check for index and groupby
            if not hasattr(interest.index, 'month'):
                logger.warning("Interest data index does not have month attribute")
                return {}

            # Group by month with error handling
            try:
                monthly = interest.groupby(interest.index.month)[keyword].mean()
            except (KeyError, AttributeError, TypeError) as e:
                logger.warning(f"Could not group by month: {e}")
                return {}

            month_names = [
                "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
            ]

            # Build result with safe access
            result = {}
            for i in range(1, 13):
                try:
                    val = monthly.get(i, 0)
                    result[month_names[i-1]] = round(float(val), 1) if val is not None else 0.0
                except (TypeError, ValueError, IndexError):
                    result[month_names[i-1]] = 0.0

            return result

        except (requests.RequestException, ConnectionError, TimeoutError, RateLimitError) as e:
            logger.warning(f"Seasonal analysis failed (network/rate limit): {e}")
            return {}
        except (KeyError, ValueError, IndexError, AttributeError, TypeError) as e:
            logger.warning(f"Seasonal analysis failed (data parsing): {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error in seasonal analysis: {e}")
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
