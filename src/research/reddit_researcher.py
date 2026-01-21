"""
Reddit Research Module for YouTube Automation

Comprehensive Reddit research for content discovery and trend analysis.
Uses PRAW (Python Reddit API Wrapper) for API access.

Setup:
    1. Go to https://www.reddit.com/prefs/apps
    2. Click "create another app..."
    3. Select "script" type
    4. Name: "youtube-automation-research" (or any name)
    5. Redirect URI: http://localhost:8080 (not used but required)
    6. Get client_id (under app name) and client_secret
    7. Add to config/.env:
        REDDIT_CLIENT_ID=your_client_id
        REDDIT_CLIENT_SECRET=your_client_secret
        REDDIT_USER_AGENT=youtube-automation/1.0

Usage:
    from src.research.reddit_researcher import RedditResearcher

    researcher = RedditResearcher()

    # Get trending topics for a niche
    trends = researcher.get_trending_topics("finance", limit=20)

    # Find viral content ideas
    viral = researcher.find_viral_content("psychology", min_upvotes=500)

    # Extract questions for FAQ videos
    questions = researcher.get_questions("storytelling", limit=50)

    # Get full research report
    report = researcher.full_research("finance")
"""

import os
import re
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import Counter
from pathlib import Path
from loguru import logger

try:
    import praw
    from praw.models import Submission, Comment
except ImportError:
    logger.error("PRAW not installed. Run: pip install praw")
    raise ImportError("Please install praw: pip install praw")

try:
    import yaml
except ImportError:
    yaml = None


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class RedditPost:
    """Represents a Reddit post with full metadata."""
    id: str
    title: str
    subreddit: str
    author: str
    score: int
    upvote_ratio: float
    num_comments: int
    url: str
    permalink: str
    created_utc: datetime
    selftext: str
    flair: Optional[str]
    is_question: bool
    is_self: bool
    viral_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            'created_utc': self.created_utc.isoformat(),
        }


@dataclass
class VideoIdea:
    """A potential video idea extracted from Reddit."""
    topic: str
    title_suggestion: str
    source_title: str
    source_url: str
    subreddit: str
    popularity_score: int
    viral_score: float
    idea_type: str  # question, discussion, request, tutorial, viral
    keywords: List[str] = field(default_factory=list)
    related_posts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SubredditStats:
    """Statistics for a subreddit."""
    name: str
    subscribers: int
    active_users: Optional[int]
    avg_post_score: float
    avg_comments: float
    posts_per_day: float
    top_flairs: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class RedditResearchReport:
    """Complete research report for a niche."""
    niche: str
    timestamp: datetime
    trending_topics: List[str]
    viral_ideas: List[VideoIdea]
    questions: List[VideoIdea]
    sentiment_summary: Dict[str, float]
    subreddit_stats: List[SubredditStats]
    keyword_frequency: Dict[str, int]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'niche': self.niche,
            'timestamp': self.timestamp.isoformat(),
            'trending_topics': self.trending_topics,
            'viral_ideas': [i.to_dict() for i in self.viral_ideas],
            'questions': [q.to_dict() for q in self.questions],
            'sentiment_summary': self.sentiment_summary,
            'subreddit_stats': [s.to_dict() for s in self.subreddit_stats],
            'keyword_frequency': self.keyword_frequency,
            'recommendations': self.recommendations,
        }

    def summary(self) -> str:
        """Generate a text summary of the report."""
        lines = [
            f"\n{'='*60}",
            f"REDDIT RESEARCH REPORT: {self.niche.upper()}",
            f"{'='*60}",
            f"Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M')}",
            f"\nTrending Topics ({len(self.trending_topics)}):",
        ]
        for i, topic in enumerate(self.trending_topics[:10], 1):
            lines.append(f"  {i}. {topic}")

        lines.append(f"\nViral Ideas ({len(self.viral_ideas)}):")
        for i, idea in enumerate(self.viral_ideas[:5], 1):
            lines.append(f"  {i}. {idea.title_suggestion}")
            lines.append(f"     Score: {idea.viral_score:.0f} | Type: {idea.idea_type}")

        lines.append(f"\nTop Questions ({len(self.questions)}):")
        for i, q in enumerate(self.questions[:5], 1):
            lines.append(f"  {i}. {q.topic}")

        lines.append(f"\nRecommendations:")
        for rec in self.recommendations:
            lines.append(f"  - {rec}")

        lines.append("="*60)
        return "\n".join(lines)


# ============================================================
# SUBREDDIT CONFIGURATION
# ============================================================

# Default subreddits by niche (fallback if YAML not available)
DEFAULT_SUBREDDITS = {
    "finance": {
        "subreddits": [
            "personalfinance",
            "financialindependence",
            "investing",
            "stocks",
            "passive_income",
            "Fire",
            "Bogleheads",
            "dividends",
            "wallstreetbets",
            "options",
        ],
        "min_upvotes": 100,
        "keywords": ["money", "invest", "stock", "wealth", "income", "budget", "savings"],
    },
    "psychology": {
        "subreddits": [
            "psychology",
            "selfimprovement",
            "getdisciplined",
            "productivity",
            "DecidingToBeBetter",
            "socialskills",
            "confidence",
            "introvert",
            "mentalhealth",
            "Stoicism",
        ],
        "min_upvotes": 50,
        "keywords": ["mind", "brain", "habit", "motivation", "anxiety", "behavior", "therapy"],
    },
    "storytelling": {
        "subreddits": [
            "nosleep",
            "tifu",
            "AmItheAsshole",
            "relationship_advice",
            "TrueOffMyChest",
            "pettyrevenge",
            "ProRevenge",
            "MaliciousCompliance",
            "entitledparents",
            "talesfromtechsupport",
        ],
        "min_upvotes": 500,
        "keywords": ["story", "happened", "experience", "never", "finally", "realized"],
    },
    "technology": {
        "subreddits": [
            "technology",
            "programming",
            "webdev",
            "learnprogramming",
            "Python",
            "javascript",
            "MachineLearning",
            "artificial",
            "gadgets",
            "Futurology",
        ],
        "min_upvotes": 100,
        "keywords": ["code", "developer", "software", "AI", "tech", "build", "learn"],
    },
}


# ============================================================
# MAIN RESEARCHER CLASS
# ============================================================

class RedditResearcher:
    """
    Comprehensive Reddit research for YouTube content discovery.

    Features:
    - Subreddit monitoring by niche
    - Hot/Rising/Top post fetching
    - Comment sentiment analysis
    - Question extraction for FAQ videos
    - Viral score calculation
    - Rate limiting handling
    """

    # Keywords indicating questions/requests
    QUESTION_PATTERNS = [
        r"^how (do|can|to|does|did|would|should|is|are)",
        r"^what (is|are|does|would|should|makes|causes)",
        r"^why (do|does|did|is|are|would|should|can't|don't)",
        r"^when (should|is|are|do|does)",
        r"^where (can|do|should|is)",
        r"^who (is|are|should|can|would)",
        r"^is it (possible|true|worth|better|normal|okay)",
        r"^can (i|you|someone|anyone|we)",
        r"^should (i|you|we|someone)",
        r"^does (anyone|someone|this|it)",
        r"^has (anyone|someone)",
        r"^eli5",
        r"\?$",
        r"^help",
        r"need advice",
        r"looking for advice",
        r"am i wrong",
    ]

    # Sentiment keywords
    POSITIVE_KEYWORDS = [
        "love", "amazing", "great", "excellent", "awesome", "best",
        "wonderful", "fantastic", "perfect", "brilliant", "incredible",
        "helpful", "thank", "grateful", "finally", "success", "worked",
    ]

    NEGATIVE_KEYWORDS = [
        "hate", "terrible", "awful", "worst", "horrible", "bad",
        "disappointed", "frustrated", "angry", "annoying", "useless",
        "scam", "avoid", "warning", "mistake", "regret", "failed",
    ]

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize Reddit researcher.

        Args:
            client_id: Reddit app client ID (or from REDDIT_CLIENT_ID env)
            client_secret: Reddit app client secret (or from REDDIT_CLIENT_SECRET env)
            user_agent: User agent string (or from REDDIT_USER_AGENT env)
            config_path: Path to subreddits.yaml config file
        """
        self.client_id = client_id or os.getenv("REDDIT_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET")
        self.user_agent = user_agent or os.getenv(
            "REDDIT_USER_AGENT",
            "youtube-automation-research/1.0"
        )

        # Rate limiting tracking
        self._request_count = 0
        self._last_request_time = 0
        self._rate_limit_remaining = 100

        # Initialize Reddit client
        self.reddit = None
        if self.client_id and self.client_secret:
            try:
                self.reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent,
                )
                # Test connection
                _ = self.reddit.user.me()
                logger.info("Reddit researcher initialized successfully")
            except Exception as e:
                logger.warning(f"Reddit authentication failed (read-only mode): {e}")
                # Try read-only mode
                try:
                    self.reddit = praw.Reddit(
                        client_id=self.client_id,
                        client_secret=self.client_secret,
                        user_agent=self.user_agent,
                    )
                    self.reddit.read_only = True
                    logger.info("Reddit researcher initialized in read-only mode")
                except Exception as e2:
                    logger.error(f"Reddit initialization failed: {e2}")
                    self.reddit = None
        else:
            logger.warning(
                "Reddit credentials not found. "
                "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env file. "
                "See module docstring for setup instructions."
            )

        # Load subreddit configuration
        self.subreddit_config = self._load_subreddit_config(config_path)

    def _load_subreddit_config(self, config_path: Optional[str] = None) -> Dict:
        """Load subreddit configuration from YAML or use defaults."""
        if config_path is None:
            # Try default paths
            possible_paths = [
                "config/subreddits.yaml",
                Path(__file__).parent.parent.parent / "config" / "subreddits.yaml",
            ]
            for path in possible_paths:
                if Path(path).exists():
                    config_path = str(path)
                    break

        if config_path and Path(config_path).exists() and yaml:
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    logger.info(f"Loaded subreddit config from {config_path}")
                    return config
            except Exception as e:
                logger.warning(f"Failed to load subreddit config: {e}")

        logger.info("Using default subreddit configuration")
        return DEFAULT_SUBREDDITS

    def _rate_limit_wait(self):
        """Implement rate limiting to avoid API bans."""
        self._request_count += 1

        # Reddit rate limit: 60 requests per minute
        if self._request_count >= 55:  # Leave some buffer
            elapsed = time.time() - self._last_request_time
            if elapsed < 60:
                wait_time = 60 - elapsed
                logger.debug(f"Rate limiting: waiting {wait_time:.1f}s")
                time.sleep(wait_time)
            self._request_count = 0
            self._last_request_time = time.time()

    def _is_question(self, text: str) -> bool:
        """Check if text appears to be a question or request."""
        text_lower = text.lower().strip()

        for pattern in self.QUESTION_PATTERNS:
            if re.search(pattern, text_lower):
                return True

        return False

    def _calculate_viral_score(self, post: Submission) -> float:
        """
        Calculate a viral potential score for a post.

        Factors:
        - Upvotes (weighted)
        - Upvote ratio (quality indicator)
        - Comments (engagement)
        - Freshness (recent posts valued higher)
        - Award count (social proof)
        """
        try:
            # Base score from upvotes
            score = post.score * 1.0

            # Upvote ratio multiplier (0.5-1.0 maps to 0.7-1.3)
            ratio_multiplier = 0.7 + (post.upvote_ratio * 0.6)
            score *= ratio_multiplier

            # Comment engagement bonus
            if post.num_comments > 0:
                engagement_ratio = min(post.num_comments / max(post.score, 1), 0.5)
                score *= (1 + engagement_ratio)

            # Freshness bonus (posts from last 24h get up to 50% boost)
            age_hours = (datetime.now() - datetime.fromtimestamp(post.created_utc)).total_seconds() / 3600
            if age_hours < 24:
                freshness_bonus = 1 + (0.5 * (1 - age_hours / 24))
                score *= freshness_bonus

            # Question bonus (more tutorial potential)
            if self._is_question(post.title):
                score *= 1.3

            return round(score, 2)

        except Exception as e:
            logger.debug(f"Error calculating viral score: {e}")
            return post.score * 1.0

    def _analyze_sentiment(self, texts: List[str]) -> Dict[str, float]:
        """
        Simple sentiment analysis on text list.

        Returns:
            Dict with positive, negative, neutral percentages
        """
        if not texts:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for text in texts:
            text_lower = text.lower()

            pos_hits = sum(1 for kw in self.POSITIVE_KEYWORDS if kw in text_lower)
            neg_hits = sum(1 for kw in self.NEGATIVE_KEYWORDS if kw in text_lower)

            if pos_hits > neg_hits:
                positive_count += 1
            elif neg_hits > pos_hits:
                negative_count += 1
            else:
                neutral_count += 1

        total = len(texts)
        return {
            "positive": round(positive_count / total, 3),
            "negative": round(negative_count / total, 3),
            "neutral": round(neutral_count / total, 3),
        }

    def _extract_keywords(self, texts: List[str], top_n: int = 20) -> Dict[str, int]:
        """Extract most common meaningful keywords from texts."""
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "to", "of",
            "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
            "during", "before", "after", "above", "below", "between", "under",
            "again", "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
            "very", "just", "i", "me", "my", "we", "our", "you", "your", "it",
            "its", "this", "that", "these", "those", "am", "or", "and", "but",
            "if", "because", "until", "while", "about", "any", "both", "which",
            "who", "whom", "up", "down", "out", "over", "like", "get", "got",
            "also", "really", "even", "still", "now", "one", "two", "three",
            "first", "new", "old", "good", "bad", "right", "wrong", "way",
        }

        words = []
        for text in texts:
            # Clean and tokenize
            text_clean = re.sub(r'[^\w\s]', ' ', text.lower())
            text_words = [w for w in text_clean.split()
                         if w not in stopwords and len(w) > 3]
            words.extend(text_words)

        counter = Counter(words)
        return dict(counter.most_common(top_n))

    def _post_to_dataclass(self, post: Submission) -> RedditPost:
        """Convert PRAW submission to RedditPost dataclass."""
        return RedditPost(
            id=post.id,
            title=post.title,
            subreddit=post.subreddit.display_name,
            author=str(post.author) if post.author else "[deleted]",
            score=post.score,
            upvote_ratio=post.upvote_ratio,
            num_comments=post.num_comments,
            url=post.url,
            permalink=f"https://reddit.com{post.permalink}",
            created_utc=datetime.fromtimestamp(post.created_utc),
            selftext=post.selftext[:1000] if post.selftext else "",
            flair=post.link_flair_text,
            is_question=self._is_question(post.title),
            is_self=post.is_self,
            viral_score=self._calculate_viral_score(post),
        )

    def _post_to_idea(self, post: RedditPost, idea_type: str = "discussion") -> VideoIdea:
        """Convert RedditPost to VideoIdea."""
        # Generate a video title suggestion
        title = post.title

        # Clean up common patterns
        title = re.sub(r'\[.*?\]', '', title)  # Remove [tags]
        title = re.sub(r'\(.*?\)', '', title)  # Remove (parentheticals)
        title = title.strip()

        # Capitalize properly
        if title:
            title = title[0].upper() + title[1:]

        # Remove trailing punctuation for title
        title_clean = title.rstrip('?!.')

        return VideoIdea(
            topic=title_clean,
            title_suggestion=title_clean[:60] if len(title_clean) > 60 else title_clean,
            source_title=post.title,
            source_url=post.permalink,
            subreddit=post.subreddit,
            popularity_score=post.score,
            viral_score=post.viral_score,
            idea_type=idea_type,
            keywords=[],
        )

    # ============================================================
    # PUBLIC API METHODS
    # ============================================================

    def get_subreddits_for_niche(self, niche: str) -> List[str]:
        """Get list of subreddits for a niche."""
        niche_lower = niche.lower()

        if niche_lower in self.subreddit_config:
            config = self.subreddit_config[niche_lower]
            if isinstance(config, dict):
                return config.get("subreddits", [])
            return config

        # Fallback to defaults
        if niche_lower in DEFAULT_SUBREDDITS:
            return DEFAULT_SUBREDDITS[niche_lower]["subreddits"]

        logger.warning(f"No subreddits configured for niche: {niche}")
        return []

    def get_min_upvotes_for_niche(self, niche: str) -> int:
        """Get minimum upvotes threshold for a niche."""
        niche_lower = niche.lower()

        if niche_lower in self.subreddit_config:
            config = self.subreddit_config[niche_lower]
            if isinstance(config, dict):
                return config.get("min_upvotes", 50)

        if niche_lower in DEFAULT_SUBREDDITS:
            return DEFAULT_SUBREDDITS[niche_lower]["min_upvotes"]

        return 50  # Default

    def get_hot_posts(
        self,
        niche: str,
        limit: int = 50,
        min_upvotes: Optional[int] = None
    ) -> List[RedditPost]:
        """
        Get hot posts from subreddits in a niche.

        Args:
            niche: Content niche (finance, psychology, storytelling, etc.)
            limit: Max posts to fetch
            min_upvotes: Minimum upvotes filter (uses niche default if None)

        Returns:
            List of RedditPost objects sorted by viral score
        """
        if not self.reddit:
            logger.error("Reddit not initialized. Check credentials.")
            return []

        subreddits = self.get_subreddits_for_niche(niche)
        if not subreddits:
            logger.warning(f"No subreddits for niche: {niche}")
            return []

        min_score = min_upvotes or self.get_min_upvotes_for_niche(niche)
        posts = []

        for subreddit_name in subreddits:
            try:
                self._rate_limit_wait()
                subreddit = self.reddit.subreddit(subreddit_name)

                for post in subreddit.hot(limit=limit // len(subreddits) + 5):
                    if post.score >= min_score and not post.stickied:
                        posts.append(self._post_to_dataclass(post))

                logger.debug(f"Fetched posts from r/{subreddit_name}")

            except Exception as e:
                logger.warning(f"Error fetching r/{subreddit_name}: {e}")
                continue

        # Sort by viral score
        posts.sort(key=lambda p: p.viral_score, reverse=True)

        logger.info(f"Found {len(posts)} hot posts for {niche}")
        return posts[:limit]

    def get_rising_posts(
        self,
        niche: str,
        limit: int = 30
    ) -> List[RedditPost]:
        """
        Get rising posts (potential viral content).

        Rising posts are newer but gaining traction quickly.
        """
        if not self.reddit:
            logger.error("Reddit not initialized.")
            return []

        subreddits = self.get_subreddits_for_niche(niche)
        posts = []

        for subreddit_name in subreddits:
            try:
                self._rate_limit_wait()
                subreddit = self.reddit.subreddit(subreddit_name)

                for post in subreddit.rising(limit=10):
                    if not post.stickied:
                        posts.append(self._post_to_dataclass(post))

            except Exception as e:
                logger.warning(f"Error fetching rising from r/{subreddit_name}: {e}")
                continue

        posts.sort(key=lambda p: p.viral_score, reverse=True)

        logger.info(f"Found {len(posts)} rising posts for {niche}")
        return posts[:limit]

    def get_top_posts(
        self,
        niche: str,
        time_filter: str = "week",
        limit: int = 50,
        min_upvotes: Optional[int] = None
    ) -> List[RedditPost]:
        """
        Get top posts from a time period.

        Args:
            niche: Content niche
            time_filter: hour, day, week, month, year, all
            limit: Max posts to return
            min_upvotes: Minimum upvotes

        Returns:
            List of top posts sorted by viral score
        """
        if not self.reddit:
            logger.error("Reddit not initialized.")
            return []

        subreddits = self.get_subreddits_for_niche(niche)
        min_score = min_upvotes or self.get_min_upvotes_for_niche(niche)
        posts = []

        for subreddit_name in subreddits:
            try:
                self._rate_limit_wait()
                subreddit = self.reddit.subreddit(subreddit_name)

                for post in subreddit.top(time_filter=time_filter, limit=limit // len(subreddits) + 5):
                    if post.score >= min_score:
                        posts.append(self._post_to_dataclass(post))

            except Exception as e:
                logger.warning(f"Error fetching top from r/{subreddit_name}: {e}")
                continue

        posts.sort(key=lambda p: p.viral_score, reverse=True)

        logger.info(f"Found {len(posts)} top posts for {niche} ({time_filter})")
        return posts[:limit]

    def get_trending_topics(
        self,
        niche: str,
        limit: int = 20
    ) -> List[str]:
        """
        Extract trending topics from hot and rising posts.

        Returns list of topic strings suitable for video titles.
        """
        hot_posts = self.get_hot_posts(niche, limit=30)
        rising_posts = self.get_rising_posts(niche, limit=15)

        all_posts = hot_posts + rising_posts

        # Extract keywords and generate topics
        titles = [p.title for p in all_posts]
        keywords = self._extract_keywords(titles, top_n=30)

        # Generate topic strings from top posts
        topics = []
        seen = set()

        for post in sorted(all_posts, key=lambda p: p.viral_score, reverse=True):
            # Clean and normalize topic
            topic = post.title
            topic = re.sub(r'\[.*?\]', '', topic)
            topic = re.sub(r'\(.*?\)', '', topic)
            topic = topic.strip()

            if topic and topic.lower() not in seen:
                seen.add(topic.lower())
                topics.append(topic)

        logger.success(f"Found {len(topics)} trending topics for {niche}")
        return topics[:limit]

    def get_questions(
        self,
        niche: str,
        limit: int = 50,
        min_upvotes: int = 10
    ) -> List[VideoIdea]:
        """
        Extract questions from Reddit for FAQ-style videos.

        Questions are great for tutorial content because they
        directly address what people want to know.
        """
        if not self.reddit:
            logger.error("Reddit not initialized.")
            return []

        subreddits = self.get_subreddits_for_niche(niche)
        questions = []

        for subreddit_name in subreddits:
            try:
                self._rate_limit_wait()
                subreddit = self.reddit.subreddit(subreddit_name)

                # Search for question-like posts
                for post in subreddit.top(time_filter="month", limit=100):
                    if post.score >= min_upvotes and self._is_question(post.title):
                        reddit_post = self._post_to_dataclass(post)
                        idea = self._post_to_idea(reddit_post, "question")
                        questions.append(idea)

            except Exception as e:
                logger.warning(f"Error fetching questions from r/{subreddit_name}: {e}")
                continue

        # Sort by viral score
        questions.sort(key=lambda q: q.viral_score, reverse=True)

        # Deduplicate similar questions
        unique_questions = []
        seen_topics = set()

        for q in questions:
            # Simple dedup by first few words
            key = ' '.join(q.topic.lower().split()[:5])
            if key not in seen_topics:
                seen_topics.add(key)
                unique_questions.append(q)

        logger.success(f"Found {len(unique_questions)} unique questions for {niche}")
        return unique_questions[:limit]

    def find_viral_content(
        self,
        niche: str,
        min_upvotes: Optional[int] = None,
        limit: int = 20
    ) -> List[VideoIdea]:
        """
        Find viral content ideas with high engagement potential.

        Combines hot, rising, and top posts to find the best
        content opportunities.
        """
        min_score = min_upvotes or self.get_min_upvotes_for_niche(niche) * 2

        # Get posts from multiple sources
        hot = self.get_hot_posts(niche, limit=30, min_upvotes=min_score)
        rising = self.get_rising_posts(niche, limit=20)
        top_week = self.get_top_posts(niche, "week", limit=30, min_upvotes=min_score)

        # Combine and deduplicate
        all_posts = {}
        for post in hot + rising + top_week:
            if post.id not in all_posts:
                all_posts[post.id] = post

        posts = list(all_posts.values())
        posts.sort(key=lambda p: p.viral_score, reverse=True)

        # Convert to video ideas
        ideas = []
        for post in posts[:limit]:
            idea_type = "question" if post.is_question else "viral"
            ideas.append(self._post_to_idea(post, idea_type))

        logger.success(f"Found {len(ideas)} viral content ideas for {niche}")
        return ideas

    def analyze_subreddit(self, subreddit_name: str) -> Optional[SubredditStats]:
        """
        Analyze a subreddit to understand its characteristics.
        """
        if not self.reddit:
            return None

        try:
            self._rate_limit_wait()
            subreddit = self.reddit.subreddit(subreddit_name)

            # Get basic stats
            subscribers = subreddit.subscribers
            active = getattr(subreddit, 'accounts_active', None)

            # Analyze recent posts
            posts = list(subreddit.hot(limit=50))

            scores = [p.score for p in posts]
            comments = [p.num_comments for p in posts]
            flairs = [p.link_flair_text for p in posts if p.link_flair_text]

            # Calculate posts per day (rough estimate)
            if len(posts) >= 2:
                time_span = posts[0].created_utc - posts[-1].created_utc
                if time_span > 0:
                    posts_per_day = len(posts) / (time_span / 86400)
                else:
                    posts_per_day = 0
            else:
                posts_per_day = 0

            return SubredditStats(
                name=subreddit_name,
                subscribers=subscribers,
                active_users=active,
                avg_post_score=sum(scores) / len(scores) if scores else 0,
                avg_comments=sum(comments) / len(comments) if comments else 0,
                posts_per_day=round(posts_per_day, 1),
                top_flairs=list(set(flairs))[:10],
            )

        except Exception as e:
            logger.warning(f"Error analyzing r/{subreddit_name}: {e}")
            return None

    def full_research(
        self,
        niche: str,
        include_sentiment: bool = True
    ) -> RedditResearchReport:
        """
        Perform comprehensive Reddit research for a niche.

        Args:
            niche: Content niche to research
            include_sentiment: Whether to analyze comment sentiment

        Returns:
            RedditResearchReport with all findings
        """
        logger.info(f"Starting full Reddit research for: {niche}")

        # Gather data
        trending = self.get_trending_topics(niche, limit=20)
        viral_ideas = self.find_viral_content(niche, limit=15)
        questions = self.get_questions(niche, limit=20)

        # Analyze subreddits
        subreddit_stats = []
        for sub_name in self.get_subreddits_for_niche(niche)[:5]:
            stats = self.analyze_subreddit(sub_name)
            if stats:
                subreddit_stats.append(stats)

        # Extract keywords
        all_titles = [v.source_title for v in viral_ideas] + [q.topic for q in questions]
        keywords = self._extract_keywords(all_titles, top_n=25)

        # Sentiment analysis
        if include_sentiment:
            texts = [v.source_title for v in viral_ideas]
            sentiment = self._analyze_sentiment(texts)
        else:
            sentiment = {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

        # Generate recommendations
        recommendations = self._generate_recommendations(niche, viral_ideas, questions, keywords)

        report = RedditResearchReport(
            niche=niche,
            timestamp=datetime.now(),
            trending_topics=trending,
            viral_ideas=viral_ideas,
            questions=questions,
            sentiment_summary=sentiment,
            subreddit_stats=subreddit_stats,
            keyword_frequency=keywords,
            recommendations=recommendations,
        )

        logger.success(f"Reddit research complete for {niche}")
        return report

    def _generate_recommendations(
        self,
        niche: str,
        viral_ideas: List[VideoIdea],
        questions: List[VideoIdea],
        keywords: Dict[str, int]
    ) -> List[str]:
        """Generate content recommendations based on research."""
        recommendations = []

        # Top keyword recommendations
        top_keywords = list(keywords.keys())[:5]
        if top_keywords:
            recommendations.append(
                f"Focus on these trending keywords: {', '.join(top_keywords)}"
            )

        # Question-based content
        if questions:
            top_question = questions[0].topic[:80]
            recommendations.append(
                f"Create a tutorial answering: \"{top_question}\""
            )

        # Viral content type
        if viral_ideas:
            top_idea = viral_ideas[0]
            recommendations.append(
                f"High-viral potential: \"{top_idea.title_suggestion}\" (Score: {top_idea.viral_score:.0f})"
            )

        # Engagement timing
        recommendations.append(
            "Post content when subreddits are most active (typically 9-11 AM EST)"
        )

        # Content mix
        question_ratio = len(questions) / (len(viral_ideas) + len(questions) + 1)
        if question_ratio > 0.6:
            recommendations.append(
                f"This niche is question-heavy ({question_ratio:.0%}). "
                "Tutorial/FAQ content will perform well."
            )
        else:
            recommendations.append(
                "Mix of discussion and Q&A content works best for this niche."
            )

        return recommendations

    def search_posts(
        self,
        query: str,
        niche: Optional[str] = None,
        subreddits: Optional[List[str]] = None,
        sort: str = "relevance",
        time_filter: str = "month",
        limit: int = 30
    ) -> List[RedditPost]:
        """
        Search Reddit for specific topics.

        Args:
            query: Search query
            niche: Niche for subreddit list (optional)
            subreddits: Specific subreddits to search (overrides niche)
            sort: relevance, hot, top, new, comments
            time_filter: hour, day, week, month, year, all
            limit: Max results
        """
        if not self.reddit:
            logger.error("Reddit not initialized.")
            return []

        # Determine subreddits
        if subreddits:
            sub_list = subreddits
        elif niche:
            sub_list = self.get_subreddits_for_niche(niche)
        else:
            sub_list = ["all"]

        posts = []

        try:
            self._rate_limit_wait()

            subreddit_str = "+".join(sub_list)
            subreddit = self.reddit.subreddit(subreddit_str)

            for post in subreddit.search(
                query,
                sort=sort,
                time_filter=time_filter,
                limit=limit
            ):
                posts.append(self._post_to_dataclass(post))

        except Exception as e:
            logger.error(f"Search failed: {e}")

        logger.info(f"Found {len(posts)} results for '{query}'")
        return posts


# ============================================================
# CLI INTERFACE (for testing)
# ============================================================

if __name__ == "__main__":
    import sys

    researcher = RedditResearcher()

    if not researcher.reddit:
        print("Reddit not configured. Please set credentials in .env:")
        print("  REDDIT_CLIENT_ID=your_client_id")
        print("  REDDIT_CLIENT_SECRET=your_client_secret")
        print("\nTo get credentials:")
        print("  1. Go to https://www.reddit.com/prefs/apps")
        print("  2. Click 'create another app...'")
        print("  3. Select 'script' type")
        print("  4. Copy client_id and client_secret")
        sys.exit(1)

    # Default to finance niche
    niche = sys.argv[1] if len(sys.argv) > 1 else "finance"

    print(f"\n{'='*60}")
    print(f"REDDIT RESEARCH: {niche.upper()}")
    print(f"{'='*60}\n")

    # Quick test
    print("Fetching trending topics...")
    topics = researcher.get_trending_topics(niche, limit=10)

    print("\nTrending Topics:")
    for i, topic in enumerate(topics, 1):
        print(f"  {i}. {topic[:70]}...")

    print("\nFetching viral content...")
    viral = researcher.find_viral_content(niche, limit=5)

    print("\nViral Ideas:")
    for i, idea in enumerate(viral, 1):
        print(f"  {i}. {idea.title_suggestion}")
        print(f"     Score: {idea.viral_score:.0f} | Subreddit: r/{idea.subreddit}")

    print("\nFetching questions...")
    questions = researcher.get_questions(niche, limit=5)

    print("\nTop Questions:")
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q.topic[:70]}...")

    print(f"\n{'='*60}")
