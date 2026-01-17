"""
Reddit Research Module

Finds video ideas by monitoring popular questions and discussions on Reddit.
Requires Reddit API credentials (free).

Setup:
1. Go to https://www.reddit.com/prefs/apps
2. Create an app (script type)
3. Set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT in .env

Usage:
    researcher = RedditResearcher()
    ideas = researcher.get_video_ideas(subreddits=["learnprogramming", "webdev"])
"""

import os
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger

try:
    import praw
except ImportError:
    raise ImportError("Please install praw: pip install praw")


@dataclass
class RedditPost:
    """Represents a Reddit post that could be a video idea."""
    title: str
    subreddit: str
    score: int              # Upvotes
    num_comments: int
    url: str
    created_utc: datetime
    flair: Optional[str]
    is_question: bool       # Likely a question/tutorial request


@dataclass
class VideoIdea:
    """A potential video idea extracted from Reddit."""
    topic: str
    source_title: str
    source_url: str
    subreddit: str
    popularity_score: int   # Combined score based on upvotes/comments
    idea_type: str          # question, discussion, request, tutorial


class RedditResearcher:
    """Research video ideas from Reddit discussions."""

    # Default subreddits for educational/tutorial content
    DEFAULT_SUBREDDITS = [
        "learnprogramming",
        "webdev",
        "Python",
        "javascript",
        "reactjs",
        "node",
        "coding",
        "cscareerquestions",
        "explainlikeimfive",
        "howto",
        "techsupport",
    ]

    # Keywords that indicate a question/request
    QUESTION_KEYWORDS = [
        "how to", "how do", "how can", "what is", "what are",
        "why does", "why is", "explain", "help", "tutorial",
        "guide", "learn", "beginner", "newbie", "stuck",
        "can someone", "eli5", "please help", "need help",
        "best way to", "should i", "difference between"
    ]

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """
        Initialize Reddit researcher.

        Args:
            client_id: Reddit app client ID
            client_secret: Reddit app client secret
            user_agent: User agent string for API requests
        """
        self.client_id = client_id or os.getenv("REDDIT_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET")
        self.user_agent = user_agent or os.getenv(
            "REDDIT_USER_AGENT",
            "youtube-automation-bot/1.0"
        )

        if not self.client_id or not self.client_secret:
            logger.warning(
                "Reddit credentials not found. Set REDDIT_CLIENT_ID and "
                "REDDIT_CLIENT_SECRET in .env file."
            )
            self.reddit = None
        else:
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent
            )
            logger.info("Reddit researcher initialized")

    def _is_question(self, text: str) -> bool:
        """Check if text appears to be a question or request."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.QUESTION_KEYWORDS) or "?" in text

    def _calculate_popularity(self, post: praw.models.Submission) -> int:
        """Calculate a popularity score for a post."""
        # Weighted score: upvotes + (comments * 2)
        # Comments indicate engagement and interest
        score = post.score + (post.num_comments * 2)

        # Boost for questions (more likely to be tutorial material)
        if self._is_question(post.title):
            score *= 1.5

        return int(score)

    def get_hot_posts(
        self,
        subreddits: Optional[List[str]] = None,
        limit: int = 50,
        time_filter: str = "week"
    ) -> List[RedditPost]:
        """
        Get hot/popular posts from subreddits.

        Args:
            subreddits: List of subreddit names
            limit: Max posts per subreddit
            time_filter: Time filter (hour, day, week, month, year, all)

        Returns:
            List of RedditPost objects
        """
        if not self.reddit:
            logger.error("Reddit not initialized. Check credentials.")
            return []

        subreddits = subreddits or self.DEFAULT_SUBREDDITS
        posts = []

        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)

                for post in subreddit.top(time_filter=time_filter, limit=limit):
                    reddit_post = RedditPost(
                        title=post.title,
                        subreddit=subreddit_name,
                        score=post.score,
                        num_comments=post.num_comments,
                        url=f"https://reddit.com{post.permalink}",
                        created_utc=datetime.fromtimestamp(post.created_utc),
                        flair=post.link_flair_text,
                        is_question=self._is_question(post.title)
                    )
                    posts.append(reddit_post)

                logger.debug(f"Fetched {limit} posts from r/{subreddit_name}")

            except Exception as e:
                logger.error(f"Error fetching r/{subreddit_name}: {e}")
                continue

        logger.info(f"Total posts fetched: {len(posts)}")
        return posts

    def get_video_ideas(
        self,
        subreddits: Optional[List[str]] = None,
        limit: int = 20,
        min_score: int = 50,
        questions_only: bool = True
    ) -> List[VideoIdea]:
        """
        Extract video ideas from Reddit posts.

        Args:
            subreddits: List of subreddit names to search
            limit: Max ideas to return
            min_score: Minimum upvote score
            questions_only: Only include questions/requests

        Returns:
            List of VideoIdea objects sorted by popularity
        """
        if not self.reddit:
            logger.error("Reddit not initialized. Check credentials.")
            return []

        subreddits = subreddits or self.DEFAULT_SUBREDDITS
        ideas = []

        logger.info(f"Searching {len(subreddits)} subreddits for video ideas...")

        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)

                # Search for questions in hot and top posts
                for post in subreddit.top(time_filter="week", limit=30):
                    if post.score < min_score:
                        continue

                    is_question = self._is_question(post.title)

                    if questions_only and not is_question:
                        continue

                    # Determine idea type
                    if is_question:
                        idea_type = "question"
                    elif "tutorial" in post.title.lower():
                        idea_type = "tutorial"
                    elif "help" in post.title.lower():
                        idea_type = "request"
                    else:
                        idea_type = "discussion"

                    idea = VideoIdea(
                        topic=self._clean_topic(post.title),
                        source_title=post.title,
                        source_url=f"https://reddit.com{post.permalink}",
                        subreddit=subreddit_name,
                        popularity_score=self._calculate_popularity(post),
                        idea_type=idea_type
                    )
                    ideas.append(idea)

            except Exception as e:
                logger.error(f"Error searching r/{subreddit_name}: {e}")
                continue

        # Sort by popularity and limit
        ideas.sort(key=lambda x: x.popularity_score, reverse=True)
        ideas = ideas[:limit]

        logger.success(f"Found {len(ideas)} video ideas")
        return ideas

    def _clean_topic(self, title: str) -> str:
        """Clean a Reddit title into a video topic."""
        # Remove common prefixes
        prefixes = [
            "[Question]", "[Help]", "[Tutorial]", "[Beginner]",
            "Question:", "Help:", "ELI5:", "How to"
        ]

        topic = title
        for prefix in prefixes:
            if topic.lower().startswith(prefix.lower()):
                topic = topic[len(prefix):].strip()

        # Remove trailing punctuation
        topic = topic.rstrip("?!.")

        # Capitalize first letter
        if topic:
            topic = topic[0].upper() + topic[1:]

        return topic

    def search_posts(
        self,
        query: str,
        subreddits: Optional[List[str]] = None,
        limit: int = 20,
        sort: str = "relevance",
        time_filter: str = "month"
    ) -> List[RedditPost]:
        """
        Search Reddit for specific topics.

        Args:
            query: Search query
            subreddits: Subreddits to search (None = all)
            limit: Max results
            sort: Sort order (relevance, hot, top, new, comments)
            time_filter: Time filter

        Returns:
            List of matching RedditPost objects
        """
        if not self.reddit:
            logger.error("Reddit not initialized. Check credentials.")
            return []

        logger.info(f"Searching Reddit for: {query}")

        posts = []

        try:
            if subreddits:
                subreddit_str = "+".join(subreddits)
                subreddit = self.reddit.subreddit(subreddit_str)
            else:
                subreddit = self.reddit.subreddit("all")

            for post in subreddit.search(
                query,
                sort=sort,
                time_filter=time_filter,
                limit=limit
            ):
                reddit_post = RedditPost(
                    title=post.title,
                    subreddit=post.subreddit.display_name,
                    score=post.score,
                    num_comments=post.num_comments,
                    url=f"https://reddit.com{post.permalink}",
                    created_utc=datetime.fromtimestamp(post.created_utc),
                    flair=post.link_flair_text,
                    is_question=self._is_question(post.title)
                )
                posts.append(reddit_post)

        except Exception as e:
            logger.error(f"Search failed: {e}")

        logger.info(f"Found {len(posts)} results for '{query}'")
        return posts

    def get_trending_topics(
        self,
        subreddits: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[str]:
        """
        Get currently trending topics/keywords.

        Args:
            subreddits: Subreddits to analyze
            limit: Number of topics to return

        Returns:
            List of trending topic strings
        """
        if not self.reddit:
            return []

        subreddits = subreddits or self.DEFAULT_SUBREDDITS[:5]
        posts = self.get_hot_posts(subreddits, limit=20)

        # Extract common words/phrases
        from collections import Counter
        words = []

        for post in posts:
            # Simple word extraction (could be improved with NLP)
            title_words = post.title.lower().split()
            # Filter out common words
            stopwords = {
                "the", "a", "an", "is", "are", "was", "were", "be", "been",
                "being", "have", "has", "had", "do", "does", "did", "will",
                "would", "could", "should", "may", "might", "must", "shall",
                "can", "need", "to", "of", "in", "for", "on", "with", "at",
                "by", "from", "as", "into", "through", "during", "before",
                "after", "above", "below", "between", "under", "again",
                "further", "then", "once", "here", "there", "when", "where",
                "why", "how", "all", "each", "few", "more", "most", "other",
                "some", "such", "no", "nor", "not", "only", "own", "same",
                "so", "than", "too", "very", "just", "i", "me", "my", "we",
                "our", "you", "your", "it", "its", "this", "that", "these"
            }
            words.extend([w for w in title_words if w not in stopwords and len(w) > 3])

        # Get most common
        counter = Counter(words)
        trending = [word for word, count in counter.most_common(limit)]

        return trending


# Example usage
if __name__ == "__main__":
    researcher = RedditResearcher()

    if researcher.reddit:
        print("\n" + "="*60)
        print("VIDEO IDEAS FROM REDDIT")
        print("="*60 + "\n")

        ideas = researcher.get_video_ideas(
            subreddits=["learnprogramming", "Python"],
            limit=10
        )

        for i, idea in enumerate(ideas, 1):
            print(f"{i}. {idea.topic}")
            print(f"   Subreddit: r/{idea.subreddit}")
            print(f"   Popularity: {idea.popularity_score}")
            print(f"   Type: {idea.idea_type}")
            print(f"   Source: {idea.source_url}")
            print()
    else:
        print("Reddit credentials not configured.")
        print("Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env")
