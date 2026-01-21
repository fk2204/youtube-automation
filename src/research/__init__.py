# Research modules
from .trends import TrendResearcher
from .reddit import RedditResearcher
from .idea_generator import IdeaGenerator

# Enhanced Reddit researcher (optional - requires praw)
try:
    from .reddit_researcher import (
        RedditResearcher as EnhancedRedditResearcher,
        RedditPost,
        VideoIdea as RedditVideoIdea,
        RedditResearchReport,
        SubredditStats,
    )
    ENHANCED_REDDIT_AVAILABLE = True
except ImportError:
    ENHANCED_REDDIT_AVAILABLE = False
    EnhancedRedditResearcher = None
    RedditPost = None
    RedditVideoIdea = None
    RedditResearchReport = None
    SubredditStats = None

__all__ = [
    "TrendResearcher",
    "RedditResearcher",
    "IdeaGenerator",
    # Enhanced Reddit (optional)
    "EnhancedRedditResearcher",
    "RedditPost",
    "RedditVideoIdea",
    "RedditResearchReport",
    "SubredditStats",
    "ENHANCED_REDDIT_AVAILABLE",
]
