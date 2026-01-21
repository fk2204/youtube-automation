"""
Social Media Integration for YouTube Automation

This module provides tools for posting to multiple social media platforms
to drive first-hour engagement after YouTube video uploads.

Supported Platforms:
- Twitter/X
- Reddit
- Discord (via webhooks)
- LinkedIn
- Facebook (Page posts)

Usage:
    from src.social import (
        SocialMediaManager,
        post_video_to_social,
        schedule_first_hour_boost,
        Platform,
        SocialPost
    )

    # Quick post to all configured platforms
    results = post_video_to_social(
        video_title="My Video Title",
        video_url="https://youtube.com/watch?v=abc123",
        niche="finance"
    )

    # Or use the manager for more control
    manager = SocialMediaManager()
    manager.schedule_first_hour_posts(
        video_title="My Video",
        video_url="https://youtube.com/watch?v=abc123",
        niche="tech",
        reddit_subreddits=["technology", "programming"]
    )

Environment Variables Required:
    Twitter:
        - TWITTER_API_KEY
        - TWITTER_API_SECRET
        - TWITTER_ACCESS_TOKEN
        - TWITTER_ACCESS_SECRET
        - TWITTER_BEARER_TOKEN (optional)

    Reddit:
        - REDDIT_CLIENT_ID
        - REDDIT_CLIENT_SECRET
        - REDDIT_USERNAME
        - REDDIT_PASSWORD
        - REDDIT_USER_AGENT (optional)

    Discord:
        - DISCORD_WEBHOOK_URL
        - DISCORD_WEBHOOK_URL_<channel> (for multiple channels)

    LinkedIn:
        - LINKEDIN_ACCESS_TOKEN
        - LINKEDIN_PERSON_URN

    Facebook:
        - FACEBOOK_PAGE_ACCESS_TOKEN
        - FACEBOOK_PAGE_ID
"""

from .social_poster import (
    # Enums
    Platform,

    # Data classes
    SocialPost,

    # Base class
    SocialPlatform,

    # Platform implementations
    TwitterPoster,
    RedditPoster,
    DiscordPoster,
    LinkedInPoster,
    FacebookPoster,

    # Main manager
    SocialMediaManager,

    # Convenience functions
    post_video_to_social,
    schedule_first_hour_boost,
)

__all__ = [
    # Enums
    "Platform",

    # Data classes
    "SocialPost",

    # Base class
    "SocialPlatform",

    # Platform implementations
    "TwitterPoster",
    "RedditPoster",
    "DiscordPoster",
    "LinkedInPoster",
    "FacebookPoster",

    # Main manager
    "SocialMediaManager",

    # Convenience functions
    "post_video_to_social",
    "schedule_first_hour_boost",
]

__version__ = "1.0.0"
