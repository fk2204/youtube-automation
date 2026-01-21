"""
Social Media Integration for First-Hour Boost

Post to social media platforms immediately after YouTube upload
to drive initial engagement and algorithm boost.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from abc import ABC, abstractmethod
from enum import Enum
import json
import os
import asyncio
import threading
import time
from loguru import logger


class Platform(Enum):
    TWITTER = "twitter"
    REDDIT = "reddit"
    LINKEDIN = "linkedin"
    FACEBOOK = "facebook"
    DISCORD = "discord"


@dataclass
class SocialPost:
    """A social media post."""
    platform: Platform
    content: str
    url: Optional[str] = None
    image_path: Optional[str] = None
    hashtags: List[str] = field(default_factory=list)
    scheduled_time: Optional[datetime] = None
    posted: bool = False
    post_id: Optional[str] = None
    error: Optional[str] = None
    posted_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "platform": self.platform.value,
            "content": self.content,
            "url": self.url,
            "image_path": self.image_path,
            "hashtags": self.hashtags,
            "scheduled_time": self.scheduled_time.isoformat() if self.scheduled_time else None,
            "posted": self.posted,
            "post_id": self.post_id,
            "error": self.error,
            "posted_at": self.posted_at.isoformat() if self.posted_at else None,
        }


class SocialPlatform(ABC):
    """Abstract base class for social media platforms."""

    @abstractmethod
    def post(self, content: str, url: str = None, image: str = None, **kwargs) -> Dict[str, Any]:
        """Post content to the platform."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if platform is properly configured."""
        pass

    @abstractmethod
    def get_platform_name(self) -> str:
        """Get the display name of the platform."""
        pass

    def validate_content(self, content: str) -> tuple[bool, str]:
        """Validate content for platform-specific requirements."""
        return True, ""


class TwitterPoster(SocialPlatform):
    """Twitter/X posting integration."""

    MAX_TWEET_LENGTH = 280

    def __init__(self):
        self.api_key = os.getenv("TWITTER_API_KEY")
        self.api_secret = os.getenv("TWITTER_API_SECRET")
        self.access_token = os.getenv("TWITTER_ACCESS_TOKEN")
        self.access_secret = os.getenv("TWITTER_ACCESS_SECRET")
        self.bearer_token = os.getenv("TWITTER_BEARER_TOKEN")

    def get_platform_name(self) -> str:
        return "Twitter/X"

    def is_configured(self) -> bool:
        return all([self.api_key, self.api_secret, self.access_token, self.access_secret])

    def validate_content(self, content: str) -> tuple[bool, str]:
        """Validate tweet length (URLs count as 23 chars)."""
        # Twitter shortens URLs to 23 characters
        effective_length = len(content)
        if effective_length > self.MAX_TWEET_LENGTH:
            return False, f"Tweet too long: {effective_length}/{self.MAX_TWEET_LENGTH} characters"
        return True, ""

    def post(self, content: str, url: str = None, image: str = None, **kwargs) -> Dict[str, Any]:
        """Post to Twitter using Twitter API v2."""
        if not self.is_configured():
            logger.warning("[Twitter] Not configured - would post: {}", content[:50])
            return {
                "success": False,
                "error": "Twitter not configured - missing API credentials",
                "platform": "twitter",
                "simulated": True,
                "content_preview": content[:100]
            }

        # Validate content
        is_valid, error_msg = self.validate_content(content)
        if not is_valid:
            return {"success": False, "error": error_msg, "platform": "twitter"}

        try:
            # Try to use tweepy if available
            import tweepy

            client = tweepy.Client(
                consumer_key=self.api_key,
                consumer_secret=self.api_secret,
                access_token=self.access_token,
                access_token_secret=self.access_secret
            )

            # Handle media upload if image provided
            media_ids = None
            if image and os.path.exists(image):
                # Need v1.1 API for media upload
                auth = tweepy.OAuth1UserHandler(
                    self.api_key, self.api_secret,
                    self.access_token, self.access_secret
                )
                api_v1 = tweepy.API(auth)
                media = api_v1.media_upload(image)
                media_ids = [media.media_id]

            # Post tweet
            response = client.create_tweet(text=content, media_ids=media_ids)

            logger.info("[Twitter] Posted successfully: {}", response.data['id'])
            return {
                "success": True,
                "post_id": str(response.data['id']),
                "platform": "twitter",
                "url": f"https://twitter.com/i/web/status/{response.data['id']}"
            }

        except ImportError:
            logger.warning("[Twitter] tweepy not installed - simulating post")
            return self._simulate_post(content, url, image)
        except Exception as e:
            logger.error("[Twitter] Failed to post: {}", str(e))
            return {"success": False, "error": str(e), "platform": "twitter"}

    def _simulate_post(self, content: str, url: str = None, image: str = None) -> Dict[str, Any]:
        """Simulate a post when API is not available."""
        logger.info("[Twitter] SIMULATED POST:")
        logger.info("  Content: {}", content)
        if url:
            logger.info("  URL: {}", url)
        if image:
            logger.info("  Image: {}", image)
        return {
            "success": True,
            "post_id": f"sim_tweet_{int(time.time())}",
            "platform": "twitter",
            "simulated": True
        }


class RedditPoster(SocialPlatform):
    """Reddit posting integration."""

    def __init__(self):
        self.client_id = os.getenv("REDDIT_CLIENT_ID")
        self.client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        self.username = os.getenv("REDDIT_USERNAME")
        self.password = os.getenv("REDDIT_PASSWORD")
        self.user_agent = os.getenv("REDDIT_USER_AGENT", "youtube-automation-bot/1.0")

    def get_platform_name(self) -> str:
        return "Reddit"

    def is_configured(self) -> bool:
        return all([self.client_id, self.client_secret, self.username, self.password])

    def post(self, content: str, url: str = None, image: str = None,
             subreddit: str = None, title: str = None, **kwargs) -> Dict[str, Any]:
        """Post to Reddit."""
        if not subreddit:
            return {"success": False, "error": "Subreddit is required", "platform": "reddit"}

        if not self.is_configured():
            logger.warning("[Reddit] Not configured - would post to r/{}: {}", subreddit, content[:50])
            return {
                "success": False,
                "error": "Reddit not configured - missing API credentials",
                "platform": "reddit",
                "simulated": True,
                "subreddit": subreddit,
                "content_preview": content[:100]
            }

        try:
            import praw

            reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                username=self.username,
                password=self.password,
                user_agent=self.user_agent
            )

            sub = reddit.subreddit(subreddit)

            # Determine post type
            post_title = title or content[:300]  # Reddit title limit

            if url:
                # Link post
                submission = sub.submit(post_title, url=url)
            elif image and os.path.exists(image):
                # Image post
                submission = sub.submit_image(post_title, image)
            else:
                # Text post
                submission = sub.submit(post_title, selftext=content)

            logger.info("[Reddit] Posted to r/{}: {}", subreddit, submission.id)
            return {
                "success": True,
                "post_id": submission.id,
                "platform": "reddit",
                "url": f"https://reddit.com{submission.permalink}",
                "subreddit": subreddit
            }

        except ImportError:
            logger.warning("[Reddit] praw not installed - simulating post")
            return self._simulate_post(content, url, subreddit, title)
        except Exception as e:
            logger.error("[Reddit] Failed to post to r/{}: {}", subreddit, str(e))
            return {"success": False, "error": str(e), "platform": "reddit"}

    def _simulate_post(self, content: str, url: str, subreddit: str, title: str) -> Dict[str, Any]:
        """Simulate a Reddit post."""
        logger.info("[Reddit] SIMULATED POST to r/{}:", subreddit)
        logger.info("  Title: {}", title or content[:100])
        if url:
            logger.info("  URL: {}", url)
        return {
            "success": True,
            "post_id": f"sim_reddit_{int(time.time())}",
            "platform": "reddit",
            "simulated": True,
            "subreddit": subreddit
        }


class DiscordPoster(SocialPlatform):
    """Discord webhook integration."""

    def __init__(self):
        self.webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
        self.webhook_urls = self._load_webhook_urls()

    def _load_webhook_urls(self) -> Dict[str, str]:
        """Load multiple webhook URLs from environment."""
        urls = {}
        # Support multiple webhooks: DISCORD_WEBHOOK_URL_channelname
        for key, value in os.environ.items():
            if key.startswith("DISCORD_WEBHOOK_URL_"):
                channel = key.replace("DISCORD_WEBHOOK_URL_", "").lower()
                urls[channel] = value
        return urls

    def get_platform_name(self) -> str:
        return "Discord"

    def is_configured(self) -> bool:
        return bool(self.webhook_url or self.webhook_urls)

    def post(self, content: str, url: str = None, image: str = None,
             channel: str = None, embed: bool = True, **kwargs) -> Dict[str, Any]:
        """Post to Discord via webhook."""
        # Determine which webhook to use
        webhook = self.webhook_url
        if channel and channel in self.webhook_urls:
            webhook = self.webhook_urls[channel]

        if not webhook:
            logger.warning("[Discord] Not configured - would post: {}", content[:50])
            return {
                "success": False,
                "error": "Discord not configured - no webhook URL",
                "platform": "discord",
                "simulated": True,
                "content_preview": content[:100]
            }

        try:
            import requests

            # Build payload
            if embed and url:
                # Create rich embed
                payload = {
                    "embeds": [{
                        "title": content[:256],  # Discord embed title limit
                        "url": url,
                        "color": 16711680,  # Red color for YouTube
                        "footer": {"text": "YouTube Automation Bot"}
                    }]
                }
                if image and os.path.exists(image):
                    # For local images, we'd need to upload separately
                    # For now, just note it in the embed
                    payload["embeds"][0]["image"] = {"url": image} if image.startswith("http") else None
            else:
                # Simple text message
                full_content = content
                if url:
                    full_content += f"\n{url}"
                payload = {"content": full_content}

            response = requests.post(webhook, json=payload, timeout=10)

            if response.status_code in (200, 204):
                logger.info("[Discord] Posted successfully to channel")
                return {
                    "success": True,
                    "platform": "discord",
                    "channel": channel
                }
            else:
                error = f"HTTP {response.status_code}: {response.text}"
                logger.error("[Discord] Failed: {}", error)
                return {"success": False, "error": error, "platform": "discord"}

        except ImportError:
            logger.error("[Discord] requests library not available")
            return {"success": False, "error": "requests library not installed", "platform": "discord"}
        except Exception as e:
            logger.error("[Discord] Failed to post: {}", str(e))
            return {"success": False, "error": str(e), "platform": "discord"}


class LinkedInPoster(SocialPlatform):
    """LinkedIn posting integration."""

    def __init__(self):
        self.access_token = os.getenv("LINKEDIN_ACCESS_TOKEN")
        self.person_urn = os.getenv("LINKEDIN_PERSON_URN")  # Format: urn:li:person:xxxxx

    def get_platform_name(self) -> str:
        return "LinkedIn"

    def is_configured(self) -> bool:
        return all([self.access_token, self.person_urn])

    def post(self, content: str, url: str = None, image: str = None, **kwargs) -> Dict[str, Any]:
        """Post to LinkedIn."""
        if not self.is_configured():
            logger.warning("[LinkedIn] Not configured - would post: {}", content[:50])
            return {
                "success": False,
                "error": "LinkedIn not configured - missing access token or person URN",
                "platform": "linkedin",
                "simulated": True,
                "content_preview": content[:100]
            }

        try:
            import requests

            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
                "X-Restli-Protocol-Version": "2.0.0"
            }

            # Build share content
            share_content = {
                "author": self.person_urn,
                "lifecycleState": "PUBLISHED",
                "specificContent": {
                    "com.linkedin.ugc.ShareContent": {
                        "shareCommentary": {"text": content},
                        "shareMediaCategory": "ARTICLE" if url else "NONE"
                    }
                },
                "visibility": {
                    "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
                }
            }

            # Add URL if provided
            if url:
                share_content["specificContent"]["com.linkedin.ugc.ShareContent"]["media"] = [{
                    "status": "READY",
                    "originalUrl": url
                }]

            response = requests.post(
                "https://api.linkedin.com/v2/ugcPosts",
                headers=headers,
                json=share_content,
                timeout=15
            )

            if response.status_code == 201:
                post_id = response.json().get("id", "")
                logger.info("[LinkedIn] Posted successfully: {}", post_id)
                return {
                    "success": True,
                    "post_id": post_id,
                    "platform": "linkedin"
                }
            else:
                error = f"HTTP {response.status_code}: {response.text}"
                logger.error("[LinkedIn] Failed: {}", error)
                return {"success": False, "error": error, "platform": "linkedin"}

        except ImportError:
            logger.error("[LinkedIn] requests library not available")
            return {"success": False, "error": "requests library not installed", "platform": "linkedin"}
        except Exception as e:
            logger.error("[LinkedIn] Failed to post: {}", str(e))
            return {"success": False, "error": str(e), "platform": "linkedin"}


class FacebookPoster(SocialPlatform):
    """Facebook page posting integration."""

    def __init__(self):
        self.page_access_token = os.getenv("FACEBOOK_PAGE_ACCESS_TOKEN")
        self.page_id = os.getenv("FACEBOOK_PAGE_ID")

    def get_platform_name(self) -> str:
        return "Facebook"

    def is_configured(self) -> bool:
        return all([self.page_access_token, self.page_id])

    def post(self, content: str, url: str = None, image: str = None, **kwargs) -> Dict[str, Any]:
        """Post to Facebook page."""
        if not self.is_configured():
            logger.warning("[Facebook] Not configured - would post: {}", content[:50])
            return {
                "success": False,
                "error": "Facebook not configured - missing page access token or page ID",
                "platform": "facebook",
                "simulated": True,
                "content_preview": content[:100]
            }

        try:
            import requests

            endpoint = f"https://graph.facebook.com/v18.0/{self.page_id}/feed"

            payload = {
                "message": content,
                "access_token": self.page_access_token
            }

            if url:
                payload["link"] = url

            response = requests.post(endpoint, data=payload, timeout=15)

            if response.status_code == 200:
                post_id = response.json().get("id", "")
                logger.info("[Facebook] Posted successfully: {}", post_id)
                return {
                    "success": True,
                    "post_id": post_id,
                    "platform": "facebook",
                    "url": f"https://facebook.com/{post_id}"
                }
            else:
                error = f"HTTP {response.status_code}: {response.json()}"
                logger.error("[Facebook] Failed: {}", error)
                return {"success": False, "error": error, "platform": "facebook"}

        except ImportError:
            logger.error("[Facebook] requests library not available")
            return {"success": False, "error": "requests library not installed", "platform": "facebook"}
        except Exception as e:
            logger.error("[Facebook] Failed to post: {}", str(e))
            return {"success": False, "error": str(e), "platform": "facebook"}


class SocialMediaManager:
    """
    Manage posting across multiple social media platforms.
    Optimized for first-hour engagement boost after YouTube uploads.
    """

    def __init__(self):
        self.platforms: Dict[Platform, SocialPlatform] = {
            Platform.TWITTER: TwitterPoster(),
            Platform.REDDIT: RedditPoster(),
            Platform.DISCORD: DiscordPoster(),
            Platform.LINKEDIN: LinkedInPoster(),
            Platform.FACEBOOK: FacebookPoster(),
        }
        self.post_history: List[SocialPost] = []
        self._scheduled_tasks: List[Dict[str, Any]] = []
        self._scheduler_running = False

    def get_configured_platforms(self) -> List[Platform]:
        """Get list of configured platforms."""
        return [p for p, poster in self.platforms.items() if poster.is_configured()]

    def get_platform_status(self) -> Dict[str, Dict[str, Any]]:
        """Get configuration status for all platforms."""
        status = {}
        for platform, poster in self.platforms.items():
            status[platform.value] = {
                "name": poster.get_platform_name(),
                "configured": poster.is_configured(),
                "platform": platform.value
            }
        return status

    def generate_post_content(
        self,
        video_title: str,
        video_url: str,
        niche: str,
        platform: Platform,
        custom_message: str = None
    ) -> str:
        """Generate platform-specific post content."""
        hashtags = self._get_niche_hashtags(niche)
        hashtag_str = " ".join(f"#{h}" for h in hashtags[:5])

        templates = {
            Platform.TWITTER: (
                "{custom}\n\n"
                "{title}\n\n"
                "{url}\n\n"
                "{hashtags}"
            ) if custom_message else (
                "New video just dropped!\n\n"
                "{title}\n\n"
                "{url}\n\n"
                "{hashtags}"
            ),
            Platform.REDDIT: (
                "{title}"
            ),
            Platform.DISCORD: (
                "**New Upload!**\n\n"
                "**{title}**\n\n"
                "{url}"
            ),
            Platform.LINKEDIN: (
                "I just published a new video on a topic I'm passionate about:\n\n"
                "{title}\n\n"
                "Would love to hear your thoughts!\n\n"
                "{url}\n\n"
                "{hashtags}"
            ),
            Platform.FACEBOOK: (
                "New video alert!\n\n"
                "{title}\n\n"
                "Watch here: {url}"
            ),
        }

        template = templates.get(platform, "{title}\n{url}")

        return template.format(
            title=video_title,
            url=video_url,
            hashtags=hashtag_str,
            custom=custom_message or ""
        ).strip()

    def _get_niche_hashtags(self, niche: str) -> List[str]:
        """Get relevant hashtags for a niche."""
        hashtag_map = {
            "finance": ["investing", "money", "finance", "wealth", "passiveincome", "financialfreedom"],
            "psychology": ["psychology", "mindset", "selfimprovement", "mentalhealth", "motivation", "growth"],
            "storytelling": ["storytelling", "truecrime", "mystery", "documentary", "stories", "realstories"],
            "tech": ["technology", "tech", "coding", "programming", "ai", "innovation"],
            "gaming": ["gaming", "gamer", "videogames", "gameplay", "twitch", "esports"],
            "education": ["education", "learning", "knowledge", "facts", "educational", "didyouknow"],
            "entertainment": ["entertainment", "viral", "trending", "funny", "comedy", "mustwatch"],
            "health": ["health", "fitness", "wellness", "healthylifestyle", "nutrition", "workout"],
            "business": ["business", "entrepreneur", "startup", "success", "hustle", "leadership"],
        }

        # Also check for partial matches
        niche_lower = niche.lower()
        for key, tags in hashtag_map.items():
            if key in niche_lower or niche_lower in key:
                return tags

        return ["youtube", "newvideo", "content", "creator", "subscribe"]

    def post_to_platform(
        self,
        platform: Platform,
        video_title: str,
        video_url: str,
        niche: str,
        thumbnail_path: Optional[str] = None,
        custom_message: Optional[str] = None,
        **kwargs
    ) -> SocialPost:
        """Post to a specific platform."""
        content = self.generate_post_content(video_title, video_url, niche, platform, custom_message)

        post = SocialPost(
            platform=platform,
            content=content,
            url=video_url,
            image_path=thumbnail_path,
            hashtags=self._get_niche_hashtags(niche)
        )

        try:
            # Add Reddit-specific kwargs
            if platform == Platform.REDDIT:
                kwargs.setdefault("title", video_title)

            result = self.platforms[platform].post(content, video_url, thumbnail_path, **kwargs)
            post.posted = result.get("success", False)
            post.post_id = result.get("post_id")
            post.error = result.get("error")
            if post.posted:
                post.posted_at = datetime.now()
        except Exception as e:
            post.error = str(e)
            logger.error("Error posting to {}: {}", platform.value, str(e))

        self.post_history.append(post)
        return post

    def post_to_all(
        self,
        video_title: str,
        video_url: str,
        niche: str,
        thumbnail_path: Optional[str] = None,
        platforms: Optional[List[Platform]] = None,
        reddit_subreddits: Optional[List[str]] = None
    ) -> List[SocialPost]:
        """Post to all configured platforms (or specified subset)."""
        results = []

        target_platforms = platforms or self.get_configured_platforms()

        for platform in target_platforms:
            if platform not in self.platforms:
                continue

            if platform == Platform.REDDIT and reddit_subreddits:
                # Post to multiple subreddits
                for subreddit in reddit_subreddits:
                    post = self.post_to_platform(
                        platform, video_title, video_url, niche,
                        thumbnail_path, subreddit=subreddit
                    )
                    results.append(post)
            else:
                post = self.post_to_platform(
                    platform, video_title, video_url, niche, thumbnail_path
                )
                results.append(post)

        return results

    def schedule_post(
        self,
        platform: Platform,
        video_title: str,
        video_url: str,
        niche: str,
        delay_minutes: int,
        thumbnail_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Schedule a post for later."""
        scheduled_time = datetime.now() + timedelta(minutes=delay_minutes)

        task = {
            "platform": platform,
            "video_title": video_title,
            "video_url": video_url,
            "niche": niche,
            "thumbnail_path": thumbnail_path,
            "scheduled_time": scheduled_time,
            "kwargs": kwargs,
            "executed": False
        }

        self._scheduled_tasks.append(task)

        logger.info("Scheduled {} post for {} ({} minutes from now)",
                   platform.value, scheduled_time.isoformat(), delay_minutes)

        return {
            "scheduled": True,
            "platform": platform.value,
            "scheduled_time": scheduled_time.isoformat(),
            "delay_minutes": delay_minutes
        }

    def schedule_first_hour_posts(
        self,
        video_title: str,
        video_url: str,
        niche: str,
        thumbnail_path: Optional[str] = None,
        reddit_subreddits: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Schedule posts optimized for first-hour engagement boost.

        Strategy:
        - Immediate: Twitter, Discord (instant notification platforms)
        - 15 minutes: Reddit (benefits from some initial views)
        - 30 minutes: LinkedIn (professional audience, different timezone coverage)
        - 45 minutes: Facebook (broader reach)
        - 60 minutes: Twitter reminder (catch different timezones)
        """
        schedule = {
            0: [Platform.TWITTER, Platform.DISCORD],
            15: [Platform.REDDIT],
            30: [Platform.LINKEDIN],
            45: [Platform.FACEBOOK],
            60: [Platform.TWITTER],  # Reminder post
        }

        scheduled_posts = []

        for delay_minutes, platforms in schedule.items():
            for platform in platforms:
                if platform not in self.get_configured_platforms():
                    logger.debug("Skipping {} - not configured", platform.value)
                    continue

                if delay_minutes == 0:
                    # Post immediately
                    if platform == Platform.REDDIT and reddit_subreddits:
                        for subreddit in reddit_subreddits:
                            post = self.post_to_platform(
                                platform, video_title, video_url, niche,
                                thumbnail_path, subreddit=subreddit
                            )
                            scheduled_posts.append({
                                "platform": platform.value,
                                "delay": 0,
                                "status": "posted" if post.posted else "failed",
                                "post_id": post.post_id
                            })
                    else:
                        # Add custom message for reminder post
                        custom_msg = None
                        if delay_minutes == 60:
                            custom_msg = "In case you missed it..."

                        post = self.post_to_platform(
                            platform, video_title, video_url, niche,
                            thumbnail_path, custom_message=custom_msg
                        )
                        scheduled_posts.append({
                            "platform": platform.value,
                            "delay": 0,
                            "status": "posted" if post.posted else "failed",
                            "post_id": post.post_id
                        })
                else:
                    # Schedule for later
                    kwargs = {}
                    if platform == Platform.REDDIT and reddit_subreddits:
                        kwargs["subreddit"] = reddit_subreddits[0] if reddit_subreddits else None
                    if delay_minutes == 60:
                        kwargs["custom_message"] = "In case you missed it..."

                    result = self.schedule_post(
                        platform, video_title, video_url, niche,
                        delay_minutes, thumbnail_path, **kwargs
                    )
                    scheduled_posts.append({
                        "platform": platform.value,
                        "delay": delay_minutes,
                        "status": "scheduled",
                        "scheduled_time": result["scheduled_time"]
                    })

        return {
            "strategy": "first_hour_boost",
            "total_posts": len(scheduled_posts),
            "configured_platforms": [p.value for p in self.get_configured_platforms()],
            "posts": scheduled_posts
        }

    def run_scheduled_posts(self) -> List[SocialPost]:
        """Execute any scheduled posts that are due."""
        results = []
        now = datetime.now()

        for task in self._scheduled_tasks:
            if task["executed"]:
                continue

            if task["scheduled_time"] <= now:
                post = self.post_to_platform(
                    task["platform"],
                    task["video_title"],
                    task["video_url"],
                    task["niche"],
                    task["thumbnail_path"],
                    **task["kwargs"]
                )
                task["executed"] = True
                results.append(post)

                logger.info("Executed scheduled post to {}: {}",
                           task["platform"].value,
                           "success" if post.posted else f"failed - {post.error}")

        return results

    def start_scheduler(self, check_interval: int = 60):
        """Start background scheduler for scheduled posts."""
        if self._scheduler_running:
            logger.warning("Scheduler already running")
            return

        def scheduler_loop():
            self._scheduler_running = True
            logger.info("Social media scheduler started (checking every {}s)", check_interval)

            while self._scheduler_running:
                try:
                    self.run_scheduled_posts()
                except Exception as e:
                    logger.error("Scheduler error: {}", str(e))
                time.sleep(check_interval)

        thread = threading.Thread(target=scheduler_loop, daemon=True)
        thread.start()

    def stop_scheduler(self):
        """Stop the background scheduler."""
        self._scheduler_running = False
        logger.info("Social media scheduler stopped")

    def get_post_history(self, platform: Optional[Platform] = None) -> List[Dict[str, Any]]:
        """Get post history, optionally filtered by platform."""
        history = self.post_history
        if platform:
            history = [p for p in history if p.platform == platform]
        return [p.to_dict() for p in history]

    def export_history(self, filepath: str):
        """Export post history to JSON file."""
        with open(filepath, 'w') as f:
            json.dump([p.to_dict() for p in self.post_history], f, indent=2)
        logger.info("Exported {} posts to {}", len(self.post_history), filepath)


# Convenience functions
def post_video_to_social(
    video_title: str,
    video_url: str,
    niche: str,
    thumbnail: str = None,
    platforms: List[Platform] = None
) -> List[SocialPost]:
    """Quick function to post video to all social platforms."""
    manager = SocialMediaManager()
    return manager.post_to_all(video_title, video_url, niche, thumbnail, platforms)


def schedule_first_hour_boost(
    video_title: str,
    video_url: str,
    niche: str,
    thumbnail: str = None,
    reddit_subreddits: List[str] = None
) -> Dict[str, Any]:
    """Schedule posts for first-hour engagement boost."""
    manager = SocialMediaManager()
    return manager.schedule_first_hour_posts(
        video_title, video_url, niche, thumbnail, reddit_subreddits
    )


if __name__ == "__main__":
    # Test the social media manager
    logger.info("=" * 60)
    logger.info("Social Media Integration Test")
    logger.info("=" * 60)

    manager = SocialMediaManager()

    # Show platform status
    print("\nPlatform Configuration Status:")
    print("-" * 40)
    for platform, status in manager.get_platform_status().items():
        configured = "YES" if status["configured"] else "NO"
        print(f"  {status['name']:12} : {configured}")

    print(f"\nConfigured platforms: {[p.value for p in manager.get_configured_platforms()]}")

    # Test post generation for different platforms
    print("\n" + "=" * 60)
    print("Sample Generated Posts:")
    print("=" * 60)

    test_title = "5 Money Mistakes Costing You $10,000/Year"
    test_url = "https://youtube.com/watch?v=abc123"
    test_niche = "finance"

    for platform in Platform:
        content = manager.generate_post_content(test_title, test_url, test_niche, platform)
        print(f"\n--- {platform.value.upper()} ---")
        print(content)

    # Test posting (will simulate if not configured)
    print("\n" + "=" * 60)
    print("Test Post Results:")
    print("=" * 60)

    results = manager.post_to_all(
        test_title,
        test_url,
        test_niche,
        thumbnail_path=None,
        reddit_subreddits=["personalfinance"]
    )

    for post in results:
        status = "SUCCESS" if post.posted else f"FAILED ({post.error})"
        print(f"  {post.platform.value:12} : {status}")

    # Test first-hour scheduling
    print("\n" + "=" * 60)
    print("First-Hour Boost Schedule:")
    print("=" * 60)

    schedule_result = manager.schedule_first_hour_posts(
        "Test Video Title",
        "https://youtube.com/watch?v=test123",
        "tech"
    )

    print(f"\nTotal scheduled posts: {schedule_result['total_posts']}")
    for post_info in schedule_result['posts']:
        print(f"  {post_info['platform']:12} : +{post_info['delay']}min - {post_info['status']}")
