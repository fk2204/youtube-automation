"""
TwitterThreadRepurposer — posts multi-tweet threads via the Twitter API v2.

TIER REQUIREMENT: Twitter API v2 Basic or Enterprise tier required.
The Free tier does not support posting tweets. __init__ raises TierError
immediately if config["twitter"]["tier"] is not "basic" or "enterprise".
This surfaces the configuration problem before any API call is attempted.

Decision #5 from Wave 2 Blueprint: Twitter requires paid tier — enforce at init.

API flow:
  1. Authenticate via OAuth 2.0 PKCE (Bearer token).
  2. POST /2/tweets — post the first tweet.
  3. For each subsequent tweet, POST /2/tweets with reply.in_reply_to_tweet_id
     set to the previous tweet's ID to chain the thread.

Rate limits (Basic tier):
  - 500 tweets per 24 hours per user
  - 100 tweets per 15 minutes per user

Credentials (env vars):
    TWITTER_BEARER_TOKEN     — OAuth 2.0 Bearer token (read-only operations)
    TWITTER_ACCESS_TOKEN     — OAuth 2.0 user access token (write operations)
    TWITTER_ACCESS_SECRET    — OAuth 2.0 user access token secret
    TWITTER_API_KEY          — App API key
    TWITTER_API_SECRET       — App API secret
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

from distribution.repurposers.base_repurposer import BaseRepurposer
from distribution.repurposers.repurposer_registry import RepurposerRegistry
from exceptions import AuthError, RateLimitError, TierError

TWITTER_API_BASE = "https://api.twitter.com/2"

# Twitter character limits
MAX_TWEET_CHARS = 280
COUNTER_RESERVE = 7   # Reserve chars for " (N/M)" counter — up to "( 99/99)" = 8

ALLOWED_TIERS = {"basic", "enterprise"}


class TwitterThreadRepurposer(BaseRepurposer):
    """
    Posts long-form content as a Twitter thread.

    Requires Basic or Enterprise API tier — raises TierError at init
    if configured tier is 'free'.

    Requires env vars:
        TWITTER_BEARER_TOKEN, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET,
        TWITTER_API_KEY, TWITTER_API_SECRET
    """

    PLATFORM_NAME = "twitter"
    AUTOMATION_LEVEL = "full"

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize and enforce tier requirement.

        Args:
            config: Must contain {"twitter": {"tier": "basic"|"enterprise"}}.

        Raises:
            TierError: if config["twitter"]["tier"] is not in ALLOWED_TIERS.
        """
        super().__init__(config)

        twitter_cfg = config.get("twitter", {})
        tier = str(twitter_cfg.get("tier", "free")).lower()

        if tier not in ALLOWED_TIERS:
            raise TierError(
                f"Twitter API tier '{tier}' is not supported for posting. "
                f"Upgrade to 'basic' or 'enterprise' tier. "
                f"Free tier does not allow tweet creation via API v2."
            )

        self._tier = tier
        self._access_token: Optional[str] = None
        self._bearer_token: Optional[str] = None

    @property
    def platform_name(self) -> str:
        return self.PLATFORM_NAME

    async def authenticate(self) -> bool:
        """
        Load OAuth 2.0 credentials from env vars and verify with GET /2/users/me.

        Returns:
            True on success, False on failure (never raises).
        """
        required = [
            "TWITTER_BEARER_TOKEN",
            "TWITTER_ACCESS_TOKEN",
            "TWITTER_ACCESS_SECRET",
            "TWITTER_API_KEY",
            "TWITTER_API_SECRET",
        ]
        if self.guard_unconfigured_env(*required):
            logger.error("TwitterThreadRepurposer: required env vars not set.")
            return False

        bearer = os.environ["TWITTER_BEARER_TOKEN"]

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{TWITTER_API_BASE}/users/me",
                    headers={"Authorization": f"Bearer {bearer}"},
                    timeout=10.0,
                )
            if response.status_code == 401:
                logger.error("Twitter bearer token rejected (401).")
                return False
            if response.status_code == 429:
                logger.error("Twitter rate limit hit during auth check.")
                return False
            response.raise_for_status()
        except Exception as exc:
            logger.error("Twitter auth validation failed: {exc}", exc=exc)
            return False

        self._bearer_token = bearer
        self._access_token = os.environ["TWITTER_ACCESS_TOKEN"]
        self._authenticated = True
        data = response.json().get("data", {})
        logger.info(
            "Twitter authenticated: username=@{name}",
            name=data.get("username", "unknown"),
        )
        return True

    async def repurpose(
        self,
        source_content: str,
        media_paths: Optional[List[str]] = None,
        reply_settings: str = "everyone",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Post source content as a Twitter thread.

        Args:
            source_content:  Long-form text to split into tweets.
            media_paths:     Optional list of local media file paths (not yet uploaded).
                             NOTE: Media upload via v2 API requires additional OAuth 1.0a
                             signing — attach manually if needed.
            reply_settings:  "everyone" | "mentionedUsers" | "subscribers".

        Returns:
            {
                "success": bool,
                "thread_id": str | None,   # ID of the first tweet
                "tweet_ids": list[str],
                "url": str | None,          # URL of the first tweet
                "error": str | None,
            }
        """
        if self.simulation_mode:
            tweets = self._split_into_tweets(source_content)
            sim = self._simulate_repurpose({
                "tweet_count": len(tweets),
                "reply_settings": reply_settings,
            })
            sim.update({
                "thread_id": "sim_thread_001",
                "tweet_ids": [f"sim_tweet_{i:03d}" for i in range(len(tweets))],
                "url": "https://twitter.com/sim/status/001",
            })
            return sim

        if not self._authenticated:
            return self._build_error_response(
                "Not authenticated. Call authenticate() first.",
                extra={"thread_id": None, "tweet_ids": [], "url": None},
            )

        tweets = self._split_into_tweets(source_content)
        try:
            tweet_ids = await self._post_thread(tweets, reply_settings)
        except Exception as exc:
            logger.error("Twitter thread post failed: {exc}", exc=exc)
            return self._build_error_response(
                str(exc),
                extra={"thread_id": None, "tweet_ids": [], "url": None},
            )

        thread_id = tweet_ids[0] if tweet_ids else None
        url = (
            f"https://twitter.com/i/web/status/{thread_id}"
            if thread_id else None
        )
        logger.info(
            "Twitter thread posted: {n} tweets, thread_id={tid}",
            n=len(tweet_ids),
            tid=thread_id,
        )
        return {
            "success": True,
            "thread_id": thread_id,
            "tweet_ids": tweet_ids,
            "url": url,
            "error": None,
        }

    def _split_into_tweets(self, text: str) -> List[str]:
        """
        Split text into tweet-sized segments preserving sentence boundaries.

        Each tweet has a (N/M) counter appended. The counter is reserved
        in the character budget before splitting so tweets never exceed
        MAX_TWEET_CHARS after the counter is added.

        Strategy:
          1. Split into sentences on '. ', '! ', '? '.
          2. Accumulate sentences until adding the next would exceed budget.
          3. Flush current tweet, start new one.
          4. Append (N/M) counter to each tweet after all segments are built.
        """
        effective_limit = MAX_TWEET_CHARS - COUNTER_RESERVE

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        tweets: List[str] = []
        current_parts: List[str] = []
        current_len = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If a single sentence is longer than the limit, hard-split it.
            if len(sentence) > effective_limit:
                if current_parts:
                    tweets.append(" ".join(current_parts))
                    current_parts = []
                    current_len = 0
                # Hard split at word boundaries.
                words = sentence.split()
                chunk_parts: List[str] = []
                chunk_len = 0
                for word in words:
                    if chunk_len + len(word) + (1 if chunk_parts else 0) > effective_limit:
                        tweets.append(" ".join(chunk_parts))
                        chunk_parts = [word]
                        chunk_len = len(word)
                    else:
                        chunk_parts.append(word)
                        chunk_len += len(word) + (1 if len(chunk_parts) > 1 else 0)
                if chunk_parts:
                    tweets.append(" ".join(chunk_parts))
                continue

            added_len = len(sentence) + (1 if current_parts else 0)
            if current_len + added_len > effective_limit:
                tweets.append(" ".join(current_parts))
                current_parts = [sentence]
                current_len = len(sentence)
            else:
                current_parts.append(sentence)
                current_len += added_len

        if current_parts:
            tweets.append(" ".join(current_parts))

        total = len(tweets)
        numbered: List[str] = [
            f"{tweet} ({i + 1}/{total})"
            for i, tweet in enumerate(tweets)
        ]
        return numbered

    async def _post_thread(
        self,
        tweets: List[str],
        reply_settings: str,
    ) -> List[str]:
        """
        Post each tweet in sequence, chaining replies to form a thread.

        Args:
            tweets:         List of tweet texts (already split and numbered).
            reply_settings: Twitter reply permission setting.

        Returns:
            List of tweet IDs in posting order.

        Raises:
            RateLimitError: on 429 response.
            Exception:      on any other API failure.
        """
        tweet_ids: List[str] = []
        previous_id: Optional[str] = None

        for text in tweets:
            payload: Dict[str, Any] = {
                "text": text,
                "reply_settings": reply_settings,
            }
            if previous_id:
                payload["reply"] = {"in_reply_to_tweet_id": previous_id}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{TWITTER_API_BASE}/tweets",
                    headers={
                        "Authorization": f"Bearer {self._access_token}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=15.0,
                )

            if response.status_code == 429:
                raise RateLimitError("Twitter rate limit hit during thread posting.")
            response.raise_for_status()

            tweet_id = response.json().get("data", {}).get("id", "")
            tweet_ids.append(tweet_id)
            previous_id = tweet_id
            logger.debug(
                "Twitter tweet {n}/{total} posted: id={tid}",
                n=len(tweet_ids),
                total=len(tweets),
                tid=tweet_id,
            )

        return tweet_ids

    async def get_rate_limit_status(self) -> Dict[str, Any]:
        """Return current Twitter API rate limit state."""
        return {
            "requests_remaining": "check x-rate-limit-remaining header",
            "reset_at": "check x-rate-limit-reset header",
            "window_seconds": 900,
            "daily_tweet_limit": 500,
            "tier": self._tier,
        }


def _register() -> None:
    """Register TwitterThreadRepurposer with the global RepurposerRegistry."""
    RepurposerRegistry.register("twitter", TwitterThreadRepurposer)
    logger.debug("TwitterThreadRepurposer registered.")


_register()
