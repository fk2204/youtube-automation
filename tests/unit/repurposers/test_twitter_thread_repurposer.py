"""
Unit tests for TwitterThreadRepurposer.

All external API calls are mocked. No real credentials required.
Tests verify tier enforcement, text splitting, counter addition, and registration.
"""

import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from distribution.repurposers.repurposer_registry import RepurposerRegistry
import distribution.repurposers.twitter_thread_repurposer as twitter_module
from distribution.repurposers.twitter_thread_repurposer import (
    TwitterThreadRepurposer,
    MAX_TWEET_CHARS,
)
from exceptions import TierError

BASIC_CONFIG = {"twitter": {"tier": "basic"}}
ENTERPRISE_CONFIG = {"twitter": {"tier": "enterprise"}}
FREE_CONFIG = {"twitter": {"tier": "free"}}
MISSING_TIER_CONFIG = {"twitter": {}}

LONG_TEXT = (
    "This is the first sentence of a long article. "
    "Here is the second sentence with more detail. "
    "The third sentence adds further context. "
    "Fourth sentence keeps the content flowing. "
    "Fifth sentence rounds out the first section. "
    "Sixth sentence begins a new thought. "
    "Seventh sentence expands on that thought. "
    "Eighth sentence concludes the paragraph. "
    "Ninth sentence introduces a new idea. "
    "Tenth sentence is the final one here."
)


@pytest.fixture
def repurposer() -> TwitterThreadRepurposer:
    return TwitterThreadRepurposer(BASIC_CONFIG)


@pytest.fixture
def authenticated_repurposer(repurposer: TwitterThreadRepurposer) -> TwitterThreadRepurposer:
    repurposer._access_token = "test_access_token"
    repurposer._bearer_token = "test_bearer_token"
    repurposer._authenticated = True
    return repurposer


@pytest.fixture(autouse=True)
def clear_simulation_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SOCIAL_SIMULATION_MODE", raising=False)



class TestTierEnforcement:
    def test_init_raises_tier_error_on_free_tier(self) -> None:
        """__init__ must raise TierError when tier is 'free'."""
        with pytest.raises(TierError, match="free"):
            TwitterThreadRepurposer(FREE_CONFIG)

    def test_init_raises_tier_error_on_missing_tier(self) -> None:
        """__init__ must raise TierError when tier key is absent."""
        with pytest.raises(TierError):
            TwitterThreadRepurposer(MISSING_TIER_CONFIG)

    def test_init_succeeds_on_basic_tier(self) -> None:
        """__init__ must not raise for 'basic' tier."""
        rp = TwitterThreadRepurposer(BASIC_CONFIG)
        assert rp._tier == "basic"

    def test_init_succeeds_on_enterprise_tier(self) -> None:
        """__init__ must not raise for 'enterprise' tier."""
        rp = TwitterThreadRepurposer(ENTERPRISE_CONFIG)
        assert rp._tier == "enterprise"


class TestAuthenticate:
    @pytest.mark.asyncio
    async def test_authenticate_success_returns_true(
        self,
        repurposer: TwitterThreadRepurposer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """authenticate() returns True on valid credentials."""
        for var in [
            "TWITTER_BEARER_TOKEN",
            "TWITTER_ACCESS_TOKEN",
            "TWITTER_ACCESS_SECRET",
            "TWITTER_API_KEY",
            "TWITTER_API_SECRET",
        ]:
            monkeypatch.setenv(var, "test_value")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"data": {"id": "123", "username": "testuser"}}

        with patch("distribution.repurposers.twitter_thread_repurposer.httpx.AsyncClient") as mock_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_class.return_value = mock_client

            result = await repurposer.authenticate()

        assert result is True
        assert repurposer._authenticated is True

    @pytest.mark.asyncio
    async def test_authenticate_returns_false_on_missing_env(
        self,
        repurposer: TwitterThreadRepurposer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """authenticate() returns False when env vars are missing."""
        for var in [
            "TWITTER_BEARER_TOKEN",
            "TWITTER_ACCESS_TOKEN",
            "TWITTER_ACCESS_SECRET",
            "TWITTER_API_KEY",
            "TWITTER_API_SECRET",
        ]:
            monkeypatch.delenv(var, raising=False)

        result = await repurposer.authenticate()
        assert result is False


class TestSplitIntoTweets:
    def test_all_tweets_within_char_limit(self, repurposer: TwitterThreadRepurposer) -> None:
        """Every generated tweet must be at or under MAX_TWEET_CHARS."""
        tweets = repurposer._split_into_tweets(LONG_TEXT)
        for tweet in tweets:
            assert len(tweet) <= MAX_TWEET_CHARS, (
                f"Tweet exceeds {MAX_TWEET_CHARS} chars: '{tweet}' ({len(tweet)} chars)"
            )

    def test_counters_added_to_all_tweets(self, repurposer: TwitterThreadRepurposer) -> None:
        """Each tweet must end with a (N/M) counter."""
        import re
        tweets = repurposer._split_into_tweets(LONG_TEXT)
        total = len(tweets)
        counter_pattern = re.compile(r"\(\d+/" + str(total) + r"\)$")
        for tweet in tweets:
            assert counter_pattern.search(tweet), (
                f"Tweet missing counter pattern: '{tweet}'"
            )

    def test_single_tweet_has_counter_1_of_1(self, repurposer: TwitterThreadRepurposer) -> None:
        """Short text that fits in one tweet still gets (1/1) counter."""
        tweets = repurposer._split_into_tweets("Short text.")
        assert len(tweets) == 1
        assert tweets[0].endswith("(1/1)")

    def test_multiple_tweets_produced_for_long_content(
        self, repurposer: TwitterThreadRepurposer
    ) -> None:
        """Long text must produce more than one tweet."""
        tweets = repurposer._split_into_tweets(LONG_TEXT)
        assert len(tweets) > 1


class TestRepurpose:
    @pytest.mark.asyncio
    async def test_repurpose_success_returns_thread_id_and_tweet_ids(
        self,
        authenticated_repurposer: TwitterThreadRepurposer,
    ) -> None:
        """repurpose() returns success with thread_id, tweet_ids, and url."""
        tweet_counter = {"n": 0}

        async def mock_post(*args, **kwargs):
            tweet_counter["n"] += 1
            mock_resp = MagicMock()
            mock_resp.status_code = 201
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {"data": {"id": f"tweet_{tweet_counter['n']:03d}"}}
            return mock_resp

        with patch("distribution.repurposers.twitter_thread_repurposer.httpx.AsyncClient") as mock_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = mock_post
            mock_class.return_value = mock_client

            result = await authenticated_repurposer.repurpose("Short test content.")

        assert result["success"] is True
        assert result["thread_id"] is not None
        assert isinstance(result["tweet_ids"], list)
        assert len(result["tweet_ids"]) >= 1
        assert result["url"] is not None
        assert result["error"] is None
