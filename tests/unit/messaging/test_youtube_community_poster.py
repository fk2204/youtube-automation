"""
Unit tests for YouTubeCommunityPoster.

Verifies that all methods raise NotImplementedError (API unavailable)
and that simulate() returns a valid mock success response for pipeline testing.
No credentials required — this poster never makes API calls.
"""

import os
import sys
import pytest

# Ensure src/ is on path for direct pytest invocation from project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from registry.poster_registry import PosterRegistry

# Import triggers _register() at module level.
import social.youtube_community_poster as youtube_community_module
from social.youtube_community_poster import YouTubeCommunityPoster, BLOCKER_REASON


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def poster() -> YouTubeCommunityPoster:
  return YouTubeCommunityPoster()


# ---------------------------------------------------------------------------
# Registration test
# ---------------------------------------------------------------------------


class TestAllMethodsRaiseNotImplementedError:
  @pytest.mark.asyncio
  async def test_authenticate_raises_not_implemented(
    self,
    poster: YouTubeCommunityPoster,
  ) -> None:
    """authenticate() must raise NotImplementedError with the blocker reason."""
    with pytest.raises(NotImplementedError, match="unavailable"):
      await poster.authenticate()

  @pytest.mark.asyncio
  async def test_post_community_update_raises_not_implemented(
    self,
    poster: YouTubeCommunityPoster,
  ) -> None:
    """post_community_update() must raise NotImplementedError."""
    with pytest.raises(NotImplementedError, match="unavailable"):
      await poster.post_community_update("Hello community!")

  @pytest.mark.asyncio
  async def test_send_video_raises_not_implemented(
    self,
    poster: YouTubeCommunityPoster,
  ) -> None:
    """send_video() must raise NotImplementedError."""
    with pytest.raises(NotImplementedError, match="unavailable"):
      await poster.send_video("/tmp/video.mp4", "caption")

  @pytest.mark.asyncio
  async def test_send_message_raises_not_implemented(
    self,
    poster: YouTubeCommunityPoster,
  ) -> None:
    """send_message() must raise NotImplementedError."""
    with pytest.raises(NotImplementedError, match="unavailable"):
      await poster.send_message("Community post text")

  @pytest.mark.asyncio
  async def test_get_rate_limit_status_raises_not_implemented(
    self,
    poster: YouTubeCommunityPoster,
  ) -> None:
    """get_rate_limit_status() must raise NotImplementedError."""
    with pytest.raises(NotImplementedError, match="unavailable"):
      await poster.get_rate_limit_status()


# ---------------------------------------------------------------------------
# simulate() test — must return mock success without raising
# ---------------------------------------------------------------------------

class TestSimulate:
  def test_simulate_returns_mock_success_response(
    self,
    poster: YouTubeCommunityPoster,
  ) -> None:
    """simulate() must return a valid mock success dict without raising."""
    result = poster.simulate()

    assert result["success"] is True
    assert result["simulated"] is True
    assert result["platform"] == "youtube_community"
    assert result["message_id"] is not None
    assert result["error"] is None
    assert "note" in result
