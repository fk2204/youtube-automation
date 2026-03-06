"""
Integration tests for module imports and circular dependency checks.

Verifies that all refactored modules can be imported without circular
dependencies and that short-form platform constants are correctly defined.
"""

import pytest
import importlib.util
from pathlib import Path


class TestImports:
    """Verify all module imports work without circular dependencies."""

    def test_platform_base_imports(self):
        """Platform base module imports successfully."""
        from src.social.platform_base import BasePoster
        assert BasePoster is not None
        assert callable(BasePoster)

    def test_social_utils_imports(self):
        """Social utils module imports and short-form platforms defined."""
        from src.social.social_utils import (
            SHORT_FORM_PLATFORMS,
            SHORT_FORM_ASPECT_RATIOS,
            SHORT_FORM_DURATION,
        )
        assert SHORT_FORM_PLATFORMS == ["youtube_shorts", "tiktok", "instagram_reels"]
        assert len(SHORT_FORM_PLATFORMS) == 3

    def test_social_poster_imports(self):
        """Social poster classes import successfully."""
        from src.social.social_poster import (
            SocialMediaManager,
            TwitterPoster,
            RedditPoster,
            DiscordPoster,
            LinkedInPoster,
            FacebookPoster,
        )
        assert TwitterPoster is not None
        assert RedditPoster is not None
        assert DiscordPoster is not None
        assert LinkedInPoster is not None
        assert FacebookPoster is not None
        assert SocialMediaManager is not None

    def test_multi_platform_imports(self):
        """Multi-platform module imports successfully."""
        from src.social.multi_platform import MultiPlatformDistributor, Platform
        assert MultiPlatformDistributor is not None
        assert Platform is not None

    def test_video_utils_imports(self):
        """Video utilities module imports successfully."""
        from src.content.video_utils import (
            find_ffmpeg,
            two_pass_encode,
            FFMPEG_PARAMS_REGULAR,
            FFMPEG_PARAMS_SHORTS,
        )
        assert callable(find_ffmpeg)
        assert callable(two_pass_encode)
        assert isinstance(FFMPEG_PARAMS_REGULAR, list)
        assert isinstance(FFMPEG_PARAMS_SHORTS, list)

    def test_no_circular_dependency_forward(self):
        """No circular dependencies when importing in forward order."""
        import src.social.platform_base
        import src.social.social_utils
        import src.social.social_poster
        import src.social.multi_platform
        # If we get here, no circular import occurred

    def test_no_circular_dependency_reverse(self):
        """No circular dependencies when importing in reverse order."""
        import src.social.multi_platform
        import src.social.social_poster
        import src.social.social_utils
        import src.social.platform_base
        # If we get here, no circular import occurred

    def test_short_form_constants_complete(self):
        """Short-form constants all present and consistent."""
        from src.social.social_utils import (
            SHORT_FORM_PLATFORMS,
            SHORT_FORM_ASPECT_RATIOS,
            SHORT_FORM_DURATION,
        )

        # All platforms should have aspect ratio and duration definitions
        assert set(SHORT_FORM_ASPECT_RATIOS.keys()) == set(SHORT_FORM_PLATFORMS)
        assert set(SHORT_FORM_DURATION.keys()) == set(SHORT_FORM_PLATFORMS)

        # All aspect ratios should be 9:16 (1080x1920)
        for platform, ratio in SHORT_FORM_ASPECT_RATIOS.items():
            assert ratio == (1080, 1920), f"{platform} has incorrect aspect ratio: {ratio}"

        # All durations should have reasonable min/max
        for platform, (min_dur, max_dur) in SHORT_FORM_DURATION.items():
            assert min_dur > 0, f"{platform} has invalid min duration: {min_dur}"
            assert max_dur > min_dur, f"{platform} has max_dur <= min_dur"
            assert max_dur <= 600, f"{platform} has unreasonable max duration: {max_dur}"
