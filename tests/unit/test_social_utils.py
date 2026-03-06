"""
Unit tests for social_utils helper functions.

Tests video writing, video info reading, HTTP error parsing,
and short-form platform constants.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.social.social_utils import (
    parse_http_error,
    write_video,
    get_video_duration,
    VideoInfo,
    SHORT_FORM_PLATFORMS,
    SHORT_FORM_ASPECT_RATIOS,
    SHORT_FORM_DURATION,
)


class TestParseHttpError:
    """Test HTTP error parsing."""

    def test_parse_json_error_field(self):
        """Parse error from JSON response."""
        response = Mock()
        response.status_code = 400
        response.json.return_value = {"error": "bad input"}

        result = parse_http_error(response, "TestPlatform")

        assert result["success"] is False
        assert "HTTP 400" in result["error"]
        assert "bad input" in result["error"]
        assert result["status_code"] == 400
        assert result["platform"] == "testplatform"

    def test_parse_text_fallback(self):
        """Fall back to str(response) when JSON parsing fails."""
        response = Mock()
        response.status_code = 500
        response.json.side_effect = ValueError("Not JSON")
        response.text = "Internal Server Error"

        result = parse_http_error(response, "TestPlatform")

        assert "HTTP 500" in result["error"]
        # The error text will be either response.text (if hasattr succeeds) or str(response)
        assert result["error"].startswith("HTTP 500:")

    def test_parse_message_field(self):
        """Parse message field from JSON if error field missing."""
        response = Mock()
        response.status_code = 401
        response.json.return_value = {"message": "unauthorized"}

        result = parse_http_error(response, "TestPlatform")

        assert "unauthorized" in result["error"]


class TestWriteVideo:
    """Test video writing function."""

    def test_write_video_success(self):
        """Write video returns True when successful."""
        clip = Mock()
        clip.write_videofile = Mock()

        with patch("src.social.social_utils.os.makedirs"), \
             patch("src.social.social_utils.os.path.exists", return_value=True), \
             patch("src.social.social_utils.os.path.getsize", return_value=5 * 1024 * 1024):
            result = write_video(clip, "/tmp/test.mp4")

        assert result is True
        clip.write_videofile.assert_called_once()

    def test_write_video_failure_on_exception(self):
        """Write video returns False on exception."""
        clip = Mock()
        clip.write_videofile.side_effect = RuntimeError("Encoding failed")

        with patch("src.social.social_utils.os.makedirs"):
            result = write_video(clip, "/tmp/test.mp4")

        assert result is False

    def test_write_video_checks_file_created(self):
        """Write video verifies output file exists."""
        clip = Mock()

        with patch("src.social.social_utils.os.makedirs"), \
             patch("src.social.social_utils.os.path.exists", return_value=False):
            result = write_video(clip, "/tmp/test.mp4")

        assert result is False

    def test_write_video_passes_parameters(self):
        """Write video passes bitrate and other params to clip."""
        clip = Mock()

        with patch("src.social.social_utils.os.makedirs"), \
             patch("src.social.social_utils.os.path.exists", return_value=True), \
             patch("src.social.social_utils.os.path.getsize", return_value=1024):
            write_video(clip, "/tmp/test.mp4", bitrate="6M", fps=24, preset="slow")

        call_kwargs = clip.write_videofile.call_args[1]
        assert call_kwargs["bitrate"] == "6M"
        assert call_kwargs["fps"] == 24
        assert call_kwargs["preset"] == "slow"


class TestGetVideoDuration:
    """Test video duration extraction."""

    def test_get_duration_returns_zero_without_moviepy(self):
        """Get duration returns 0 if MoviePy not available."""
        with patch("src.social.social_utils.get_video_info") as mock_get_info:
            mock_get_info.return_value.__enter__.return_value = VideoInfo(
                duration=0, width=0, height=0
            )
            result = get_video_duration("/tmp/video.mp4")

        assert result == 0.0

    def test_get_duration_returns_actual_duration(self):
        """Get duration returns actual video duration."""
        with patch("src.social.social_utils.get_video_info") as mock_get_info:
            mock_info = VideoInfo(duration=30.5, width=1920, height=1080)
            mock_get_info.return_value.__enter__.return_value = mock_info
            result = get_video_duration("/tmp/video.mp4")

        assert result == 30.5


class TestVideoInfo:
    """Test VideoInfo dataclass."""

    def test_video_info_instantiation(self):
        """VideoInfo dataclass instantiates with all fields."""
        info = VideoInfo(duration=30.0, width=1920, height=1080, fps=30.0)

        assert info.duration == 30.0
        assert info.width == 1920
        assert info.height == 1080
        assert info.fps == 30.0

    def test_video_info_default_fps(self):
        """VideoInfo has default fps of 30."""
        info = VideoInfo(duration=10.0, width=1080, height=1920)

        assert info.fps == 30.0


class TestShortFormConstants:
    """Test short-form platform constants."""

    def test_platforms_are_exactly_three(self):
        """SHORT_FORM_PLATFORMS contains exactly 3 platforms."""
        assert len(SHORT_FORM_PLATFORMS) == 3

    def test_platforms_are_correct(self):
        """SHORT_FORM_PLATFORMS contains correct platforms."""
        assert "youtube_shorts" in SHORT_FORM_PLATFORMS
        assert "tiktok" in SHORT_FORM_PLATFORMS
        assert "instagram_reels" in SHORT_FORM_PLATFORMS

    def test_aspect_ratios_are_9_16(self):
        """All aspect ratios are 9:16 (1080x1920)."""
        for platform, ratio in SHORT_FORM_ASPECT_RATIOS.items():
            assert ratio == (1080, 1920), f"{platform} has wrong ratio: {ratio}"

    def test_aspect_ratios_all_platforms(self):
        """All platforms have aspect ratio defined."""
        for platform in SHORT_FORM_PLATFORMS:
            assert platform in SHORT_FORM_ASPECT_RATIOS

    def test_duration_constraints_reasonable(self):
        """Duration constraints have reasonable min/max."""
        for platform, (min_dur, max_dur) in SHORT_FORM_DURATION.items():
            assert min_dur > 0, f"{platform} min_dur <= 0"
            assert max_dur > min_dur, f"{platform} max_dur <= min_dur"
            assert max_dur <= 600, f"{platform} max_dur too large"

    def test_duration_all_platforms(self):
        """All platforms have duration constraints defined."""
        for platform in SHORT_FORM_PLATFORMS:
            assert platform in SHORT_FORM_DURATION
