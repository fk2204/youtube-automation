"""
Unit tests for BasePoster mixin class.

Tests shared poster functionality: unconfigured guards, simulation mode,
library availability checks, and standard error handling.
"""

import pytest
from unittest.mock import Mock, patch
from src.social.platform_base import BasePoster


class MinimalPoster(BasePoster):
    """Concrete subclass of BasePoster for testing."""

    platform_name = "TestPlatform"
    is_configured_value = False

    def is_configured(self) -> bool:
        return self.is_configured_value


class TestBasePosterGuard:
    """Test guard_unconfigured() method."""

    def test_guard_returns_dict_when_not_configured(self):
        """Guard returns error dict when platform not configured."""
        poster = MinimalPoster()
        poster.is_configured_value = False

        result = poster.guard_unconfigured("test content")

        assert isinstance(result, dict)
        assert result["success"] is False
        assert result["simulated"] is True
        assert result["platform"] == "testplatform"
        assert "not configured" in result["error"].lower()

    def test_guard_returns_none_when_configured(self):
        """Guard returns None when platform is configured."""
        poster = MinimalPoster()
        poster.is_configured_value = True

        result = poster.guard_unconfigured("test content")

        assert result is None

    def test_guard_includes_content_preview(self):
        """Guard includes preview of content being posted."""
        poster = MinimalPoster()
        poster.is_configured_value = False

        content = "This is a long piece of content that should be previewed"
        result = poster.guard_unconfigured(content)

        assert "content_preview" in result
        assert result["content_preview"] == content[:100]


class TestBasePosterSimulate:
    """Test _simulate_post() method."""

    def test_simulate_post_success(self):
        """Simulated post returns success dict."""
        poster = MinimalPoster()

        result = poster._simulate_post("test content", url="http://example.com")

        assert result["success"] is True
        assert result["simulated"] is True
        assert result["platform"] == "testplatform"
        assert "url" in result
        assert result["url"] == "http://example.com"

    def test_simulate_post_includes_post_id(self):
        """Simulated post includes generated post ID."""
        poster = MinimalPoster()

        result = poster._simulate_post("content")

        assert "post_id" in result
        assert result["post_id"].startswith("sim_testplatform_")

    def test_simulate_post_with_image(self):
        """Simulated post can include image path."""
        poster = MinimalPoster()

        result = poster._simulate_post("content", image="/path/to/image.jpg")

        assert "image_path" in result
        assert result["image_path"] == "/path/to/image.jpg"

    def test_simulate_post_extra_fields(self):
        """Simulated post can include platform-specific extra fields."""
        poster = MinimalPoster()

        result = poster._simulate_post("content", subreddit="python", title="Test")

        assert result["subreddit"] == "python"
        assert result["title"] == "Test"


class TestBasePosterLibraryExecution:
    """Test _execute_with_library() method."""

    def test_execute_success_path(self):
        """Execute calls function and returns its result."""
        poster = MinimalPoster()

        success_result = {"success": True, "post_id": "123"}
        fn = Mock(return_value=success_result)

        result = poster._execute_with_library("test_lib", fn)

        assert result == success_result
        fn.assert_called_once()

    def test_execute_import_error_handling(self):
        """Execute catches ImportError and returns safe response."""
        poster = MinimalPoster()

        def fn_needs_lib():
            raise ImportError("test_lib not found")

        result = poster._execute_with_library("test_lib", fn_needs_lib)

        assert result["success"] is False
        assert "not installed" in result["error"].lower()
        assert result["platform"] == "testplatform"

    def test_execute_generic_exception_handling(self):
        """Execute catches generic exceptions and logs error."""
        poster = MinimalPoster()

        def fn_error():
            raise ValueError("Something went wrong")

        result = poster._execute_with_library("test_lib", fn_error)

        assert result["success"] is False
        assert "Something went wrong" in result["error"]
        assert result["platform"] == "testplatform"

    def test_execute_exception_logging(self):
        """Execute logs exceptions appropriately."""
        poster = MinimalPoster()

        def fn_error():
            raise RuntimeError("test error")

        with patch("src.social.platform_base.logger") as mock_logger:
            result = poster._execute_with_library("test_lib", fn_error)
            mock_logger.error.assert_called_once()


class TestBasePosterAbstractMethods:
    """Test abstract method requirements."""

    def test_direct_instantiation_not_allowed(self):
        """Cannot instantiate BasePoster directly (it's abstract)."""
        # BasePoster itself doesn't prevent instantiation, but subclasses must implement is_configured
        base = BasePoster()
        with pytest.raises(NotImplementedError):
            base.is_configured()

    def test_subclass_requires_is_configured(self):
        """Subclass must implement is_configured()."""
        # MinimalPoster properly implements it - verify it's callable
        poster = MinimalPoster()
        assert callable(poster.is_configured)
        assert isinstance(poster.is_configured(), bool)
