"""
Tests for the VideoAssembler module.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


class TestVideoAssemblerInit:
    """Tests for VideoAssembler initialization."""

    def test_default_initialization(self):
        """Test VideoAssembler initializes with default values."""
        with patch("src.content.video_assembler.AUDIO_PROCESSOR_AVAILABLE", False):
            from src.content.video_assembler import VideoAssembler

            assembler = VideoAssembler()

            assert assembler.resolution == (1920, 1080)
            assert assembler.fps == 30
            assert assembler.background_color == (20, 20, 30)
            assert assembler.width == 1920
            assert assembler.height == 1080

    def test_custom_resolution(self):
        """Test VideoAssembler with custom resolution."""
        with patch("src.content.video_assembler.AUDIO_PROCESSOR_AVAILABLE", False):
            from src.content.video_assembler import VideoAssembler

            assembler = VideoAssembler(resolution=(1280, 720))

            assert assembler.resolution == (1280, 720)
            assert assembler.width == 1280
            assert assembler.height == 720

    def test_custom_fps(self):
        """Test VideoAssembler with custom FPS."""
        with patch("src.content.video_assembler.AUDIO_PROCESSOR_AVAILABLE", False):
            from src.content.video_assembler import VideoAssembler

            assembler = VideoAssembler(fps=60)

            assert assembler.fps == 60

    def test_custom_background_color(self):
        """Test VideoAssembler with custom background color."""
        with patch("src.content.video_assembler.AUDIO_PROCESSOR_AVAILABLE", False):
            from src.content.video_assembler import VideoAssembler

            assembler = VideoAssembler(background_color=(255, 0, 0))

            assert assembler.background_color == (255, 0, 0)


class TestVideoSegment:
    """Tests for VideoSegment dataclass."""

    def test_video_segment_creation(self):
        """Test VideoSegment can be created with required fields."""
        from src.content.video_assembler import VideoSegment

        segment = VideoSegment(
            start_time=0.0,
            end_time=10.0,
            content_type="text",
            content="Hello World"
        )

        assert segment.start_time == 0.0
        assert segment.end_time == 10.0
        assert segment.content_type == "text"
        assert segment.content == "Hello World"
        assert segment.text_overlay is None

    def test_video_segment_with_overlay(self):
        """Test VideoSegment with text overlay."""
        from src.content.video_assembler import VideoSegment

        segment = VideoSegment(
            start_time=0.0,
            end_time=5.0,
            content_type="image",
            content="/path/to/image.png",
            text_overlay="Caption"
        )

        assert segment.text_overlay == "Caption"


class TestCreateBackgroundClip:
    """Tests for create_background_clip method."""

    def test_create_background_clip_default_color(self):
        """Test creating background clip with default color."""
        with patch("src.content.video_assembler.AUDIO_PROCESSOR_AVAILABLE", False):
            with patch("src.content.video_assembler.ColorClip") as mock_color_clip:
                from src.content.video_assembler import VideoAssembler

                mock_instance = MagicMock()
                mock_color_clip.return_value = mock_instance

                assembler = VideoAssembler()
                clip = assembler.create_background_clip(duration=10.0)

                mock_color_clip.assert_called_once_with(
                    size=(1920, 1080),
                    color=(20, 20, 30),
                    duration=10.0
                )

    def test_create_background_clip_custom_color(self):
        """Test creating background clip with custom color."""
        with patch("src.content.video_assembler.AUDIO_PROCESSOR_AVAILABLE", False):
            with patch("src.content.video_assembler.ColorClip") as mock_color_clip:
                from src.content.video_assembler import VideoAssembler

                mock_instance = MagicMock()
                mock_color_clip.return_value = mock_instance

                assembler = VideoAssembler()
                clip = assembler.create_background_clip(
                    duration=5.0,
                    color=(100, 100, 100)
                )

                mock_color_clip.assert_called_once_with(
                    size=(1920, 1080),
                    color=(100, 100, 100),
                    duration=5.0
                )


class TestCreateThumbnail:
    """Tests for create_thumbnail method."""

    def test_create_thumbnail_creates_file(self, temp_dir):
        """Test that create_thumbnail creates an image file."""
        with patch("src.content.video_assembler.AUDIO_PROCESSOR_AVAILABLE", False):
            from src.content.video_assembler import VideoAssembler

            assembler = VideoAssembler()
            output_path = str(temp_dir / "thumbnail.png")

            result = assembler.create_thumbnail(
                output_file=output_path,
                title="Test Title"
            )

            assert result == output_path
            assert os.path.exists(output_path)

    def test_create_thumbnail_with_subtitle(self, temp_dir):
        """Test create_thumbnail with subtitle."""
        with patch("src.content.video_assembler.AUDIO_PROCESSOR_AVAILABLE", False):
            from src.content.video_assembler import VideoAssembler

            assembler = VideoAssembler()
            output_path = str(temp_dir / "thumbnail_with_sub.png")

            result = assembler.create_thumbnail(
                output_file=output_path,
                title="Main Title",
                subtitle="Subtitle Text"
            )

            assert result == output_path
            assert os.path.exists(output_path)

    def test_create_thumbnail_custom_colors(self, temp_dir):
        """Test create_thumbnail with custom colors."""
        with patch("src.content.video_assembler.AUDIO_PROCESSOR_AVAILABLE", False):
            from src.content.video_assembler import VideoAssembler

            assembler = VideoAssembler()
            output_path = str(temp_dir / "thumbnail_custom.png")

            result = assembler.create_thumbnail(
                output_file=output_path,
                title="Custom",
                background_color=(255, 0, 0),
                text_color="yellow"
            )

            assert result == output_path
            assert os.path.exists(output_path)


class TestCreateTitleCard:
    """Tests for create_title_card method."""

    def test_create_title_card_basic(self):
        """Test creating a basic title card."""
        with patch("src.content.video_assembler.AUDIO_PROCESSOR_AVAILABLE", False):
            with patch("src.content.video_assembler.ColorClip") as mock_color:
                with patch("src.content.video_assembler.TextClip") as mock_text:
                    with patch("src.content.video_assembler.CompositeVideoClip") as mock_composite:
                        from src.content.video_assembler import VideoAssembler

                        # Setup mocks
                        mock_color_instance = MagicMock()
                        mock_color.return_value = mock_color_instance

                        mock_text_instance = MagicMock()
                        mock_text_instance.set_duration.return_value = mock_text_instance
                        mock_text_instance.set_position.return_value = mock_text_instance
                        mock_text.return_value = mock_text_instance

                        mock_composite_instance = MagicMock()
                        mock_composite.return_value = mock_composite_instance

                        assembler = VideoAssembler()
                        result = assembler.create_title_card(
                            title="Test Title",
                            duration=5.0
                        )

                        mock_composite.assert_called_once()
                        assert result == mock_composite_instance


class TestVideoAssemblerIntegration:
    """Integration-style tests (with mocking of external dependencies)."""

    def test_video_assembler_resolution_affects_clips(self):
        """Test that resolution setting affects clip creation."""
        with patch("src.content.video_assembler.AUDIO_PROCESSOR_AVAILABLE", False):
            with patch("src.content.video_assembler.ColorClip") as mock_color:
                from src.content.video_assembler import VideoAssembler

                mock_instance = MagicMock()
                mock_color.return_value = mock_instance

                # Test with 4K resolution
                assembler = VideoAssembler(resolution=(3840, 2160))
                assembler.create_background_clip(duration=1.0)

                mock_color.assert_called_with(
                    size=(3840, 2160),
                    color=(20, 20, 30),
                    duration=1.0
                )
