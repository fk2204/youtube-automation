"""
Pytest configuration and fixtures for YouTube Automation tests.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_audio_path(temp_dir):
    """Create a path for a sample audio file."""
    return temp_dir / "test_audio.mp3"


@pytest.fixture
def sample_video_path(temp_dir):
    """Create a path for a sample video file."""
    return temp_dir / "test_video.mp4"


@pytest.fixture
def sample_thumbnail_path(temp_dir):
    """Create a path for a sample thumbnail file."""
    return temp_dir / "test_thumbnail.png"


@pytest.fixture
def mock_audio_clip():
    """Mock MoviePy AudioFileClip."""
    with patch("moviepy.editor.AudioFileClip") as mock:
        mock_instance = MagicMock()
        mock_instance.duration = 60.0
        mock_instance.close = MagicMock()
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_video_file_clip():
    """Mock MoviePy VideoFileClip."""
    with patch("moviepy.editor.VideoFileClip") as mock:
        mock_instance = MagicMock()
        mock_instance.duration = 60.0
        mock_instance.audio = MagicMock()
        mock_instance.close = MagicMock()
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_color_clip():
    """Mock MoviePy ColorClip."""
    with patch("moviepy.editor.ColorClip") as mock:
        mock_instance = MagicMock()
        mock_instance.set_duration = MagicMock(return_value=mock_instance)
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_text_clip():
    """Mock MoviePy TextClip."""
    with patch("moviepy.editor.TextClip") as mock:
        mock_instance = MagicMock()
        mock_instance.set_duration = MagicMock(return_value=mock_instance)
        mock_instance.set_position = MagicMock(return_value=mock_instance)
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_composite_video_clip():
    """Mock MoviePy CompositeVideoClip."""
    with patch("moviepy.editor.CompositeVideoClip") as mock:
        mock_instance = MagicMock()
        mock_instance.set_duration = MagicMock(return_value=mock_instance)
        mock_instance.set_audio = MagicMock(return_value=mock_instance)
        mock_instance.write_videofile = MagicMock()
        mock_instance.close = MagicMock()
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def config_dir(project_root):
    """Return the config directory path."""
    return project_root / "config"


@pytest.fixture
def output_dir(project_root):
    """Return the output directory path."""
    return project_root / "output"


@pytest.fixture
def mock_env_vars():
    """Set up mock environment variables for testing."""
    env_vars = {
        "AI_PROVIDER": "ollama",
        "PEXELS_API_KEY": "test_pexels_key",
        "PIXABAY_API_KEY": "test_pixabay_key",
    }
    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture
def mock_logger():
    """Mock loguru logger."""
    with patch("loguru.logger") as mock:
        yield mock
