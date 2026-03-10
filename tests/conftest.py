"""
Pytest configuration and fixtures for Joe tests.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root and src to path for all imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ---------------------------------------------------------------------------
# General directory fixtures
# ---------------------------------------------------------------------------

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
def project_root():
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def config_dir(project_root):
    """Return the config directory path."""
    return project_root / "config"


@pytest.fixture
def output_dir(project_root):
    """Return the output directory path."""
    return project_root / "output"


# ---------------------------------------------------------------------------
# New fixtures required by test_core_modules / test_integration_basic
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_config_dir():
    """
    Temporary directory pre-populated with minimal config files.

    Allows tests to exercise config-loading logic without touching the real
    config/ directory and without requiring a network connection.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = Path(tmpdir) / "config"
        cfg.mkdir()

        (cfg / "config.yaml").write_text(
            "ai_provider: ollama\ntest_mode: true\n",
            encoding="utf-8",
        )
        (cfg / "channels.yaml").write_text(
            "channels:\n  - id: test_channel\n    name: Test Channel\n",
            encoding="utf-8",
        )
        (cfg / ".env").write_text("AI_PROVIDER=ollama\n", encoding="utf-8")

        yield cfg


@pytest.fixture
def temp_output_dir():
    """Temporary directory used as the video output location in tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "output"
        out.mkdir()
        yield out


@pytest.fixture
def mock_youtube_client():
    """
    Mock Google API YouTube client.

    The returned MagicMock mirrors the googleapiclient resource structure:
      client.videos().insert(...).execute()
      client.channels().list(...).execute()
    """
    mock_client = MagicMock()

    mock_insert = MagicMock()
    mock_insert.execute.return_value = {
        "id": "mock_video_id_123",
        "status": {"uploadStatus": "uploaded"},
        "snippet": {"title": "Mock Video", "channelId": "UC_mock_channel"},
    }
    mock_client.videos.return_value.insert.return_value = mock_insert

    mock_channel_list = MagicMock()
    mock_channel_list.execute.return_value = {
        "items": [{"id": "UC_mock_channel", "snippet": {"title": "Mock Channel"}}]
    }
    mock_client.channels.return_value.list.return_value = mock_channel_list

    yield mock_client


@pytest.fixture
def mock_database():
    """
    Mock SQLAlchemy session so database tests never touch disk.

    Patches the internal session factory used by src/database/db.py and
    yields the mock session for assertion in individual tests.
    """
    mock_session = MagicMock()
    mock_session.add = MagicMock()
    mock_session.commit = MagicMock()
    mock_session.rollback = MagicMock()
    mock_session.close = MagicMock()
    mock_session.query = MagicMock(return_value=MagicMock())

    mock_engine = MagicMock()

    with (
        patch("src.database.db._get_engine", return_value=mock_engine),
        patch("src.database.db._get_session_factory", return_value=lambda: mock_session),
    ):
        yield mock_session


@pytest.fixture
def mock_subprocess():
    """
    Mock subprocess.run and subprocess.Popen so FFmpeg calls never execute.

    Yields the patched subprocess.run mock so tests can assert call args.
    """
    mock_run = MagicMock()
    mock_run.return_value = MagicMock(returncode=0, stdout=b"", stderr=b"")

    mock_popen = MagicMock()
    mock_popen.return_value = MagicMock(
        returncode=0,
        communicate=MagicMock(return_value=(b"", b"")),
        wait=MagicMock(return_value=0),
    )

    with (
        patch("subprocess.run", mock_run),
        patch("subprocess.Popen", mock_popen),
    ):
        yield mock_run


# ---------------------------------------------------------------------------
# Environment / logging fixtures
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# MoviePy clip fixtures (kept from original conftest for existing tests)
# ---------------------------------------------------------------------------

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
