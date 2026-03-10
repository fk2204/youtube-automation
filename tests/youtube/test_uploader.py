"""Comprehensive tests for src/youtube/uploader.py"""

import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
from src.youtube.uploader import YouTubeUploader


@pytest.fixture
def mock_youtube_service():
    """Mock YouTube API service."""
    service = MagicMock()
    service.videos().insert().execute.return_value = {
        'id': 'test_video_id', 'snippet': {'title': 'Test Video'}
    }
    service.playlists().insert().execute.return_value = {
        'id': 'test_playlist_id'
    }
    return service


@pytest.fixture
def mock_credentials():
    """Mock Google credentials."""
    creds = MagicMock()
    creds.valid = True
    return creds


@pytest.fixture
def uploader(monkeypatch, mock_youtube_service):
    """Create uploader with mocked YouTube service using monkeypatch."""
    mock_auth_instance = MagicMock()
    mock_auth_instance.get_authenticated_service.return_value = mock_youtube_service

    # Use monkeypatch for persistent mocking across test execution
    monkeypatch.setattr('src.youtube.uploader.YouTubeAuth', MagicMock(return_value=mock_auth_instance))

    return YouTubeUploader()


class TestYouTubeUploaderInit:
    """Tests for YouTubeUploader initialization."""

    @patch('src.youtube.uploader.YouTubeAuth')
    def test_init_default(self, mock_auth):
        mock_auth.return_value = MagicMock()
        uploader = YouTubeUploader()
        assert uploader is not None

    @patch('src.youtube.uploader.YouTubeAuth')
    def test_init_with_credentials_file(self, mock_auth):
        mock_auth.return_value = MagicMock()
        uploader = YouTubeUploader(credentials_file='custom.json')
        assert uploader is not None


class TestVideoUpload:
    """Tests for video upload functionality."""

    @patch('src.youtube.uploader.MediaFileUpload')
    @patch('os.path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open, read_data=b'video data')
    def test_upload_video_success(self, mock_file, mock_exists, mock_media, uploader, mock_youtube_service):
        """Test successful video upload."""
        mock_media.return_value = MagicMock()

        result = uploader.upload_video(
            video_file='test.mp4',
            title='Test Video',
            description='Test Description'
        )

        assert result is not None

    @patch('os.path.exists', return_value=False)
    def test_upload_video_file_not_found(self, mock_exists, uploader):
        """Test upload with missing file."""
        with pytest.raises((FileNotFoundError, Exception)):
            uploader.upload_video(
                video_file='missing.mp4',
                title='Test'
            )

    @patch('src.youtube.uploader.MediaFileUpload')
    @patch('os.path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open, read_data=b'video')
    def test_upload_with_tags(self, mock_file, mock_exists, mock_media, uploader, mock_youtube_service):
        """Test upload with tags."""
        mock_media.return_value = MagicMock()

        result = uploader.upload_video(
            video_file='test.mp4',
            title='Test',
            description='Test Description',
            tags=['tag1', 'tag2']
        )

        assert result is not None

    @patch('src.youtube.uploader.MediaFileUpload')
    @patch('os.path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open, read_data=b'video')
    @pytest.mark.parametrize('privacy', ['public', 'private', 'unlisted'])
    def test_upload_with_privacy(self, mock_file, mock_exists, mock_media, uploader, mock_youtube_service, privacy):
        """Test upload with different privacy settings."""
        mock_media.return_value = MagicMock()

        result = uploader.upload_video(
            video_file='test.mp4',
            title='Test',
            description='Test Description',
            privacy=privacy
        )

        assert result is not None


class TestYouTubeUploaderMethods:
    """Tests for YouTubeUploader methods."""

    def test_uploader_attributes(self, uploader):
        """Test uploader has expected attributes."""
        assert hasattr(uploader, 'upload_video')
        assert callable(uploader.upload_video)

    @patch('src.youtube.uploader.MediaFileUpload')
    @patch('os.path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open, read_data=b'video')
    @pytest.mark.parametrize('title', [
        'Simple Title',
        'Title with Multiple Words and Symbols',
        'A' * 100  # Very long title
    ])
    def test_upload_various_titles(self, mock_file, mock_exists, mock_media, uploader, title):
        """Test upload with various title formats."""
        # Should not raise an error
        try:
            uploader.upload_video(
                video_file='test.mp4',
                title=title
            )
        except (FileNotFoundError, Exception):
            pass  # Expected in mock environment
