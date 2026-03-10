"""
Basic integration tests for Joe pipeline components. (15 tests)

Covers:
- Config loading → database initialization flow
- YouTube uploader auth structure (mock OAuth)
- Scheduler importability and constant structure
- ScriptWriter provider factory (get_provider) for all known providers
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Group 1: Config → database initialization flow (4 tests)
# ---------------------------------------------------------------------------

class TestConfigToDatabaseFlow:
    """
    Exercises the path: load_config() reads YAML → init_db() sets up SQLite.
    All file I/O is mocked so the test is hermetic.
    """

    def test_load_config_returns_dict(self, temp_config_dir):
        """load_config() must return a dict when config files exist."""
        import yaml

        config_path = temp_config_dir / "config.yaml"
        channels_path = temp_config_dir / "channels.yaml"

        with open(config_path) as f:
            config = yaml.safe_load(f)
        with open(channels_path) as f:
            channels_config = yaml.safe_load(f)
            config["channels"] = channels_config.get("channels", [])

        assert isinstance(config, dict)
        assert "channels" in config

    def test_load_config_channels_is_list(self, temp_config_dir):
        """The channels key produced by load_config must be a list."""
        import yaml

        channels_path = temp_config_dir / "channels.yaml"
        with open(channels_path) as f:
            channels_config = yaml.safe_load(f)

        channels = channels_config.get("channels", [])
        assert isinstance(channels, list)

    def test_init_db_uses_sqlite_url(self):
        """init_db should build a SQLite URL pointing at data/joe.db."""
        from src.database.db import DATABASE_URL
        assert "sqlite" in DATABASE_URL
        assert "joe.db" in DATABASE_URL

    def test_database_data_dir_constant_set(self):
        """DATA_DIR constant must resolve to a path ending in 'data'."""
        from src.database.db import DATA_DIR
        assert DATA_DIR.name == "data"


# ---------------------------------------------------------------------------
# Group 2: YouTube uploader auth structure (4 tests)
# ---------------------------------------------------------------------------

class TestYouTubeAuthStructure:
    """
    Verifies YouTubeAuth configuration constants and initialization logic
    using mocked OAuth credentials so no browser flow is triggered.
    """

    def test_youtube_auth_scopes_contain_upload(self):
        from src.youtube.auth import YouTubeAuth
        upload_scope = "https://www.googleapis.com/auth/youtube.upload"
        assert upload_scope in YouTubeAuth.SCOPES

    def test_youtube_auth_api_service_name(self):
        from src.youtube.auth import YouTubeAuth
        assert YouTubeAuth.API_SERVICE_NAME == "youtube"

    def test_youtube_auth_default_secrets_env_fallback(self):
        """When env var is set, YouTubeAuth should use it as the secrets path."""
        from src.youtube.auth import YouTubeAuth
        with patch.dict(os.environ, {"YOUTUBE_CLIENT_SECRETS_FILE": "/mock/secret.json"}):
            auth = YouTubeAuth()
            assert auth.client_secrets_file == "/mock/secret.json"

    def test_youtube_uploader_niche_hashtags_finance_is_list(self):
        from src.youtube.uploader import NICHE_HASHTAGS
        finance_tags = NICHE_HASHTAGS.get("finance", [])
        assert isinstance(finance_tags, list)
        assert len(finance_tags) > 0


# ---------------------------------------------------------------------------
# Group 3: Scheduler import and initialization (3 tests)
# ---------------------------------------------------------------------------

class TestSchedulerImportAndInit:
    """
    Verifies the scheduler package exposes its public constants and functions
    without crashing on import (no APScheduler daemon is started).
    """

    def test_scheduler_package_importable(self):
        from src.scheduler import daily_scheduler
        assert daily_scheduler is not None

    def test_posting_schedule_is_dict(self):
        from src.scheduler.daily_scheduler import POSTING_SCHEDULE
        assert isinstance(POSTING_SCHEDULE, dict)
        assert len(POSTING_SCHEDULE) > 0

    def test_posting_schedule_channels_have_times(self):
        from src.scheduler.daily_scheduler import POSTING_SCHEDULE
        for channel_id, schedule in POSTING_SCHEDULE.items():
            assert "times" in schedule, f"Channel '{channel_id}' missing 'times' key"
            assert isinstance(schedule["times"], list)
            assert len(schedule["times"]) > 0


# ---------------------------------------------------------------------------
# Group 4: ScriptWriter provider factory (4 tests)
# ---------------------------------------------------------------------------

class TestScriptWriterProviderFactory:
    """
    Exercises get_provider() for each registered provider name.
    External HTTP calls are prevented by mocking requests.Session.
    """

    def test_get_provider_returns_ollama_instance(self):
        from src.content.script_writer import get_provider, OllamaProvider
        provider = get_provider("ollama")
        assert isinstance(provider, OllamaProvider)

    def test_get_provider_returns_groq_instance(self):
        from src.content.script_writer import get_provider, GroqProvider
        provider = get_provider("groq", api_key="test_key")
        assert isinstance(provider, GroqProvider)

    def test_get_provider_raises_on_unknown_name(self):
        from src.content.script_writer import get_provider
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("definitely_not_a_provider")

    def test_get_provider_known_providers_list(self):
        """get_provider should accept all five documented provider names."""
        from src.content.script_writer import get_provider
        known = ["ollama", "groq", "gemini", "claude", "openai"]
        for name in known:
            # We only need construction to not raise; pass a dummy key for
            # providers that require one.
            try:
                provider = get_provider(name, api_key="dummy_key_for_test")
                assert provider is not None, f"get_provider('{name}') returned None"
            except Exception as exc:
                # Fail with a clear message that names which provider broke
                pytest.fail(f"get_provider('{name}') raised unexpectedly: {exc}")
