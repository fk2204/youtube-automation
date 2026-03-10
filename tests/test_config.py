"""
Tests for configuration files.

Verifies config.yaml, config.example.yaml, and channels.yaml load correctly
and contain the expected values for the Joe project.
"""

import pytest
import yaml
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="module")
def config_dir(project_root):
    """Return the config directory path."""
    return project_root / "config"


@pytest.fixture(scope="module")
def config_yaml(config_dir):
    """Load and return config.yaml as a dict."""
    with open(config_dir / "config.yaml", "r") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def config_example_yaml(config_dir):
    """Load and return config.example.yaml as a dict."""
    with open(config_dir / "config.example.yaml", "r") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def channels_yaml(config_dir):
    """Load and return channels.yaml as a dict."""
    with open(config_dir / "channels.yaml", "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# config.yaml tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigYaml:

    def test_config_loads_without_error(self, config_yaml):
        """config.yaml must parse into a non-None dict."""
        assert config_yaml is not None
        assert isinstance(config_yaml, dict)

    def test_database_path_uses_joe_db(self, config_yaml):
        """Database path must reference joe.db, not youtube_automation.db."""
        db_path = config_yaml["database"]["path"]
        assert "joe.db" in db_path, f"Expected joe.db but got: {db_path}"
        assert "youtube_automation.db" not in db_path

    def test_app_name_is_joe(self, config_yaml):
        """App name should be Joe, not the old project name."""
        app_name = config_yaml["app"]["name"]
        assert app_name == "Joe"

    def test_database_type_is_sqlite(self, config_yaml):
        """Database type should be sqlite."""
        assert config_yaml["database"]["type"] == "sqlite"

    def test_budget_section_exists_with_defaults(self, config_yaml):
        """Budget section must exist with daily_limit, warning_threshold, enforce."""
        budget = config_yaml.get("budget", {})
        assert "daily_limit" in budget
        assert "warning_threshold" in budget
        assert "enforce" in budget

    def test_research_default_values(self, config_yaml):
        """Research section must have ideas_per_run and min_score."""
        research = config_yaml.get("research", {})
        assert "ideas_per_run" in research
        assert "min_score" in research
        assert isinstance(research["ideas_per_run"], int)
        assert isinstance(research["min_score"], (int, float))

    def test_content_script_defaults(self, config_yaml):
        """Content script section must have min_length, max_length, target_length."""
        script = config_yaml.get("content", {}).get("script", {})
        assert "min_length" in script
        assert "max_length" in script
        assert "target_length" in script
        assert script["min_length"] < script["max_length"]

    def test_missing_key_returns_none_not_exception(self, config_yaml):
        """Accessing a missing top-level key should return None, not raise."""
        result = config_yaml.get("nonexistent_key_xyz")
        assert result is None

    def test_output_directories_defined(self, config_yaml):
        """Output section must define videos, thumbnails, audio, scripts."""
        output = config_yaml.get("output", {})
        assert "videos" in output
        assert "thumbnails" in output
        assert "audio" in output
        assert "scripts" in output


# ─────────────────────────────────────────────────────────────────────────────
# config.example.yaml tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigExampleYaml:

    def test_config_example_loads_without_error(self, config_example_yaml):
        """config.example.yaml must parse into a non-None dict."""
        assert config_example_yaml is not None
        assert isinstance(config_example_yaml, dict)

    def test_config_example_is_valid_yaml(self, config_dir):
        """config.example.yaml must be syntactically valid YAML."""
        with open(config_dir / "config.example.yaml", "r") as f:
            content = f.read()
        # yaml.safe_load raises yaml.YAMLError on invalid YAML
        result = yaml.safe_load(content)
        assert result is not None

    def test_config_example_database_uses_joe_db(self, config_example_yaml):
        """config.example.yaml database path should use joe.db."""
        db_path = config_example_yaml["database"]["path"]
        assert "joe.db" in db_path


# ─────────────────────────────────────────────────────────────────────────────
# channels.yaml tests
# ─────────────────────────────────────────────────────────────────────────────

class TestChannelsYaml:

    def test_channels_yaml_loads_without_error(self, channels_yaml):
        """channels.yaml must parse into a non-None dict."""
        assert channels_yaml is not None
        assert isinstance(channels_yaml, dict)

    def test_channels_key_exists_and_is_list(self, channels_yaml):
        """channels.yaml must have a top-level 'channels' list."""
        assert "channels" in channels_yaml
        assert isinstance(channels_yaml["channels"], list)

    def test_channels_list_is_not_empty(self, channels_yaml):
        """At least one channel must be defined."""
        assert len(channels_yaml["channels"]) > 0

    def test_each_channel_has_required_fields(self, channels_yaml):
        """Every channel entry must have id, name, enabled, settings."""
        for channel in channels_yaml["channels"]:
            assert "id" in channel, f"Channel missing 'id': {channel}"
            assert "name" in channel, f"Channel {channel.get('id')} missing 'name'"
            assert "enabled" in channel, f"Channel {channel.get('id')} missing 'enabled'"
            assert "settings" in channel, f"Channel {channel.get('id')} missing 'settings'"

    def test_channel_settings_have_niche(self, channels_yaml):
        """Every channel settings block must define a niche."""
        for channel in channels_yaml["channels"]:
            niche = channel["settings"].get("niche")
            assert niche is not None, f"Channel {channel.get('id')} missing niche"
            assert isinstance(niche, str)
