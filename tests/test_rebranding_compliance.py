"""
Rebranding compliance tests.

Verifies that the project has been fully rebranded from "YouTube Automation"
to "Joe" and that configuration, CLI commands, and error messages reflect the
new brand.
"""

import sys
import subprocess
from pathlib import Path

import pytest
import yaml

# Project root so we can locate config and source files.
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
SRC_DIR = PROJECT_ROOT / "src"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _load_channels() -> list:
    channels_path = CONFIG_DIR / "channels.yaml"
    data = _read_yaml(channels_path)
    return data.get("channels", [])


# ---------------------------------------------------------------------------
# 1. Config files use joe.db, not youtube_automation.db
# ---------------------------------------------------------------------------

class TestDatabasePathCompliance:
    """Config files must reference joe.db, never youtube_automation.db."""

    def test_config_yaml_has_no_youtube_automation_db(self):
        config_path = CONFIG_DIR / "config.yaml"
        text = config_path.read_text(encoding="utf-8")
        assert "youtube_automation.db" not in text

    def test_channels_yaml_has_no_youtube_automation_db(self):
        channels_path = CONFIG_DIR / "channels.yaml"
        text = channels_path.read_text(encoding="utf-8")
        assert "youtube_automation.db" not in text

    def test_integrations_yaml_has_no_youtube_automation_db(self):
        integrations_path = CONFIG_DIR / "integrations.yaml"
        if integrations_path.exists():
            text = integrations_path.read_text(encoding="utf-8")
            assert "youtube_automation.db" not in text

    def test_database_module_uses_joe_db(self):
        db_path = SRC_DIR / "database" / "db.py"
        text = db_path.read_text(encoding="utf-8")
        assert "joe.db" in text
        assert "youtube_automation.db" not in text


# ---------------------------------------------------------------------------
# 2. No "YouTube Automation" strings in user-facing output files
# ---------------------------------------------------------------------------

class TestBrandNameCompliance:
    """No old brand name should survive in config files that affect output."""

    def test_config_yaml_app_name_is_not_youtube_automation(self):
        config_path = CONFIG_DIR / "config.yaml"
        data = _read_yaml(config_path)
        app_name = data.get("app", {}).get("name", "")
        assert "YouTube Automation" not in app_name

    def test_channels_yaml_channel_names_do_not_contain_old_brand(self):
        channels = _load_channels()
        for channel in channels:
            channel_name = channel.get("name", "")
            assert "YouTube Automation" not in channel_name, (
                f"Channel '{channel_name}' contains old brand name"
            )

    def test_config_yaml_no_youtube_automation_string(self):
        """config.yaml must not contain 'YouTube Automation' at all."""
        config_path = CONFIG_DIR / "config.yaml"
        text = config_path.read_text(encoding="utf-8")
        assert "YouTube Automation" not in text

    def test_channels_yaml_no_youtube_automation_string(self):
        channels_path = CONFIG_DIR / "channels.yaml"
        text = channels_path.read_text(encoding="utf-8")
        assert "YouTube Automation" not in text


# ---------------------------------------------------------------------------
# 3. Removed benchmark command does not crash CLI
# ---------------------------------------------------------------------------

class TestBenchmarkCommandRemoved:
    """
    The benchmark command was removed from the codebase.
    Invoking it must not crash with an unhandled exception (exit code 1 from
    unhandled exception is different from a graceful "unknown command" exit).
    """

    def test_benchmark_command_does_not_crash(self):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "run.py"), "benchmark"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        # Must not be an unhandled Python traceback.
        assert "Traceback (most recent call last)" not in result.stderr, (
            "benchmark command raised an unhandled exception"
        )

    def test_benchmark_not_in_help_text(self):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "run.py"), "--help"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        combined = result.stdout + result.stderr
        assert "benchmark" not in combined.lower(), (
            "'benchmark' should not appear in --help output"
        )

    def test_benchmark_not_in_bare_run_help(self):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "run.py")],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        combined = result.stdout + result.stderr
        assert "benchmark" not in combined.lower(), (
            "'benchmark' should not appear in bare run.py output"
        )


# ---------------------------------------------------------------------------
# 4. All channel configs load correctly
# ---------------------------------------------------------------------------

class TestChannelConfigsLoad:
    """channels.yaml must be parseable and all required fields present."""

    def test_channels_yaml_is_parseable(self):
        channels_path = CONFIG_DIR / "channels.yaml"
        data = _read_yaml(channels_path)
        assert isinstance(data, dict)

    def test_channels_list_is_non_empty(self):
        channels = _load_channels()
        assert len(channels) > 0, "channels.yaml must define at least one channel"

    def test_each_channel_has_required_fields(self):
        channels = _load_channels()
        required_fields = ["id", "name", "enabled", "credentials_file", "settings"]
        for channel in channels:
            for field in required_fields:
                assert field in channel, (
                    f"Channel '{channel.get('id', '?')}' is missing required field '{field}'"
                )

    def test_money_blueprints_channel_loads(self):
        channels = _load_channels()
        ids = [ch["id"] for ch in channels]
        assert "money_blueprints" in ids

    def test_mind_unlocked_channel_loads(self):
        channels = _load_channels()
        ids = [ch["id"] for ch in channels]
        assert "mind_unlocked" in ids

    def test_untold_stories_channel_loads(self):
        channels = _load_channels()
        ids = [ch["id"] for ch in channels]
        assert "untold_stories" in ids


# ---------------------------------------------------------------------------
# 5. No old brand names in error messages (source files)
# ---------------------------------------------------------------------------

class TestNoOldBrandInSourceErrors:
    """Python source files must not contain the old brand name in error strings."""

    def _py_files_under(self, directory: Path):
        return list(directory.rglob("*.py"))

    def test_no_youtube_automation_in_raise_statements(self):
        violations = []
        for py_file in self._py_files_under(SRC_DIR):
            text = py_file.read_text(encoding="utf-8", errors="replace")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if "raise" in line and "YouTube Automation" in line:
                    violations.append(f"{py_file.relative_to(PROJECT_ROOT)}:{lineno}")
        assert violations == [], (
            "Old brand in raise statements:\n" + "\n".join(violations)
        )

    def test_no_youtube_automation_db_in_source_files(self):
        violations = []
        for py_file in self._py_files_under(SRC_DIR):
            text = py_file.read_text(encoding="utf-8", errors="replace")
            if "youtube_automation.db" in text:
                violations.append(str(py_file.relative_to(PROJECT_ROOT)))
        assert violations == [], (
            "Files still reference youtube_automation.db:\n" + "\n".join(violations)
        )
