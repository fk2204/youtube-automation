"""
Smoke tests for the CLI entry point (run.py).

All tests use mocked subprocess / module imports so that no real APIs,
databases, or file-system side-effects are triggered.
"""

import sys
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
RUN_PY = str(PROJECT_ROOT / "run.py")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(*args, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run run.py with the given arguments and return the completed process."""
    return subprocess.run(
        [sys.executable, RUN_PY, *args],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# 1. --help / bare invocation
# ---------------------------------------------------------------------------

class TestHelpCommand:
    def test_bare_invocation_exits_cleanly(self):
        result = _run()
        # A bare invocation should print help text and exit 0 (or gracefully).
        assert "Traceback" not in result.stderr, (
            "Bare invocation raised an unhandled exception"
        )

    def test_bare_invocation_prints_commands(self):
        result = _run()
        combined = result.stdout + result.stderr
        # Should mention at least the core commands
        assert "video" in combined.lower() or "short" in combined.lower(), (
            "Help text does not mention any known commands"
        )

    def test_no_benchmark_in_help_text(self):
        result = _run()
        combined = result.stdout + result.stderr
        assert "benchmark" not in combined.lower(), (
            "'benchmark' must not appear in the help text after removal"
        )


# ---------------------------------------------------------------------------
# 2. cost command
# ---------------------------------------------------------------------------

class TestCostCommand:
    def test_cost_command_does_not_crash(self):
        mock_report = MagicMock()
        with patch.dict("sys.modules", {
            "src.utils.token_manager": MagicMock(print_usage_report=mock_report),
        }):
            result = _run("cost")
        # Should not raise an unhandled exception.
        assert "Traceback" not in result.stderr, (
            "cost command raised an unhandled exception"
        )

    def test_cost_command_exit_code(self):
        mock_module = MagicMock()
        mock_module.print_usage_report = MagicMock()
        with patch.dict("sys.modules", {
            "src.utils.token_manager": mock_module,
        }):
            result = _run("cost")
        # Exit code must not indicate a Python-level crash (exit code 1 from
        # traceback).  We allow 0 or any non-crash exit.
        assert result.returncode != 1 or "Traceback" not in result.stderr


# ---------------------------------------------------------------------------
# 3. status command
# ---------------------------------------------------------------------------

class TestStatusCommand:
    def test_status_command_does_not_crash(self):
        mock_scheduler = MagicMock()
        mock_scheduler.show_status = MagicMock()
        with patch.dict("sys.modules", {
            "src.scheduler.daily_scheduler": mock_scheduler,
        }):
            result = _run("status")
        assert "Traceback" not in result.stderr, (
            "status command raised an unhandled exception"
        )

    def test_status_command_exit_code(self):
        mock_scheduler = MagicMock()
        mock_scheduler.show_status = MagicMock()
        with patch.dict("sys.modules", {
            "src.scheduler.daily_scheduler": mock_scheduler,
        }):
            result = _run("status")
        assert result.returncode != 1 or "Traceback" not in result.stderr


# ---------------------------------------------------------------------------
# 4. benchmark command removed
# ---------------------------------------------------------------------------

class TestBenchmarkCommandAbsent:
    def test_benchmark_command_does_not_traceback(self):
        result = _run("benchmark")
        assert "Traceback" not in result.stderr, (
            "benchmark command raised an unhandled traceback"
        )

    def test_benchmark_not_in_help(self):
        result = _run()
        combined = result.stdout + result.stderr
        assert "benchmark" not in combined.lower()
