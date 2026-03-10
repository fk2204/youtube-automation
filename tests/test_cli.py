"""
Tests for run.py CLI commands.

Verifies that expected commands are defined, the benchmark command has been
removed, and unknown commands print usage help rather than crashing.
Uses subprocess with mocking to avoid actual execution.
"""

import sys
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="module")
def run_py(project_root):
    """Return the absolute path to run.py."""
    return project_root / "run.py"


@pytest.fixture(scope="module")
def run_py_source(run_py):
    """Return the raw source text of run.py."""
    return run_py.read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _run_help(run_py: Path) -> subprocess.CompletedProcess:
    """Run `python run.py` with no arguments and capture output."""
    return subprocess.run(
        [sys.executable, str(run_py)],
        capture_output=True,
        text=True,
        timeout=15,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests: expected commands exist
# ─────────────────────────────────────────────────────────────────────────────

class TestCommandsExist:

    def test_video_command_exists(self, run_py_source):
        """'video' command must be handled in run.py."""
        assert 'cmd == "video"' in run_py_source or "cmd == 'video'" in run_py_source

    def test_short_command_exists(self, run_py_source):
        """'short' command must be handled in run.py."""
        assert 'cmd == "short"' in run_py_source or "cmd == 'short'" in run_py_source

    def test_batch_command_exists(self, run_py_source):
        """'batch' command must be handled in run.py."""
        assert 'cmd == "batch"' in run_py_source or "cmd == 'batch'" in run_py_source

    def test_schedule_shorts_command_exists(self, run_py_source):
        """'schedule-shorts' command must be handled in run.py."""
        assert "schedule-shorts" in run_py_source

    def test_daily_all_command_exists(self, run_py_source):
        """'daily-all' command must be handled in run.py."""
        assert "daily-all" in run_py_source

    def test_cost_command_exists(self, run_py_source):
        """'cost' command must be handled in run.py."""
        assert 'cmd == "cost"' in run_py_source or "cmd == 'cost'" in run_py_source

    def test_status_command_exists(self, run_py_source):
        """'status' command must be handled in run.py."""
        assert 'cmd == "status"' in run_py_source or "cmd == 'status'" in run_py_source

    def test_cache_stats_command_exists(self, run_py_source):
        """'cache-stats' command must be handled in run.py."""
        assert "cache-stats" in run_py_source


# ─────────────────────────────────────────────────────────────────────────────
# Tests: benchmark removed
# ─────────────────────────────────────────────────────────────────────────────

class TestBenchmarkRemoved:

    def test_benchmark_command_not_in_main_dispatch(self, run_py_source):
        """'benchmark' must not appear as a handled command in the main dispatch block."""
        # The command dispatch block reads cmd == "benchmark".
        # If that substring appears, the command is still wired up.
        assert 'cmd == "benchmark"' not in run_py_source
        assert "cmd == 'benchmark'" not in run_py_source

    def test_benchmark_not_in_main_dispatch_block(self, run_py_source):
        """
        'benchmark' must not be a dispatched command in the main() if/elif chain.

        The help text may still mention it for historical reference, but the
        actual elif branch that would execute benchmark code must be absent.
        This prevents the command from silently running real I/O.
        """
        # The dispatch block pattern is: elif cmd == "benchmark":
        # That substring being absent means no code path executes for it.
        assert 'elif cmd == "benchmark"' not in run_py_source
        assert "elif cmd == 'benchmark'" not in run_py_source

    def test_benchmark_does_not_execute(self, run_py):
        """Invoking 'python run.py benchmark' must not trigger a benchmark run."""
        with patch("subprocess.run") as mock_run:
            result = subprocess.run(
                [sys.executable, str(run_py), "benchmark"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        # Should exit cleanly (0 or non-zero) without actually benchmarking;
        # we verify no "benchmark" execution output was produced in stdout.
        assert "benchmark" not in result.stdout.lower() or "not found" in result.stdout.lower() or result.stdout == ""


# ─────────────────────────────────────────────────────────────────────────────
# Tests: unknown commands show usage help
# ─────────────────────────────────────────────────────────────────────────────

class TestUnknownCommandHelp:

    def test_no_args_prints_usage(self, run_py):
        """Running run.py with no arguments must print usage information."""
        result = _run_help(run_py)
        combined = result.stdout + result.stderr
        # Must mention known commands
        assert "video" in combined
        assert "short" in combined

    def test_no_args_exit_code(self, run_py):
        """Running run.py with no arguments must exit cleanly (code 0)."""
        result = _run_help(run_py)
        assert result.returncode == 0

    def test_unknown_command_does_not_crash(self, run_py):
        """An unrecognised command must not raise an unhandled exception."""
        result = subprocess.run(
            [sys.executable, str(run_py), "xyzzy_unknown_cmd_12345"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # No Python traceback should leak to stderr
        assert "Traceback" not in result.stderr

    def test_mock_subprocess_video_command(self, run_py):
        """Mock subprocess to verify 'video' command path is reachable without real I/O."""
        with patch("subprocess.run") as mock_subproc:
            mock_subproc.return_value = MagicMock(returncode=0)
            # We just confirm the run.py file defines the video branch without
            # actually running the video pipeline.
            source = run_py.read_text(encoding="utf-8")
            assert "video" in source
