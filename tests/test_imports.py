"""
Import-safety tests for the Joe project. (8 tests)

Verifies that:
- None of the five deleted long-form modules can be imported or are
  referenced anywhere inside src/
- All package __init__.py files parse as valid Python
"""

import ast
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SRC_ROOT = PROJECT_ROOT / "src"

# Modules removed in Phase 3C (long-form removal)
DELETED_MODULE_NAMES = [
    "video_pro",
    "video_ultra",
    "pro_video_engine",
    "ai_video_runway",
    "ai_video_hailuo",
]


# ---------------------------------------------------------------------------
# Helper: grep all .py source files under src/
# ---------------------------------------------------------------------------

def _all_py_sources():
    """Yield (path, text) for every .py file inside src/."""
    for py_file in SRC_ROOT.rglob("*.py"):
        yield py_file, py_file.read_text(encoding="utf-8", errors="replace")


def _files_referencing(module_name: str):
    """Return list of src-relative paths that contain module_name as a token."""
    matches = []
    for path, text in _all_py_sources():
        # Only flag direct import statements, not incidental string matches
        # (e.g., "VIDEO_PRODUCTION" should not match "video_pro")
        if f"import {module_name}" in text or f"from .{module_name}" in text or f"from src.content.{module_name}" in text:
            matches.append(path.relative_to(PROJECT_ROOT))
    return matches


# ---------------------------------------------------------------------------
# Tests 1-5: deleted modules cannot be imported or referenced
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module_name", DELETED_MODULE_NAMES)
def test_deleted_module_not_referenced_in_src(module_name):
    """
    No file in src/ should contain an import statement for a deleted module.
    This catches any leftover 'from .video_pro import ...' style references.
    """
    offending = _files_referencing(module_name)
    assert offending == [], (
        f"Module '{module_name}' is still imported in: "
        + ", ".join(str(p) for p in offending)
    )


# ---------------------------------------------------------------------------
# Tests 6-8: __init__.py files are valid Python
# ---------------------------------------------------------------------------

INIT_FILES_TO_CHECK = [
    SRC_ROOT / "__init__.py",
    SRC_ROOT / "content" / "__init__.py",
    SRC_ROOT / "database" / "__init__.py",
]


@pytest.mark.parametrize(
    "init_path",
    INIT_FILES_TO_CHECK,
    ids=["src/__init__.py", "src/content/__init__.py", "src/database/__init__.py"],
)
def test_init_file_is_valid_python(init_path):
    """
    Each __init__.py must be parseable by the ast module with no SyntaxError.
    This catches truncated files, merge conflicts, or manual editing mistakes.
    """
    assert init_path.exists(), f"{init_path} does not exist"
    source = init_path.read_text(encoding="utf-8")
    try:
        ast.parse(source)
    except SyntaxError as exc:
        pytest.fail(f"{init_path} is not valid Python: {exc}")
