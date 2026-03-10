"""
Core module tests for the Joe content automation pipeline. (20 tests)

Verifies that:
- main.py loads without importing deleted video_pro and related modules
- src/ package __init__.py files do not reference deleted modules
- Database models can be imported cleanly
- Agent classes can be imported cleanly
- YouTube upload module has the expected public interface
"""

import sys
import importlib
import types
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure project root is on sys.path (conftest.py already does this, but
# repeated here so the file is self-contained when run in isolation).
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DELETED_MODULES = [
    "video_pro",
    "video_ultra",
    "pro_video_engine",
    "ai_video_runway",
    "ai_video_hailuo",
]


def _source_text(relative_path: str) -> str:
    """Read a source file and return its text content."""
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Group 1: main.py does not import deleted modules (4 tests)
# ---------------------------------------------------------------------------

class TestMainPyDoesNotImportDeleted:
    """main.py must not import any of the five deleted long-form modules."""

    def test_main_py_exists(self):
        assert (PROJECT_ROOT / "src" / "main.py").exists(), "src/main.py is missing"

    def test_main_py_no_video_pro_import(self):
        source = _source_text("src/main.py")
        assert "video_pro" not in source, "main.py still imports video_pro"

    def test_main_py_no_video_ultra_import(self):
        source = _source_text("src/main.py")
        assert "video_ultra" not in source, "main.py still imports video_ultra"

    def test_main_py_no_pro_video_engine_import(self):
        source = _source_text("src/main.py")
        assert "pro_video_engine" not in source, "main.py still imports pro_video_engine"


# ---------------------------------------------------------------------------
# Group 2: src/__init__.py files clean of deleted modules (4 tests)
# ---------------------------------------------------------------------------

class TestSrcInitFilesClean:
    """Package __init__.py files must not reference deleted modules."""

    def test_src_content_init_no_video_pro(self):
        source = _source_text("src/content/__init__.py")
        assert "video_pro" not in source

    def test_src_content_init_no_video_ultra(self):
        source = _source_text("src/content/__init__.py")
        assert "video_ultra" not in source

    def test_src_content_init_no_pro_video_engine(self):
        source = _source_text("src/content/__init__.py")
        assert "pro_video_engine" not in source

    def test_src_content_init_no_ai_video_runway(self):
        source = _source_text("src/content/__init__.py")
        assert "ai_video_runway" not in source


# ---------------------------------------------------------------------------
# Group 3: Database models import correctly (4 tests)
# ---------------------------------------------------------------------------

class TestDatabaseModels:
    """Database models must import without errors and expose correct classes."""

    def test_models_module_importable(self):
        from src.database import models
        assert models is not None

    def test_video_model_is_class(self):
        from src.database.models import Video
        assert isinstance(Video, type)

    def test_upload_model_is_class(self):
        from src.database.models import Upload
        assert isinstance(Upload, type)

    def test_generation_enums_present(self):
        from src.database.models import GenerationStep, GenerationStatus
        assert GenerationStep.RESEARCH == "research"
        assert GenerationStatus.PENDING == "pending"


# ---------------------------------------------------------------------------
# Group 4: Agents can be imported (4 tests)
# ---------------------------------------------------------------------------

class TestAgentsImport:
    """Core agent classes must be importable without heavy external deps."""

    def test_base_agent_importable(self):
        from src.agents.base_agent import BaseAgent
        assert BaseAgent is not None

    def test_research_agent_importable(self):
        from src.agents.research_agent import ResearchAgent
        assert ResearchAgent is not None

    def test_quality_agent_importable(self):
        from src.agents.quality_agent import QualityAgent
        assert QualityAgent is not None

    def test_agents_package_version_set(self):
        from src.agents import __version__
        assert isinstance(__version__, str)
        assert len(__version__) > 0


# ---------------------------------------------------------------------------
# Group 5: YouTube upload module structure (4 tests)
# ---------------------------------------------------------------------------

class TestYouTubeUploaderStructure:
    """YouTube uploader must expose the expected public interface."""

    def test_uploader_module_has_niche_hashtags(self):
        from src.youtube.uploader import NICHE_HASHTAGS
        assert isinstance(NICHE_HASHTAGS, dict)
        assert "finance" in NICHE_HASHTAGS

    def test_uploader_module_has_niche_categories(self):
        from src.youtube.uploader import NICHE_CATEGORIES
        assert isinstance(NICHE_CATEGORIES, dict)

    def test_auth_module_has_scopes(self):
        from src.youtube.auth import YouTubeAuth
        assert hasattr(YouTubeAuth, "SCOPES")
        assert isinstance(YouTubeAuth.SCOPES, list)
        assert len(YouTubeAuth.SCOPES) > 0

    def test_auth_module_api_version(self):
        from src.youtube.auth import YouTubeAuth
        assert YouTubeAuth.API_VERSION == "v3"
