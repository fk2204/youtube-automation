"""
Database compliance tests.

Verifies that the database layer uses joe.db, not the old youtube_automation.db
path, that the schema initialises correctly, and that all model classes can be
imported without errors.
"""

import sys
import importlib
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from sqlalchemy import create_engine, inspect as sa_inspect
from sqlalchemy.orm import sessionmaker

# Ensure the src package is on the path.
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_in_memory_engine():
    """Return a fresh in-memory SQLite engine."""
    return create_engine("sqlite:///:memory:", echo=False)


# ---------------------------------------------------------------------------
# 1. Database module uses joe.db path
# ---------------------------------------------------------------------------

class TestDatabasePath:
    def test_db_path_constant_uses_joe_db(self):
        from src.database.db import DB_PATH
        assert DB_PATH.name == "joe.db", (
            f"Expected joe.db, got {DB_PATH.name}"
        )

    def test_database_url_references_joe_db(self):
        from src.database.db import DATABASE_URL
        assert "joe.db" in DATABASE_URL, (
            f"DATABASE_URL does not reference joe.db: {DATABASE_URL}"
        )

    def test_database_url_does_not_reference_old_name(self):
        from src.database.db import DATABASE_URL
        assert "youtube_automation" not in DATABASE_URL, (
            f"DATABASE_URL still references old name: {DATABASE_URL}"
        )

    def test_db_module_source_has_no_youtube_automation_db(self):
        db_file = PROJECT_ROOT / "src" / "database" / "db.py"
        text = db_file.read_text(encoding="utf-8")
        assert "youtube_automation.db" not in text


# ---------------------------------------------------------------------------
# 2. Database initialisation creates correct schema (in-memory SQLite)
# ---------------------------------------------------------------------------

class TestDatabaseInitialisation:
    def _init_schema(self):
        """Create all tables in a fresh in-memory database."""
        from src.database.models import Base
        engine = _make_in_memory_engine()
        Base.metadata.create_all(engine)
        return engine

    def test_init_db_creates_videos_table(self):
        engine = self._init_schema()
        inspector = sa_inspect(engine)
        assert "videos" in inspector.get_table_names(), (
            "videos table was not created"
        )

    def test_init_db_creates_uploads_table(self):
        engine = self._init_schema()
        inspector = sa_inspect(engine)
        assert "uploads" in inspector.get_table_names(), (
            "uploads table was not created"
        )

    def test_init_db_creates_generations_table(self):
        engine = self._init_schema()
        inspector = sa_inspect(engine)
        assert "generations" in inspector.get_table_names(), (
            "generations table was not created"
        )

    def test_videos_table_has_expected_columns(self):
        engine = self._init_schema()
        inspector = sa_inspect(engine)
        columns = {col["name"] for col in inspector.get_columns("videos")}
        for expected in ("id", "title", "topic", "channel_id", "created_at"):
            assert expected in columns, (
                f"videos table is missing column '{expected}'"
            )

    def test_uploads_table_has_expected_columns(self):
        engine = self._init_schema()
        inspector = sa_inspect(engine)
        columns = {col["name"] for col in inspector.get_columns("uploads")}
        for expected in ("id", "video_id", "status", "youtube_url"):
            assert expected in columns, (
                f"uploads table is missing column '{expected}'"
            )

    def test_generations_table_has_expected_columns(self):
        engine = self._init_schema()
        inspector = sa_inspect(engine)
        columns = {col["name"] for col in inspector.get_columns("generations")}
        for expected in ("id", "video_id", "step", "status"):
            assert expected in columns, (
                f"generations table is missing column '{expected}'"
            )

    def test_init_db_is_idempotent(self):
        """Calling create_all twice must not raise."""
        from src.database.models import Base
        engine = _make_in_memory_engine()
        Base.metadata.create_all(engine)
        Base.metadata.create_all(engine)  # Should not raise.


# ---------------------------------------------------------------------------
# 3. Old youtube_automation.db references are gone
# ---------------------------------------------------------------------------

class TestNoOldDbReferences:
    def _py_files_under(self, directory: Path):
        return list(directory.rglob("*.py"))

    def test_no_youtube_automation_db_in_database_package(self):
        db_pkg = PROJECT_ROOT / "src" / "database"
        for py_file in self._py_files_under(db_pkg):
            text = py_file.read_text(encoding="utf-8", errors="replace")
            assert "youtube_automation.db" not in text, (
                f"{py_file.relative_to(PROJECT_ROOT)} still references youtube_automation.db"
            )

    def test_no_youtube_automation_db_in_src(self):
        violations = []
        for py_file in self._py_files_under(PROJECT_ROOT / "src"):
            text = py_file.read_text(encoding="utf-8", errors="replace")
            if "youtube_automation.db" in text:
                violations.append(str(py_file.relative_to(PROJECT_ROOT)))
        assert violations == [], (
            "Files still reference youtube_automation.db:\n" + "\n".join(violations)
        )


# ---------------------------------------------------------------------------
# 4. All database models import correctly
# ---------------------------------------------------------------------------

class TestModelImports:
    def test_base_imports(self):
        from src.database.models import Base
        assert Base is not None

    def test_video_model_imports(self):
        from src.database.models import Video
        assert Video.__tablename__ == "videos"

    def test_upload_model_imports(self):
        from src.database.models import Upload
        assert Upload.__tablename__ == "uploads"

    def test_generation_model_imports(self):
        from src.database.models import Generation
        assert Generation.__tablename__ == "generations"

    def test_generation_step_enum_imports(self):
        from src.database.models import GenerationStep
        assert "research" in [s.value for s in GenerationStep]

    def test_generation_status_enum_imports(self):
        from src.database.models import GenerationStatus
        values = [s.value for s in GenerationStatus]
        assert "pending" in values
        assert "completed" in values

    def test_upload_status_enum_imports(self):
        from src.database.models import UploadStatus
        values = [s.value for s in UploadStatus]
        assert "pending" in values
        assert "completed" in values

    def test_db_module_helper_functions_importable(self):
        from src.database.db import (
            init_db,
            get_session,
            get_session_context,
            log_video,
            log_upload,
        )
        assert callable(init_db)
        assert callable(get_session)
        assert callable(get_session_context)
        assert callable(log_video)
        assert callable(log_upload)
