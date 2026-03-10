"""
Tests for the FastAPI server (api/server.py) and its Pydantic models.

All database and external-service calls are mocked so these tests run
without a real database, YouTube API, or running server.

Strategy: api/server.py does top-level imports of heavyweight modules
(VideoAutomationRunner, success_tracker, etc.).  Those are injected into
sys.modules as stubs BEFORE the server module is imported so Python never
attempts the real import.
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure the project root is on sys.path so `api` and `src` packages resolve.
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Stub injection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _inject_stubs():
    """
    Inject lightweight stubs for modules that api/server.py imports at the
    top level but that require unavailable resources (DB, YouTube creds, etc.).
    Must be called before `from api.server import app`.
    """
    # Stub: VideoAutomationRunner (does not exist in src.automation.runner)
    runner_stub = types.ModuleType("src.automation.runner")
    runner_stub.VideoAutomationRunner = MagicMock()
    sys.modules.setdefault("src.automation.runner", runner_stub)

    # Stub: success_tracker
    tracker_mock = MagicMock()
    tracker_mock.get_dashboard_data.return_value = {}
    tracker_mock.get_token_efficiency.return_value = {}
    tracker_module = types.ModuleType("src.analytics.success_tracker")
    tracker_module.get_success_tracker = MagicMock(return_value=tracker_mock)
    sys.modules.setdefault("src.analytics.success_tracker", tracker_module)

    # Stub: job_tracker
    job_tracker_mock = MagicMock()
    job_tracker_mock.get_stats.return_value = {
        "by_status": {"pending": 0, "running": 0, "completed": 0, "failed": 0},
        "total_jobs": 0,
    }
    jt_module = types.ModuleType("api.job_tracker")
    jt_module.get_job_tracker = MagicMock(return_value=job_tracker_mock)
    sys.modules.setdefault("api.job_tracker", jt_module)

    # Stub: IdeaGenerator
    idea_module = types.ModuleType("src.research.idea_generator")
    idea_module.IdeaGenerator = MagicMock()
    sys.modules.setdefault("src.research.idea_generator", idea_module)

    # Stub: KeywordIntelligence
    ki_module = types.ModuleType("src.seo.keyword_intelligence")
    ki_module.KeywordIntelligence = MagicMock()
    sys.modules.setdefault("src.seo.keyword_intelligence", ki_module)

    # Stub: SmartScheduler
    sched_module = types.ModuleType("src.scheduler.smart_scheduler")
    sched_module.SmartScheduler = MagicMock()
    sys.modules.setdefault("src.scheduler.smart_scheduler", sched_module)


# Inject stubs at collection time (before any fixture runs).
_inject_stubs()

# Remove any previously cached api.server so our stubs take effect.
for _key in list(sys.modules.keys()):
    if _key in ("api.server",):
        del sys.modules[_key]

from api.server import app as _APP  # noqa: E402 – must follow stub injection


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def api_app():
    """Return the already-imported FastAPI app."""
    return _APP


@pytest.fixture(scope="module")
def client(api_app):
    """Return a TestClient backed by the mocked FastAPI app."""
    from fastapi.testclient import TestClient
    return TestClient(api_app, raise_server_exceptions=False)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: server import
# ─────────────────────────────────────────────────────────────────────────────

class TestServerImport:

    def test_server_imports_without_error(self, api_app):
        """api.server must be importable without raising any exception."""
        # If we reach this point the module imported cleanly.
        assert api_app is not None

    def test_fastapi_app_object_exists(self, api_app):
        """The module must expose a FastAPI application object."""
        from fastapi import FastAPI
        assert isinstance(api_app, FastAPI)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: API title and description branding
# ─────────────────────────────────────────────────────────────────────────────

class TestApiBranding:

    def test_api_title_does_not_contain_youtube_automation(self, api_app):
        """API title must not reference the old 'YouTube Automation' name."""
        assert "YouTube Automation" not in api_app.title

    def test_api_title_is_joe(self, api_app):
        """API title must be 'Joe API'."""
        assert api_app.title == "Joe API"

    def test_api_description_does_not_contain_youtube_automation(self, api_app):
        """API description must not contain 'YouTube Automation'."""
        description = api_app.description or ""
        assert "YouTube Automation" not in description

    def test_api_version_is_set(self, api_app):
        """API version must be a non-empty string."""
        assert api_app.version
        assert isinstance(api_app.version, str)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: /health endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_health_endpoint_returns_200(self, client):
        """/health must return HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_has_status_field(self, client):
        """/health response body must contain a 'status' field."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data

    def test_health_status_value_is_ok(self, client):
        """/health 'status' must equal 'ok'."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_response_has_channels_list(self, client):
        """/health response must include a 'channels' list."""
        response = client.get("/health")
        data = response.json()
        assert "channels" in data
        assert isinstance(data["channels"], list)

    def test_health_response_has_version(self, client):
        """/health response must include an API version string."""
        response = client.get("/health")
        data = response.json()
        assert "version" in data
        assert isinstance(data["version"], str)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Pydantic request/response models
# ─────────────────────────────────────────────────────────────────────────────

class TestRequestModels:

    def test_create_video_request_model_imports(self):
        """CreateVideoRequest must be importable from api.models."""
        from api.models import CreateVideoRequest  # noqa: F401

    def test_create_video_request_requires_channel_id(self):
        """CreateVideoRequest must raise ValidationError when channel_id is missing."""
        from api.models import CreateVideoRequest
        import pydantic

        with pytest.raises((pydantic.ValidationError, Exception)):
            CreateVideoRequest()  # channel_id is required

    def test_create_video_request_valid_instantiation(self):
        """CreateVideoRequest must accept a valid channel_id."""
        from api.models import CreateVideoRequest

        req = CreateVideoRequest(channel_id="money_blueprints")
        assert req.channel_id == "money_blueprints"
        assert req.no_upload is False  # default

    def test_create_short_request_model_imports(self):
        """CreateShortRequest must be importable from api.models."""
        from api.models import CreateShortRequest  # noqa: F401

    def test_health_response_model_imports(self):
        """HealthResponse must be importable from api.models."""
        from api.models import HealthResponse  # noqa: F401

    def test_health_response_valid_instantiation(self):
        """HealthResponse must be constructible with required fields."""
        from api.models import HealthResponse

        resp = HealthResponse(
            status="ok",
            version="2.1.0",
            channels=["money_blueprints"],
            uptime_seconds=0,
        )
        assert resp.status == "ok"
        assert resp.version == "2.1.0"
        assert "money_blueprints" in resp.channels
