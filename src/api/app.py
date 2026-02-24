"""
Content Empire API — FastAPI application.

This is the REST API service layer that Open Claw Bot uses to control
the content creation and distribution pipeline.

Launch:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000

Or programmatically:
    from src.api.app import app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

import asyncio
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.auth import verify_api_key, get_api_key
from src.api.job_manager import get_job_manager
from src.api.models import (
    ChannelInfo,
    ChannelListResponse,
    HealthResponse,
    PlatformName,
    TriggerDailyRequest,
    JobResponse,
)

# Import route modules
from src.api.routes import content, jobs, platforms, analytics, research

# --- Application lifecycle ---

_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    logger.info("Content Empire API starting up...")

    # Initialize job manager and register handlers
    mgr = get_job_manager()
    loop = asyncio.get_running_loop()
    mgr.set_event_loop(loop)

    # Register job handlers
    mgr.register_handler("create_content", content._handle_create_content)
    mgr.register_handler("distribute_content", content._handle_distribute_content)
    mgr.register_handler("create_and_distribute", content._handle_create_and_distribute)
    mgr.register_handler("research_topics", research._handle_research)
    mgr.register_handler("daily_automation", _handle_daily_automation)

    # Recover any incomplete jobs from previous run
    await mgr.start_recovery()

    api_key = get_api_key()
    logger.info(f"API ready. Key: {api_key[:8]}...")
    logger.info("Endpoints: http://0.0.0.0:8000/docs")

    yield

    logger.info("Content Empire API shutting down...")


# --- FastAPI app ---

app = FastAPI(
    title="Content Empire API",
    description="Multi-platform content creation and distribution service. "
                "Controlled by Open Claw Bot via REST API.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow Open Claw Bot from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Error handling ---

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": type(exc).__name__,
            "detail": str(exc),
            "path": str(request.url),
        },
    )


# --- Include route modules ---

app.include_router(content.router)
app.include_router(jobs.router)
app.include_router(platforms.router)
app.include_router(analytics.router)
app.include_router(research.router)


# --- Core endpoints ---

@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Service health check. No authentication required."""
    mgr = get_job_manager()
    stats = mgr.get_stats()

    # Check database
    db_ok = True
    try:
        from src.database.db import get_engine
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute("SELECT 1")
    except Exception:
        db_ok = False

    return HealthResponse(
        status="ok",
        version="1.0.0",
        uptime_seconds=round(time.time() - _start_time, 1),
        active_jobs=stats["running"],
        queued_jobs=stats["queued"],
        platforms_connected=0,  # Updated when platforms are checked
        database_ok=db_ok,
    )


@app.get("/api/v1/channels", response_model=ChannelListResponse)
async def list_channels(_: str = Depends(verify_api_key)):
    """List all configured YouTube channels."""
    channels = _load_channels()
    return ChannelListResponse(channels=channels)


@app.post("/api/v1/schedule/daily", response_model=JobResponse)
async def trigger_daily_automation(
    req: Optional[TriggerDailyRequest] = None,
    _: str = Depends(verify_api_key),
):
    """Trigger the full daily automation pipeline.

    Creates content for all (or specified) channels and distributes
    across configured platforms.
    """
    if req is None:
        req = TriggerDailyRequest()

    mgr = get_job_manager()
    job = mgr.submit(
        job_type="daily_automation",
        params={
            "channels": req.channels,
            "content_types": [ct.value for ct in req.content_types],
            "distribute": req.distribute,
        },
    )

    return JobResponse(
        job_id=job.id,
        status=job.status,
        estimated_duration_seconds=1800,  # ~30 min for full daily run
        created_at=job.created_at,
    )


# --- Helper functions ---

def _load_channels() -> List[ChannelInfo]:
    """Load channel definitions from config/channels.yaml."""
    channels_path = Path("config/channels.yaml")
    if not channels_path.exists():
        # Return default channels
        return [
            ChannelInfo(
                channel_id="money_blueprints",
                name="Money Blueprints",
                niche="finance",
                enabled=True,
                platforms=[PlatformName.YOUTUBE, PlatformName.TIKTOK],
                voice="en-US-GuyNeural",
                posting_days=["tuesday", "wednesday", "thursday"],
            ),
            ChannelInfo(
                channel_id="mind_unlocked",
                name="Mind Unlocked",
                niche="psychology",
                enabled=True,
                platforms=[PlatformName.YOUTUBE, PlatformName.TIKTOK],
                voice="en-US-JennyNeural",
                posting_days=["tuesday", "wednesday", "thursday"],
            ),
            ChannelInfo(
                channel_id="untold_stories",
                name="Untold Stories",
                niche="storytelling",
                enabled=True,
                platforms=[PlatformName.YOUTUBE, PlatformName.TIKTOK],
                voice="en-GB-RyanNeural",
                posting_days=["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
            ),
        ]

    try:
        with open(channels_path) as f:
            data = yaml.safe_load(f)

        channels = []
        for ch_id, ch_data in data.get("channels", {}).items():
            channels.append(ChannelInfo(
                channel_id=ch_id,
                name=ch_data.get("name", ch_id),
                niche=ch_data.get("niche", "general"),
                enabled=ch_data.get("enabled", True),
                platforms=[PlatformName.YOUTUBE, PlatformName.TIKTOK],
                voice=ch_data.get("voice", {}).get("name", "en-US-GuyNeural") if isinstance(ch_data.get("voice"), dict) else ch_data.get("voice"),
                posting_days=ch_data.get("posting_days", []),
            ))
        return channels
    except Exception as e:
        logger.error(f"Failed to load channels config: {e}")
        return []


async def _handle_daily_automation(job) -> dict:
    """Job handler for daily automation."""
    params = job.params
    mgr = get_job_manager()

    try:
        # Try the new daily empire pipeline (Phase 4)
        try:
            from src.automation.daily_empire import run_daily_empire
            result = await run_daily_empire(
                channels=params.get("channels"),
                content_types=params.get("content_types"),
                distribute=params.get("distribute", True),
            )
            return result
        except ImportError:
            pass

        # Fallback to existing unified launcher
        from src.automation.unified_launcher import UnifiedLauncher
        launcher = UnifiedLauncher()
        result = await launcher.launch_parallel_batch(
            channels=params.get("channels"),
            videos_per_channel=1,
            include_shorts=True,
        )

        return {
            "status": "success" if result.success else "partial",
            "videos_created": result.videos_created,
            "shorts_created": result.shorts_created,
            "errors": result.errors,
            "duration_seconds": result.duration_seconds,
        }

    except Exception as e:
        logger.error(f"Daily automation failed: {e}")
        return {"status": "failed", "errors": [str(e)]}


# --- CLI entry point ---

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("API_PORT", "8000"))
    host = os.environ.get("API_HOST", "0.0.0.0")
    uvicorn.run("src.api.app:app", host=host, port=port, reload=False, log_level="info")
