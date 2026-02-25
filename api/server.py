"""
YouTube Automation FastAPI Server
Exposes youtube-automation as HTTP API for integration with Dubravko backend
"""

import os
import asyncio
import json
from datetime import datetime
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import yaml

from api.models import (
    CreateVideoRequest, CreateShortRequest, BatchRequest,
    GenerateIdeasRequest, KeywordResearchRequest, RedditResearchRequest,
    OptimalTimeRequest, CalendarRequest, UpcomingScheduleRequest,
    SEOStrategyRequest, GenerateTitlesRequest,
    JobResponse, JobStatusResponse, HealthResponse, ChannelsResponse, DashboardResponse
)
from api.job_tracker import get_job_tracker
from src.automation.runner import VideoAutomationRunner
from src.analytics.success_tracker import get_success_tracker
from src.research.idea_generator import IdeaGenerator
from src.seo.keyword_intelligence import KeywordIntelligence
from src.scheduler.smart_scheduler import SmartScheduler


# ═══════════════════════════════════════════════════════════════════════════════
# APP INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="YouTube Automation API",
    description="Control youtube-automation via HTTP API",
    version="2.1.0"
)

job_tracker = get_job_tracker()
success_tracker = get_success_tracker()

# Load configuration
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_FILE = PROJECT_ROOT / "config" / "channels.yaml"

with open(CONFIG_FILE, 'r') as f:
    CONFIG = yaml.safe_load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# AUTH
# ═══════════════════════════════════════════════════════════════════════════════

def verify_api_key(x_api_key: str = Header(None)) -> str:
    """Verify X-API-Key header"""
    expected_key = os.getenv("YOUTUBE_API_KEY", "")

    if not x_api_key:
        raise HTTPException(status_code=401, detail="X-API-Key header required")

    if not expected_key:
        # In development, skip auth
        return x_api_key

    if x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return x_api_key


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH & STATUS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse)
async def health_check(api_key: str = Header(None)):
    """Health check endpoint"""
    channels = [ch['id'] for ch in CONFIG['channels']]

    return HealthResponse(
        status="ok",
        version="2.1.0",
        channels=channels,
        uptime_seconds=0  # TODO: track uptime
    )


@app.get("/status")
async def get_status(api_key: str = Header(None)):
    """Get system status"""
    channels = [ch['id'] for ch in CONFIG['channels']]
    stats = job_tracker.get_stats()

    return {
        "success": True,
        "data": {
            "channels": channels,
            "jobs_pending": stats["by_status"].get("pending", 0),
            "jobs_running": stats["by_status"].get("running", 0),
            "jobs_completed": stats["by_status"].get("completed", 0),
            "jobs_failed": stats["by_status"].get("failed", 0),
            "total_jobs": stats["total_jobs"]
        }
    }


@app.get("/channels", response_model=ChannelsResponse)
async def list_channels(api_key: str = Header(None)):
    """List configured YouTube channels"""
    channels = []

    for ch_config in CONFIG['channels']:
        channels.append({
            "id": ch_config['id'],
            "name": ch_config['name'],
            "niche": ch_config['settings']['niche'],
            "enabled": ch_config['enabled'],
            "subscribers": None  # Could fetch from YouTube API
        })

    return ChannelsResponse(channels=channels)


# ═══════════════════════════════════════════════════════════════════════════════
# VIDEO CREATION (ASYNC)
# ═══════════════════════════════════════════════════════════════════════════════

async def _run_video_job(job_id: str, channel_id: str, topic: Optional[str], no_upload: bool):
    """Background task for video creation"""
    job_tracker.update_job(job_id, "running", progress=0)

    try:
        runner = VideoAutomationRunner()

        # Build command args
        args = ["--channel", channel_id]
        if topic:
            args.extend(["--topic", topic])
        if no_upload:
            args.append("--no-upload")

        # Run video creation
        result = await runner.run_video(channel_id, topic, no_upload=no_upload)

        # Mark as completed
        job_tracker.update_job(
            job_id, "completed", progress=100,
            result={
                "output_file": str(result.get("output_file", "")),
                "duration": result.get("duration", 0),
                "size_mb": result.get("size_mb", 0),
                "video_url": result.get("video_url", "")
            }
        )

        # Notify backend via webhook
        await _notify_backend(job_id)

    except Exception as e:
        job_tracker.update_job(job_id, "failed", error=str(e))
        print(f"Error in video job {job_id}: {str(e)}")


@app.post("/video/create", response_model=JobResponse)
async def create_video(req: CreateVideoRequest, bg_tasks: BackgroundTasks, api_key: str = Header(None)):
    """Create and upload a YouTube video (async)"""
    try:
        verify_api_key(api_key)

        # Create job
        job_id = job_tracker.create_job("video", channel_id=req.channel_id, topic=req.topic)

        # Queue background task
        bg_tasks.add_task(_run_video_job, job_id, req.channel_id, req.topic, req.no_upload)

        return JobResponse(
            job_id=job_id,
            status="pending",
            message=f"Video creation started for {req.channel_id}. Check status with /jobs/{job_id}"
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e), "code": "VIDEO_CREATE_ERROR"}
        )


@app.post("/video/short", response_model=JobResponse)
async def create_short(req: CreateShortRequest, bg_tasks: BackgroundTasks, api_key: str = Header(None)):
    """Create and upload a YouTube Short (async)"""
    try:
        verify_api_key(api_key)

        job_id = job_tracker.create_job("short", channel_id=req.channel_id, topic=req.topic)
        bg_tasks.add_task(_run_video_job, job_id, req.channel_id, req.topic, req.no_upload)

        return JobResponse(
            job_id=job_id,
            status="pending",
            message=f"Short creation started for {req.channel_id}"
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e), "code": "SHORT_CREATE_ERROR"}
        )


@app.post("/video/batch", response_model=JobResponse)
async def batch_videos(req: BatchRequest, bg_tasks: BackgroundTasks, api_key: str = Header(None)):
    """Create multiple videos across channels (async)"""
    try:
        verify_api_key(api_key)

        job_id = job_tracker.create_job("batch", topic=f"{len(req.channels)} channels, {req.count} per channel")

        async def _batch_job():
            job_tracker.update_job(job_id, "running", progress=0)
            try:
                runner = VideoAutomationRunner()
                results = await runner.run_batch(req.channels, req.count, parallel=req.parallel)
                job_tracker.update_job(job_id, "completed", progress=100, result={"videos_created": results})
            except Exception as e:
                job_tracker.update_job(job_id, "failed", error=str(e))

        bg_tasks.add_task(_batch_job)

        return JobResponse(
            job_id=job_id,
            status="pending",
            message=f"Batch job started: {len(req.channels)} channels, {req.count} videos each"
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e), "code": "BATCH_ERROR"}
        )


# ═══════════════════════════════════════════════════════════════════════════════
# JOB TRACKING
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str, api_key: str = Header(None)):
    """Get status of an async job"""
    verify_api_key(api_key)

    job = job_tracker.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=job['id'],
        status=job['status'],
        job_type=job['job_type'],
        channel_id=job['channel_id'],
        progress=job['progress'],
        result=job['result'],
        error=job['error'],
        created_at=job['created_at'],
        updated_at=job['updated_at']
    )


@app.get("/jobs")
async def list_jobs(limit: int = 20, status: Optional[str] = None, api_key: str = Header(None)):
    """List recent jobs"""
    verify_api_key(api_key)

    jobs = job_tracker.list_jobs(limit=limit, status=status)

    return {
        "success": True,
        "data": {
            "total": len(jobs),
            "jobs": [
                {
                    "job_id": j['id'],
                    "status": j['status'],
                    "job_type": j['job_type'],
                    "channel_id": j['channel_id'],
                    "progress": j['progress'],
                    "created_at": j['created_at'],
                    "updated_at": j['updated_at']
                }
                for j in jobs
            ]
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RESEARCH
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/research/ideas")
async def generate_ideas(req: GenerateIdeasRequest, api_key: str = Header(None)):
    """Generate AI-scored video ideas"""
    try:
        verify_api_key(api_key)

        gen = IdeaGenerator()
        ideas = gen.generate_ideas(req.niche, req.count)

        return {
            "success": True,
            "data": {
                "niche": req.niche,
                "count": len(ideas),
                "ideas": [
                    {
                        "title": idea.title,
                        "description": idea.description,
                        "score": idea.score,
                        "hook": idea.hook
                    }
                    for idea in ideas
                ]
            }
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e), "code": "IDEA_GENERATION_ERROR"}
        )


@app.post("/research/keyword")
async def research_keyword(req: KeywordResearchRequest, api_key: str = Header(None)):
    """Full keyword research and analysis"""
    try:
        verify_api_key(api_key)

        ki = KeywordIntelligence()
        result = ki.full_analysis(req.keyword, niche=req.niche)

        return {
            "success": True,
            "data": {
                "keyword": req.keyword,
                "opportunity_score": result.opportunity_score,
                "competition_score": result.competition_score,
                "search_volume": result.search_volume,
                "trend": result.trend,
                "long_tail_variations": result.long_tail_variations[:5]
            }
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e), "code": "KEYWORD_RESEARCH_ERROR"}
        )


@app.post("/research/reddit")
async def reddit_research(req: RedditResearchRequest, api_key: str = Header(None)):
    """Research trending Reddit content"""
    try:
        verify_api_key(api_key)

        # This would call RedditResearcher from src/research/reddit.py
        # Placeholder for now
        return {
            "success": True,
            "data": {
                "niche": req.niche,
                "type": req.type,
                "topics": [
                    {"title": "Example topic", "subreddit": "AskReddit", "upvotes": 1000}
                ]
            }
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e), "code": "REDDIT_RESEARCH_ERROR"}
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/analytics/dashboard")
async def get_dashboard(api_key: str = Header(None)):
    """Get full dashboard"""
    try:
        verify_api_key(api_key)

        dashboard = success_tracker.get_dashboard_data()

        return {
            "success": True,
            "data": dashboard
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e), "code": "DASHBOARD_ERROR"}
        )


@app.post("/analytics/channel")
async def get_channel_analytics(req: dict, api_key: str = Header(None)):
    """Get channel-specific analytics"""
    try:
        verify_api_key(api_key)

        channel_id = req.get("channel_id")
        period = req.get("period", "30d")

        # Would fetch from YouTube Analytics API
        return {
            "success": True,
            "data": {
                "channel_id": channel_id,
                "period": period,
                "views": 0,
                "watch_time_hours": 0,
                "average_view_duration_percent": 0,
                "engagement_rate": 0
            }
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e), "code": "ANALYTICS_ERROR"}
        )


@app.get("/analytics/costs")
async def get_api_costs(api_key: str = Header(None)):
    """Get API cost breakdown"""
    try:
        verify_api_key(api_key)

        costs = success_tracker.get_token_efficiency()

        return {
            "success": True,
            "data": costs
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e), "code": "COST_ERROR"}
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEDULING
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/schedule/optimal-time")
async def get_optimal_time(req: OptimalTimeRequest, api_key: str = Header(None)):
    """Get optimal upload time for channel"""
    try:
        verify_api_key(api_key)

        scheduler = SmartScheduler()
        best_time = scheduler.get_optimal_time(req.channel_id)

        return {
            "success": True,
            "data": {
                "channel_id": req.channel_id,
                "optimal_time": best_time,
                "timezone": "UTC"
            }
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e), "code": "SCHEDULING_ERROR"}
        )


@app.post("/schedule/calendar")
async def plan_calendar(req: CalendarRequest, api_key: str = Header(None)):
    """Generate content calendar"""
    try:
        verify_api_key(api_key)

        scheduler = SmartScheduler()
        calendar = scheduler.plan_content_calendar(req.channel_id, weeks=req.weeks)

        return {
            "success": True,
            "data": {
                "channel_id": req.channel_id,
                "weeks": req.weeks,
                "calendar": calendar
            }
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e), "code": "CALENDAR_ERROR"}
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SEO
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/seo/strategy")
async def get_seo_strategy(req: SEOStrategyRequest, api_key: str = Header(None)):
    """Get SEO strategy for keyword"""
    try:
        verify_api_key(api_key)

        ki = KeywordIntelligence()
        strategy = ki.get_seo_strategy(req.keyword, niche=req.niche)

        return {
            "success": True,
            "data": strategy
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e), "code": "SEO_STRATEGY_ERROR"}
        )


@app.post("/seo/titles")
async def generate_titles(req: GenerateTitlesRequest, api_key: str = Header(None)):
    """Generate viral title variants"""
    try:
        verify_api_key(api_key)

        # Would call TitleGenerator
        return {
            "success": True,
            "data": {
                "topic": req.topic,
                "count": req.count,
                "titles": [
                    f"Title variant {i+1} for {req.topic}"
                    for i in range(req.count)
                ]
            }
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e), "code": "TITLE_GENERATION_ERROR"}
        )


# ═══════════════════════════════════════════════════════════════════════════════
# WEBHOOKS
# ═══════════════════════════════════════════════════════════════════════════════

async def _notify_backend(job_id: str):
    """Notify backend of job completion"""
    webhook_url = os.getenv("BACKEND_WEBHOOK_URL", "http://localhost:3001/webhooks/youtube/job-complete")
    api_key = os.getenv("YOUTUBE_API_KEY", "")

    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            await session.post(
                webhook_url,
                json={"job_id": job_id},
                headers={"X-Webhook-Secret": api_key},
                timeout=aiohttp.ClientTimeout(total=5)
            )
    except Exception as e:
        print(f"Webhook notification failed for {job_id}: {str(e)}")
        # Non-fatal error


# ═══════════════════════════════════════════════════════════════════════════════
# STARTUP/SHUTDOWN
# ═══════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("YouTube Automation API starting...")
    print(f"Configured channels: {[ch['id'] for ch in CONFIG['channels']]}")
    print(f"API Key: {'Set' if os.getenv('YOUTUBE_API_KEY') else 'Not set (development mode)'}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("YOUTUBE_API_PORT", 8002)),
        log_level="info"
    )
