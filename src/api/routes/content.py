"""
Content creation and distribution endpoints.

These are the primary endpoints Open Claw Bot calls to create and distribute content.
All creation operations are async — they return a job_id immediately.
"""

from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from src.api.auth import verify_api_key
from src.api.job_manager import get_job_manager, Job
from src.api.models import (
    CreateContentRequest,
    DistributeContentRequest,
    CreateAndDistributeRequest,
    JobResponse,
    ContentType,
)

router = APIRouter(prefix="/api/v1/content", tags=["content"])

# Estimated durations by content type (seconds)
DURATION_ESTIMATES = {
    ContentType.VIDEO_LONG: 600,
    ContentType.VIDEO_SHORT: 180,
    ContentType.BLOG: 120,
    ContentType.IMAGE: 30,
    ContentType.CAROUSEL: 60,
    ContentType.TEXT_POST: 15,
}


async def _handle_create_content(job: Job) -> Dict[str, Any]:
    """Job handler for content creation."""
    params = job.params
    content_type = params["content_type"]
    topic = params["topic"]
    niche = params["niche"]
    channel = params["channel"]
    options = params.get("options", {})
    mgr = get_job_manager()

    artifacts = []
    metrics = {"started_at": datetime.utcnow().isoformat()}

    try:
        if content_type in (ContentType.VIDEO_LONG.value, ContentType.VIDEO_SHORT.value):
            mgr.update_progress(job.id, 10.0)
            result = await _create_video(
                topic=topic,
                niche=niche,
                channel=channel,
                short=(content_type == ContentType.VIDEO_SHORT.value),
                options=options,
                job=job,
            )
            artifacts.append({
                "type": "video",
                "path": result.get("video_path", ""),
                "platform": "youtube",
                "url": result.get("youtube_url"),
            })

            # Create shorts if requested
            if options.get("include_shorts") and content_type == ContentType.VIDEO_LONG.value:
                mgr.update_progress(job.id, 80.0)
                short_result = await _create_video(
                    topic=topic, niche=niche, channel=channel,
                    short=True, options=options, job=job,
                )
                artifacts.append({
                    "type": "video_short",
                    "path": short_result.get("video_path", ""),
                    "platform": "youtube_shorts",
                })

        elif content_type == ContentType.BLOG.value:
            mgr.update_progress(job.id, 10.0)
            result = await _create_blog(topic=topic, niche=niche, channel=channel, options=options)
            artifacts.append({
                "type": "blog",
                "path": result.get("blog_path", ""),
                "content": result.get("content", ""),
            })

        elif content_type == ContentType.IMAGE.value:
            mgr.update_progress(job.id, 10.0)
            result = await _create_images(topic=topic, niche=niche, channel=channel, options=options)
            for img in result.get("images", []):
                artifacts.append({"type": "image", "path": img["path"], "style": img.get("style")})

        elif content_type == ContentType.TEXT_POST.value:
            mgr.update_progress(job.id, 10.0)
            result = await _create_text_posts(topic=topic, niche=niche, channel=channel, options=options)
            for post in result.get("posts", []):
                artifacts.append({"type": "text_post", "platform": post["platform"], "content": post["text"]})

        else:
            result = {"message": f"Content type {content_type} not yet implemented"}

        mgr.update_progress(job.id, 100.0)
        metrics["completed_at"] = datetime.utcnow().isoformat()
        return {"artifacts": artifacts, "metrics": metrics, **result}

    except Exception as e:
        logger.error(f"Content creation failed for job {job.id}: {e}")
        raise


async def _handle_distribute_content(job: Job) -> Dict[str, Any]:
    """Job handler for content distribution."""
    params = job.params
    mgr = get_job_manager()
    results = []

    try:
        mgr.update_progress(job.id, 10.0)

        # Import distributor (Phase 1)
        try:
            from src.distribution.distributor import ContentDistributor
            distributor = ContentDistributor()
            dist_result = await distributor.distribute(
                content_path=params["content_path"],
                content_type=params["content_type"],
                title=params["title"],
                description=params.get("description", ""),
                tags=params.get("tags", []),
                niche=params["niche"],
                channel=params["channel"],
                platforms=params.get("platforms"),
            )
            results = dist_result.get("results", [])
        except ImportError:
            logger.warning("ContentDistributor not yet available. Using legacy distribution.")
            results = [{"platform": "youtube", "status": "legacy_mode"}]

        mgr.update_progress(job.id, 100.0)
        return {"distribution_results": results, "artifacts": results}

    except Exception as e:
        logger.error(f"Distribution failed for job {job.id}: {e}")
        raise


async def _handle_create_and_distribute(job: Job) -> Dict[str, Any]:
    """Job handler for full pipeline: create then distribute."""
    params = job.params
    mgr = get_job_manager()

    # Step 1: Create content
    create_job = Job(
        job_type="create_content",
        params=params,
    )
    create_result = await _handle_create_content(create_job)

    mgr.update_progress(job.id, 60.0)

    # Step 2: Distribute to platforms
    artifacts = create_result.get("artifacts", [])
    dist_results = []

    for artifact in artifacts:
        if artifact.get("path") and artifact["type"] in ("video", "video_short"):
            try:
                from src.distribution.distributor import ContentDistributor
                distributor = ContentDistributor()
                dr = await distributor.distribute(
                    content_path=artifact["path"],
                    content_type=artifact["type"],
                    title=params.get("topic", ""),
                    niche=params["niche"],
                    channel=params["channel"],
                    platforms=params.get("platforms"),
                )
                dist_results.extend(dr.get("results", []))
            except ImportError:
                logger.info("ContentDistributor not yet available")
                dist_results.append({"platform": "youtube", "status": "uploaded", "url": artifact.get("url")})

    mgr.update_progress(job.id, 100.0)
    return {
        "creation": create_result,
        "distribution": dist_results,
        "artifacts": artifacts + dist_results,
        "metrics": create_result.get("metrics", {}),
    }


# --- Video creation helper ---

async def _create_video(
    topic: str, niche: str, channel: str, short: bool,
    options: Dict[str, Any], job: Job,
) -> Dict[str, Any]:
    """Create a video using the existing pipeline."""
    mgr = get_job_manager()

    try:
        from src.automation.unified_launcher import UnifiedLauncher
        launcher = UnifiedLauncher()
        video_type = "short" if short else "video"
        result = await launcher.launch_full_pipeline(channel, video_type)

        if result.success:
            return {
                "status": "success",
                "video_path": result.details.get("video_path", ""),
                "youtube_url": result.details.get("youtube_url", ""),
                "title": result.details.get("title", topic),
                **result.details,
            }
        else:
            return {"status": "failed", "errors": result.errors}

    except ImportError:
        logger.warning("UnifiedLauncher not available, using direct pipeline")
        return await _create_video_direct(topic, niche, channel, short, options, job)


async def _create_video_direct(
    topic: str, niche: str, channel: str, short: bool,
    options: Dict[str, Any], job: Job,
) -> Dict[str, Any]:
    """Direct video creation without UnifiedLauncher (fallback)."""
    mgr = get_job_manager()

    try:
        from src.content.script_writer import ScriptWriter
        mgr.update_progress(job.id, 20.0)

        writer = ScriptWriter(provider="groq")
        script = writer.generate_script(topic=topic, niche=niche)

        mgr.update_progress(job.id, 40.0)

        from src.content.tts import TextToSpeech
        tts = TextToSpeech()
        audio_file = f"output/audio/{channel}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.mp3"
        narration = writer.get_full_narration(script)
        await tts.generate(narration, audio_file)

        mgr.update_progress(job.id, 60.0)

        return {
            "status": "success",
            "title": script.title,
            "audio_path": audio_file,
            "script_sections": len(script.sections) if hasattr(script, "sections") else 0,
        }

    except Exception as e:
        return {"status": "failed", "errors": [str(e)]}


async def _create_blog(topic: str, niche: str, channel: str, options: Dict) -> Dict[str, Any]:
    """Create a blog article (Phase 2 implementation)."""
    try:
        from src.content.blog_engine import BlogEngine
        engine = BlogEngine()
        result = await engine.create_article(topic=topic, niche=niche, channel=channel)
        return result
    except ImportError:
        # Phase 2 not yet built — use ScriptWriter as fallback
        try:
            from src.content.script_writer import ScriptWriter
            writer = ScriptWriter(provider="groq")
            script = writer.generate_script(topic=topic, niche=niche)
            narration = writer.get_full_narration(script)
            return {
                "status": "success",
                "content": narration,
                "title": script.title,
                "blog_path": "",
                "note": "Generated from script (BlogEngine not yet available)",
            }
        except Exception as e:
            return {"status": "failed", "errors": [str(e)]}


async def _create_images(topic: str, niche: str, channel: str, options: Dict) -> Dict[str, Any]:
    """Create images (Phase 2 implementation)."""
    try:
        from src.content.image_engine import ImageEngine
        engine = ImageEngine()
        return await engine.create_all(topic=topic, niche=niche, channel=channel)
    except ImportError:
        return {"status": "pending", "images": [], "note": "ImageEngine not yet available (Phase 2)"}


async def _create_text_posts(topic: str, niche: str, channel: str, options: Dict) -> Dict[str, Any]:
    """Create social media text posts."""
    try:
        from src.content.script_writer import ScriptWriter
        writer = ScriptWriter(provider="groq")
        script = writer.generate_script(topic=topic, niche=niche)
        title = script.title if hasattr(script, "title") else topic

        posts = [
            {"platform": "twitter", "text": f"{title}\n\n#{''.join(niche.split())} #ContentEmpire"},
            {"platform": "reddit", "text": title},
            {"platform": "linkedin", "text": f"New insight: {title}"},
        ]
        return {"status": "success", "posts": posts}
    except Exception as e:
        return {"status": "failed", "posts": [], "errors": [str(e)]}


# --- Route handlers ---

@router.post("/create", response_model=JobResponse)
async def create_content(req: CreateContentRequest, _: str = Depends(verify_api_key)):
    """Create new content (video, blog, image, etc.).

    Returns a job_id immediately. Poll /jobs/{job_id} for status.
    Optionally provide callback_url for webhook notification on completion.
    """
    mgr = get_job_manager()
    job = mgr.submit(
        job_type="create_content",
        params={
            "content_type": req.content_type.value,
            "topic": req.topic,
            "niche": req.niche,
            "channel": req.channel,
            "platforms": [p.value for p in req.platforms] if req.platforms else None,
            "options": req.options.model_dump(),
        },
        priority=req.priority,
        callback_url=req.callback_url,
    )

    return JobResponse(
        job_id=job.id,
        status=job.status,
        estimated_duration_seconds=DURATION_ESTIMATES.get(req.content_type, 300),
        created_at=job.created_at,
    )


@router.post("/distribute", response_model=JobResponse)
async def distribute_content(req: DistributeContentRequest, _: str = Depends(verify_api_key)):
    """Distribute existing content to platforms.

    Takes a path to an already-created content file and uploads/posts
    to the specified platforms.
    """
    mgr = get_job_manager()
    job = mgr.submit(
        job_type="distribute_content",
        params={
            "content_path": req.content_path,
            "content_type": req.content_type.value,
            "title": req.title,
            "description": req.description,
            "tags": req.tags,
            "niche": req.niche,
            "channel": req.channel,
            "platforms": [p.value for p in req.platforms] if req.platforms else None,
        },
        callback_url=req.callback_url,
    )

    return JobResponse(
        job_id=job.id,
        status=job.status,
        estimated_duration_seconds=60,
        created_at=job.created_at,
    )


@router.post("/create-and-distribute", response_model=JobResponse)
async def create_and_distribute(req: CreateAndDistributeRequest, _: str = Depends(verify_api_key)):
    """Full pipeline: create content then distribute to all platforms.

    This is the primary endpoint for Open Claw Bot to trigger the complete
    content creation and distribution pipeline.
    """
    mgr = get_job_manager()
    job = mgr.submit(
        job_type="create_and_distribute",
        params={
            "content_type": req.content_type.value,
            "topic": req.topic,
            "niche": req.niche,
            "channel": req.channel,
            "platforms": [p.value for p in req.platforms] if req.platforms else None,
            "options": req.options.model_dump(),
        },
        priority=req.priority,
        callback_url=req.callback_url,
    )

    base_estimate = DURATION_ESTIMATES.get(req.content_type, 300)
    return JobResponse(
        job_id=job.id,
        status=job.status,
        estimated_duration_seconds=int(base_estimate * 1.5),  # Extra time for distribution
        created_at=job.created_at,
    )
