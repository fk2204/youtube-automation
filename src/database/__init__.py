"""
YouTube Automation Database Module

Provides SQLAlchemy models and database utilities for tracking
videos, uploads, and generation pipeline progress.

Usage:
    from src.database import init_db, log_video, log_upload

    # Initialize database
    init_db()

    # Log a new video
    video = log_video(
        title="My Tutorial",
        topic="Python basics",
        niche="programming"
    )

    # Log an upload attempt
    upload = log_upload(
        video_id=video.id,
        status=UploadStatus.COMPLETED,
        youtube_url="https://youtube.com/watch?v=abc123"
    )
"""

# Models
from .models import (
    Base,
    Video,
    Upload,
    Generation,
    GenerationStep,
    GenerationStatus,
    UploadStatus,
)

# Database functions
from .db import (
    init_db,
    get_session,
    get_session_context,
    log_video,
    log_upload,
    update_upload_status,
    log_generation_step,
    update_generation_step,
    get_recent_videos,
    get_failed_uploads,
    get_pending_uploads,
    get_video_generations,
    DB_PATH,
    DATA_DIR,
)

__all__ = [
    # Models
    "Base",
    "Video",
    "Upload",
    "Generation",
    "GenerationStep",
    "GenerationStatus",
    "UploadStatus",
    # Functions
    "init_db",
    "get_session",
    "get_session_context",
    "log_video",
    "log_upload",
    "update_upload_status",
    "log_generation_step",
    "update_generation_step",
    "get_recent_videos",
    "get_failed_uploads",
    "get_pending_uploads",
    "get_video_generations",
    # Constants
    "DB_PATH",
    "DATA_DIR",
]
