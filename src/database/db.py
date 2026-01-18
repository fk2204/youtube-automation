"""
Database Session Management and Helper Functions

Provides functions for database initialization, session management,
and common database operations for the YouTube automation pipeline.
"""

import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List
from contextlib import contextmanager

from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker, Session
from loguru import logger

from .models import (
    Base,
    Video,
    Upload,
    Generation,
    GenerationStep,
    GenerationStatus,
    UploadStatus,
)


# Database configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "youtube_automation.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

# Create engine and session factory
_engine = None
_SessionFactory = None


def _get_engine():
    """Get or create the database engine."""
    global _engine
    if _engine is None:
        # Ensure data directory exists
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(DATABASE_URL, echo=False)
        logger.debug(f"Database engine created: {DB_PATH}")
    return _engine


def _get_session_factory():
    """Get or create the session factory."""
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=_get_engine(), expire_on_commit=False)
    return _SessionFactory


def init_db() -> None:
    """
    Initialize the database by creating all tables.

    Creates the data directory if it doesn't exist and
    creates all tables defined in the models.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    engine = _get_engine()
    Base.metadata.create_all(engine)
    logger.info(f"Database initialized at: {DB_PATH}")


def get_session() -> Session:
    """
    Get a new database session.

    Returns:
        SQLAlchemy Session object

    Note:
        Caller is responsible for closing the session.
        Consider using get_session_context() for automatic cleanup.
    """
    SessionFactory = _get_session_factory()
    return SessionFactory()


@contextmanager
def get_session_context():
    """
    Context manager for database sessions with automatic cleanup.

    Usage:
        with get_session_context() as session:
            video = session.query(Video).first()

    Yields:
        SQLAlchemy Session object
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        session.close()


def log_video(
    title: str,
    topic: str,
    niche: Optional[str] = None,
    channel_id: Optional[str] = None,
    file_path: Optional[str] = None,
    duration: Optional[float] = None,
) -> Video:
    """
    Record a new video in the database.

    Args:
        title: Video title
        topic: Video topic/subject
        niche: Content niche (e.g., 'python programming')
        channel_id: Target YouTube channel ID
        file_path: Path to the generated video file
        duration: Video duration in seconds

    Returns:
        The created Video object (detached from session)
    """
    with get_session_context() as session:
        video = Video(
            title=title,
            topic=topic,
            niche=niche,
            channel_id=channel_id,
            file_path=file_path,
            duration=duration,
        )
        session.add(video)
        session.flush()  # Get the ID before commit
        logger.info(f"Logged video: {title} (ID: {video.id})")
        # Expunge to detach from session while keeping data
        session.expunge(video)
        return video


def log_upload(
    video_id: int,
    youtube_url: Optional[str] = None,
    youtube_id: Optional[str] = None,
    privacy: str = "unlisted",
    status: UploadStatus = UploadStatus.PENDING,
    error_msg: Optional[str] = None,
) -> Upload:
    """
    Record an upload attempt for a video.

    Args:
        video_id: ID of the video being uploaded
        youtube_url: Full YouTube URL after successful upload
        youtube_id: YouTube video ID
        privacy: Privacy setting (public, unlisted, private)
        status: Upload status
        error_msg: Error message if upload failed

    Returns:
        The created Upload object (detached from session)
    """
    with get_session_context() as session:
        uploaded_at = datetime.now(timezone.utc) if status == UploadStatus.COMPLETED else None

        upload = Upload(
            video_id=video_id,
            youtube_url=youtube_url,
            youtube_id=youtube_id,
            privacy=privacy,
            status=status,
            uploaded_at=uploaded_at,
            error_msg=error_msg,
        )
        session.add(upload)
        session.flush()
        logger.info(f"Logged upload for video {video_id} (Upload ID: {upload.id}, Status: {status})")
        session.expunge(upload)
        return upload


def update_upload_status(
    upload_id: int,
    status: UploadStatus,
    youtube_url: Optional[str] = None,
    youtube_id: Optional[str] = None,
    error_msg: Optional[str] = None,
) -> Optional[Upload]:
    """
    Update the status of an existing upload.

    Args:
        upload_id: ID of the upload to update
        status: New upload status
        youtube_url: YouTube URL (on success)
        youtube_id: YouTube video ID (on success)
        error_msg: Error message (on failure)

    Returns:
        Updated Upload object or None if not found
    """
    with get_session_context() as session:
        upload = session.query(Upload).filter(Upload.id == upload_id).first()
        if upload is None:
            logger.warning(f"Upload {upload_id} not found")
            return None

        upload.status = status
        if youtube_url:
            upload.youtube_url = youtube_url
        if youtube_id:
            upload.youtube_id = youtube_id
        if error_msg:
            upload.error_msg = error_msg
        if status == UploadStatus.COMPLETED:
            upload.uploaded_at = datetime.now(timezone.utc)

        logger.info(f"Updated upload {upload_id} status to {status}")
        session.expunge(upload)
        return upload


def log_generation_step(
    video_id: int,
    step: GenerationStep,
    status: GenerationStatus = GenerationStatus.PENDING,
    error_msg: Optional[str] = None,
) -> Generation:
    """
    Track a pipeline generation step for a video.

    Args:
        video_id: ID of the video being generated
        step: Pipeline step (research, script, audio, video, upload)
        status: Step status
        error_msg: Error message if step failed

    Returns:
        The created Generation object (detached from session)
    """
    with get_session_context() as session:
        started_at = datetime.now(timezone.utc) if status == GenerationStatus.IN_PROGRESS else None
        completed_at = datetime.now(timezone.utc) if status in (
            GenerationStatus.COMPLETED, GenerationStatus.FAILED
        ) else None

        generation = Generation(
            video_id=video_id,
            step=step,
            status=status,
            error_msg=error_msg,
            started_at=started_at,
            completed_at=completed_at,
        )
        session.add(generation)
        session.flush()
        logger.debug(f"Logged generation step: {step} for video {video_id} (Status: {status})")
        session.expunge(generation)
        return generation


def update_generation_step(
    generation_id: int,
    status: GenerationStatus,
    error_msg: Optional[str] = None,
) -> Optional[Generation]:
    """
    Update the status of an existing generation step.

    Args:
        generation_id: ID of the generation step to update
        status: New status
        error_msg: Error message (on failure)

    Returns:
        Updated Generation object or None if not found
    """
    with get_session_context() as session:
        generation = session.query(Generation).filter(
            Generation.id == generation_id
        ).first()

        if generation is None:
            logger.warning(f"Generation {generation_id} not found")
            return None

        generation.status = status
        if status == GenerationStatus.IN_PROGRESS and generation.started_at is None:
            generation.started_at = datetime.now(timezone.utc)
        if status in (GenerationStatus.COMPLETED, GenerationStatus.FAILED):
            generation.completed_at = datetime.now(timezone.utc)
        if error_msg:
            generation.error_msg = error_msg

        logger.debug(f"Updated generation {generation_id} status to {status}")
        session.expunge(generation)
        return generation


def get_recent_videos(limit: int = 10) -> List[Video]:
    """
    Query recent videos from the database.

    Args:
        limit: Maximum number of videos to return

    Returns:
        List of Video objects ordered by creation date (newest first)
    """
    with get_session_context() as session:
        videos = (
            session.query(Video)
            .order_by(desc(Video.created_at))
            .limit(limit)
            .all()
        )
        # Detach from session
        for video in videos:
            session.expunge(video)
        return videos


def get_failed_uploads() -> List[Upload]:
    """
    Get all uploads that have failed.

    Returns:
        List of Upload objects with failed status
    """
    with get_session_context() as session:
        uploads = (
            session.query(Upload)
            .filter(Upload.status == UploadStatus.FAILED)
            .order_by(desc(Upload.id))
            .all()
        )
        # Detach from session
        for upload in uploads:
            session.expunge(upload)
        return uploads


def get_pending_uploads() -> List[Upload]:
    """
    Get all uploads that are pending.

    Returns:
        List of Upload objects with pending status
    """
    with get_session_context() as session:
        uploads = (
            session.query(Upload)
            .filter(Upload.status == UploadStatus.PENDING)
            .order_by(Upload.id)
            .all()
        )
        # Detach from session
        for upload in uploads:
            session.expunge(upload)
        return uploads


def get_video_generations(video_id: int) -> List[Generation]:
    """
    Get all generation steps for a specific video.

    Args:
        video_id: ID of the video

    Returns:
        List of Generation objects for the video
    """
    with get_session_context() as session:
        generations = (
            session.query(Generation)
            .filter(Generation.video_id == video_id)
            .order_by(Generation.id)
            .all()
        )
        # Detach from session
        for generation in generations:
            session.expunge(generation)
        return generations


# Example usage and testing
if __name__ == "__main__":
    from loguru import logger
    import sys

    # Configure loguru for testing
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    print("\n" + "=" * 60)
    print("DATABASE TEST")
    print("=" * 60 + "\n")

    # Initialize database
    init_db()
    print(f"Database created at: {DB_PATH}\n")

    # Create a sample video
    video = log_video(
        title="How to Learn Python in 2024",
        topic="Python programming tutorial for beginners",
        niche="python programming",
        channel_id="UC123456789",
        file_path="output/python_tutorial.mp4",
        duration=600.5,
    )
    print(f"Created video: {video}")

    # Log generation steps
    for step in GenerationStep:
        gen = log_generation_step(
            video_id=video.id,
            step=step,
            status=GenerationStatus.COMPLETED,
        )
        print(f"Logged step: {gen}")

    # Log an upload
    upload = log_upload(
        video_id=video.id,
        youtube_url="https://youtube.com/watch?v=abc123",
        youtube_id="abc123",
        privacy="unlisted",
        status=UploadStatus.COMPLETED,
    )
    print(f"\nCreated upload: {upload}")

    # Query recent videos
    recent = get_recent_videos(5)
    print(f"\nRecent videos ({len(recent)}):")
    for v in recent:
        print(f"  - {v.title}")

    # Get failed uploads (should be empty)
    failed = get_failed_uploads()
    print(f"\nFailed uploads: {len(failed)}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60 + "\n")
