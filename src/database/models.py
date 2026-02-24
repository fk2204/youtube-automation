"""
SQLAlchemy Models for YouTube Automation Database

Defines the database schema for tracking videos, uploads, and generation pipeline.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Float,
    ForeignKey,
    Enum as SQLEnum,
    Index,
)
from sqlalchemy.orm import DeclarativeBase, relationship
import enum


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


class GenerationStep(str, enum.Enum):
    """Pipeline generation steps."""
    RESEARCH = "research"
    SCRIPT = "script"
    AUDIO = "audio"
    VIDEO = "video"
    UPLOAD = "upload"


class GenerationStatus(str, enum.Enum):
    """Status of a generation step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class UploadStatus(str, enum.Enum):
    """Status of a video upload."""
    PENDING = "pending"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Video(Base):
    """
    Represents a generated video in the automation pipeline.

    Tracks metadata about videos created by the system.
    """
    __tablename__ = "videos"
    __table_args__ = (
        Index('ix_videos_channel_niche', 'channel_id', 'niche'),  # Common query pattern
        Index('ix_videos_created_at', 'created_at'),  # Time-based queries
        Index('ix_videos_channel_created', 'channel_id', 'created_at'),  # Channel analytics
    )

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    title: str = Column(String(200), nullable=False)
    topic: str = Column(String(500), nullable=False)
    niche: Optional[str] = Column(String(100), nullable=True, index=True)  # Frequently filtered
    channel_id: Optional[str] = Column(String(100), nullable=True, index=True)  # Frequently filtered
    file_path: Optional[str] = Column(String(500), nullable=True)
    duration: Optional[float] = Column(Float, nullable=True)
    created_at: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    uploads = relationship("Upload", back_populates="video", cascade="all, delete-orphan")
    generations = relationship("Generation", back_populates="video", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Video(id={self.id}, title='{self.title[:30]}...')>"


class Upload(Base):
    """
    Represents an upload attempt to YouTube.

    Tracks upload status and YouTube metadata.
    """
    __tablename__ = "uploads"
    __table_args__ = (
        Index('ix_uploads_status_uploaded', 'status', 'uploaded_at'),  # Status monitoring
        Index('ix_uploads_youtube_id', 'youtube_id'),  # Lookup by YouTube ID
    )

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    video_id: int = Column(Integer, ForeignKey("videos.id"), nullable=False, index=True)  # FK lookup
    youtube_url: Optional[str] = Column(String(200), nullable=True)
    youtube_id: Optional[str] = Column(String(50), nullable=True)
    privacy: str = Column(String(20), default="unlisted", nullable=False)
    uploaded_at: Optional[datetime] = Column(DateTime, nullable=True)
    status: str = Column(
        SQLEnum(UploadStatus),
        default=UploadStatus.PENDING,
        nullable=False,
        index=True  # Frequently filtered by status
    )
    error_msg: Optional[str] = Column(Text, nullable=True)

    # Relationships
    video = relationship("Video", back_populates="uploads")

    def __repr__(self) -> str:
        return f"<Upload(id={self.id}, video_id={self.video_id}, status='{self.status}')>"


class Generation(Base):
    """
    Tracks individual steps in the video generation pipeline.

    Each video goes through multiple generation steps (research, script, audio, video, upload).
    """
    __tablename__ = "generations"
    __table_args__ = (
        Index('ix_generations_video_step', 'video_id', 'step'),  # Pipeline tracking
        Index('ix_generations_status_started', 'status', 'started_at'),  # Active jobs
    )

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    video_id: int = Column(Integer, ForeignKey("videos.id"), nullable=False, index=True)  # FK lookup
    step: str = Column(
        SQLEnum(GenerationStep),
        nullable=False,
        index=True  # Step-based queries
    )
    status: str = Column(
        SQLEnum(GenerationStatus),
        default=GenerationStatus.PENDING,
        nullable=False,
        index=True  # Status monitoring
    )
    error_msg: Optional[str] = Column(Text, nullable=True)
    started_at: Optional[datetime] = Column(DateTime, nullable=True)
    completed_at: Optional[datetime] = Column(DateTime, nullable=True)

    # Relationships
    video = relationship("Video", back_populates="generations")

    def __repr__(self) -> str:
        return f"<Generation(id={self.id}, step='{self.step}', status='{self.status}')>"


# ============================================================
# Cross-Platform Distribution Models
# ============================================================


class PlatformPost(Base):
    """
    Tracks content distributed to external platforms.

    Each row represents one piece of content posted to one platform
    (e.g., a TikTok upload, a Pinterest pin, a Medium article).
    """
    __tablename__ = "platform_posts"
    __table_args__ = (
        Index('ix_platform_posts_content', 'content_id'),
        Index('ix_platform_posts_platform', 'platform'),
        Index('ix_platform_posts_channel', 'channel'),
        Index('ix_platform_posts_posted', 'posted_at'),
    )

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    content_id: str = Column(String(100), nullable=False)
    platform: str = Column(String(50), nullable=False)
    post_id: Optional[str] = Column(String(200), nullable=True)
    url: Optional[str] = Column(String(500), nullable=True)
    title: Optional[str] = Column(String(500), nullable=True)
    niche: Optional[str] = Column(String(100), nullable=True)
    channel: Optional[str] = Column(String(100), nullable=True)
    content_type: Optional[str] = Column(String(50), nullable=True)
    posted_at: Optional[datetime] = Column(DateTime, nullable=True)
    created_at: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    metrics = relationship("PlatformMetric", back_populates="post", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<PlatformPost(id={self.id}, platform='{self.platform}', content_id='{self.content_id}')>"


class PlatformMetric(Base):
    """
    Performance metrics for a platform post, scraped periodically.

    Multiple rows per post as metrics are collected over time.
    """
    __tablename__ = "platform_metrics"
    __table_args__ = (
        Index('ix_platform_metrics_post', 'post_id'),
        Index('ix_platform_metrics_scraped', 'scraped_at'),
    )

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    post_id: int = Column(Integer, ForeignKey("platform_posts.id"), nullable=False, index=True)
    views: int = Column(Integer, default=0)
    likes: int = Column(Integer, default=0)
    comments: int = Column(Integer, default=0)
    shares: int = Column(Integer, default=0)
    saves: int = Column(Integer, default=0)
    impressions: int = Column(Integer, default=0)
    click_through_rate: float = Column(Float, default=0.0)
    avg_watch_time: float = Column(Float, default=0.0)
    retention_pct: float = Column(Float, default=0.0)
    engagement_rate: float = Column(Float, default=0.0)
    scraped_at: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    post = relationship("PlatformPost", back_populates="metrics")

    def __repr__(self) -> str:
        return f"<PlatformMetric(id={self.id}, post_id={self.post_id}, views={self.views})>"
