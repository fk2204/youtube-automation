"""
Pydantic request/response models for the Content Empire API.

All API communication uses these typed schemas for validation and serialization.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# --- Enums ---

class ContentType(str, Enum):
    """Supported content types for creation."""
    VIDEO_LONG = "video_long"
    VIDEO_SHORT = "video_short"
    BLOG = "blog"
    IMAGE = "image"
    CAROUSEL = "carousel"
    TEXT_POST = "text_post"


class JobStatus(str, Enum):
    """Async job lifecycle states."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(str, Enum):
    """Job execution priority."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class PlatformName(str, Enum):
    """All supported distribution platforms."""
    YOUTUBE = "youtube"
    YOUTUBE_SHORTS = "youtube_shorts"
    TIKTOK = "tiktok"
    INSTAGRAM_REELS = "instagram_reels"
    INSTAGRAM_POST = "instagram_post"
    PINTEREST = "pinterest"
    TWITTER = "twitter"
    REDDIT = "reddit"
    LINKEDIN = "linkedin"
    FACEBOOK = "facebook"
    DISCORD = "discord"
    MEDIUM = "medium"
    QUORA = "quora"
    THREADS = "threads"


class PlatformStatus(str, Enum):
    """Platform connection status."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    NOT_CONFIGURED = "not_configured"


# --- Request Models ---

class ContentOptions(BaseModel):
    """Optional parameters for content creation."""
    duration_minutes: Optional[int] = Field(None, ge=1, le=60, description="Target video duration")
    voice: Optional[str] = Field(None, description="TTS voice ID (e.g. en-US-GuyNeural)")
    include_shorts: bool = Field(False, description="Also create short-form clips")
    include_blog: bool = Field(False, description="Also create blog article")
    include_images: bool = Field(False, description="Also create images/pins")
    include_social_posts: bool = Field(False, description="Also create social media posts")
    style: Optional[str] = Field(None, description="Content style: educational, casual, professional")
    thumbnail_style: Optional[str] = Field(None, description="Thumbnail style override")


class CreateContentRequest(BaseModel):
    """Request to create new content."""
    content_type: ContentType
    topic: str = Field(..., min_length=3, max_length=500)
    niche: str = Field(..., min_length=2, max_length=100)
    channel: str = Field(..., min_length=2, max_length=100)
    platforms: Optional[List[PlatformName]] = Field(None, description="Target platforms (None = all enabled)")
    priority: JobPriority = JobPriority.NORMAL
    callback_url: Optional[str] = Field(None, description="Webhook URL for job completion notification")
    options: ContentOptions = Field(default_factory=ContentOptions)


class DistributeContentRequest(BaseModel):
    """Request to distribute existing content to platforms."""
    content_path: str = Field(..., description="Path to content file (video, image, etc.)")
    content_type: ContentType
    title: str = Field(..., min_length=3, max_length=200)
    description: Optional[str] = Field(None, max_length=5000)
    tags: List[str] = Field(default_factory=list)
    niche: str = Field(..., min_length=2, max_length=100)
    channel: str = Field(..., min_length=2, max_length=100)
    platforms: Optional[List[PlatformName]] = Field(None, description="Target platforms (None = all enabled)")
    callback_url: Optional[str] = None


class CreateAndDistributeRequest(BaseModel):
    """Full pipeline: create content then distribute everywhere."""
    content_type: ContentType
    topic: str = Field(..., min_length=3, max_length=500)
    niche: str = Field(..., min_length=2, max_length=100)
    channel: str = Field(..., min_length=2, max_length=100)
    platforms: Optional[List[PlatformName]] = None
    priority: JobPriority = JobPriority.NORMAL
    callback_url: Optional[str] = None
    options: ContentOptions = Field(default_factory=ContentOptions)


class ResearchTopicsRequest(BaseModel):
    """Request to research trending topics."""
    niche: str = Field(..., min_length=2, max_length=100)
    count: int = Field(5, ge=1, le=20)
    min_score: int = Field(60, ge=0, le=100)
    include_reddit: bool = True
    include_trends: bool = True


class TriggerDailyRequest(BaseModel):
    """Request to trigger daily automation."""
    channels: Optional[List[str]] = Field(None, description="Channels to run (None = all)")
    content_types: List[ContentType] = Field(
        default_factory=lambda: [ContentType.VIDEO_LONG, ContentType.VIDEO_SHORT]
    )
    distribute: bool = True


# --- Response Models ---

class JobResponse(BaseModel):
    """Response when an async job is created."""
    job_id: str
    status: JobStatus
    estimated_duration_seconds: Optional[int] = None
    created_at: datetime


class JobStatusResponse(BaseModel):
    """Detailed job status response."""
    job_id: str
    status: JobStatus
    job_type: str
    progress: float = Field(0.0, ge=0.0, le=100.0, description="Completion percentage")
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)


class JobResultResponse(BaseModel):
    """Full job result with output artifacts."""
    job_id: str
    status: JobStatus
    result: Dict[str, Any]
    artifacts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Output files: [{type, path, platform, url}]"
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Execution metrics: duration, tokens_used, cost"
    )


class PlatformStatusResponse(BaseModel):
    """Status of a single platform."""
    platform: PlatformName
    status: PlatformStatus
    enabled: bool
    upload_type: str = Field(description="api, browser, or webhook")
    last_upload_at: Optional[datetime] = None
    error: Optional[str] = None


class PlatformListResponse(BaseModel):
    """Status of all platforms."""
    platforms: List[PlatformStatusResponse]
    total_enabled: int
    total_connected: int


class ChannelInfo(BaseModel):
    """Channel configuration summary."""
    channel_id: str
    name: str
    niche: str
    enabled: bool
    platforms: List[PlatformName]
    voice: Optional[str] = None
    posting_days: List[str] = Field(default_factory=list)


class ChannelListResponse(BaseModel):
    """List of configured channels."""
    channels: List[ChannelInfo]


class TopicIdea(BaseModel):
    """A researched topic idea."""
    topic: str
    title_suggestions: List[str]
    score: float = Field(ge=0.0, le=100.0)
    source: str = Field(description="trends, reddit, ai, or combined")
    trend_data: Optional[Dict[str, Any]] = None


class ResearchResponse(BaseModel):
    """Research results."""
    job_id: str
    niche: str
    ideas: List[TopicIdea]
    total_found: int


class AnalyticsReport(BaseModel):
    """Cross-platform analytics report."""
    period: str = Field(description="e.g. 'last_7_days', 'last_30_days'")
    channels: Dict[str, Dict[str, Any]]
    platforms: Dict[str, Dict[str, Any]]
    top_performers: List[Dict[str, Any]]
    recommendations: List[str]


class HealthResponse(BaseModel):
    """Service health check response."""
    status: str = "ok"
    version: str
    uptime_seconds: float
    active_jobs: int
    queued_jobs: int
    platforms_connected: int
    database_ok: bool
