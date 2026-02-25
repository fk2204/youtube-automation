"""
Pydantic models for API request/response validation
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
# VIDEO CREATION REQUESTS
# ═══════════════════════════════════════════════════════════════════════════════

class CreateVideoRequest(BaseModel):
    """Create a long-form YouTube video"""
    channel_id: str = Field(..., description="Channel ID: money_blueprints, mind_unlocked, untold_stories")
    topic: Optional[str] = Field(None, description="Optional specific topic. If omitted, AI picks viral topic")
    no_upload: bool = Field(False, description="Create video but skip YouTube upload")


class CreateShortRequest(BaseModel):
    """Create a YouTube Short (vertical 9:16 video)"""
    channel_id: str = Field(..., description="Channel ID: money_blueprints, mind_unlocked, untold_stories")
    topic: Optional[str] = Field(None, description="Optional specific topic for the Short")
    no_upload: bool = Field(False, description="Create but skip YouTube upload")


class BatchRequest(BaseModel):
    """Create multiple videos across channels"""
    channels: List[str] = Field(..., description="List of channel IDs")
    count: int = Field(1, description="Number of videos per channel")
    parallel: bool = Field(False, description="Run in parallel (faster)")


# ═══════════════════════════════════════════════════════════════════════════════
# RESEARCH REQUESTS
# ═══════════════════════════════════════════════════════════════════════════════

class GenerateIdeasRequest(BaseModel):
    """Generate video ideas for a niche"""
    niche: str = Field(..., description="Niche: finance, psychology, storytelling")
    count: int = Field(5, description="Number of ideas to generate (max 20)")


class KeywordResearchRequest(BaseModel):
    """Research a keyword"""
    keyword: str = Field(..., description="Keyword to research")
    niche: str = Field("default", description="Niche context")


class RedditResearchRequest(BaseModel):
    """Research trending content on Reddit"""
    niche: str = Field(..., description="Niche to research")
    type: str = Field("trends", description="Type: trends, questions, viral")


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEDULING REQUESTS
# ═══════════════════════════════════════════════════════════════════════════════

class OptimalTimeRequest(BaseModel):
    """Get optimal upload time for a channel"""
    channel_id: str = Field(..., description="Channel ID")


class CalendarRequest(BaseModel):
    """Generate content calendar"""
    channel_id: str = Field(..., description="Channel ID")
    weeks: int = Field(4, description="Number of weeks to plan")


class UpcomingScheduleRequest(BaseModel):
    """Get upcoming scheduled content"""
    channel_id: str = Field(..., description="Channel ID")
    days: int = Field(7, description="Number of days to show")


# ═══════════════════════════════════════════════════════════════════════════════
# SEO REQUESTS
# ═══════════════════════════════════════════════════════════════════════════════

class SEOStrategyRequest(BaseModel):
    """Get SEO strategy for a keyword"""
    keyword: str = Field(..., description="Keyword to optimize for")
    niche: str = Field("default", description="Niche context")


class GenerateTitlesRequest(BaseModel):
    """Generate viral title variants"""
    topic: str = Field(..., description="Video topic")
    niche: str = Field("default", description="Niche context")
    count: int = Field(5, description="Number of title variants")


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSES
# ═══════════════════════════════════════════════════════════════════════════════

class JobResponse(BaseModel):
    """Response for async job creation"""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Current status: pending, running, completed, failed")
    message: str = Field(..., description="Human-readable status message")


class JobStatusResponse(BaseModel):
    """Response for job status query"""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Current status: pending, running, completed, failed")
    job_type: str = Field(..., description="Type of job: video, short, batch, scheduler")
    channel_id: Optional[str] = Field(None, description="Channel ID if applicable")
    progress: Optional[int] = Field(None, description="Progress percentage (0-100)")
    result: Optional[Dict[str, Any]] = Field(None, description="Job result if completed")
    error: Optional[str] = Field(None, description="Error message if failed")
    created_at: str = Field(..., description="ISO timestamp when job was created")
    updated_at: str = Field(..., description="ISO timestamp when job was last updated")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field("ok", description="System status")
    version: str = Field(..., description="API version")
    channels: List[str] = Field(..., description="Available channels")
    uptime_seconds: int = Field(..., description="Uptime in seconds")


class ChannelInfo(BaseModel):
    """Information about a YouTube channel"""
    id: str = Field(..., description="Channel ID")
    name: str = Field(..., description="Channel name")
    niche: str = Field(..., description="Content niche")
    enabled: bool = Field(..., description="Channel enabled for posting")
    subscribers: Optional[int] = Field(None, description="Subscriber count")


class ChannelsResponse(BaseModel):
    """List of channels"""
    channels: List[ChannelInfo] = Field(..., description="Available channels")


class AnalyticsData(BaseModel):
    """Channel analytics"""
    channel_id: str
    views: int
    watch_time_hours: float
    average_view_duration_percent: float
    engagement_rate: float
    likes: int
    comments: int
    shares: int
    subscribers_gained: int
    period: str


class DashboardResponse(BaseModel):
    """Full dashboard data"""
    total_videos: int
    total_shorts: int
    total_views: int
    average_ctr: float
    average_retention: float
    total_subscribers_gained: int
    token_usage: Dict[str, int]
    recent_jobs: List[JobStatusResponse]
    channel_stats: List[AnalyticsData]


class ErrorResponse(BaseModel):
    """Standard error response"""
    success: bool = False
    error: str = Field(..., description="Error message")
    code: str = Field(..., description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
