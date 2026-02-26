"""Data models for Openclaw plugin commands and responses."""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import uuid


class JobStatus(Enum):
    """Job execution status."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PermissionLevel(Enum):
    """User permission levels."""
    ADMIN = "admin"
    MODERATOR = "moderator"
    USER = "user"
    PUBLIC = "public"


@dataclass
class CommandRequest:
    """Incoming command request from bot."""
    command: str  # e.g., "/video", "/batch", "/analytics"
    args: Dict[str, Any]
    user_id: str
    username: str
    platform: str  # "discord" or "telegram"
    channel_id: Optional[str] = None
    guild_id: Optional[str] = None  # For Discord
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


@dataclass
class Job:
    """Async job tracking."""
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    command: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    status: JobStatus = JobStatus.QUEUED
    user_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: int = 0  # 0-100
    logs: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "command": self.command,
            "args": self.args,
            "status": self.status.value,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress,
            "logs": self.logs,
            "results": self.results,
            "error": self.error,
        }

    def add_log(self, message: str) -> None:
        """Add log message."""
        self.logs.append(f"[{datetime.utcnow().isoformat()}] {message}")

    def set_progress(self, percent: int) -> None:
        """Set job progress (0-100)."""
        self.progress = min(100, max(0, percent))


@dataclass
class PluginResponse:
    """Response to send back to bot."""
    success: bool
    message: str
    job_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    embed_data: Optional[Dict[str, Any]] = None  # For Discord embeds

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps({
            "success": self.success,
            "message": self.message,
            "job_id": self.job_id,
            "data": self.data,
            "error": self.error,
        })

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "message": self.message,
            "job_id": self.job_id,
            "data": self.data,
            "error": self.error,
            "embed_data": self.embed_data,
        }


@dataclass
class VideoCommand:
    """Create single video command."""
    topic: str
    channel: Optional[str] = None
    niche: Optional[str] = None
    duration_minutes: int = 10
    upload: bool = True
    privacy: str = "unlisted"  # "public", "unlisted", "private"


@dataclass
class BatchCommand:
    """Batch video creation command."""
    count: int
    channel: Optional[str] = None
    topics: Optional[List[str]] = None  # Auto-generate if not provided
    spacing_days: int = 1


@dataclass
class AnalyticsCommand:
    """Analytics query command."""
    channel_id: Optional[str] = None
    video_id: Optional[str] = None
    period: str = "week"  # "day", "week", "month", "all"
    metric: Optional[str] = None  # "views", "engagement", "revenue", etc


@dataclass
class ScheduleCommand:
    """Schedule content creation."""
    topic: str
    channel: Optional[str] = None
    scheduled_time: datetime = field(default_factory=datetime.utcnow)
    recurring: Optional[str] = None  # "daily", "weekly", "monthly"


@dataclass
class MultiplatformCommand:
    """Export to multiple platforms."""
    video_id: str
    platforms: List[str] = field(default_factory=lambda: ["tiktok", "instagram", "shorts"])


@dataclass
class ConfigCommand:
    """Configuration change command."""
    setting: str
    value: Any
    scope: str = "user"  # "user", "channel", "global"
