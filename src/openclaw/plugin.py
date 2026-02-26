"""Main Openclaw plugin interface."""

import asyncio
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
from loguru import logger

from .models import (
    CommandRequest,
    PluginResponse,
    Job,
    JobStatus,
    PermissionLevel,
    VideoCommand,
    BatchCommand,
    AnalyticsCommand,
    ScheduleCommand,
    MultiplatformCommand,
)
from .command_registry import CommandRegistry


class OpenclawPlugin:
    """Main plugin class for Openclaw bot integration."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize plugin.

        Args:
            config_path: Path to configuration file (optional)
        """
        self.command_registry = CommandRegistry()
        self.job_queue: Dict[str, Job] = {}
        self.config = self._load_config(config_path)
        self.rate_limiter = {}  # user_id -> {command -> timestamp}
        logger.info("Openclaw plugin initialized")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file."""
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                return json.load(f)

        # Default configuration
        return {
            "rate_limits": {
                "/video": 300,  # 5 minutes between requests
                "/batch": 3600,  # 1 hour
                "/short": 60,  # 1 minute
            },
            "max_queue_size": 100,
            "job_timeout": 3600,  # 1 hour
            "permissions": {
                "admin": ["*"],
                "moderator": ["/video", "/short", "/batch", "/schedule", "/analytics"],
                "user": ["/video", "/short", "/analytics", "/cost", "/status"],
                "public": ["/help", "/status"],
            },
        }

    async def handle_command(
        self,
        command: str,
        args: Dict[str, Any],
        user_id: str = "anonymous",
        username: str = "Anonymous",
        platform: str = "discord",
        channel_id: Optional[str] = None,
        guild_id: Optional[str] = None,
    ) -> PluginResponse:
        """Main command handler.

        Args:
            command: Command string (e.g., "/video")
            args: Command arguments
            user_id: ID of user issuing command
            username: Username of user
            platform: Platform ("discord" or "telegram")
            channel_id: Channel/chat ID
            guild_id: Guild/group ID (Discord)

        Returns:
            PluginResponse with result
        """
        try:
            # Get user permission level
            user_permission = self.get_user_permission(user_id)

            # Validate command and arguments
            is_valid, error = self.command_registry.validate_command(
                command, args, user_permission
            )
            if not is_valid:
                return PluginResponse(success=False, message=error or "Invalid command")

            # Check rate limits
            is_allowed, wait_time = self.check_rate_limit(user_id, command)
            if not is_allowed:
                return PluginResponse(
                    success=False,
                    message=f"Rate limited. Please wait {wait_time:.0f} seconds.",
                )

            # Create command request
            request = CommandRequest(
                command=command,
                args=args,
                user_id=user_id,
                username=username,
                platform=platform,
                channel_id=channel_id,
                guild_id=guild_id,
            )

            # Route to appropriate handler
            handler_name = self.command_registry.get_command(command).get("handler")
            handler = getattr(self, handler_name, None)

            if not handler:
                return PluginResponse(
                    success=False,
                    message=f"No handler for command: {command}",
                )

            # Execute handler
            response = await handler(request)
            return response

        except Exception as e:
            logger.error(f"Error handling command: {e}", exc_info=True)
            return PluginResponse(
                success=False,
                message="An error occurred processing your command",
                error=str(e),
            )

    async def handle_video_command(self, request: CommandRequest) -> PluginResponse:
        """Handle /video command."""
        topic = request.args.get("topic", "")
        if not topic:
            return PluginResponse(success=False, message="Topic is required")

        job = Job(command="/video", args=request.args, user_id=request.user_id)
        job.add_log(f"Video creation requested: {topic}")

        # Queue the job
        self.job_queue[job.job_id] = job

        # Schedule async execution
        asyncio.create_task(self._execute_video_job(job, request))

        return PluginResponse(
            success=True,
            message=f"Video creation queued: {topic}",
            job_id=job.job_id,
            data={"job_id": job.job_id},
            embed_data={
                "title": "🎥 Video Creation Started",
                "description": f"Creating video for: **{topic}**",
                "fields": [
                    {"name": "Job ID", "value": job.job_id, "inline": True},
                    {"name": "Status", "value": "Queued", "inline": True},
                ],
                "color": 3447003,  # Blue
            },
        )

    async def _execute_video_job(self, job: Job, request: CommandRequest) -> None:
        """Execute video creation job."""
        try:
            job.started_at = datetime.utcnow()
            job.status = JobStatus.RUNNING

            # TODO: Integrate with actual video creation pipeline
            # from src.content.script_writer import ScriptWriter
            # from src.content.tts import TextToSpeech
            # from src.content.video_assembler import VideoAssembler

            topic = job.args.get("topic", "")
            job.add_log("Generating script...")
            job.set_progress(20)
            await asyncio.sleep(1)  # Simulate work

            job.add_log("Generating TTS audio...")
            job.set_progress(40)
            await asyncio.sleep(1)

            job.add_log("Creating video...")
            job.set_progress(60)
            await asyncio.sleep(1)

            job.add_log("Processing thumbnails...")
            job.set_progress(80)
            await asyncio.sleep(1)

            if request.args.get("upload", True):
                job.add_log("Uploading to YouTube...")
                job.set_progress(95)
                await asyncio.sleep(1)

            job.add_log("Complete!")
            job.set_progress(100)
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.results = {
                "video_url": "https://youtube.com/watch?v=dQw4w9WgXcQ",
                "title": topic,
                "duration": "10:32",
            }

        except Exception as e:
            logger.error(f"Job {job.job_id} failed: {e}")
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.utcnow()

    async def handle_short_command(self, request: CommandRequest) -> PluginResponse:
        """Handle /short command."""
        topic = request.args.get("topic", "")
        if not topic:
            return PluginResponse(success=False, message="Topic is required")

        job = Job(command="/short", args=request.args, user_id=request.user_id)
        self.job_queue[job.job_id] = job

        asyncio.create_task(self._execute_short_job(job, request))

        return PluginResponse(
            success=True,
            message=f"YouTube Short creation queued: {topic}",
            job_id=job.job_id,
        )

    async def _execute_short_job(self, job: Job, request: CommandRequest) -> None:
        """Execute short creation job."""
        try:
            job.started_at = datetime.utcnow()
            job.status = JobStatus.RUNNING
            job.add_log("Creating YouTube Short...")
            await asyncio.sleep(2)
            job.set_progress(100)
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)

    async def handle_batch_command(self, request: CommandRequest) -> PluginResponse:
        """Handle /batch command."""
        count = request.args.get("count")
        try:
            count = int(count)
        except (ValueError, TypeError):
            return PluginResponse(success=False, message="Count must be a number")

        if count < 1 or count > 100:
            return PluginResponse(success=False, message="Count must be between 1 and 100")

        job = Job(command="/batch", args=request.args, user_id=request.user_id)
        self.job_queue[job.job_id] = job

        return PluginResponse(
            success=True,
            message=f"Batch video creation queued: {count} videos",
            job_id=job.job_id,
        )

    async def handle_multiplatform_command(self, request: CommandRequest) -> PluginResponse:
        """Handle /multiplatform command."""
        video_id = request.args.get("video_id")
        if not video_id:
            return PluginResponse(success=False, message="video_id is required")

        job = Job(command="/multiplatform", args=request.args, user_id=request.user_id)
        self.job_queue[job.job_id] = job

        return PluginResponse(
            success=True,
            message=f"Export to platforms queued for video: {video_id}",
            job_id=job.job_id,
        )

    async def handle_schedule_command(self, request: CommandRequest) -> PluginResponse:
        """Handle /schedule command."""
        topic = request.args.get("topic")
        scheduled_time = request.args.get("scheduled_time")

        if not topic or not scheduled_time:
            return PluginResponse(success=False, message="topic and scheduled_time are required")

        return PluginResponse(
            success=True,
            message=f"Video scheduled: {topic} at {scheduled_time}",
        )

    async def handle_analytics_command(self, request: CommandRequest) -> PluginResponse:
        """Handle /analytics command."""
        channel = request.args.get("channel")
        period = request.args.get("period", "week")

        # TODO: Integrate with actual analytics system
        analytics_data = {
            "period": period,
            "views": 15234,
            "ctr": 0.042,
            "retention": 0.68,
            "engagement": {
                "likes": 342,
                "comments": 23,
                "shares": 12,
            },
        }

        return PluginResponse(
            success=True,
            message=f"Analytics for {channel or 'all channels'}",
            data=analytics_data,
        )

    async def handle_cost_command(self, request: CommandRequest) -> PluginResponse:
        """Handle /cost command."""
        period = request.args.get("period", "month")

        # TODO: Integrate with token manager
        cost_data = {
            "period": period,
            "total_cost": 24.50,
            "token_count": 1234567,
            "breakdown": {
                "claude": 12.30,
                "groq": 5.20,
                "other": 7.00,
            },
        }

        return PluginResponse(
            success=True,
            message=f"API costs for {period}",
            data=cost_data,
        )

    async def handle_status_command(self, request: CommandRequest) -> PluginResponse:
        """Handle /status command."""
        queued_count = sum(1 for j in self.job_queue.values() if j.status == JobStatus.QUEUED)
        running_count = sum(1 for j in self.job_queue.values() if j.status == JobStatus.RUNNING)

        status_data = {
            "system_status": "operational",
            "queued_jobs": queued_count,
            "running_jobs": running_count,
            "uptime": "23h 45m",
        }

        return PluginResponse(
            success=True,
            message="System operational",
            data=status_data,
        )

    async def handle_configure_command(self, request: CommandRequest) -> PluginResponse:
        """Handle /configure command."""
        setting = request.args.get("setting")
        value = request.args.get("value")

        if not setting or value is None:
            return PluginResponse(success=False, message="setting and value are required")

        # TODO: Implement configuration updates
        return PluginResponse(
            success=True,
            message=f"Configuration updated: {setting} = {value}",
        )

    async def handle_help_command(self, request: CommandRequest) -> PluginResponse:
        """Handle /help command."""
        user_permission = self.get_user_permission(request.user_id)
        commands = self.command_registry.list_commands(user_permission)

        command_list = "\n".join([f"`{cmd}`" for cmd in commands])

        help_text = f"""
Available commands for {user_permission.value}s:

{command_list}

Use `/help <command>` for details on a specific command.
"""

        return PluginResponse(
            success=True,
            message=help_text,
        )

    def get_user_permission(self, user_id: str) -> PermissionLevel:
        """Get user permission level."""
        # TODO: Integrate with actual permission system
        # For now, default to USER permission
        if user_id in ["admin_id_1", "admin_id_2"]:
            return PermissionLevel.ADMIN
        return PermissionLevel.USER

    def check_rate_limit(self, user_id: str, command: str) -> tuple[bool, float]:
        """Check if user is rate limited.

        Returns:
            (is_allowed, seconds_to_wait)
        """
        import time

        if user_id not in self.rate_limiter:
            self.rate_limiter[user_id] = {}

        rate_limits = self.config.get("rate_limits", {})
        limit = rate_limits.get(command, 60)  # Default 60 seconds

        last_call = self.rate_limiter[user_id].get(command, 0)
        now = time.time()
        time_since_last = now - last_call

        if time_since_last < limit:
            return False, limit - time_since_last

        self.rate_limiter[user_id][command] = now
        return True, 0

    def get_job_status(self, job_id: str) -> Optional[Job]:
        """Get job status by ID."""
        return self.job_queue.get(job_id)

    def get_user_jobs(self, user_id: str, limit: int = 10) -> List[Job]:
        """Get recent jobs for user."""
        user_jobs = [j for j in self.job_queue.values() if j.user_id == user_id]
        return sorted(user_jobs, key=lambda j: j.created_at, reverse=True)[:limit]

    def clear_completed_jobs(self, older_than_hours: int = 24) -> int:
        """Clear completed jobs older than specified hours."""
        from datetime import timedelta

        cutoff = datetime.utcnow() - timedelta(hours=older_than_hours)
        to_remove = [
            jid
            for jid, job in self.job_queue.items()
            if job.completed_at and job.completed_at < cutoff
        ]

        for jid in to_remove:
            del self.job_queue[jid]

        logger.info(f"Cleared {len(to_remove)} completed jobs")
        return len(to_remove)
