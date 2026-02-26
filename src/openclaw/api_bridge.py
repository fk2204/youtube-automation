"""API bridge for Discord/Telegram bot communication."""

import json
import asyncio
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
from loguru import logger

from .plugin import OpenclawPlugin
from .models import PluginResponse, Job, JobStatus


class APIBridge:
    """Bridge between Openclaw bot and plugin system."""

    def __init__(self, plugin: Optional[OpenclawPlugin] = None, host: str = "0.0.0.0", port: int = 8000):
        """Initialize API bridge.

        Args:
            plugin: OpenclawPlugin instance (creates new if None)
            host: Host to bind to
            port: Port to listen on
        """
        self.plugin = plugin or OpenclawPlugin()
        self.host = host
        self.port = port
        self.callbacks: Dict[str, List[Callable]] = {
            "job_completed": [],
            "job_failed": [],
            "job_progress": [],
        }
        logger.info(f"API Bridge initialized on {host}:{port}")

    def register_callback(self, event: str, callback: Callable) -> None:
        """Register callback for event.

        Events:
            - job_completed: Called when job completes
            - job_failed: Called when job fails
            - job_progress: Called on job progress update
        """
        if event not in self.callbacks:
            self.callbacks[event] = []
        self.callbacks[event].append(callback)

    async def _trigger_callbacks(self, event: str, data: Dict[str, Any]) -> None:
        """Trigger callbacks for event."""
        for callback in self.callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")

    async def handle_webhook(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming webhook from bot.

        Payload format:
        {
            "command": "/video",
            "args": {"topic": "Passive Income Tips"},
            "user_id": "123456",
            "username": "john_doe",
            "platform": "discord",
            "channel_id": "789012",
            "guild_id": "345678"
        }
        """
        try:
            command = payload.get("command", "")
            args = payload.get("args", {})
            user_id = payload.get("user_id", "anonymous")
            username = payload.get("username", "Anonymous")
            platform = payload.get("platform", "discord")
            channel_id = payload.get("channel_id")
            guild_id = payload.get("guild_id")

            response = await self.plugin.handle_command(
                command=command,
                args=args,
                user_id=user_id,
                username=username,
                platform=platform,
                channel_id=channel_id,
                guild_id=guild_id,
            )

            return {
                "success": True,
                "data": response.to_dict(),
            }

        except Exception as e:
            logger.error(f"Webhook error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status."""
        job = self.plugin.get_job_status(job_id)
        if not job:
            return {"success": False, "error": "Job not found"}

        return {
            "success": True,
            "data": job.to_dict(),
        }

    async def get_user_jobs(
        self,
        user_id: str,
        limit: int = 10,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get user's recent jobs."""
        jobs = self.plugin.get_user_jobs(user_id, limit)

        if status:
            jobs = [j for j in jobs if j.status.value == status]

        return {
            "success": True,
            "data": {
                "jobs": [j.to_dict() for j in jobs],
                "total": len(jobs),
            },
        }

    async def cancel_job(self, job_id: str, user_id: str) -> Dict[str, Any]:
        """Cancel a job."""
        job = self.plugin.get_job_status(job_id)
        if not job:
            return {"success": False, "error": "Job not found"}

        # Only allow user to cancel their own jobs
        if job.user_id != user_id:
            return {"success": False, "error": "Permission denied"}

        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return {"success": False, "error": f"Cannot cancel job with status {job.status.value}"}

        job.status = JobStatus.CANCELLED
        return {
            "success": True,
            "message": f"Job {job_id} cancelled",
            "data": job.to_dict(),
        }

    def setup_fastapi(self) -> None:
        """Setup FastAPI app for API server.

        Usage:
            from src.openclaw.api_bridge import APIBridge
            from fastapi import FastAPI

            bridge = APIBridge()
            app = FastAPI()
            bridge.setup_fastapi()
        """
        try:
            from fastapi import FastAPI, HTTPException
            from fastapi.responses import JSONResponse
            from pydantic import BaseModel

            app = FastAPI(title="YouTube Automation API")

            class CommandPayload(BaseModel):
                command: str
                args: Dict[str, Any]
                user_id: str = "anonymous"
                username: str = "Anonymous"
                platform: str = "discord"
                channel_id: Optional[str] = None
                guild_id: Optional[str] = None

            @app.post("/command")
            async def handle_command(payload: CommandPayload):
                """Handle command from bot."""
                return await self.handle_webhook(payload.dict())

            @app.get("/job/{job_id}")
            async def get_job(job_id: str):
                """Get job status."""
                return await self.get_job_status(job_id)

            @app.get("/jobs/{user_id}")
            async def get_jobs(user_id: str, limit: int = 10, status: Optional[str] = None):
                """Get user's jobs."""
                return await self.get_user_jobs(user_id, limit, status)

            @app.post("/job/{job_id}/cancel")
            async def cancel_job(job_id: str, user_id: str):
                """Cancel a job."""
                return await self.cancel_job(job_id, user_id)

            @app.get("/status")
            async def get_status():
                """Get system status."""
                return {
                    "status": "operational",
                    "queued_jobs": sum(
                        1 for j in self.plugin.job_queue.values()
                        if j.status == JobStatus.QUEUED
                    ),
                    "running_jobs": sum(
                        1 for j in self.plugin.job_queue.values()
                        if j.status == JobStatus.RUNNING
                    ),
                }

            @app.get("/health")
            async def health_check():
                """Health check endpoint."""
                return {"status": "healthy"}

            return app

        except ImportError:
            logger.warning("FastAPI not installed. Install with: pip install fastapi uvicorn")
            return None

    def setup_discord_bot(self) -> None:
        """Setup Discord bot integration.

        Usage:
            from src.openclaw.api_bridge import APIBridge
            import discord
            from discord.ext import commands

            bridge = APIBridge()
            bridge.setup_discord_bot()
        """
        logger.info(
            "Discord bot setup requires manual implementation. "
            "See discord_handler.py for example."
        )

    def setup_telegram_bot(self) -> None:
        """Setup Telegram bot integration.

        Usage:
            from src.openclaw.api_bridge import APIBridge
            from telegram.ext import Application

            bridge = APIBridge()
            bridge.setup_telegram_bot()
        """
        logger.info(
            "Telegram bot setup requires manual implementation. "
            "See telegram_handler.py for example."
        )


# Export for easier importing
__all__ = ["APIBridge"]
