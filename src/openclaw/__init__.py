"""
Openclaw Bot Plugin System

Integrates YouTube Automation with Openclaw/Clawd bot for Discord and Telegram.
Provides command routing, job management, and real-time status updates.

Usage:
    from src.openclaw import OpenclawPlugin

    plugin = OpenclawPlugin()
    result = await plugin.handle_command("/video", {"topic": "Passive Income"})
"""

from .plugin import OpenclawPlugin
from .models import CommandRequest, JobStatus, PluginResponse
from .command_registry import CommandRegistry
from .api_bridge import APIBridge

__all__ = [
    "OpenclawPlugin",
    "CommandRequest",
    "JobStatus",
    "PluginResponse",
    "CommandRegistry",
    "APIBridge",
]

__version__ = "1.0.0"
