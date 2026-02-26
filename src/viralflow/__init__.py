"""
ViralFlow Plugin System

Integrates YouTube Automation with ViralFlow bot for Discord and Telegram.
Provides command routing, job management, and real-time status updates.

Usage:
    from src.viralflow import ViralFlowPlugin

    plugin = ViralFlowPlugin()
    result = await plugin.handle_command("/video", {"topic": "Passive Income"})
"""

from .plugin import ViralFlowPlugin
from .models import CommandRequest, JobStatus, PluginResponse
from .command_registry import CommandRegistry
from .api_bridge import APIBridge

__all__ = [
    "ViralFlowPlugin",
    "CommandRequest",
    "JobStatus",
    "PluginResponse",
    "CommandRegistry",
    "APIBridge",
]

__version__ = "1.0.0"
