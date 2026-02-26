"""Command registration and routing system."""

from typing import Dict, Callable, Any, Optional, List
from loguru import logger
from .models import CommandRequest, PluginResponse, PermissionLevel


class CommandRegistry:
    """Registry for available commands."""

    def __init__(self):
        """Initialize command registry."""
        self.commands: Dict[str, Dict[str, Any]] = {}
        self.permission_cache: Dict[str, PermissionLevel] = {}
        self._register_default_commands()

    def _register_default_commands(self) -> None:
        """Register default commands."""
        commands_to_register = [
            {
                "name": "/video",
                "aliases": ["/create", "/create-video"],
                "description": "Create a single video",
                "usage": "/video <topic> [channel] [--upload] [--privacy public|unlisted|private]",
                "handler": "handle_video_command",
                "required_args": ["topic"],
                "optional_args": ["channel", "duration", "niche"],
                "min_permission": PermissionLevel.USER,
            },
            {
                "name": "/short",
                "aliases": ["/create-short", "/shorts"],
                "description": "Create a YouTube Short",
                "usage": "/short <topic> [channel]",
                "handler": "handle_short_command",
                "required_args": ["topic"],
                "optional_args": ["channel"],
                "min_permission": PermissionLevel.USER,
            },
            {
                "name": "/batch",
                "aliases": ["/batch-videos", "/create-batch"],
                "description": "Create multiple videos",
                "usage": "/batch <count> [channel] [--spacing 1-7]",
                "handler": "handle_batch_command",
                "required_args": ["count"],
                "optional_args": ["channel", "spacing_days"],
                "min_permission": PermissionLevel.MODERATOR,
            },
            {
                "name": "/multiplatform",
                "aliases": ["/export", "/distribute"],
                "description": "Export video to all platforms",
                "usage": "/multiplatform <video_id> [platforms]",
                "handler": "handle_multiplatform_command",
                "required_args": ["video_id"],
                "optional_args": ["platforms"],
                "min_permission": PermissionLevel.USER,
            },
            {
                "name": "/schedule",
                "aliases": ["/schedule-video"],
                "description": "Schedule video creation",
                "usage": "/schedule <topic> <datetime> [channel] [--recurring daily|weekly|monthly]",
                "handler": "handle_schedule_command",
                "required_args": ["topic", "scheduled_time"],
                "optional_args": ["channel", "recurring"],
                "min_permission": PermissionLevel.MODERATOR,
            },
            {
                "name": "/analytics",
                "aliases": ["/stats", "/analytics"],
                "description": "View performance analytics",
                "usage": "/analytics [channel] [--period day|week|month|all]",
                "handler": "handle_analytics_command",
                "required_args": [],
                "optional_args": ["channel", "video_id", "period"],
                "min_permission": PermissionLevel.USER,
            },
            {
                "name": "/cost",
                "aliases": ["/tokens", "/usage"],
                "description": "View API token usage and costs",
                "usage": "/cost [--period day|week|month]",
                "handler": "handle_cost_command",
                "required_args": [],
                "optional_args": ["period"],
                "min_permission": PermissionLevel.USER,
            },
            {
                "name": "/status",
                "aliases": ["/info", "/system-status"],
                "description": "Get system status",
                "usage": "/status [--jobs] [--detailed]",
                "handler": "handle_status_command",
                "required_args": [],
                "optional_args": ["jobs", "detailed"],
                "min_permission": PermissionLevel.PUBLIC,
            },
            {
                "name": "/configure",
                "aliases": ["/config", "/set"],
                "description": "Update configuration",
                "usage": "/configure <setting> <value> [--scope user|channel|global]",
                "handler": "handle_configure_command",
                "required_args": ["setting", "value"],
                "optional_args": ["scope"],
                "min_permission": PermissionLevel.ADMIN,
            },
            {
                "name": "/help",
                "aliases": ["/commands", "/?"],
                "description": "Show available commands",
                "usage": "/help [command]",
                "handler": "handle_help_command",
                "required_args": [],
                "optional_args": ["command"],
                "min_permission": PermissionLevel.PUBLIC,
            },
        ]

        for cmd in commands_to_register:
            self.register_command(cmd)

    def register_command(self, command_info: Dict[str, Any]) -> None:
        """Register a command."""
        name = command_info.get("name", "")
        self.commands[name] = command_info

        # Register aliases
        for alias in command_info.get("aliases", []):
            self.commands[alias] = command_info

        logger.info(f"Registered command: {name}")

    def get_command(self, command_str: str) -> Optional[Dict[str, Any]]:
        """Get command info."""
        return self.commands.get(command_str)

    def parse_command(self, text: str) -> tuple[Optional[str], Dict[str, str]]:
        """Parse command text into command name and arguments.

        Examples:
            "/video Passive Income Tips" -> ("/video", {"topic": "Passive Income Tips"})
            "/batch 5 money_blueprints" -> ("/batch", {"count": "5", "channel": "money_blueprints"})
        """
        parts = text.strip().split(maxsplit=1)
        if not parts:
            return None, {}

        command = parts[0]
        args_str = parts[1] if len(parts) > 1 else ""

        command_info = self.get_command(command)
        if not command_info:
            return None, {}

        # Simple argument parsing (can be enhanced)
        args = self._parse_arguments(args_str, command_info)
        return command, args

    def _parse_arguments(self, args_str: str, command_info: Dict[str, Any]) -> Dict[str, str]:
        """Parse argument string into dict."""
        args = {}
        if not args_str:
            return args

        # Simple parsing: split by spaces, handle quoted strings
        import shlex
        try:
            tokens = shlex.split(args_str)
        except ValueError:
            tokens = args_str.split()

        # Map positional args to required_args
        required = command_info.get("required_args", [])
        optional = command_info.get("optional_args", [])

        for i, token in enumerate(tokens):
            if token.startswith("--"):
                # Handle flag arguments
                key = token[2:]
                if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                    args[key] = tokens[i + 1]
                else:
                    args[key] = "true"
            elif i < len(required):
                args[required[i]] = token
            elif i - len(required) < len(optional):
                args[optional[i - len(required)]] = token

        return args

    def validate_command(
        self,
        command: str,
        args: Dict[str, Any],
        user_permission: PermissionLevel = PermissionLevel.USER,
    ) -> tuple[bool, Optional[str]]:
        """Validate command and arguments.

        Returns:
            (is_valid, error_message)
        """
        cmd_info = self.get_command(command)
        if not cmd_info:
            return False, f"Unknown command: {command}"

        # Check permissions
        min_perm = cmd_info.get("min_permission", PermissionLevel.USER)
        perm_hierarchy = {
            PermissionLevel.PUBLIC: 0,
            PermissionLevel.USER: 1,
            PermissionLevel.MODERATOR: 2,
            PermissionLevel.ADMIN: 3,
        }

        if perm_hierarchy.get(user_permission, 0) < perm_hierarchy.get(min_perm, 0):
            return False, f"Permission denied. Requires {min_perm.value} or higher."

        # Check required arguments
        required = cmd_info.get("required_args", [])
        for arg in required:
            if arg not in args:
                return False, f"Missing required argument: {arg}"

        return True, None

    def list_commands(self, permission: PermissionLevel = PermissionLevel.USER) -> List[str]:
        """List available commands for user permission level."""
        perm_hierarchy = {
            PermissionLevel.PUBLIC: 0,
            PermissionLevel.USER: 1,
            PermissionLevel.MODERATOR: 2,
            PermissionLevel.ADMIN: 3,
        }

        available = set()
        for name, cmd_info in self.commands.items():
            min_perm = cmd_info.get("min_permission", PermissionLevel.USER)
            if perm_hierarchy.get(permission, 0) >= perm_hierarchy.get(min_perm, 0):
                # Only add the primary command name (not aliases)
                if name == cmd_info.get("name"):
                    available.add(name)

        return sorted(list(available))

    def get_command_help(self, command: str) -> Optional[str]:
        """Get help text for a command."""
        cmd_info = self.get_command(command)
        if not cmd_info:
            return None

        return f"""
**{cmd_info.get('name')}** - {cmd_info.get('description', '')}

Usage: `{cmd_info.get('usage', '')}`

Aliases: {', '.join(cmd_info.get('aliases', []))}
"""
