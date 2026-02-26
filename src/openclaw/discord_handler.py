"""Discord bot handler for Openclaw plugin.

Example Discord bot integration. Requires discord.py:
    pip install discord.py

Usage:
    python -m src.openclaw.discord_handler YOUR_TOKEN
"""

import asyncio
import json
from typing import Optional
import discord
from discord.ext import commands
from loguru import logger

from .plugin import OpenclawPlugin
from .models import PluginResponse


class OpencrawDiscordBot:
    """Discord bot for YouTube automation."""

    def __init__(self, token: str, plugin: Optional[OpenclawPlugin] = None):
        """Initialize Discord bot.

        Args:
            token: Discord bot token
            plugin: OpenclawPlugin instance
        """
        self.token = token
        self.plugin = plugin or OpenclawPlugin()
        self.bot = commands.Bot(command_prefix="/", intents=discord.Intents.default())

        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Setup Discord command handlers."""

        @self.bot.event
        async def on_ready():
            logger.info(f"Bot logged in as {self.bot.user}")
            await self.bot.change_presence(
                activity=discord.Activity(
                    type=discord.ActivityType.watching,
                    name="for /help | YouTube Automation"
                )
            )

        @self.bot.command(name="video", help="Create a video")
        async def create_video(ctx: commands.Context, *, topic: Optional[str] = None):
            """Create a video: /video <topic> [options]"""
            if not topic:
                embed = discord.Embed(
                    title="❌ Missing Topic",
                    description="Usage: `/video <topic> [channel]`\n\nExample: `/video Passive Income Tips money_blueprints`",
                    color=discord.Color.red()
                )
                await ctx.send(embed=embed)
                return

            # Parse arguments (basic parsing)
            args_parts = topic.split()
            args = {"topic": args_parts[0] if args_parts else ""}

            if len(args_parts) > 1:
                args["channel"] = args_parts[1]

            response = await self.plugin.handle_command(
                command="/video",
                args=args,
                user_id=str(ctx.author.id),
                username=str(ctx.author.name),
                platform="discord",
                channel_id=str(ctx.channel.id),
                guild_id=str(ctx.guild.id) if ctx.guild else None,
            )

            await self._send_response(ctx, response)

        @self.bot.command(name="short", help="Create a YouTube Short")
        async def create_short(ctx: commands.Context, *, topic: Optional[str] = None):
            """Create a YouTube Short: /short <topic>"""
            if not topic:
                await ctx.send("❌ Topic required. Usage: `/short <topic>`")
                return

            args = {"topic": topic}
            response = await self.plugin.handle_command(
                command="/short",
                args=args,
                user_id=str(ctx.author.id),
                username=str(ctx.author.name),
                platform="discord",
                channel_id=str(ctx.channel.id),
                guild_id=str(ctx.guild.id) if ctx.guild else None,
            )

            await self._send_response(ctx, response)

        @self.bot.command(name="batch", help="Create multiple videos")
        async def batch_videos(ctx: commands.Context, count: int = 5):
            """Create multiple videos: /batch <count> [channel]"""
            args = {"count": count}
            response = await self.plugin.handle_command(
                command="/batch",
                args=args,
                user_id=str(ctx.author.id),
                username=str(ctx.author.name),
                platform="discord",
                channel_id=str(ctx.channel.id),
                guild_id=str(ctx.guild.id) if ctx.guild else None,
            )

            await self._send_response(ctx, response)

        @self.bot.command(name="status", help="Get system status")
        async def status(ctx: commands.Context):
            """Get system status: /status"""
            response = await self.plugin.handle_command(
                command="/status",
                args={},
                user_id=str(ctx.author.id),
                username=str(ctx.author.name),
                platform="discord",
            )

            if response.success and response.data:
                embed = discord.Embed(
                    title="📊 System Status",
                    color=discord.Color.green()
                )
                for key, value in response.data.items():
                    embed.add_field(name=key.replace("_", " ").title(), value=str(value), inline=True)
                await ctx.send(embed=embed)
            else:
                await self._send_response(ctx, response)

        @self.bot.command(name="analytics", help="View analytics")
        async def analytics(ctx: commands.Context, channel: Optional[str] = None):
            """Get analytics: /analytics [channel]"""
            args = {}
            if channel:
                args["channel"] = channel

            response = await self.plugin.handle_command(
                command="/analytics",
                args=args,
                user_id=str(ctx.author.id),
                username=str(ctx.author.name),
                platform="discord",
            )

            if response.success and response.data:
                embed = discord.Embed(
                    title="📈 Analytics",
                    color=discord.Color.blue()
                )
                for key, value in response.data.items():
                    if isinstance(value, dict):
                        embed.add_field(
                            name=key.replace("_", " ").title(),
                            value=json.dumps(value, indent=2)[:1024],
                            inline=False
                        )
                    else:
                        embed.add_field(name=key.replace("_", " ").title(), value=str(value), inline=True)
                await ctx.send(embed=embed)
            else:
                await self._send_response(ctx, response)

        @self.bot.command(name="cost", help="View API costs")
        async def cost(ctx: commands.Context, period: str = "month"):
            """Get cost report: /cost [period]"""
            args = {"period": period}
            response = await self.plugin.handle_command(
                command="/cost",
                args=args,
                user_id=str(ctx.author.id),
                username=str(ctx.author.name),
                platform="discord",
            )

            if response.success and response.data:
                embed = discord.Embed(
                    title="💰 API Costs",
                    color=discord.Color.gold()
                )
                for key, value in response.data.items():
                    if isinstance(value, dict):
                        embed.add_field(
                            name=key.replace("_", " ").title(),
                            value=json.dumps(value, indent=2)[:1024],
                            inline=False
                        )
                    else:
                        embed.add_field(name=key.replace("_", " ").title(), value=str(value), inline=True)
                await ctx.send(embed=embed)
            else:
                await self._send_response(ctx, response)

        @self.bot.command(name="job", help="Check job status")
        async def check_job(ctx: commands.Context, job_id: str):
            """Check job status: /job <job_id>"""
            job = self.plugin.get_job_status(job_id)
            if not job:
                embed = discord.Embed(
                    title="❌ Job Not Found",
                    description=f"Job ID: {job_id}",
                    color=discord.Color.red()
                )
                await ctx.send(embed=embed)
                return

            color_map = {
                "queued": discord.Color.yellow(),
                "running": discord.Color.blue(),
                "completed": discord.Color.green(),
                "failed": discord.Color.red(),
                "cancelled": discord.Color.orange(),
            }

            embed = discord.Embed(
                title=f"📋 Job {job.command}",
                description=f"ID: `{job.job_id}`",
                color=color_map.get(job.status.value, discord.Color.default()),
            )

            embed.add_field(name="Status", value=job.status.value.upper(), inline=True)
            embed.add_field(name="Progress", value=f"{job.progress}%", inline=True)
            embed.add_field(name="Created", value=job.created_at.isoformat(), inline=False)

            if job.logs:
                logs_text = "\n".join(job.logs[-5:])  # Last 5 logs
                embed.add_field(name="Recent Logs", value=f"```{logs_text}```", inline=False)

            if job.results:
                embed.add_field(name="Results", value=json.dumps(job.results, indent=2)[:1024], inline=False)

            if job.error:
                embed.add_field(name="Error", value=f"```{job.error}```", inline=False)

            await ctx.send(embed=embed)

        @self.bot.command(name="help", help="Show help")
        async def help_command(ctx: commands.Context, command: Optional[str] = None):
            """Get help: /help [command]"""
            response = await self.plugin.handle_command(
                command="/help",
                args={"command": command} if command else {},
                user_id=str(ctx.author.id),
                username=str(ctx.author.name),
                platform="discord",
            )

            embed = discord.Embed(
                title="📖 Help",
                description=response.message,
                color=discord.Color.blue()
            )

            await ctx.send(embed=embed)

        @self.bot.event
        async def on_command_error(ctx: commands.Context, error: Exception):
            """Handle command errors."""
            logger.error(f"Command error: {error}")
            embed = discord.Embed(
                title="❌ Error",
                description=str(error)[:1024],
                color=discord.Color.red()
            )
            await ctx.send(embed=embed)

    async def _send_response(self, ctx: commands.Context, response: PluginResponse) -> None:
        """Send plugin response to Discord."""
        if response.embed_data:
            embed = discord.Embed(**response.embed_data)
            await ctx.send(embed=embed)
        else:
            color = discord.Color.green() if response.success else discord.Color.red()
            status = "✅" if response.success else "❌"
            embed = discord.Embed(
                title=f"{status} {response.message}",
                color=color
            )
            if response.job_id:
                embed.add_field(name="Job ID", value=f"`{response.job_id}`", inline=False)
            if response.error:
                embed.add_field(name="Error", value=f"```{response.error}```", inline=False)
            await ctx.send(embed=embed)

    def run(self) -> None:
        """Start the bot."""
        logger.info("Starting Discord bot...")
        self.bot.run(self.token)


async def main():
    """Main entry point."""
    import sys
    import os

    token = os.getenv("DISCORD_TOKEN")
    if not token and len(sys.argv) > 1:
        token = sys.argv[1]

    if not token:
        logger.error("Discord token required. Set DISCORD_TOKEN env var or pass as argument.")
        return

    bot = OpencrawDiscordBot(token=token)
    bot.run()


if __name__ == "__main__":
    asyncio.run(main())
