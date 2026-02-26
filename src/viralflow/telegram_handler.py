"""Telegram bot handler for ViralFlow plugin.

Example Telegram bot integration. Requires python-telegram-bot:
    pip install python-telegram-bot

Usage:
    export TELEGRAM_TOKEN="YOUR_BOT_TOKEN"
    python -m src.viralflow.telegram_handler
"""

import os
import asyncio
import json
from typing import Optional
from datetime import datetime
from loguru import logger

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
)

from .plugin import ViralFlowPlugin
from .models import PluginResponse


class ViralFlowTelegramBot:
    """Telegram bot for YouTube automation."""

    def __init__(self, token: str, plugin: Optional[ViralFlowPlugin] = None):
        """Initialize Telegram bot.

        Args:
            token: Telegram bot token
            plugin: ViralFlowPlugin instance
        """
        self.token = token
        self.plugin = plugin or ViralFlowPlugin()
        self.application = Application.builder().token(token).build()

        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Setup Telegram command handlers."""

        async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
            """Handle /start command."""
            welcome_text = """
🎥 **YouTube Automation Bot**

Welcome! I can help you create and distribute videos across multiple platforms.

**Available Commands:**
• `/video` - Create a video
• `/short` - Create a YouTube Short
• `/batch` - Create multiple videos
• `/multiplatform` - Export to all platforms
• `/analytics` - View analytics
• `/cost` - Check API costs
• `/status` - System status
• `/help` - Show all commands

**Example:**
`/video Passive Income Tips money_blueprints`

For more info, use `/help`
"""
            await update.message.reply_text(welcome_text, parse_mode="Markdown")

        async def video_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
            """Handle /video command."""
            if not context.args:
                await update.message.reply_text(
                    "Usage: `/video <topic> [channel]`\n\n"
                    "Example: `/video Passive Income Tips`",
                    parse_mode="Markdown"
                )
                return

            topic = " ".join(context.args)
            await self._send_processing_message(update, "Creating video...")

            response = await self.plugin.handle_command(
                command="/video",
                args={"topic": topic},
                user_id=str(update.effective_user.id),
                username=update.effective_user.username or update.effective_user.first_name,
                platform="telegram",
                channel_id=str(update.effective_chat.id),
            )

            await self._send_response(update, response)

        async def short_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
            """Handle /short command."""
            if not context.args:
                await update.message.reply_text(
                    "Usage: `/short <topic>`",
                    parse_mode="Markdown"
                )
                return

            topic = " ".join(context.args)
            response = await self.plugin.handle_command(
                command="/short",
                args={"topic": topic},
                user_id=str(update.effective_user.id),
                username=update.effective_user.username or update.effective_user.first_name,
                platform="telegram",
                channel_id=str(update.effective_chat.id),
            )

            await self._send_response(update, response)

        async def batch_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
            """Handle /batch command."""
            if not context.args:
                await update.message.reply_text(
                    "Usage: `/batch <count> [channel]`",
                    parse_mode="Markdown"
                )
                return

            try:
                count = int(context.args[0])
            except ValueError:
                await update.message.reply_text("❌ Count must be a number")
                return

            args = {"count": count}
            if len(context.args) > 1:
                args["channel"] = context.args[1]

            response = await self.plugin.handle_command(
                command="/batch",
                args=args,
                user_id=str(update.effective_user.id),
                username=update.effective_user.username or update.effective_user.first_name,
                platform="telegram",
                channel_id=str(update.effective_chat.id),
            )

            await self._send_response(update, response)

        async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
            """Handle /status command."""
            response = await self.plugin.handle_command(
                command="/status",
                args={},
                user_id=str(update.effective_user.id),
                username=update.effective_user.username or update.effective_user.first_name,
                platform="telegram",
            )

            if response.success and response.data:
                status_text = "📊 **System Status**\n\n"
                for key, value in response.data.items():
                    status_text += f"• **{key.replace('_', ' ').title()}:** {value}\n"
                await update.message.reply_text(status_text, parse_mode="Markdown")
            else:
                await self._send_response(update, response)

        async def analytics_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
            """Handle /analytics command."""
            args = {}
            if context.args:
                args["channel"] = context.args[0]

            response = await self.plugin.handle_command(
                command="/analytics",
                args=args,
                user_id=str(update.effective_user.id),
                username=update.effective_user.username or update.effective_user.first_name,
                platform="telegram",
            )

            if response.success and response.data:
                analytics_text = "📈 **Analytics**\n\n"
                for key, value in response.data.items():
                    if isinstance(value, dict):
                        analytics_text += f"**{key.title()}:**\n"
                        for k, v in value.items():
                            analytics_text += f"  • {k.replace('_', ' ').title()}: {v}\n"
                    else:
                        analytics_text += f"• **{key.replace('_', ' ').title()}:** {value}\n"
                await update.message.reply_text(analytics_text, parse_mode="Markdown")
            else:
                await self._send_response(update, response)

        async def cost_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
            """Handle /cost command."""
            period = context.args[0] if context.args else "month"
            response = await self.plugin.handle_command(
                command="/cost",
                args={"period": period},
                user_id=str(update.effective_user.id),
                username=update.effective_user.username or update.effective_user.first_name,
                platform="telegram",
            )

            if response.success and response.data:
                cost_text = f"💰 **API Costs ({period.title()})**\n\n"
                for key, value in response.data.items():
                    if isinstance(value, dict):
                        cost_text += f"**{key.title()}:**\n"
                        for k, v in value.items():
                            cost_text += f"  • {k.replace('_', ' ').title()}: ${v:.2f}\n"
                    else:
                        cost_text += f"• **{key.replace('_', ' ').title()}:** ${value:.2f}\n"
                await update.message.reply_text(cost_text, parse_mode="Markdown")
            else:
                await self._send_response(update, response)

        async def job_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
            """Handle /job command."""
            if not context.args:
                await update.message.reply_text(
                    "Usage: `/job <job_id>`",
                    parse_mode="Markdown"
                )
                return

            job_id = context.args[0]
            job = self.plugin.get_job_status(job_id)

            if not job:
                await update.message.reply_text(f"❌ Job not found: `{job_id}`", parse_mode="Markdown")
                return

            job_text = f"""
📋 **Job Status**

**ID:** `{job.job_id}`
**Command:** `{job.command}`
**Status:** `{job.status.value.upper()}`
**Progress:** {job.progress}%
**Created:** {job.created_at.isoformat()}

"""
            if job.logs:
                logs = "\n".join(job.logs[-3:])
                job_text += f"**Recent Logs:**\n```\n{logs}\n```\n"

            if job.error:
                job_text += f"**Error:**\n```\n{job.error}\n```"

            if job.results:
                job_text += f"\n**Results:**\n`{json.dumps(job.results, indent=2)}`"

            await update.message.reply_text(job_text, parse_mode="Markdown")

        async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
            """Handle /help command."""
            response = await self.plugin.handle_command(
                command="/help",
                args={},
                user_id=str(update.effective_user.id),
                username=update.effective_user.username or update.effective_user.first_name,
                platform="telegram",
            )

            await update.message.reply_text(response.message, parse_mode="Markdown")

        async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
            """Handle errors."""
            logger.error(f"Telegram error: {context.error}")
            if update:
                await update.message.reply_text(
                    f"❌ An error occurred: {str(context.error)[:100]}",
                    parse_mode="Markdown"
                )

        # Register handlers
        self.application.add_handler(CommandHandler("start", start))
        self.application.add_handler(CommandHandler("video", video_command))
        self.application.add_handler(CommandHandler("short", short_command))
        self.application.add_handler(CommandHandler("batch", batch_command))
        self.application.add_handler(CommandHandler("status", status_command))
        self.application.add_handler(CommandHandler("analytics", analytics_command))
        self.application.add_handler(CommandHandler("cost", cost_command))
        self.application.add_handler(CommandHandler("job", job_command))
        self.application.add_handler(CommandHandler("help", help_command))

        self.application.add_error_handler(error_handler)

    async def _send_processing_message(self, update: Update, message: str) -> None:
        """Send processing message."""
        await update.message.reply_text(f"⏳ {message}", parse_mode="Markdown")

    async def _send_response(self, update: Update, response: PluginResponse) -> None:
        """Send plugin response to Telegram."""
        if response.success:
            text = f"✅ {response.message}"
        else:
            text = f"❌ {response.message}"

        if response.job_id:
            text += f"\n\n**Job ID:** `{response.job_id}`"

        if response.error:
            text += f"\n\n**Error:** ```{response.error}```"

        await update.message.reply_text(text, parse_mode="Markdown")

    async def run(self) -> None:
        """Start the bot."""
        logger.info("Starting Telegram bot...")
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()

        logger.info("Telegram bot running. Press Ctrl+C to stop.")
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            await self.application.stop()


async def main():
    """Main entry point."""
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        logger.error("TELEGRAM_TOKEN environment variable required")
        return

    bot = ViralFlowTelegramBot(token=token)
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
