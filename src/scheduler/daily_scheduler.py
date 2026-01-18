#!/usr/bin/env python3
"""
Daily Scheduler - Automated YouTube Posting

Schedules 3 videos per channel per day (9 total).

Usage:
    python src/scheduler/daily_scheduler.py
    python src/scheduler/daily_scheduler.py --test  # Run once immediately

Schedule (UTC):
    Money Blueprints:  06:00, 14:00, 20:00
    Mind Unlocked:     08:00, 15:00, 21:00
    Untold Stories:    10:00, 16:00, 22:00
"""

import sys
import random
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/scheduler_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="DEBUG"
)


# ============================================================
# SCHEDULE CONFIGURATION
# ============================================================

# 3 posts per channel, spaced throughout the day (UTC times)
POSTING_SCHEDULE = {
    "money_blueprints": {
        "times": ["06:00", "14:00", "20:00"],
        "topics": [
            "passive income ideas for beginners",
            "how to save money fast",
            "investing for beginners",
            "side hustle ideas 2024",
            "financial freedom tips",
            "budgeting tips that work",
            "money mistakes to avoid",
            "wealth building strategies"
        ],
        "voice": "en-US-GuyNeural"
    },
    "mind_unlocked": {
        "times": ["08:00", "15:00", "21:00"],
        "topics": [
            "dark psychology tricks",
            "stoicism life lessons",
            "signs of manipulation",
            "body language secrets",
            "how to read people",
            "emotional intelligence tips",
            "psychology of persuasion",
            "cognitive biases explained"
        ],
        "voice": "en-US-JennyNeural"
    },
    "untold_stories": {
        "times": ["10:00", "16:00", "22:00"],
        "topics": [
            "unsolved mysteries",
            "true crime documentary",
            "creepy reddit stories",
            "historical mysteries",
            "internet mysteries explained",
            "famous disappearances",
            "scary stories that are true",
            "conspiracy theories debunked"
        ],
        "voice": "en-GB-RyanNeural"
    }
}

# Default privacy (change to "public" when ready)
DEFAULT_PRIVACY = "unlisted"


# ============================================================
# VIDEO CREATION FUNCTION
# ============================================================

def create_and_upload_video(
    channel_id: str,
    topic: Optional[str] = None,
    privacy: str = DEFAULT_PRIVACY
) -> dict:
    """
    Create and upload a video to a specific channel.

    Returns:
        dict with success status and details
    """
    from src.youtube.multi_channel import MultiChannelManager
    from src.agents.subagents import AgentOrchestrator

    config = POSTING_SCHEDULE.get(channel_id)
    if not config:
        return {"success": False, "error": f"Unknown channel: {channel_id}"}

    # Pick random topic if not specified
    if not topic:
        topic = random.choice(config["topics"])

    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"  CREATING VIDEO FOR: {channel_id.upper()}")
    logger.info(f"  Topic: {topic}")
    logger.info(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'='*60}")

    try:
        # Initialize
        manager = MultiChannelManager()
        orchestrator = AgentOrchestrator()

        # Check channel is authenticated
        if channel_id not in manager.channels:
            return {"success": False, "error": f"Channel not authenticated: {channel_id}"}

        # Set voice for this channel
        orchestrator.production_agent.set_voice(config["voice"])

        # Create video (without upload - we'll use multi_channel manager)
        project = orchestrator.create_video(
            niche=topic,
            channel=channel_id,
            upload=False
        )

        if project.status != "completed" or not project.video_file:
            return {
                "success": False,
                "error": f"Video creation failed: {project.errors}",
                "channel": channel_id,
                "topic": topic
            }

        # Prepare metadata
        description = f"""{project.script.description if project.script else ''}

Subscribe for more content!

#shorts #viral #trending"""

        tags = project.script.tags if project.script else []
        title = project.script.title if project.script else topic

        # Upload to channel
        video_url = manager.upload_to_channel(
            channel_id=channel_id,
            video_file=project.video_file,
            title=title,
            description=description,
            tags=tags,
            privacy=privacy,
            thumbnail_file=project.thumbnail_file
        )

        if video_url:
            logger.success(f"Uploaded: {video_url}")
            return {
                "success": True,
                "channel": channel_id,
                "channel_name": manager.channels[channel_id].name,
                "title": title,
                "video_url": video_url,
                "topic": topic
            }
        else:
            return {
                "success": False,
                "error": "Upload failed",
                "channel": channel_id,
                "topic": topic
            }

    except Exception as e:
        logger.error(f"Error: {e}")
        return {
            "success": False,
            "error": str(e),
            "channel": channel_id,
            "topic": topic
        }


# ============================================================
# SCHEDULER
# ============================================================

def run_scheduler():
    """Run the automated posting scheduler."""
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.error("APScheduler not installed!")
        logger.info("Install with: pip install apscheduler")
        return

    scheduler = BlockingScheduler()

    # Add jobs for each channel and time slot
    job_count = 0
    for channel_id, config in POSTING_SCHEDULE.items():
        for time_str in config["times"]:
            hour, minute = time_str.split(":")

            job_id = f"{channel_id}_{time_str.replace(':', '')}"

            scheduler.add_job(
                create_and_upload_video,
                CronTrigger(hour=int(hour), minute=int(minute)),
                args=[channel_id],
                id=job_id,
                name=f"{channel_id} at {time_str}",
                misfire_grace_time=3600  # 1 hour grace period
            )
            job_count += 1
            logger.info(f"  Scheduled: {channel_id} at {time_str} UTC")

    print()
    print("=" * 60)
    print("  YOUTUBE AUTOMATION SCHEDULER")
    print("=" * 60)
    print()
    print(f"  Total jobs scheduled: {job_count}")
    print(f"  Videos per day: 9 (3 per channel)")
    print()
    print("  Schedule (UTC):")
    print("  ---------------")
    for channel_id, config in POSTING_SCHEDULE.items():
        times = ", ".join(config["times"])
        print(f"  {channel_id}: {times}")
    print()
    print("  Status: RUNNING")
    print("  Press Ctrl+C to stop")
    print()
    print("=" * 60)

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
    except Exception as e:
        logger.error(f"Scheduler error: {e}")


def run_test():
    """Run one video for each channel immediately (for testing)."""
    print()
    print("=" * 60)
    print("  TEST MODE - Creating 1 video per channel")
    print("=" * 60)
    print()

    results = []

    for channel_id in POSTING_SCHEDULE.keys():
        result = create_and_upload_video(channel_id)
        results.append(result)

    # Summary
    print()
    print("=" * 60)
    print("  TEST RESULTS")
    print("=" * 60)
    print()

    for result in results:
        status = "SUCCESS" if result["success"] else "FAILED"
        channel = result.get("channel_name", result.get("channel", "Unknown"))
        print(f"  [{status}] {channel}")
        if result["success"]:
            print(f"           {result.get('video_url', 'No URL')}")
        else:
            print(f"           Error: {result.get('error', 'Unknown')}")
        print()

    successful = sum(1 for r in results if r["success"])
    print(f"  Total: {successful}/{len(results)} successful")
    print()


def show_status():
    """Show current schedule and channel status."""
    from src.youtube.multi_channel import MultiChannelManager

    manager = MultiChannelManager()

    print()
    print("=" * 60)
    print("  SCHEDULER STATUS")
    print("=" * 60)
    print()

    print("  Channels:")
    print("  ---------")
    for channel_id, config in POSTING_SCHEDULE.items():
        if channel_id in manager.channels:
            status = "[OK] Authenticated"
            name = manager.channels[channel_id].name
        else:
            status = "[--] Not authenticated"
            name = channel_id

        times = ", ".join(config["times"])
        print(f"  {name}")
        print(f"    Status: {status}")
        print(f"    Times (UTC): {times}")
        print(f"    Topics: {len(config['topics'])} available")
        print()

    print("  Daily Output:")
    print("  -------------")
    print(f"  Videos per channel: 3")
    print(f"  Total videos per day: 9")
    print(f"  Privacy: {DEFAULT_PRIVACY}")
    print()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YouTube Automation Scheduler")
    parser.add_argument("--test", action="store_true", help="Run one video per channel immediately")
    parser.add_argument("--status", action="store_true", help="Show scheduler status")
    parser.add_argument("--channel", type=str, help="Run single video for specific channel")
    args = parser.parse_args()

    # Create directories
    Path("logs").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)

    if args.status:
        show_status()
    elif args.test:
        run_test()
    elif args.channel:
        result = create_and_upload_video(args.channel)
        if result["success"]:
            print(f"\nSuccess: {result['video_url']}")
        else:
            print(f"\nFailed: {result['error']}")
    else:
        run_scheduler()
