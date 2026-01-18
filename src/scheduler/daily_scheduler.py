#!/usr/bin/env python3
"""
Daily Scheduler - Automated YouTube Posting

Schedules regular videos and YouTube Shorts per channel per day.

Usage:
    python src/scheduler/daily_scheduler.py           # Run full scheduler (videos + shorts)
    python src/scheduler/daily_scheduler.py --test    # Run once immediately
    python src/scheduler/daily_scheduler.py --shorts  # Run Shorts scheduler only
    python src/scheduler/daily_scheduler.py --videos  # Run regular videos only

Schedule (UTC):
    Regular Videos:
        Money Blueprints:  06:00, 14:00, 20:00
        Mind Unlocked:     08:00, 15:00, 21:00
        Untold Stories:    10:00, 16:00, 22:00

    Shorts (posted after regular videos):
        Configured via channels.yaml shorts_schedule
        Default: 2 hours after each regular video
"""

import sys
import random
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
logger.add(
    "logs/shorts_scheduler_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="DEBUG",
    filter=lambda record: "shorts" in record["message"].lower()
)


# ============================================================
# SCHEDULE CONFIGURATION
# ============================================================

# OPTIMIZED for YouTube Views:
# - Best upload window: 3-5 PM EST (19:00-21:00 UTC)
# - Best days: Wednesday-Friday for highest engagement
# - 3 posts per channel, spaced throughout the day (UTC times)
POSTING_SCHEDULE = {
    "money_blueprints": {
        "times": ["15:00", "19:00", "21:00"],  # Optimized: afternoon + prime time EST
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
        "times": ["16:00", "19:30", "21:30"],  # Optimized: staggered from money_blueprints
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
        "times": ["17:00", "20:00", "22:00"],  # Optimized: evening prime time EST
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
# SHORTS SCHEDULE CONFIGURATION
# ============================================================

def load_shorts_config() -> Dict[str, Any]:
    """
    Load Shorts scheduling configuration from channels.yaml.

    Returns default config if not found in yaml.
    """
    default_config = {
        "enabled": True,
        "post_after_regular": True,
        "delay_hours": 2,
        "standalone_times": ["12:00", "18:00"]
    }

    try:
        config_path = PROJECT_ROOT / "config" / "channels.yaml"
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Check for global shorts_schedule
            global_shorts = config.get("global", {}).get("shorts_schedule", {})
            if global_shorts:
                # Merge with defaults
                default_config.update(global_shorts)

        return default_config
    except Exception as e:
        logger.warning(f"Failed to load shorts config: {e}, using defaults")
        return default_config


def get_channel_shorts_config(channel_id: str) -> Dict[str, Any]:
    """
    Get Shorts configuration for a specific channel.

    Checks channel-specific config first, then falls back to global config.
    """
    global_config = load_shorts_config()

    try:
        config_path = PROJECT_ROOT / "config" / "channels.yaml"
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Find channel-specific shorts config
            for channel in config.get("channels", []):
                if channel.get("id") == channel_id:
                    channel_shorts = channel.get("shorts_schedule", {})
                    if channel_shorts:
                        # Merge channel config with global defaults
                        merged = global_config.copy()
                        merged.update(channel_shorts)
                        return merged

        return global_config
    except Exception as e:
        logger.warning(f"Failed to load channel shorts config for {channel_id}: {e}")
        return global_config


def calculate_shorts_times(regular_times: List[str], delay_hours: int = 2) -> List[str]:
    """
    Calculate Shorts posting times based on regular video times.

    Args:
        regular_times: List of regular video posting times (e.g., ["06:00", "14:00"])
        delay_hours: Hours to wait after regular video (default: 2)

    Returns:
        List of Shorts posting times
    """
    shorts_times = []

    for time_str in regular_times:
        hour, minute = map(int, time_str.split(":"))

        # Add delay hours
        new_hour = (hour + delay_hours) % 24
        shorts_time = f"{new_hour:02d}:{minute:02d}"
        shorts_times.append(shorts_time)

    return shorts_times


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
# SHORTS CREATION FUNCTION
# ============================================================

def create_and_upload_short(
    channel_id: str,
    topic: Optional[str] = None,
    privacy: str = DEFAULT_PRIVACY
) -> dict:
    """
    Create and upload a YouTube Short to a specific channel.

    Uses the Shorts pipeline from runner.py for vertical video creation.

    Returns:
        dict with success status and details
    """
    from src.automation.runner import task_short_with_upload, task_short_pipeline

    config = POSTING_SCHEDULE.get(channel_id)
    if not config:
        return {"success": False, "error": f"Unknown channel: {channel_id}"}

    # Pick random topic if not specified
    if not topic:
        topic = random.choice(config["topics"])

    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"  CREATING YOUTUBE SHORT FOR: {channel_id.upper()}")
    logger.info(f"  Topic: {topic}")
    logger.info(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Format: 1080x1920 vertical (9:16)")
    logger.info(f"{'='*60}")

    try:
        # Use the Shorts pipeline from runner
        if privacy == "unlisted":
            # Create and upload
            result = task_short_with_upload(channel_id, topic)
        else:
            # Just create without upload for testing
            result = task_short_pipeline(channel_id, topic)

        if result.get("success"):
            video_url = result.get("results", {}).get("video_url", "N/A")
            video_file = result.get("results", {}).get("video_file", "N/A")
            title = result.get("results", {}).get("title", topic)

            logger.success(f"Shorts uploaded: {video_url}")
            return {
                "success": True,
                "channel": channel_id,
                "title": title,
                "video_url": video_url,
                "video_file": video_file,
                "topic": topic,
                "format": "short"
            }
        else:
            return {
                "success": False,
                "error": result.get("error", "Shorts creation failed"),
                "channel": channel_id,
                "topic": topic,
                "format": "short"
            }

    except Exception as e:
        logger.error(f"Shorts error: {e}")
        return {
            "success": False,
            "error": str(e),
            "channel": channel_id,
            "topic": topic,
            "format": "short"
        }


# ============================================================
# SCHEDULER
# ============================================================

def run_scheduler(include_videos: bool = True, include_shorts: bool = True):
    """
    Run the automated posting scheduler.

    Args:
        include_videos: Schedule regular video posts (default: True)
        include_shorts: Schedule YouTube Shorts posts (default: True)
    """
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.error("APScheduler not installed!")
        logger.info("Install with: pip install apscheduler")
        return

    scheduler = BlockingScheduler()

    video_job_count = 0
    shorts_job_count = 0

    # Add regular video jobs for each channel and time slot
    if include_videos:
        for channel_id, config in POSTING_SCHEDULE.items():
            for time_str in config["times"]:
                hour, minute = time_str.split(":")

                job_id = f"{channel_id}_video_{time_str.replace(':', '')}"

                scheduler.add_job(
                    create_and_upload_video,
                    CronTrigger(hour=int(hour), minute=int(minute)),
                    args=[channel_id],
                    id=job_id,
                    name=f"{channel_id} video at {time_str}",
                    misfire_grace_time=3600  # 1 hour grace period
                )
                video_job_count += 1
                logger.info(f"  Scheduled video: {channel_id} at {time_str} UTC")

    # Add Shorts jobs based on configuration
    if include_shorts:
        for channel_id, config in POSTING_SCHEDULE.items():
            shorts_config = get_channel_shorts_config(channel_id)

            if not shorts_config.get("enabled", True):
                logger.info(f"  Shorts disabled for {channel_id}")
                continue

            shorts_times = []

            # Add Shorts scheduled after regular videos
            if shorts_config.get("post_after_regular", True):
                delay_hours = shorts_config.get("delay_hours", 2)
                regular_times = config["times"]
                after_video_times = calculate_shorts_times(regular_times, delay_hours)
                shorts_times.extend(after_video_times)
                logger.info(f"  Shorts for {channel_id} after regular videos (+{delay_hours}h): {after_video_times}")

            # Add standalone Shorts times
            standalone_times = shorts_config.get("standalone_times", [])
            for time_str in standalone_times:
                if time_str not in shorts_times:
                    shorts_times.append(time_str)
            if standalone_times:
                logger.info(f"  Standalone Shorts for {channel_id}: {standalone_times}")

            # Schedule each Shorts time
            for time_str in shorts_times:
                hour, minute = time_str.split(":")

                job_id = f"{channel_id}_short_{time_str.replace(':', '')}"

                scheduler.add_job(
                    create_and_upload_short,
                    CronTrigger(hour=int(hour), minute=int(minute)),
                    args=[channel_id],
                    id=job_id,
                    name=f"{channel_id} short at {time_str}",
                    misfire_grace_time=3600  # 1 hour grace period
                )
                shorts_job_count += 1
                logger.info(f"  Scheduled short: {channel_id} at {time_str} UTC")

    total_job_count = video_job_count + shorts_job_count

    print()
    print("=" * 60)
    print("  YOUTUBE AUTOMATION SCHEDULER")
    print("=" * 60)
    print()
    print(f"  Total jobs scheduled: {total_job_count}")
    if include_videos:
        print(f"    Regular videos: {video_job_count}")
    if include_shorts:
        print(f"    YouTube Shorts: {shorts_job_count}")
    print()
    print("  Regular Video Schedule (UTC):")
    print("  ------------------------------")
    if include_videos:
        for channel_id, config in POSTING_SCHEDULE.items():
            times = ", ".join(config["times"])
            print(f"    {channel_id}: {times}")
    else:
        print("    (disabled)")
    print()
    print("  Shorts Schedule (UTC):")
    print("  -----------------------")
    if include_shorts:
        for channel_id, config in POSTING_SCHEDULE.items():
            shorts_config = get_channel_shorts_config(channel_id)
            if shorts_config.get("enabled", True):
                delay = shorts_config.get("delay_hours", 2)
                shorts_times = calculate_shorts_times(config["times"], delay)
                standalone = shorts_config.get("standalone_times", [])
                all_times = sorted(set(shorts_times + standalone))
                print(f"    {channel_id}: {', '.join(all_times)}")
            else:
                print(f"    {channel_id}: (disabled)")
    else:
        print("    (disabled)")
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


def run_test(include_shorts: bool = False):
    """
    Run one video for each channel immediately (for testing).

    Args:
        include_shorts: Also create one Short per channel (default: False)
    """
    print()
    print("=" * 60)
    if include_shorts:
        print("  TEST MODE - Creating 1 video + 1 Short per channel")
    else:
        print("  TEST MODE - Creating 1 video per channel")
    print("=" * 60)
    print()

    results = []

    for channel_id in POSTING_SCHEDULE.keys():
        # Create regular video
        result = create_and_upload_video(channel_id)
        result["type"] = "video"
        results.append(result)

        # Create Short if requested
        if include_shorts:
            short_result = create_and_upload_short(channel_id)
            short_result["type"] = "short"
            results.append(short_result)

    # Summary
    print()
    print("=" * 60)
    print("  TEST RESULTS")
    print("=" * 60)
    print()

    for result in results:
        status = "SUCCESS" if result["success"] else "FAILED"
        channel = result.get("channel_name", result.get("channel", "Unknown"))
        content_type = result.get("type", "video").upper()
        print(f"  [{status}] {channel} ({content_type})")
        if result["success"]:
            print(f"           {result.get('video_url', 'No URL')}")
        else:
            print(f"           Error: {result.get('error', 'Unknown')}")
        print()

    successful = sum(1 for r in results if r["success"])
    print(f"  Total: {successful}/{len(results)} successful")
    print()


def run_shorts_test():
    """Run one YouTube Short for each channel immediately (for testing)."""
    print()
    print("=" * 60)
    print("  TEST MODE - Creating 1 YouTube Short per channel")
    print("=" * 60)
    print()

    results = []

    for channel_id in POSTING_SCHEDULE.keys():
        result = create_and_upload_short(channel_id)
        results.append(result)

    # Summary
    print()
    print("=" * 60)
    print("  SHORTS TEST RESULTS")
    print("=" * 60)
    print()

    for result in results:
        status = "SUCCESS" if result["success"] else "FAILED"
        channel = result.get("channel", "Unknown")
        print(f"  [{status}] {channel} (SHORT)")
        if result["success"]:
            print(f"           {result.get('video_url', 'No URL')}")
        else:
            print(f"           Error: {result.get('error', 'Unknown')}")
        print()

    successful = sum(1 for r in results if r["success"])
    print(f"  Total: {successful}/{len(results)} successful")
    print()


def show_status():
    """Show current schedule and channel status including Shorts configuration."""
    from src.youtube.multi_channel import MultiChannelManager

    manager = MultiChannelManager()

    print()
    print("=" * 60)
    print("  SCHEDULER STATUS")
    print("=" * 60)
    print()

    print("  Channels:")
    print("  ---------")
    total_videos = 0
    total_shorts = 0

    for channel_id, config in POSTING_SCHEDULE.items():
        if channel_id in manager.channels:
            status = "[OK] Authenticated"
            name = manager.channels[channel_id].name
        else:
            status = "[--] Not authenticated"
            name = channel_id

        # Regular video times
        video_times = ", ".join(config["times"])
        num_videos = len(config["times"])
        total_videos += num_videos

        # Shorts configuration
        shorts_config = get_channel_shorts_config(channel_id)
        shorts_enabled = shorts_config.get("enabled", True)

        print(f"  {name}")
        print(f"    Status: {status}")
        print(f"    Regular Videos (UTC): {video_times}")
        print(f"    Topics: {len(config['topics'])} available")

        if shorts_enabled:
            delay = shorts_config.get("delay_hours", 2)
            shorts_times = []

            if shorts_config.get("post_after_regular", True):
                after_times = calculate_shorts_times(config["times"], delay)
                shorts_times.extend(after_times)

            standalone = shorts_config.get("standalone_times", [])
            for t in standalone:
                if t not in shorts_times:
                    shorts_times.append(t)

            shorts_times = sorted(set(shorts_times))
            num_shorts = len(shorts_times)
            total_shorts += num_shorts

            print(f"    Shorts (UTC): {', '.join(shorts_times)}")
            print(f"    Shorts delay: {delay} hours after regular videos")
        else:
            print(f"    Shorts: DISABLED")

        print()

    print("  Daily Output:")
    print("  -------------")
    print(f"  Regular videos per day: {total_videos}")
    print(f"  Shorts per day: {total_shorts}")
    print(f"  Total content per day: {total_videos + total_shorts}")
    print(f"  Privacy: {DEFAULT_PRIVACY}")
    print()

    # Show Shorts configuration summary
    print("  Shorts Configuration:")
    print("  ---------------------")
    global_config = load_shorts_config()
    print(f"  Enabled: {global_config.get('enabled', True)}")
    print(f"  Post after regular: {global_config.get('post_after_regular', True)}")
    print(f"  Delay hours: {global_config.get('delay_hours', 2)}")
    print(f"  Standalone times: {global_config.get('standalone_times', [])}")
    print()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YouTube Automation Scheduler")
    parser.add_argument("--test", action="store_true", help="Run one video per channel immediately")
    parser.add_argument("--test-shorts", action="store_true", help="Run one Short per channel immediately")
    parser.add_argument("--test-all", action="store_true", help="Run one video AND one Short per channel")
    parser.add_argument("--status", action="store_true", help="Show scheduler status")
    parser.add_argument("--channel", type=str, help="Run single video for specific channel")
    parser.add_argument("--short", type=str, help="Run single Short for specific channel")
    parser.add_argument("--shorts", action="store_true", help="Run Shorts scheduler only")
    parser.add_argument("--videos", action="store_true", help="Run regular videos scheduler only")
    args = parser.parse_args()

    # Create directories
    Path("logs").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)

    if args.status:
        show_status()
    elif args.test_shorts:
        run_shorts_test()
    elif args.test_all:
        run_test(include_shorts=True)
    elif args.test:
        run_test(include_shorts=False)
    elif args.channel:
        result = create_and_upload_video(args.channel)
        if result["success"]:
            print(f"\nSuccess: {result['video_url']}")
        else:
            print(f"\nFailed: {result['error']}")
    elif args.short:
        result = create_and_upload_short(args.short)
        if result["success"]:
            print(f"\nShort Success: {result['video_url']}")
        else:
            print(f"\nShort Failed: {result['error']}")
    elif args.shorts:
        # Run only Shorts scheduler
        run_scheduler(include_videos=False, include_shorts=True)
    elif args.videos:
        # Run only regular videos scheduler (backwards compatible)
        run_scheduler(include_videos=True, include_shorts=False)
    else:
        # Run both regular videos and Shorts
        run_scheduler(include_videos=True, include_shorts=True)
