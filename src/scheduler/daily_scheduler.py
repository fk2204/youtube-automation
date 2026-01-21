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
    Regular Videos (2 per day):
        Money Blueprints:  14:00, 18:00 (Tue-Thu)
        Mind Unlocked:     15:00, 20:00 (Tue-Thu)
        Untold Stories:    13:00, 19:00 (Daily)

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

# OPTIMIZED for YouTube Views (2026 Algorithm Research):
# - 2 videos per day per channel for better spacing
# - Shorts follow each video at +1.5h/+3h (finance) or +2h/+4h (psychology/stories)
# - Times optimized for US morning + UK afternoon, US afternoon + UK evening
POSTING_SCHEDULE = {
    "money_blueprints": {
        "times": ["14:00", "18:00"],  # 2 videos/day: US morning (9AM EST) + afternoon (1PM EST)
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
        "times": ["15:00", "20:00"],  # 2 videos/day: US morning (10AM EST) + prime evening (3PM EST)
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
        "times": ["13:00", "19:00"],  # 2 videos/day: lunch viewers + prime evening slot
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


def get_channel_posting_days(channel_id: str) -> Optional[List[int]]:
    """
    Get posting_days configuration for a specific channel.
    Returns list of day numbers (0=Monday, 6=Sunday) or None for every day.
    """
    try:
        config_path = PROJECT_ROOT / "config" / "channels.yaml"
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
            for channel in config.get("channels", []):
                if channel.get("id") == channel_id:
                    settings = channel.get("settings", {})
                    return settings.get("posting_days")
        return None
    except Exception as e:
        logger.warning(f"Failed to load posting_days for {channel_id}: {e}")
        return None


def convert_posting_days_to_cron(posting_days: Optional[List[int]]) -> Optional[str]:
    """Convert posting_days list to cron day_of_week string."""
    if posting_days is None:
        return None
    day_names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    cron_days = [day_names[day] for day in posting_days if 0 <= day <= 6]
    return ",".join(cron_days) if cron_days else None


def calculate_shorts_times(regular_times: List[str], delay_hours: Any = 2) -> List[tuple]:
    """
    Calculate Shorts posting times based on regular video times.

    Supports multiple Shorts per video by accepting a list of delays.

    Args:
        regular_times: List of regular video posting times (e.g., ["06:00", "14:00"])
        delay_hours: Hours to wait after regular video.
                    Can be int (single delay) or list of ints (multiple delays for multiple Shorts)
                    Example: [2, 4] creates 2 Shorts per video at +2h and +4h

    Returns:
        List of tuples: (shorts_time, short_index) where short_index is 0 for first Short, 1 for second, etc.
        This allows scheduling unique content for each Short.
    """
    shorts_times = []

    # Normalize delay_hours to a list
    if isinstance(delay_hours, (int, float)):
        delays = [int(delay_hours)]
    elif isinstance(delay_hours, list):
        delays = [int(d) for d in delay_hours]
    else:
        delays = [2]  # Default fallback

    for time_str in regular_times:
        hour, minute = map(int, time_str.split(":"))

        # Create a Short for each delay
        for short_index, delay in enumerate(delays):
            new_hour = (hour + delay) % 24
            shorts_time = f"{new_hour:02d}:{minute:02d}"
            shorts_times.append((shorts_time, short_index))

    return shorts_times


def get_shorts_times_flat(regular_times: List[str], delay_hours: Any = 2) -> List[str]:
    """
    Get flat list of Shorts times (for display purposes).

    Args:
        regular_times: List of regular video posting times
        delay_hours: Hours to wait (int or list of ints)

    Returns:
        Flat list of unique Shorts times (sorted)
    """
    shorts_with_index = calculate_shorts_times(regular_times, delay_hours)
    times = sorted(set([t[0] for t in shorts_with_index]))
    return times


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
    privacy: str = DEFAULT_PRIVACY,
    short_index: int = 0
) -> dict:
    """
    Create and upload a YouTube Short to a specific channel.

    Uses the Shorts pipeline from runner.py for vertical video creation.

    Args:
        channel_id: The channel to upload to
        topic: Optional specific topic (if None, randomly selected)
        privacy: Video privacy setting
        short_index: Index of this Short (0 = first, 1 = second, etc.)
                    Used to vary topic selection when multiple Shorts per video

    Returns:
        dict with success status and details
    """
    from src.automation.runner import task_short_with_upload, task_short_pipeline

    config = POSTING_SCHEDULE.get(channel_id)
    if not config:
        return {"success": False, "error": f"Unknown channel: {channel_id}"}

    # Pick topic based on short_index to ensure variety
    if not topic:
        topics = config["topics"]
        # Use different topic selection strategy based on short_index
        # This ensures each Short in a batch gets a different topic
        if short_index == 0:
            # First Short: random from first half of topics
            topic = random.choice(topics[:max(1, len(topics)//2)])
        else:
            # Second/subsequent Shorts: random from second half of topics
            topic = random.choice(topics[max(1, len(topics)//2):])

    short_num = short_index + 1
    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"  CREATING YOUTUBE SHORT #{short_num} FOR: {channel_id.upper()}")
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

def run_scheduled_cleanup():
    """
    Run disk cleanup as a scheduled task.

    Called daily by the scheduler to clean up old files and free disk space.
    """
    try:
        from src.utils.cleanup import run_scheduled_cleanup as do_cleanup
        result = do_cleanup()

        if result.get("skipped"):
            logger.info(f"Cleanup skipped: {result.get('reason')} ({result.get('free_gb', 0):.1f} GB free)")
        else:
            logger.success(
                f"Cleanup complete: {result.get('files_deleted', 0)} files deleted, "
                f"{result.get('space_freed_gb', 0):.2f} GB freed"
            )
        return result
    except Exception as e:
        logger.error(f"Scheduled cleanup failed: {e}")
        return {"success": False, "error": str(e)}


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

    # Add daily cleanup job (runs at 04:00 UTC every day)
    scheduler.add_job(
        run_scheduled_cleanup,
        CronTrigger(hour=4, minute=0),
        id="daily_cleanup",
        name="Daily disk cleanup",
        misfire_grace_time=3600  # 1 hour grace period
    )
    logger.info("  Scheduled daily cleanup: 04:00 UTC")

    # Add regular video jobs for each channel and time slot
    if include_videos:
        for channel_id, config in POSTING_SCHEDULE.items():
            # Get posting_days from channels.yaml
            posting_days = get_channel_posting_days(channel_id)
            day_of_week = convert_posting_days_to_cron(posting_days)

            for time_str in config["times"]:
                hour, minute = time_str.split(":")
                job_id = f"{channel_id}_video_{time_str.replace(':', '')}"

                # Build trigger with day_of_week if configured
                if day_of_week:
                    trigger = CronTrigger(day_of_week=day_of_week, hour=int(hour), minute=int(minute))
                    days_str = day_of_week
                else:
                    trigger = CronTrigger(hour=int(hour), minute=int(minute))
                    days_str = "daily"

                scheduler.add_job(
                    create_and_upload_video,
                    trigger,
                    args=[channel_id],
                    id=job_id,
                    name=f"{channel_id} video at {time_str}",
                    misfire_grace_time=3600  # 1 hour grace period
                )
                video_job_count += 1
                logger.info(f"  Scheduled video: {channel_id} at {time_str} UTC ({days_str})")

    # Add Shorts jobs based on configuration
    if include_shorts:
        for channel_id, config in POSTING_SCHEDULE.items():
            shorts_config = get_channel_shorts_config(channel_id)

            if not shorts_config.get("enabled", True):
                logger.info(f"  Shorts disabled for {channel_id}")
                continue

            # Get posting_days (Shorts follow same schedule as regular videos)
            posting_days = get_channel_posting_days(channel_id)
            day_of_week = convert_posting_days_to_cron(posting_days)

            # List of (time_str, short_index) tuples for scheduled Shorts
            shorts_schedule_list = []

            # Add Shorts scheduled after regular videos (supports multiple Shorts per video)
            if shorts_config.get("post_after_regular", True):
                delay_hours = shorts_config.get("delay_hours", [2])  # Default to [2] if not specified
                regular_times = config["times"]
                after_video_shorts = calculate_shorts_times(regular_times, delay_hours)
                shorts_schedule_list.extend(after_video_shorts)

                # Display the delays being used
                if isinstance(delay_hours, list):
                    delays_str = ", ".join([f"+{d}h" for d in delay_hours])
                    shorts_per_video = len(delay_hours)
                else:
                    delays_str = f"+{delay_hours}h"
                    shorts_per_video = 1
                times_flat = get_shorts_times_flat(regular_times, delay_hours)
                logger.info(f"  Shorts for {channel_id}: {shorts_per_video} per video ({delays_str}): {times_flat}")

            # Add standalone Shorts times (these are always short_index=0)
            standalone_times = shorts_config.get("standalone_times", [])
            for time_str in standalone_times:
                # Check if this time already exists in the schedule
                existing_times = [t[0] for t in shorts_schedule_list]
                if time_str not in existing_times:
                    shorts_schedule_list.append((time_str, 0))  # Standalone Shorts use index 0
            if standalone_times:
                logger.info(f"  Standalone Shorts for {channel_id}: {standalone_times}")

            # Schedule each Short with its index
            for time_str, short_index in shorts_schedule_list:
                hour, minute = time_str.split(":")
                # Include short_index in job_id to make it unique
                job_id = f"{channel_id}_short_{time_str.replace(':', '')}_{short_index}"

                # Build trigger with day_of_week if configured
                if day_of_week:
                    trigger = CronTrigger(day_of_week=day_of_week, hour=int(hour), minute=int(minute))
                    days_str = day_of_week
                else:
                    trigger = CronTrigger(hour=int(hour), minute=int(minute))
                    days_str = "daily"

                # Pass short_index as keyword argument for topic variation
                scheduler.add_job(
                    create_and_upload_short,
                    trigger,
                    args=[channel_id],
                    kwargs={"short_index": short_index},
                    id=job_id,
                    name=f"{channel_id} short #{short_index+1} at {time_str}",
                    misfire_grace_time=3600  # 1 hour grace period
                )
                shorts_job_count += 1
                logger.info(f"  Scheduled short #{short_index+1}: {channel_id} at {time_str} UTC")

    total_job_count = video_job_count + shorts_job_count + 1  # +1 for cleanup job

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
    print(f"    Maintenance: 1 (daily cleanup)")
    print()
    print("  Maintenance Schedule (UTC):")
    print("  ----------------------------")
    print("    Daily cleanup: 04:00 (removes files > 30 days old)")
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
                delay_hours = shorts_config.get("delay_hours", [2])
                shorts_times_flat = get_shorts_times_flat(config["times"], delay_hours)
                standalone = shorts_config.get("standalone_times", [])
                all_times = sorted(set(shorts_times_flat + standalone))
                # Calculate shorts per video
                if isinstance(delay_hours, list):
                    shorts_per_video = len(delay_hours)
                else:
                    shorts_per_video = 1
                print(f"    {channel_id}: {', '.join(all_times)} ({shorts_per_video} per video)")
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
            delay_hours = shorts_config.get("delay_hours", [2])
            shorts_per_video = shorts_config.get("shorts_per_video", 1)

            # Handle both list and single value for delay_hours
            if isinstance(delay_hours, list):
                delays = delay_hours
                shorts_per_video = len(delays)
            else:
                delays = [delay_hours]

            shorts_times_flat = []

            if shorts_config.get("post_after_regular", True):
                after_times = get_shorts_times_flat(config["times"], delay_hours)
                shorts_times_flat.extend(after_times)

            standalone = shorts_config.get("standalone_times", [])
            for t in standalone:
                if t not in shorts_times_flat:
                    shorts_times_flat.append(t)

            shorts_times_flat = sorted(set(shorts_times_flat))
            # Total Shorts = (videos * shorts_per_video) + standalone
            num_shorts_from_videos = len(config["times"]) * shorts_per_video
            num_standalone = len([t for t in standalone if t not in get_shorts_times_flat(config["times"], delay_hours)])
            num_shorts = num_shorts_from_videos + num_standalone
            total_shorts += num_shorts

            print(f"    Shorts (UTC): {', '.join(shorts_times_flat)}")
            delays_str = ", ".join([f"+{d}h" for d in delays])
            print(f"    Shorts per video: {shorts_per_video} (delays: {delays_str})")
            if shorts_config.get("vary_topics", False):
                print(f"    Topic variation: ENABLED (unique content per Short)")
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
    print("  Shorts Configuration (Global):")
    print("  -------------------------------")
    global_config = load_shorts_config()
    print(f"  Enabled: {global_config.get('enabled', True)}")
    print(f"  Post after regular: {global_config.get('post_after_regular', True)}")
    delay_hours = global_config.get("delay_hours", [2])
    if isinstance(delay_hours, list):
        shorts_per_video = len(delay_hours)
        delays_str = ", ".join([f"+{d}h" for d in delay_hours])
        print(f"  Shorts per video: {shorts_per_video}")
        print(f"  Delay hours: {delays_str}")
    else:
        print(f"  Shorts per video: 1")
        print(f"  Delay hours: +{delay_hours}h")
    print(f"  Vary topics: {global_config.get('vary_topics', False)}")
    print(f"  Standalone times: {global_config.get('standalone_times', [])}")
    print()

    # Show Cleanup configuration
    print("  Disk Cleanup:")
    print("  -------------")
    print("  Schedule: Daily at 04:00 UTC")
    print("  Max file age: 30 days")
    print("  Threshold: Runs if disk < 10 GB free")
    print("  Directories cleaned:")
    print("    - output/videos, output/audio, output/thumbnails, output/shorts")
    print("    - data/stock_cache, cache, logs")
    print("    - System temp (video_ultra, video_shorts, video_fast)")
    print()

    # Show current disk usage summary
    try:
        from src.utils.cleanup import get_disk_usage
        usage = get_disk_usage()
        system_disk = usage.get("system_disk", {})
        if system_disk:
            print("  Current Disk Status:")
            print("  --------------------")
            print(f"  Project size: {usage.get('total_formatted', 'N/A')}")
            print(f"  Disk free: {system_disk.get('free_formatted', 'N/A')} ({100 - system_disk.get('used_percent', 0):.1f}%)")
            print()
    except Exception:
        pass  # Silently skip if cleanup module not available


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
