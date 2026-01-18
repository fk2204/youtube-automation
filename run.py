#!/usr/bin/env python
"""
Quick automation launcher.

Usage:
    python run.py video money_blueprints
    python run.py video mind_unlocked
    python run.py video untold_stories
    python run.py short money_blueprints     # Create YouTube Short
    python run.py batch 3
    python run.py batch-all
    python run.py schedule-shorts            # Run Shorts scheduler only
    python run.py daily-all                  # Run both videos and Shorts scheduler
"""

import sys
import os

# Add project root
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv("config/.env")

def main():
    if len(sys.argv) < 2:
        print("""
YouTube Automation Quick Launcher
=================================

Commands:
  python run.py video <channel>     Create & upload 1 regular video
  python run.py short <channel>     Create & upload 1 YouTube Short (vertical)
  python run.py test <channel>      Test regular video creation (no upload)
  python run.py test-short <channel> Test Short creation (no upload)
  python run.py batch <count>       Create <count> videos for all channels
  python run.py batch-all           Create 1 video for each channel

Scheduler Commands:
  python run.py schedule-shorts     Run Shorts scheduler only
  python run.py schedule-videos     Run regular videos scheduler only
  python run.py daily-all           Run both videos and Shorts scheduler
  python run.py status              Show scheduler status

Video Formats:
  video    - Regular 1920x1080 horizontal video (5-10 min)
  short    - YouTube Shorts 1080x1920 vertical video (15-60 sec)

Channels:
  money_blueprints    Finance content
  mind_unlocked       Psychology content
  untold_stories      Storytelling content

Shorts Schedule:
  - Shorts are posted 2-3 hours after each regular video
  - Configure in config/channels.yaml under shorts_schedule
  - Each channel can have custom delay_hours and standalone_times

Examples:
  python run.py video money_blueprints
  python run.py short money_blueprints
  python run.py test-short mind_unlocked
  python run.py batch 3
  python run.py daily-all              # Start full scheduler
  python run.py schedule-shorts        # Shorts only
        """)
        return

    cmd = sys.argv[1]

    if cmd == "video":
        channel = sys.argv[2] if len(sys.argv) > 2 else "money_blueprints"
        from src.automation.runner import task_full_with_upload
        result = task_full_with_upload(channel)
        if result["success"]:
            print(f"\n[OK] Video uploaded: {result['results'].get('video_url', 'N/A')}")
        else:
            print(f"\n[FAIL] Failed: {result.get('error')}")

    elif cmd == "short":
        # Create and upload a YouTube Short (vertical video)
        channel = sys.argv[2] if len(sys.argv) > 2 else "money_blueprints"
        topic = sys.argv[3] if len(sys.argv) > 3 else None
        from src.automation.runner import task_short_with_upload
        result = task_short_with_upload(channel, topic)
        if result["success"]:
            print(f"\n[OK] YouTube Short uploaded: {result['results'].get('video_url', 'N/A')}")
        else:
            print(f"\n[FAIL] Failed: {result.get('error')}")

    elif cmd == "batch":
        count = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        from src.automation.batch import run_batch
        run_batch(videos_per_channel=count, upload=True)

    elif cmd == "batch-all":
        from src.automation.batch import run_batch
        run_batch(channels=None, videos_per_channel=1, upload=True)

    elif cmd == "test":
        # Test regular video creation (no upload)
        channel = sys.argv[2] if len(sys.argv) > 2 else "money_blueprints"
        from src.automation.runner import task_full_pipeline
        result = task_full_pipeline(channel)
        if result["success"]:
            print(f"\n[OK] Video created: {result['results'].get('video_file')}")
        else:
            print(f"\n[FAIL] Failed: {result.get('error')}")

    elif cmd == "test-short":
        # Test YouTube Short creation (no upload)
        channel = sys.argv[2] if len(sys.argv) > 2 else "money_blueprints"
        topic = sys.argv[3] if len(sys.argv) > 3 else None
        from src.automation.runner import task_short_pipeline
        result = task_short_pipeline(channel, topic)
        if result["success"]:
            print(f"\n[OK] YouTube Short created: {result['results'].get('video_file')}")
            print(f"    Format: 1080x1920 vertical (9:16)")
            print(f"    Duration: 15-60 seconds")
        else:
            print(f"\n[FAIL] Failed: {result.get('error')}")

    elif cmd == "schedule-shorts":
        # Run Shorts scheduler only
        print("\n[INFO] Starting Shorts scheduler...")
        print("       Shorts will be posted after regular videos based on config/channels.yaml")
        from src.scheduler.daily_scheduler import run_scheduler
        run_scheduler(include_videos=False, include_shorts=True)

    elif cmd == "schedule-videos":
        # Run regular videos scheduler only (backwards compatible)
        print("\n[INFO] Starting regular videos scheduler...")
        from src.scheduler.daily_scheduler import run_scheduler
        run_scheduler(include_videos=True, include_shorts=False)

    elif cmd == "daily-all":
        # Run both regular videos and Shorts scheduler
        print("\n[INFO] Starting full scheduler (videos + Shorts)...")
        print("       Regular videos will be posted at scheduled times")
        print("       Shorts will be posted 2-3 hours after each regular video")
        from src.scheduler.daily_scheduler import run_scheduler
        run_scheduler(include_videos=True, include_shorts=True)

    elif cmd == "status":
        # Show scheduler status
        from src.scheduler.daily_scheduler import show_status
        show_status()

    else:
        print(f"Unknown command: {cmd}")
        print("Run 'python run.py' for help")


if __name__ == "__main__":
    main()
