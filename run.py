#!/usr/bin/env python
"""
Quick automation launcher.

Usage:
    python run.py video money_blueprints
    python run.py video mind_unlocked
    python run.py video untold_stories
    python run.py batch 3
    python run.py batch-all
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
  python run.py video <channel>     Create & upload 1 video
  python run.py batch <count>       Create <count> videos for all channels
  python run.py batch-all           Create 1 video for each channel
  python run.py test                Test video creation (no upload)

Channels:
  money_blueprints    Finance content
  mind_unlocked       Psychology content
  untold_stories      Storytelling content

Examples:
  python run.py video money_blueprints
  python run.py batch 3
  python run.py test
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

    elif cmd == "batch":
        count = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        from src.automation.batch import run_batch
        run_batch(videos_per_channel=count, upload=True)

    elif cmd == "batch-all":
        from src.automation.batch import run_batch
        run_batch(channels=None, videos_per_channel=1, upload=True)

    elif cmd == "test":
        channel = sys.argv[2] if len(sys.argv) > 2 else "money_blueprints"
        from src.automation.runner import task_full_pipeline
        result = task_full_pipeline(channel)
        if result["success"]:
            print(f"\n[OK] Video created: {result['results'].get('video_file')}")
        else:
            print(f"\n[FAIL] Failed: {result.get('error')}")

    else:
        print(f"Unknown command: {cmd}")
        print("Run 'python run.py' for help")


if __name__ == "__main__":
    main()
