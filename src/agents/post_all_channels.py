#!/usr/bin/env python3
"""
Post Videos to All Channels

Creates and uploads videos to all 3 YouTube channels.

Usage:
    python src/agents/post_all_channels.py
    python src/agents/post_all_channels.py --privacy public
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger
from src.youtube.multi_channel import MultiChannelManager
from src.agents.subagents import AgentOrchestrator


# Channel-specific configurations
CHANNEL_TOPICS = {
    "money_blueprints": {
        "topics": [
            "passive income ideas for beginners",
            "how to save money fast",
            "investing for beginners",
            "side hustle ideas 2024",
            "financial freedom tips"
        ],
        "voice": "en-US-GuyNeural"
    },
    "mind_unlocked": {
        "topics": [
            "dark psychology tricks",
            "stoicism life lessons",
            "signs of manipulation",
            "body language secrets",
            "how to read people"
        ],
        "voice": "en-US-JennyNeural"
    },
    "untold_stories": {
        "topics": [
            "unsolved mysteries",
            "true crime documentary",
            "creepy reddit stories",
            "historical mysteries",
            "internet mysteries explained"
        ],
        "voice": "en-GB-RyanNeural"
    }
}


def create_video_for_channel(channel_id: str, orchestrator: AgentOrchestrator):
    """Create a video tailored for a specific channel."""
    config = CHANNEL_TOPICS[channel_id]

    # Pick a random topic
    import random
    topic = random.choice(config["topics"])

    # Set voice for production agent
    orchestrator.production_agent.set_voice(config["voice"])

    # Create video
    project = orchestrator.create_video(
        niche=topic,
        channel=channel_id,
        upload=False  # We'll upload separately
    )

    return project


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Post videos to all channels")
    parser.add_argument("--privacy", default="unlisted", choices=["public", "unlisted", "private"])
    parser.add_argument("--channels", nargs="+", default=["money_blueprints", "mind_unlocked", "untold_stories"])
    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

    print("\n" + "="*60)
    print("  POST TO ALL YOUTUBE CHANNELS")
    print("="*60)
    print(f"\n  Channels: {', '.join(args.channels)}")
    print(f"  Privacy: {args.privacy}")
    print()

    # Initialize
    manager = MultiChannelManager()
    orchestrator = AgentOrchestrator()

    results = []

    for channel_id in args.channels:
        if channel_id not in manager.channels:
            logger.warning(f"Channel not authenticated: {channel_id}")
            continue

        print("\n" + "-"*60)
        print(f"  CHANNEL: {manager.channels[channel_id].name}")
        print("-"*60)

        # Create video
        project = create_video_for_channel(channel_id, orchestrator)

        if project.status != "completed" or not project.video_file:
            logger.error(f"Failed to create video for {channel_id}")
            results.append({
                "channel": channel_id,
                "success": False,
                "error": "Video creation failed"
            })
            continue

        # Upload to channel
        description = f"""{project.script.description if project.script else ''}

Subscribe for more content!

#shorts #viral #trending"""

        tags = project.script.tags if project.script else []

        video_url = manager.upload_to_channel(
            channel_id=channel_id,
            video_file=project.video_file,
            title=project.script.title if project.script else project.topic.title,
            description=description,
            tags=tags,
            privacy=args.privacy,
            thumbnail_file=project.thumbnail_file
        )

        results.append({
            "channel": channel_id,
            "channel_name": manager.channels[channel_id].name,
            "success": video_url is not None,
            "video_url": video_url,
            "title": project.script.title if project.script else "Unknown"
        })

    # Summary
    print("\n" + "="*60)
    print("  RESULTS SUMMARY")
    print("="*60 + "\n")

    for result in results:
        status = "SUCCESS" if result["success"] else "FAILED"
        print(f"  [{status}] {result.get('channel_name', result['channel'])}")
        if result["success"]:
            print(f"           {result['video_url']}")
        print()

    successful = sum(1 for r in results if r["success"])
    print(f"  Total: {successful}/{len(results)} uploaded successfully")
    print()


if __name__ == "__main__":
    main()
