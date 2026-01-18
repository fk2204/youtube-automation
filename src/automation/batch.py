"""
Batch Automation Runner

Creates multiple videos across channels efficiently.
Designed to be called by Claude subagents.

Usage:
    python -m src.automation.batch --videos 3
    python -m src.automation.batch --channel money_blueprints --videos 2
    python -m src.automation.batch --all-channels
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / "config" / ".env")

from loguru import logger
import yaml

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")


def get_channels() -> List[Dict]:
    """Load enabled channels from config."""
    with open(PROJECT_ROOT / "config" / "channels.yaml") as f:
        config = yaml.safe_load(f)
    return [ch for ch in config["channels"] if ch.get("enabled", True)]


def create_video_for_channel(channel_id: str, upload: bool = True) -> Dict:
    """Create and optionally upload a video for a channel."""
    from src.automation.runner import task_full_pipeline, task_upload

    logger.info(f"\n{'='*50}")
    logger.info(f"Creating video for: {channel_id}")
    logger.info(f"{'='*50}")

    # Run pipeline
    result = task_full_pipeline(channel_id)

    if not result["success"]:
        logger.error(f"Pipeline failed: {result.get('error')}")
        return result

    # Upload if requested
    if upload:
        logger.info("\nUploading to YouTube...")
        upload_result = task_upload(
            video_file=result["results"]["video_file"],
            channel_id=channel_id,
            title=result["results"]["title"],
            description=result["results"]["description"],
            tags=result["results"]["tags"]
        )

        result["results"]["upload"] = upload_result
        if upload_result["success"]:
            result["results"]["video_url"] = upload_result["video_url"]
            logger.success(f"Uploaded: {upload_result['video_url']}")

    return result


def run_batch(
    channels: List[str] = None,
    videos_per_channel: int = 1,
    upload: bool = True
) -> Dict:
    """
    Run batch video creation.

    Args:
        channels: List of channel IDs (None = all enabled)
        videos_per_channel: Videos to create per channel
        upload: Whether to upload to YouTube

    Returns:
        Summary of results
    """
    all_channels = get_channels()

    if channels:
        target_channels = [ch for ch in all_channels if ch["id"] in channels]
    else:
        target_channels = all_channels

    logger.info(f"\n{'='*60}")
    logger.info(f"BATCH VIDEO CREATION")
    logger.info(f"Channels: {len(target_channels)}")
    logger.info(f"Videos per channel: {videos_per_channel}")
    logger.info(f"Total videos: {len(target_channels) * videos_per_channel}")
    logger.info(f"Upload: {'Yes' if upload else 'No'}")
    logger.info(f"{'='*60}\n")

    results = {
        "started_at": datetime.now().isoformat(),
        "channels": [],
        "total_videos": 0,
        "successful": 0,
        "failed": 0,
        "uploaded": 0
    }

    for channel in target_channels:
        channel_id = channel["id"]
        channel_results = {
            "channel": channel_id,
            "name": channel["name"],
            "videos": []
        }

        for i in range(videos_per_channel):
            logger.info(f"\n[{channel_id}] Video {i+1}/{videos_per_channel}")

            result = create_video_for_channel(channel_id, upload=upload)
            results["total_videos"] += 1

            video_info = {
                "success": result["success"],
                "title": result.get("results", {}).get("title", "Unknown"),
                "video_file": result.get("results", {}).get("video_file"),
                "video_url": result.get("results", {}).get("video_url")
            }

            if result["success"]:
                results["successful"] += 1
                if result.get("results", {}).get("video_url"):
                    results["uploaded"] += 1
            else:
                results["failed"] += 1
                video_info["error"] = result.get("error")

            channel_results["videos"].append(video_info)

        results["channels"].append(channel_results)

    results["completed_at"] = datetime.now().isoformat()

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("BATCH COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total: {results['total_videos']}")
    logger.info(f"Successful: {results['successful']}")
    logger.info(f"Failed: {results['failed']}")
    logger.info(f"Uploaded: {results['uploaded']}")

    # List uploaded videos
    if results["uploaded"] > 0:
        logger.info("\nUploaded Videos:")
        for ch in results["channels"]:
            for v in ch["videos"]:
                if v.get("video_url"):
                    logger.info(f"  {v['video_url']}")
                    logger.info(f"    \"{v['title']}\"")

    return results


def main():
    parser = argparse.ArgumentParser(description="Batch Video Creation")
    parser.add_argument("--channel", "-c", help="Specific channel ID")
    parser.add_argument("--videos", "-v", type=int, default=1, help="Videos per channel")
    parser.add_argument("--all-channels", "-a", action="store_true", help="Run for all channels")
    parser.add_argument("--no-upload", action="store_true", help="Don't upload to YouTube")
    parser.add_argument("--output", "-o", help="Save results to JSON file")

    args = parser.parse_args()

    # Determine channels
    if args.channel:
        channels = [args.channel]
    elif args.all_channels:
        channels = None  # All channels
    else:
        channels = ["money_blueprints"]  # Default

    # Run batch
    results = run_batch(
        channels=channels,
        videos_per_channel=args.videos,
        upload=not args.no_upload
    )

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nResults saved: {args.output}")

    print("\n" + json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
