#!/usr/bin/env python3
"""
Run Subagents - Simple CLI for video creation

Usage:
    python src/agents/run_agents.py "passive income"
    python src/agents/run_agents.py "dark psychology" --channel mind_unlocked
    python src/agents/run_agents.py "true crime" --upload
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger
from src.agents.subagents import AgentOrchestrator, quick_video


def main():
    parser = argparse.ArgumentParser(
        description="Create YouTube videos using AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/agents/run_agents.py "passive income tips"
  python src/agents/run_agents.py "stoicism wisdom" --channel mind_unlocked
  python src/agents/run_agents.py "scary stories" --upload --privacy unlisted

Channels:
  money_blueprints  - Finance & Wealth ($10-25 CPM)
  mind_unlocked     - Psychology ($5-12 CPM)
  untold_stories    - Stories & Entertainment ($4-10 CPM)
        """
    )

    parser.add_argument(
        "niche",
        type=str,
        help="Topic niche for the video"
    )
    parser.add_argument(
        "--channel",
        type=str,
        default=None,
        help="Channel ID from channels.yaml"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to YouTube after creation"
    )
    parser.add_argument(
        "--privacy",
        type=str,
        default="unlisted",
        choices=["public", "unlisted", "private"],
        help="YouTube privacy setting"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="ollama",
        choices=["ollama", "groq", "claude", "openai"],
        help="AI provider for script generation"
    )

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

    print("\n" + "="*60)
    print("  YOUTUBE VIDEO CREATION - SUBAGENT SYSTEM")
    print("="*60)
    print(f"\n  Niche: {args.niche}")
    if args.channel:
        print(f"  Channel: {args.channel}")
    print(f"  Upload: {'Yes' if args.upload else 'No'}")
    print(f"  Provider: {args.provider}")
    print()

    # Create orchestrator and run
    orchestrator = AgentOrchestrator(ai_provider=args.provider)
    project = orchestrator.create_video(
        niche=args.niche,
        channel=args.channel,
        upload=args.upload,
        privacy=args.privacy
    )

    # Print results
    print("\n" + "="*60)
    print("  RESULTS")
    print("="*60)
    print(f"\n  Status: {project.status}")

    if project.topic:
        print(f"  Topic: {project.topic.title}")

    if project.video_file:
        print(f"  Video: {project.video_file}")

    if project.thumbnail_file:
        print(f"  Thumbnail: {project.thumbnail_file}")

    if project.errors:
        print(f"\n  Errors:")
        for err in project.errors:
            print(f"    - {err}")

    print()


if __name__ == "__main__":
    main()
