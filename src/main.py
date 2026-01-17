#!/usr/bin/env python3
"""
YouTube Automation Tool - Main Entry Point

This is the main script to run the YouTube automation pipeline.

Usage:
    # Run once for a specific niche
    python src/main.py --niche "python tutorials"

    # Run with upload enabled
    python src/main.py --niche "python tutorials" --upload

    # Run scheduler (daily automation)
    python src/main.py --schedule

    # Test individual components
    python src/main.py --test-tts
    python src/main.py --test-script
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from dotenv import load_dotenv

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/youtube_automation_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="7 days",
    level="DEBUG"
)


def load_config():
    """Load configuration from files."""
    import yaml

    # Load environment variables
    load_dotenv("config/.env")

    # Load main config
    config_path = Path("config/config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Load channels config
    channels_path = Path("config/channels.yaml")
    if channels_path.exists():
        with open(channels_path) as f:
            channels_config = yaml.safe_load(f)
            config["channels"] = channels_config.get("channels", [])
    else:
        config["channels"] = []

    return config


def get_ai_provider():
    """Get AI provider from environment."""
    provider = os.getenv("AI_PROVIDER", "ollama")
    api_key = None

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
    elif provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
    elif provider == "claude":
        api_key = os.getenv("ANTHROPIC_API_KEY")
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")

    return provider, api_key


def run_pipeline(
    niche: str,
    upload: bool = False,
    privacy: str = "unlisted"
):
    """Run the video creation pipeline."""
    from src.agents.crew import YouTubeCrew

    provider, api_key = get_ai_provider()
    logger.info(f"Using AI provider: {provider}")

    crew = YouTubeCrew(provider=provider, api_key=api_key)
    result = crew.run_pipeline(niche=niche, upload=upload, privacy=privacy)

    if result.success:
        logger.success(f"Pipeline completed successfully!")
        logger.info(f"  Title: {result.title}")
        logger.info(f"  Video: {result.video_file}")
        if result.video_url:
            logger.info(f"  URL: {result.video_url}")
    else:
        logger.error(f"Pipeline failed: {result.error}")

    return result


def run_scheduler():
    """Run the automated scheduler."""
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.error("APScheduler not installed. Run: pip install apscheduler")
        return

    config = load_config()
    scheduler = BlockingScheduler()

    def daily_job():
        """Daily video creation job."""
        logger.info("Starting daily video creation job...")

        from src.agents.crew import YouTubeCrew

        provider, api_key = get_ai_provider()
        crew = YouTubeCrew(provider=provider, api_key=api_key)

        channels = config.get("channels", [])
        if not channels:
            # Default to single channel with generic niche
            channels = [{"name": "Default", "enabled": True, "settings": {"niche": "tutorials"}}]

        results = crew.run_daily(channels)

        for result in results:
            if result.success:
                logger.success(f"Created: {result.title}")
            else:
                logger.error(f"Failed: {result.error}")

    # Schedule daily at 10:00 AM
    schedule_time = config.get("scheduler", {}).get("pipeline", {}).get("video_creation", "10:00")
    hour, minute = schedule_time.split(":")

    scheduler.add_job(
        daily_job,
        CronTrigger(hour=int(hour), minute=int(minute)),
        id="daily_video_creation",
        name="Daily Video Creation"
    )

    logger.info(f"Scheduler started. Daily job at {schedule_time}")
    logger.info("Press Ctrl+C to stop")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped")


async def test_tts():
    """Test the TTS module."""
    from src.content.tts import TextToSpeech

    logger.info("Testing Edge-TTS...")

    tts = TextToSpeech()

    # List voices
    logger.info("Available English voices:")
    await tts.print_voices("en")

    # Generate test audio
    test_text = "Hello! This is a test of the text to speech system. It's completely free and sounds natural."

    output_file = "output/test_tts.mp3"
    Path("output").mkdir(exist_ok=True)

    await tts.generate(test_text, output_file)
    logger.success(f"Test audio saved to: {output_file}")


def test_script():
    """Test the script writer."""
    from src.content.script_writer import ScriptWriter

    provider, api_key = get_ai_provider()
    logger.info(f"Testing script writer with provider: {provider}")

    writer = ScriptWriter(provider=provider, api_key=api_key)

    script = writer.generate_script(
        topic="How to make your first Python program",
        duration_minutes=5
    )

    print("\n" + "="*60)
    print(f"TITLE: {script.title}")
    print("="*60)
    print(f"\nDescription:\n{script.description[:200]}...")
    print(f"\nTags: {', '.join(script.tags[:5])}")
    print(f"\nSections: {len(script.sections)}")
    print(f"Total duration: {script.total_duration} seconds")

    print("\n" + "="*60)
    print("NARRATION PREVIEW:")
    print("="*60)
    narration = writer.get_full_narration(script)
    print(narration[:500] + "...")


def test_research():
    """Test the research module."""
    from src.research.trends import TrendResearcher
    from src.research.idea_generator import IdeaGenerator

    logger.info("Testing trend research...")

    researcher = TrendResearcher()
    trends = researcher.get_trending_topics("python programming")

    print("\n" + "="*60)
    print("TRENDING TOPICS")
    print("="*60 + "\n")

    for trend in trends[:5]:
        print(f"  {trend.keyword} ({trend.trend_direction})")

    logger.info("Testing idea generator...")

    provider, api_key = get_ai_provider()
    generator = IdeaGenerator(provider=provider, api_key=api_key)

    ideas = generator.generate_ideas("python programming", count=3)

    print("\n" + "="*60)
    print("VIDEO IDEAS")
    print("="*60 + "\n")

    for i, idea in enumerate(ideas, 1):
        print(f"{i}. {idea.title}")
        print(f"   Score: {idea.score}/100")
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="YouTube Automation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py --niche "python tutorials"
  python src/main.py --niche "web development" --upload
  python src/main.py --schedule
  python src/main.py --test-tts
        """
    )

    parser.add_argument(
        "--niche",
        type=str,
        help="Topic niche for video generation"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload video to YouTube after creation"
    )
    parser.add_argument(
        "--privacy",
        type=str,
        default="unlisted",
        choices=["public", "unlisted", "private"],
        help="YouTube privacy setting (default: unlisted)"
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Run the automated scheduler"
    )
    parser.add_argument(
        "--test-tts",
        action="store_true",
        help="Test the TTS module"
    )
    parser.add_argument(
        "--test-script",
        action="store_true",
        help="Test the script writer"
    )
    parser.add_argument(
        "--test-research",
        action="store_true",
        help="Test the research module"
    )

    args = parser.parse_args()

    # Create output directory
    Path("output").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    print("\n" + "="*60)
    print("  YOUTUBE AUTOMATION TOOL")
    print("="*60 + "\n")

    if args.test_tts:
        asyncio.run(test_tts())
    elif args.test_script:
        test_script()
    elif args.test_research:
        test_research()
    elif args.schedule:
        run_scheduler()
    elif args.niche:
        run_pipeline(
            niche=args.niche,
            upload=args.upload,
            privacy=args.privacy
        )
    else:
        parser.print_help()
        print("\n" + "-"*60)
        print("Quick Start:")
        print("  1. Install Ollama: https://ollama.ai/download")
        print("  2. Run: ollama pull llama3.2")
        print("  3. Run: python src/main.py --niche 'python tutorials'")
        print("-"*60 + "\n")


if __name__ == "__main__":
    main()
