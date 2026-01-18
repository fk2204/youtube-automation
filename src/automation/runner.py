"""
Token-Optimized Automation Runner

Runs each task independently to minimize context usage.
Each function is self-contained and can be called by subagents.

Usage:
    python -m src.automation.runner research finance
    python -m src.automation.runner script "passive income ideas"
    python -m src.automation.runner video "script.json"
    python -m src.automation.runner upload "video.mp4" money_blueprints
    python -m src.automation.runner full money_blueprints
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / "config" / ".env")

from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")


def task_research(niche: str, count: int = 5) -> Dict[str, Any]:
    """
    TASK: Research trending topics for a niche.

    Returns dict with topics list.
    """
    logger.info(f"Researching topics for: {niche}")

    from src.research.idea_generator import IdeaGenerator

    generator = IdeaGenerator(provider="ollama")
    ideas = generator.generate_ideas(niche=niche, count=count)

    if not ideas:
        return {"success": False, "error": "No topics found"}

    topics = []
    for idea in ideas:
        topics.append({
            "title": idea.title,
            "description": idea.description,
            "score": idea.score,
            "tags": idea.tags
        })

    logger.success(f"Found {len(topics)} topics")
    return {"success": True, "topics": topics}


def task_script(topic: str, niche: str = "default", duration: int = 5) -> Dict[str, Any]:
    """
    TASK: Generate a video script for a topic.

    Returns dict with script data.
    """
    logger.info(f"Writing script for: {topic}")

    from src.content.script_writer import ScriptWriter

    writer = ScriptWriter(provider="ollama")
    script = writer.generate_script(
        topic=topic,
        duration_minutes=duration,
        niche=niche
    )

    if not script:
        return {"success": False, "error": "Script generation failed"}

    # Convert to dict for JSON serialization
    script_data = {
        "title": script.title,
        "description": script.description,
        "tags": script.tags,
        "total_duration": script.total_duration,
        "sections": []
    }

    for section in script.sections:
        script_data["sections"].append({
            "timestamp": section.timestamp,
            "section_type": section.section_type,
            "title": section.title,
            "narration": section.narration,
            "screen_action": section.screen_action,
            "keywords": section.keywords,
            "duration_seconds": section.duration_seconds
        })

    # Get full narration
    narration = writer.get_full_narration(script)
    script_data["full_narration"] = narration

    logger.success(f"Script complete: {len(script.sections)} sections")
    return {"success": True, "script": script_data}


def task_audio(narration: str, output_file: str, voice: str = "en-US-GuyNeural") -> Dict[str, Any]:
    """
    TASK: Generate audio from narration text.

    Returns dict with audio file path.
    """
    logger.info(f"Generating audio: {output_file}")

    from src.content.tts import TextToSpeech

    tts = TextToSpeech(default_voice=voice)

    async def generate():
        await tts.generate(narration, output_file)

    asyncio.run(generate())

    if os.path.exists(output_file):
        size_kb = os.path.getsize(output_file) / 1024
        logger.success(f"Audio created: {size_kb:.1f} KB")
        return {"success": True, "audio_file": output_file}

    return {"success": False, "error": "Audio generation failed"}


def task_video(audio_file: str, script_data: Dict, output_file: str, niche: str = "default") -> Dict[str, Any]:
    """
    TASK: Create video from audio and script.

    Returns dict with video file path.
    """
    logger.info(f"Creating video: {output_file}")

    from src.content.video_ultra import UltraVideoGenerator
    from src.content.script_writer import VideoScript, ScriptSection

    # Reconstruct script object
    sections = []
    for s in script_data.get("sections", []):
        sections.append(ScriptSection(
            timestamp=s.get("timestamp", "00:00"),
            section_type=s.get("section_type", "content"),
            title=s.get("title", ""),
            narration=s.get("narration", ""),
            screen_action=s.get("screen_action", ""),
            keywords=s.get("keywords", []),
            duration_seconds=s.get("duration_seconds", 10)
        ))

    script = VideoScript(
        title=script_data.get("title", "Video"),
        description=script_data.get("description", ""),
        tags=script_data.get("tags", []),
        sections=sections,
        total_duration=script_data.get("total_duration", 60),
        thumbnail_idea=""
    )

    generator = UltraVideoGenerator()
    result = generator.create_video(
        audio_file=audio_file,
        script=script,
        output_file=output_file,
        niche=niche
    )

    if result and os.path.exists(result):
        size_mb = os.path.getsize(result) / (1024 * 1024)
        logger.success(f"Video created: {size_mb:.1f} MB")
        return {"success": True, "video_file": result}

    return {"success": False, "error": "Video generation failed"}


def task_short(audio_file: str, script_data: Dict, output_file: str, niche: str = "default") -> Dict[str, Any]:
    """
    TASK: Create YouTube Short (vertical video) from audio and script.

    YouTube Shorts requirements:
    - Resolution: 1080x1920 (9:16 vertical)
    - Duration: 15-60 seconds max
    - Faster pacing (visual change every 2-3 seconds)
    - Larger text overlays (readable on mobile)

    Returns dict with video file path.
    """
    logger.info(f"Creating YouTube Short: {output_file}")

    from src.content.video_shorts import ShortsVideoGenerator
    from src.content.script_writer import VideoScript, ScriptSection

    # Reconstruct script object
    sections = []
    for s in script_data.get("sections", []):
        sections.append(ScriptSection(
            timestamp=s.get("timestamp", "00:00"),
            section_type=s.get("section_type", "content"),
            title=s.get("title", ""),
            narration=s.get("narration", ""),
            screen_action=s.get("screen_action", ""),
            keywords=s.get("keywords", []),
            duration_seconds=s.get("duration_seconds", 10)
        ))

    script = VideoScript(
        title=script_data.get("title", "Short"),
        description=script_data.get("description", ""),
        tags=script_data.get("tags", []),
        sections=sections,
        total_duration=script_data.get("total_duration", 60),
        thumbnail_idea=""
    )

    generator = ShortsVideoGenerator()
    result = generator.create_short(
        audio_file=audio_file,
        script=script,
        output_file=output_file,
        niche=niche
    )

    if result and os.path.exists(result):
        size_mb = os.path.getsize(result) / (1024 * 1024)
        logger.success(f"YouTube Short created: {size_mb:.1f} MB")
        return {"success": True, "video_file": result}

    return {"success": False, "error": "Short generation failed"}


def task_upload(video_file: str, channel_id: str, title: str, description: str, tags: list, thumbnail: str = None) -> Dict[str, Any]:
    """
    TASK: Upload video to YouTube.

    Returns dict with video URL.
    """
    logger.info(f"Uploading to {channel_id}: {title}")

    from src.youtube.uploader import YouTubeUploader
    import yaml

    # Load channel config
    with open(PROJECT_ROOT / "config" / "channels.yaml") as f:
        config = yaml.safe_load(f)

    # Find channel
    channel_config = None
    for ch in config["channels"]:
        if ch["id"] == channel_id:
            channel_config = ch
            break

    if not channel_config:
        return {"success": False, "error": f"Channel not found: {channel_id}"}

    # Initialize uploader with channel credentials
    creds_file = PROJECT_ROOT / channel_config["credentials_file"]
    uploader = YouTubeUploader(credentials_file=str(creds_file))

    # Upload
    result = uploader.upload_video(
        video_file=video_file,
        title=title,
        description=description,
        tags=tags,
        privacy="unlisted",
        thumbnail_file=thumbnail
    )

    if result.success:
        logger.success(f"Uploaded: {result.video_url}")
        return {
            "success": True,
            "video_id": result.video_id,
            "video_url": result.video_url
        }

    return {"success": False, "error": result.error}


def task_full_pipeline(channel_id: str, topic: str = None) -> Dict[str, Any]:
    """
    TASK: Run full video creation and upload pipeline.

    Returns dict with all outputs.
    """
    import yaml
    import re

    logger.info(f"Starting full pipeline for: {channel_id}")

    # Load channel config
    with open(PROJECT_ROOT / "config" / "channels.yaml") as f:
        config = yaml.safe_load(f)

    # Find channel
    channel_config = None
    for ch in config["channels"]:
        if ch["id"] == channel_id:
            channel_config = ch
            break

    if not channel_config:
        return {"success": False, "error": f"Channel not found: {channel_id}"}

    niche = channel_config["settings"]["niche"]
    voice = channel_config["settings"]["voice"]

    results = {
        "channel": channel_id,
        "niche": niche,
        "steps": {}
    }

    # Step 1: Research (if no topic provided)
    if not topic:
        logger.info("\n[1/4] RESEARCH")
        topics = channel_config["settings"].get("topics", [])
        if topics:
            import random
            topic = random.choice(topics)
        else:
            research = task_research(niche, count=3)
            if not research["success"]:
                return {"success": False, "error": "Research failed", "results": results}
            topic = research["topics"][0]["title"]

    results["topic"] = topic
    results["steps"]["research"] = {"success": True, "topic": topic}

    # Step 2: Script
    logger.info("\n[2/4] SCRIPT")
    script_result = task_script(topic, niche=niche, duration=5)
    if not script_result["success"]:
        return {"success": False, "error": "Script failed", "results": results}

    script_data = script_result["script"]
    results["steps"]["script"] = {"success": True, "title": script_data["title"]}

    # Step 3: Audio
    logger.info("\n[3/4] AUDIO")
    safe_name = re.sub(r'[^\w\s-]', '', script_data["title"])[:40].replace(' ', '_')
    audio_file = str(PROJECT_ROOT / "output" / f"{safe_name}_audio.mp3")

    audio_result = task_audio(script_data["full_narration"], audio_file, voice=voice)
    if not audio_result["success"]:
        return {"success": False, "error": "Audio failed", "results": results}

    results["steps"]["audio"] = {"success": True, "file": audio_file}

    # Step 4: Video
    logger.info("\n[4/4] VIDEO")
    video_file = str(PROJECT_ROOT / "output" / f"{safe_name}.mp4")

    video_result = task_video(audio_file, script_data, video_file, niche=niche)
    if not video_result["success"]:
        return {"success": False, "error": "Video failed", "results": results}

    results["steps"]["video"] = {"success": True, "file": video_file}
    results["video_file"] = video_file
    results["title"] = script_data["title"]
    results["description"] = script_data["description"]
    results["tags"] = script_data["tags"]

    logger.success(f"\nPipeline complete: {video_file}")
    return {"success": True, "results": results}


def task_full_with_upload(channel_id: str, topic: str = None) -> Dict[str, Any]:
    """
    TASK: Run full pipeline and upload to YouTube.
    """
    # Run pipeline
    result = task_full_pipeline(channel_id, topic)

    if not result["success"]:
        return result

    # Upload
    logger.info("\n[5/5] UPLOAD")
    upload_result = task_upload(
        video_file=result["results"]["video_file"],
        channel_id=channel_id,
        title=result["results"]["title"],
        description=result["results"]["description"],
        tags=result["results"]["tags"]
    )

    result["results"]["steps"]["upload"] = upload_result
    if upload_result["success"]:
        result["results"]["video_url"] = upload_result["video_url"]

    return result


def task_short_pipeline(channel_id: str, topic: str = None) -> Dict[str, Any]:
    """
    TASK: Run full YouTube Shorts creation pipeline.

    Creates a vertical short-form video (15-60 seconds) optimized for YouTube Shorts.
    Uses shorter script, faster pacing, and vertical 9:16 format.

    Returns dict with all outputs.
    """
    import yaml
    import re

    logger.info(f"Starting YouTube Shorts pipeline for: {channel_id}")

    # Load channel config
    with open(PROJECT_ROOT / "config" / "channels.yaml") as f:
        config = yaml.safe_load(f)

    # Find channel
    channel_config = None
    for ch in config["channels"]:
        if ch["id"] == channel_id:
            channel_config = ch
            break

    if not channel_config:
        return {"success": False, "error": f"Channel not found: {channel_id}"}

    niche = channel_config["settings"]["niche"]
    voice = channel_config["settings"]["voice"]

    results = {
        "channel": channel_id,
        "niche": niche,
        "format": "short",
        "steps": {}
    }

    # Step 1: Research (if no topic provided)
    if not topic:
        logger.info("\n[1/4] RESEARCH (Shorts)")
        topics = channel_config["settings"].get("topics", [])
        if topics:
            import random
            topic = random.choice(topics)
        else:
            research = task_research(niche, count=3)
            if not research["success"]:
                return {"success": False, "error": "Research failed", "results": results}
            topic = research["topics"][0]["title"]

    results["topic"] = topic
    results["steps"]["research"] = {"success": True, "topic": topic}

    # Step 2: Script (shorter duration for Shorts - 30-45 seconds)
    logger.info("\n[2/4] SCRIPT (Short format)")
    # For shorts, we use a shorter duration (approximately 45 seconds of narration)
    script_result = task_script(topic, niche=niche, duration=1)  # ~1 minute target, will be trimmed
    if not script_result["success"]:
        return {"success": False, "error": "Script failed", "results": results}

    script_data = script_result["script"]

    # Trim narration for Shorts (max 60 seconds of speech ~ 150 words)
    full_narration = script_data.get("full_narration", "")
    words = full_narration.split()
    if len(words) > 150:
        # Take first 150 words for a ~60 second short
        trimmed_narration = " ".join(words[:150])
        script_data["full_narration"] = trimmed_narration
        logger.info(f"Trimmed narration from {len(words)} to 150 words for Shorts")

    results["steps"]["script"] = {"success": True, "title": script_data["title"]}

    # Step 3: Audio
    logger.info("\n[3/4] AUDIO (Short format)")
    safe_name = re.sub(r'[^\w\s-]', '', script_data["title"])[:40].replace(' ', '_')
    audio_file = str(PROJECT_ROOT / "output" / f"{safe_name}_short_audio.mp3")

    audio_result = task_audio(script_data["full_narration"], audio_file, voice=voice)
    if not audio_result["success"]:
        return {"success": False, "error": "Audio failed", "results": results}

    results["steps"]["audio"] = {"success": True, "file": audio_file}

    # Step 4: Video (using ShortsVideoGenerator)
    logger.info("\n[4/4] VIDEO (Shorts - 1080x1920 vertical)")
    video_file = str(PROJECT_ROOT / "output" / f"{safe_name}_short.mp4")

    video_result = task_short(audio_file, script_data, video_file, niche=niche)
    if not video_result["success"]:
        return {"success": False, "error": "Short video generation failed", "results": results}

    results["steps"]["video"] = {"success": True, "file": video_file}
    results["video_file"] = video_file
    results["title"] = script_data["title"] + " #shorts"  # Add shorts tag
    results["description"] = script_data["description"] + "\n\n#shorts #youtubeshorts"
    results["tags"] = script_data["tags"] + ["shorts", "youtubeshorts", "short"]

    logger.success(f"\nShorts pipeline complete: {video_file}")
    return {"success": True, "results": results}


def task_short_with_upload(channel_id: str, topic: str = None) -> Dict[str, Any]:
    """
    TASK: Run full Shorts pipeline and upload to YouTube.
    """
    # Run pipeline
    result = task_short_pipeline(channel_id, topic)

    if not result["success"]:
        return result

    # Upload
    logger.info("\n[5/5] UPLOAD (Shorts)")
    upload_result = task_upload(
        video_file=result["results"]["video_file"],
        channel_id=channel_id,
        title=result["results"]["title"],
        description=result["results"]["description"],
        tags=result["results"]["tags"]
    )

    result["results"]["steps"]["upload"] = upload_result
    if upload_result["success"]:
        result["results"]["video_url"] = upload_result["video_url"]

    return result


# CLI Interface
def main():
    parser = argparse.ArgumentParser(description="YouTube Automation Runner")
    parser.add_argument("task", choices=["research", "script", "audio", "video", "short", "upload", "full", "full-upload", "short-pipeline", "short-upload"])
    parser.add_argument("args", nargs="*", help="Task arguments")
    parser.add_argument("--niche", default="default", help="Content niche")
    parser.add_argument("--channel", help="Channel ID")
    parser.add_argument("--output", help="Output file path")

    args = parser.parse_args()

    if args.task == "research":
        niche = args.args[0] if args.args else args.niche
        result = task_research(niche)

    elif args.task == "script":
        topic = args.args[0] if args.args else "passive income ideas"
        result = task_script(topic, niche=args.niche)

    elif args.task == "audio":
        if len(args.args) < 2:
            print("Usage: runner.py audio <narration> <output_file>")
            return
        result = task_audio(args.args[0], args.args[1])

    elif args.task == "video":
        if len(args.args) < 2:
            print("Usage: runner.py video <audio_file> <script_json>")
            return
        with open(args.args[1]) as f:
            script_data = json.load(f)
        output = args.output or "output/video.mp4"
        result = task_video(args.args[0], script_data, output, niche=args.niche)

    elif args.task == "short":
        # Create a single YouTube Short from audio + script
        if len(args.args) < 2:
            print("Usage: runner.py short <audio_file> <script_json>")
            return
        with open(args.args[1]) as f:
            script_data = json.load(f)
        output = args.output or "output/short.mp4"
        result = task_short(args.args[0], script_data, output, niche=args.niche)

    elif args.task == "upload":
        if len(args.args) < 2:
            print("Usage: runner.py upload <video_file> <channel_id>")
            return
        result = task_upload(
            args.args[0],
            args.args[1],
            title="Test Video",
            description="Test",
            tags=["test"]
        )

    elif args.task == "full":
        channel = args.args[0] if args.args else args.channel or "money_blueprints"
        topic = args.args[1] if len(args.args) > 1 else None
        result = task_full_pipeline(channel, topic)

    elif args.task == "full-upload":
        channel = args.args[0] if args.args else args.channel or "money_blueprints"
        topic = args.args[1] if len(args.args) > 1 else None
        result = task_full_with_upload(channel, topic)

    elif args.task == "short-pipeline":
        # Full Shorts pipeline: research -> script -> audio -> short video
        channel = args.args[0] if args.args else args.channel or "money_blueprints"
        topic = args.args[1] if len(args.args) > 1 else None
        result = task_short_pipeline(channel, topic)

    elif args.task == "short-upload":
        # Full Shorts pipeline with upload
        channel = args.args[0] if args.args else args.channel or "money_blueprints"
        topic = args.args[1] if len(args.args) > 1 else None
        result = task_short_with_upload(channel, topic)

    # Print result
    print("\n" + "="*50)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
