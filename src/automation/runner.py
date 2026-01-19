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

# Import budget enforcement utilities
from src.utils.token_manager import (
    BudgetExceededError,
    check_budget_status,
    enforce_budget,
    BudgetGuard,
    load_budget_config
)

# Import best practices for pre-publish checklist
try:
    from src.utils.best_practices import pre_publish_checklist, PrePublishChecklist
    BEST_PRACTICES_AVAILABLE = True
except ImportError:
    BEST_PRACTICES_AVAILABLE = False
    logger.debug("Best practices module not available - pre-publish checklist disabled")

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")


@enforce_budget()
def task_research(niche: str, count: int = 5) -> Dict[str, Any]:
    """
    TASK: Research trending topics for a niche.

    Returns dict with topics list.
    Budget-enforced: Will raise BudgetExceededError if daily limit exceeded.
    """
    logger.info(f"Researching topics for: {niche}")

    from src.research.idea_generator import IdeaGenerator

    generator = IdeaGenerator(provider=os.getenv("AI_PROVIDER", "ollama"))
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


@enforce_budget()
def task_script(topic: str, niche: str = "default", duration: int = 5) -> Dict[str, Any]:
    """
    TASK: Generate a video script for a topic.

    Returns dict with script data.
    Budget-enforced: Will raise BudgetExceededError if daily limit exceeded.
    """
    logger.info(f"Writing script for: {topic}")

    from src.content.script_writer import ScriptWriter

    writer = ScriptWriter(provider=os.getenv("AI_PROVIDER", "ollama"))
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


def task_audio(
    narration: str,
    output_file: str,
    voice: str = "en-US-GuyNeural",
    voice_settings: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    TASK: Generate audio from narration text.

    Args:
        narration: Text to convert to speech
        output_file: Output audio file path
        voice: Voice to use (e.g., "en-US-GuyNeural")
        voice_settings: Optional dict with rate, pitch, volume, use_ssml, dramatic_pauses

    Returns dict with audio file path.
    """
    logger.info(f"Generating audio: {output_file}")

    from src.content.tts import TextToSpeech

    # Extract voice settings with defaults
    settings = voice_settings or {}
    rate = settings.get("rate", "+0%")
    pitch = settings.get("pitch", "+0Hz")
    volume = settings.get("volume", "+0%")
    use_ssml = settings.get("use_ssml", False)
    dramatic_pauses = settings.get("dramatic_pauses", False)

    logger.info(f"Voice settings: rate={rate}, pitch={pitch}, ssml={use_ssml}, pauses={dramatic_pauses}")

    tts = TextToSpeech(default_voice=voice)

    async def generate():
        if use_ssml or dramatic_pauses:
            # Use SSML generation with optional dramatic pauses
            await tts.generate_with_ssml(
                narration,
                output_file,
                rate=rate,
                pitch=pitch,
                volume=volume,
                add_pauses=dramatic_pauses
            )
        else:
            # Standard generation with rate/pitch settings
            await tts.generate(
                narration,
                output_file,
                rate=rate,
                pitch=pitch,
                volume=volume
            )

    asyncio.run(generate())

    if os.path.exists(output_file):
        size_kb = os.path.getsize(output_file) / 1024
        logger.success(f"Audio created: {size_kb:.1f} KB")
        return {"success": True, "audio_file": output_file}

    return {"success": False, "error": "Audio generation failed"}


def task_video(
    audio_file: str,
    script_data: Dict,
    output_file: str,
    niche: str = "default",
    music_enabled: bool = True,
    music_volume: Optional[float] = None,
    subtitles_enabled: bool = True,
    subtitle_style: str = "regular"
) -> Dict[str, Any]:
    """
    TASK: Create video from audio and script.

    Args:
        audio_file: Path to narration audio
        script_data: Script data dict with title, sections, etc.
        output_file: Output video path
        niche: Content niche for styling
        music_enabled: Whether to add background music (default: True)
        music_volume: Optional music volume override (0.0-1.0)
        subtitles_enabled: Burn subtitles into video for 15-25% retention boost (default: True)
        subtitle_style: Subtitle style - "regular", "shorts", "minimal", "cinematic" (default: "regular")

    Returns dict with video file path.
    """
    logger.info(f"Creating video: {output_file}")
    logger.info(f"Music settings: enabled={music_enabled}, volume={music_volume}")
    logger.info(f"Subtitle settings: enabled={subtitles_enabled}, style={subtitle_style}")

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

    # Determine background music path (None if disabled)
    background_music = None
    if music_enabled:
        background_music = generator.get_niche_music_path(niche)

    result = generator.create_video(
        audio_file=audio_file,
        script=script,
        output_file=output_file,
        niche=niche,
        background_music=background_music,
        music_volume=music_volume,
        subtitles_enabled=subtitles_enabled,
        subtitle_style=subtitle_style
    )

    if result and os.path.exists(result):
        size_mb = os.path.getsize(result) / (1024 * 1024)
        logger.success(f"Video created: {size_mb:.1f} MB")
        return {"success": True, "video_file": result}

    return {"success": False, "error": "Video generation failed"}


def task_short(
    audio_file: str,
    script_data: Dict,
    output_file: str,
    niche: str = "default",
    music_enabled: bool = True,
    music_volume: Optional[float] = None,
    subtitles_enabled: bool = False,
    use_pika: bool = False,
    pika_prompts: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    TASK: Create YouTube Short (vertical video) from audio and script.

    YouTube Shorts requirements:
    - Resolution: 1080x1920 (9:16 vertical)
    - Duration: 15-60 seconds max
    - Faster pacing (visual change every 2-3 seconds)
    - Larger text overlays (readable on mobile)

    Args:
        audio_file: Path to narration audio
        script_data: Script data dict with title, sections, etc.
        output_file: Output video path
        niche: Content niche for styling
        music_enabled: Whether to add background music (default: True)
        music_volume: Optional music volume override (0.0-1.0), defaults to 0.15 for Shorts
        subtitles_enabled: Whether to burn subtitles into the video
        use_pika: Use Pika Labs AI for intro/outro clips (requires PIKA_API_KEY)
        pika_prompts: Custom Pika prompts dict with 'intro' and 'outro' keys

    Returns dict with video file path.
    """
    logger.info(f"Creating YouTube Short: {output_file}")
    logger.info(f"Music settings: enabled={music_enabled}, volume={music_volume}")
    logger.info(f"Pika hybrid mode: enabled={use_pika}")

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

    # Determine background music path (None if disabled)
    # Use slightly higher volume for Shorts (0.15 default)
    background_music = None
    shorts_volume = music_volume if music_volume is not None else 0.15
    if music_enabled:
        background_music = generator.get_niche_music_path(niche)

    # Get script text for subtitles
    script_text = script_data.get("full_narration", "")
    if not script_text:
        # Build from sections
        narrations = [s.get("narration", "") for s in script_data.get("sections", []) if s.get("narration")]
        script_text = " ".join(narrations)

    # Convert pika_prompts dict to list format expected by create_short
    # pika_prompts should be [intro_prompt, outro_prompt]
    pika_prompt_list = None
    if pika_prompts:
        pika_prompt_list = [
            pika_prompts.get("intro", ""),
            pika_prompts.get("outro", "")
        ]
        # Filter out empty strings
        pika_prompt_list = [p for p in pika_prompt_list if p] or None

    result = generator.create_short(
        audio_file=audio_file,
        script=script,
        output_file=output_file,
        niche=niche,
        background_music=background_music,
        music_volume=shorts_volume,
        use_pika=use_pika,
        pika_prompts=pika_prompt_list,
        subtitles_enabled=subtitles_enabled,
        script_text=script_text if subtitles_enabled else None
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


def _check_budget_for_pipeline() -> Dict[str, Any]:
    """
    Check budget status before running an expensive pipeline.

    Returns:
        Dict with budget status and whether to proceed
    """
    config = load_budget_config()
    status = check_budget_status()

    if status["exceeded"]:
        logger.error(f"Cannot start pipeline: Daily budget exceeded (${status['spent_today']:.4f} spent)")
        return {"can_proceed": False, "reason": "budget_exceeded", "status": status}

    if status["warning"]:
        logger.warning(
            f"Budget warning: {status['usage_percent']:.1f}% used "
            f"(${status['spent_today']:.4f} of ${config['daily_limit']:.2f})"
        )

    return {"can_proceed": True, "status": status}


def task_full_pipeline(channel_id: str, topic: str = None) -> Dict[str, Any]:
    """
    TASK: Run full video creation and upload pipeline.

    Returns dict with all outputs.
    Budget-enforced: Checks budget before starting and at each step.
    """
    import yaml
    import re

    logger.info(f"Starting full pipeline for: {channel_id}")

    # Check budget before starting pipeline
    budget_check = _check_budget_for_pipeline()
    if not budget_check["can_proceed"]:
        return {
            "success": False,
            "error": f"Budget exceeded: ${budget_check['status']['spent_today']:.4f} spent of ${budget_check['status']['daily_budget']:.2f} limit",
            "budget_status": budget_check["status"]
        }

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
    voice_settings = channel_config["settings"].get("voice_settings", {})

    # Get music settings from channel config, with fallback to global settings
    global_music = config.get("global", {}).get("music", {})
    channel_music_enabled = channel_config["settings"].get("music_enabled", global_music.get("enabled", True))
    channel_music_volume = channel_config["settings"].get("music_volume", global_music.get("volume", 0.12))

    logger.info(f"Music configuration: enabled={channel_music_enabled}, volume={channel_music_volume}")

    results = {
        "channel": channel_id,
        "niche": niche,
        "voice_settings": voice_settings,
        "music_enabled": channel_music_enabled,
        "music_volume": channel_music_volume,
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

    audio_result = task_audio(
        script_data["full_narration"],
        audio_file,
        voice=voice,
        voice_settings=voice_settings
    )
    if not audio_result["success"]:
        return {"success": False, "error": "Audio failed", "results": results}

    results["steps"]["audio"] = {"success": True, "file": audio_file}

    # Step 4: Video
    logger.info("\n[4/4] VIDEO")
    video_file = str(PROJECT_ROOT / "output" / f"{safe_name}.mp4")

    video_result = task_video(
        audio_file,
        script_data,
        video_file,
        niche=niche,
        music_enabled=channel_music_enabled,
        music_volume=channel_music_volume
    )
    if not video_result["success"]:
        return {"success": False, "error": "Video failed", "results": results}

    results["steps"]["video"] = {"success": True, "file": video_file}
    results["video_file"] = video_file
    results["title"] = script_data["title"]
    results["description"] = script_data["description"]
    results["tags"] = script_data["tags"]

    logger.success(f"\nPipeline complete: {video_file}")
    return {"success": True, "results": results}


def task_quality_check(
    video_file: str,
    script_data: Optional[Dict[str, Any]] = None,
    is_short: bool = False,
    threshold: int = 70,
    skip_ai_checks: bool = False
) -> Dict[str, Any]:
    """
    TASK: Run quality check on a video before upload.

    Args:
        video_file: Path to the video file
        script_data: Optional script data dict with title, description, tags, sections
        is_short: Whether this is a YouTube Short
        threshold: Minimum score to pass (0-100)
        skip_ai_checks: Skip AI-based content analysis (faster)

    Returns dict with quality check results.
    """
    logger.info(f"Running quality check on: {video_file}")

    from src.content.quality_checker import VideoQualityChecker

    checker = VideoQualityChecker()
    report = checker.check_video(
        video_file=video_file,
        script_data=script_data,
        is_short=is_short,
        threshold=threshold,
        skip_ai_checks=skip_ai_checks
    )

    logger.info(f"Quality score: {report.overall_score}/100 (threshold: {threshold})")

    if report.passed:
        logger.success(f"Quality check PASSED ({report.overall_score}/100)")
    else:
        logger.warning(f"Quality check FAILED ({report.overall_score}/100)")
        for issue in report.issues[:5]:  # Show top 5 issues
            logger.warning(f"  - {issue.category}: {issue.issue}")

    return {
        "success": True,
        "passed": report.passed,
        "score": report.overall_score,
        "threshold": threshold,
        "issues_count": len(report.issues),
        "issues": [
            {
                "category": issue.category,
                "issue": issue.issue,
                "severity": issue.severity.value,
                "recommendation": issue.recommendation
            }
            for issue in report.issues
        ],
        "recommendations": report.recommendations,
        "summary": report.summary(),
        "report": report.to_dict()
    }


def task_pre_publish_checklist(
    script_data: Dict[str, Any],
    niche: str = "default",
    strict_mode: bool = False
) -> Dict[str, Any]:
    """
    TASK: Run pre-publish checklist to validate content quality.

    Validates content against 2026 YouTube best practices:
    - Title optimization (CTR patterns, length, power words)
    - Hook quality (first 5 seconds)
    - Retention techniques (open loops, micro-payoffs)
    - SEO compliance (tags, description)
    - Visual quality indicators

    Args:
        script_data: Dict with title, description, tags, sections, full_narration
        niche: Content niche (finance, psychology, storytelling)
        strict_mode: If True, requires higher score threshold (75 vs 70)

    Returns:
        Dict with checklist results, score, and ready_to_publish status
    """
    if not BEST_PRACTICES_AVAILABLE:
        logger.warning("Best practices module not available - skipping pre-publish checklist")
        return {
            "success": True,
            "ready_to_publish": True,
            "score": 100,
            "skipped": True,
            "reason": "Best practices module not available"
        }

    logger.info("Running pre-publish checklist...")

    # Build script object for checklist
    from src.content.script_writer import VideoScript, ScriptSection

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
        thumbnail_idea="",
        hook_text=script_data.get("hook_text", "")
    )

    # Run checklist
    checklist = pre_publish_checklist(script, niche)

    # Determine threshold
    threshold = 75 if strict_mode else 70
    passed = checklist.overall_score >= threshold

    # Log results
    if passed:
        logger.success(f"Pre-publish checklist PASSED ({checklist.overall_score:.0%})")
    else:
        logger.warning(f"Pre-publish checklist FAILED ({checklist.overall_score:.0%})")
        for item in checklist.items:
            if not item.passed:
                logger.warning(f"  - FAIL: {item.name}: {item.details}")

    return {
        "success": True,
        "ready_to_publish": passed,
        "score": int(checklist.overall_score * 100),
        "threshold": threshold,
        "strict_mode": strict_mode,
        "items": [
            {
                "name": item.name,
                "passed": item.passed,
                "score": item.score,
                "details": item.details
            }
            for item in checklist.items
        ],
        "critical_issues": [
            item.details for item in checklist.items
            if not item.passed and item.score < 0.5
        ]
    }


def task_full_with_upload(channel_id: str, topic: str = None) -> Dict[str, Any]:
    """
    TASK: Run full pipeline and upload to YouTube.

    Includes pre-publish checklist and quality check before upload.
    Quality gates enforce content standards based on 2026 YouTube best practices.
    """
    import yaml

    # Run pipeline
    result = task_full_pipeline(channel_id, topic)

    if not result["success"]:
        return result

    # Load channel config for quality check settings
    with open(PROJECT_ROOT / "config" / "channels.yaml") as f:
        config = yaml.safe_load(f)

    # Find channel config
    channel_config = None
    for ch in config["channels"]:
        if ch["id"] == channel_id:
            channel_config = ch
            break

    niche = channel_config.get("settings", {}).get("niche", "default")

    # Get quality gate settings (from channel or global)
    global_settings = config.get("global", {})

    # Load quality gates config
    try:
        with open(PROJECT_ROOT / "config" / "config.yaml") as f:
            app_config = yaml.safe_load(f)
        quality_gates = app_config.get("quality_gates", {})
    except (FileNotFoundError, yaml.YAMLError):
        quality_gates = {}

    quality_check_enabled = channel_config.get("settings", {}).get(
        "quality_check_enabled",
        global_settings.get("quality_check_enabled", False)
    )
    quality_threshold = channel_config.get("settings", {}).get(
        "quality_threshold",
        global_settings.get("quality_threshold", 70)
    )
    skip_upload_on_fail = channel_config.get("settings", {}).get(
        "skip_upload_on_quality_fail",
        global_settings.get("skip_upload_on_quality_fail", False)
    )

    # Quality gates: strict mode and pre-publish checklist
    strict_mode = quality_gates.get("strict_mode", False)
    pre_publish_enabled = quality_gates.get("pre_publish_checklist", True)
    block_on_fail = quality_gates.get("block_on_fail", skip_upload_on_fail)

    # Build script_data for checks
    script_data = {
        "title": result["results"]["title"],
        "description": result["results"]["description"],
        "tags": result["results"]["tags"],
        "sections": result["results"]["steps"].get("script", {}).get("sections", []),
        "full_narration": result["results"]["steps"].get("script", {}).get("full_narration", ""),
        "hook_text": result["results"]["steps"].get("script", {}).get("hook_text", "")
    }

    # PRE-PUBLISH CHECKLIST (Phase 1.1 - Quality Gate Enforcement)
    if pre_publish_enabled and BEST_PRACTICES_AVAILABLE:
        logger.info("\n[5/7] PRE-PUBLISH CHECKLIST")

        checklist_result = task_pre_publish_checklist(
            script_data=script_data,
            niche=niche,
            strict_mode=strict_mode
        )

        result["results"]["steps"]["pre_publish_checklist"] = checklist_result
        result["results"]["checklist_score"] = checklist_result["score"]

        if not checklist_result["ready_to_publish"]:
            logger.warning(f"Pre-publish checklist failed: {checklist_result['score']}/{'75' if strict_mode else '70'}")
            if block_on_fail:
                logger.warning("Blocking upload due to failed pre-publish checklist")
                result["results"]["upload_skipped"] = True
                result["results"]["upload_skip_reason"] = "Pre-publish checklist failed"
                result["results"]["critical_issues"] = checklist_result.get("critical_issues", [])
                return result
            else:
                logger.warning("Proceeding with upload despite failed checklist (block_on_fail=false)")

    # Quality check
    if quality_check_enabled:
        step_label = "[6/7] QUALITY CHECK" if pre_publish_enabled else "[5/6] QUALITY CHECK"
        logger.info(f"\n{step_label}")

        quality_result = task_quality_check(
            video_file=result["results"]["video_file"],
            script_data=script_data,
            is_short=False,
            threshold=quality_threshold
        )

        result["results"]["steps"]["quality_check"] = quality_result
        result["results"]["quality_score"] = quality_result["score"]

        if not quality_result["passed"]:
            logger.warning(f"Quality check failed: {quality_result['score']}/{quality_threshold}")
            if skip_upload_on_fail:
                logger.warning("Skipping upload due to failed quality check")
                result["results"]["upload_skipped"] = True
                result["results"]["upload_skip_reason"] = "Quality check failed"
                return result
            else:
                logger.warning("Proceeding with upload despite failed quality check")
    else:
        logger.info("\n[Quality check disabled]")

    # Upload
    if pre_publish_enabled and quality_check_enabled:
        step_num = "7/7"
    elif pre_publish_enabled or quality_check_enabled:
        step_num = "6/6"
    else:
        step_num = "5/5"
    logger.info(f"\n[{step_num}] UPLOAD")
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
    Budget-enforced: Checks budget before starting and at each step.
    """
    import yaml
    import re

    logger.info(f"Starting YouTube Shorts pipeline for: {channel_id}")

    # Check budget before starting pipeline
    budget_check = _check_budget_for_pipeline()
    if not budget_check["can_proceed"]:
        return {
            "success": False,
            "error": f"Budget exceeded: ${budget_check['status']['spent_today']:.4f} spent of ${budget_check['status']['daily_budget']:.2f} limit",
            "budget_status": budget_check["status"]
        }

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
    voice_settings = channel_config["settings"].get("voice_settings", {})

    # Get music settings from channel config, with fallback to global settings
    # For Shorts, use shorts_volume if specified, otherwise use regular volume
    global_music = config.get("global", {}).get("music", {})
    channel_music_enabled = channel_config["settings"].get("music_enabled", global_music.get("enabled", True))
    channel_music_volume = channel_config["settings"].get("music_volume", global_music.get("shorts_volume", 0.15))

    # Get subtitle settings from channel config, with fallback to global settings
    global_subtitles = config.get("global", {}).get("subtitles", {})
    channel_subtitles_enabled = channel_config["settings"].get(
        "subtitles_enabled",
        global_subtitles.get("enabled", True)  # Default enabled - subtitles increase retention by 15-25%
    )

    # Get Pika hybrid settings from channel shorts_schedule, with fallback to global settings
    global_shorts_schedule = config.get("global", {}).get("shorts_schedule", {})
    channel_shorts_schedule = channel_config.get("shorts_schedule", {})
    use_pika = channel_shorts_schedule.get("use_hybrid", global_shorts_schedule.get("use_hybrid", False))
    pika_prompts = channel_shorts_schedule.get("pika_prompts", None)

    logger.info(f"Shorts music configuration: enabled={channel_music_enabled}, volume={channel_music_volume}")
    logger.info(f"Subtitles configuration: enabled={channel_subtitles_enabled}")
    logger.info(f"Pika hybrid configuration: enabled={use_pika}")

    results = {
        "channel": channel_id,
        "niche": niche,
        "format": "short",
        "voice_settings": voice_settings,
        "music_enabled": channel_music_enabled,
        "music_volume": channel_music_volume,
        "subtitles_enabled": channel_subtitles_enabled,
        "use_pika": use_pika,
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

    audio_result = task_audio(
        script_data["full_narration"],
        audio_file,
        voice=voice,
        voice_settings=voice_settings
    )
    if not audio_result["success"]:
        return {"success": False, "error": "Audio failed", "results": results}

    results["steps"]["audio"] = {"success": True, "file": audio_file}

    # Step 4: Video (using ShortsVideoGenerator)
    logger.info("\n[4/4] VIDEO (Shorts - 1080x1920 vertical)")
    video_file = str(PROJECT_ROOT / "output" / f"{safe_name}_short.mp4")

    video_result = task_short(
        audio_file,
        script_data,
        video_file,
        niche=niche,
        music_enabled=channel_music_enabled,
        music_volume=channel_music_volume,
        subtitles_enabled=channel_subtitles_enabled,
        use_pika=use_pika,
        pika_prompts=pika_prompts
    )
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

    Includes pre-publish checklist and quality check before upload.
    Quality gates enforce content standards based on 2026 YouTube best practices.
    """
    import yaml

    # Run pipeline
    result = task_short_pipeline(channel_id, topic)

    if not result["success"]:
        return result

    # Load channel config for quality check settings
    with open(PROJECT_ROOT / "config" / "channels.yaml") as f:
        config = yaml.safe_load(f)

    # Find channel config
    channel_config = None
    for ch in config["channels"]:
        if ch["id"] == channel_id:
            channel_config = ch
            break

    niche = channel_config.get("settings", {}).get("niche", "default")

    # Get quality gate settings (from channel or global)
    global_settings = config.get("global", {})

    # Load quality gates config
    try:
        with open(PROJECT_ROOT / "config" / "config.yaml") as f:
            app_config = yaml.safe_load(f)
        quality_gates = app_config.get("quality_gates", {})
    except (FileNotFoundError, yaml.YAMLError):
        quality_gates = {}

    quality_check_enabled = channel_config.get("settings", {}).get(
        "quality_check_enabled",
        global_settings.get("quality_check_enabled", False)
    )
    quality_threshold = channel_config.get("settings", {}).get(
        "quality_threshold",
        global_settings.get("quality_threshold", 70)
    )
    skip_upload_on_fail = channel_config.get("settings", {}).get(
        "skip_upload_on_quality_fail",
        global_settings.get("skip_upload_on_quality_fail", False)
    )

    # Quality gates: strict mode and pre-publish checklist
    strict_mode = quality_gates.get("strict_mode", False)
    pre_publish_enabled = quality_gates.get("pre_publish_checklist", True)
    block_on_fail = quality_gates.get("block_on_fail", skip_upload_on_fail)

    # Build script_data for checks
    script_data = {
        "title": result["results"]["title"],
        "description": result["results"]["description"],
        "tags": result["results"]["tags"],
        "sections": result["results"]["steps"].get("script", {}).get("sections", []),
        "full_narration": result["results"]["steps"].get("script", {}).get("full_narration", ""),
        "hook_text": result["results"]["steps"].get("script", {}).get("hook_text", "")
    }

    # PRE-PUBLISH CHECKLIST (Phase 1.1 - Quality Gate Enforcement)
    if pre_publish_enabled and BEST_PRACTICES_AVAILABLE:
        logger.info("\n[5/7] PRE-PUBLISH CHECKLIST (Shorts)")

        checklist_result = task_pre_publish_checklist(
            script_data=script_data,
            niche=niche,
            strict_mode=strict_mode
        )

        result["results"]["steps"]["pre_publish_checklist"] = checklist_result
        result["results"]["checklist_score"] = checklist_result["score"]

        if not checklist_result["ready_to_publish"]:
            logger.warning(f"Pre-publish checklist failed: {checklist_result['score']}/{'75' if strict_mode else '70'}")
            if block_on_fail:
                logger.warning("Blocking upload due to failed pre-publish checklist")
                result["results"]["upload_skipped"] = True
                result["results"]["upload_skip_reason"] = "Pre-publish checklist failed"
                result["results"]["critical_issues"] = checklist_result.get("critical_issues", [])
                return result
            else:
                logger.warning("Proceeding with upload despite failed checklist (block_on_fail=false)")

    # Quality check
    if quality_check_enabled:
        step_label = "[6/7] QUALITY CHECK (Shorts)" if pre_publish_enabled else "[5/6] QUALITY CHECK (Shorts)"
        logger.info(f"\n{step_label}")

        quality_result = task_quality_check(
            video_file=result["results"]["video_file"],
            script_data=script_data,
            is_short=True,  # This is a Short
            threshold=quality_threshold
        )

        result["results"]["steps"]["quality_check"] = quality_result
        result["results"]["quality_score"] = quality_result["score"]

        if not quality_result["passed"]:
            logger.warning(f"Quality check failed: {quality_result['score']}/{quality_threshold}")
            if skip_upload_on_fail:
                logger.warning("Skipping upload due to failed quality check")
                result["results"]["upload_skipped"] = True
                result["results"]["upload_skip_reason"] = "Quality check failed"
                return result
            else:
                logger.warning("Proceeding with upload despite failed quality check")
    else:
        logger.info("\n[Quality check disabled]")

    # Upload
    if pre_publish_enabled and quality_check_enabled:
        step_num = "7/7"
    elif pre_publish_enabled or quality_check_enabled:
        step_num = "6/6"
    else:
        step_num = "5/5"
    logger.info(f"\n[{step_num}] UPLOAD (Shorts)")
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
    parser.add_argument("task", choices=["research", "script", "audio", "video", "short", "upload", "full", "full-upload", "short-pipeline", "short-upload", "budget"])
    parser.add_argument("args", nargs="*", help="Task arguments")
    parser.add_argument("--niche", default="default", help="Content niche")
    parser.add_argument("--channel", help="Channel ID")
    parser.add_argument("--output", help="Output file path")

    args = parser.parse_args()
    result = None

    try:
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

        elif args.task == "budget":
            # Check current budget status
            config = load_budget_config()
            status = check_budget_status()
            result = {
                "success": True,
                "config": config,
                "status": status,
                "message": "Budget EXCEEDED - pipeline will not run" if status["exceeded"]
                          else "Budget WARNING - approaching limit" if status["warning"]
                          else "Budget OK - within limits"
            }
            # Print a nice summary
            print("\n" + "="*50)
            print("         BUDGET STATUS")
            print("="*50)
            print(f"\nDaily Limit:    ${config['daily_limit']:.2f}")
            print(f"Spent Today:    ${status['spent_today']:.4f}")
            print(f"Remaining:      ${status['remaining']:.4f}")
            print(f"Usage:          {status['usage_percent']:.1f}%")
            print(f"Warning at:     {config['warning_threshold']*100:.0f}%")
            print(f"Enforcement:    {'ENABLED' if config['enforce'] else 'DISABLED'}")
            print(f"\nStatus:         {result['message']}")
            print("="*50)
            return  # Skip the JSON output for budget check

        # Print result
        if result is not None:
            print("\n" + "="*50)
            print(json.dumps(result, indent=2, default=str))

    except BudgetExceededError as e:
        # Handle budget exceeded gracefully
        logger.error(f"Budget exceeded: {e}")
        result = {
            "success": False,
            "error": str(e),
            "error_type": "BudgetExceededError",
            "spent": e.spent,
            "limit": e.limit
        }
        print("\n" + "="*50)
        print("         BUDGET EXCEEDED")
        print("="*50)
        print(f"\nError: {e}")
        print(f"Spent today: ${e.spent:.4f}")
        print(f"Daily limit: ${e.limit:.2f}")
        print("\nPipeline stopped to prevent overspending.")
        print("To continue, either:")
        print("  1. Wait until tomorrow (budget resets daily)")
        print("  2. Increase daily_limit in config/config.yaml")
        print("  3. Set budget.enforce: false to disable enforcement")
        print("="*50)


if __name__ == "__main__":
    main()
