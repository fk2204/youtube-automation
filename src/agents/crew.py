"""
CrewAI Multi-Agent System for YouTube Automation

Orchestrates multiple AI agents to handle the complete video pipeline:
- Researcher: Finds trending topics
- Scriptwriter: Creates engaging scripts
- Producer: Generates videos
- Publisher: Uploads to YouTube

Usage:
    crew = YouTubeCrew(provider="ollama")
    result = crew.run_pipeline(niche="python tutorials")
"""

import os
import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
from loguru import logger

# Try to import CrewAI (optional dependency)
try:
    from crewai import Agent, Task, Crew, Process
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    logger.warning("CrewAI not installed. Using simplified pipeline.")


@dataclass
class PipelineResult:
    """Result of the video pipeline."""
    success: bool
    video_id: Optional[str]
    video_url: Optional[str]
    video_file: Optional[str]
    title: str
    topic: str
    error: Optional[str]


class YouTubeCrew:
    """
    Multi-agent crew for YouTube video automation.

    If CrewAI is installed, uses full agent orchestration.
    Otherwise, falls back to a simplified sequential pipeline.
    """

    def __init__(
        self,
        provider: str = "ollama",
        api_key: Optional[str] = None,
        output_dir: str = "output"
    ):
        """
        Initialize the YouTube crew.

        Args:
            provider: AI provider for agents (ollama, groq, claude, etc.)
            api_key: API key for cloud providers
            output_dir: Directory for output files
        """
        self.provider = provider
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"YouTubeCrew initialized with provider: {provider}")

        if CREWAI_AVAILABLE:
            self._setup_crewai_agents()
        else:
            self._setup_simple_pipeline()

    def _setup_crewai_agents(self):
        """Setup CrewAI agents."""
        logger.info("Setting up CrewAI agents...")

        # Import tools
        from ..research.idea_generator import IdeaGenerator
        from ..content.script_writer import ScriptWriter
        from ..content.tts import TextToSpeech
        from ..content.video_assembler import VideoAssembler

        # Store components for agents
        self.idea_gen = IdeaGenerator(provider=self.provider, api_key=self.api_key)
        self.script_writer = ScriptWriter(provider=self.provider, api_key=self.api_key)
        self.tts = TextToSpeech()
        self.video_assembler = VideoAssembler()

        # Define agents
        self.researcher = Agent(
            role="Trend Researcher",
            goal="Find viral video topics with high demand and low competition",
            backstory="""You are an expert content strategist who specializes in
            finding trending topics that will perform well on YouTube. You analyze
            Google Trends, Reddit discussions, and competitor content to identify
            opportunities.""",
            verbose=True,
            allow_delegation=False
        )

        self.scriptwriter = Agent(
            role="Script Writer",
            goal="Write engaging, educational YouTube scripts that keep viewers watching",
            backstory="""You are a professional YouTube scriptwriter with years of
            experience creating viral educational content. You know how to hook
            viewers in the first 10 seconds and keep them engaged throughout.""",
            verbose=True,
            allow_delegation=False
        )

        self.producer = Agent(
            role="Video Producer",
            goal="Create high-quality tutorial videos with professional narration",
            backstory="""You are a video production expert who creates polished
            educational content. You handle text-to-speech narration, video
            assembly, and thumbnail creation.""",
            verbose=True,
            allow_delegation=False
        )

        self.publisher = Agent(
            role="YouTube Publisher",
            goal="Upload and optimize videos for maximum reach and engagement",
            backstory="""You are a YouTube SEO expert who knows how to optimize
            titles, descriptions, and tags for maximum discoverability. You
            understand the YouTube algorithm.""",
            verbose=True,
            allow_delegation=False
        )

    def _setup_simple_pipeline(self):
        """Setup simple sequential pipeline (no CrewAI)."""
        logger.info("Setting up simple pipeline (CrewAI not available)...")

        from ..research.idea_generator import IdeaGenerator
        from ..content.script_writer import ScriptWriter
        from ..content.tts import TextToSpeech
        from ..content.video_assembler import VideoAssembler

        self.idea_gen = IdeaGenerator(provider=self.provider, api_key=self.api_key)
        self.script_writer = ScriptWriter(provider=self.provider, api_key=self.api_key)
        self.tts = TextToSpeech()
        self.video_assembler = VideoAssembler()

    def run_pipeline(
        self,
        niche: str,
        upload: bool = False,
        privacy: str = "unlisted"
    ) -> PipelineResult:
        """
        Run the complete video creation pipeline.

        Args:
            niche: Topic niche (e.g., "python programming")
            upload: Whether to upload to YouTube
            privacy: YouTube privacy setting

        Returns:
            PipelineResult with video details
        """
        logger.info(f"Starting pipeline for niche: {niche}")

        try:
            # Step 1: Research
            logger.info("Step 1/4: Researching topics...")
            idea = self.idea_gen.get_best_idea(niche)

            if not idea:
                return PipelineResult(
                    success=False,
                    video_id=None,
                    video_url=None,
                    video_file=None,
                    title="",
                    topic=niche,
                    error="No video ideas generated"
                )

            logger.info(f"Selected topic: {idea.title}")

            # Step 2: Script
            logger.info("Step 2/4: Writing script...")
            script = self.script_writer.generate_script(
                topic=idea.title,
                duration_minutes=10
            )

            logger.info(f"Script generated: {len(script.sections)} sections")

            # Step 3: Create Video
            logger.info("Step 3/4: Creating video...")
            result = asyncio.run(self._create_video(script, idea.title))

            if not result["success"]:
                return PipelineResult(
                    success=False,
                    video_id=None,
                    video_url=None,
                    video_file=None,
                    title=script.title,
                    topic=idea.title,
                    error=result.get("error", "Video creation failed")
                )

            video_file = result["video_file"]
            thumbnail_file = result.get("thumbnail_file")

            # Step 4: Upload (optional)
            video_id = None
            video_url = None

            if upload:
                logger.info("Step 4/4: Uploading to YouTube...")
                from ..youtube.uploader import YouTubeUploader

                uploader = YouTubeUploader()
                upload_result = uploader.upload_video(
                    video_file=video_file,
                    title=script.title,
                    description=script.description,
                    tags=script.tags,
                    privacy=privacy,
                    thumbnail_file=thumbnail_file
                )

                if upload_result.success:
                    video_id = upload_result.video_id
                    video_url = upload_result.video_url
                    logger.success(f"Uploaded: {video_url}")
                else:
                    logger.warning(f"Upload failed: {upload_result.error}")
            else:
                logger.info("Step 4/4: Skipping upload (upload=False)")

            return PipelineResult(
                success=True,
                video_id=video_id,
                video_url=video_url,
                video_file=video_file,
                title=script.title,
                topic=idea.title,
                error=None
            )

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return PipelineResult(
                success=False,
                video_id=None,
                video_url=None,
                video_file=None,
                title="",
                topic=niche,
                error=str(e)
            )

    async def _create_video(self, script, topic: str) -> Dict[str, Any]:
        """Create video from script."""
        import re

        # Generate safe filename
        safe_title = re.sub(r'[^\w\s-]', '', script.title)[:50]
        safe_title = safe_title.replace(' ', '_')

        audio_file = self.output_dir / f"{safe_title}_audio.mp3"
        video_file = self.output_dir / f"{safe_title}.mp4"
        thumbnail_file = self.output_dir / f"{safe_title}_thumb.png"

        try:
            # Generate audio
            narration = self.script_writer.get_full_narration(script)
            await self.tts.generate(narration, str(audio_file))

            # Create video
            self.video_assembler.create_video_from_audio(
                audio_file=str(audio_file),
                output_file=str(video_file),
                title=script.title
            )

            # Create thumbnail
            self.video_assembler.create_thumbnail(
                output_file=str(thumbnail_file),
                title=script.title[:30],
                subtitle=topic[:40]
            )

            return {
                "success": True,
                "video_file": str(video_file),
                "audio_file": str(audio_file),
                "thumbnail_file": str(thumbnail_file)
            }

        except Exception as e:
            logger.error(f"Video creation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def run_daily(self, channels: List[Dict[str, Any]]) -> List[PipelineResult]:
        """
        Run daily video creation for multiple channels.

        Args:
            channels: List of channel configs from channels.yaml

        Returns:
            List of PipelineResult for each channel
        """
        results = []

        for channel in channels:
            if not channel.get("enabled", True):
                continue

            niche = channel.get("settings", {}).get("niche", "tutorials")
            logger.info(f"Processing channel: {channel.get('name', 'Unknown')}")

            result = self.run_pipeline(
                niche=niche,
                upload=True,
                privacy=channel.get("settings", {}).get("default_privacy", "unlisted")
            )

            results.append(result)

        return results


# Simple pipeline for when CrewAI is not installed
class SimplePipeline:
    """Simplified pipeline without CrewAI dependency."""

    def __init__(self, provider: str = "ollama", api_key: Optional[str] = None):
        self.crew = YouTubeCrew(provider=provider, api_key=api_key)

    def run(self, niche: str, upload: bool = False) -> PipelineResult:
        return self.crew.run_pipeline(niche=niche, upload=upload)


# Example usage
if __name__ == "__main__":
    print("\n" + "="*60)
    print("YOUTUBE AUTOMATION CREW")
    print("="*60 + "\n")

    crew = YouTubeCrew(provider="ollama")

    print("Running pipeline for: Python programming tutorials")
    print("(This will use Ollama for AI - make sure it's running)\n")

    result = crew.run_pipeline(
        niche="python programming tutorials",
        upload=False  # Don't upload, just create video
    )

    if result.success:
        print(f"\nSuccess!")
        print(f"  Title: {result.title}")
        print(f"  Video: {result.video_file}")
    else:
        print(f"\nFailed: {result.error}")
