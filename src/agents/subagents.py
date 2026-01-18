"""
YouTube Automation Subagents

Specialized agents for different parts of the video production pipeline.
Each agent handles a specific task and can work independently or together.

Usage:
    from src.agents.subagents import AgentOrchestrator

    orchestrator = AgentOrchestrator()

    # Run full pipeline for a channel
    result = orchestrator.run_for_channel("money_blueprints")

    # Or run individual agents
    topics = orchestrator.research_agent.find_topics("passive income")
    script = orchestrator.script_agent.write_script(topics[0])
    video = orchestrator.production_agent.create_video(script)
"""

import os
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from loguru import logger

# Import our modules
from ..research.trends import TrendResearcher
from ..research.idea_generator import IdeaGenerator, ScoredIdea
from ..content.script_writer import ScriptWriter, VideoScript
from ..content.tts import TextToSpeech
from ..content.video_fast import FastVideoGenerator
from ..content.video_pro import ProVideoGenerator
from ..content.video_ultra import UltraVideoGenerator


@dataclass
class AgentResult:
    """Result from any agent operation."""
    success: bool
    agent: str
    task: str
    data: Any = None
    error: Optional[str] = None
    duration_seconds: float = 0


@dataclass
class VideoProject:
    """Tracks a video through the entire pipeline."""
    project_id: str
    channel: str
    niche: str
    status: str = "created"

    # Pipeline outputs
    topic: Optional[ScoredIdea] = None
    script: Optional[VideoScript] = None
    audio_file: Optional[str] = None
    video_file: Optional[str] = None
    thumbnail_file: Optional[str] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)


# ============================================================
# RESEARCH AGENT
# ============================================================
class ResearchAgent:
    """
    Agent specialized in finding trending topics and video ideas.

    Capabilities:
    - Analyze Google Trends
    - Find low-competition topics
    - Score ideas by viral potential
    """

    def __init__(self, ai_provider: str = "ollama", api_key: Optional[str] = None):
        self.name = "ResearchAgent"
        self.trend_researcher = TrendResearcher()
        self.idea_generator = IdeaGenerator(provider=ai_provider, api_key=api_key)
        logger.info(f"{self.name} initialized")

    def find_topics(self, niche: str, count: int = 5) -> List[ScoredIdea]:
        """Find trending topics for a niche."""
        logger.info(f"[{self.name}] Finding {count} topics for: {niche}")

        ideas = self.idea_generator.generate_ideas(niche=niche, count=count)

        if ideas:
            logger.success(f"[{self.name}] Found {len(ideas)} topic ideas")
            for i, idea in enumerate(ideas[:3], 1):
                logger.info(f"  {i}. {idea.title} (score: {idea.score})")
        else:
            logger.warning(f"[{self.name}] No topics found")

        return ideas

    def get_best_topic(self, niche: str) -> Optional[ScoredIdea]:
        """Get the single best topic for a niche."""
        ideas = self.find_topics(niche, count=5)
        return ideas[0] if ideas else None

    def analyze_competition(self, topic: str) -> Dict[str, Any]:
        """Analyze competition for a topic."""
        logger.info(f"[{self.name}] Analyzing competition for: {topic}")

        # Get trend data
        trends = self.trend_researcher.get_trending_topics(topic)

        return {
            "topic": topic,
            "trend_count": len(trends),
            "trending": any(t.trend_direction == "rising" for t in trends),
            "related_keywords": trends[0].related_queries if trends else []
        }


# ============================================================
# SCRIPT AGENT
# ============================================================
class ScriptAgent:
    """
    Agent specialized in writing video scripts.

    Capabilities:
    - Write engaging scripts
    - Optimize for retention
    - Create hooks and CTAs
    """

    def __init__(self, ai_provider: str = "ollama", api_key: Optional[str] = None):
        self.name = "ScriptAgent"
        self.writer = ScriptWriter(provider=ai_provider, api_key=api_key)
        logger.info(f"{self.name} initialized")

    def write_script(
        self,
        topic: ScoredIdea,
        duration_minutes: int = 5,
        style: str = "documentary",
        niche: str = "default"
    ) -> Optional[VideoScript]:
        """Write a complete video script."""
        logger.info(f"[{self.name}] Writing {duration_minutes}-min script for: {topic.title}")

        try:
            script = self.writer.generate_script(
                topic=topic.title,
                style=style,
                duration_minutes=duration_minutes,
                niche=niche
            )

            logger.success(f"[{self.name}] Script complete: {len(script.sections)} sections, ~{script.total_duration}s")
            return script

        except Exception as e:
            logger.error(f"[{self.name}] Script failed: {e}")
            return None

    def get_narration(self, script: VideoScript) -> str:
        """Extract full narration from script."""
        return self.writer.get_full_narration(script)

    def improve_hook(self, script: VideoScript) -> VideoScript:
        """Improve the opening hook of a script."""
        # TODO: Implement hook optimization
        return script


# ============================================================
# PRODUCTION AGENT
# ============================================================
class ProductionAgent:
    """
    Agent specialized in video production.

    Capabilities:
    - Generate voiceover audio
    - Create video from audio (basic, pro, or ULTRA with Ken Burns effects)
    - Generate thumbnails
    """

    def __init__(self, voice: str = "en-US-GuyNeural", use_ultra: bool = True):
        self.name = "ProductionAgent"
        self.tts = TextToSpeech(default_voice=voice)
        self.video_gen = FastVideoGenerator()
        self.use_ultra = use_ultra

        # Ultra video generator with Ken Burns, stock footage, animations
        if use_ultra:
            try:
                self.ultra_video_gen = UltraVideoGenerator()
                logger.info(f"{self.name} initialized with ULTRA video generator (Ken Burns + Stock)")
            except Exception as e:
                logger.warning(f"Ultra video generator unavailable: {e}")
                # Fallback to Pro
                try:
                    self.ultra_video_gen = ProVideoGenerator()
                    logger.info(f"{self.name} falling back to PRO video generator")
                except Exception as e2:
                    logger.warning(f"Pro video generator also unavailable: {e2}")
                    self.ultra_video_gen = None
                    self.use_ultra = False
        else:
            self.ultra_video_gen = None

        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        logger.info(f"{self.name} initialized with voice: {voice}")

    def set_voice(self, voice: str):
        """Change the TTS voice."""
        self.tts = TextToSpeech(default_voice=voice)
        logger.info(f"[{self.name}] Voice changed to: {voice}")

    async def create_audio(self, narration: str, filename: str) -> Optional[str]:
        """Generate voiceover audio."""
        logger.info(f"[{self.name}] Creating audio: {filename}")

        try:
            output_path = self.output_dir / filename
            await self.tts.generate(narration, str(output_path))

            if output_path.exists() and output_path.stat().st_size > 0:
                logger.success(f"[{self.name}] Audio created: {output_path}")
                return str(output_path)
            else:
                logger.error(f"[{self.name}] Audio file empty or missing")
                return None

        except Exception as e:
            logger.error(f"[{self.name}] Audio failed: {e}")
            return None

    def create_video(
        self,
        audio_file: str,
        title: str,
        filename: str
    ) -> Optional[str]:
        """Create video from audio."""
        logger.info(f"[{self.name}] Creating video: {filename}")

        try:
            output_path = self.output_dir / filename
            result = self.video_gen.create_video(
                audio_file=audio_file,
                output_file=str(output_path),
                title=title
            )

            if result:
                logger.success(f"[{self.name}] Video created: {result}")
            return result

        except Exception as e:
            logger.error(f"[{self.name}] Video failed: {e}")
            return None

    def create_thumbnail(self, title: str, filename: str, subtitle: str = None) -> Optional[str]:
        """Create video thumbnail."""
        logger.info(f"[{self.name}] Creating thumbnail: {filename}")

        try:
            output_path = self.output_dir / filename
            result = self.video_gen.create_thumbnail(
                output_file=str(output_path),
                title=title[:30],
                subtitle=subtitle
            )

            if result:
                logger.success(f"[{self.name}] Thumbnail created: {result}")
            return result

        except Exception as e:
            logger.error(f"[{self.name}] Thumbnail failed: {e}")
            return None

    async def produce_full_video(
        self,
        script: VideoScript,
        narration: str,
        project_id: str,
        niche: str = "default"
    ) -> Dict[str, Optional[str]]:
        """Produce complete video package (audio, video, thumbnail)."""
        logger.info(f"[{self.name}] Starting full production for: {script.title}")

        # Generate safe filename
        import re
        safe_name = re.sub(r'[^\w\s-]', '', script.title)[:40].replace(' ', '_')

        # Create audio
        audio_file = await self.create_audio(
            narration,
            f"{safe_name}_audio.mp3"
        )

        if not audio_file:
            return {"audio": None, "video": None, "thumbnail": None}

        # Create video (use ULTRA generator with Ken Burns if available)
        video_path = str(self.output_dir / f"{safe_name}.mp4")

        if self.use_ultra and self.ultra_video_gen:
            logger.info(f"[{self.name}] Using ULTRA video generator (Ken Burns + Stock Footage)")
            video_file = self.ultra_video_gen.create_video(
                audio_file=audio_file,
                script=script,
                output_file=video_path,
                niche=niche
            )
        else:
            # Fallback to basic video generator
            video_file = self.create_video(
                audio_file,
                script.title,
                f"{safe_name}.mp4"
            )

        # Create thumbnail
        thumb_path = str(self.output_dir / f"{safe_name}_thumb.png")

        if self.use_ultra and hasattr(self.ultra_video_gen, 'create_title_card'):
            # Use ULTRA title card as thumbnail
            self.ultra_video_gen.create_title_card(
                title=script.title,
                output_path=thumb_path,
                style=self.ultra_video_gen.NICHE_STYLES.get(niche, self.ultra_video_gen.NICHE_STYLES["default"]),
                subtitle=script.description[:50] if script.description else None
            )
            thumbnail_file = thumb_path if os.path.exists(thumb_path) else None
        elif self.use_ultra and hasattr(self.ultra_video_gen, 'create_thumbnail_pro'):
            thumbnail_file = self.ultra_video_gen.create_thumbnail_pro(
                title=script.title,
                output_file=thumb_path,
                niche=niche
            )
        else:
            thumbnail_file = self.create_thumbnail(
                script.title,
                f"{safe_name}_thumb.png",
                subtitle=script.description[:50] if script.description else None
            )

        return {
            "audio": audio_file,
            "video": video_file,
            "thumbnail": thumbnail_file
        }


# ============================================================
# UPLOAD AGENT
# ============================================================
class UploadAgent:
    """
    Agent specialized in YouTube uploads.

    Capabilities:
    - Upload videos to YouTube
    - Set metadata and thumbnails
    - Schedule uploads
    """

    def __init__(self):
        self.name = "UploadAgent"
        self._uploader = None
        logger.info(f"{self.name} initialized")

    @property
    def uploader(self):
        """Lazy load uploader."""
        if self._uploader is None:
            try:
                from ..youtube.uploader import YouTubeUploader
                self._uploader = YouTubeUploader()
            except Exception as e:
                logger.error(f"[{self.name}] Failed to initialize: {e}")
        return self._uploader

    def upload(
        self,
        video_file: str,
        title: str,
        description: str,
        tags: List[str],
        thumbnail_file: Optional[str] = None,
        privacy: str = "unlisted"
    ) -> Dict[str, Any]:
        """Upload video to YouTube."""
        logger.info(f"[{self.name}] Uploading: {title}")

        if not self.uploader:
            return {"success": False, "error": "Uploader not initialized"}

        try:
            result = self.uploader.upload_video(
                video_file=video_file,
                title=title,
                description=description,
                tags=tags,
                privacy=privacy,
                thumbnail_file=thumbnail_file
            )

            if result.success:
                logger.success(f"[{self.name}] Uploaded: {result.video_url}")
                return {
                    "success": True,
                    "video_id": result.video_id,
                    "video_url": result.video_url
                }
            else:
                logger.error(f"[{self.name}] Upload failed: {result.error}")
                return {"success": False, "error": result.error}

        except Exception as e:
            logger.error(f"[{self.name}] Upload error: {e}")
            return {"success": False, "error": str(e)}


# ============================================================
# AGENT ORCHESTRATOR
# ============================================================
class AgentOrchestrator:
    """
    Master orchestrator that coordinates all agents.

    Usage:
        orchestrator = AgentOrchestrator()
        result = orchestrator.create_video("passive income", channel="money_blueprints")
    """

    def __init__(
        self,
        ai_provider: str = "ollama",
        api_key: Optional[str] = None
    ):
        self.ai_provider = ai_provider
        self.api_key = api_key

        # Initialize agents
        self.research_agent = ResearchAgent(ai_provider, api_key)
        self.script_agent = ScriptAgent(ai_provider, api_key)
        self.production_agent = ProductionAgent()
        self.upload_agent = UploadAgent()

        # Track projects
        self.projects: Dict[str, VideoProject] = {}

        logger.info("AgentOrchestrator initialized with all agents")

    def create_video(
        self,
        niche: str,
        channel: Optional[str] = None,
        upload: bool = False,
        privacy: str = "unlisted"
    ) -> VideoProject:
        """
        Create a complete video using all agents.

        Args:
            niche: Topic niche (e.g., "passive income")
            channel: Channel ID from channels.yaml
            upload: Whether to upload to YouTube
            privacy: YouTube privacy setting

        Returns:
            VideoProject with all outputs
        """
        # Create project
        project_id = f"{niche.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        project = VideoProject(
            project_id=project_id,
            channel=channel or "default",
            niche=niche
        )
        self.projects[project_id] = project

        logger.info(f"")
        logger.info(f"{'='*60}")
        logger.info(f"  CREATING VIDEO: {niche}")
        logger.info(f"{'='*60}")

        try:
            # Step 1: Research
            logger.info(f"\n[STEP 1/4] Research Agent finding topics...")
            project.status = "researching"
            topic = self.research_agent.get_best_topic(niche)

            if not topic:
                project.status = "failed"
                project.errors.append("No topics found")
                return project

            project.topic = topic
            logger.info(f"  Selected: {topic.title}")

            # Determine niche from channel
            niche = "default"
            if channel:
                niche_map = {
                    "money_blueprints": "finance",
                    "mind_unlocked": "psychology",
                    "untold_stories": "storytelling"
                }
                niche = niche_map.get(channel, "default")

            # Step 2: Script (5-6 minute videos)
            logger.info(f"\n[STEP 2/4] Script Agent writing script...")
            project.status = "scripting"
            script = self.script_agent.write_script(
                topic,
                duration_minutes=5,
                style="documentary",
                niche=niche
            )

            if not script:
                project.status = "failed"
                project.errors.append("Script generation failed")
                return project

            project.script = script
            narration = self.script_agent.get_narration(script)

            # Step 3: Production
            logger.info(f"\n[STEP 3/4] Production Agent creating video...")
            project.status = "producing"

            # Run async production
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.production_agent.produce_full_video(script, narration, project_id, niche=niche)
            )
            loop.close()

            project.audio_file = result.get("audio")
            project.video_file = result.get("video")
            project.thumbnail_file = result.get("thumbnail")

            if not project.video_file:
                project.status = "failed"
                project.errors.append("Video production failed")
                return project

            # Step 4: Upload (optional)
            if upload:
                logger.info(f"\n[STEP 4/4] Upload Agent uploading to YouTube...")
                project.status = "uploading"

                upload_result = self.upload_agent.upload(
                    video_file=project.video_file,
                    title=script.title,
                    description=script.description,
                    tags=script.tags,
                    thumbnail_file=project.thumbnail_file,
                    privacy=privacy
                )

                if not upload_result.get("success"):
                    project.errors.append(f"Upload failed: {upload_result.get('error')}")
            else:
                logger.info(f"\n[STEP 4/4] Skipping upload (upload=False)")

            # Complete
            project.status = "completed"
            project.completed_at = datetime.now()

            logger.info(f"")
            logger.info(f"{'='*60}")
            logger.success(f"  VIDEO COMPLETE!")
            logger.info(f"  Title: {script.title}")
            logger.info(f"  Video: {project.video_file}")
            logger.info(f"{'='*60}")

            return project

        except Exception as e:
            project.status = "failed"
            project.errors.append(str(e))
            logger.error(f"Pipeline failed: {e}")
            return project

    def create_batch(
        self,
        niches: List[str],
        channel: Optional[str] = None
    ) -> List[VideoProject]:
        """Create multiple videos in batch."""
        results = []

        for i, niche in enumerate(niches, 1):
            logger.info(f"\n[BATCH {i}/{len(niches)}]")
            project = self.create_video(niche, channel)
            results.append(project)

        # Summary
        successful = sum(1 for p in results if p.status == "completed")
        logger.info(f"\nBatch complete: {successful}/{len(niches)} successful")

        return results


# ============================================================
# QUICK FUNCTIONS
# ============================================================
def quick_video(niche: str, upload: bool = False) -> VideoProject:
    """Quick function to create a video."""
    orchestrator = AgentOrchestrator()
    return orchestrator.create_video(niche, upload=upload)


def quick_batch(niches: List[str]) -> List[VideoProject]:
    """Quick function to create multiple videos."""
    orchestrator = AgentOrchestrator()
    return orchestrator.create_batch(niches)


# Example usage
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  SUBAGENT SYSTEM TEST")
    print("="*60 + "\n")

    # Test the orchestrator
    orchestrator = AgentOrchestrator()

    # Create a video
    project = orchestrator.create_video("money saving tips")

    print(f"\nProject Status: {project.status}")
    if project.video_file:
        print(f"Video: {project.video_file}")
