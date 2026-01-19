"""
Hybrid YouTube Shorts Generator

Uses Pika Labs AI for intro/outro clips and stock footage for main content.
Cost-effective approach: ~$0.40-0.60 per Short instead of $1.20+ for full AI.

Structure:
- Intro (5s): Pika AI - Eye-catching hook
- Middle (15-20s): Stock footage - Main content
- Outro (5s): Pika AI - Call-to-action / Loop-friendly ending

Usage:
    from src.content.shorts_hybrid import HybridShortsGenerator

    generator = HybridShortsGenerator()
    await generator.create_hybrid_short(
        audio_file="narration.mp3",
        script=script_object,
        output_file="short.mp4",
        niche="psychology"
    )
"""

import os
import asyncio
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
_env_paths = [
    Path(__file__).parent.parent.parent / "config" / ".env",
    Path.cwd() / "config" / ".env",
]
for _env_path in _env_paths:
    if _env_path.exists():
        load_dotenv(_env_path)
        break


@dataclass
class HybridShortResult:
    """Result from hybrid short generation."""
    success: bool
    output_path: Optional[str] = None
    duration: Optional[float] = None
    pika_cost: float = 0.0
    intro_path: Optional[str] = None
    outro_path: Optional[str] = None
    error: Optional[str] = None


class HybridShortsGenerator:
    """
    Hybrid YouTube Shorts generator combining Pika AI and stock footage.

    Cost breakdown per 30s Short:
    - Intro (5s Pika 720p): $0.20
    - Middle (20s stock): $0.00
    - Outro (5s Pika 720p): $0.20
    - Total: $0.40
    """

    # Resolution for Shorts
    WIDTH = 1080
    HEIGHT = 1920
    FPS = 30

    # Pika clip durations
    INTRO_DURATION = 5  # seconds
    OUTRO_DURATION = 5  # seconds

    # Niche-specific Pika prompts
    NICHE_PROMPTS = {
        "finance": {
            "intro": [
                "Dramatic close-up of gold coins falling in slow motion, dark background, cinematic lighting",
                "Stock market graph going up with green glow, futuristic digital aesthetic",
                "Luxury lifestyle montage, wealth symbols, golden hour lighting",
                "Money bills floating in air, slow motion, dramatic spotlight",
                "Cityscape with financial district skyscrapers at night, neon reflections",
            ],
            "outro": [
                "Subscribe button animation with coins exploding, dark background",
                "Person walking towards bright future, silhouette, inspirational",
                "Upward trending arrow made of light, motivational ending",
                "Golden sunrise over city, new beginnings, hope",
                "Loop-friendly: coins starting to fall (connects to intro)",
            ]
        },
        "psychology": {
            "intro": [
                "Close-up of human eye with neural network reflections, mysterious atmosphere",
                "Abstract brain visualization with glowing synapses, dark purple aesthetic",
                "Person deep in thought, dramatic side lighting, contemplative mood",
                "Surreal dreamscape with floating objects, psychological imagery",
                "Mirror reflection showing different emotions, split personality concept",
            ],
            "outro": [
                "Mind expanding visualization, enlightenment imagery",
                "Person having an aha moment, lightbulb effect, revelation",
                "Neural pathways forming connections, learning visualization",
                "Peaceful meditation scene, inner peace achieved",
                "Loop-friendly: eye opening (connects to intro eye shot)",
            ]
        },
        "storytelling": {
            "intro": [
                "Old book opening with magical dust particles, cinematic lighting",
                "Mysterious figure walking into fog, dramatic silhouette",
                "Time-lapse of day to night, epic landscape, story beginning",
                "Vintage film grain effect, nostalgic opening shot",
                "Map unfolding with adventure route appearing, exploration theme",
                # True crime / mystery style intros
                "Crime scene tape fluttering in the wind, dark atmospheric lighting, noir aesthetic",
                "Shadowy figure in doorway, dramatic backlighting, suspenseful mood",
                "Old newspaper headlines spinning into frame, vintage documentary style",
                "Clock hands moving rapidly in reverse, time-travel flashback effect",
                "Crumbling corporate building facade, rise and fall imagery, documentary feel",
            ],
            "outro": [
                "Book closing gently, satisfying ending, warm lighting",
                "Character walking into sunset, journey complete",
                "The end title card with elegant typography, classic style",
                "Stars appearing in night sky, peaceful conclusion",
                "Loop-friendly: book about to open (connects to intro)",
                # True crime / mystery style outros
                "Prison bars closing with dramatic clang, justice served imagery",
                "Question mark fading into darkness, unresolved mystery vibe",
                "Gavel striking with dramatic slow motion, verdict delivered",
                "Candle flame flickering then extinguishing, story ends, vertical format",
                "Loop-friendly: shadowy figure beginning to appear (connects to intro)",
            ]
        },
        "default": {
            "intro": [
                "Cinematic slow motion abstract visuals, eye-catching colors",
                "Dynamic light rays through atmosphere, dramatic opening",
                "Mysterious silhouette reveal, building anticipation",
            ],
            "outro": [
                "Satisfying visual conclusion, subscribe hint",
                "Uplifting ending shot, positive energy",
                "Loop-friendly seamless transition back to start",
            ]
        }
    }

    def __init__(self):
        """Initialize hybrid shorts generator."""
        self.ffmpeg = self._find_ffmpeg()
        self.pika = None
        self.stock = None

        # Initialize Pika
        try:
            from .video_pika import PikaVideoGenerator
            self.pika = PikaVideoGenerator()
            if self.pika.api_key:
                logger.info("HybridShortsGenerator: Pika Labs ready")
            else:
                self.pika = None
                logger.warning("Pika API key not configured")
        except ImportError:
            logger.warning("Pika Labs not available")
        except Exception as e:
            logger.warning(f"Pika initialization failed: {e}")

        # Initialize stock footage
        try:
            from .stock_footage import StockFootageProvider
            self.stock = StockFootageProvider()
            logger.info("HybridShortsGenerator: Stock footage ready")
        except ImportError:
            try:
                from .multi_stock import MultiStockProvider
                self.stock = MultiStockProvider()
                logger.info("HybridShortsGenerator: Multi-stock footage ready")
            except ImportError:
                logger.warning("Stock footage provider not available")

        # Temp directory
        self.temp_dir = Path(tempfile.gettempdir()) / "hybrid_shorts"
        self.temp_dir.mkdir(exist_ok=True)

        logger.info("HybridShortsGenerator initialized")

    def _find_ffmpeg(self) -> Optional[str]:
        """Find FFmpeg executable."""
        if shutil.which("ffmpeg"):
            return "ffmpeg"

        common_paths = [
            os.path.expanduser("~\\AppData\\Local\\Microsoft\\WinGet\\Packages"),
            "C:\\ffmpeg\\bin",
            "C:\\Program Files\\ffmpeg\\bin",
        ]

        for base_path in common_paths:
            if os.path.exists(base_path):
                for root, dirs, files in os.walk(base_path):
                    if "ffmpeg.exe" in files:
                        return os.path.join(root, "ffmpeg.exe")

        return None

    def _get_prompt_for_niche(self, niche: str, clip_type: str, topic: str = "") -> str:
        """Get a Pika prompt for the specified niche and clip type."""
        prompts = self.NICHE_PROMPTS.get(niche, self.NICHE_PROMPTS["default"])
        prompt_list = prompts.get(clip_type, prompts.get("intro", []))

        base_prompt = random.choice(prompt_list) if prompt_list else "Cinematic opening shot"

        # Add topic context if provided
        if topic and clip_type == "intro":
            base_prompt = f"{base_prompt}, related to {topic[:50]}"

        # Add vertical format specification
        base_prompt += ", vertical 9:16 format, mobile-optimized, high quality"

        return base_prompt

    async def generate_pika_intro(
        self,
        niche: str,
        topic: str = "",
        custom_prompt: Optional[str] = None
    ) -> Optional[str]:
        """Generate an AI intro clip with Pika Labs."""
        if not self.pika:
            logger.error("Pika not available for intro generation")
            return None

        prompt = custom_prompt or self._get_prompt_for_niche(niche, "intro", topic)
        output_path = str(self.temp_dir / f"intro_{os.urandom(4).hex()}.mp4")

        logger.info(f"Generating Pika intro: {prompt[:60]}...")

        try:
            result = await self.pika.generate_short_clip(
                prompt=prompt,
                output_file=output_path,
                duration=self.INTRO_DURATION,
                aspect_ratio="9:16"
            )

            if result.success and result.local_path:
                logger.success(f"Pika intro generated: {result.local_path}")
                return result.local_path
            else:
                logger.error(f"Pika intro failed: {result.error}")
                return None

        except Exception as e:
            logger.error(f"Pika intro generation error: {e}")
            return None

    async def generate_pika_outro(
        self,
        niche: str,
        topic: str = "",
        custom_prompt: Optional[str] = None
    ) -> Optional[str]:
        """Generate an AI outro clip with Pika Labs."""
        if not self.pika:
            logger.error("Pika not available for outro generation")
            return None

        prompt = custom_prompt or self._get_prompt_for_niche(niche, "outro", topic)
        output_path = str(self.temp_dir / f"outro_{os.urandom(4).hex()}.mp4")

        logger.info(f"Generating Pika outro: {prompt[:60]}...")

        try:
            result = await self.pika.generate_short_clip(
                prompt=prompt,
                output_file=output_path,
                duration=self.OUTRO_DURATION,
                aspect_ratio="9:16"
            )

            if result.success and result.local_path:
                logger.success(f"Pika outro generated: {result.local_path}")
                return result.local_path
            else:
                logger.error(f"Pika outro failed: {result.error}")
                return None

        except Exception as e:
            logger.error(f"Pika outro generation error: {e}")
            return None

    def get_audio_duration(self, audio_file: str) -> float:
        """Get audio file duration in seconds."""
        if not self.ffmpeg:
            return 30.0  # Default

        try:
            ffprobe = self.ffmpeg.replace("ffmpeg", "ffprobe")
            cmd = [
                ffprobe, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_file
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return float(result.stdout.strip())
        except Exception:
            return 30.0

    def _convert_to_vertical(self, input_path: str, output_path: str, duration: float) -> Optional[str]:
        """Convert a clip to vertical 9:16 format with Ken Burns effect."""
        if not self.ffmpeg:
            return None

        try:
            # Ken Burns parameters
            zoom_start = random.uniform(1.0, 1.1)
            zoom_end = random.uniform(1.1, 1.25)

            cmd = [
                self.ffmpeg, "-y",
                "-i", input_path,
                "-t", str(duration),
                "-vf", (
                    f"scale=-2:2160,"
                    f"zoompan=z='if(lte(zoom,{zoom_start}),{zoom_start},max({zoom_end},zoom-0.001))':"
                    f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                    f"d={int(duration * self.FPS)}:s={self.WIDTH}x{self.HEIGHT}:fps={self.FPS},"
                    f"setsar=1"
                ),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-an",  # No audio for segments
                output_path
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=120)

            if result.returncode == 0 and os.path.exists(output_path):
                return output_path
            return None

        except Exception as e:
            logger.warning(f"Vertical conversion failed: {e}")
            return None

    def _create_gradient_clip(self, output_path: str, duration: float, niche: str) -> Optional[str]:
        """Create a gradient background clip as fallback."""
        if not self.ffmpeg:
            return None

        # Niche colors
        colors = {
            "finance": ("0a1628", "1a3a5c"),
            "psychology": ("150520", "2a1040"),
            "storytelling": ("1a0a0a", "3a2020"),
            "default": ("0a0a14", "1a1a2e"),
        }

        c1, c2 = colors.get(niche, colors["default"])

        try:
            cmd = [
                self.ffmpeg, "-y",
                "-f", "lavfi",
                "-i", f"color=c=#{c1}:s={self.WIDTH}x{self.HEIGHT}:d={duration}",
                "-vf", f"format=rgb24,geq=r='r(X,Y)*({self.HEIGHT}-Y)/{self.HEIGHT}+128*Y/{self.HEIGHT}':g='g(X,Y)*({self.HEIGHT}-Y)/{self.HEIGHT}+64*Y/{self.HEIGHT}':b='b(X,Y)*({self.HEIGHT}-Y)/{self.HEIGHT}+180*Y/{self.HEIGHT}'",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-t", str(duration),
                output_path
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=60)

            if result.returncode == 0 and os.path.exists(output_path):
                return output_path
            return None

        except Exception as e:
            logger.warning(f"Gradient clip creation failed: {e}")
            return None

    async def create_hybrid_short(
        self,
        audio_file: str,
        output_file: str,
        niche: str = "default",
        topic: str = "",
        script=None,
        intro_prompt: Optional[str] = None,
        outro_prompt: Optional[str] = None,
        skip_pika_on_error: bool = True
    ) -> HybridShortResult:
        """
        Create a hybrid YouTube Short with Pika intro/outro and stock middle.

        Args:
            audio_file: Path to narration audio
            output_file: Output video path
            niche: Content niche (finance, psychology, storytelling)
            topic: Video topic for context
            script: Optional script object with title/sections
            intro_prompt: Custom Pika prompt for intro
            outro_prompt: Custom Pika prompt for outro
            skip_pika_on_error: If True, use stock footage if Pika fails

        Returns:
            HybridShortResult with paths and cost info
        """
        if not self.ffmpeg:
            return HybridShortResult(success=False, error="FFmpeg not available")

        if not os.path.exists(audio_file):
            return HybridShortResult(success=False, error=f"Audio file not found: {audio_file}")

        logger.info(f"Creating hybrid Short: {output_file}")
        logger.info(f"Niche: {niche}, Topic: {topic[:50] if topic else 'N/A'}")

        # Get topic from script if not provided
        if not topic and script:
            topic = getattr(script, 'title', str(script)[:50])

        # Get audio duration
        audio_duration = self.get_audio_duration(audio_file)
        logger.info(f"Audio duration: {audio_duration:.1f}s")

        # Calculate segment durations
        middle_duration = audio_duration - self.INTRO_DURATION - self.OUTRO_DURATION
        if middle_duration < 5:
            # Short audio - use minimal Pika (just intro)
            middle_duration = audio_duration - self.INTRO_DURATION
            use_outro = False
            logger.info("Short audio - using intro only (no outro)")
        else:
            use_outro = True

        pika_cost = 0.0
        segment_files = []

        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Generate Pika intro (or fallback)
            logger.info("Step 1/3: Generating intro...")
            intro_path = None

            if self.pika:
                intro_path = await self.generate_pika_intro(niche, topic, intro_prompt)
                if intro_path:
                    pika_cost += 0.20
                    segment_files.append(intro_path)

            if not intro_path and skip_pika_on_error:
                logger.warning("Pika intro failed, using gradient fallback")
                fallback_intro = str(self.temp_dir / "fallback_intro.mp4")
                intro_path = self._create_gradient_clip(fallback_intro, self.INTRO_DURATION, niche)
                if intro_path:
                    segment_files.append(intro_path)

            # 2. Generate middle content from stock footage
            logger.info("Step 2/3: Creating middle content from stock...")
            middle_segments = await self._create_middle_content(
                topic, niche, middle_duration, script
            )
            segment_files.extend(middle_segments)

            # 3. Generate Pika outro (or fallback)
            if use_outro:
                logger.info("Step 3/3: Generating outro...")
                outro_path = None

                if self.pika:
                    outro_path = await self.generate_pika_outro(niche, topic, outro_prompt)
                    if outro_path:
                        pika_cost += 0.20
                        segment_files.append(outro_path)

                if not outro_path and skip_pika_on_error:
                    logger.warning("Pika outro failed, using gradient fallback")
                    fallback_outro = str(self.temp_dir / "fallback_outro.mp4")
                    outro_path = self._create_gradient_clip(fallback_outro, self.OUTRO_DURATION, niche)
                    if outro_path:
                        segment_files.append(outro_path)

            if not segment_files:
                return HybridShortResult(success=False, error="No video segments created")

            # 4. Concatenate all segments
            logger.info("Concatenating segments...")
            concat_video = str(self.temp_dir / "concat_video.mp4")

            if not self._concatenate_clips(segment_files, concat_video):
                return HybridShortResult(success=False, error="Failed to concatenate clips")

            # 5. Add audio
            logger.info("Adding audio...")
            if not self._add_audio(concat_video, audio_file, output_file):
                return HybridShortResult(success=False, error="Failed to add audio")

            logger.success(f"Hybrid Short created: {output_file}")
            logger.info(f"Pika cost: ${pika_cost:.2f}")

            return HybridShortResult(
                success=True,
                output_path=output_file,
                duration=audio_duration,
                pika_cost=pika_cost,
                intro_path=intro_path if intro_path else None,
                outro_path=outro_path if use_outro and outro_path else None
            )

        except Exception as e:
            logger.error(f"Hybrid short creation failed: {e}")
            return HybridShortResult(success=False, error=str(e))

        finally:
            # Cleanup temp files
            self._cleanup_temp_files(segment_files)

    async def _create_middle_content(
        self,
        topic: str,
        niche: str,
        duration: float,
        script=None
    ) -> List[str]:
        """Create middle content from stock footage."""
        segment_files = []
        segment_duration = 2.5  # Fast pacing for Shorts
        num_segments = int(duration / segment_duration) + 1

        # Get stock footage
        downloaded_clips = []

        if self.stock:
            try:
                # Build search terms
                search_terms = [topic] if topic else ["abstract", "cinematic"]

                if script:
                    sections = getattr(script, 'sections', [])
                    for s in sections[:3]:
                        if hasattr(s, 'keywords') and s.keywords:
                            search_terms.extend(s.keywords[:2])

                # Search and download
                if hasattr(self.stock, 'get_clips_for_topic'):
                    clips = self.stock.get_clips_for_topic(
                        topic=topic,
                        niche=niche,
                        count=num_segments + 2,
                        min_total_duration=int(duration * 1.5)
                    )
                    for clip in clips:
                        path = self.stock.download_clip(clip)
                        if path:
                            downloaded_clips.append(path)
                else:
                    # Basic search
                    for term in search_terms[:3]:
                        if hasattr(self.stock, 'search_videos'):
                            found = self.stock.search_videos(term, count=3)
                            for clip in found:
                                if hasattr(self.stock, 'download_video'):
                                    path = self.temp_dir / f"stock_{len(downloaded_clips)}.mp4"
                                    result = self.stock.download_video(clip, str(path))
                                    if result:
                                        downloaded_clips.append(result)

                logger.info(f"Downloaded {len(downloaded_clips)} stock clips for middle content")

            except Exception as e:
                logger.warning(f"Stock footage fetch failed: {e}")

        # Create segments from downloaded clips
        current_time = 0.0
        clip_index = 0
        segment_num = 0

        while current_time < duration:
            remaining = duration - current_time
            seg_dur = min(segment_duration, remaining)

            if seg_dur < 0.5:
                break

            segment_path = str(self.temp_dir / f"middle_{segment_num}.mp4")

            if clip_index < len(downloaded_clips):
                # Convert stock clip to vertical format
                result = self._convert_to_vertical(
                    downloaded_clips[clip_index],
                    segment_path,
                    seg_dur
                )
                clip_index += 1

                if result:
                    segment_files.append(result)
                    current_time += seg_dur
                    segment_num += 1
                    continue

            # Fallback: gradient
            result = self._create_gradient_clip(segment_path, seg_dur, niche)
            if result:
                segment_files.append(result)

            current_time += seg_dur
            segment_num += 1

            # Loop clips if needed
            if clip_index >= len(downloaded_clips) and downloaded_clips:
                clip_index = 0

        return segment_files

    def _concatenate_clips(self, clip_files: List[str], output_path: str) -> bool:
        """Concatenate video clips using FFmpeg."""
        if not clip_files:
            return False

        try:
            # Create concat file
            concat_file = self.temp_dir / "concat_list.txt"
            with open(concat_file, "w") as f:
                for clip in clip_files:
                    f.write(f"file '{clip}'\n")

            cmd = [
                self.ffmpeg, "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-an",
                output_path
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=300)

            return result.returncode == 0 and os.path.exists(output_path)

        except Exception as e:
            logger.error(f"Concatenation failed: {e}")
            return False

    def _add_audio(self, video_path: str, audio_path: str, output_path: str) -> bool:
        """Add audio to video."""
        try:
            cmd = [
                self.ffmpeg, "-y",
                "-i", video_path,
                "-i", audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                output_path
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=120)

            return result.returncode == 0 and os.path.exists(output_path)

        except Exception as e:
            logger.error(f"Audio addition failed: {e}")
            return False

    def _cleanup_temp_files(self, files: List[str]):
        """Clean up temporary files."""
        for f in files:
            try:
                if f and os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass


# Convenience function
async def create_hybrid_short(
    audio_file: str,
    output_file: str,
    niche: str = "default",
    topic: str = ""
) -> HybridShortResult:
    """Quick function to create a hybrid short."""
    generator = HybridShortsGenerator()
    return await generator.create_hybrid_short(
        audio_file=audio_file,
        output_file=output_file,
        niche=niche,
        topic=topic
    )


# Test
if __name__ == "__main__":
    async def test():
        print("=" * 50)
        print("HYBRID SHORTS GENERATOR TEST")
        print("=" * 50)

        generator = HybridShortsGenerator()
        print(f"Pika available: {generator.pika is not None}")
        print(f"Stock available: {generator.stock is not None}")
        print(f"FFmpeg available: {generator.ffmpeg is not None}")

        # Test prompt generation
        for niche in ["finance", "psychology", "storytelling"]:
            intro = generator._get_prompt_for_niche(niche, "intro", "test topic")
            outro = generator._get_prompt_for_niche(niche, "outro", "test topic")
            print(f"\n{niche.upper()}:")
            print(f"  Intro: {intro[:60]}...")
            print(f"  Outro: {outro[:60]}...")

    asyncio.run(test())
