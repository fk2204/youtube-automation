"""
Video Assembly Module using MoviePy

Combines audio, video clips, text overlays, and effects to create
complete YouTube videos.

Usage:
    assembler = VideoAssembler()
    video_path = assembler.create_video(
        audio_file="narration.mp3",
        output_file="output.mp4",
        title="My Tutorial"
    )

Audio Enhancement (optional):
    video_path = assembler.create_video_from_audio(
        audio_file="narration.mp3",
        output_file="output.mp4",
        normalize_audio=True  # Normalize to -14 LUFS
    )
"""

import os
import gc
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from loguru import logger

# Import audio processor for normalization
try:
    from .audio_processor import AudioProcessor
    AUDIO_PROCESSOR_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSOR_AVAILABLE = False
    logger.debug("AudioProcessor not available - audio enhancement disabled")

# Import AI video providers for B-roll generation
try:
    from .ai_video_providers import AIVideoProviderRouter, get_ai_video_provider
    AI_VIDEO_AVAILABLE = True
except ImportError:
    AI_VIDEO_AVAILABLE = False
    AIVideoProviderRouter = None
    get_ai_video_provider = None
    logger.debug("AI video providers not available - AI B-roll generation disabled")

try:
    from moviepy.editor import (
        VideoFileClip, AudioFileClip, ImageClip, TextClip,
        CompositeVideoClip, concatenate_videoclips, ColorClip
    )
    from moviepy.video.fx.all import fadein, fadeout
except ImportError:
    raise ImportError("Please install moviepy: pip install moviepy")

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    raise ImportError("Please install pillow: pip install pillow")


@dataclass
class VideoSegment:
    """Represents a segment of the video."""
    start_time: float       # Start time in seconds
    end_time: float         # End time in seconds
    content_type: str       # text, image, video, color
    content: str            # File path or text content
    text_overlay: Optional[str] = None


class VideoAssembler:
    """Assemble videos from audio, images, and text."""

    # Optimized FFmpeg parameters for YouTube streaming
    FFMPEG_PARAMS_REGULAR = [
        "-movflags", "+faststart",    # Enable web streaming
        "-profile:v", "high",          # H.264 High Profile
        "-level", "4.2",               # Level 4.2 for 1080p30
        "-bf", "3",                    # 3 B-frames
        "-g", "60",                    # GOP size = 2x framerate
        "-keyint_min", "30",           # Minimum GOP
        "-sc_threshold", "0",          # Fixed GOP
        "-threads", "0",               # Auto thread detection
    ]

    FFMPEG_PARAMS_SHORTS = [
        "-movflags", "+faststart",    # Enable web streaming
        "-profile:v", "high",          # H.264 High Profile
        "-level", "4.2",               # Level 4.2 for 1080p
        "-bf", "2",                    # 2 B-frames for faster encode
        "-g", "60",                    # GOP size
        "-keyint_min", "30",           # Minimum GOP
        "-sc_threshold", "0",          # Fixed GOP
        "-threads", "0",               # Auto thread detection
    ]

    def __init__(
        self,
        resolution: Tuple[int, int] = (1920, 1080),
        fps: int = 30,
        background_color: Tuple[int, int, int] = (20, 20, 30),
        content_type: str = "regular"
    ):
        """
        Initialize the video assembler.

        Args:
            resolution: Video resolution (width, height)
            fps: Frames per second
            background_color: RGB background color
            content_type: "regular" for standard videos, "shorts" for YouTube Shorts
        """
        self.resolution = resolution
        self.fps = fps
        self.background_color = background_color
        self.width, self.height = resolution
        self.content_type = content_type

        # Select preset and FFmpeg params based on content type
        if content_type == "shorts":
            self.encoding_preset = "faster"  # Faster encode for shorts
            self.ffmpeg_params = self.FFMPEG_PARAMS_SHORTS
        else:
            self.encoding_preset = "medium"  # Balanced quality/speed for regular videos
            self.ffmpeg_params = self.FFMPEG_PARAMS_REGULAR

        # Check for hardware acceleration
        self._nvenc_available = None  # Lazy detection

        # Initialize audio processor for normalization
        self.audio_processor = None
        if AUDIO_PROCESSOR_AVAILABLE:
            try:
                self.audio_processor = AudioProcessor()
                logger.debug("AudioProcessor initialized for audio enhancement")
            except Exception as e:
                logger.debug(f"AudioProcessor initialization failed: {e}")

        logger.info(f"VideoAssembler: {resolution[0]}x{resolution[1]} @ {fps}fps (content_type={content_type})")

    def _check_nvenc_available(self) -> bool:
        """Check if NVIDIA NVENC hardware encoder is available."""
        if self._nvenc_available is not None:
            return self._nvenc_available

        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=5
            )
            self._nvenc_available = "h264_nvenc" in result.stdout
            if self._nvenc_available:
                logger.info("NVENC hardware encoder detected")
            return self._nvenc_available
        except Exception:
            self._nvenc_available = False
            return False

    def _get_video_codec(self) -> str:
        """Get the best available video codec (NVENC if available, otherwise libx264)."""
        if self._check_nvenc_available():
            return "h264_nvenc"
        return "libx264"

    def _get_ffmpeg_params(self, use_nvenc: bool = False) -> List[str]:
        """Get FFmpeg parameters optimized for the encoder."""
        params = list(self.ffmpeg_params)  # Copy base params

        if use_nvenc and self._check_nvenc_available():
            # NVENC-specific optimizations
            params.extend([
                "-rc", "vbr",           # Variable bitrate
                "-cq", "23",            # Constant quality
                "-spatial-aq", "1",     # Spatial adaptive quantization
                "-temporal-aq", "1",    # Temporal adaptive quantization
            ])
        else:
            # libx264-specific optimization
            params.extend([
                "-crf", "23",           # Constant rate factor
            ])

        return params

    def _calculate_optimal_bitrate(self, has_motion: bool = False, content_type: str = "regular") -> str:
        """Calculate optimal bitrate based on content characteristics."""
        if content_type == "shorts":
            return "6000k" if has_motion else "5000k"
        # Regular videos
        if has_motion:
            return "10000k"  # 10 Mbps for motion content
        return "6000k"  # 6 Mbps for static content

    def two_pass_encode(
        self,
        input_file: str,
        output_file: str,
        target_bitrate: str = "8M",
        max_bitrate: str = "12M"
    ) -> Optional[str]:
        """
        Perform two-pass encoding for maximum quality at target bitrate.

        Args:
            input_file: Input video file path
            output_file: Output video file path
            target_bitrate: Target bitrate (e.g., "8M" for 8 Mbps)
            max_bitrate: Maximum bitrate (e.g., "12M" for 12 Mbps)

        Returns:
            Path to encoded file or None on failure
        """
        import tempfile
        passlog = tempfile.mktemp(prefix="ffmpeg_pass_")

        try:
            # Pass 1: Analysis
            pass1_cmd = [
                "ffmpeg", "-y",
                "-i", input_file,
                "-c:v", "libx264",
                "-preset", self.encoding_preset,
                "-b:v", target_bitrate,
                "-maxrate", max_bitrate,
                "-bufsize", "16M",
                "-pass", "1",
                "-passlogfile", passlog,
                "-an",  # No audio for pass 1
                "-f", "null",
                "NUL" if os.name == "nt" else "/dev/null"
            ]

            logger.info("Two-pass encoding: Pass 1 (analysis)...")
            subprocess.run(pass1_cmd, capture_output=True, timeout=600)

            # Pass 2: Encode
            pass2_cmd = [
                "ffmpeg", "-y",
                "-i", input_file,
                "-c:v", "libx264",
                "-preset", self.encoding_preset,
                "-b:v", target_bitrate,
                "-maxrate", max_bitrate,
                "-bufsize", "16M",
                "-pass", "2",
                "-passlogfile", passlog,
                "-c:a", "aac",
                "-b:a", "256k",
            ] + self.FFMPEG_PARAMS_REGULAR + [output_file]

            logger.info("Two-pass encoding: Pass 2 (encoding)...")
            result = subprocess.run(pass2_cmd, capture_output=True, timeout=600)

            if os.path.exists(output_file):
                logger.success(f"Two-pass encoding complete: {output_file}")
                return output_file

            logger.error(f"Two-pass encoding failed: {result.stderr.decode()[:500]}")
            return None

        except Exception as e:
            logger.error(f"Two-pass encoding error: {e}")
            return None
        finally:
            # Cleanup passlog files
            for ext in ["", "-0.log", "-0.log.mbtree"]:
                try:
                    log_file = passlog + ext
                    if os.path.exists(log_file):
                        os.remove(log_file)
                except Exception:
                    pass

    def create_text_clip(
        self,
        text: str,
        duration: float,
        fontsize: int = 60,
        color: str = "white",
        bg_color: Optional[str] = None,
        position: str = "center"
    ) -> TextClip:
        """
        Create a text clip.

        Args:
            text: Text to display
            duration: Duration in seconds
            fontsize: Font size
            color: Text color
            bg_color: Background color (None for transparent)
            position: Position (center, top, bottom)
        """
        try:
            clip = TextClip(
                text,
                fontsize=fontsize,
                color=color,
                bg_color=bg_color,
                size=(self.width - 100, None),
                method="caption"
            ).set_duration(duration).set_position(position)
            return clip
        except Exception as e:
            logger.warning(f"TextClip failed: {e}. Using image-based text.")
            return self._create_text_image_clip(text, duration, fontsize, color, position)

    def _create_text_image_clip(
        self,
        text: str,
        duration: float,
        fontsize: int,
        color: str,
        position: str
    ) -> ImageClip:
        """Create text as an image clip (fallback method)."""
        # Create image with PIL
        img = Image.new('RGBA', self.resolution, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Try to use a system font
        try:
            font = ImageFont.truetype("arial.ttf", fontsize)
        except (OSError, IOError) as e:
            logger.debug(f"Could not load arial.ttf font: {e}, using default")
            font = ImageFont.load_default()

        # Calculate text position
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (self.width - text_width) // 2
        if position == "top":
            y = 50
        elif position == "bottom":
            y = self.height - text_height - 50
        else:
            y = (self.height - text_height) // 2

        # Draw text
        draw.text((x, y), text, fill=color, font=font)

        # Convert to clip
        img_path = f"temp_text_{hash(text)}.png"
        img.save(img_path)

        clip = ImageClip(img_path).set_duration(duration)

        # Clean up temp file
        try:
            os.remove(img_path)
        except (OSError, PermissionError) as e:
            logger.debug(f"Could not remove temp file {img_path}: {e}")

        return clip

    def create_background_clip(
        self,
        duration: float,
        color: Optional[Tuple[int, int, int]] = None
    ) -> ColorClip:
        """Create a solid color background clip."""
        color = color or self.background_color
        return ColorClip(
            size=self.resolution,
            color=color,
            duration=duration
        )

    def create_title_card(
        self,
        title: str,
        subtitle: Optional[str] = None,
        duration: float = 5.0
    ) -> CompositeVideoClip:
        """
        Create a title card for the video intro.

        Args:
            title: Main title text
            subtitle: Optional subtitle
            duration: Duration in seconds
        """
        clips = []

        # Background
        bg = self.create_background_clip(duration)
        clips.append(bg)

        # Title
        title_clip = self.create_text_clip(
            title,
            duration,
            fontsize=80,
            color="white",
            position="center"
        )
        clips.append(title_clip)

        # Subtitle
        if subtitle:
            sub_clip = self.create_text_clip(
                subtitle,
                duration,
                fontsize=40,
                color="#aaaaaa",
                position=("center", self.height // 2 + 80)
            )
            clips.append(sub_clip)

        return CompositeVideoClip(clips, size=self.resolution)

    def create_video_from_audio(
        self,
        audio_file: str,
        output_file: str,
        title: Optional[str] = None,
        segments: Optional[List[VideoSegment]] = None,
        add_captions: bool = True,
        subtitle_file: Optional[str] = None,
        normalize_audio: bool = False,
        background_music: Optional[str] = None,
        music_volume: float = 0.15
    ) -> str:
        """
        Create a video from audio narration.

        Args:
            audio_file: Path to audio file (MP3)
            output_file: Output video file path
            title: Optional title for intro card
            segments: Optional list of VideoSegments for custom visuals
            add_captions: Whether to add caption overlays
            subtitle_file: Path to VTT/SRT subtitle file
            normalize_audio: Normalize audio to YouTube's -14 LUFS standard
            background_music: Optional background music file to mix with voice
            music_volume: Volume level for background music (0.0-1.0, default 0.15)

        Returns:
            Path to created video file
        """
        logger.info(f"Creating video from audio: {audio_file}")

        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        # Process audio (normalization and/or music mixing)
        processed_audio_file = audio_file
        temp_audio_file = None

        if self.audio_processor and (normalize_audio or background_music):
            temp_audio_file = str(Path(output_file).with_suffix('.processed_audio.mp3'))

            if background_music and os.path.exists(background_music):
                # Mix with background music (this also normalizes)
                logger.info(f"Mixing audio with background music at {music_volume*100:.0f}% volume")
                result = self.audio_processor.mix_with_background_music(
                    voice_file=audio_file,
                    music_file=background_music,
                    output_file=temp_audio_file,
                    music_volume=music_volume,
                    normalize_before_mix=True
                )
                if result:
                    processed_audio_file = result
            elif normalize_audio:
                # Just normalize
                logger.info("Normalizing audio to YouTube's -14 LUFS standard")
                result = self.audio_processor.normalize_audio(audio_file, temp_audio_file)
                if result:
                    processed_audio_file = result

        # Load audio (use processed if available)
        audio = AudioFileClip(processed_audio_file)
        duration = audio.duration

        logger.info(f"Audio duration: {duration:.1f} seconds")

        clips = []

        # Add title card if provided
        if title:
            title_card = self.create_title_card(title, duration=4.0)
            title_card = fadein(title_card, 0.5)
            title_card = fadeout(title_card, 0.5)
            clips.append(title_card)
            remaining_duration = duration - 4.0
        else:
            remaining_duration = duration

        # Create main content clip
        if segments:
            # Use provided segments
            for segment in segments:
                seg_duration = segment.end_time - segment.start_time

                if segment.content_type == "text":
                    clip = self.create_text_clip(
                        segment.content,
                        seg_duration,
                        fontsize=50
                    )
                elif segment.content_type == "image":
                    clip = ImageClip(segment.content).set_duration(seg_duration)
                    clip = clip.resize(self.resolution)
                elif segment.content_type == "color":
                    clip = self.create_background_clip(seg_duration)
                else:
                    clip = self.create_background_clip(seg_duration)

                clips.append(clip)
        else:
            # Create simple background with animated text
            main_clip = self.create_background_clip(remaining_duration)
            clips.append(main_clip)

        # Concatenate all clips
        if len(clips) > 1:
            video = concatenate_videoclips(clips, method="compose")
        else:
            video = clips[0]

        # Ensure video matches audio duration
        if video.duration != duration:
            video = video.set_duration(duration)

        # Add audio
        video = video.set_audio(audio)

        # Write output file
        logger.info(f"Rendering video to: {output_file}")

        # Collect all clips for cleanup
        all_clips = [video, audio] + clips

        try:
            video.write_videofile(
                output_file,
                fps=self.fps,
                codec=self._get_video_codec(),
                audio_codec="aac",
                audio_bitrate="256k",
                preset=self.encoding_preset,
                threads=0,  # Auto-detect threads
                logger=None,  # Suppress moviepy's verbose output
                ffmpeg_params=self._get_ffmpeg_params(use_nvenc=self._check_nvenc_available())
            )

            logger.success(f"Video created: {output_file}")
            return output_file

        finally:
            # Memory optimization: properly close all clips
            for clip in all_clips:
                try:
                    clip.close()
                except Exception:
                    pass

            # Clean up temp audio file
            if temp_audio_file and os.path.exists(temp_audio_file):
                try:
                    os.remove(temp_audio_file)
                except (OSError, PermissionError) as e:
                    logger.debug(f"Could not remove temp audio file: {e}")

            # Force garbage collection to free memory
            gc.collect()

    def create_thumbnail(
        self,
        output_file: str,
        title: str,
        subtitle: Optional[str] = None,
        background_color: Tuple[int, int, int] = (30, 30, 60),
        text_color: str = "white"
    ) -> str:
        """
        Create a video thumbnail.

        Args:
            output_file: Output image path
            title: Main title text
            subtitle: Optional subtitle
            background_color: RGB background color
            text_color: Text color

        Returns:
            Path to created thumbnail
        """
        logger.info(f"Creating thumbnail: {output_file}")

        # Create image
        size = (1280, 720)  # YouTube thumbnail size
        img = Image.new('RGB', size, background_color)
        draw = ImageDraw.Draw(img)

        # Load fonts
        try:
            title_font = ImageFont.truetype("arial.ttf", 72)
            sub_font = ImageFont.truetype("arial.ttf", 36)
        except (OSError, IOError) as e:
            logger.debug(f"Could not load fonts for thumbnail: {e}, using defaults")
            title_font = ImageFont.load_default()
            sub_font = ImageFont.load_default()

        # Draw title
        bbox = draw.textbbox((0, 0), title, font=title_font)
        text_width = bbox[2] - bbox[0]
        x = (size[0] - text_width) // 2
        y = size[1] // 2 - 50

        # Add shadow
        draw.text((x + 3, y + 3), title, fill="black", font=title_font)
        draw.text((x, y), title, fill=text_color, font=title_font)

        # Draw subtitle
        if subtitle:
            bbox = draw.textbbox((0, 0), subtitle, font=sub_font)
            sub_width = bbox[2] - bbox[0]
            sub_x = (size[0] - sub_width) // 2
            sub_y = y + 80
            draw.text((sub_x, sub_y), subtitle, fill="#cccccc", font=sub_font)

        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        # Save
        img.save(output_file, quality=95)

        logger.success(f"Thumbnail created: {output_file}")
        return output_file

    def add_background_music(
        self,
        video_file: str,
        music_file: str,
        output_file: str,
        music_volume: float = 0.1
    ) -> str:
        """
        Add background music to a video.

        Args:
            video_file: Input video path
            music_file: Background music path
            output_file: Output video path
            music_volume: Music volume (0.0-1.0)

        Returns:
            Path to output video
        """
        logger.info("Adding background music...")

        video = None
        music = None
        clips_to_close = []

        try:
            video = VideoFileClip(video_file)
            music = AudioFileClip(music_file)
            clips_to_close = [video, music]

            # Loop music if shorter than video
            if music.duration < video.duration:
                loops = int(video.duration / music.duration) + 1
                music = concatenate_videoclips([music] * loops)
                clips_to_close.append(music)

            # Trim music to video length
            music = music.subclip(0, video.duration)

            # Adjust volume
            music = music.volumex(music_volume)

            # Mix with original audio
            if video.audio:
                from moviepy.audio.AudioClip import CompositeAudioClip
                final_audio = CompositeAudioClip([video.audio, music])
                video = video.set_audio(final_audio)
            else:
                video = video.set_audio(music)

            # Write output with optimized FFmpeg params
            video.write_videofile(
                output_file,
                fps=self.fps,
                codec=self._get_video_codec(),
                audio_codec="aac",
                audio_bitrate="256k",
                preset=self.encoding_preset,
                threads=0,
                logger=None,
                ffmpeg_params=self._get_ffmpeg_params(use_nvenc=self._check_nvenc_available())
            )

            logger.success(f"Music added: {output_file}")
            return output_file

        finally:
            # Memory optimization: properly close all clips
            for clip in clips_to_close:
                try:
                    clip.close()
                except Exception:
                    pass

            # Force garbage collection
            gc.collect()

    async def generate_ai_broll(
        self,
        script_segments: List[str],
        output_dir: str,
        style: str = "cinematic",
        niche: str = "default",
        duration: int = 5,
        provider: Optional[str] = None,
        fallback_to_stock: bool = True
    ) -> List[Optional[str]]:
        """
        Generate AI B-roll clips for script segments.

        Uses Runway, Pika, or other AI video providers to generate
        visually relevant B-roll footage for each script segment.
        Falls back to stock footage if AI generation fails.

        Args:
            script_segments: List of script text segments to visualize
            output_dir: Directory to save generated clips
            style: Visual style (cinematic, documentary, corporate, abstract)
            niche: Content niche (finance, psychology, storytelling, technology)
            duration: Duration per clip in seconds
            provider: Specific provider to use (None for auto-select)
            fallback_to_stock: If True, use stock footage when AI generation fails

        Returns:
            List of paths to generated video clips (None for failed generations)
        """
        if not AI_VIDEO_AVAILABLE:
            logger.warning(
                "AI video providers not available. "
                "Install with: pip install runwayml httpx"
            )
            if fallback_to_stock:
                return await self._fallback_to_stock_footage(
                    script_segments, output_dir, niche
                )
            return [None] * len(script_segments)

        import asyncio

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Get provider/router
        if provider:
            ai_provider = get_ai_video_provider(provider)
            if not ai_provider or not ai_provider.is_available:
                logger.warning(f"Provider {provider} not available, using router")
                ai_provider = None
        else:
            ai_provider = None

        router = None
        if not ai_provider:
            router = AIVideoProviderRouter()
            if not router.providers:
                logger.warning("No AI video providers available")
                if fallback_to_stock:
                    return await self._fallback_to_stock_footage(
                        script_segments, output_dir, niche
                    )
                return [None] * len(script_segments)

        # Style and niche prompt enhancements
        style_prompts = {
            "cinematic": "cinematic shot, dramatic lighting, film grain, professional",
            "documentary": "documentary style, natural lighting, realistic",
            "corporate": "clean, professional, modern, well-lit",
            "abstract": "abstract visuals, artistic, symbolic",
        }
        niche_context = {
            "finance": "business, money, investment, professional setting",
            "psychology": "human emotion, mind, thinking, relationships",
            "storytelling": "dramatic, narrative, atmospheric",
            "technology": "futuristic, digital, tech, innovation",
            "default": "",
        }

        style_addition = style_prompts.get(style, style_prompts["cinematic"])
        niche_addition = niche_context.get(niche, "")

        results = []
        for i, segment in enumerate(script_segments):
            output_file = str(Path(output_dir) / f"ai_broll_{i:03d}.mp4")

            # Build enhanced prompt
            enhanced_prompt = (
                f"B-roll footage: {segment}. "
                f"{style_addition}. {niche_addition}. "
                f"16:9 widescreen, high quality."
            )

            logger.info(f"Generating AI B-roll [{i+1}/{len(script_segments)}]: {segment[:50]}...")

            try:
                if ai_provider:
                    result = await ai_provider.generate_video(
                        prompt=enhanced_prompt,
                        output_file=output_file,
                        duration=duration,
                        aspect_ratio="16:9"
                    )
                else:
                    result = await router.generate_video(
                        prompt=enhanced_prompt,
                        output_file=output_file,
                        duration=duration,
                        aspect_ratio="16:9"
                    )

                if result.success and result.local_path:
                    logger.success(f"AI B-roll generated: {result.local_path}")
                    results.append(result.local_path)
                else:
                    logger.warning(f"AI generation failed: {result.error}")
                    if fallback_to_stock:
                        fallback = await self._fallback_single_stock(
                            segment, output_dir, niche, i
                        )
                        results.append(fallback)
                    else:
                        results.append(None)

            except Exception as e:
                logger.error(f"AI B-roll generation error: {e}")
                if fallback_to_stock:
                    fallback = await self._fallback_single_stock(
                        segment, output_dir, niche, i
                    )
                    results.append(fallback)
                else:
                    results.append(None)

        return results

    async def _fallback_to_stock_footage(
        self,
        script_segments: List[str],
        output_dir: str,
        niche: str
    ) -> List[Optional[str]]:
        """Fallback to stock footage for all segments."""
        logger.info("Falling back to stock footage for all segments")
        results = []
        for i, segment in enumerate(script_segments):
            result = await self._fallback_single_stock(segment, output_dir, niche, i)
            results.append(result)
        return results

    async def _fallback_single_stock(
        self,
        segment: str,
        output_dir: str,
        niche: str,
        index: int
    ) -> Optional[str]:
        """Get single stock footage clip as fallback."""
        try:
            from .multi_stock import get_stock_provider

            provider = get_stock_provider()
            if not provider:
                logger.warning("No stock footage provider available")
                return None

            # Extract keywords from segment
            keywords = self._extract_keywords(segment, niche)
            search_query = " ".join(keywords[:3])

            logger.info(f"Fetching stock footage for: {search_query}")

            # Search and download
            results = await provider.search(search_query, count=1)
            if results:
                output_file = str(Path(output_dir) / f"stock_broll_{index:03d}.mp4")
                downloaded = await provider.download(results[0], output_file)
                if downloaded:
                    return downloaded

            return None
        except Exception as e:
            logger.error(f"Stock footage fallback error: {e}")
            return None

    def _extract_keywords(self, text: str, niche: str) -> List[str]:
        """Extract relevant keywords from text for stock footage search."""
        import re

        # Niche-specific keyword mappings
        niche_keywords = {
            "finance": ["money", "business", "investment", "chart", "success"],
            "psychology": ["mind", "brain", "emotion", "thinking", "person"],
            "storytelling": ["story", "dramatic", "scene", "journey", "adventure"],
            "technology": ["tech", "computer", "digital", "future", "innovation"],
        }

        # Get words from text
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())

        # Remove common words
        stopwords = {"this", "that", "with", "from", "have", "will", "your", "about",
                     "when", "what", "which", "where", "there", "their", "they",
                     "would", "could", "should", "into", "more", "some", "very"}
        keywords = [w for w in words if w not in stopwords]

        # Add niche-specific keywords
        niche_kw = niche_keywords.get(niche, [])
        keywords = keywords[:5] + niche_kw[:2]

        return keywords[:5]  # Return top 5 keywords

    def create_video_with_ai_broll(
        self,
        audio_file: str,
        output_file: str,
        script_segments: List[str],
        title: Optional[str] = None,
        niche: str = "default",
        style: str = "cinematic",
        ai_provider: Optional[str] = None,
        fallback_to_stock: bool = True,
        normalize_audio: bool = True
    ) -> str:
        """
        Create a video with AI-generated B-roll clips.

        This is a high-level convenience method that:
        1. Generates AI B-roll for script segments
        2. Assembles the video with audio
        3. Falls back to stock footage if AI generation fails

        Args:
            audio_file: Path to audio file
            output_file: Output video path
            script_segments: List of script text segments for B-roll
            title: Optional title for intro card
            niche: Content niche
            style: Visual style for B-roll
            ai_provider: Specific AI provider to use
            fallback_to_stock: Use stock footage when AI fails
            normalize_audio: Normalize audio to YouTube standards

        Returns:
            Path to created video file
        """
        import asyncio

        logger.info(f"Creating video with AI B-roll for {len(script_segments)} segments")

        # Generate AI B-roll
        broll_dir = str(Path(output_file).parent / "ai_broll_temp")

        async def _generate():
            return await self.generate_ai_broll(
                script_segments=script_segments,
                output_dir=broll_dir,
                style=style,
                niche=niche,
                provider=ai_provider,
                fallback_to_stock=fallback_to_stock
            )

        broll_clips = asyncio.run(_generate())

        # Filter out failed generations
        valid_clips = [c for c in broll_clips if c and os.path.exists(c)]

        if not valid_clips:
            logger.warning("No B-roll clips generated, creating simple video")
            return self.create_video_from_audio(
                audio_file=audio_file,
                output_file=output_file,
                title=title,
                normalize_audio=normalize_audio
            )

        # Load audio to get duration
        audio = AudioFileClip(audio_file)
        total_duration = audio.duration

        # Calculate segment durations
        segment_duration = total_duration / len(valid_clips)

        # Create video segments
        segments = []
        for i, clip_path in enumerate(valid_clips):
            segment = VideoSegment(
                start_time=i * segment_duration,
                end_time=(i + 1) * segment_duration,
                content_type="video",
                content=clip_path
            )
            segments.append(segment)

        # Create video with segments
        result = self.create_video_from_audio(
            audio_file=audio_file,
            output_file=output_file,
            title=title,
            segments=segments,
            normalize_audio=normalize_audio
        )

        # Cleanup temp B-roll directory
        try:
            import shutil
            shutil.rmtree(broll_dir, ignore_errors=True)
        except Exception:
            pass

        audio.close()
        return result


# Example usage
if __name__ == "__main__":
    assembler = VideoAssembler()

    # Create a simple test video
    print("\nCreating test thumbnail...")
    assembler.create_thumbnail(
        output_file="output/test_thumbnail.png",
        title="Python Tutorial",
        subtitle="Learn Python in 10 Minutes"
    )

    print("\nTo create a full video, you need an audio file.")
    print("Use the TTS module to generate audio first.")
