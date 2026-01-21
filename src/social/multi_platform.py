"""
Multi-Platform Distribution Module for YouTube Automation

Repurpose long-form content to short-form formats for maximum reach across:
- YouTube Shorts
- TikTok-ready exports
- Instagram Reels-ready exports

Features:
- Auto-resize videos (16:9 to 9:16)
- Platform-specific metadata templates
- Cross-posting scheduler
- Smart segment extraction for best moments

Usage:
    from src.social.multi_platform import MultiPlatformDistributor

    distributor = MultiPlatformDistributor()

    # Repurpose a long-form video to Shorts
    shorts = distributor.create_shorts(
        video_path="output/video.mp4",
        title="How I Made $10,000",
        niche="finance",
        num_shorts=3
    )

    # Export for all platforms
    exports = await distributor.export_all_platforms(
        video_path="output/video.mp4",
        title="My Video Title",
        niche="finance"
    )
"""

import os
import asyncio
import json
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from loguru import logger

try:
    from moviepy.editor import (
        VideoFileClip, AudioFileClip, CompositeVideoClip,
        TextClip, ColorClip, concatenate_videoclips
    )
    MOVIEPY_AVAILABLE = True
except ImportError:
    logger.warning("MoviePy not installed. Install with: pip install moviepy")
    MOVIEPY_AVAILABLE = False


class Platform(Enum):
    """Supported distribution platforms."""
    YOUTUBE_SHORTS = "youtube_shorts"
    TIKTOK = "tiktok"
    INSTAGRAM_REELS = "instagram_reels"
    YOUTUBE_LONG = "youtube_long"


class AspectRatio(Enum):
    """Video aspect ratios."""
    LANDSCAPE = "16:9"    # 1920x1080
    PORTRAIT = "9:16"     # 1080x1920
    SQUARE = "1:1"        # 1080x1080


@dataclass
class PlatformSpec:
    """Platform-specific specifications."""
    platform: Platform
    aspect_ratio: AspectRatio
    max_duration: int  # seconds
    min_duration: int  # seconds
    resolution: Tuple[int, int]
    max_file_size_mb: int
    supports_captions: bool = True
    supports_music: bool = True
    hashtag_limit: int = 30
    caption_limit: int = 2200

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["platform"] = self.platform.value
        result["aspect_ratio"] = self.aspect_ratio.value
        return result


# Platform specifications
PLATFORM_SPECS: Dict[Platform, PlatformSpec] = {
    Platform.YOUTUBE_SHORTS: PlatformSpec(
        platform=Platform.YOUTUBE_SHORTS,
        aspect_ratio=AspectRatio.PORTRAIT,
        max_duration=60,
        min_duration=15,
        resolution=(1080, 1920),
        max_file_size_mb=256,
        hashtag_limit=60,
        caption_limit=100,
    ),
    Platform.TIKTOK: PlatformSpec(
        platform=Platform.TIKTOK,
        aspect_ratio=AspectRatio.PORTRAIT,
        max_duration=180,  # 3 minutes for most accounts
        min_duration=5,
        resolution=(1080, 1920),
        max_file_size_mb=287,
        hashtag_limit=30,
        caption_limit=2200,
    ),
    Platform.INSTAGRAM_REELS: PlatformSpec(
        platform=Platform.INSTAGRAM_REELS,
        aspect_ratio=AspectRatio.PORTRAIT,
        max_duration=90,
        min_duration=3,
        resolution=(1080, 1920),
        max_file_size_mb=250,
        hashtag_limit=30,
        caption_limit=2200,
    ),
    Platform.YOUTUBE_LONG: PlatformSpec(
        platform=Platform.YOUTUBE_LONG,
        aspect_ratio=AspectRatio.LANDSCAPE,
        max_duration=43200,  # 12 hours
        min_duration=1,
        resolution=(1920, 1080),
        max_file_size_mb=256000,  # 256GB
        hashtag_limit=500,
        caption_limit=5000,
    ),
}


@dataclass
class ShortSegment:
    """A segment extracted for short-form content."""
    start_time: float
    end_time: float
    duration: float
    score: float  # Virality score 0-100
    hook_text: str
    source_video: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlatformExport:
    """Result of exporting for a platform."""
    platform: Platform
    video_path: str
    thumbnail_path: Optional[str]
    title: str
    description: str
    hashtags: List[str]
    duration: float
    resolution: Tuple[int, int]
    file_size_mb: float
    ready_to_upload: bool
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["platform"] = self.platform.value
        return result


@dataclass
class MetadataTemplate:
    """Platform-specific metadata template."""
    platform: Platform
    title_template: str
    description_template: str
    hashtag_sets: Dict[str, List[str]]
    cta_phrases: List[str]

    def generate_title(self, base_title: str, niche: str) -> str:
        """Generate platform-specific title."""
        return self.title_template.format(title=base_title, niche=niche)

    def generate_description(self, base_title: str, niche: str, video_url: str = "") -> str:
        """Generate platform-specific description."""
        return self.description_template.format(
            title=base_title,
            niche=niche,
            video_url=video_url
        )

    def get_hashtags(self, niche: str, count: int = 10) -> List[str]:
        """Get relevant hashtags for niche."""
        niche_tags = self.hashtag_sets.get(niche.lower(), [])
        general_tags = self.hashtag_sets.get("general", [])

        # Combine niche-specific and general tags
        combined = niche_tags[:count//2] + general_tags[:count//2]
        return combined[:count]


# Platform-specific metadata templates
METADATA_TEMPLATES: Dict[Platform, MetadataTemplate] = {
    Platform.YOUTUBE_SHORTS: MetadataTemplate(
        platform=Platform.YOUTUBE_SHORTS,
        title_template="{title} #shorts",
        description_template="{title}\n\nFull video: {video_url}\n\n#shorts",
        hashtag_sets={
            "finance": ["money", "finance", "investing", "wealthbuilding", "passiveincome", "financialfreedom"],
            "psychology": ["psychology", "mindset", "mentalhealth", "selfimprovement", "motivation"],
            "storytelling": ["story", "truecrime", "mystery", "documentary", "storytelling"],
            "general": ["viral", "trending", "fyp", "shorts", "youtubeshorts"],
        },
        cta_phrases=[
            "Follow for more!",
            "Like for part 2!",
            "Comment what you think!",
        ],
    ),
    Platform.TIKTOK: MetadataTemplate(
        platform=Platform.TIKTOK,
        title_template="{title}",
        description_template="{title} #fyp #viral #trending",
        hashtag_sets={
            "finance": ["moneytok", "financetok", "investing", "wealthtok", "money", "financetips"],
            "psychology": ["psychologyfacts", "mindset", "mentalhealthawareness", "selfimprovement", "psychtok"],
            "storytelling": ["storytimetiktok", "truecrime", "mysterytok", "storytime", "scary"],
            "general": ["fyp", "foryou", "viral", "trending", "tiktok"],
        },
        cta_phrases=[
            "Follow for more!",
            "Like if you agree!",
            "Part 2?",
        ],
    ),
    Platform.INSTAGRAM_REELS: MetadataTemplate(
        platform=Platform.INSTAGRAM_REELS,
        title_template="{title}",
        description_template="{title}\n\nSave this for later!\n\n.",
        hashtag_sets={
            "finance": ["finance", "money", "investing", "wealth", "financialeducation", "moneytips"],
            "psychology": ["psychology", "mindset", "personalgrowth", "selfcare", "motivation"],
            "storytelling": ["storytelling", "truecrime", "mystery", "stories", "viral"],
            "general": ["reels", "reelsinstagram", "viral", "trending", "explore"],
        },
        cta_phrases=[
            "Save this post!",
            "Tag someone who needs this!",
            "Follow for daily content!",
        ],
    ),
}


@dataclass
class CrossPostSchedule:
    """Schedule for cross-posting across platforms."""
    video_title: str
    source_video: str
    platform_exports: Dict[str, PlatformExport]
    schedule: Dict[str, datetime]  # platform -> scheduled time
    status: Dict[str, str]  # platform -> status (pending, posted, failed)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_title": self.video_title,
            "source_video": self.source_video,
            "platform_exports": {k: v.to_dict() for k, v in self.platform_exports.items()},
            "schedule": {k: v.isoformat() for k, v in self.schedule.items()},
            "status": self.status,
            "created_at": self.created_at.isoformat(),
        }


class VideoResizer:
    """Handle video resizing and cropping for different aspect ratios."""

    def __init__(self, output_dir: str = "output/resized"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def resize_video(
        self,
        input_path: str,
        target_resolution: Tuple[int, int],
        output_path: Optional[str] = None,
        crop_mode: str = "center"  # center, top, bottom, smart
    ) -> str:
        """
        Resize video to target resolution with cropping.

        Args:
            input_path: Path to input video
            target_resolution: (width, height) tuple
            output_path: Optional output path
            crop_mode: How to crop (center, top, bottom, smart)

        Returns:
            Path to resized video
        """
        if not MOVIEPY_AVAILABLE:
            raise ImportError("MoviePy required for video resizing")

        logger.info(f"Resizing video to {target_resolution[0]}x{target_resolution[1]}")

        if not output_path:
            input_name = Path(input_path).stem
            output_path = str(self.output_dir / f"{input_name}_{target_resolution[0]}x{target_resolution[1]}.mp4")

        clip = VideoFileClip(input_path)

        # Calculate scaling and cropping
        target_w, target_h = target_resolution
        target_aspect = target_w / target_h
        source_aspect = clip.w / clip.h

        if source_aspect > target_aspect:
            # Source is wider - scale by height, crop width
            new_h = target_h
            new_w = int(clip.w * (target_h / clip.h))
            resized = clip.resize(height=new_h)

            # Crop width
            crop_x = (new_w - target_w) // 2
            if crop_mode == "top":
                crop_x = 0
            elif crop_mode == "bottom":
                crop_x = new_w - target_w

            cropped = resized.crop(x1=crop_x, x2=crop_x + target_w)
        else:
            # Source is taller - scale by width, crop height
            new_w = target_w
            new_h = int(clip.h * (target_w / clip.w))
            resized = clip.resize(width=new_w)

            # Crop height
            crop_y = (new_h - target_h) // 2
            if crop_mode == "top":
                crop_y = 0
            elif crop_mode == "bottom":
                crop_y = new_h - target_h

            cropped = resized.crop(y1=crop_y, y2=crop_y + target_h)

        # Ensure exact resolution
        final = cropped.resize(newsize=target_resolution)

        # Write output
        final.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            bitrate="8M",
            fps=30,
            preset="medium",
            threads=4
        )

        clip.close()
        resized.close()
        final.close()

        logger.success(f"Resized video saved: {output_path}")
        return output_path

    def convert_landscape_to_portrait(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        add_blur_background: bool = True
    ) -> str:
        """
        Convert 16:9 video to 9:16 with optional blur background.

        This creates a more visually appealing result than simple cropping
        by adding a blurred version of the video as background.

        Args:
            input_path: Path to input video
            output_path: Optional output path
            add_blur_background: Whether to add blurred background

        Returns:
            Path to converted video
        """
        if not MOVIEPY_AVAILABLE:
            raise ImportError("MoviePy required for video conversion")

        logger.info("Converting landscape to portrait format")

        if not output_path:
            input_name = Path(input_path).stem
            output_path = str(self.output_dir / f"{input_name}_portrait.mp4")

        clip = VideoFileClip(input_path)
        target_w, target_h = 1080, 1920

        if add_blur_background:
            # Create blurred background
            bg_clip = clip.resize(height=target_h)

            # Center crop for background
            crop_x = (bg_clip.w - target_w) // 2
            bg_cropped = bg_clip.crop(x1=max(0, crop_x), x2=min(bg_clip.w, crop_x + target_w))
            bg_final = bg_cropped.resize(newsize=(target_w, target_h))

            # Apply blur (using PIL via moviepy)
            def blur_frame(frame):
                from PIL import Image, ImageFilter
                img = Image.fromarray(frame)
                blurred = img.filter(ImageFilter.GaussianBlur(radius=30))
                # Darken
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Brightness(blurred)
                darkened = enhancer.enhance(0.5)
                return np.array(darkened)

            try:
                import numpy as np
                bg_blurred = bg_final.fl_image(blur_frame)
            except Exception:
                # Fallback to just darkened background
                bg_blurred = bg_final.fl_image(lambda f: (f * 0.3).astype('uint8'))

            # Scale main content to fit height-wise centered
            main_scale = min(target_w / clip.w, target_h * 0.6 / clip.h)
            main_clip = clip.resize(main_scale)

            # Center the main clip
            main_x = (target_w - main_clip.w) // 2
            main_y = (target_h - main_clip.h) // 2

            # Composite
            final = CompositeVideoClip([
                bg_blurred,
                main_clip.set_position((main_x, main_y))
            ], size=(target_w, target_h))
        else:
            # Simple crop from center
            final = self.resize_video(input_path, (target_w, target_h))
            clip.close()
            return final

        # Write output
        final.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            bitrate="8M",
            fps=30,
            preset="medium",
            threads=4
        )

        clip.close()
        final.close()

        logger.success(f"Portrait video saved: {output_path}")
        return output_path


class SegmentExtractor:
    """Extract best segments from long-form videos for short-form content."""

    # Hook patterns that indicate good short-form content
    HOOK_INDICATORS = [
        "here's the thing",
        "let me tell you",
        "the truth is",
        "what most people don't know",
        "the secret is",
        "this changed everything",
        "stop doing this",
        "biggest mistake",
        "the real reason",
        "nobody talks about",
    ]

    def __init__(self):
        self.segments: List[ShortSegment] = []

    async def find_best_segments(
        self,
        video_path: str,
        transcript: Optional[str] = None,
        num_segments: int = 3,
        min_duration: int = 15,
        max_duration: int = 60
    ) -> List[ShortSegment]:
        """
        Find the best segments for short-form content.

        Args:
            video_path: Path to source video
            transcript: Optional transcript with timestamps
            num_segments: Number of segments to extract
            min_duration: Minimum segment duration in seconds
            max_duration: Maximum segment duration in seconds

        Returns:
            List of ShortSegment objects
        """
        logger.info(f"Finding {num_segments} best segments from {video_path}")

        segments = []

        if transcript:
            # Use transcript to find hook points
            segments = await self._find_segments_from_transcript(
                transcript, video_path, num_segments, min_duration, max_duration
            )
        else:
            # Fall back to evenly spaced segments
            segments = await self._find_segments_by_intervals(
                video_path, num_segments, min_duration, max_duration
            )

        self.segments = segments
        return segments

    async def _find_segments_from_transcript(
        self,
        transcript: str,
        video_path: str,
        num_segments: int,
        min_duration: int,
        max_duration: int
    ) -> List[ShortSegment]:
        """Find segments based on transcript analysis."""
        segments = []

        # Parse transcript for timestamps and text
        # Expected format: "[00:00] text" or SRT format
        lines = transcript.split('\n')

        potential_segments = []

        for i, line in enumerate(lines):
            line_lower = line.lower()

            # Check for hook indicators
            for hook in self.HOOK_INDICATORS:
                if hook in line_lower:
                    # Try to extract timestamp
                    timestamp = self._extract_timestamp(line)
                    if timestamp is not None:
                        score = self._calculate_segment_score(line, i, len(lines))
                        potential_segments.append({
                            "start_time": timestamp,
                            "hook_text": line,
                            "score": score,
                            "line_index": i
                        })

        # Sort by score and take top segments
        potential_segments.sort(key=lambda x: x["score"], reverse=True)

        for seg_info in potential_segments[:num_segments]:
            segment = ShortSegment(
                start_time=seg_info["start_time"],
                end_time=min(seg_info["start_time"] + max_duration, self._get_video_duration(video_path)),
                duration=min(max_duration, self._get_video_duration(video_path) - seg_info["start_time"]),
                score=seg_info["score"],
                hook_text=seg_info["hook_text"][:100],
                source_video=video_path
            )
            segments.append(segment)

        # If not enough segments found, add interval-based ones
        if len(segments) < num_segments:
            additional = await self._find_segments_by_intervals(
                video_path,
                num_segments - len(segments),
                min_duration,
                max_duration,
                exclude_times=[s.start_time for s in segments]
            )
            segments.extend(additional)

        return segments[:num_segments]

    async def _find_segments_by_intervals(
        self,
        video_path: str,
        num_segments: int,
        min_duration: int,
        max_duration: int,
        exclude_times: List[float] = None
    ) -> List[ShortSegment]:
        """Find segments at evenly spaced intervals."""
        segments = []
        exclude_times = exclude_times or []

        video_duration = self._get_video_duration(video_path)

        if video_duration <= max_duration:
            # Video is short enough to be one segment
            return [ShortSegment(
                start_time=0,
                end_time=video_duration,
                duration=video_duration,
                score=50.0,
                hook_text="Full video",
                source_video=video_path
            )]

        # Calculate segment positions
        # Good positions: intro (5-10%), middle hooks (30%, 50%, 70%), climax (80-90%)
        positions = [0.05, 0.30, 0.50, 0.70, 0.85]

        for i, pos in enumerate(positions[:num_segments]):
            start_time = video_duration * pos

            # Skip if too close to excluded times
            if any(abs(start_time - et) < max_duration for et in exclude_times):
                continue

            segment = ShortSegment(
                start_time=start_time,
                end_time=min(start_time + max_duration, video_duration),
                duration=min(max_duration, video_duration - start_time),
                score=70.0 - (i * 5),  # Earlier segments score higher
                hook_text=f"Segment at {int(pos*100)}%",
                source_video=video_path
            )
            segments.append(segment)

        return segments[:num_segments]

    def _extract_timestamp(self, line: str) -> Optional[float]:
        """Extract timestamp from a line of text."""
        import re

        # Pattern: [00:00] or [00:00:00] or 00:00 or 00:00:00
        patterns = [
            r'\[(\d{1,2}):(\d{2})(?::(\d{2}))?\]',
            r'(\d{1,2}):(\d{2})(?::(\d{2}))?',
        ]

        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                groups = match.groups()
                hours = 0
                if groups[2]:  # Has hours
                    hours = int(groups[0])
                    minutes = int(groups[1])
                    seconds = int(groups[2])
                else:
                    minutes = int(groups[0])
                    seconds = int(groups[1])

                return hours * 3600 + minutes * 60 + seconds

        return None

    def _calculate_segment_score(self, text: str, line_index: int, total_lines: int) -> float:
        """Calculate virality score for a potential segment."""
        score = 50.0

        text_lower = text.lower()

        # Bonus for hook phrases
        hook_phrases = ["secret", "truth", "mistake", "never", "always", "shocking", "revealed"]
        for phrase in hook_phrases:
            if phrase in text_lower:
                score += 10

        # Bonus for numbers/statistics
        import re
        if re.search(r'\d+', text):
            score += 5

        # Bonus for questions
        if "?" in text:
            score += 5

        # Position bonus (early content often has good hooks)
        position_ratio = line_index / max(total_lines, 1)
        if position_ratio < 0.2:
            score += 10
        elif position_ratio < 0.4:
            score += 5

        return min(100.0, score)

    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds."""
        if not MOVIEPY_AVAILABLE:
            return 0.0

        try:
            clip = VideoFileClip(video_path)
            duration = clip.duration
            clip.close()
            return duration
        except Exception as e:
            logger.warning(f"Could not get video duration: {e}")
            return 0.0


class MultiPlatformDistributor:
    """
    Main class for multi-platform content distribution.

    Handles repurposing long-form content to short-form formats,
    auto-resizing, metadata generation, and cross-posting scheduling.
    """

    def __init__(self, output_dir: str = "output/multiplatform"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.resizer = VideoResizer(str(self.output_dir / "resized"))
        self.segment_extractor = SegmentExtractor()

        # Cross-post schedules
        self.schedules: List[CrossPostSchedule] = []

        logger.info(f"MultiPlatformDistributor initialized. Output: {self.output_dir}")

    def create_shorts(
        self,
        video_path: str,
        title: str,
        niche: str,
        num_shorts: int = 3,
        transcript: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Create YouTube Shorts from a long-form video.

        Args:
            video_path: Path to source video
            title: Video title for metadata
            niche: Content niche
            num_shorts: Number of Shorts to create
            transcript: Optional transcript for better segment selection

        Returns:
            List of created Short information dicts
        """
        logger.info(f"Creating {num_shorts} Shorts from {video_path}")

        # Find best segments
        segments = asyncio.run(
            self.segment_extractor.find_best_segments(
                video_path, transcript, num_shorts, 15, 60
            )
        )

        shorts = []
        spec = PLATFORM_SPECS[Platform.YOUTUBE_SHORTS]
        template = METADATA_TEMPLATES[Platform.YOUTUBE_SHORTS]

        for i, segment in enumerate(segments):
            try:
                # Extract segment
                segment_path = self._extract_segment(
                    video_path, segment.start_time, segment.end_time, i
                )

                # Resize to portrait
                portrait_path = self.resizer.convert_landscape_to_portrait(
                    segment_path,
                    add_blur_background=True
                )

                # Generate metadata
                short_title = f"{title} Part {i+1}"
                metadata = {
                    "title": template.generate_title(short_title, niche),
                    "description": template.generate_description(short_title, niche),
                    "hashtags": template.get_hashtags(niche, 10),
                }

                shorts.append({
                    "video_path": portrait_path,
                    "segment": segment.to_dict(),
                    "metadata": metadata,
                    "platform": Platform.YOUTUBE_SHORTS.value,
                    "index": i + 1,
                })

                logger.info(f"Created Short {i+1}: {portrait_path}")

            except Exception as e:
                logger.error(f"Failed to create Short {i+1}: {e}")

        return shorts

    def _extract_segment(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        index: int
    ) -> str:
        """Extract a segment from a video."""
        if not MOVIEPY_AVAILABLE:
            raise ImportError("MoviePy required for segment extraction")

        output_path = str(self.output_dir / f"segment_{index}_{int(start_time)}_{int(end_time)}.mp4")

        clip = VideoFileClip(video_path)
        segment = clip.subclip(start_time, min(end_time, clip.duration))

        segment.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            bitrate="8M",
            fps=30,
            preset="medium"
        )

        clip.close()
        segment.close()

        return output_path

    async def export_for_platform(
        self,
        video_path: str,
        platform: Platform,
        title: str,
        niche: str,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> PlatformExport:
        """
        Export a video for a specific platform.

        Args:
            video_path: Path to source video
            platform: Target platform
            title: Video title
            niche: Content niche
            custom_metadata: Optional custom metadata overrides

        Returns:
            PlatformExport object with all export details
        """
        logger.info(f"Exporting {video_path} for {platform.value}")

        spec = PLATFORM_SPECS[platform]
        template = METADATA_TEMPLATES.get(platform)
        warnings = []

        # Get video info
        if MOVIEPY_AVAILABLE:
            clip = VideoFileClip(video_path)
            source_duration = clip.duration
            source_resolution = (clip.w, clip.h)
            clip.close()
        else:
            source_duration = 0
            source_resolution = (0, 0)

        # Check duration
        if source_duration > spec.max_duration:
            warnings.append(f"Video duration ({source_duration:.1f}s) exceeds platform max ({spec.max_duration}s)")

        # Resize if needed
        target_resolution = spec.resolution
        if source_resolution != target_resolution:
            logger.info(f"Resizing from {source_resolution} to {target_resolution}")

            if spec.aspect_ratio == AspectRatio.PORTRAIT:
                output_path = self.resizer.convert_landscape_to_portrait(
                    video_path,
                    add_blur_background=True
                )
            else:
                output_path = self.resizer.resize_video(
                    video_path,
                    target_resolution
                )
        else:
            output_path = video_path

        # Generate metadata
        if template:
            generated_title = template.generate_title(title, niche)
            generated_desc = template.generate_description(title, niche)
            hashtags = template.get_hashtags(niche, spec.hashtag_limit)
        else:
            generated_title = title
            generated_desc = title
            hashtags = []

        # Apply custom metadata overrides
        if custom_metadata:
            generated_title = custom_metadata.get("title", generated_title)
            generated_desc = custom_metadata.get("description", generated_desc)
            hashtags = custom_metadata.get("hashtags", hashtags)

        # Check title length
        if len(generated_title) > spec.caption_limit:
            warnings.append(f"Title too long for platform ({len(generated_title)} > {spec.caption_limit})")

        # Get file size
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024) if os.path.exists(output_path) else 0
        if file_size_mb > spec.max_file_size_mb:
            warnings.append(f"File size ({file_size_mb:.1f}MB) exceeds platform max ({spec.max_file_size_mb}MB)")

        export = PlatformExport(
            platform=platform,
            video_path=output_path,
            thumbnail_path=None,  # Could generate platform-specific thumbnail
            title=generated_title,
            description=generated_desc,
            hashtags=hashtags,
            duration=source_duration,
            resolution=target_resolution,
            file_size_mb=file_size_mb,
            ready_to_upload=len(warnings) == 0,
            warnings=warnings
        )

        logger.success(f"Export complete for {platform.value}: {output_path}")
        return export

    async def export_all_platforms(
        self,
        video_path: str,
        title: str,
        niche: str,
        platforms: Optional[List[Platform]] = None
    ) -> Dict[str, PlatformExport]:
        """
        Export video for all specified platforms.

        Args:
            video_path: Path to source video
            title: Video title
            niche: Content niche
            platforms: List of target platforms (default: all short-form)

        Returns:
            Dict mapping platform name to PlatformExport
        """
        if platforms is None:
            platforms = [
                Platform.YOUTUBE_SHORTS,
                Platform.TIKTOK,
                Platform.INSTAGRAM_REELS,
            ]

        exports = {}

        for platform in platforms:
            try:
                export = await self.export_for_platform(
                    video_path, platform, title, niche
                )
                exports[platform.value] = export
            except Exception as e:
                logger.error(f"Failed to export for {platform.value}: {e}")

        return exports

    def create_cross_post_schedule(
        self,
        video_path: str,
        title: str,
        niche: str,
        base_time: Optional[datetime] = None,
        platform_delays: Optional[Dict[str, int]] = None
    ) -> CrossPostSchedule:
        """
        Create a cross-posting schedule for multiple platforms.

        Staggers posts across platforms for maximum reach without
        appearing spammy or cannibalizing views.

        Args:
            video_path: Path to source video
            title: Video title
            niche: Content niche
            base_time: Starting time for schedule (default: now)
            platform_delays: Dict of platform -> minutes delay (default: staggered)

        Returns:
            CrossPostSchedule object
        """
        base_time = base_time or datetime.now()

        # Default delays: stagger posts over 2 hours
        if platform_delays is None:
            platform_delays = {
                Platform.YOUTUBE_SHORTS.value: 0,      # Immediate
                Platform.TIKTOK.value: 30,             # 30 minutes later
                Platform.INSTAGRAM_REELS.value: 90,    # 1.5 hours later
            }

        # Export for all platforms
        exports = asyncio.run(self.export_all_platforms(video_path, title, niche))

        # Create schedule
        schedule = {}
        status = {}

        for platform_name, delay_minutes in platform_delays.items():
            if platform_name in exports:
                scheduled_time = base_time + timedelta(minutes=delay_minutes)
                schedule[platform_name] = scheduled_time
                status[platform_name] = "pending"

        cross_post = CrossPostSchedule(
            video_title=title,
            source_video=video_path,
            platform_exports=exports,
            schedule=schedule,
            status=status
        )

        self.schedules.append(cross_post)

        # Save schedule to file
        schedule_path = self.output_dir / f"schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(schedule_path, 'w') as f:
            json.dump(cross_post.to_dict(), f, indent=2)

        logger.success(f"Cross-post schedule created: {schedule_path}")
        return cross_post

    def get_platform_requirements(self, platform: Platform) -> Dict[str, Any]:
        """Get the requirements/specifications for a platform."""
        spec = PLATFORM_SPECS.get(platform)
        if spec:
            return spec.to_dict()
        return {}

    def validate_for_platform(self, video_path: str, platform: Platform) -> Dict[str, Any]:
        """
        Validate a video for a specific platform.

        Returns:
            Dict with validation results and any issues found
        """
        spec = PLATFORM_SPECS[platform]
        issues = []
        warnings = []

        # Check file exists
        if not os.path.exists(video_path):
            return {
                "valid": False,
                "issues": ["Video file not found"],
                "warnings": [],
            }

        # Check file size
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        if file_size_mb > spec.max_file_size_mb:
            issues.append(f"File size ({file_size_mb:.1f}MB) exceeds max ({spec.max_file_size_mb}MB)")

        # Check duration and resolution with MoviePy
        if MOVIEPY_AVAILABLE:
            try:
                clip = VideoFileClip(video_path)
                duration = clip.duration
                resolution = (clip.w, clip.h)
                clip.close()

                if duration > spec.max_duration:
                    issues.append(f"Duration ({duration:.1f}s) exceeds max ({spec.max_duration}s)")
                elif duration < spec.min_duration:
                    issues.append(f"Duration ({duration:.1f}s) below min ({spec.min_duration}s)")

                if resolution != spec.resolution:
                    warnings.append(f"Resolution {resolution} differs from optimal {spec.resolution}")

            except Exception as e:
                warnings.append(f"Could not analyze video: {e}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "platform": platform.value,
            "file_size_mb": file_size_mb,
        }


# Convenience functions
def create_shorts_from_video(
    video_path: str,
    title: str,
    niche: str,
    num_shorts: int = 3
) -> List[Dict[str, Any]]:
    """Convenience function to create Shorts from a video."""
    distributor = MultiPlatformDistributor()
    return distributor.create_shorts(video_path, title, niche, num_shorts)


async def export_for_all_platforms(
    video_path: str,
    title: str,
    niche: str
) -> Dict[str, PlatformExport]:
    """Convenience function to export for all platforms."""
    distributor = MultiPlatformDistributor()
    return await distributor.export_all_platforms(video_path, title, niche)


if __name__ == "__main__":
    import sys

    print("\n" + "=" * 60)
    print("MULTI-PLATFORM DISTRIBUTION MODULE")
    print("=" * 60 + "\n")

    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        title = sys.argv[2] if len(sys.argv) > 2 else "Test Video"
        niche = sys.argv[3] if len(sys.argv) > 3 else "finance"

        distributor = MultiPlatformDistributor()

        # Show platform requirements
        print("Platform Requirements:")
        print("-" * 40)
        for platform in [Platform.YOUTUBE_SHORTS, Platform.TIKTOK, Platform.INSTAGRAM_REELS]:
            req = distributor.get_platform_requirements(platform)
            print(f"\n{platform.value}:")
            print(f"  Resolution: {req['resolution']}")
            print(f"  Max Duration: {req['max_duration']}s")
            print(f"  Max File Size: {req['max_file_size_mb']}MB")

        # Validate video
        print(f"\nValidating {video_path}...")
        for platform in [Platform.YOUTUBE_SHORTS, Platform.TIKTOK, Platform.INSTAGRAM_REELS]:
            result = distributor.validate_for_platform(video_path, platform)
            status = "VALID" if result["valid"] else "INVALID"
            print(f"\n{platform.value}: {status}")
            for issue in result.get("issues", []):
                print(f"  [ERROR] {issue}")
            for warning in result.get("warnings", []):
                print(f"  [WARN] {warning}")

        # Create Shorts
        print(f"\nCreating Shorts from {video_path}...")
        shorts = distributor.create_shorts(video_path, title, niche, 2)
        print(f"\nCreated {len(shorts)} Shorts:")
        for short in shorts:
            print(f"  - {short['video_path']}")
    else:
        print("Usage: python -m src.social.multi_platform <video_path> [title] [niche]")
        print("\nSupported platforms:")
        for platform in Platform:
            print(f"  - {platform.value}")
