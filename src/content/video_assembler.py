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
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from loguru import logger

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

    def __init__(
        self,
        resolution: Tuple[int, int] = (1920, 1080),
        fps: int = 30,
        background_color: Tuple[int, int, int] = (20, 20, 30)
    ):
        """
        Initialize the video assembler.

        Args:
            resolution: Video resolution (width, height)
            fps: Frames per second
            background_color: RGB background color
        """
        self.resolution = resolution
        self.fps = fps
        self.background_color = background_color
        self.width, self.height = resolution

        logger.info(f"VideoAssembler: {resolution[0]}x{resolution[1]} @ {fps}fps")

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
        subtitle_file: Optional[str] = None
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

        Returns:
            Path to created video file
        """
        logger.info(f"Creating video from audio: {audio_file}")

        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        # Load audio
        audio = AudioFileClip(audio_file)
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

        video.write_videofile(
            output_file,
            fps=self.fps,
            codec="libx264",
            audio_codec="aac",
            preset="medium",
            threads=4,
            logger=None  # Suppress moviepy's verbose output
        )

        # Clean up
        video.close()
        audio.close()

        logger.success(f"Video created: {output_file}")
        return output_file

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

        video = VideoFileClip(video_file)
        music = AudioFileClip(music_file)

        # Loop music if shorter than video
        if music.duration < video.duration:
            loops = int(video.duration / music.duration) + 1
            music = concatenate_videoclips([music] * loops)

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

        # Write output
        video.write_videofile(
            output_file,
            fps=self.fps,
            codec="libx264",
            audio_codec="aac",
            logger=None
        )

        video.close()
        music.close()

        logger.success(f"Music added: {output_file}")
        return output_file


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
