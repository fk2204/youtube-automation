"""
Fast Video Generator using FFmpeg directly.

Much faster than MoviePy for simple video generation.
Creates videos from audio + background in seconds.

Usage:
    generator = FastVideoGenerator()
    generator.create_video(
        audio_file="narration.mp3",
        output_file="video.mp4",
        title="My Tutorial"
    )

Audio Enhancement (optional):
    generator.create_video(
        audio_file="narration.mp3",
        output_file="video.mp4",
        normalize_audio=True  # Normalize to -14 LUFS
    )

Subtitles (optional):
    generator.create_video(
        audio_file="narration.mp3",
        output_file="video.mp4",
        script_text="Your narration text...",
        subtitles_enabled=True,
        subtitle_style="regular"  # or "shorts", "minimal", "cinematic"
    )
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    raise ImportError("Please install pillow: pip install pillow")

# Import audio processor for normalization
try:
    from .audio_processor import AudioProcessor
    AUDIO_PROCESSOR_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSOR_AVAILABLE = False
    logger.debug("AudioProcessor not available - audio enhancement disabled")

# Import subtitle generator
try:
    from .subtitles import SubtitleGenerator, SubtitleTrack, SUBTITLE_STYLES
    SUBTITLES_AVAILABLE = True
except ImportError:
    SUBTITLES_AVAILABLE = False
    logger.debug("SubtitleGenerator not available - subtitles disabled")


class FastVideoGenerator:
    """
    Fast video generation using FFmpeg.

    Much faster than MoviePy for:
    - Simple background + audio videos
    - Text overlay videos
    - Thumbnail generation
    """

    def __init__(
        self,
        resolution: Tuple[int, int] = (1920, 1080),
        fps: int = 30,
        background_color: str = "#14141e"
    ):
        self.resolution = resolution
        self.fps = fps
        self.background_color = background_color
        self.width, self.height = resolution

        # Check FFmpeg
        self.ffmpeg = self._find_ffmpeg()
        if self.ffmpeg:
            logger.info(f"FastVideoGenerator ready (FFmpeg: {self.ffmpeg})")
        else:
            logger.warning("FFmpeg not found. Install from https://ffmpeg.org/download.html")

        # Initialize audio processor for normalization
        self.audio_processor = None
        if AUDIO_PROCESSOR_AVAILABLE:
            try:
                self.audio_processor = AudioProcessor()
                logger.debug("AudioProcessor initialized for audio enhancement")
            except Exception as e:
                logger.debug(f"AudioProcessor initialization failed: {e}")

        # Initialize subtitle generator
        self.subtitle_generator = None
        if SUBTITLES_AVAILABLE:
            try:
                self.subtitle_generator = SubtitleGenerator()
                logger.debug("SubtitleGenerator initialized for caption support")
            except Exception as e:
                logger.debug(f"SubtitleGenerator initialization failed: {e}")

    def _find_ffmpeg(self) -> Optional[str]:
        """Find FFmpeg executable."""
        # Check if ffmpeg is in PATH
        if shutil.which("ffmpeg"):
            return "ffmpeg"

        # Common Windows locations
        common_paths = [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            os.path.expanduser("~\\ffmpeg\\bin\\ffmpeg.exe"),
        ]

        # Check WinGet installation paths
        winget_base = os.path.expanduser("~\\AppData\\Local\\Microsoft\\WinGet\\Packages")
        if os.path.exists(winget_base):
            for folder in os.listdir(winget_base):
                if "FFmpeg" in folder:
                    # Search for ffmpeg.exe in this package
                    package_path = os.path.join(winget_base, folder)
                    for root, dirs, files in os.walk(package_path):
                        if "ffmpeg.exe" in files:
                            common_paths.insert(0, os.path.join(root, "ffmpeg.exe"))

        for path in common_paths:
            if os.path.exists(path):
                return path

        return None

    def create_background_image(
        self,
        output_path: str,
        title: Optional[str] = None,
        subtitle: Optional[str] = None
    ) -> str:
        """Create a background image with optional text."""
        # Parse hex color
        color = self.background_color.lstrip('#')
        rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))

        # Create image
        img = Image.new('RGB', self.resolution, rgb)
        draw = ImageDraw.Draw(img)

        # Add gradient effect (darker at edges)
        for i in range(50):
            alpha = int(255 * (1 - i / 50) * 0.3)
            draw.rectangle(
                [i, i, self.width - i, self.height - i],
                outline=tuple(max(0, c - alpha // 3) for c in rgb)
            )

        # Add title text if provided
        if title:
            try:
                # Try to load a nice font
                font_size = 72
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    try:
                        font = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", font_size)
                    except:
                        font = ImageFont.load_default()

                # Calculate text position
                bbox = draw.textbbox((0, 0), title, font=font)
                text_width = bbox[2] - bbox[0]
                x = (self.width - text_width) // 2
                y = self.height // 2 - 50

                # Draw shadow
                draw.text((x + 3, y + 3), title, fill="#000000", font=font)
                # Draw text
                draw.text((x, y), title, fill="#ffffff", font=font)

            except Exception as e:
                logger.warning(f"Could not add title: {e}")

        # Add subtitle
        if subtitle:
            try:
                font_size = 36
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()

                bbox = draw.textbbox((0, 0), subtitle, font=font)
                text_width = bbox[2] - bbox[0]
                x = (self.width - text_width) // 2
                y = self.height // 2 + 50

                draw.text((x, y), subtitle, fill="#aaaaaa", font=font)
            except:
                pass

        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, quality=95)
        logger.debug(f"Background image saved: {output_path}")

        return output_path

    def create_video(
        self,
        audio_file: str,
        output_file: str,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        background_image: Optional[str] = None,
        normalize_audio: bool = False,
        background_music: Optional[str] = None,
        music_volume: float = 0.15,
        script_text: Optional[str] = None,
        subtitles_enabled: bool = False,
        subtitle_style: str = "regular",
        niche: Optional[str] = None
    ) -> Optional[str]:
        """
        Create a video from audio file.

        Args:
            audio_file: Path to audio file (MP3, WAV)
            output_file: Output video path (MP4)
            title: Optional title overlay
            subtitle: Optional subtitle
            background_image: Custom background (or auto-generate)
            normalize_audio: Normalize audio to YouTube's -14 LUFS standard
            background_music: Optional background music file to mix with voice
            music_volume: Volume level for background music (0.0-1.0, default 0.15)
            script_text: Script/narration text for subtitle generation
            subtitles_enabled: Whether to burn subtitles into video
            subtitle_style: Subtitle style ("regular", "shorts", "minimal", "cinematic")
            niche: Content niche for subtitle styling ("finance", "psychology", etc.)

        Returns:
            Path to created video or None on failure
        """
        if not self.ffmpeg:
            logger.error("FFmpeg not available")
            return None

        if not os.path.exists(audio_file):
            logger.error(f"Audio file not found: {audio_file}")
            return None

        # Create output directory
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        # Process audio (normalization and/or music mixing)
        processed_audio = audio_file
        temp_audio = None

        if self.audio_processor and (normalize_audio or background_music):
            temp_audio = str(Path(output_file).with_suffix('.processed_audio.mp3'))

            if background_music and os.path.exists(background_music):
                # Mix with background music (this also normalizes)
                logger.info(f"Mixing audio with background music at {music_volume*100:.0f}% volume")
                result = self.audio_processor.mix_with_background_music(
                    voice_file=audio_file,
                    music_file=background_music,
                    output_file=temp_audio,
                    music_volume=music_volume,
                    normalize_before_mix=True
                )
                if result:
                    processed_audio = result
            elif normalize_audio:
                # Just normalize
                logger.info("Normalizing audio to YouTube's -14 LUFS standard")
                result = self.audio_processor.normalize_audio(audio_file, temp_audio)
                if result:
                    processed_audio = result

        # Create or use background image
        temp_bg = None
        if not background_image:
            temp_bg = str(Path(output_file).with_suffix('.bg.png'))
            background_image = self.create_background_image(temp_bg, title, subtitle)

        try:
            logger.info(f"Creating video: {output_file}")

            # FFmpeg command to create video from image + audio
            cmd = [
                self.ffmpeg,
                '-y',  # Overwrite output
                '-loop', '1',  # Loop image
                '-i', background_image,  # Input image
                '-i', processed_audio,  # Input audio (may be normalized/mixed)
                '-c:v', 'libx264',  # Video codec
                '-tune', 'stillimage',  # Optimize for still image
                '-crf', '23',  # Constant rate factor for quality
                '-b:v', '8000k',  # Video bitrate (8 Mbps for YouTube 1080p)
                '-c:a', 'aac',  # Audio codec
                '-b:a', '256k',  # Audio bitrate (improved from 192k)
                '-pix_fmt', 'yuv420p',  # Pixel format
                '-shortest',  # End when audio ends
                '-vf', f'scale={self.width}:{self.height}',  # Resolution
                '-r', str(self.fps),  # Frame rate
                output_file
            ]

            # Run FFmpeg
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr[:500]}")
                return None

            logger.success(f"Video created: {output_file}")

            # Apply subtitles if enabled
            if subtitles_enabled and script_text and self.subtitle_generator:
                logger.info("Adding burned-in subtitles...")
                try:
                    # Get audio duration for subtitle timing
                    audio_duration = self.get_audio_duration(audio_file)
                    if not audio_duration:
                        audio_duration = len(script_text.split()) / 2.5  # Fallback estimate

                    # Generate subtitle track
                    style_config = self.subtitle_generator.get_style(subtitle_style, niche)
                    max_chars = style_config.get("max_chars", 50)
                    track = self.subtitle_generator.sync_subtitles_with_audio(
                        script_text, audio_file, max_chars
                    )

                    if track.cues:
                        # Burn subtitles into video
                        temp_video = str(Path(output_file).with_suffix('.temp_no_subs.mp4'))
                        os.rename(output_file, temp_video)

                        result = self.subtitle_generator.burn_subtitles(
                            temp_video,
                            track,
                            output_file,
                            style=subtitle_style,
                            niche=niche
                        )

                        # Clean up temp video
                        if os.path.exists(temp_video):
                            try:
                                os.remove(temp_video)
                            except:
                                pass

                        if result:
                            logger.success(f"Subtitles burned into video: {output_file}")
                        else:
                            # Restore original video if subtitle burning failed
                            logger.warning("Subtitle burning failed, video saved without subtitles")
                except Exception as e:
                    logger.warning(f"Subtitle processing failed: {e}")

            return output_file

        except subprocess.TimeoutExpired:
            logger.error("FFmpeg timed out")
            return None
        except Exception as e:
            logger.error(f"Video creation failed: {e}")
            return None
        finally:
            # Clean up temp background
            if temp_bg and os.path.exists(temp_bg):
                try:
                    os.remove(temp_bg)
                except:
                    pass
            # Clean up temp audio
            if temp_audio and os.path.exists(temp_audio):
                try:
                    os.remove(temp_audio)
                except:
                    pass

    def create_thumbnail(
        self,
        output_file: str,
        title: str,
        subtitle: Optional[str] = None,
        background_color: str = "#1e1e3f"
    ) -> str:
        """Create a YouTube thumbnail (1280x720)."""
        size = (1280, 720)

        # Parse color
        color = background_color.lstrip('#')
        rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))

        img = Image.new('RGB', size, rgb)
        draw = ImageDraw.Draw(img)

        # Add gradient
        for i in range(40):
            alpha = int(255 * (1 - i / 40) * 0.4)
            draw.rectangle(
                [i, i, size[0] - i, size[1] - i],
                outline=tuple(max(0, c - alpha // 2) for c in rgb)
            )

        # Title
        try:
            font_size = 64
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("C:\\Windows\\Fonts\\arialbd.ttf", font_size)
                except:
                    font = ImageFont.load_default()

            # Wrap title if too long
            max_chars = 25
            if len(title) > max_chars:
                words = title.split()
                lines = []
                current_line = ""
                for word in words:
                    if len(current_line + " " + word) <= max_chars:
                        current_line += (" " + word if current_line else word)
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                if current_line:
                    lines.append(current_line)
                title = "\n".join(lines[:2])

            # Draw title with shadow
            bbox = draw.textbbox((0, 0), title, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (size[0] - text_width) // 2
            y = (size[1] - text_height) // 2 - 30

            draw.text((x + 4, y + 4), title, fill="#000000", font=font)
            draw.text((x, y), title, fill="#ffffff", font=font)

        except Exception as e:
            logger.warning(f"Font error: {e}")

        # Subtitle
        if subtitle:
            try:
                font = ImageFont.truetype("arial.ttf", 32)
                bbox = draw.textbbox((0, 0), subtitle, font=font)
                x = (size[0] - (bbox[2] - bbox[0])) // 2
                y = size[1] // 2 + 60
                draw.text((x, y), subtitle, fill="#cccccc", font=font)
            except:
                pass

        # Save
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        img.save(output_file, quality=95)
        logger.success(f"Thumbnail created: {output_file}")

        return output_file

    def get_audio_duration(self, audio_file: str) -> Optional[float]:
        """Get duration of audio file in seconds."""
        if not self.ffmpeg:
            return None

        try:
            ffprobe = self.ffmpeg.replace('ffmpeg', 'ffprobe')
            cmd = [
                ffprobe,
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                audio_file
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return float(result.stdout.strip())
        except:
            return None


# Example usage
if __name__ == "__main__":
    generator = FastVideoGenerator()

    # Create test thumbnail
    print("\nCreating test thumbnail...")
    generator.create_thumbnail(
        output_file="output/test_thumb.png",
        title="Python Tutorial",
        subtitle="Learn Python in 10 Minutes"
    )

    # If we have test audio, create video
    if os.path.exists("output/test_voice.mp3"):
        print("\nCreating test video...")
        generator.create_video(
            audio_file="output/test_voice.mp3",
            output_file="output/test_video.mp4",
            title="Test Video",
            subtitle="Created with FFmpeg"
        )
    else:
        print("\nNo test audio found. Run TTS test first.")
