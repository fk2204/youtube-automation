"""
Professional Video Generator

Creates high-quality videos with:
- Stock footage B-roll
- Text overlays and animations
- Smooth transitions
- Subtitles/captions
- Dynamic visual effects

Usage:
    from src.content.video_pro import ProVideoGenerator

    generator = ProVideoGenerator()
    generator.create_video(
        audio_file="narration.mp3",
        script=script_object,
        output_file="video.mp4"
    )
"""

import os
import re
import json
import subprocess
import tempfile
import random
from pathlib import Path
import shutil
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

try:
    from moviepy.editor import (
        VideoFileClip, AudioFileClip, ImageClip, TextClip,
        CompositeVideoClip, concatenate_videoclips, ColorClip
    )
    from moviepy.video.fx.all import fadein, fadeout, resize
    from moviepy.config import change_settings
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    logger.warning("MoviePy not installed. Run: pip install moviepy")

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    raise ImportError("Please install pillow: pip install pillow")


@dataclass
class VideoSection:
    """A section of the video with timing and content."""
    start_time: float
    end_time: float
    text: str
    keywords: List[str]
    visual_type: str = "stock"  # stock, text, image


class ProVideoGenerator:
    """
    Professional video generator with stock footage and effects.
    """

    # Color schemes for different niches
    COLOR_SCHEMES = {
        "finance": {
            "primary": "#00d4aa",
            "secondary": "#1a1a2e",
            "accent": "#ffd700",
            "text": "#ffffff"
        },
        "psychology": {
            "primary": "#9b59b6",
            "secondary": "#0f0f1a",
            "accent": "#e74c3c",
            "text": "#ffffff"
        },
        "storytelling": {
            "primary": "#e74c3c",
            "secondary": "#0d0d0d",
            "accent": "#f39c12",
            "text": "#ffffff"
        },
        "default": {
            "primary": "#3498db",
            "secondary": "#1a1a2e",
            "accent": "#e74c3c",
            "text": "#ffffff"
        }
    }

    def __init__(
        self,
        resolution: Tuple[int, int] = (1920, 1080),
        fps: int = 30,
        use_gpu: bool = True,
        prefer_quality: bool = False
    ):
        self.resolution = resolution
        self.width, self.height = resolution
        self.fps = fps

        # Find FFmpeg and ffprobe
        self.ffmpeg = self._find_ffmpeg()
        self.ffprobe = self._find_ffprobe() if self.ffmpeg else None

        # Initialize GPU acceleration
        self.use_gpu = use_gpu
        self.gpu = None
        if use_gpu and self.ffmpeg:
            try:
                from ..utils.gpu_utils import GPUAccelerator
                self.gpu = GPUAccelerator(self.ffmpeg, prefer_quality)
                if self.gpu.is_available():
                    logger.success(f"GPU acceleration enabled: {self.gpu.get_gpu_type().value}")
                else:
                    logger.info("GPU not available, using CPU encoding")
            except Exception as e:
                logger.warning(f"GPU initialization failed: {e}")
                self.gpu = None

        # Initialize stock footage client
        try:
            from .stock_footage import StockFootage
            self.stock = StockFootage()
        except Exception as e:
            logger.warning(f"Stock footage unavailable: {e}")
            self.stock = None

        # Temp directory for intermediate files
        self.temp_dir = Path(tempfile.gettempdir()) / "video_pro"
        self.temp_dir.mkdir(exist_ok=True)

        gpu_status = "GPU" if self.gpu and self.gpu.is_available() else "CPU"
        logger.info(f"ProVideoGenerator initialized ({self.width}x{self.height}, {gpu_status})")

    def _run_ffmpeg(self, cmd: list, timeout: int = 120) -> bool:
        """Run FFmpeg command with error handling."""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr[:200]}")
                return False
            return True
        except subprocess.TimeoutExpired:
            logger.error(f"FFmpeg command timed out after {timeout}s")
            return False
        except Exception as e:
            logger.error(f"FFmpeg command failed: {e}")
            return False

    def _get_encoder_args(self, bitrate: str = "8M", quality: int = 23) -> List[str]:
        """Get encoder arguments based on GPU availability."""
        if self.gpu and self.gpu.is_available():
            return self.gpu.get_ffmpeg_args(
                preset="fast",
                bitrate=bitrate,
                quality=quality,
                width=self.width,
                height=self.height
            )
        else:
            # CPU fallback
            return [
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", str(quality),
                "-b:v", bitrate,
                "-pix_fmt", "yuv420p"
            ]

    def _find_ffmpeg(self) -> Optional[str]:
        """Find FFmpeg executable."""
        if shutil.which("ffmpeg"):
            return "ffmpeg"

        # Check WinGet paths
        winget_base = os.path.expanduser("~\\AppData\\Local\\Microsoft\\WinGet\\Packages")
        if os.path.exists(winget_base):
            for folder in os.listdir(winget_base):
                if "FFmpeg" in folder:
                    package_path = os.path.join(winget_base, folder)
                    for root, dirs, files in os.walk(package_path):
                        if "ffmpeg.exe" in files:
                            return os.path.join(root, "ffmpeg.exe")
        return None

    def _find_ffprobe(self) -> Optional[str]:
        """Find ffprobe executable (companion to ffmpeg)."""
        # Try system PATH first
        if shutil.which("ffprobe"):
            return "ffprobe"

        # If ffmpeg is found, ffprobe is usually in the same directory
        if self.ffmpeg:
            ffmpeg_dir = os.path.dirname(self.ffmpeg)
            ffprobe_path = os.path.join(ffmpeg_dir, 'ffprobe.exe' if os.name == 'nt' else 'ffprobe')
            if os.path.exists(ffprobe_path):
                return ffprobe_path

        return None

    def create_text_clip(
        self,
        text: str,
        duration: float,
        font_size: int = 60,
        color: str = "#ffffff",
        bg_color: str = None,
        position: str = "center",
        animation: str = "fade"
    ) -> str:
        """
        Create a video clip with animated text.

        Returns path to the generated clip.
        """
        output_path = self.temp_dir / f"text_{hash(text)}.mp4"

        # Create text image
        img = Image.new('RGBA', self.resolution, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Load font
        try:
            font = ImageFont.truetype("C:\\Windows\\Fonts\\arialbd.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()

        # Wrap text
        max_width = self.width - 200
        lines = []
        words = text.split()
        current_line = ""

        for word in words:
            test_line = f"{current_line} {word}".strip()
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        text = "\n".join(lines)

        # Calculate position
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        if position == "center":
            x = (self.width - text_width) // 2
            y = (self.height - text_height) // 2
        elif position == "bottom":
            x = (self.width - text_width) // 2
            y = self.height - text_height - 100
        else:  # top
            x = (self.width - text_width) // 2
            y = 100

        # Draw text with shadow
        shadow_offset = 3
        draw.text((x + shadow_offset, y + shadow_offset), text, fill="#000000", font=font)
        draw.text((x, y), text, fill=color, font=font)

        # Save frame
        frame_path = self.temp_dir / f"text_frame_{hash(text)}.png"
        img.save(str(frame_path))

        # Create video with FFmpeg
        if self.ffmpeg:
            cmd = [
                self.ffmpeg, '-y',
                '-loop', '1',
                '-i', str(frame_path),
                '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo',
                '-t', str(duration),
                '-vf', f'fade=in:0:30,fade=out:st={duration-1}:d=1',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac',
                str(output_path)
            ]
            subprocess.run(cmd, capture_output=True, timeout=60)

        return str(output_path) if output_path.exists() else None

    def create_subtitle_overlay(
        self,
        text: str,
        output_path: str,
        font_size: int = 48,
        bg_opacity: int = 180
    ) -> str:
        """Create a subtitle/caption image overlay."""
        # Create transparent image
        img = Image.new('RGBA', self.resolution, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Load font
        try:
            font = ImageFont.truetype("C:\\Windows\\Fonts\\arialbd.ttf", font_size)
        except:
            font = ImageFont.load_default()

        # Wrap text to 2 lines max
        max_width = self.width - 200
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = f"{current_line} {word}".strip()
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        # Limit to 2 lines
        lines = lines[:2]
        text = "\n".join(lines)

        # Calculate position (bottom center)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (self.width - text_width) // 2
        y = self.height - text_height - 120

        # Draw background box
        padding = 20
        bg_rect = [
            x - padding,
            y - padding,
            x + text_width + padding,
            y + text_height + padding
        ]
        draw.rectangle(bg_rect, fill=(0, 0, 0, bg_opacity))

        # Draw text
        draw.text((x, y), text, fill="#ffffff", font=font)

        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)

        return output_path

    def create_title_card(
        self,
        title: str,
        subtitle: str = None,
        duration: float = 4.0,
        scheme: str = "default"
    ) -> str:
        """Create an animated title card."""
        colors = self.COLOR_SCHEMES.get(scheme, self.COLOR_SCHEMES["default"])
        output_path = self.temp_dir / f"title_{hash(title)}.mp4"

        # Create background
        bg_color = colors["secondary"].lstrip('#')
        bg_rgb = tuple(int(bg_color[i:i+2], 16) for i in (0, 2, 4))

        img = Image.new('RGB', self.resolution, bg_rgb)
        draw = ImageDraw.Draw(img)

        # Add gradient effect
        for i in range(100):
            alpha = int(255 * (1 - i / 100) * 0.5)
            draw.rectangle(
                [i, i, self.width - i, self.height - i],
                outline=tuple(min(255, c + alpha // 4) for c in bg_rgb)
            )

        # Draw title
        try:
            title_font = ImageFont.truetype("C:\\Windows\\Fonts\\arialbd.ttf", 80)
        except:
            title_font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), title, font=title_font)
        x = (self.width - (bbox[2] - bbox[0])) // 2
        y = self.height // 2 - 60

        # Shadow
        draw.text((x + 4, y + 4), title, fill="#000000", font=title_font)
        # Title
        draw.text((x, y), title, fill=colors["primary"], font=title_font)

        # Subtitle
        if subtitle:
            try:
                sub_font = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 40)
            except:
                sub_font = ImageFont.load_default()

            bbox = draw.textbbox((0, 0), subtitle, font=sub_font)
            x = (self.width - (bbox[2] - bbox[0])) // 2
            y = self.height // 2 + 60
            draw.text((x, y), subtitle, fill="#cccccc", font=sub_font)

        # Save frame
        frame_path = self.temp_dir / f"title_frame_{hash(title)}.png"
        img.save(str(frame_path))

        # Create video with fade effects (using GPU if available)
        if self.ffmpeg:
            encoder_args = self._get_encoder_args(bitrate="6M", quality=20)
            cmd = [
                self.ffmpeg, '-y',
                '-loop', '1',
                '-i', str(frame_path),
                '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo',
                '-t', str(duration),
                '-vf', f'fade=in:0:30,fade=out:st={duration-0.5}:d=0.5',
            ]
            cmd.extend(encoder_args)
            cmd.extend(['-c:a', 'aac', str(output_path)])
            subprocess.run(cmd, capture_output=True, timeout=60)

        return str(output_path) if output_path.exists() else None

    def process_stock_clip(
        self,
        input_path: str,
        output_path: str,
        target_duration: float,
        add_overlay: bool = True,
        darken: float = 0.4
    ) -> Optional[str]:
        """
        Process a stock video clip:
        - Resize to target resolution
        - Trim/loop to target duration
        - Add dark overlay for text visibility
        - Add fade transitions
        """
        if not self.ffmpeg or not os.path.exists(input_path):
            return None

        try:
            # Get input duration
            ffprobe = self.ffmpeg.replace('ffmpeg', 'ffprobe')
            probe_cmd = [
                ffprobe, '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                input_path
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
            input_duration = float(result.stdout.strip())

            # Build filter
            filters = []

            # Scale and crop to fill frame
            filters.append(f"scale={self.width}:{self.height}:force_original_aspect_ratio=increase")
            filters.append(f"crop={self.width}:{self.height}")

            # Darken for text visibility
            if add_overlay and darken > 0:
                brightness = 1 - darken
                filters.append(f"eq=brightness={-darken}:saturation=0.8")

            # Fade in/out
            fade_duration = min(0.5, target_duration / 4)
            filters.append(f"fade=in:0:{int(fade_duration * self.fps)}")
            filters.append(f"fade=out:st={target_duration - fade_duration}:d={fade_duration}")

            filter_str = ",".join(filters)

            # Get encoder args with GPU support
            encoder_args = self._get_encoder_args(bitrate="8M", quality=23)

            # Handle duration (loop if needed, trim if too long)
            if input_duration < target_duration:
                # Loop the video
                loop_count = int(target_duration / input_duration) + 1
                cmd = [self.ffmpeg, '-y']

                # Add GPU input args if available
                if self.gpu and self.gpu.is_available():
                    cmd.extend(self.gpu.get_input_args(use_hwaccel=True))

                cmd.extend([
                    '-stream_loop', str(loop_count),
                    '-i', input_path,
                    '-t', str(target_duration),
                    '-vf', filter_str,
                ])
                cmd.extend(encoder_args)
                cmd.extend(['-an', output_path])  # Remove audio from stock clips
            else:
                # Trim the video
                cmd = [self.ffmpeg, '-y']

                # Add GPU input args if available
                if self.gpu and self.gpu.is_available():
                    cmd.extend(self.gpu.get_input_args(use_hwaccel=True))

                cmd.extend([
                    '-i', input_path,
                    '-t', str(target_duration),
                    '-vf', filter_str,
                ])
                cmd.extend(encoder_args)
                cmd.extend(['-an', output_path])

            subprocess.run(cmd, capture_output=True, timeout=120)
            return output_path if os.path.exists(output_path) else None

        except Exception as e:
            logger.error(f"Clip processing failed: {e}")
            return None

    def create_video(
        self,
        audio_file: str,
        script: 'VideoScript',
        output_file: str,
        niche: str = "default",
        use_stock: bool = True
    ) -> Optional[str]:
        """
        Create a professional video with stock footage and effects.

        Args:
            audio_file: Path to narration audio
            script: VideoScript object with sections
            output_file: Output video path
            niche: Content niche for color scheme
            use_stock: Whether to fetch stock footage

        Returns:
            Path to created video or None
        """
        if not self.ffmpeg:
            logger.error("FFmpeg not available")
            return None

        if not os.path.exists(audio_file):
            logger.error(f"Audio file not found: {audio_file}")
            return None

        logger.info(f"Creating professional video: {output_file}")

        try:
            # Get audio duration using FFprobe
            if not self.ffprobe:
                logger.warning("ffprobe not found, will estimate duration")
                file_size = os.path.getsize(audio_file)
                audio_duration = file_size / 16000  # Rough estimate for 128kbps audio
            else:
                probe_cmd = [
                    self.ffprobe, '-v', 'error',
                    '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    audio_file
                ]
                result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)

                if result.returncode != 0 or not result.stdout.strip():
                    # Fallback: estimate duration from file size (rough estimate)
                    file_size = os.path.getsize(audio_file)
                    audio_duration = file_size / 16000  # Rough estimate for 128kbps audio
                    logger.warning(f"Could not get exact duration, estimating: {audio_duration:.1f}s")
                else:
                    audio_duration = float(result.stdout.strip())

            logger.info(f"Audio duration: {audio_duration:.1f}s")

            # Calculate section timings
            sections = script.sections if hasattr(script, 'sections') else []
            num_sections = len(sections) if sections else 5

            # Create video segments
            segment_files = []
            segment_duration = audio_duration / num_sections

            # Title card (first 4 seconds)
            title_card = self.create_title_card(
                title=script.title if hasattr(script, 'title') else "Video",
                subtitle=getattr(script, 'description', '')[:50] if hasattr(script, 'description') else None,
                duration=min(4.0, segment_duration),
                scheme=niche
            )
            if title_card:
                segment_files.append(title_card)

            # Get stock footage if available
            stock_clips = []
            if use_stock and self.stock and self.stock.api_key:
                # Build search queries from script
                search_terms = []
                if hasattr(script, 'title'):
                    search_terms.append(script.title)
                if sections:
                    for s in sections[:5]:
                        if hasattr(s, 'keywords') and s.keywords:
                            search_terms.extend(s.keywords[:2])
                        elif hasattr(s, 'title'):
                            search_terms.append(s.title)

                # Search for clips
                for term in search_terms[:5]:
                    clips = self.stock.search_videos(term, count=2, min_duration=5, max_duration=30)
                    stock_clips.extend(clips)

                logger.info(f"Found {len(stock_clips)} stock clips")

            # Download and process stock clips
            downloaded_clips = []
            for i, clip in enumerate(stock_clips[:num_sections]):
                clip_path = self.temp_dir / f"stock_{i}.mp4"
                if self.stock.download_video(clip, str(clip_path)):
                    downloaded_clips.append(str(clip_path))

            # Create segments with stock footage or gradient backgrounds
            colors = self.COLOR_SCHEMES.get(niche, self.COLOR_SCHEMES["default"])

            for i in range(num_sections):
                seg_start = i * segment_duration
                seg_end = (i + 1) * segment_duration
                seg_duration = seg_end - seg_start

                # Skip title card time for first segment
                if i == 0 and title_card:
                    seg_duration -= 4.0
                    if seg_duration <= 0:
                        continue

                segment_path = self.temp_dir / f"segment_{i}.mp4"

                # Use stock clip if available
                if i < len(downloaded_clips):
                    processed = self.process_stock_clip(
                        downloaded_clips[i],
                        str(segment_path),
                        seg_duration
                    )
                    if processed:
                        segment_files.append(processed)
                        continue

                # Fallback: Create gradient background
                self._create_gradient_segment(str(segment_path), seg_duration, colors)
                if segment_path.exists():
                    segment_files.append(str(segment_path))

            if not segment_files:
                logger.error("No video segments created")
                return None

            # Concatenate all segments
            concat_file = self.temp_dir / "concat_list.txt"
            with open(concat_file, 'w') as f:
                for seg in segment_files:
                    f.write(f"file '{seg}'\n")

            video_only = self.temp_dir / "video_only.mp4"
            encoder_args = self._get_encoder_args(bitrate="8M", quality=23)

            concat_cmd = [
                self.ffmpeg, '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
            ]
            concat_cmd.extend(encoder_args)
            concat_cmd.append(str(video_only))

            subprocess.run(concat_cmd, capture_output=True, timeout=300)

            # Combine with audio
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)

            final_cmd = [
                self.ffmpeg, '-y',
                '-i', str(video_only),
                '-i', audio_file,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-shortest',
                output_file
            ]
            subprocess.run(final_cmd, capture_output=True, timeout=300)

            if os.path.exists(output_file):
                logger.success(f"Professional video created: {output_file}")
                self._cleanup_temp_files()
                return output_file

        except Exception as e:
            logger.error(f"Video creation failed: {e}")

        return None

    def _create_gradient_segment(
        self,
        output_path: str,
        duration: float,
        colors: Dict[str, str]
    ):
        """Create a gradient background segment."""
        # Create gradient image
        bg_color = colors["secondary"].lstrip('#')
        bg_rgb = tuple(int(bg_color[i:i+2], 16) for i in (0, 2, 4))

        accent_color = colors["primary"].lstrip('#')
        accent_rgb = tuple(int(accent_color[i:i+2], 16) for i in (0, 2, 4))

        img = Image.new('RGB', self.resolution, bg_rgb)
        draw = ImageDraw.Draw(img)

        # Radial gradient effect
        center_x, center_y = self.width // 2, self.height // 2
        max_radius = max(self.width, self.height)

        for r in range(0, max_radius, 10):
            ratio = r / max_radius
            color = tuple(
                int(bg_rgb[i] + (accent_rgb[i] - bg_rgb[i]) * ratio * 0.3)
                for i in range(3)
            )
            draw.ellipse(
                [center_x - r, center_y - r, center_x + r, center_y + r],
                outline=color
            )

        frame_path = Path(output_path).with_suffix('.bg.png')
        img.save(str(frame_path))

        # Create video (with GPU support)
        if self.ffmpeg:
            encoder_args = self._get_encoder_args(bitrate="6M", quality=23)
            cmd = [
                self.ffmpeg, '-y',
                '-loop', '1',
                '-i', str(frame_path),
                '-t', str(duration),
                '-vf', 'fade=in:0:15,fade=out:st=' + str(duration - 0.5) + ':d=0.5',
            ]
            cmd.extend(encoder_args)
            cmd.extend(['-an', output_path])
            subprocess.run(cmd, capture_output=True, timeout=60)

        # Cleanup frame
        if frame_path.exists():
            frame_path.unlink()

    def _cleanup_temp_files(self):
        """Clean up temporary files."""
        try:
            for f in self.temp_dir.glob("*"):
                if f.is_file():
                    f.unlink()
        except:
            pass

    def create_thumbnail_pro(
        self,
        title: str,
        output_file: str,
        background_image: str = None,
        niche: str = "default"
    ) -> str:
        """Create a professional YouTube thumbnail."""
        size = (1280, 720)
        colors = self.COLOR_SCHEMES.get(niche, self.COLOR_SCHEMES["default"])

        # Load or create background
        if background_image and os.path.exists(background_image):
            img = Image.open(background_image).convert('RGB')
            img = img.resize(size, Image.LANCZOS)
            # Darken
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(0.5)
        else:
            # Create gradient background
            bg_color = colors["secondary"].lstrip('#')
            bg_rgb = tuple(int(bg_color[i:i+2], 16) for i in (0, 2, 4))
            img = Image.new('RGB', size, bg_rgb)

        draw = ImageDraw.Draw(img)

        # Add accent color bar at bottom
        accent = colors["primary"].lstrip('#')
        accent_rgb = tuple(int(accent[i:i+2], 16) for i in (0, 2, 4))
        draw.rectangle([0, size[1] - 10, size[0], size[1]], fill=accent_rgb)

        # Load font
        try:
            font = ImageFont.truetype("C:\\Windows\\Fonts\\arialbd.ttf", 72)
        except:
            font = ImageFont.load_default()

        # Wrap title
        max_chars = 20
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
        title = "\n".join(lines[:3])

        # Draw title
        bbox = draw.textbbox((0, 0), title, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (size[0] - text_width) // 2
        y = (size[1] - text_height) // 2 - 20

        # Shadow
        draw.text((x + 4, y + 4), title, fill="#000000", font=font)
        # Text
        draw.text((x, y), title, fill="#ffffff", font=font)

        # Save
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        img.save(output_file, quality=95)
        logger.success(f"Thumbnail created: {output_file}")

        return output_file


# Example usage
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  PROFESSIONAL VIDEO GENERATOR TEST")
    print("="*60 + "\n")

    generator = ProVideoGenerator()

    # Create test thumbnail
    generator.create_thumbnail_pro(
        title="10 Passive Income Ideas",
        output_file="output/test_pro_thumb.png",
        niche="finance"
    )

    print("\nThumbnail created!")
