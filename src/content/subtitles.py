"""
Subtitle/Caption System for YouTube Automation

Generates and burns subtitles into videos for better accessibility and retention.
Subtitles increase retention by 15-25% according to research.

Features:
- Generate subtitles from script text synced with audio duration
- Generate subtitles from audio using speech-to-text (Whisper)
- Create SRT/VTT subtitle files
- Burn subtitles into video with configurable styles
- Niche-specific styling presets

Usage:
    from src.content.subtitles import SubtitleGenerator

    generator = SubtitleGenerator()

    # Generate subtitles from script
    track = generator.generate_subtitles_from_script(script_text, audio_duration)

    # Generate subtitles from audio (requires Whisper)
    track = generator.generate_subtitles_from_audio("audio.mp3")

    # Burn subtitles into video
    output = generator.burn_subtitles("video.mp4", track, "output.mp4", style="shorts")

    # Create SRT file
    srt_path = generator.create_srt_file(track, "subtitles.srt")
"""

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


# Check for Whisper availability
WHISPER_AVAILABLE = False
FASTER_WHISPER_AVAILABLE = False
WHISPER_TYPE = None

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
    WHISPER_TYPE = "faster-whisper"
    logger.debug("faster-whisper is available for speech-to-text")
except ImportError:
    pass

if not FASTER_WHISPER_AVAILABLE:
    try:
        import whisper
        WHISPER_AVAILABLE = True
        WHISPER_TYPE = "openai-whisper"
        logger.debug("openai-whisper is available for speech-to-text")
    except ImportError:
        pass


class SubtitlePosition(Enum):
    """Subtitle position on screen."""
    TOP = "top"
    CENTER = "center"
    BOTTOM = "bottom"


@dataclass
class SubtitleCue:
    """A single subtitle cue with timing and text."""
    index: int
    start_time: float  # seconds
    end_time: float    # seconds
    text: str
    words: List[Dict[str, Union[str, float]]] = field(default_factory=list)  # word-level timing

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_srt_timestamp(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def to_vtt_timestamp(self, seconds: float) -> str:
        """Convert seconds to VTT timestamp format (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def to_srt(self) -> str:
        """Convert to SRT format block."""
        start = self.to_srt_timestamp(self.start_time)
        end = self.to_srt_timestamp(self.end_time)
        return f"{self.index}\n{start} --> {end}\n{self.text}\n"

    def to_vtt(self) -> str:
        """Convert to VTT format block."""
        start = self.to_vtt_timestamp(self.start_time)
        end = self.to_vtt_timestamp(self.end_time)
        return f"{start} --> {end}\n{self.text}\n"


@dataclass
class SubtitleTrack:
    """A collection of subtitle cues."""
    cues: List[SubtitleCue] = field(default_factory=list)
    language: str = "en"

    def add_cue(self, start: float, end: float, text: str, words: List = None):
        """Add a new cue to the track."""
        cue = SubtitleCue(
            index=len(self.cues) + 1,
            start_time=start,
            end_time=end,
            text=text,
            words=words or []
        )
        self.cues.append(cue)
        return cue

    def to_srt(self) -> str:
        """Convert entire track to SRT format."""
        return "\n".join(cue.to_srt() for cue in self.cues)

    def to_vtt(self) -> str:
        """Convert entire track to VTT format."""
        header = "WEBVTT\n\n"
        return header + "\n".join(cue.to_vtt() for cue in self.cues)

    @property
    def total_duration(self) -> float:
        """Get total duration of subtitles."""
        if not self.cues:
            return 0
        return max(cue.end_time for cue in self.cues)


# Subtitle styling presets
SUBTITLE_STYLES = {
    "shorts": {
        "font_size": 48,
        "font_name": "Arial",
        "font_color": "white",
        "outline_color": "black",
        "outline_width": 3,
        "position": SubtitlePosition.CENTER,
        "margin_v": 100,  # Vertical margin from position
        "max_chars": 30,  # Max characters per line
        "max_lines": 2,
        "highlight_keywords": True,
        "background_box": True,
        "background_color": "black",
        "background_opacity": 0.6,
        "bold": True,
        "shadow": True,
        "shadow_offset": 2,
    },
    "regular": {
        "font_size": 32,
        "font_name": "Arial",
        "font_color": "white",
        "outline_color": "black",
        "outline_width": 2,
        "position": SubtitlePosition.BOTTOM,
        "margin_v": 50,
        "max_chars": 50,
        "max_lines": 2,
        "highlight_keywords": False,
        "background_box": True,
        "background_color": "black",
        "background_opacity": 0.5,
        "bold": False,
        "shadow": True,
        "shadow_offset": 1,
    },
    "minimal": {
        "font_size": 28,
        "font_name": "Arial",
        "font_color": "white",
        "outline_color": "black",
        "outline_width": 1,
        "position": SubtitlePosition.BOTTOM,
        "margin_v": 30,
        "max_chars": 60,
        "max_lines": 2,
        "highlight_keywords": False,
        "background_box": False,
        "background_color": None,
        "background_opacity": 0,
        "bold": False,
        "shadow": True,
        "shadow_offset": 1,
    },
    "cinematic": {
        "font_size": 36,
        "font_name": "Arial",
        "font_color": "white",
        "outline_color": "black",
        "outline_width": 2,
        "position": SubtitlePosition.BOTTOM,
        "margin_v": 80,
        "max_chars": 45,
        "max_lines": 2,
        "highlight_keywords": False,
        "background_box": True,
        "background_color": "black",
        "background_opacity": 0.4,
        "bold": True,
        "shadow": True,
        "shadow_offset": 2,
    },
}

# Niche-specific style overrides
NICHE_SUBTITLE_STYLES = {
    "finance": {
        "font_color": "white",
        "outline_color": "#003333",  # Dark teal
        "background_color": "#001a1a",
        "highlight_color": "#00d4aa",  # Teal accent
    },
    "psychology": {
        "font_color": "white",
        "outline_color": "#1a0033",  # Dark purple
        "background_color": "#0d001a",
        "highlight_color": "#9b59b6",  # Purple accent
    },
    "storytelling": {
        "font_color": "white",
        "outline_color": "#330000",  # Dark red
        "background_color": "#1a0000",
        "highlight_color": "#e74c3c",  # Red accent
    },
    "default": {
        "font_color": "white",
        "outline_color": "black",
        "background_color": "black",
        "highlight_color": "#3498db",  # Blue accent
    },
}


class SubtitleGenerator:
    """
    Generate and burn subtitles into videos.

    Supports:
    - Script-based subtitle generation (syncs with audio duration)
    - Audio-based transcription (using Whisper)
    - Multiple output formats (SRT, VTT)
    - Customizable styles for Shorts and regular videos
    - Burned-in subtitles via FFmpeg
    """

    # Average speaking rates for timing estimation
    WORDS_PER_MINUTE = 150  # Average speaking rate
    CHARS_PER_SECOND = 15   # Average reading rate

    # Timing constraints
    MIN_CUE_DURATION = 0.8   # Minimum subtitle display time
    MAX_CUE_DURATION = 5.0   # Maximum subtitle display time
    CUE_GAP = 0.05           # Gap between cues

    def __init__(self, whisper_model: str = "base"):
        """
        Initialize the subtitle generator.

        Args:
            whisper_model: Whisper model size for transcription
                          ("tiny", "base", "small", "medium", "large")
        """
        self.whisper_model_name = whisper_model
        self.whisper = None
        self._init_whisper()

        # Find FFmpeg
        self.ffmpeg = self._find_ffmpeg()
        self.ffprobe = self._find_ffprobe()

        if self.ffmpeg:
            logger.info(f"SubtitleGenerator initialized (FFmpeg: {self.ffmpeg})")
        else:
            logger.warning("FFmpeg not found - subtitle burning will not work")

        if WHISPER_TYPE:
            logger.info(f"Speech-to-text available via {WHISPER_TYPE}")
        else:
            logger.info("No Whisper available - using script-based subtitles only")

    def _init_whisper(self):
        """Initialize Whisper model if available."""
        if FASTER_WHISPER_AVAILABLE:
            try:
                # Use CPU by default, can be changed to "cuda" for GPU
                self.whisper = WhisperModel(
                    self.whisper_model_name,
                    device="cpu",
                    compute_type="int8"
                )
                logger.debug(f"Loaded faster-whisper model: {self.whisper_model_name}")
            except Exception as e:
                logger.warning(f"Failed to load faster-whisper: {e}")
                self.whisper = None
        elif WHISPER_AVAILABLE:
            try:
                self.whisper = whisper.load_model(self.whisper_model_name)
                logger.debug(f"Loaded openai-whisper model: {self.whisper_model_name}")
            except Exception as e:
                logger.warning(f"Failed to load openai-whisper: {e}")
                self.whisper = None

    def _find_ffmpeg(self) -> Optional[str]:
        """Find FFmpeg executable."""
        if shutil.which("ffmpeg"):
            return "ffmpeg"

        common_paths = [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            os.path.expanduser("~\\ffmpeg\\bin\\ffmpeg.exe"),
        ]

        winget_base = os.path.expanduser("~\\AppData\\Local\\Microsoft\\WinGet\\Packages")
        if os.path.exists(winget_base):
            for folder in os.listdir(winget_base):
                if "FFmpeg" in folder:
                    package_path = os.path.join(winget_base, folder)
                    for root, dirs, files in os.walk(package_path):
                        if "ffmpeg.exe" in files:
                            return os.path.join(root, "ffmpeg.exe")

        for path in common_paths:
            if os.path.exists(path):
                return path

        return None

    def _find_ffprobe(self) -> Optional[str]:
        """Find FFprobe executable."""
        if shutil.which("ffprobe"):
            return "ffprobe"

        if self.ffmpeg:
            ffprobe = self.ffmpeg.replace("ffmpeg.exe", "ffprobe.exe")
            if ffprobe != self.ffmpeg and os.path.exists(ffprobe):
                return ffprobe
            ffprobe = self.ffmpeg.replace("ffmpeg", "ffprobe")
            if os.path.exists(ffprobe):
                return ffprobe

        return None

    def get_audio_duration(self, audio_file: str) -> Optional[float]:
        """Get duration of audio file in seconds."""
        if not self.ffprobe or not os.path.exists(audio_file):
            return None

        try:
            cmd = [
                self.ffprobe,
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                audio_file
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError):
            pass

        return None

    def _split_into_phrases(self, text: str, max_chars: int = 50) -> List[str]:
        """
        Split text into phrases suitable for subtitles.

        Splits on natural boundaries (periods, commas, etc.) while
        respecting max character limits.
        """
        # Clean text
        text = re.sub(r'\s+', ' ', text.strip())

        # Split on sentence boundaries first
        sentences = re.split(r'(?<=[.!?])\s+', text)

        phrases = []
        for sentence in sentences:
            if len(sentence) <= max_chars:
                phrases.append(sentence)
            else:
                # Split on commas, colons, semicolons
                parts = re.split(r'(?<=[,;:])\s+', sentence)
                current = ""

                for part in parts:
                    if len(current) + len(part) + 1 <= max_chars:
                        current = f"{current} {part}".strip() if current else part
                    else:
                        if current:
                            phrases.append(current)

                        # If part is still too long, split on words
                        if len(part) > max_chars:
                            words = part.split()
                            current = ""
                            for word in words:
                                if len(current) + len(word) + 1 <= max_chars:
                                    current = f"{current} {word}".strip() if current else word
                                else:
                                    if current:
                                        phrases.append(current)
                                    current = word
                        else:
                            current = part

                if current:
                    phrases.append(current)

        return [p.strip() for p in phrases if p.strip()]

    def generate_subtitles_from_script(
        self,
        script: str,
        audio_duration: float,
        max_chars: int = 50,
        words_per_minute: float = None
    ) -> SubtitleTrack:
        """
        Generate subtitles from script text, synced to audio duration.

        Uses estimated timing based on word count and speaking rate.

        Args:
            script: The script/narration text
            audio_duration: Total duration of the audio in seconds
            max_chars: Maximum characters per subtitle line
            words_per_minute: Speaking rate (default: 150 WPM)

        Returns:
            SubtitleTrack with timed cues
        """
        logger.info(f"Generating subtitles from script ({len(script)} chars, {audio_duration:.1f}s)")

        wpm = words_per_minute or self.WORDS_PER_MINUTE

        # Split into phrases
        phrases = self._split_into_phrases(script, max_chars)

        if not phrases:
            logger.warning("No phrases extracted from script")
            return SubtitleTrack()

        # Calculate timing for each phrase based on word count
        total_words = sum(len(p.split()) for p in phrases)

        if total_words == 0:
            logger.warning("No words in script")
            return SubtitleTrack()

        # Time per word (seconds)
        seconds_per_word = audio_duration / total_words

        track = SubtitleTrack()
        current_time = 0.0

        for phrase in phrases:
            word_count = len(phrase.split())
            duration = word_count * seconds_per_word

            # Apply duration constraints
            duration = max(self.MIN_CUE_DURATION, min(self.MAX_CUE_DURATION, duration))

            # Ensure we don't exceed audio duration
            end_time = min(current_time + duration, audio_duration)

            if current_time < audio_duration:
                track.add_cue(current_time, end_time, phrase)

            current_time = end_time + self.CUE_GAP

        logger.info(f"Generated {len(track.cues)} subtitle cues")
        return track

    def generate_subtitles_from_audio(
        self,
        audio_path: str,
        language: str = "en"
    ) -> SubtitleTrack:
        """
        Generate subtitles from audio using speech-to-text.

        Uses Whisper (faster-whisper or openai-whisper) for transcription
        with word-level timestamps.

        Args:
            audio_path: Path to audio file
            language: Language code (default: "en")

        Returns:
            SubtitleTrack with accurate timing from transcription
        """
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return SubtitleTrack()

        if not self.whisper:
            logger.warning("Whisper not available - falling back to script-based timing")
            logger.info("Install faster-whisper: pip install faster-whisper")
            logger.info("Or openai-whisper: pip install openai-whisper")
            return SubtitleTrack()

        logger.info(f"Transcribing audio with {WHISPER_TYPE}: {audio_path}")

        track = SubtitleTrack(language=language)

        try:
            if FASTER_WHISPER_AVAILABLE:
                return self._transcribe_faster_whisper(audio_path, language)
            elif WHISPER_AVAILABLE:
                return self._transcribe_openai_whisper(audio_path, language)
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return SubtitleTrack()

        return track

    def _transcribe_faster_whisper(
        self,
        audio_path: str,
        language: str
    ) -> SubtitleTrack:
        """Transcribe using faster-whisper."""
        track = SubtitleTrack(language=language)

        segments, info = self.whisper.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
            vad_filter=True  # Filter out silence
        )

        cue_index = 1
        for segment in segments:
            # Get word-level timing if available
            words_data = []
            if hasattr(segment, 'words') and segment.words:
                words_data = [
                    {"word": w.word, "start": w.start, "end": w.end}
                    for w in segment.words
                ]

            track.add_cue(
                start=segment.start,
                end=segment.end,
                text=segment.text.strip(),
                words=words_data
            )
            cue_index += 1

        logger.info(f"Transcribed {len(track.cues)} segments")
        return track

    def _transcribe_openai_whisper(
        self,
        audio_path: str,
        language: str
    ) -> SubtitleTrack:
        """Transcribe using openai-whisper."""
        track = SubtitleTrack(language=language)

        result = self.whisper.transcribe(
            audio_path,
            language=language,
            word_timestamps=True
        )

        for segment in result.get("segments", []):
            words_data = []
            if "words" in segment:
                words_data = [
                    {"word": w["word"], "start": w["start"], "end": w["end"]}
                    for w in segment["words"]
                ]

            track.add_cue(
                start=segment["start"],
                end=segment["end"],
                text=segment["text"].strip(),
                words=words_data
            )

        logger.info(f"Transcribed {len(track.cues)} segments")
        return track

    def sync_subtitles_with_audio(
        self,
        script_text: str,
        audio_path: str,
        max_chars: int = 50
    ) -> SubtitleTrack:
        """
        Sync script text with audio, using transcription if available.

        If Whisper is available, uses audio transcription for accurate timing.
        Otherwise, falls back to script-based estimation.

        Args:
            script_text: The script/narration text
            audio_path: Path to audio file
            max_chars: Maximum characters per subtitle line

        Returns:
            SubtitleTrack with synced timing
        """
        logger.info("Syncing subtitles with audio")

        # Get audio duration
        duration = self.get_audio_duration(audio_path)
        if not duration:
            logger.warning("Could not get audio duration")
            duration = len(script_text.split()) / (self.WORDS_PER_MINUTE / 60)

        # Try transcription first for best timing
        if self.whisper:
            track = self.generate_subtitles_from_audio(audio_path)
            if track.cues:
                return track

        # Fall back to script-based timing
        return self.generate_subtitles_from_script(script_text, duration, max_chars)

    def create_srt_file(
        self,
        subtitle_track: SubtitleTrack,
        output_path: str
    ) -> str:
        """
        Create an SRT subtitle file from a track.

        Args:
            subtitle_track: SubtitleTrack to convert
            output_path: Output file path (.srt)

        Returns:
            Path to created SRT file
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        srt_content = subtitle_track.to_srt()

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)

        logger.info(f"Created SRT file: {output_path}")
        return output_path

    def create_vtt_file(
        self,
        subtitle_track: SubtitleTrack,
        output_path: str
    ) -> str:
        """
        Create a VTT subtitle file from a track.

        Args:
            subtitle_track: SubtitleTrack to convert
            output_path: Output file path (.vtt)

        Returns:
            Path to created VTT file
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        vtt_content = subtitle_track.to_vtt()

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(vtt_content)

        logger.info(f"Created VTT file: {output_path}")
        return output_path

    def get_style(
        self,
        style_name: str = "regular",
        niche: str = None
    ) -> Dict:
        """
        Get subtitle style configuration.

        Args:
            style_name: Base style ("shorts", "regular", "minimal", "cinematic")
            niche: Optional niche for color customization

        Returns:
            Dict with style settings
        """
        # Get base style
        style = SUBTITLE_STYLES.get(style_name, SUBTITLE_STYLES["regular"]).copy()

        # Apply niche overrides
        if niche:
            niche_style = NICHE_SUBTITLE_STYLES.get(niche, NICHE_SUBTITLE_STYLES["default"])
            for key, value in niche_style.items():
                if key in style or key.endswith("_color"):
                    style[key] = value

        return style

    def _build_subtitle_filter(
        self,
        srt_path: str,
        style: Dict,
        video_width: int = 1920,
        video_height: int = 1080
    ) -> str:
        """
        Build FFmpeg subtitle filter string.

        Creates a complex filter for styled subtitles with optional background box.
        """
        # Escape path for FFmpeg filter
        srt_escaped = srt_path.replace("\\", "/").replace(":", "\\:")

        # Determine vertical alignment
        position = style.get("position", SubtitlePosition.BOTTOM)
        margin_v = style.get("margin_v", 50)

        if position == SubtitlePosition.TOP:
            alignment = 8  # ASS alignment: top center
            y_position = margin_v
        elif position == SubtitlePosition.CENTER:
            alignment = 5  # ASS alignment: middle center
            y_position = video_height // 2
        else:  # BOTTOM
            alignment = 2  # ASS alignment: bottom center
            y_position = video_height - margin_v

        # Build force_style string
        force_style_parts = [
            f"FontName={style.get('font_name', 'Arial')}",
            f"FontSize={style.get('font_size', 32)}",
            f"PrimaryColour=&H00FFFFFF",  # White (ABGR format)
            f"OutlineColour=&H00000000",  # Black outline
            f"Outline={style.get('outline_width', 2)}",
            f"Alignment={alignment}",
            f"MarginV={margin_v}",
        ]

        if style.get("bold"):
            force_style_parts.append("Bold=1")

        if style.get("shadow"):
            force_style_parts.append(f"Shadow={style.get('shadow_offset', 1)}")

        # Background box (BorderStyle=4 for opaque box)
        if style.get("background_box"):
            opacity = style.get("background_opacity", 0.5)
            # Convert opacity to hex alpha (inverted: 0=opaque, 255=transparent)
            alpha = int((1 - opacity) * 255)
            force_style_parts.append("BorderStyle=4")
            force_style_parts.append(f"BackColour=&H{alpha:02X}000000")  # Semi-transparent black

        force_style = ",".join(force_style_parts)

        # Build the subtitles filter
        subtitle_filter = f"subtitles='{srt_escaped}':force_style='{force_style}'"

        return subtitle_filter

    def burn_subtitles(
        self,
        video_path: str,
        subtitle_track: SubtitleTrack,
        output_path: str,
        style: Union[str, Dict] = "regular",
        niche: str = None
    ) -> Optional[str]:
        """
        Burn subtitles into a video file using FFmpeg.

        Creates hardcoded/burned-in subtitles that are part of the video stream.

        Args:
            video_path: Input video path
            subtitle_track: SubtitleTrack to burn in
            output_path: Output video path
            style: Style name ("shorts", "regular", etc.) or custom style dict
            niche: Optional niche for color customization

        Returns:
            Path to output video or None on failure
        """
        if not self.ffmpeg:
            logger.error("FFmpeg not available for subtitle burning")
            return None

        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return None

        if not subtitle_track.cues:
            logger.warning("No subtitles to burn")
            # Just copy the video
            shutil.copy(video_path, output_path)
            return output_path

        # Get style configuration
        if isinstance(style, str):
            style_config = self.get_style(style, niche)
        else:
            style_config = style

        logger.info(f"Burning subtitles into video: {video_path}")

        # Create temporary SRT file
        temp_dir = tempfile.mkdtemp()
        temp_srt = os.path.join(temp_dir, "subtitles.srt")

        try:
            # Write SRT file
            self.create_srt_file(subtitle_track, temp_srt)

            # Get video dimensions
            video_info = self._get_video_info(video_path)
            width = video_info.get("width", 1920)
            height = video_info.get("height", 1080)

            # Build subtitle filter
            sub_filter = self._build_subtitle_filter(temp_srt, style_config, width, height)

            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # FFmpeg command to burn subtitles
            cmd = [
                self.ffmpeg, '-y',
                '-i', video_path,
                '-vf', sub_filter,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-c:a', 'copy',
                output_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr[:500]}")
                return None

            if os.path.exists(output_path):
                logger.success(f"Subtitles burned into: {output_path}")
                return output_path

            return None

        except subprocess.TimeoutExpired:
            logger.error("Subtitle burning timed out")
            return None
        except Exception as e:
            logger.error(f"Subtitle burning failed: {e}")
            return None
        finally:
            # Cleanup temp files
            try:
                if os.path.exists(temp_srt):
                    os.remove(temp_srt)
                os.rmdir(temp_dir)
            except OSError:
                pass

    def _get_video_info(self, video_path: str) -> Dict:
        """Get video dimensions and other info."""
        if not self.ffprobe:
            return {"width": 1920, "height": 1080}

        try:
            cmd = [
                self.ffprobe,
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'csv=p=0',
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(',')
                if len(parts) >= 2:
                    return {
                        "width": int(parts[0]),
                        "height": int(parts[1])
                    }
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError):
            pass

        return {"width": 1920, "height": 1080}

    def add_subtitles_to_video(
        self,
        video_path: str,
        script: str,
        output_path: str,
        audio_path: str = None,
        style: str = "regular",
        niche: str = None,
        use_transcription: bool = True
    ) -> Optional[str]:
        """
        Convenience method to add subtitles to a video from script.

        Handles the full workflow: generate subtitles, burn into video.

        Args:
            video_path: Input video path
            script: Script/narration text
            output_path: Output video path
            audio_path: Optional audio file for transcription
            style: Subtitle style name
            niche: Content niche for styling
            use_transcription: Try to use audio transcription if available

        Returns:
            Path to output video with subtitles or None on failure
        """
        logger.info(f"Adding subtitles to video: {video_path}")

        # Generate subtitles
        if audio_path and use_transcription and self.whisper:
            track = self.sync_subtitles_with_audio(script, audio_path)
        else:
            # Get video duration if no audio
            duration = self.get_audio_duration(video_path)
            if not duration:
                duration = len(script.split()) / (self.WORDS_PER_MINUTE / 60)

            style_config = self.get_style(style, niche)
            max_chars = style_config.get("max_chars", 50)
            track = self.generate_subtitles_from_script(script, duration, max_chars)

        # Burn subtitles
        return self.burn_subtitles(video_path, track, output_path, style, niche)


# Example usage and testing
if __name__ == "__main__":
    print("\n" + "="*60)
    print("SUBTITLE GENERATOR TEST")
    print("="*60 + "\n")

    generator = SubtitleGenerator()

    print(f"FFmpeg: {generator.ffmpeg}")
    print(f"FFprobe: {generator.ffprobe}")
    print(f"Whisper available: {WHISPER_TYPE or 'No'}")

    # Test script-based subtitle generation
    test_script = """
    Welcome to this video about financial freedom. Today we'll explore
    five powerful strategies that can transform your relationship with money.
    First, let's talk about the importance of budgeting. A good budget is
    the foundation of financial success. Second, consider investing early.
    Time in the market beats timing the market. Third, eliminate high-interest
    debt as quickly as possible. Fourth, build an emergency fund with at least
    three to six months of expenses. And finally, diversify your income streams.
    """

    print("\n--- Testing Script-Based Subtitles ---")
    track = generator.generate_subtitles_from_script(test_script, audio_duration=60)
    print(f"Generated {len(track.cues)} cues")

    if track.cues:
        print("\nFirst 3 cues:")
        for cue in track.cues[:3]:
            print(f"  [{cue.start_time:.1f}s - {cue.end_time:.1f}s] {cue.text}")

    # Test SRT generation
    print("\n--- Testing SRT Output ---")
    print(track.cues[0].to_srt() if track.cues else "No cues")

    # Test style configurations
    print("\n--- Style Configurations ---")
    for style_name in SUBTITLE_STYLES:
        style = generator.get_style(style_name, niche="finance")
        print(f"{style_name}: {style.get('font_size')}px, {style.get('position').value}")

    print("\nSubtitle generator ready for use!")
