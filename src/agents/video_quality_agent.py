"""
Video Quality Agent - Video Quality Validation

Validates video files for YouTube optimization using FFprobe for analysis.
Checks resolution, bitrate, frame rate, codec, and encoding quality.

Usage:
    from src.agents.video_quality_agent import VideoQualityAgent

    agent = VideoQualityAgent()

    # Check video quality
    result = agent.run(video_file="path/to/video.mp4")

    if result.success:
        quality = result.data
        print(f"Passed: {quality['passed']}")
        print(f"Resolution: {quality['resolution']}")
        print(f"Bitrate: {quality['bitrate']}")

Example:
    >>> agent = VideoQualityAgent()
    >>> result = agent.run(video_file="output.mp4", is_short=False)
    >>> print(result.data['passed'])
    True
    >>> print(result.data['resolution'])
    1920x1080
"""

import os
import subprocess
import shutil
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

from .base_agent import BaseAgent, AgentResult


@dataclass
class VideoQualityResult:
    """
    Result of video quality check.

    Attributes:
        passed: Whether the video passes all quality checks
        resolution: Video resolution as "WIDTHxHEIGHT"
        width: Video width in pixels
        height: Video height in pixels
        bitrate: Video bitrate in Mbps
        frame_rate: Video frame rate in fps
        codec: Video codec (e.g., "h264", "h265")
        pixel_format: Pixel format (e.g., "yuv420p")
        duration: Video duration in seconds
        file_size_mb: File size in megabytes
        has_audio: Whether video has audio track
        issues: List of quality issues found
        warnings: Non-critical warnings
        recommendations: Suggested improvements
    """
    passed: bool
    resolution: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    bitrate: Optional[float] = None
    frame_rate: Optional[float] = None
    codec: Optional[str] = None
    pixel_format: Optional[str] = None
    duration: Optional[float] = None
    file_size_mb: Optional[float] = None
    has_audio: bool = False
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "resolution": self.resolution,
            "width": self.width,
            "height": self.height,
            "bitrate": self.bitrate,
            "frame_rate": self.frame_rate,
            "codec": self.codec,
            "pixel_format": self.pixel_format,
            "duration": self.duration,
            "file_size_mb": self.file_size_mb,
            "has_audio": self.has_audio,
            "issues": self.issues,
            "warnings": self.warnings,
            "recommendations": self.recommendations
        }


class VideoQualityAgent(BaseAgent):
    """
    Agent for validating video quality for YouTube.

    Uses FFprobe for video analysis and validates against
    YouTube's recommended video specifications.

    Features:
    - Resolution verification (1920x1080 or 1080x1920 for Shorts)
    - Bitrate validation (>= 8 Mbps recommended)
    - Frame rate checking (30fps standard)
    - Codec verification (h264/h265)
    - Encoding issue detection
    """

    # YouTube recommended video specs
    RESOLUTION_REGULAR = (1920, 1080)  # 16:9 landscape
    RESOLUTION_SHORTS = (1080, 1920)   # 9:16 portrait
    MIN_BITRATE_MBPS = 8.0             # Minimum recommended bitrate
    TARGET_BITRATE_MBPS = 12.0         # Optimal bitrate for 1080p
    TARGET_FRAME_RATE = 30.0           # Standard frame rate
    MIN_FRAME_RATE = 24.0              # Minimum acceptable frame rate
    MAX_FRAME_RATE = 60.0              # Maximum common frame rate
    SUPPORTED_CODECS = ["h264", "hevc", "h265", "vp9", "av1"]
    RECOMMENDED_CODECS = ["h264", "hevc", "h265"]
    RECOMMENDED_PIXEL_FORMAT = "yuv420p"

    # Duration limits
    MAX_DURATION_REGULAR = 12 * 60 * 60  # 12 hours
    MAX_DURATION_SHORTS = 60             # 60 seconds
    MIN_DURATION_SHORTS = 15             # 15 seconds
    OPTIMAL_DURATION_SHORTS = (20, 45)   # Optimal range for engagement

    # File size limits
    MAX_FILE_SIZE_GB = 256  # YouTube's limit

    def __init__(self, provider: str = "ffprobe", api_key: str = None):
        """
        Initialize the video quality agent.

        Args:
            provider: Analysis provider (default: ffprobe)
            api_key: Not used for this agent
        """
        super().__init__(provider=provider, api_key=api_key)
        self.ffprobe = self._find_ffprobe()
        self.ffmpeg = self._find_ffmpeg()

        if self.ffprobe:
            logger.info(f"VideoQualityAgent initialized (FFprobe: {self.ffprobe})")
        else:
            logger.warning(
                "FFprobe not found! Video quality analysis will be limited. "
                "Install FFmpeg: https://ffmpeg.org/download.html"
            )

    def _find_ffprobe(self) -> Optional[str]:
        """Find FFprobe executable."""
        if shutil.which("ffprobe"):
            return "ffprobe"

        # Common Windows locations
        common_paths = [
            r"C:\ffmpeg\bin\ffprobe.exe",
            r"C:\Program Files\ffmpeg\bin\ffprobe.exe",
            os.path.expanduser("~\\ffmpeg\\bin\\ffprobe.exe"),
        ]

        # Check WinGet installation paths
        winget_base = os.path.expanduser("~\\AppData\\Local\\Microsoft\\WinGet\\Packages")
        if os.path.exists(winget_base):
            for folder in os.listdir(winget_base):
                if "FFmpeg" in folder:
                    package_path = os.path.join(winget_base, folder)
                    for root, dirs, files in os.walk(package_path):
                        if "ffprobe.exe" in files:
                            return os.path.join(root, "ffprobe.exe")

        for path in common_paths:
            if os.path.exists(path):
                return path

        return None

    def _find_ffmpeg(self) -> Optional[str]:
        """Find FFmpeg executable."""
        if shutil.which("ffmpeg"):
            return "ffmpeg"

        if self.ffprobe:
            ffmpeg = self.ffprobe.replace("ffprobe.exe", "ffmpeg.exe")
            if ffmpeg != self.ffprobe and os.path.exists(ffmpeg):
                return ffmpeg
            ffmpeg = self.ffprobe.replace("ffprobe", "ffmpeg")
            if os.path.exists(ffmpeg):
                return ffmpeg

        return None

    def run(
        self,
        video_file: str = "",
        is_short: bool = False,
        check_encoding: bool = True,
        **kwargs
    ) -> AgentResult:
        """
        Analyze video file quality for YouTube.

        Args:
            video_file: Path to video file to analyze
            is_short: Whether this is a YouTube Short
            check_encoding: Whether to check for encoding issues (slower)
            **kwargs: Additional parameters

        Returns:
            AgentResult with VideoQualityResult data

        Example:
            >>> agent = VideoQualityAgent()
            >>> result = agent.run(video_file="output.mp4", is_short=True)
            >>> print(f"Resolution: {result.data['resolution']}")
            Resolution: 1080x1920
        """
        logger.info(f"[VideoQualityAgent] Analyzing video: {video_file} (Short: {is_short})")

        # Validate input
        if not video_file:
            return AgentResult(
                success=False,
                data=VideoQualityResult(passed=False, issues=["No video file provided"]).to_dict(),
                error="No video file provided"
            )

        if not os.path.exists(video_file):
            return AgentResult(
                success=False,
                data=VideoQualityResult(passed=False, issues=["Video file not found"]).to_dict(),
                error=f"Video file not found: {video_file}"
            )

        quality_result = VideoQualityResult(passed=True)

        # Get file size
        file_size = os.path.getsize(video_file)
        quality_result.file_size_mb = file_size / (1024 * 1024)

        # Check if FFprobe is available
        if not self.ffprobe:
            quality_result.issues.append("FFprobe not available for detailed analysis")
            quality_result.passed = False
            return AgentResult(
                success=False,
                data=quality_result.to_dict(),
                error="FFprobe not found"
            )

        # Run all quality checks
        self._get_video_info(video_file, quality_result)
        self._check_resolution(quality_result, is_short)
        self._check_bitrate(quality_result)
        self._check_frame_rate(quality_result)
        self._check_codec(quality_result)
        self._check_duration(quality_result, is_short)
        self._check_audio_track(video_file, quality_result)

        if check_encoding and self.ffmpeg:
            self._check_encoding_issues(video_file, quality_result)

        # Generate recommendations
        self._generate_recommendations(quality_result, is_short)

        # Determine final pass/fail status
        quality_result.passed = len(quality_result.issues) == 0

        # Log results
        if quality_result.passed:
            logger.success(
                f"[VideoQualityAgent] Video passed quality check "
                f"({quality_result.resolution}, {quality_result.bitrate:.1f} Mbps, "
                f"{quality_result.frame_rate:.1f} fps)"
            )
        else:
            logger.warning(
                f"[VideoQualityAgent] Video quality issues found: "
                f"{len(quality_result.issues)} issues"
            )

        return AgentResult(
            success=True,
            data=quality_result.to_dict(),
            tokens_used=0,
            cost=0.0,
            metadata={
                "video_file": video_file,
                "is_short": is_short,
                "checks_performed": [
                    "video_info", "resolution", "bitrate", "frame_rate",
                    "codec", "duration", "audio_track",
                    "encoding" if check_encoding else None
                ]
            }
        )

    def _get_video_info(self, video_file: str, result: VideoQualityResult):
        """Get video file metadata using FFprobe."""
        try:
            cmd = [
                self.ffprobe,
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,codec_name,pix_fmt,r_frame_rate,bit_rate,duration',
                '-show_entries', 'format=duration,bit_rate',
                '-of', 'json',
                video_file
            ]

            proc_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if proc_result.returncode != 0:
                result.warnings.append("Could not read video metadata")
                return

            data = json.loads(proc_result.stdout)

            # Extract stream info
            if 'streams' in data and data['streams']:
                stream = data['streams'][0]

                result.width = int(stream.get('width', 0))
                result.height = int(stream.get('height', 0))
                result.resolution = f"{result.width}x{result.height}"
                result.codec = stream.get('codec_name', '').lower()
                result.pixel_format = stream.get('pix_fmt', '')

                # Parse frame rate (can be "30/1" or "30000/1001")
                fps_str = stream.get('r_frame_rate', '0/1')
                if '/' in fps_str:
                    num, den = fps_str.split('/')
                    result.frame_rate = float(num) / float(den) if float(den) != 0 else 0
                else:
                    result.frame_rate = float(fps_str)

                # Get bitrate from stream or format
                bit_rate = stream.get('bit_rate')
                if bit_rate:
                    result.bitrate = int(bit_rate) / 1_000_000  # Convert to Mbps

            # Get format-level info (fallback for bitrate and duration)
            if 'format' in data:
                fmt = data['format']

                if not result.duration:
                    result.duration = float(fmt.get('duration', 0))

                if not result.bitrate:
                    bit_rate = fmt.get('bit_rate')
                    if bit_rate:
                        result.bitrate = int(bit_rate) / 1_000_000

            # Also get duration from stream if not in format
            if not result.duration and 'streams' in data and data['streams']:
                stream_duration = data['streams'][0].get('duration')
                if stream_duration:
                    result.duration = float(stream_duration)

            logger.debug(
                f"Video info: {result.resolution}, {result.codec}, "
                f"{result.bitrate:.1f} Mbps, {result.frame_rate:.1f} fps"
            )

        except subprocess.TimeoutExpired:
            logger.error("Video info extraction timed out")
            result.warnings.append("Video info extraction timed out")
        except Exception as e:
            logger.error(f"Video info extraction failed: {e}")
            result.warnings.append(f"Video info error: {str(e)[:50]}")

    def _check_resolution(self, result: VideoQualityResult, is_short: bool):
        """Check video resolution."""
        if not result.width or not result.height:
            result.issues.append("Could not determine video resolution")
            return

        target_res = self.RESOLUTION_SHORTS if is_short else self.RESOLUTION_REGULAR
        target_width, target_height = target_res

        # Check exact match
        if (result.width, result.height) == (target_width, target_height):
            return  # Perfect match

        # Check aspect ratio
        target_aspect = target_width / target_height
        actual_aspect = result.width / result.height
        aspect_diff = abs(target_aspect - actual_aspect)

        if aspect_diff > 0.1:  # More than 10% difference
            expected = "9:16 portrait" if is_short else "16:9 landscape"
            result.issues.append(
                f"Wrong aspect ratio: {result.resolution} (expected {expected})"
            )
        elif result.width < target_width or result.height < target_height:
            result.warnings.append(
                f"Resolution below optimal: {result.resolution} "
                f"(recommended: {target_width}x{target_height})"
            )

    def _check_bitrate(self, result: VideoQualityResult):
        """Check video bitrate."""
        if result.bitrate is None:
            result.warnings.append("Could not determine video bitrate")
            return

        if result.bitrate < self.MIN_BITRATE_MBPS:
            result.issues.append(
                f"Bitrate too low: {result.bitrate:.1f} Mbps "
                f"(minimum: {self.MIN_BITRATE_MBPS} Mbps)"
            )
        elif result.bitrate < self.TARGET_BITRATE_MBPS:
            result.warnings.append(
                f"Bitrate below optimal: {result.bitrate:.1f} Mbps "
                f"(recommended: {self.TARGET_BITRATE_MBPS} Mbps for 1080p)"
            )

    def _check_frame_rate(self, result: VideoQualityResult):
        """Check video frame rate."""
        if result.frame_rate is None:
            result.warnings.append("Could not determine frame rate")
            return

        if result.frame_rate < self.MIN_FRAME_RATE:
            result.issues.append(
                f"Frame rate too low: {result.frame_rate:.1f} fps "
                f"(minimum: {self.MIN_FRAME_RATE} fps)"
            )
        elif result.frame_rate < self.TARGET_FRAME_RATE - 1:
            result.warnings.append(
                f"Frame rate below standard: {result.frame_rate:.1f} fps "
                f"(recommended: {self.TARGET_FRAME_RATE} fps)"
            )
        elif result.frame_rate > self.MAX_FRAME_RATE:
            result.warnings.append(
                f"Very high frame rate: {result.frame_rate:.1f} fps "
                f"(may increase file size significantly)"
            )

    def _check_codec(self, result: VideoQualityResult):
        """Check video codec."""
        if not result.codec:
            result.warnings.append("Could not determine video codec")
            return

        codec_lower = result.codec.lower()

        if codec_lower not in self.SUPPORTED_CODECS:
            result.issues.append(
                f"Unsupported codec: {result.codec} "
                f"(supported: {', '.join(self.SUPPORTED_CODECS)})"
            )
        elif codec_lower not in self.RECOMMENDED_CODECS:
            result.warnings.append(
                f"Codec {result.codec} may have compatibility issues. "
                f"Recommended: {', '.join(self.RECOMMENDED_CODECS)}"
            )

        # Check pixel format
        if result.pixel_format and result.pixel_format != self.RECOMMENDED_PIXEL_FORMAT:
            result.warnings.append(
                f"Pixel format {result.pixel_format} may cause issues. "
                f"Recommended: {self.RECOMMENDED_PIXEL_FORMAT}"
            )

    def _check_duration(self, result: VideoQualityResult, is_short: bool):
        """Check video duration."""
        if not result.duration:
            result.warnings.append("Could not determine video duration")
            return

        if is_short:
            if result.duration > self.MAX_DURATION_SHORTS:
                result.issues.append(
                    f"Short too long: {result.duration:.1f}s "
                    f"(max: {self.MAX_DURATION_SHORTS}s)"
                )
            elif result.duration < self.MIN_DURATION_SHORTS:
                result.issues.append(
                    f"Short too brief: {result.duration:.1f}s "
                    f"(min: {self.MIN_DURATION_SHORTS}s)"
                )
            elif not (self.OPTIMAL_DURATION_SHORTS[0] <= result.duration <= self.OPTIMAL_DURATION_SHORTS[1]):
                result.warnings.append(
                    f"Short duration ({result.duration:.1f}s) outside optimal range "
                    f"({self.OPTIMAL_DURATION_SHORTS[0]}-{self.OPTIMAL_DURATION_SHORTS[1]}s)"
                )
        else:
            if result.duration > self.MAX_DURATION_REGULAR:
                result.issues.append(
                    f"Video too long: {result.duration / 3600:.1f} hours "
                    f"(max: {self.MAX_DURATION_REGULAR / 3600:.0f} hours)"
                )

    def _check_audio_track(self, video_file: str, result: VideoQualityResult):
        """Check if video has audio track."""
        try:
            cmd = [
                self.ffprobe,
                '-v', 'error',
                '-select_streams', 'a',
                '-show_entries', 'stream=codec_type',
                '-of', 'json',
                video_file
            ]

            proc_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if proc_result.returncode == 0:
                data = json.loads(proc_result.stdout)
                streams = data.get('streams', [])
                result.has_audio = len(streams) > 0

                if not result.has_audio:
                    result.issues.append("No audio track detected")

        except Exception as e:
            logger.debug(f"Audio track check failed: {e}")
            result.warnings.append("Could not verify audio track")

    def _check_encoding_issues(self, video_file: str, result: VideoQualityResult):
        """Check for encoding issues by analyzing a portion of the video."""
        try:
            # Check for errors in the first 10 seconds
            cmd = [
                self.ffmpeg,
                '-v', 'error',
                '-i', video_file,
                '-t', '10',
                '-f', 'null',
                '-y',
                'NUL' if os.name == 'nt' else '/dev/null'
            ]

            proc_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            # Check stderr for errors
            if proc_result.stderr:
                error_lines = [
                    line for line in proc_result.stderr.split('\n')
                    if 'error' in line.lower() or 'corrupt' in line.lower()
                ]
                if error_lines:
                    result.warnings.append(
                        f"Potential encoding issues detected: {len(error_lines)} errors"
                    )

        except subprocess.TimeoutExpired:
            result.warnings.append("Encoding check timed out")
        except Exception as e:
            logger.debug(f"Encoding check failed: {e}")

    def _generate_recommendations(self, result: VideoQualityResult, is_short: bool):
        """Generate actionable recommendations based on analysis."""
        if result.bitrate and result.bitrate < self.MIN_BITRATE_MBPS:
            result.recommendations.append(
                f"Re-encode video with higher bitrate (target: {self.TARGET_BITRATE_MBPS} Mbps)"
            )

        if result.frame_rate and result.frame_rate < self.MIN_FRAME_RATE:
            result.recommendations.append(
                f"Re-encode video with {self.TARGET_FRAME_RATE} fps"
            )

        if result.codec and result.codec.lower() not in self.RECOMMENDED_CODECS:
            result.recommendations.append(
                "Re-encode video using H.264 codec for best compatibility"
            )

        if result.width and result.height:
            target = self.RESOLUTION_SHORTS if is_short else self.RESOLUTION_REGULAR
            if (result.width, result.height) != target:
                result.recommendations.append(
                    f"Resize video to {target[0]}x{target[1]} for optimal quality"
                )

        if not result.has_audio:
            result.recommendations.append(
                "Add audio track (narration or background music)"
            )

        if not result.recommendations and not result.issues:
            result.recommendations.append(
                "Video quality meets YouTube recommendations"
            )

    def quick_check(self, video_file: str, is_short: bool = False) -> Dict[str, Any]:
        """
        Quick video quality check with minimal output.

        Args:
            video_file: Path to video file
            is_short: Whether this is a YouTube Short

        Returns:
            Dictionary with passed status and key metrics
        """
        result = self.run(video_file=video_file, is_short=is_short, check_encoding=False)
        data = result.data

        return {
            "passed": data["passed"],
            "resolution": data["resolution"],
            "bitrate": data["bitrate"],
            "fps": data["frame_rate"],
            "codec": data["codec"],
            "issues_count": len(data["issues"])
        }


# CLI entry point
def main():
    """CLI entry point for video quality agent."""
    import sys

    if len(sys.argv) < 2:
        print("""
Video Quality Agent - Video Quality Validation for YouTube

Usage:
    python -m src.agents.video_quality_agent <video_file> [options]

Options:
    --short         Check as YouTube Short (9:16 aspect ratio)
    --quick         Quick check (minimal output)
    --json          Output as JSON
    --no-encoding   Skip encoding issue check

Examples:
    python -m src.agents.video_quality_agent video.mp4
    python -m src.agents.video_quality_agent short.mp4 --short
    python -m src.agents.video_quality_agent output.mp4 --json

Targets:
    - Resolution: 1920x1080 (regular) or 1080x1920 (Shorts)
    - Bitrate: >= 8 Mbps (12 Mbps recommended)
    - Frame Rate: 30 fps
    - Codec: H.264 or H.265
        """)
        return

    # Parse arguments
    video_file = sys.argv[1]
    is_short = "--short" in sys.argv
    quick_mode = "--quick" in sys.argv
    output_json = "--json" in sys.argv
    check_encoding = "--no-encoding" not in sys.argv

    # Run agent
    agent = VideoQualityAgent()

    if quick_mode:
        result = agent.quick_check(video_file, is_short)
        if output_json:
            print(json.dumps(result, indent=2))
        else:
            status = "PASS" if result["passed"] else "FAIL"
            print(
                f"{status} | {result['resolution']} | "
                f"{result['bitrate']:.1f} Mbps | {result['fps']:.1f} fps | "
                f"{result['codec']} | Issues: {result['issues_count']}"
            )
    else:
        result = agent.run(
            video_file=video_file,
            is_short=is_short,
            check_encoding=check_encoding
        )

        if output_json:
            print(json.dumps(result.data, indent=2))
        else:
            print("\n" + "=" * 60)
            print("VIDEO QUALITY AGENT RESULT")
            print("=" * 60)

            data = result.data
            status = "PASSED" if data["passed"] else "FAILED"
            format_type = "YouTube Short" if is_short else "Regular Video"

            print(f"Status: {status}")
            print(f"Format: {format_type}")
            print(f"\nMetrics:")
            print(f"  Resolution: {data['resolution']}")
            print(f"  Bitrate: {data['bitrate']:.1f} Mbps" if data['bitrate'] else "  Bitrate: N/A")
            print(f"  Frame Rate: {data['frame_rate']:.1f} fps" if data['frame_rate'] else "  Frame Rate: N/A")
            print(f"  Codec: {data['codec']}")
            print(f"  Pixel Format: {data['pixel_format']}")
            print(f"  Duration: {data['duration']:.1f}s" if data['duration'] else "  Duration: N/A")
            print(f"  File Size: {data['file_size_mb']:.1f} MB" if data['file_size_mb'] else "  File Size: N/A")
            print(f"  Has Audio: {'Yes' if data['has_audio'] else 'No'}")

            if data["issues"]:
                print(f"\nIssues ({len(data['issues'])}):")
                for issue in data["issues"]:
                    print(f"  [X] {issue}")

            if data["warnings"]:
                print(f"\nWarnings ({len(data['warnings'])}):")
                for warning in data["warnings"]:
                    print(f"  [!] {warning}")

            if data["recommendations"]:
                print(f"\nRecommendations:")
                for rec in data["recommendations"]:
                    print(f"  -> {rec}")


if __name__ == "__main__":
    main()
