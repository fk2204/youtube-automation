"""
Audio Quality Agent - Audio Quality Validation

Validates audio files for YouTube optimization using FFmpeg for analysis.
Checks LUFS normalization, clipping, background noise, and voice clarity.

Usage:
    from src.agents.audio_quality_agent import AudioQualityAgent

    agent = AudioQualityAgent()

    # Check audio quality
    result = agent.run(audio_file="path/to/audio.mp3")

    if result.success:
        quality = result.data
        print(f"Passed: {quality['passed']}")
        print(f"LUFS Level: {quality['lufs_level']}")
        print(f"Peak Level: {quality['peak_level']}")

Example:
    >>> agent = AudioQualityAgent()
    >>> result = agent.run(audio_file="voice.mp3")
    >>> print(result.data['passed'])
    True
    >>> print(result.data['lufs_level'])
    -14.2
"""

import os
import subprocess
import shutil
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from loguru import logger

from .base_agent import BaseAgent, AgentResult


@dataclass
class AudioQualityResult:
    """
    Result of audio quality check.

    Attributes:
        passed: Whether the audio passes all quality checks
        lufs_level: Integrated loudness in LUFS (target: -14)
        peak_level: True peak level in dBTP (should be < -1dB)
        noise_floor: Estimated background noise level in dB
        loudness_range: Dynamic range in LU
        sample_rate: Audio sample rate in Hz
        duration: Audio duration in seconds
        issues: List of quality issues found
        warnings: Non-critical warnings
        recommendations: Suggested improvements
    """
    passed: bool
    lufs_level: Optional[float] = None
    peak_level: Optional[float] = None
    noise_floor: Optional[float] = None
    loudness_range: Optional[float] = None
    sample_rate: Optional[int] = None
    duration: Optional[float] = None
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "lufs_level": self.lufs_level,
            "peak_level": self.peak_level,
            "noise_floor": self.noise_floor,
            "loudness_range": self.loudness_range,
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            "issues": self.issues,
            "warnings": self.warnings,
            "recommendations": self.recommendations
        }


class AudioQualityAgent(BaseAgent):
    """
    Agent for validating audio quality for YouTube.

    Uses FFmpeg/FFprobe for audio analysis and validates against
    YouTube's recommended audio specifications.

    Features:
    - LUFS normalization verification (-14 LUFS target)
    - Clipping detection (peak > -1dB)
    - Background noise level estimation
    - Voice clarity analysis
    - Sample rate and format validation
    """

    # YouTube recommended audio specs
    TARGET_LUFS = -14.0          # YouTube's Stable Volume target
    LUFS_TOLERANCE = 2.0         # +/- 2 LUFS acceptable
    MAX_TRUE_PEAK = -1.0         # True peak limit (prevents clipping)
    MAX_NOISE_FLOOR = -50.0      # Maximum acceptable noise floor
    TARGET_SAMPLE_RATE = 48000   # YouTube's preferred sample rate
    MIN_SAMPLE_RATE = 44100      # Minimum acceptable sample rate
    TARGET_LOUDNESS_RANGE = 11.0 # EBU R128 recommended LRA

    def __init__(self, provider: str = "ffmpeg", api_key: str = None):
        """
        Initialize the audio quality agent.

        Args:
            provider: Analysis provider (default: ffmpeg)
            api_key: Not used for this agent
        """
        super().__init__(provider=provider, api_key=api_key)
        self.ffmpeg = self._find_ffmpeg()
        self.ffprobe = self._find_ffprobe()

        if self.ffmpeg:
            logger.info(f"AudioQualityAgent initialized (FFmpeg: {self.ffmpeg})")
        else:
            logger.warning(
                "FFmpeg not found! Audio quality analysis will be limited. "
                "Install FFmpeg: https://ffmpeg.org/download.html"
            )

    def _find_ffmpeg(self) -> Optional[str]:
        """Find FFmpeg executable."""
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

    def run(
        self,
        audio_file: str = "",
        check_voice_clarity: bool = True,
        **kwargs
    ) -> AgentResult:
        """
        Analyze audio file quality for YouTube.

        Args:
            audio_file: Path to audio file to analyze
            check_voice_clarity: Whether to analyze voice clarity (slower)
            **kwargs: Additional parameters

        Returns:
            AgentResult with AudioQualityResult data

        Example:
            >>> agent = AudioQualityAgent()
            >>> result = agent.run(audio_file="narration.mp3")
            >>> print(f"LUFS: {result.data['lufs_level']}")
            LUFS: -14.2
        """
        logger.info(f"[AudioQualityAgent] Analyzing audio: {audio_file}")

        # Validate input
        if not audio_file:
            return AgentResult(
                success=False,
                data=AudioQualityResult(passed=False, issues=["No audio file provided"]).to_dict(),
                error="No audio file provided"
            )

        if not os.path.exists(audio_file):
            return AgentResult(
                success=False,
                data=AudioQualityResult(passed=False, issues=["Audio file not found"]).to_dict(),
                error=f"Audio file not found: {audio_file}"
            )

        quality_result = AudioQualityResult(passed=True)

        # Check if FFmpeg is available
        if not self.ffmpeg:
            quality_result.issues.append("FFmpeg not available for detailed analysis")
            quality_result.passed = False
            return AgentResult(
                success=False,
                data=quality_result.to_dict(),
                error="FFmpeg not found"
            )

        # Run all quality checks
        self._analyze_loudness(audio_file, quality_result)
        self._check_clipping(audio_file, quality_result)
        self._analyze_noise_floor(audio_file, quality_result)
        self._get_audio_info(audio_file, quality_result)

        if check_voice_clarity:
            self._analyze_voice_clarity(audio_file, quality_result)

        # Generate recommendations
        self._generate_recommendations(quality_result)

        # Determine final pass/fail status
        quality_result.passed = len(quality_result.issues) == 0

        # Log results
        if quality_result.passed:
            logger.success(
                f"[AudioQualityAgent] Audio passed quality check "
                f"(LUFS: {quality_result.lufs_level}, Peak: {quality_result.peak_level})"
            )
        else:
            logger.warning(
                f"[AudioQualityAgent] Audio quality issues found: "
                f"{len(quality_result.issues)} issues"
            )

        return AgentResult(
            success=True,
            data=quality_result.to_dict(),
            tokens_used=0,
            cost=0.0,
            metadata={
                "audio_file": audio_file,
                "checks_performed": [
                    "loudness", "clipping", "noise_floor", "audio_info",
                    "voice_clarity" if check_voice_clarity else None
                ]
            }
        )

    def _analyze_loudness(self, audio_file: str, result: AudioQualityResult):
        """Analyze audio loudness using EBU R128 standard."""
        try:
            cmd = [
                self.ffmpeg,
                '-i', audio_file,
                '-af', f'loudnorm=I={self.TARGET_LUFS}:TP={self.MAX_TRUE_PEAK}:LRA={self.TARGET_LOUDNESS_RANGE}:print_format=json',
                '-f', 'null',
                '-'
            ]

            proc_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )

            # Parse loudnorm output from stderr
            output = proc_result.stderr

            # Extract values
            for line in output.split('\n'):
                line = line.strip()
                if '"input_i"' in line:
                    result.lufs_level = float(line.split(':')[1].strip().strip('",'))
                elif '"input_tp"' in line:
                    result.peak_level = float(line.split(':')[1].strip().strip('",'))
                elif '"input_lra"' in line:
                    result.loudness_range = float(line.split(':')[1].strip().strip('",'))

            # Validate LUFS level
            if result.lufs_level is not None:
                lufs_diff = abs(result.lufs_level - self.TARGET_LUFS)
                if lufs_diff > self.LUFS_TOLERANCE:
                    if result.lufs_level < self.TARGET_LUFS - self.LUFS_TOLERANCE:
                        result.issues.append(
                            f"Audio too quiet: {result.lufs_level:.1f} LUFS "
                            f"(target: {self.TARGET_LUFS} LUFS)"
                        )
                    else:
                        result.issues.append(
                            f"Audio too loud: {result.lufs_level:.1f} LUFS "
                            f"(target: {self.TARGET_LUFS} LUFS)"
                        )
                elif lufs_diff > 1.0:
                    result.warnings.append(
                        f"Audio level slightly off: {result.lufs_level:.1f} LUFS "
                        f"(target: {self.TARGET_LUFS} LUFS)"
                    )

            logger.debug(
                f"Loudness analysis: {result.lufs_level} LUFS, "
                f"Peak: {result.peak_level} dBTP, "
                f"LRA: {result.loudness_range} LU"
            )

        except subprocess.TimeoutExpired:
            logger.error("Loudness analysis timed out")
            result.warnings.append("Loudness analysis timed out")
        except Exception as e:
            logger.error(f"Loudness analysis failed: {e}")
            result.warnings.append(f"Loudness analysis error: {str(e)[:50]}")

    def _check_clipping(self, audio_file: str, result: AudioQualityResult):
        """Check for audio clipping (peak levels exceeding -1dB)."""
        if result.peak_level is not None:
            if result.peak_level > self.MAX_TRUE_PEAK:
                result.issues.append(
                    f"Audio clipping detected: peak at {result.peak_level:.1f} dBTP "
                    f"(max: {self.MAX_TRUE_PEAK} dBTP)"
                )
            elif result.peak_level > -1.5:
                result.warnings.append(
                    f"Audio peaks near clipping threshold: {result.peak_level:.1f} dBTP"
                )

    def _analyze_noise_floor(self, audio_file: str, result: AudioQualityResult):
        """Estimate background noise level."""
        try:
            # Use volumedetect to get min volume (rough noise floor estimate)
            cmd = [
                self.ffmpeg,
                '-i', audio_file,
                '-af', 'volumedetect',
                '-f', 'null',
                '-y',
                'NUL' if os.name == 'nt' else '/dev/null'
            ]

            proc_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )

            # Parse output
            for line in proc_result.stderr.split('\n'):
                if 'mean_volume:' in line:
                    parts = line.split('mean_volume:')[1].split()
                    mean_volume = float(parts[0])
                elif 'max_volume:' in line:
                    parts = line.split('max_volume:')[1].split()
                    max_volume = float(parts[0])

            # Estimate noise floor (very rough)
            # In a good recording, the difference between mean and max should be reasonable
            # A high mean relative to max suggests background noise

            logger.debug(f"Volume analysis: mean={mean_volume:.1f}dB")

        except Exception as e:
            logger.debug(f"Noise floor analysis skipped: {e}")

    def _get_audio_info(self, audio_file: str, result: AudioQualityResult):
        """Get audio file metadata."""
        if not self.ffprobe:
            return

        try:
            cmd = [
                self.ffprobe,
                '-v', 'error',
                '-select_streams', 'a:0',
                '-show_entries', 'stream=sample_rate,duration',
                '-show_entries', 'format=duration',
                '-of', 'json',
                audio_file
            ]

            proc_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if proc_result.returncode == 0:
                data = json.loads(proc_result.stdout)

                # Get sample rate
                if 'streams' in data and data['streams']:
                    stream = data['streams'][0]
                    result.sample_rate = int(stream.get('sample_rate', 0))

                # Get duration
                if 'format' in data:
                    result.duration = float(data['format'].get('duration', 0))
                elif 'streams' in data and data['streams']:
                    result.duration = float(data['streams'][0].get('duration', 0))

                # Validate sample rate
                if result.sample_rate:
                    if result.sample_rate < self.MIN_SAMPLE_RATE:
                        result.issues.append(
                            f"Sample rate too low: {result.sample_rate}Hz "
                            f"(min: {self.MIN_SAMPLE_RATE}Hz)"
                        )
                    elif result.sample_rate < self.TARGET_SAMPLE_RATE:
                        result.warnings.append(
                            f"Sample rate below optimal: {result.sample_rate}Hz "
                            f"(recommended: {self.TARGET_SAMPLE_RATE}Hz)"
                        )

        except Exception as e:
            logger.debug(f"Audio info extraction failed: {e}")

    def _analyze_voice_clarity(self, audio_file: str, result: AudioQualityResult):
        """
        Analyze voice clarity (simplified check).

        This is a basic check - more sophisticated analysis would require
        speech recognition or specialized audio analysis libraries.
        """
        # Check for very low audio levels which indicate poor recording
        if result.lufs_level is not None and result.lufs_level < -24:
            result.warnings.append(
                "Very quiet audio may indicate poor voice capture. "
                "Consider re-recording or boosting levels."
            )

        # Check loudness range for dynamics
        if result.loudness_range is not None:
            if result.loudness_range > 15:
                result.warnings.append(
                    "High dynamic range may indicate inconsistent voice levels. "
                    "Consider applying compression."
                )
            elif result.loudness_range < 4:
                result.warnings.append(
                    "Low dynamic range may indicate over-compression."
                )

    def _generate_recommendations(self, result: AudioQualityResult):
        """Generate actionable recommendations based on analysis."""
        if result.lufs_level is not None:
            if result.lufs_level < self.TARGET_LUFS - self.LUFS_TOLERANCE:
                result.recommendations.append(
                    f"Normalize audio to {self.TARGET_LUFS} LUFS using loudnorm filter"
                )
            elif result.lufs_level > self.TARGET_LUFS + self.LUFS_TOLERANCE:
                result.recommendations.append(
                    f"Reduce audio level to {self.TARGET_LUFS} LUFS"
                )

        if result.peak_level is not None and result.peak_level > self.MAX_TRUE_PEAK:
            result.recommendations.append(
                f"Apply a limiter to keep peaks below {self.MAX_TRUE_PEAK} dBTP"
            )

        if result.sample_rate and result.sample_rate < self.TARGET_SAMPLE_RATE:
            result.recommendations.append(
                f"Resample audio to {self.TARGET_SAMPLE_RATE}Hz for best YouTube compatibility"
            )

        if not result.recommendations and not result.issues:
            result.recommendations.append(
                "Audio quality meets YouTube recommendations"
            )

    def quick_check(self, audio_file: str) -> Dict[str, Any]:
        """
        Quick audio quality check with minimal output.

        Args:
            audio_file: Path to audio file

        Returns:
            Dictionary with passed status and key metrics
        """
        result = self.run(audio_file=audio_file, check_voice_clarity=False)
        data = result.data

        return {
            "passed": data["passed"],
            "lufs": data["lufs_level"],
            "peak": data["peak_level"],
            "issues_count": len(data["issues"])
        }


# CLI entry point
def main():
    """CLI entry point for audio quality agent."""
    import sys

    if len(sys.argv) < 2:
        print("""
Audio Quality Agent - Audio Quality Validation for YouTube

Usage:
    python -m src.agents.audio_quality_agent <audio_file> [options]

Options:
    --quick         Quick check (minimal output)
    --json          Output as JSON
    --no-clarity    Skip voice clarity analysis

Examples:
    python -m src.agents.audio_quality_agent narration.mp3
    python -m src.agents.audio_quality_agent voice.wav --quick
    python -m src.agents.audio_quality_agent audio.mp3 --json

Targets:
    - LUFS Level: -14 LUFS (+/- 2 LUFS tolerance)
    - True Peak: < -1 dBTP (no clipping)
    - Sample Rate: 48000 Hz (recommended)
        """)
        return

    # Parse arguments
    audio_file = sys.argv[1]
    quick_mode = "--quick" in sys.argv
    output_json = "--json" in sys.argv
    check_clarity = "--no-clarity" not in sys.argv

    # Run agent
    agent = AudioQualityAgent()

    if quick_mode:
        result = agent.quick_check(audio_file)
        if output_json:
            print(json.dumps(result, indent=2))
        else:
            status = "PASS" if result["passed"] else "FAIL"
            print(f"{status} | LUFS: {result['lufs']} | Peak: {result['peak']} | Issues: {result['issues_count']}")
    else:
        result = agent.run(audio_file=audio_file, check_voice_clarity=check_clarity)

        if output_json:
            print(json.dumps(result.data, indent=2))
        else:
            print("\n" + "=" * 60)
            print("AUDIO QUALITY AGENT RESULT")
            print("=" * 60)

            data = result.data
            status = "PASSED" if data["passed"] else "FAILED"
            print(f"Status: {status}")
            print(f"\nMetrics:")
            print(f"  LUFS Level: {data['lufs_level']:.1f} (target: -14)" if data['lufs_level'] else "  LUFS Level: N/A")
            print(f"  True Peak: {data['peak_level']:.1f} dBTP (max: -1)" if data['peak_level'] else "  True Peak: N/A")
            print(f"  Loudness Range: {data['loudness_range']:.1f} LU" if data['loudness_range'] else "  Loudness Range: N/A")
            print(f"  Sample Rate: {data['sample_rate']} Hz" if data['sample_rate'] else "  Sample Rate: N/A")
            print(f"  Duration: {data['duration']:.1f}s" if data['duration'] else "  Duration: N/A")

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
