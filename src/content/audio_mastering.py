"""
Professional Audio Mastering Pipeline

Broadcast-quality audio processing for YouTube content.
Applies industry-standard mastering chain for professional sound.

Pipeline stages:
1. FFT Noise Reduction - Removes background noise
2. High-Pass Filter - Removes rumble below 80Hz
3. Niche-Specific EQ - Tailored frequency response
4. De-esser - Tames harsh sibilants
5. Compression - Smooths dynamics
6. True Peak Limiting - Prevents clipping
7. Loudness Normalization - Targets -14 LUFS
"""

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger


@dataclass
class MasteringPreset:
    """Audio mastering preset for a niche."""

    name: str
    eq_bands: List[Tuple[int, float, float]]  # (freq, width, gain)
    compression_threshold: float
    compression_ratio: float
    target_lufs: float
    description: str


# Niche-specific mastering presets
MASTERING_PRESETS: Dict[str, MasteringPreset] = {
    "finance": MasteringPreset(
        name="Finance/Authority",
        eq_bands=[
            (200, 1.0, 2.0),    # Boost low-mids for authority
            (5000, 2.0, -1.0),  # Slight cut for smoothness
        ],
        compression_threshold=-18.0,
        compression_ratio=3.0,
        target_lufs=-14.0,
        description="Authoritative, clear voice for finance content"
    ),
    "psychology": MasteringPreset(
        name="Psychology/Warm",
        eq_bands=[
            (100, 1.0, 3.0),    # Boost lows for warmth
            (3500, 1.0, 2.0),   # Boost presence for clarity
        ],
        compression_threshold=-20.0,
        compression_ratio=2.5,
        target_lufs=-14.0,
        description="Warm, intimate voice for psychology content"
    ),
    "storytelling": MasteringPreset(
        name="Storytelling/Dramatic",
        eq_bands=[
            (80, 1.0, 4.0),     # Deep bass for drama
            (10000, 1.0, 3.0),  # Air for presence
        ],
        compression_threshold=-16.0,
        compression_ratio=4.0,
        target_lufs=-14.0,
        description="Dramatic, cinematic voice for storytelling"
    ),
    "default": MasteringPreset(
        name="Default/Balanced",
        eq_bands=[
            (150, 1.0, 1.5),    # Slight low boost
            (4000, 1.5, 1.0),   # Slight presence boost
        ],
        compression_threshold=-18.0,
        compression_ratio=3.0,
        target_lufs=-14.0,
        description="Balanced mastering for general content"
    ),
}


class AudioMasteringPipeline:
    """
    Professional audio mastering for YouTube.

    Matches quality of top creators like:
    - MrBallen (exceptional clarity)
    - MKBHD (broadcast-quality voiceovers)
    - Kurzgesagt (clean, punchy narration)
    """

    def __init__(self, ffmpeg_path: Optional[str] = None):
        """
        Initialize mastering pipeline.

        Args:
            ffmpeg_path: Path to FFmpeg binary (auto-detected if not provided)
        """
        self.ffmpeg = ffmpeg_path or self._find_ffmpeg()
        if not self.ffmpeg:
            raise RuntimeError("FFmpeg not found. Please install FFmpeg.")

        logger.info(f"AudioMasteringPipeline initialized with FFmpeg: {self.ffmpeg}")

    def master_voice_track(
        self,
        input_file: str,
        output_file: str,
        niche: str = "default",
        target_lufs: Optional[float] = None,
        noise_reduction: bool = True,
        normalize: bool = True
    ) -> str:
        """
        Apply full mastering chain to voice track.

        Args:
            input_file: Input audio file
            output_file: Output audio file
            niche: Content niche for preset selection
            target_lufs: Override target loudness (default: -14 LUFS)
            noise_reduction: Apply FFT noise reduction
            normalize: Apply loudness normalization

        Returns:
            Path to mastered audio
        """
        # Get preset
        preset = MASTERING_PRESETS.get(niche, MASTERING_PRESETS["default"])
        lufs = target_lufs or preset.target_lufs

        # Build filter chain
        filter_chain = self._build_mastering_chain(
            preset=preset,
            target_lufs=lufs,
            noise_reduction=noise_reduction,
            normalize=normalize
        )

        # Run FFmpeg
        cmd = [
            self.ffmpeg,
            "-i", input_file,
            "-af", filter_chain,
            "-c:a", "aac",
            "-b:a", "192k",
            "-y",  # Overwrite
            output_file
        ]

        logger.info(f"Mastering voice: {input_file} -> {output_file}")
        logger.debug(f"Filter chain: {filter_chain}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Mastering complete: {output_file}")

            # Verify loudness
            if normalize:
                actual_lufs = self._measure_lufs(output_file)
                if actual_lufs:
                    logger.info(f"Final loudness: {actual_lufs:.1f} LUFS (target: {lufs})")

            return output_file
        except subprocess.CalledProcessError as e:
            logger.error(f"Mastering failed: {e.stderr}")
            raise

    def apply_music_ducking(
        self,
        voice_file: str,
        music_file: str,
        output_file: str,
        duck_level: float = -12.0,
        attack_ms: float = 200,
        release_ms: float = 500
    ) -> str:
        """
        Apply sidechain ducking - lower music when voice is present.

        Args:
            voice_file: Voice track
            music_file: Background music track
            output_file: Mixed output file
            duck_level: How much to duck music (dB)
            attack_ms: Duck attack time
            release_ms: Duck release time

        Returns:
            Path to mixed audio
        """
        # FFmpeg sidechain compression filter
        filter_complex = (
            f"[0:a]asplit=2[voice][sc];"
            f"[1:a][sc]sidechaincompress="
            f"threshold=0.02:ratio=10:attack={attack_ms}:release={release_ms}:"
            f"level_sc=1[ducked];"
            f"[voice][ducked]amix=inputs=2:duration=first:weights=1 0.15"
        )

        cmd = [
            self.ffmpeg,
            "-i", voice_file,
            "-i", music_file,
            "-filter_complex", filter_complex,
            "-c:a", "aac",
            "-b:a", "192k",
            "-y",
            output_file
        ]

        logger.info(f"Applying music ducking: {voice_file} + {music_file}")

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Ducking complete: {output_file}")
            return output_file
        except subprocess.CalledProcessError as e:
            logger.error(f"Ducking failed: {e.stderr}")
            raise

    def master_batch(
        self,
        files: List[str],
        output_dir: str,
        niche: str = "default"
    ) -> List[str]:
        """
        Master multiple audio files.

        Args:
            files: List of input files
            output_dir: Output directory
            niche: Content niche

        Returns:
            List of mastered file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        mastered = []
        for input_file in files:
            input_path = Path(input_file)
            output_file = output_path / f"{input_path.stem}_mastered{input_path.suffix}"

            try:
                result = self.master_voice_track(
                    input_file=input_file,
                    output_file=str(output_file),
                    niche=niche
                )
                mastered.append(result)
            except Exception as e:
                logger.error(f"Failed to master {input_file}: {e}")

        logger.info(f"Batch mastering complete: {len(mastered)}/{len(files)} successful")
        return mastered

    def enhance_for_youtube(
        self,
        input_file: str,
        output_file: str
    ) -> str:
        """
        Quick enhancement optimized for YouTube.

        Applies:
        - Noise reduction
        - High-pass filter (80Hz)
        - Loudness normalization (-14 LUFS)
        - True peak limiting (-1dB)

        Args:
            input_file: Input audio
            output_file: Output audio

        Returns:
            Path to enhanced audio
        """
        filter_chain = (
            "afftdn=nf=-25,"
            "highpass=f=80,"
            "loudnorm=I=-14:TP=-1:LRA=11"
        )

        cmd = [
            self.ffmpeg,
            "-i", input_file,
            "-af", filter_chain,
            "-c:a", "aac",
            "-b:a", "192k",
            "-y",
            output_file
        ]

        subprocess.run(cmd, capture_output=True, check=True)
        logger.info(f"YouTube enhancement complete: {output_file}")
        return output_file

    def _build_mastering_chain(
        self,
        preset: MasteringPreset,
        target_lufs: float,
        noise_reduction: bool,
        normalize: bool
    ) -> str:
        """Build FFmpeg audio filter chain."""
        filters = []

        # 1. Noise reduction (FFT-based)
        if noise_reduction:
            filters.append("afftdn=nf=-25")

        # 2. High-pass filter (remove rumble)
        filters.append("highpass=f=80")

        # 3. EQ bands from preset
        for freq, width, gain in preset.eq_bands:
            filters.append(
                f"equalizer=f={freq}:width_type=o:width={width}:g={gain}"
            )

        # 4. De-esser (reduce harsh S sounds)
        filters.append("deesser=i=0.05:m=0.5:f=6000:s=o")

        # 5. Compression
        filters.append(
            f"acompressor="
            f"threshold={preset.compression_threshold}dB:"
            f"ratio={preset.compression_ratio}:"
            f"attack=5:release=50"
        )

        # 6. Limiter (prevent clipping)
        filters.append("alimiter=limit=-1dB:attack=5:release=50")

        # 7. Loudness normalization
        if normalize:
            filters.append(f"loudnorm=I={target_lufs}:TP=-1:LRA=11")

        return ",".join(filters)

    def _measure_lufs(self, audio_file: str) -> Optional[float]:
        """Measure integrated loudness of audio file."""
        cmd = [
            self.ffmpeg,
            "-i", audio_file,
            "-af", "loudnorm=I=-14:print_format=json",
            "-f", "null",
            "-"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            # Parse LUFS from output
            import re
            match = re.search(r'"input_i"\s*:\s*"(-?\d+\.?\d*)"', result.stderr)
            if match:
                return float(match.group(1))
        except Exception as e:
            logger.debug(f"Could not measure LUFS: {e}")

        return None

    def _find_ffmpeg(self) -> Optional[str]:
        """Find FFmpeg binary."""
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            return ffmpeg

        # Check common locations
        common_paths = [
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "C:\\ffmpeg\\bin\\ffmpeg.exe",
            "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe",
        ]

        for path in common_paths:
            if Path(path).exists():
                return path

        return None


# Convenience functions

def master_voice(
    input_file: str,
    output_file: str,
    niche: str = "default"
) -> str:
    """Quick voice mastering function."""
    pipeline = AudioMasteringPipeline()
    return pipeline.master_voice_track(input_file, output_file, niche)


def enhance_for_youtube(input_file: str, output_file: str) -> str:
    """Quick YouTube enhancement function."""
    pipeline = AudioMasteringPipeline()
    return pipeline.enhance_for_youtube(input_file, output_file)


def get_preset(niche: str) -> MasteringPreset:
    """Get mastering preset for niche."""
    return MASTERING_PRESETS.get(niche, MASTERING_PRESETS["default"])
