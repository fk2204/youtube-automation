"""
Audio Processing and Enhancement
Normalizes audio levels for YouTube optimization

YouTube Audio Guidelines (based on research):
- Overall mix: -13 to -15 LUFS (Loudness Units Full Scale)
- Dialogue: -12dB
- Background music: -25 to -30dB (10-20dB below voice)
- YouTube's Stable Volume targets -14 LUFS
- True peak should not exceed -1dB to prevent clipping

Usage:
    from src.content.audio_processor import AudioProcessor

    processor = AudioProcessor()

    # Normalize audio to YouTube's recommended levels
    normalized = processor.normalize_audio("voice.mp3", "voice_normalized.mp3")

    # Mix voice with background music at proper levels
    mixed = processor.mix_with_background_music(
        voice_file="voice.mp3",
        music_file="background.mp3",
        output_file="mixed_audio.mp3"
    )

    # Analyze audio levels
    levels = processor.analyze_loudness("audio.mp3")
    print(f"Integrated loudness: {levels['integrated_loudness']} LUFS")

    # Phase 3 - Broadcast Quality Enhancement:

    # Apply professional 6-band EQ
    eq_result = processor.apply_multiband_eq("voice.mp3", "voice_eq.mp3")

    # Apply sidechain ducking (music ducks when voice present)
    ducked = processor.apply_sidechain_ducking(
        voice_file="voice.mp3",
        music_file="background.mp3",
        output_file="mixed_ducked.mp3"
    )

    # Apply broadcast compression (2-stage: compressor + limiter)
    compressed = processor.apply_broadcast_compression("voice.mp3", "voice_compressed.mp3")

    # Combined broadcast-quality processing (EQ + compression + normalization)
    broadcast = processor.process_broadcast_quality(
        input_file="voice.mp3",
        output_file="broadcast_voice.mp3",
        apply_eq=True,
        apply_compression=True
    )
"""

import os
import math
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class AudioLevels:
    """Audio level measurements."""
    integrated_loudness: float  # LUFS
    true_peak: float           # dBTP
    loudness_range: float      # LU
    threshold: float           # LUFS


class AudioProcessor:
    """
    Process and enhance audio for YouTube optimization.

    Uses FFmpeg's loudnorm filter for EBU R128 compliant loudness normalization.
    This ensures consistent audio levels across all videos.
    """

    # YouTube recommended levels
    TARGET_LUFS = -14.0        # YouTube's Stable Volume target
    TARGET_PEAK = -1.0         # True peak limit (prevents clipping)
    TARGET_LRA = 11.0          # Loudness range (dynamics)

    # Music mixing levels
    MUSIC_VOLUME_PERCENT = 15  # 15% volume for background music
    MUSIC_DB_REDUCTION = -18   # ~18dB below voice (-30dB absolute)

    # Sample rates
    YOUTUBE_SAMPLE_RATE = 48000  # YouTube's preferred sample rate

    def __init__(self):
        """Initialize the audio processor."""
        self.ffmpeg = self._find_ffmpeg()
        self.ffprobe = self._find_ffprobe()

        if self.ffmpeg:
            logger.info(f"AudioProcessor initialized (FFmpeg: {self.ffmpeg})")
        else:
            logger.warning(
                "FFmpeg not found! Audio processing will not work.\n"
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
            # Try to find ffprobe next to ffmpeg
            ffprobe = self.ffmpeg.replace("ffmpeg.exe", "ffprobe.exe")
            if ffprobe != self.ffmpeg and os.path.exists(ffprobe):
                return ffprobe

            # Non-Windows
            ffprobe = self.ffmpeg.replace("ffmpeg", "ffprobe")
            if os.path.exists(ffprobe):
                return ffprobe

        return None

    def analyze_loudness(self, audio_file: str) -> Optional[Dict[str, float]]:
        """
        Analyze audio file loudness using EBU R128 standard.

        Args:
            audio_file: Path to audio file

        Returns:
            Dict with integrated_loudness, true_peak, loudness_range, threshold
            or None if analysis fails
        """
        if not self.ffmpeg or not os.path.exists(audio_file):
            return None

        try:
            # Use loudnorm filter in measurement mode
            cmd = [
                self.ffmpeg,
                '-i', audio_file,
                '-af', 'loudnorm=I=-14:TP=-1:LRA=11:print_format=json',
                '-f', 'null',
                '-'
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )

            # Parse loudnorm output from stderr
            output = result.stderr

            # Extract values using simple parsing
            levels = {}

            for line in output.split('\n'):
                line = line.strip()
                if '"input_i"' in line:
                    levels['integrated_loudness'] = float(line.split(':')[1].strip().strip('",'))
                elif '"input_tp"' in line:
                    levels['true_peak'] = float(line.split(':')[1].strip().strip('",'))
                elif '"input_lra"' in line:
                    levels['loudness_range'] = float(line.split(':')[1].strip().strip('",'))
                elif '"input_thresh"' in line:
                    levels['threshold'] = float(line.split(':')[1].strip().strip('",'))

            if levels:
                logger.debug(
                    f"Audio analysis: {levels.get('integrated_loudness', 'N/A')} LUFS, "
                    f"Peak: {levels.get('true_peak', 'N/A')} dBTP"
                )
                return levels

            return None

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError) as e:
            logger.error(f"Loudness analysis failed: {e}")
            return None

    def normalize_audio(
        self,
        input_file: str,
        output_file: str = None,
        target_lufs: float = None,
        target_peak: float = None,
        two_pass: bool = True
    ) -> Optional[str]:
        """
        Normalize audio to YouTube's recommended levels (-14 LUFS).

        Uses EBU R128 loudness normalization for broadcast-quality results.
        Two-pass mode provides more accurate normalization.

        Args:
            input_file: Path to input audio file
            output_file: Path for output (default: adds _normalized suffix)
            target_lufs: Target integrated loudness (default: -14 LUFS)
            target_peak: Target true peak (default: -1 dBTP)
            two_pass: Use two-pass normalization for better accuracy

        Returns:
            Path to normalized audio file or None on failure
        """
        if not self.ffmpeg:
            logger.error("FFmpeg not available for audio normalization")
            return None

        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return None

        # Set defaults
        target_lufs = target_lufs or self.TARGET_LUFS
        target_peak = target_peak or self.TARGET_PEAK

        # Generate output filename if not provided
        if not output_file:
            input_path = Path(input_file)
            output_file = str(input_path.with_stem(f"{input_path.stem}_normalized"))

        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Normalizing audio to {target_lufs} LUFS: {input_file}")

        try:
            if two_pass:
                # Two-pass normalization for better accuracy
                return self._normalize_two_pass(
                    input_file, output_file, target_lufs, target_peak
                )
            else:
                # Single-pass (faster but less accurate)
                return self._normalize_single_pass(
                    input_file, output_file, target_lufs, target_peak
                )

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as e:
            logger.error(f"Audio normalization failed: {e}")
            return None

    def _normalize_single_pass(
        self,
        input_file: str,
        output_file: str,
        target_lufs: float,
        target_peak: float
    ) -> Optional[str]:
        """Single-pass loudness normalization."""
        cmd = [
            self.ffmpeg, '-y',
            '-i', input_file,
            '-af', f'loudnorm=I={target_lufs}:TP={target_peak}:LRA={self.TARGET_LRA}',
            '-ar', str(self.YOUTUBE_SAMPLE_RATE),
            '-b:a', '256k',
            output_file
        ]

        result = subprocess.run(cmd, capture_output=True, timeout=300)

        if result.returncode == 0 and os.path.exists(output_file):
            logger.info(f"Audio normalized (single-pass): {output_file}")
            return output_file

        logger.error(f"Single-pass normalization failed: {result.stderr.decode()[:500]}")
        return None

    def _normalize_two_pass(
        self,
        input_file: str,
        output_file: str,
        target_lufs: float,
        target_peak: float
    ) -> Optional[str]:
        """
        Two-pass loudness normalization for better accuracy.

        Pass 1: Analyze the audio to get exact loudness measurements
        Pass 2: Apply normalization with measured values
        """
        # Pass 1: Measure current loudness
        logger.debug("Pass 1: Analyzing audio loudness...")

        measure_cmd = [
            self.ffmpeg,
            '-i', input_file,
            '-af', f'loudnorm=I={target_lufs}:TP={target_peak}:LRA={self.TARGET_LRA}:print_format=json',
            '-f', 'null',
            '-'
        ]

        result = subprocess.run(measure_cmd, capture_output=True, text=True, timeout=300)

        # Parse measured values from first pass
        measured = self._parse_loudnorm_output(result.stderr)

        if not measured:
            logger.warning("Two-pass measurement failed, falling back to single-pass")
            return self._normalize_single_pass(input_file, output_file, target_lufs, target_peak)

        # Pass 2: Apply normalization with measured values
        logger.debug("Pass 2: Applying normalization...")

        normalize_cmd = [
            self.ffmpeg, '-y',
            '-i', input_file,
            '-af', (
                f'loudnorm=I={target_lufs}:TP={target_peak}:LRA={self.TARGET_LRA}:'
                f'measured_I={measured["input_i"]}:'
                f'measured_TP={measured["input_tp"]}:'
                f'measured_LRA={measured["input_lra"]}:'
                f'measured_thresh={measured["input_thresh"]}:'
                f'offset={measured.get("target_offset", 0)}:'
                'linear=true'
            ),
            '-ar', str(self.YOUTUBE_SAMPLE_RATE),
            '-b:a', '256k',
            output_file
        ]

        result = subprocess.run(normalize_cmd, capture_output=True, timeout=300)

        if result.returncode == 0 and os.path.exists(output_file):
            logger.success(f"Audio normalized (two-pass): {output_file}")
            return output_file

        logger.error(f"Two-pass normalization failed: {result.stderr.decode()[:500]}")
        return None

    def _parse_loudnorm_output(self, output: str) -> Optional[Dict[str, float]]:
        """Parse loudnorm filter JSON output."""
        try:
            values = {}
            keys = ['input_i', 'input_tp', 'input_lra', 'input_thresh', 'target_offset']

            for key in keys:
                for line in output.split('\n'):
                    if f'"{key}"' in line:
                        value_str = line.split(':')[1].strip().strip('",')
                        values[key] = float(value_str)
                        break

            if len(values) >= 4:  # Need at least the 4 main values
                return values

            return None

        except (ValueError, IndexError, KeyError) as e:
            logger.debug(f"Failed to parse loudnorm output: {e}")
            return None

    def mix_with_background_music(
        self,
        voice_file: str,
        music_file: str,
        output_file: str,
        music_volume: float = 0.15,
        fade_in_duration: float = 1.0,
        fade_out_duration: float = 2.0,
        normalize_before_mix: bool = True
    ) -> Optional[str]:
        """
        Mix voice narration with background music at proper levels.

        Voice stays at full volume, music is reduced to ~15% (about -18dB below voice).
        This ensures dialogue is clearly audible over music.

        Args:
            voice_file: Path to voice/narration audio
            music_file: Path to background music
            output_file: Path for mixed output
            music_volume: Music volume as fraction (0.15 = 15%, ~-16dB)
            fade_in_duration: Music fade-in at start (seconds)
            fade_out_duration: Music fade-out at end (seconds)
            normalize_before_mix: Normalize voice before mixing

        Returns:
            Path to mixed audio file or None on failure
        """
        if not self.ffmpeg:
            logger.error("FFmpeg not available for audio mixing")
            return None

        if not os.path.exists(voice_file):
            logger.error(f"Voice file not found: {voice_file}")
            return None

        if not os.path.exists(music_file):
            logger.warning(f"Music file not found: {music_file}")
            # Return normalized voice without music
            if normalize_before_mix:
                return self.normalize_audio(voice_file, output_file)
            return None

        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Mixing audio: voice + music at {music_volume*100:.0f}% volume")

        try:
            # Get voice duration
            voice_duration = self._get_audio_duration(voice_file)
            if not voice_duration:
                voice_duration = 60  # Fallback

            # Convert volume to dB for logging
            music_db = 20 * math.log10(music_volume) if music_volume > 0 else -60
            logger.debug(f"Music volume: {music_volume*100:.1f}% ({music_db:.1f} dB)")

            # Build filter complex for mixing
            # 1. Voice track: optionally normalize
            # 2. Music track: adjust volume, loop if needed, fade in/out
            # 3. Mix both tracks together
            # 4. Final normalization

            voice_filter = ""
            if normalize_before_mix:
                voice_filter = f"loudnorm=I={self.TARGET_LUFS}:TP={self.TARGET_PEAK}:LRA={self.TARGET_LRA}"

            # Build the filter complex
            # [0:a] = voice, [1:a] = music
            filters = []

            # Voice processing
            if voice_filter:
                filters.append(f"[0:a]{voice_filter}[voice]")
            else:
                filters.append("[0:a]acopy[voice]")

            # Music processing: volume, fade in/out
            music_filter = (
                f"[1:a]volume={music_volume},"
                f"afade=t=in:st=0:d={fade_in_duration},"
                f"afade=t=out:st={voice_duration - fade_out_duration}:d={fade_out_duration}"
                f"[music]"
            )
            filters.append(music_filter)

            # Mix together
            filters.append("[voice][music]amix=inputs=2:duration=first:dropout_transition=2[mixed]")

            # Final output normalization
            filters.append(
                f"[mixed]loudnorm=I={self.TARGET_LUFS}:TP={self.TARGET_PEAK}:LRA={self.TARGET_LRA}[out]"
            )

            filter_complex = ";".join(filters)

            cmd = [
                self.ffmpeg, '-y',
                '-i', voice_file,
                '-stream_loop', '-1',  # Loop music to match voice duration
                '-i', music_file,
                '-filter_complex', filter_complex,
                '-map', '[out]',
                '-t', str(voice_duration),
                '-ar', str(self.YOUTUBE_SAMPLE_RATE),
                '-b:a', '256k',
                output_file
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=600)

            if result.returncode == 0 and os.path.exists(output_file):
                logger.success(f"Audio mixed successfully: {output_file}")
                return output_file

            logger.error(f"Audio mixing failed: {result.stderr.decode()[:500]}")
            return None

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as e:
            logger.error(f"Audio mixing failed: {e}")
            return None

    def _get_audio_duration(self, audio_file: str) -> Optional[float]:
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

    def apply_multiband_eq(
        self,
        input_file: str,
        output_file: str
    ) -> Optional[str]:
        """
        Apply professional 6-band EQ for broadcast-quality voice audio.

        This replaces the single 3kHz boost with a complete frequency shaping chain:
        - 80Hz high-pass filter: Removes low-frequency rumble (mic handling, HVAC, etc.)
        - 150Hz -2dB cut: Reduces muddiness in the low-mids
        - 3kHz +3dB boost: Adds voice presence and intelligibility
        - 6.5kHz +2dB boost: Adds clarity and "air" to the voice
        - 11kHz +1.5dB boost: Adds brilliance and sparkle

        These frequency adjustments are based on professional broadcast standards
        and help voice audio cut through background music while remaining pleasant.

        Args:
            input_file: Path to input audio file
            output_file: Path for EQ-processed output file

        Returns:
            Path to processed audio file or None on failure

        Example:
            processor = AudioProcessor()
            result = processor.apply_multiband_eq(
                "raw_voice.mp3",
                "eq_voice.mp3"
            )
        """
        if not self.ffmpeg:
            logger.error("FFmpeg not available for multi-band EQ")
            return None

        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return None

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Applying 6-band professional EQ to: {input_file}")

        try:
            # Build the 6-band EQ filter chain
            # Each filter is applied in sequence for precise frequency shaping
            eq_filters = [
                # Band 1: High-pass at 80Hz - removes rumble
                "highpass=f=80",

                # Band 2: 150Hz -2dB cut - reduces muddiness
                # width_type=o means octave width, width=1 gives a gentle Q
                "equalizer=f=150:width_type=o:width=1:g=-2",

                # Band 3: 3kHz +3dB boost - voice presence
                # This is the primary intelligibility range for speech
                "equalizer=f=3000:width_type=o:width=1:g=3",

                # Band 4: 6.5kHz +2dB boost - clarity/air
                # Adds brightness without harshness
                "equalizer=f=6500:width_type=o:width=1:g=2",

                # Band 5: 11kHz +1.5dB boost - brilliance
                # Adds sparkle and definition to consonants
                "equalizer=f=11000:width_type=o:width=1.5:g=1.5",
            ]

            filter_chain = ",".join(eq_filters)

            logger.debug("EQ bands applied:")
            logger.debug("  - 80Hz high-pass (rumble removal)")
            logger.debug("  - 150Hz -2dB (muddiness reduction)")
            logger.debug("  - 3kHz +3dB (voice presence)")
            logger.debug("  - 6.5kHz +2dB (clarity/air)")
            logger.debug("  - 11kHz +1.5dB (brilliance)")

            cmd = [
                self.ffmpeg, '-y',
                '-i', input_file,
                '-af', filter_chain,
                '-ar', str(self.YOUTUBE_SAMPLE_RATE),
                '-b:a', '256k',
                output_file
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=300)

            if result.returncode == 0 and os.path.exists(output_file):
                logger.success(f"Multi-band EQ applied: {output_file}")
                return output_file

            error_msg = result.stderr.decode() if result.stderr else "Unknown error"
            logger.error(f"Multi-band EQ failed: {error_msg[:500]}")
            return None

        except subprocess.TimeoutExpired:
            logger.error("Multi-band EQ timed out (exceeded 5 minutes)")
            return None
        except (subprocess.SubprocessError, OSError) as e:
            logger.error(f"Multi-band EQ failed: {e}")
            return None

    def apply_sidechain_ducking(
        self,
        voice_file: str,
        music_file: str,
        output_file: str
    ) -> Optional[str]:
        """
        Apply automatic sidechain ducking - music volume drops when voice is present.

        This creates a professional podcast/broadcast sound where background music
        automatically ducks (reduces volume) when the speaker talks. Parameters:
        - Detection threshold: -35dB (voice above this triggers ducking)
        - Duck ratio: 30% (music drops to 70% volume when voice detected)
        - Attack time: 10ms (fast response to voice onset)
        - Release time: 100ms (smooth return when voice stops)

        The voice track controls the music volume via FFmpeg's sidechaincompress filter.

        Args:
            voice_file: Path to voice/narration audio (the control signal)
            music_file: Path to background music (will be ducked)
            output_file: Path for mixed output with ducking applied

        Returns:
            Path to mixed audio file with ducking or None on failure

        Example:
            processor = AudioProcessor()
            result = processor.apply_sidechain_ducking(
                "voice.mp3",
                "background_music.mp3",
                "mixed_ducked.mp3"
            )
        """
        if not self.ffmpeg:
            logger.error("FFmpeg not available for sidechain ducking")
            return None

        if not os.path.exists(voice_file):
            logger.error(f"Voice file not found: {voice_file}")
            return None

        if not os.path.exists(music_file):
            logger.error(f"Music file not found: {music_file}")
            return None

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Applying sidechain ducking: voice={voice_file}, music={music_file}")

        try:
            # Get voice duration to limit music length
            voice_duration = self._get_audio_duration(voice_file)
            if not voice_duration:
                voice_duration = 60  # Fallback

            # Sidechain ducking parameters
            # threshold: -35dB - voice level that triggers ducking
            # ratio: ~3.3:1 to achieve 30% reduction (duck to 70%)
            # attack: 10ms - fast response
            # release: 100ms - smooth release
            # level_sc: 1 - sidechain signal level

            # The sidechaincompress filter uses the voice as the sidechain input
            # to control compression (ducking) of the music track

            # Calculate ratio for 30% ducking (music at 70% when triggered)
            # With threshold=-35dB and makeup gain, ratio of 3.33:1 gives ~30% reduction
            duck_ratio = 3.33

            # Filter complex explanation:
            # [0:a] = voice (main signal and sidechain control)
            # [1:a] = music (will be ducked)
            # sidechaincompress: compresses music based on voice level
            # amix: combines voice and ducked music
            filter_complex = (
                # First, prepare the music with volume adjustment and loop
                f"[1:a]volume=0.15[music_vol];"
                # Apply sidechain compression: voice controls music ducking
                # level_sc=1 means full sidechain signal
                # detection=peak for faster response
                f"[music_vol][0:a]sidechaincompress="
                f"threshold=0.018:"  # -35dB in linear scale (10^(-35/20))
                f"ratio={duck_ratio}:"
                f"attack=10:"
                f"release=100:"
                f"level_sc=1:"
                f"detection=peak[ducked_music];"
                # Mix voice and ducked music together
                f"[0:a][ducked_music]amix=inputs=2:duration=first:dropout_transition=2[out]"
            )

            logger.debug("Sidechain ducking parameters:")
            logger.debug("  - Detection threshold: -35dB")
            logger.debug("  - Duck ratio: 30% (music at 70% when voice present)")
            logger.debug("  - Attack time: 10ms")
            logger.debug("  - Release time: 100ms")

            cmd = [
                self.ffmpeg, '-y',
                '-i', voice_file,
                '-stream_loop', '-1',  # Loop music to match voice duration
                '-i', music_file,
                '-filter_complex', filter_complex,
                '-map', '[out]',
                '-t', str(voice_duration),
                '-ar', str(self.YOUTUBE_SAMPLE_RATE),
                '-b:a', '256k',
                output_file
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=600)

            if result.returncode == 0 and os.path.exists(output_file):
                logger.success(f"Sidechain ducking applied: {output_file}")
                return output_file

            error_msg = result.stderr.decode() if result.stderr else "Unknown error"
            logger.error(f"Sidechain ducking failed: {error_msg[:500]}")
            return None

        except subprocess.TimeoutExpired:
            logger.error("Sidechain ducking timed out (exceeded 10 minutes)")
            return None
        except (subprocess.SubprocessError, OSError) as e:
            logger.error(f"Sidechain ducking failed: {e}")
            return None

    def apply_broadcast_compression(
        self,
        input_file: str,
        output_file: str
    ) -> Optional[str]:
        """
        Apply two-stage broadcast compression for professional loudness and consistency.

        This implements a broadcast-standard compression chain:

        Stage 1 - Compressor:
        - Threshold: -24dB (compress signals above this level)
        - Ratio: 4:1 (moderate-heavy compression)
        - Attack: 10ms (catches transients while preserving punch)
        - Release: 100ms (smooth release maintains natural feel)

        Stage 2 - Limiter:
        - Threshold: -1dB (brick-wall limit)
        - Prevents any clipping or digital distortion

        This two-stage approach is used in broadcast to:
        1. Even out dynamic range (compressor)
        2. Prevent clipping while maximizing loudness (limiter)

        Args:
            input_file: Path to input audio file
            output_file: Path for compressed output file

        Returns:
            Path to compressed audio file or None on failure

        Example:
            processor = AudioProcessor()
            result = processor.apply_broadcast_compression(
                "normalized_voice.mp3",
                "broadcast_voice.mp3"
            )
        """
        if not self.ffmpeg:
            logger.error("FFmpeg not available for broadcast compression")
            return None

        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return None

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Applying broadcast compression to: {input_file}")

        try:
            # Two-stage compression chain
            compression_filters = [
                # Stage 1: Main compressor
                # threshold=-24dB: compress signals above -24dB
                # ratio=4: 4:1 compression ratio
                # attack=10ms: fast enough to catch transients
                # release=100ms: smooth release for natural sound
                # makeup=2: add 2dB makeup gain to compensate for compression
                "acompressor=threshold=-24dB:ratio=4:attack=10:release=100:makeup=2",

                # Stage 2: Brick-wall limiter
                # This prevents any signal from exceeding -1dB
                # Using alimiter filter with limit=-1dB
                # attack=5ms: very fast to catch peaks
                # release=50ms: quick release
                # level=false: don't auto-level (we control levels explicitly)
                "alimiter=limit=0.891:"  # -1dB in linear scale (10^(-1/20) = 0.891)
                "attack=5:"
                "release=50:"
                "level=false"
            ]

            filter_chain = ",".join(compression_filters)

            logger.debug("Broadcast compression chain:")
            logger.debug("  Stage 1 - Compressor:")
            logger.debug("    - Threshold: -24dB")
            logger.debug("    - Ratio: 4:1")
            logger.debug("    - Attack: 10ms")
            logger.debug("    - Release: 100ms")
            logger.debug("  Stage 2 - Limiter:")
            logger.debug("    - Threshold: -1dB (brick-wall)")

            cmd = [
                self.ffmpeg, '-y',
                '-i', input_file,
                '-af', filter_chain,
                '-ar', str(self.YOUTUBE_SAMPLE_RATE),
                '-b:a', '256k',
                output_file
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=300)

            if result.returncode == 0 and os.path.exists(output_file):
                logger.success(f"Broadcast compression applied: {output_file}")
                return output_file

            error_msg = result.stderr.decode() if result.stderr else "Unknown error"
            logger.error(f"Broadcast compression failed: {error_msg[:500]}")
            return None

        except subprocess.TimeoutExpired:
            logger.error("Broadcast compression timed out (exceeded 5 minutes)")
            return None
        except (subprocess.SubprocessError, OSError) as e:
            logger.error(f"Broadcast compression failed: {e}")
            return None

    def process_broadcast_quality(
        self,
        input_file: str,
        output_file: str,
        apply_eq: bool = True,
        apply_compression: bool = True
    ) -> Optional[str]:
        """
        Apply complete broadcast-quality processing chain in a single pass.

        Combines multi-band EQ and broadcast compression into an optimized
        processing chain. This is more efficient than calling individual
        methods separately as it processes the audio in a single FFmpeg pass.

        Processing order (when all enabled):
        1. Multi-band EQ (frequency shaping)
           - 80Hz high-pass (rumble removal)
           - 150Hz -2dB (muddiness cut)
           - 3kHz +3dB (presence boost)
           - 6.5kHz +2dB (clarity/air)
           - 11kHz +1.5dB (brilliance)
        2. Broadcast compression
           - Stage 1: Compressor (-24dB threshold, 4:1 ratio)
           - Stage 2: Limiter (-1dB brick-wall)
        3. Final loudness normalization to -14 LUFS

        Args:
            input_file: Path to input audio file
            output_file: Path for processed output file
            apply_eq: Apply 6-band professional EQ (default: True)
            apply_compression: Apply two-stage broadcast compression (default: True)

        Returns:
            Path to processed audio file or None on failure

        Example:
            processor = AudioProcessor()

            # Full broadcast processing
            result = processor.process_broadcast_quality(
                "raw_voice.mp3",
                "broadcast_voice.mp3"
            )

            # EQ only (no compression)
            result = processor.process_broadcast_quality(
                "raw_voice.mp3",
                "eq_only.mp3",
                apply_compression=False
            )

            # Compression only (no EQ)
            result = processor.process_broadcast_quality(
                "raw_voice.mp3",
                "compressed_only.mp3",
                apply_eq=False
            )
        """
        if not self.ffmpeg:
            logger.error("FFmpeg not available for broadcast quality processing")
            return None

        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return None

        if not apply_eq and not apply_compression:
            logger.warning("No processing requested, returning normalized audio")
            return self.normalize_audio(input_file, output_file)

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Applying broadcast-quality processing to: {input_file}")
        logger.info(f"  - Multi-band EQ: {'enabled' if apply_eq else 'disabled'}")
        logger.info(f"  - Broadcast compression: {'enabled' if apply_compression else 'disabled'}")

        try:
            filters = []

            # Stage 1: Multi-band EQ (if enabled)
            if apply_eq:
                eq_filters = [
                    # 80Hz high-pass - remove rumble
                    "highpass=f=80",
                    # 150Hz -2dB - reduce muddiness
                    "equalizer=f=150:width_type=o:width=1:g=-2",
                    # 3kHz +3dB - voice presence
                    "equalizer=f=3000:width_type=o:width=1:g=3",
                    # 6.5kHz +2dB - clarity/air
                    "equalizer=f=6500:width_type=o:width=1:g=2",
                    # 11kHz +1.5dB - brilliance
                    "equalizer=f=11000:width_type=o:width=1.5:g=1.5",
                ]
                filters.extend(eq_filters)
                logger.debug("Added 6-band EQ to processing chain")

            # Stage 2: Broadcast compression (if enabled)
            if apply_compression:
                compression_filters = [
                    # Main compressor
                    "acompressor=threshold=-24dB:ratio=4:attack=10:release=100:makeup=2",
                    # Brick-wall limiter at -1dB
                    "alimiter=limit=0.891:attack=5:release=50:level=false",
                ]
                filters.extend(compression_filters)
                logger.debug("Added broadcast compression to processing chain")

            # Stage 3: Final loudness normalization
            filters.append(f"loudnorm=I={self.TARGET_LUFS}:TP={self.TARGET_PEAK}:LRA={self.TARGET_LRA}")
            logger.debug(f"Added loudness normalization ({self.TARGET_LUFS} LUFS)")

            filter_chain = ",".join(filters)

            cmd = [
                self.ffmpeg, '-y',
                '-i', input_file,
                '-af', filter_chain,
                '-ar', str(self.YOUTUBE_SAMPLE_RATE),
                '-b:a', '256k',
                output_file
            ]

            logger.debug(f"FFmpeg filter chain: {filter_chain}")
            result = subprocess.run(cmd, capture_output=True, timeout=300)

            if result.returncode == 0 and os.path.exists(output_file):
                logger.success(f"Broadcast-quality processing complete: {output_file}")
                return output_file

            error_msg = result.stderr.decode() if result.stderr else "Unknown error"
            logger.error(f"Broadcast-quality processing failed: {error_msg[:500]}")
            return None

        except subprocess.TimeoutExpired:
            logger.error("Broadcast-quality processing timed out (exceeded 5 minutes)")
            return None
        except (subprocess.SubprocessError, OSError) as e:
            logger.error(f"Broadcast-quality processing failed: {e}")
            return None

    def enhance_voice(
        self,
        input_file: str,
        output_file: str,
        remove_noise: bool = True,
        add_compression: bool = True,
        eq_boost: bool = True
    ) -> Optional[str]:
        """
        Enhance voice audio for better clarity and presence (legacy method).

        For professional broadcast-quality enhancement, use enhance_voice_professional() instead.

        Applies:
        - Noise reduction (highpass filter to remove rumble)
        - Compression (even out volume levels)
        - EQ boost (add presence and clarity)

        Args:
            input_file: Path to voice audio
            output_file: Path for enhanced output
            remove_noise: Apply highpass filter to remove low rumble
            add_compression: Apply dynamic compression
            eq_boost: Boost presence frequencies

        Returns:
            Path to enhanced audio or None on failure
        """
        if not self.ffmpeg or not os.path.exists(input_file):
            return None

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        logger.info("Enhancing voice audio...")

        try:
            filters = []

            # Noise reduction: highpass filter removes low rumble (below 80Hz)
            if remove_noise:
                filters.append("highpass=f=80")
                # Also add a lowpass to remove very high frequencies
                filters.append("lowpass=f=12000")

            # EQ boost for voice presence
            if eq_boost:
                # Boost 2-4kHz for presence, slight cut at 500Hz for less mud
                filters.append("equalizer=f=500:t=q:w=1:g=-2")
                filters.append("equalizer=f=3000:t=q:w=1.5:g=3")

            # Compression for even levels
            if add_compression:
                # Gentle compression: threshold -20dB, ratio 3:1, attack 20ms, release 100ms
                filters.append(
                    "acompressor=threshold=-20dB:ratio=3:attack=20:release=100:makeup=2"
                )

            if not filters:
                # No processing requested
                return self.normalize_audio(input_file, output_file)

            filter_str = ",".join(filters)

            cmd = [
                self.ffmpeg, '-y',
                '-i', input_file,
                '-af', filter_str,
                '-ar', str(self.YOUTUBE_SAMPLE_RATE),
                '-b:a', '256k',
                output_file
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=300)

            if result.returncode == 0 and os.path.exists(output_file):
                logger.info(f"Voice enhanced: {output_file}")
                return output_file

            return None

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as e:
            logger.error(f"Voice enhancement failed: {e}")
            return None

    def enhance_voice_professional(
        self,
        input_file: str,
        output_file: str,
        noise_reduction: bool = True,
        normalize_lufs: float = -14.0
    ) -> Optional[str]:
        """
        Professional broadcast-quality voice enhancement for YouTube.

        Applies a complete audio processing chain used in broadcast production:
        1. FFT-based noise reduction - removes background noise and hum
        2. High-pass filter - removes low-frequency rumble below 80Hz
        3. Presence EQ boost - enhances voice clarity in 2-4kHz range
        4. De-essing - reduces harsh sibilant 's' sounds
        5. Dynamic compression - smooths volume levels for consistent audio
        6. Loudness normalization - targets YouTube's -14 LUFS standard

        This is the recommended method for all YouTube voice content.

        Args:
            input_file: Path to input voice audio file
            output_file: Path for enhanced output file
            noise_reduction: Apply FFT-based noise reduction (default: True)
            normalize_lufs: Target loudness in LUFS (default: -14 for YouTube)

        Returns:
            Path to enhanced audio file or None on failure

        Example:
            processor = AudioProcessor()
            result = processor.enhance_voice_professional(
                "raw_voice.mp3",
                "enhanced_voice.mp3",
                noise_reduction=True,
                normalize_lufs=-14
            )
        """
        if not self.ffmpeg:
            logger.error("FFmpeg not available for professional voice enhancement")
            return None

        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return None

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Applying professional voice enhancement to: {input_file}")

        try:
            # Build the professional filter chain
            filters = []

            # 1. Noise reduction using FFT-based denoiser
            # nf=-20 sets noise floor at -20dB (removes quiet noise)
            if noise_reduction:
                filters.append("afftdn=nf=-20")
                logger.debug("Applied FFT noise reduction (nf=-20)")

            # 2. EQ for voice clarity - boost presence frequencies (2-4kHz)
            # This makes voice cut through and sound more professional
            # width_type=h means half-width in Hz, width=2000 gives gentle curve
            filters.append("equalizer=f=3000:width_type=h:width=2000:g=3")
            logger.debug("Applied presence EQ boost (+3dB at 3kHz)")

            # 3. High-pass filter - remove rumble below 80Hz
            # Voice fundamentals are above 80Hz, this removes room noise
            filters.append("highpass=f=80")
            logger.debug("Applied high-pass filter (80Hz)")

            # 4. De-essing - reduce harsh sibilant sounds
            # FFmpeg's deesser targets 4-8kHz sibilant frequencies
            filters.append("deesser")
            logger.debug("Applied de-esser")

            # 5. Compression - smooth dynamics for broadcast quality
            # threshold=-18dB: compress signals above -18dB
            # ratio=3: 3:1 compression ratio (moderate)
            # attack=5ms: fast attack catches transients
            # release=50ms: quick release maintains natural feel
            filters.append("acompressor=threshold=-18dB:ratio=3:attack=5:release=50")
            logger.debug("Applied compression (threshold=-18dB, ratio=3:1)")

            # 6. Final loudness normalization to YouTube target
            # I=-14: Integrated loudness target of -14 LUFS
            # TP=-1.5: True peak limit of -1.5dB (headroom for encoding)
            # LRA=11: Loudness range of 11 LU (natural dynamics)
            filters.append(f"loudnorm=I={normalize_lufs}:TP=-1.5:LRA=11")
            logger.debug(f"Applied loudness normalization ({normalize_lufs} LUFS)")

            filter_chain = ",".join(filters)

            # Run FFmpeg with the complete filter chain
            cmd = [
                self.ffmpeg, '-y',
                '-i', input_file,
                '-af', filter_chain,
                '-ar', str(self.YOUTUBE_SAMPLE_RATE),
                '-b:a', '256k',
                output_file
            ]

            logger.debug(f"FFmpeg command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, timeout=300)

            if result.returncode == 0 and os.path.exists(output_file):
                logger.success(f"Professional voice enhancement complete: {output_file}")
                return output_file

            # Log error details if failed
            error_msg = result.stderr.decode() if result.stderr else "Unknown error"
            logger.error(f"Professional enhancement failed: {error_msg[:500]}")
            return None

        except subprocess.TimeoutExpired:
            logger.error("Voice enhancement timed out (exceeded 5 minutes)")
            return None
        except (subprocess.SubprocessError, OSError) as e:
            logger.error(f"Voice enhancement failed: {e}")
            return None

    def process_for_youtube(
        self,
        input_file: str,
        output_file: str = None,
        music_file: str = None,
        enhance_voice: bool = False,
        professional_enhancement: bool = True,
        noise_reduction: bool = True,
        normalize_lufs: float = -14.0
    ) -> Optional[str]:
        """
        Complete audio processing pipeline for YouTube.

        Applies all necessary processing in optimal order:
        1. Professional voice enhancement (recommended) or legacy enhancement
        2. Background music mixing (if provided)
        3. Final loudness normalization

        Args:
            input_file: Path to voice/narration audio
            output_file: Path for processed output
            music_file: Optional background music to mix
            enhance_voice: Whether to apply voice enhancement (legacy method)
            professional_enhancement: Use professional broadcast-quality enhancement (default: True)
            noise_reduction: Apply FFT noise reduction in professional mode (default: True)
            normalize_lufs: Target loudness in LUFS (default: -14 for YouTube)

        Returns:
            Path to processed audio file or None on failure
        """
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return None

        # Generate output filename if not provided
        if not output_file:
            input_path = Path(input_file)
            output_file = str(input_path.with_stem(f"{input_path.stem}_youtube"))

        logger.info(f"Processing audio for YouTube: {input_file}")

        current_file = input_file
        temp_files = []

        try:
            # Step 1: Voice enhancement
            if professional_enhancement:
                # Use professional broadcast-quality enhancement
                enhanced_file = str(Path(output_file).with_stem("_temp_enhanced"))
                result = self.enhance_voice_professional(
                    current_file,
                    enhanced_file,
                    noise_reduction=noise_reduction,
                    normalize_lufs=normalize_lufs
                )
                if result:
                    temp_files.append(enhanced_file)
                    current_file = enhanced_file
                    logger.info("Applied professional voice enhancement")
            elif enhance_voice:
                # Legacy enhancement (for backwards compatibility)
                enhanced_file = str(Path(output_file).with_stem("_temp_enhanced"))
                result = self.enhance_voice(current_file, enhanced_file)
                if result:
                    temp_files.append(enhanced_file)
                    current_file = enhanced_file

            # Step 2: Mix with background music (if provided)
            if music_file and os.path.exists(music_file):
                mixed_file = str(Path(output_file).with_stem("_temp_mixed"))
                result = self.mix_with_background_music(
                    current_file,
                    music_file,
                    mixed_file,
                    normalize_before_mix=not (enhance_voice or professional_enhancement)
                )
                if result:
                    temp_files.append(mixed_file)
                    current_file = mixed_file
                    # Skip final normalization as mixing already normalizes
                    # Just copy to output
                    if current_file != output_file:
                        shutil.copy(current_file, output_file)
                    current_file = output_file
            else:
                # Step 3: Final normalization (if no music mixing and no enhancement was applied)
                if not professional_enhancement and not enhance_voice:
                    result = self.normalize_audio(current_file, output_file)
                    if result:
                        current_file = output_file
                else:
                    # Enhancement already normalized, just copy to output
                    if current_file != output_file:
                        shutil.copy(current_file, output_file)
                    current_file = output_file

            # Cleanup temp files
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file) and temp_file != output_file:
                        os.remove(temp_file)
                except OSError:
                    pass

            if os.path.exists(output_file):
                logger.success(f"Audio processed for YouTube: {output_file}")
                return output_file

            return None

        except Exception as e:
            logger.error(f"YouTube audio processing failed: {e}")
            # Cleanup on error
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except OSError:
                    pass
            return None


# Example usage
if __name__ == "__main__":
    print("\n" + "="*60)
    print("AUDIO PROCESSOR TEST")
    print("="*60 + "\n")

    processor = AudioProcessor()

    print(f"FFmpeg: {processor.ffmpeg}")
    print(f"FFprobe: {processor.ffprobe}")
    print(f"Target LUFS: {processor.TARGET_LUFS}")
    print(f"Target Peak: {processor.TARGET_PEAK} dBTP")
    print(f"Sample Rate: {processor.YOUTUBE_SAMPLE_RATE} Hz")

    # Test with a sample file if it exists
    test_file = "output/test_voice.mp3"
    if os.path.exists(test_file):
        print(f"\nAnalyzing: {test_file}")
        levels = processor.analyze_loudness(test_file)
        if levels:
            print(f"  Integrated Loudness: {levels.get('integrated_loudness', 'N/A')} LUFS")
            print(f"  True Peak: {levels.get('true_peak', 'N/A')} dBTP")
            print(f"  Loudness Range: {levels.get('loudness_range', 'N/A')} LU")

        print("\nNormalizing audio...")
        normalized = processor.normalize_audio(test_file, "output/test_voice_normalized.mp3")
        if normalized:
            print(f"  Output: {normalized}")
    else:
        print(f"\nNo test file found at {test_file}")
        print("Run TTS test first to generate a test audio file.")
