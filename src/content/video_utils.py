"""
Shared video encoding utilities for all video generators.

Consolidates common FFmpeg operations:
- Finding FFmpeg executable across systems
- Two-pass H.264 encoding for optimal quality
- Shared FFmpeg parameters
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional
from loguru import logger


# Shared FFmpeg parameters for different video types
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


def find_ffmpeg() -> Optional[str]:
    """
    Find FFmpeg executable on the system.

    Searches in:
    1. System PATH (via shutil.which)
    2. Common installation directories (Windows)
    3. WinGet package manager locations

    Returns:
        Path to ffmpeg executable or "ffmpeg" if found in PATH, None otherwise.
    """
    # Check system PATH first
    if shutil.which("ffmpeg"):
        return "ffmpeg"

    # Check common Windows locations
    common_paths = [
        os.path.expanduser("~\\AppData\\Local\\Microsoft\\WinGet\\Packages"),
        "C:\\ffmpeg\\bin\\ffmpeg.exe",
        "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe",
        os.path.expanduser("~\\ffmpeg\\bin\\ffmpeg.exe"),
    ]

    # Check direct paths first
    for path in common_paths[1:]:
        if os.path.exists(path):
            return path

    # Walk WinGet directory if it exists
    winget_path = common_paths[0]
    if os.path.isdir(winget_path):
        try:
            for root, dirs, files in os.walk(winget_path):
                if "ffmpeg.exe" in files:
                    full_path = os.path.join(root, "ffmpeg.exe")
                    return full_path
        except (OSError, PermissionError):
            pass

    # Not found
    logger.warning("FFmpeg not found on system. Install from https://ffmpeg.org/download.html")
    return None


def two_pass_encode(
    input_file: str,
    output_file: str,
    ffmpeg_path: str,
    encoding_preset: str,
    ffmpeg_params: List[str],
    target_bitrate: str = "8M",
    max_bitrate: str = "12M",
) -> Optional[str]:
    """
    Perform two-pass H.264 encoding for maximum quality at target bitrate.

    Two-pass encoding analyzes the video in pass 1, then encodes with optimal
    parameters in pass 2, achieving better quality per bitrate than single-pass.

    Args:
        input_file: Path to input video file
        output_file: Path to output video file
        ffmpeg_path: Path or command to ffmpeg executable
        encoding_preset: FFmpeg preset (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
        ffmpeg_params: List of additional FFmpeg parameters to apply
        target_bitrate: Target video bitrate (e.g., "8M" for 8 Mbps)
        max_bitrate: Maximum bitrate for VBV buffer (e.g., "12M" for 12 Mbps)

    Returns:
        Path to encoded output file if successful, None if encoding failed.
    """
    passlog = tempfile.mktemp(prefix="ffmpeg_pass_")

    try:
        # Pass 1: Analyze video to build rate control model
        pass1_cmd = [
            ffmpeg_path, "-y",
            "-i", input_file,
            "-c:v", "libx264",
            "-preset", encoding_preset,
            "-b:v", target_bitrate,
            "-maxrate", max_bitrate,
            "-bufsize", "16M",
            "-pass", "1",
            "-passlogfile", passlog,
            "-an",  # No audio in analysis pass
            "-f", "null",
            "NUL" if os.name == "nt" else "/dev/null"
        ]

        logger.info("Two-pass encoding: Pass 1 (analysis)...")
        subprocess.run(pass1_cmd, capture_output=True, timeout=600)

        # Pass 2: Encode with optimal parameters from pass 1
        pass2_cmd = [
            ffmpeg_path, "-y",
            "-i", input_file,
            "-c:v", "libx264",
            "-preset", encoding_preset,
            "-b:v", target_bitrate,
            "-maxrate", max_bitrate,
            "-bufsize", "16M",
            "-pass", "2",
            "-passlogfile", passlog,
            "-c:a", "aac",
            "-b:a", "256k",
        ] + ffmpeg_params + [output_file]

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
        # Cleanup passlog files generated by FFmpeg
        for ext in ["", "-0.log", "-0.log.mbtree"]:
            try:
                log_file = passlog + ext
                if os.path.exists(log_file):
                    os.remove(log_file)
            except Exception:
                pass
