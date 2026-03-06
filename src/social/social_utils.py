"""
Shared utility functions for social media distribution.

Consolidates:
- Video encoding/writing
- Video metadata reading
- HTTP error parsing
- Platform-agnostic helpers
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from contextlib import contextmanager
from loguru import logger


@dataclass
class VideoInfo:
    """Basic video metadata."""
    duration: float
    width: int
    height: int
    fps: float = 30.0


def parse_http_error(response: Any, platform_name: str) -> Dict[str, Any]:
    """
    Parse HTTP error response and return standard error dict.

    Args:
        response: HTTP response object (requests.Response)
        platform_name: Display name of platform

    Returns:
        Standard error response dict
    """
    error_text = ""
    try:
        if hasattr(response, "json"):
            error_data = response.json()
            error_text = str(error_data.get("error", error_data.get("message", str(error_data))))
        else:
            error_text = response.text
    except Exception:
        error_text = str(response)

    error = f"HTTP {response.status_code}: {error_text}"
    logger.error(f"[{platform_name}] Failed: {error}")

    return {
        "success": False,
        "error": error,
        "platform": platform_name.lower(),
        "status_code": getattr(response, "status_code", None),
    }


def write_video(
    clip: Any,
    output_path: str,
    bitrate: str = "8M",
    fps: int = 30,
    preset: str = "medium",
    threads: int = 4,
) -> bool:
    """
    Write a MoviePy video clip to file with standard YouTube-friendly settings.

    Args:
        clip: MoviePy VideoClip
        output_path: Path to write video to
        bitrate: Video bitrate (e.g., "8M", "6M")
        fps: Frames per second
        preset: FFmpeg preset (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
        threads: Number of FFmpeg threads

    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        logger.debug(f"Writing video: {output_path} ({bitrate}, {fps}fps, preset={preset})")

        clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            verbose=False,
            logger=None,  # Suppress MoviePy logging
            bitrate=bitrate,
            fps=fps,
            preset=preset,
            ffmpeg_params=["-threads", str(threads)],
        )

        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.success(f"Video written: {output_path} ({size_mb:.1f} MB)")
            return True
        else:
            logger.error(f"Video write failed - file not created: {output_path}")
            return False

    except Exception as e:
        logger.error(f"Error writing video {output_path}: {str(e)}")
        return False


@contextmanager
def get_video_info(video_path: str):
    """
    Context manager to read video info and automatically close the clip.

    Args:
        video_path: Path to video file

    Yields:
        VideoInfo dataclass with duration, width, height, fps

    Example:
        with get_video_info("video.mp4") as info:
            print(f"Duration: {info.duration}s")
    """
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        logger.error("MoviePy not installed - cannot read video info")
        yield VideoInfo(duration=0, width=0, height=0)
        return

    clip = None
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        width = clip.w
        height = clip.h
        fps = clip.fps if hasattr(clip, "fps") else 30.0

        yield VideoInfo(duration=duration, width=width, height=height, fps=fps)

    except Exception as e:
        logger.error(f"Error reading video info from {video_path}: {str(e)}")
        yield VideoInfo(duration=0, width=0, height=0)

    finally:
        if clip:
            try:
                clip.close()
            except Exception:
                pass


def get_video_duration(video_path: str) -> float:
    """
    Get duration of a video file in seconds.

    Args:
        video_path: Path to video file

    Returns:
        Duration in seconds, or 0 if unable to determine
    """
    with get_video_info(video_path) as info:
        return info.duration


# Short-form platform constant - used by multi_platform.py
SHORT_FORM_PLATFORMS = [
    "youtube_shorts",
    "tiktok",
    "instagram_reels",
]

# Aspect ratios for short-form platforms
SHORT_FORM_ASPECT_RATIOS = {
    "youtube_shorts": (1080, 1920),  # 9:16
    "tiktok": (1080, 1920),  # 9:16
    "instagram_reels": (1080, 1920),  # 9:16
}

# Min/max duration constraints for short-form
SHORT_FORM_DURATION = {
    "youtube_shorts": (15, 60),  # 15-60 seconds
    "tiktok": (3, 600),  # 3-600 seconds but optimal 15-60
    "instagram_reels": (3, 90),  # 3-90 seconds but optimal 15-60
}
