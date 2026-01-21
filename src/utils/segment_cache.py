"""
Video Segment Cache Module

Caches reusable video segments (intros, outros, transitions) to avoid
regenerating them for every video. This can save 30-60 seconds per video.

Features:
- Channel-specific intro/outro caching
- Transition segment caching
- Automatic expiration and cleanup
- Size-based cache management
- Thread-safe operations

Usage:
    from src.utils.segment_cache import SegmentCache

    cache = SegmentCache()

    # Get cached intro
    intro = cache.get_intro_segment("money_blueprints")
    if intro:
        print(f"Using cached intro: {intro}")
    else:
        # Generate intro and cache it
        intro_path = generate_intro(...)
        cache.cache_segment("intro", "money_blueprints", intro_path)

    # Get cached outro
    outro = cache.get_outro_segment("money_blueprints")

    # Cache a transition
    cache.cache_segment("transition_fade", "global", fade_path)
"""

import os
import json
import shutil
import hashlib
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from loguru import logger


# Default configuration
DEFAULT_CACHE_DIR = Path("data/segment_cache")
CACHE_METADATA_FILE = "segment_metadata.json"
DEFAULT_EXPIRY_DAYS = 30
MAX_CACHE_SIZE_MB = 1000  # 1 GB max cache


@dataclass
class SegmentEntry:
    """Represents a cached video segment."""
    segment_type: str  # intro, outro, transition, etc.
    channel_id: str
    file_path: str
    file_size: int
    duration: float
    resolution: str
    created_at: str
    last_accessed: str
    access_count: int
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "SegmentEntry":
        return cls(**data)


@dataclass
class SegmentCacheStats:
    """Cache statistics."""
    total_segments: int
    total_size_mb: float
    segments_by_type: Dict[str, int]
    segments_by_channel: Dict[str, int]
    total_hits: int
    oldest_segment_days: int
    newest_segment_days: int

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            "=" * 60,
            "  VIDEO SEGMENT CACHE STATISTICS",
            "=" * 60,
            f"  Total cached segments: {self.total_segments}",
            f"  Total cache size: {self.total_size_mb:.1f} MB",
            f"  Total cache hits: {self.total_hits}",
            "",
            "  Segments by type:",
        ]
        for seg_type, count in self.segments_by_type.items():
            lines.append(f"    - {seg_type}: {count}")

        lines.extend([
            "",
            "  Segments by channel:",
        ])
        for channel, count in self.segments_by_channel.items():
            lines.append(f"    - {channel}: {count}")

        lines.extend([
            "",
            f"  Oldest segment: {self.oldest_segment_days} days",
            f"  Newest segment: {self.newest_segment_days} days",
            "=" * 60,
        ])
        return "\n".join(lines)


class SegmentCache:
    """
    Video segment cache manager.

    Caches reusable video segments like intros, outros, and transitions
    to avoid regenerating them for every video.
    """

    # Supported segment types
    SEGMENT_TYPES = [
        "intro",
        "outro",
        "transition_fade",
        "transition_wipe",
        "transition_dissolve",
        "lower_third",
        "subscribe_cta",
        "end_screen",
        "background_loop"
    ]

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        expiry_days: int = DEFAULT_EXPIRY_DAYS,
        max_size_mb: int = MAX_CACHE_SIZE_MB
    ):
        """
        Initialize the segment cache.

        Args:
            cache_dir: Directory to store cached segments
            expiry_days: Days before segments expire
            max_size_mb: Maximum cache size in MB
        """
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.expiry_days = expiry_days
        self.max_size_mb = max_size_mb
        self.metadata_file = self.cache_dir / CACHE_METADATA_FILE
        self._lock = threading.Lock()

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load metadata
        self.metadata: Dict[str, SegmentEntry] = self._load_metadata()

        logger.info(f"SegmentCache initialized: {self.cache_dir} ({len(self.metadata)} segments)")

    def _get_cache_key(self, segment_type: str, channel_id: str) -> str:
        """Generate unique cache key."""
        return f"{channel_id}_{segment_type}"

    def _load_metadata(self) -> Dict[str, SegmentEntry]:
        """Load cache metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return {
                    key: SegmentEntry.from_dict(entry)
                    for key, entry in data.items()
                }
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"Failed to load segment cache metadata: {e}")
        return {}

    def _save_metadata(self) -> None:
        """Save cache metadata to file."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                data = {key: entry.to_dict() for key, entry in self.metadata.items()}
                json.dump(data, f, indent=2)
        except (OSError, IOError) as e:
            logger.error(f"Failed to save segment cache metadata: {e}")

    def get_intro_segment(self, channel_id: str) -> Optional[str]:
        """
        Get cached intro segment for a channel.

        Args:
            channel_id: Channel identifier

        Returns:
            Path to cached intro or None if not found
        """
        return self._get_segment("intro", channel_id)

    def get_outro_segment(self, channel_id: str) -> Optional[str]:
        """
        Get cached outro segment for a channel.

        Args:
            channel_id: Channel identifier

        Returns:
            Path to cached outro or None if not found
        """
        return self._get_segment("outro", channel_id)

    def get_transition_segment(self, transition_type: str, channel_id: str = "global") -> Optional[str]:
        """
        Get cached transition segment.

        Args:
            transition_type: Type of transition (fade, wipe, dissolve)
            channel_id: Channel ID or "global" for shared transitions

        Returns:
            Path to cached transition or None
        """
        return self._get_segment(f"transition_{transition_type}", channel_id)

    def _get_segment(self, segment_type: str, channel_id: str) -> Optional[str]:
        """
        Get a cached segment.

        Args:
            segment_type: Type of segment
            channel_id: Channel identifier

        Returns:
            Path to cached segment or None
        """
        cache_key = self._get_cache_key(segment_type, channel_id)
        cache_file = self.cache_dir / f"{cache_key}.mp4"

        with self._lock:
            # Check if file exists
            if not cache_file.exists():
                return None

            # Check expiry
            if cache_key in self.metadata:
                entry = self.metadata[cache_key]
                try:
                    created = datetime.fromisoformat(entry.created_at)
                    age_days = (datetime.now() - created).days
                    if age_days > self.expiry_days:
                        logger.debug(f"Segment expired ({age_days} days): {cache_key}")
                        self._remove_segment(cache_key)
                        return None
                except (ValueError, TypeError):
                    pass

                # Update access stats
                entry.last_accessed = datetime.now().isoformat()
                entry.access_count += 1
                self._save_metadata()

            logger.debug(f"Segment cache hit: {cache_key}")
            return str(cache_file)

    def cache_segment(
        self,
        segment_type: str,
        channel_id: str,
        video_path: str,
        duration: float = 0.0,
        resolution: str = "1920x1080",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Cache a video segment.

        Args:
            segment_type: Type of segment (intro, outro, transition_*, etc.)
            channel_id: Channel identifier or "global"
            video_path: Path to video file to cache
            duration: Segment duration in seconds
            resolution: Video resolution
            metadata: Additional metadata

        Returns:
            Path to cached file or None on failure
        """
        if not os.path.exists(video_path):
            logger.error(f"Cannot cache non-existent file: {video_path}")
            return None

        cache_key = self._get_cache_key(segment_type, channel_id)
        cache_file = self.cache_dir / f"{cache_key}.mp4"

        with self._lock:
            try:
                # Check cache size before adding
                self._enforce_cache_limit()

                # Copy file to cache
                shutil.copy2(video_path, cache_file)
                file_size = os.path.getsize(cache_file)

                # Create metadata entry
                now = datetime.now().isoformat()
                entry = SegmentEntry(
                    segment_type=segment_type,
                    channel_id=channel_id,
                    file_path=str(cache_file),
                    file_size=file_size,
                    duration=duration,
                    resolution=resolution,
                    created_at=now,
                    last_accessed=now,
                    access_count=0,
                    metadata=metadata or {}
                )

                self.metadata[cache_key] = entry
                self._save_metadata()

                logger.info(f"Cached segment: {cache_key} ({file_size / 1024:.1f} KB)")
                return str(cache_file)

            except (OSError, IOError, shutil.Error) as e:
                logger.error(f"Failed to cache segment: {e}")
                return None

    def _remove_segment(self, cache_key: str) -> None:
        """Remove a segment from cache."""
        cache_file = self.cache_dir / f"{cache_key}.mp4"

        # Remove file
        if cache_file.exists():
            try:
                cache_file.unlink()
            except OSError:
                pass

        # Remove metadata
        if cache_key in self.metadata:
            del self.metadata[cache_key]
            self._save_metadata()

    def _enforce_cache_limit(self) -> None:
        """Remove old segments if cache exceeds size limit."""
        total_size = sum(entry.file_size for entry in self.metadata.values())
        max_bytes = self.max_size_mb * 1024 * 1024

        if total_size <= max_bytes:
            return

        # Sort by last accessed (oldest first)
        sorted_entries = sorted(
            self.metadata.items(),
            key=lambda x: x[1].last_accessed
        )

        # Remove oldest until under limit
        for cache_key, entry in sorted_entries:
            if total_size <= max_bytes:
                break

            total_size -= entry.file_size
            self._remove_segment(cache_key)
            logger.debug(f"Removed old segment to enforce limit: {cache_key}")

    def invalidate_channel_segments(self, channel_id: str) -> int:
        """
        Invalidate all cached segments for a channel.

        Useful when channel branding changes.

        Args:
            channel_id: Channel identifier

        Returns:
            Number of segments removed
        """
        removed = 0
        keys_to_remove = []

        with self._lock:
            for cache_key, entry in self.metadata.items():
                if entry.channel_id == channel_id:
                    keys_to_remove.append(cache_key)

            for cache_key in keys_to_remove:
                self._remove_segment(cache_key)
                removed += 1

        logger.info(f"Invalidated {removed} segments for channel: {channel_id}")
        return removed

    def cleanup_expired(self) -> Tuple[int, int]:
        """
        Remove expired segments.

        Returns:
            Tuple of (segments_removed, bytes_freed)
        """
        removed = 0
        bytes_freed = 0
        cutoff = datetime.now() - timedelta(days=self.expiry_days)

        with self._lock:
            keys_to_remove = []

            for cache_key, entry in self.metadata.items():
                try:
                    created = datetime.fromisoformat(entry.created_at)
                    if created < cutoff:
                        keys_to_remove.append((cache_key, entry.file_size))
                except (ValueError, TypeError):
                    keys_to_remove.append((cache_key, entry.file_size))

            for cache_key, file_size in keys_to_remove:
                self._remove_segment(cache_key)
                removed += 1
                bytes_freed += file_size

        logger.info(f"Cleanup: removed {removed} expired segments, freed {bytes_freed / 1024 / 1024:.1f} MB")
        return removed, bytes_freed

    def get_stats(self) -> SegmentCacheStats:
        """Get cache statistics."""
        total_segments = 0
        total_size = 0
        total_hits = 0
        oldest_days = 0
        newest_days = float('inf')
        by_type: Dict[str, int] = {}
        by_channel: Dict[str, int] = {}

        now = datetime.now()

        with self._lock:
            for entry in self.metadata.values():
                total_segments += 1
                total_size += entry.file_size
                total_hits += entry.access_count

                # Count by type
                by_type[entry.segment_type] = by_type.get(entry.segment_type, 0) + 1

                # Count by channel
                by_channel[entry.channel_id] = by_channel.get(entry.channel_id, 0) + 1

                # Calculate age
                try:
                    created = datetime.fromisoformat(entry.created_at)
                    age = (now - created).days
                    oldest_days = max(oldest_days, age)
                    newest_days = min(newest_days, age)
                except (ValueError, TypeError):
                    pass

        if newest_days == float('inf'):
            newest_days = 0

        return SegmentCacheStats(
            total_segments=total_segments,
            total_size_mb=total_size / 1024 / 1024,
            segments_by_type=by_type,
            segments_by_channel=by_channel,
            total_hits=total_hits,
            oldest_segment_days=oldest_days,
            newest_segment_days=newest_days
        )

    def clear_cache(self) -> Tuple[int, int]:
        """
        Clear all cached segments.

        Returns:
            Tuple of (segments_removed, bytes_freed)
        """
        removed = 0
        bytes_freed = 0

        with self._lock:
            for file_path in self.cache_dir.glob("*.mp4"):
                try:
                    bytes_freed += file_path.stat().st_size
                    file_path.unlink()
                    removed += 1
                except OSError:
                    pass

            self.metadata.clear()
            self._save_metadata()

        logger.info(f"Cache cleared: {removed} segments, {bytes_freed / 1024 / 1024:.1f} MB freed")
        return removed, bytes_freed

    def pregenerate_channel_segments(
        self,
        channel_id: str,
        intro_generator: Optional[callable] = None,
        outro_generator: Optional[callable] = None
    ) -> Dict[str, Optional[str]]:
        """
        Pre-generate and cache intro/outro for a channel.

        Args:
            channel_id: Channel identifier
            intro_generator: Function to generate intro video
            outro_generator: Function to generate outro video

        Returns:
            Dict with paths to cached intro and outro
        """
        results = {"intro": None, "outro": None}

        # Check/generate intro
        intro = self.get_intro_segment(channel_id)
        if intro:
            results["intro"] = intro
        elif intro_generator:
            try:
                intro_path = intro_generator(channel_id)
                if intro_path and os.path.exists(intro_path):
                    results["intro"] = self.cache_segment("intro", channel_id, intro_path)
            except Exception as e:
                logger.error(f"Failed to generate intro for {channel_id}: {e}")

        # Check/generate outro
        outro = self.get_outro_segment(channel_id)
        if outro:
            results["outro"] = outro
        elif outro_generator:
            try:
                outro_path = outro_generator(channel_id)
                if outro_path and os.path.exists(outro_path):
                    results["outro"] = self.cache_segment("outro", channel_id, outro_path)
            except Exception as e:
                logger.error(f"Failed to generate outro for {channel_id}: {e}")

        return results


def print_segment_cache_stats():
    """Print segment cache statistics to console."""
    cache = SegmentCache()
    stats = cache.get_stats()
    print(stats.summary())


# CLI entry point
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "stats":
            print_segment_cache_stats()
        elif cmd == "cleanup":
            cache = SegmentCache()
            removed, freed = cache.cleanup_expired()
            print(f"Cleanup complete: {removed} segments, {freed / 1024 / 1024:.1f} MB freed")
        elif cmd == "clear":
            cache = SegmentCache()
            removed, freed = cache.clear_cache()
            print(f"Cache cleared: {removed} segments, {freed / 1024 / 1024:.1f} MB freed")
        elif cmd == "invalidate" and len(sys.argv) > 2:
            channel_id = sys.argv[2]
            cache = SegmentCache()
            removed = cache.invalidate_channel_segments(channel_id)
            print(f"Invalidated {removed} segments for channel: {channel_id}")
        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python segment_cache.py [stats|cleanup|clear|invalidate <channel_id>]")
    else:
        print_segment_cache_stats()
