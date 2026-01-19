"""
Stock Footage Cache Module

Caches downloaded stock footage to save 80% download time on similar topics.
Uses query-based caching with clip ID to avoid re-downloading the same content.

Features:
- Query + clip ID based cache keys (MD5 hash)
- Configurable cache duration (default 30 days)
- Automatic cleanup of old cache files
- Cache statistics and management
- Thread-safe operations

Usage:
    from src.content.stock_cache import StockCache

    cache = StockCache()

    # Check for cached clip
    cached = cache.get_cached_clip("passive income", "pexels_12345")
    if cached:
        print(f"Using cached: {cached}")
    else:
        # Download and save to cache
        downloaded_path = download_clip(...)
        cache.save_to_cache("passive income", "pexels_12345", downloaded_path)

    # Get cache stats
    stats = cache.get_stats()
    print(f"Cache size: {stats['total_size_mb']:.1f} MB")

    # Cleanup old files
    cache.cleanup_old_files()
"""

import os
import json
import shutil
import hashlib
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
from loguru import logger


# Default cache configuration
CACHE_DIR = Path("data/stock_cache")
CACHE_METADATA_FILE = "cache_metadata.json"
CACHE_DURATION_DAYS = 30
MIN_VALID_FILE_SIZE = 10000  # 10KB minimum for valid video


@dataclass
class CacheEntry:
    """Represents a cached stock footage entry."""
    query: str
    clip_id: str
    cache_key: str
    file_path: str
    source: str  # "pexels" or "pixabay"
    duration: int  # seconds
    file_size: int  # bytes
    created_at: str  # ISO format
    last_accessed: str  # ISO format
    access_count: int

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "CacheEntry":
        return cls(**data)


@dataclass
class CacheStats:
    """Cache statistics."""
    total_files: int
    total_size_bytes: int
    total_size_mb: float
    oldest_file_days: int
    newest_file_days: int
    total_hits: int
    unique_queries: int
    clips_by_source: Dict[str, int]

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            "=" * 60,
            "  STOCK FOOTAGE CACHE STATISTICS",
            "=" * 60,
            f"  Total cached clips: {self.total_files}",
            f"  Total cache size: {self.total_size_mb:.1f} MB",
            f"  Unique queries: {self.unique_queries}",
            f"  Total cache hits: {self.total_hits}",
            "",
            "  Clips by source:",
        ]
        for source, count in self.clips_by_source.items():
            lines.append(f"    - {source}: {count}")

        lines.extend([
            "",
            f"  Oldest file: {self.oldest_file_days} days",
            f"  Newest file: {self.newest_file_days} days",
            "=" * 60,
        ])
        return "\n".join(lines)


class StockCache:
    """
    Stock footage cache manager.

    Caches downloaded clips by query + clip_id to avoid re-downloading
    the same content for similar topics.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        cache_duration_days: int = CACHE_DURATION_DAYS
    ):
        """
        Initialize the stock cache.

        Args:
            cache_dir: Custom cache directory (default: data/stock_cache)
            cache_duration_days: Days to keep cached files (default: 30)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.cache_duration_days = cache_duration_days
        self.metadata_file = self.cache_dir / CACHE_METADATA_FILE
        self._lock = threading.Lock()

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load or initialize metadata
        self.metadata: Dict[str, CacheEntry] = self._load_metadata()

        logger.info(f"StockCache initialized: {self.cache_dir} ({len(self.metadata)} entries)")

    def _load_metadata(self) -> Dict[str, CacheEntry]:
        """Load cache metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return {
                    key: CacheEntry.from_dict(entry)
                    for key, entry in data.items()
                }
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        return {}

    def _save_metadata(self) -> None:
        """Save cache metadata to file."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                data = {key: entry.to_dict() for key, entry in self.metadata.items()}
                json.dump(data, f, indent=2)
        except (OSError, IOError) as e:
            logger.error(f"Failed to save cache metadata: {e}")

    @staticmethod
    def get_cache_key(query: str, clip_id: str) -> str:
        """
        Generate a unique cache key from query and clip ID.

        Args:
            query: Search query used to find the clip
            clip_id: Unique clip identifier (e.g., "pexels_12345")

        Returns:
            MD5 hash of query + clip_id
        """
        combined = f"{query.lower().strip()}_{clip_id}"
        return hashlib.md5(combined.encode()).hexdigest()

    def get_cached_clip(
        self,
        query: str,
        clip_id: str,
        check_expiry: bool = True
    ) -> Optional[Path]:
        """
        Get a cached clip if it exists and is valid.

        Args:
            query: Original search query
            clip_id: Clip identifier
            check_expiry: Whether to check cache expiration

        Returns:
            Path to cached file or None if not found/expired
        """
        cache_key = self.get_cache_key(query, clip_id)
        cache_file = self.cache_dir / f"{cache_key}.mp4"

        with self._lock:
            # Check if file exists
            if not cache_file.exists():
                return None

            # Check file size (must be valid video)
            file_size = cache_file.stat().st_size
            if file_size < MIN_VALID_FILE_SIZE:
                logger.warning(f"Cached file too small ({file_size} bytes), removing: {cache_key}")
                self._remove_cache_entry(cache_key)
                return None

            # Check expiry
            if check_expiry:
                age_days = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days
                if age_days > self.cache_duration_days:
                    logger.debug(f"Cache expired ({age_days} days old): {cache_key}")
                    self._remove_cache_entry(cache_key)
                    return None

            # Update access stats
            if cache_key in self.metadata:
                entry = self.metadata[cache_key]
                entry.last_accessed = datetime.now().isoformat()
                entry.access_count += 1
                self._save_metadata()

            logger.debug(f"Cache hit: {clip_id} (query: {query[:30]}...)")
            return cache_file

    def save_to_cache(
        self,
        query: str,
        clip_id: str,
        video_path: str,
        source: str = "unknown",
        duration: int = 0
    ) -> Optional[Path]:
        """
        Save a downloaded clip to the cache.

        Args:
            query: Search query used
            clip_id: Clip identifier
            video_path: Path to downloaded video file
            source: Video source ("pexels", "pixabay", etc.)
            duration: Video duration in seconds

        Returns:
            Path to cached file or None on failure
        """
        if not os.path.exists(video_path):
            logger.error(f"Cannot cache non-existent file: {video_path}")
            return None

        file_size = os.path.getsize(video_path)
        if file_size < MIN_VALID_FILE_SIZE:
            logger.warning(f"File too small to cache ({file_size} bytes): {video_path}")
            return None

        cache_key = self.get_cache_key(query, clip_id)
        cache_file = self.cache_dir / f"{cache_key}.mp4"

        with self._lock:
            try:
                # Copy file to cache
                shutil.copy2(video_path, cache_file)

                # Create metadata entry
                now = datetime.now().isoformat()
                entry = CacheEntry(
                    query=query,
                    clip_id=clip_id,
                    cache_key=cache_key,
                    file_path=str(cache_file),
                    source=source,
                    duration=duration,
                    file_size=file_size,
                    created_at=now,
                    last_accessed=now,
                    access_count=0
                )

                self.metadata[cache_key] = entry
                self._save_metadata()

                logger.info(f"Cached: {clip_id} ({file_size / 1024:.1f} KB)")
                return cache_file

            except (OSError, IOError, shutil.Error) as e:
                logger.error(f"Failed to cache file: {e}")
                return None

    def _remove_cache_entry(self, cache_key: str) -> None:
        """Remove a cache entry and its file."""
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

    def cleanup_old_files(self, max_age_days: Optional[int] = None) -> Tuple[int, int]:
        """
        Remove cached files older than max_age_days.

        Args:
            max_age_days: Maximum age in days (default: cache_duration_days)

        Returns:
            Tuple of (files_removed, bytes_freed)
        """
        max_age = max_age_days if max_age_days is not None else self.cache_duration_days
        cutoff_date = datetime.now() - timedelta(days=max_age)

        files_removed = 0
        bytes_freed = 0

        with self._lock:
            # Find expired entries
            expired_keys = []

            for cache_key, entry in self.metadata.items():
                try:
                    created = datetime.fromisoformat(entry.created_at)
                    if created < cutoff_date:
                        expired_keys.append(cache_key)
                except (ValueError, TypeError):
                    # Invalid date, remove it
                    expired_keys.append(cache_key)

            # Also check for orphaned files (files without metadata)
            for file_path in self.cache_dir.glob("*.mp4"):
                if file_path.name == CACHE_METADATA_FILE:
                    continue

                cache_key = file_path.stem
                if cache_key not in self.metadata:
                    # Check file age
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if mtime < cutoff_date:
                        try:
                            bytes_freed += file_path.stat().st_size
                            file_path.unlink()
                            files_removed += 1
                            logger.debug(f"Removed orphaned cache file: {file_path.name}")
                        except OSError:
                            pass

            # Remove expired entries
            for cache_key in expired_keys:
                cache_file = self.cache_dir / f"{cache_key}.mp4"
                if cache_file.exists():
                    try:
                        bytes_freed += cache_file.stat().st_size
                        cache_file.unlink()
                        files_removed += 1
                    except OSError:
                        pass

                if cache_key in self.metadata:
                    del self.metadata[cache_key]

            if expired_keys:
                self._save_metadata()

        logger.info(f"Cache cleanup: removed {files_removed} files, freed {bytes_freed / 1024 / 1024:.1f} MB")
        return files_removed, bytes_freed

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        total_files = 0
        total_size = 0
        total_hits = 0
        oldest_days = 0
        newest_days = float('inf')
        unique_queries = set()
        clips_by_source: Dict[str, int] = {}

        now = datetime.now()

        with self._lock:
            for entry in self.metadata.values():
                total_files += 1
                total_size += entry.file_size
                total_hits += entry.access_count
                unique_queries.add(entry.query.lower().strip())

                # Count by source
                source = entry.source
                clips_by_source[source] = clips_by_source.get(source, 0) + 1

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

        return CacheStats(
            total_files=total_files,
            total_size_bytes=total_size,
            total_size_mb=total_size / 1024 / 1024,
            oldest_file_days=oldest_days,
            newest_file_days=newest_days,
            total_hits=total_hits,
            unique_queries=len(unique_queries),
            clips_by_source=clips_by_source
        )

    def clear_cache(self) -> Tuple[int, int]:
        """
        Clear all cached files.

        Returns:
            Tuple of (files_removed, bytes_freed)
        """
        files_removed = 0
        bytes_freed = 0

        with self._lock:
            for file_path in self.cache_dir.glob("*.mp4"):
                try:
                    bytes_freed += file_path.stat().st_size
                    file_path.unlink()
                    files_removed += 1
                except OSError:
                    pass

            self.metadata.clear()
            self._save_metadata()

        logger.info(f"Cache cleared: removed {files_removed} files, freed {bytes_freed / 1024 / 1024:.1f} MB")
        return files_removed, bytes_freed

    def get_cached_clips_for_query(self, query: str) -> List[CacheEntry]:
        """
        Get all cached clips matching a query (fuzzy match).

        Args:
            query: Search query

        Returns:
            List of matching cache entries
        """
        query_lower = query.lower().strip()
        matches = []

        with self._lock:
            for entry in self.metadata.values():
                if query_lower in entry.query.lower():
                    matches.append(entry)

        return matches

    def estimate_savings(self) -> Dict[str, float]:
        """
        Estimate download time savings from cache usage.

        Returns:
            Dict with savings statistics
        """
        stats = self.get_stats()

        # Assume average download speed of 5 MB/s
        avg_download_speed_mbps = 5.0

        # Total MB downloaded from cache
        total_cached_mb = stats.total_size_mb

        # Estimated time saved per cache hit (based on average clip size)
        avg_clip_size_mb = total_cached_mb / max(stats.total_files, 1)
        time_per_clip_sec = avg_clip_size_mb / avg_download_speed_mbps

        total_time_saved_sec = stats.total_hits * time_per_clip_sec
        total_time_saved_min = total_time_saved_sec / 60

        return {
            "total_cache_hits": stats.total_hits,
            "total_cached_mb": total_cached_mb,
            "avg_clip_size_mb": avg_clip_size_mb,
            "estimated_time_saved_seconds": total_time_saved_sec,
            "estimated_time_saved_minutes": total_time_saved_min,
            "estimated_bandwidth_saved_mb": stats.total_hits * avg_clip_size_mb
        }


def print_cache_stats():
    """Print cache statistics to console."""
    cache = StockCache()
    stats = cache.get_stats()
    print(stats.summary())

    # Savings estimate
    savings = cache.estimate_savings()
    print("\n  Estimated Savings:")
    print(f"    Time saved: {savings['estimated_time_saved_minutes']:.1f} minutes")
    print(f"    Bandwidth saved: {savings['estimated_bandwidth_saved_mb']:.1f} MB")
    print()


def cleanup_cache(max_age_days: int = CACHE_DURATION_DAYS):
    """Run cache cleanup."""
    cache = StockCache()
    files_removed, bytes_freed = cache.cleanup_old_files(max_age_days)
    print(f"Cleanup complete: removed {files_removed} files, freed {bytes_freed / 1024 / 1024:.1f} MB")


# Quick test
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "stats":
            print_cache_stats()
        elif cmd == "cleanup":
            max_age = int(sys.argv[2]) if len(sys.argv) > 2 else CACHE_DURATION_DAYS
            cleanup_cache(max_age)
        elif cmd == "clear":
            cache = StockCache()
            files, bytes_freed = cache.clear_cache()
            print(f"Cache cleared: {files} files, {bytes_freed / 1024 / 1024:.1f} MB")
        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python stock_cache.py [stats|cleanup|clear]")
    else:
        print_cache_stats()
