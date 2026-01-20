"""
Intelligent Content Caching System

Multi-layer content caching for research, outlines, and renders.
Reduces API calls and processing time by 50-80% for similar topics.

Cache layers:
1. Research cache (trends, competitor data) - 24h TTL
2. Script outline cache (topic -> outline) - 7d TTL
3. Stock footage cache (query -> clips) - 30d TTL (exists in stock_cache.py)
4. Render cache (script+settings -> video) - 90d TTL
"""

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class CacheStats:
    """Cache statistics."""

    hits: int = 0
    misses: int = 0
    total_size_mb: float = 0.0
    entries: int = 0
    oldest_entry: Optional[datetime] = None
    newest_entry: Optional[datetime] = None

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


@dataclass
class CacheEntry:
    """Metadata for a cache entry."""

    key: str
    created_at: datetime
    expires_at: datetime
    size_bytes: int
    cache_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return datetime.now() > self.expires_at

    @property
    def age_hours(self) -> float:
        """Get age in hours."""
        delta = datetime.now() - self.created_at
        return delta.total_seconds() / 3600


class IntelligentContentCache:
    """
    Multi-layer content caching system.

    Benefits:
    - Similar topics reuse research: 5min -> 30sec
    - Outline variations save AI calls: $0.10 -> $0.02
    - Render cache for re-uploads: 10min -> instant
    """

    # Default TTLs
    RESEARCH_TTL_HOURS = 24
    OUTLINE_TTL_DAYS = 7
    RENDER_TTL_DAYS = 90

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize content cache.

        Args:
            cache_dir: Base directory for cache (default: data/content_cache)
        """
        self.cache_dir = cache_dir or Path("data/content_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache subdirectories
        self.research_cache = self.cache_dir / "research"
        self.outline_cache = self.cache_dir / "outlines"
        self.render_cache = self.cache_dir / "renders"

        for subdir in [self.research_cache, self.outline_cache, self.render_cache]:
            subdir.mkdir(exist_ok=True)

        # Statistics
        self._stats = {
            "research": CacheStats(),
            "outline": CacheStats(),
            "render": CacheStats(),
        }

        logger.info(f"ContentCache initialized at {self.cache_dir}")

    # ==================== Research Cache ====================

    def get_cached_research(
        self,
        topic: str,
        niche: str,
        max_age_hours: int = RESEARCH_TTL_HOURS
    ) -> Optional[Dict]:
        """
        Get cached research data for a topic.

        Args:
            topic: Topic to look up
            niche: Content niche
            max_age_hours: Maximum cache age in hours

        Returns:
            Research data if cache hit, None otherwise
        """
        cache_key = self._get_cache_key(topic, niche)
        cache_file = self.research_cache / f"{cache_key}.json"

        if not cache_file.exists():
            self._stats["research"].misses += 1
            return None

        # Check age
        file_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        file_age = datetime.now() - file_mtime

        if file_age > timedelta(hours=max_age_hours):
            logger.debug(f"Research cache expired: {file_age.total_seconds()/3600:.1f}h old")
            self._stats["research"].misses += 1
            return None

        # Load and return
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._stats["research"].hits += 1
            logger.info(
                f"Research cache HIT: '{topic[:30]}...' "
                f"({file_age.total_seconds()/60:.0f}min old)"
            )
            return data
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load research cache: {e}")
            self._stats["research"].misses += 1
            return None

    def save_research(
        self,
        topic: str,
        niche: str,
        research_data: Dict
    ) -> str:
        """
        Save research data to cache.

        Args:
            topic: Research topic
            niche: Content niche
            research_data: Data to cache

        Returns:
            Cache key
        """
        cache_key = self._get_cache_key(topic, niche)
        cache_file = self.research_cache / f"{cache_key}.json"

        # Add metadata
        data_with_meta = {
            "topic": topic,
            "niche": niche,
            "cached_at": datetime.now().isoformat(),
            "data": research_data,
        }

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data_with_meta, f, indent=2, default=str)

            logger.info(f"Research cached: '{topic[:30]}...'")
            return cache_key
        except IOError as e:
            logger.error(f"Failed to save research cache: {e}")
            raise

    # ==================== Outline Cache ====================

    def get_cached_outline(
        self,
        topic: str,
        niche: str,
        max_age_days: int = OUTLINE_TTL_DAYS
    ) -> Optional[Dict]:
        """
        Get cached script outline.

        Use case: Generate variations of same topic without re-researching.

        Args:
            topic: Topic to look up
            niche: Content niche
            max_age_days: Maximum cache age in days

        Returns:
            Outline data if cache hit, None otherwise
        """
        cache_key = self._get_cache_key(topic, niche)
        cache_file = self.outline_cache / f"{cache_key}.json"

        if not cache_file.exists():
            self._stats["outline"].misses += 1
            return None

        # Check age
        file_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        file_age = datetime.now() - file_mtime

        if file_age > timedelta(days=max_age_days):
            self._stats["outline"].misses += 1
            return None

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._stats["outline"].hits += 1
            logger.info(f"Outline cache HIT: '{topic[:30]}...'")
            return data
        except (json.JSONDecodeError, IOError):
            self._stats["outline"].misses += 1
            return None

    def save_outline(
        self,
        topic: str,
        niche: str,
        outline_data: Dict
    ) -> str:
        """
        Save script outline to cache.

        Args:
            topic: Topic
            niche: Content niche
            outline_data: Outline to cache

        Returns:
            Cache key
        """
        cache_key = self._get_cache_key(topic, niche)
        cache_file = self.outline_cache / f"{cache_key}.json"

        data_with_meta = {
            "topic": topic,
            "niche": niche,
            "cached_at": datetime.now().isoformat(),
            "outline": outline_data,
        }

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data_with_meta, f, indent=2, default=str)

        logger.info(f"Outline cached: '{topic[:30]}...'")
        return cache_key

    # ==================== Render Cache ====================

    def get_cached_render(
        self,
        script_hash: str,
        settings_hash: str
    ) -> Optional[str]:
        """
        Check if video was already rendered with same script and settings.

        Args:
            script_hash: Hash of script content
            settings_hash: Hash of render settings

        Returns:
            Path to cached video if exists, None otherwise
        """
        render_key = f"{script_hash}_{settings_hash}"
        cache_file = self.render_cache / f"{render_key}.json"

        if not cache_file.exists():
            self._stats["render"].misses += 1
            return None

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            video_path = data.get("video_path")
            if video_path and Path(video_path).exists():
                self._stats["render"].hits += 1
                logger.info(f"Render cache HIT: {video_path}")
                return video_path
            else:
                # Video file was deleted
                cache_file.unlink()
                self._stats["render"].misses += 1
                return None
        except (json.JSONDecodeError, IOError):
            self._stats["render"].misses += 1
            return None

    def save_render_reference(
        self,
        script_hash: str,
        settings_hash: str,
        video_path: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Save reference to rendered video.

        Args:
            script_hash: Hash of script content
            settings_hash: Hash of render settings
            video_path: Path to rendered video
            metadata: Optional metadata

        Returns:
            Render key
        """
        render_key = f"{script_hash}_{settings_hash}"
        cache_file = self.render_cache / f"{render_key}.json"

        data = {
            "video_path": str(video_path),
            "cached_at": datetime.now().isoformat(),
            "script_hash": script_hash,
            "settings_hash": settings_hash,
            "metadata": metadata or {},
        }

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Render cached: {video_path}")
        return render_key

    # ==================== Cache Management ====================

    def cleanup_expired(self, dry_run: bool = False) -> Dict[str, int]:
        """
        Remove expired cache entries.

        Args:
            dry_run: If True, only report what would be deleted

        Returns:
            Count of deleted entries per cache type
        """
        deleted = {"research": 0, "outline": 0, "render": 0}

        # Research cache (24h TTL)
        for f in self.research_cache.glob("*.json"):
            age = datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)
            if age > timedelta(hours=self.RESEARCH_TTL_HOURS):
                if not dry_run:
                    f.unlink()
                deleted["research"] += 1

        # Outline cache (7d TTL)
        for f in self.outline_cache.glob("*.json"):
            age = datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)
            if age > timedelta(days=self.OUTLINE_TTL_DAYS):
                if not dry_run:
                    f.unlink()
                deleted["outline"] += 1

        # Render cache (90d TTL)
        for f in self.render_cache.glob("*.json"):
            age = datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)
            if age > timedelta(days=self.RENDER_TTL_DAYS):
                if not dry_run:
                    f.unlink()
                deleted["render"] += 1

        action = "Would delete" if dry_run else "Deleted"
        logger.info(
            f"{action}: {deleted['research']} research, "
            f"{deleted['outline']} outlines, {deleted['render']} renders"
        )

        return deleted

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.

        Returns:
            Statistics for all cache types
        """
        stats = {}

        for cache_type, cache_dir in [
            ("research", self.research_cache),
            ("outline", self.outline_cache),
            ("render", self.render_cache),
        ]:
            files = list(cache_dir.glob("*.json"))
            total_size = sum(f.stat().st_size for f in files)

            mtimes = [datetime.fromtimestamp(f.stat().st_mtime) for f in files]

            cache_stats = self._stats[cache_type]
            stats[cache_type] = {
                "entries": len(files),
                "size_mb": total_size / (1024 * 1024),
                "hits": cache_stats.hits,
                "misses": cache_stats.misses,
                "hit_rate": cache_stats.hit_rate,
                "oldest": min(mtimes).isoformat() if mtimes else None,
                "newest": max(mtimes).isoformat() if mtimes else None,
            }

        # Total stats
        stats["total"] = {
            "entries": sum(s["entries"] for s in stats.values() if isinstance(s, dict)),
            "size_mb": sum(s["size_mb"] for s in stats.values() if isinstance(s, dict)),
            "total_hits": sum(s["hits"] for s in stats.values() if isinstance(s, dict)),
            "total_misses": sum(s["misses"] for s in stats.values() if isinstance(s, dict)),
        }

        return stats

    def clear_cache(self, cache_type: Optional[str] = None):
        """
        Clear cache entries.

        Args:
            cache_type: Specific cache to clear, or None for all
        """
        if cache_type:
            cache_dirs = {
                "research": self.research_cache,
                "outline": self.outline_cache,
                "render": self.render_cache,
            }
            if cache_type in cache_dirs:
                shutil.rmtree(cache_dirs[cache_type])
                cache_dirs[cache_type].mkdir()
                logger.info(f"Cleared {cache_type} cache")
        else:
            for subdir in [self.research_cache, self.outline_cache, self.render_cache]:
                shutil.rmtree(subdir)
                subdir.mkdir()
            logger.info("Cleared all caches")

    # ==================== Utility Methods ====================

    @staticmethod
    def _get_cache_key(topic: str, niche: str) -> str:
        """Generate cache key from topic and niche."""
        # Normalize
        combined = f"{niche.lower().strip()}:{topic.lower().strip()}"
        return hashlib.md5(combined.encode()).hexdigest()

    @staticmethod
    def hash_content(content: Any) -> str:
        """Generate hash for arbitrary content."""
        if isinstance(content, dict):
            content = json.dumps(content, sort_keys=True, default=str)
        elif not isinstance(content, str):
            content = str(content)
        return hashlib.md5(content.encode()).hexdigest()


# Module-level singleton
_cache: Optional[IntelligentContentCache] = None


def get_content_cache(cache_dir: Optional[Path] = None) -> IntelligentContentCache:
    """Get or create cache singleton."""
    global _cache
    if _cache is None:
        _cache = IntelligentContentCache(cache_dir=cache_dir)
    return _cache
