#!/usr/bin/env python3
"""
Disk Cleanup Utility - Automatic cleanup of old files to prevent disk space issues.

Usage:
    python run.py cleanup              # Clean files older than 30 days
    python run.py cleanup --days 7     # Clean files older than 7 days
    python run.py cleanup --dry-run    # Preview what would be deleted
    python run.py disk-usage           # Show disk usage stats
    python run.py tiered-storage       # Run tiered storage migration
    python run.py predictive-cleanup   # Run predictive cleanup based on disk space

Features:
    - Tiered storage (hot/warm/cold) for automatic archival
    - Video compression using HEVC for archives
    - Predictive cleanup based on disk space thresholds
    - Auto-delete source files after successful YouTube upload
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple

from loguru import logger

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


class TieredStorage:
    """
    Tiered storage system for automatic file archival.

    Automatically moves files between storage tiers based on age:
    - Hot (0-7 days): Active working files in output/
    - Warm (8-30 days): Recent archives in archive/warm/
    - Cold (31-90 days): Long-term archives in archive/cold/ (compressed)
    - Delete (90+ days): Files are removed

    Usage:
        tiered = TieredStorage()
        tiered.run_migration()  # Move files to appropriate tiers
        tiered.compress_cold_tier()  # Compress files in cold storage
    """

    TIERS = {
        "hot": {"path": "output/", "max_age_days": 7},
        "warm": {"path": "archive/warm/", "max_age_days": 30},
        "cold": {"path": "archive/cold/", "max_age_days": 90}
    }

    # File extensions to process
    VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".ogg"}
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize tiered storage.

        Args:
            project_root: Project root directory (default: auto-detect)
        """
        self.project_root = Path(project_root) if project_root else PROJECT_ROOT

        # Ensure archive directories exist
        for tier_name, tier_config in self.TIERS.items():
            if tier_name != "hot":  # Hot tier already exists (output/)
                tier_path = self.project_root / tier_config["path"]
                tier_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"TieredStorage initialized at {self.project_root}")

    def get_file_age_days(self, file_path: Path) -> int:
        """Get file age in days from modification time."""
        try:
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            return (datetime.now() - mtime).days
        except (OSError, IOError):
            return 0

    def get_target_tier(self, age_days: int) -> Optional[str]:
        """
        Determine which tier a file should be in based on age.

        Args:
            age_days: File age in days

        Returns:
            Tier name ("hot", "warm", "cold") or None if should be deleted
        """
        if age_days <= self.TIERS["hot"]["max_age_days"]:
            return "hot"
        elif age_days <= self.TIERS["warm"]["max_age_days"]:
            return "warm"
        elif age_days <= self.TIERS["cold"]["max_age_days"]:
            return "cold"
        else:
            return None  # File should be deleted

    def move_to_tier(
        self,
        file_path: Path,
        age_days: int,
        dry_run: bool = False
    ) -> Optional[Path]:
        """
        Move a file to the appropriate tier based on age.

        Args:
            file_path: Path to the file
            age_days: File age in days
            dry_run: If True, only log what would happen

        Returns:
            New file path or None if file was deleted or not moved
        """
        target_tier = self.get_target_tier(age_days)

        if target_tier is None:
            # File is too old, should be deleted
            if dry_run:
                logger.info(f"Would delete (>90 days): {file_path}")
                return None
            else:
                try:
                    file_path.unlink()
                    logger.info(f"Deleted (>90 days): {file_path}")
                except OSError as e:
                    logger.error(f"Failed to delete {file_path}: {e}")
                return None

        # Determine current tier based on path
        rel_path = file_path.relative_to(self.project_root) if file_path.is_relative_to(self.project_root) else file_path
        current_tier = None

        for tier_name, tier_config in self.TIERS.items():
            if str(rel_path).startswith(tier_config["path"].rstrip("/")):
                current_tier = tier_name
                break

        # If already in the correct tier, no action needed
        if current_tier == target_tier:
            return file_path

        # Determine relative path within the tier
        if current_tier:
            # Remove tier prefix from path
            tier_prefix = self.TIERS[current_tier]["path"]
            inner_path = str(rel_path)[len(tier_prefix):]
        else:
            inner_path = str(file_path.name)

        # Build new path in target tier
        target_path = self.project_root / self.TIERS[target_tier]["path"] / inner_path

        if dry_run:
            logger.info(f"Would move to {target_tier}: {file_path} -> {target_path}")
            return target_path

        try:
            # Ensure target directory exists
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # Move the file
            shutil.move(str(file_path), str(target_path))
            logger.info(f"Moved to {target_tier}: {file_path.name} ({age_days} days old)")
            return target_path

        except (OSError, shutil.Error) as e:
            logger.error(f"Failed to move {file_path} to {target_tier}: {e}")
            return file_path

    def run_migration(
        self,
        dry_run: bool = False,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run tier migration for all files in output directory.

        Args:
            dry_run: If True, only preview what would happen
            verbose: If True, log each file being processed

        Returns:
            Migration statistics
        """
        stats = {
            "files_processed": 0,
            "files_moved_to_warm": 0,
            "files_moved_to_cold": 0,
            "files_deleted": 0,
            "bytes_archived": 0,
            "bytes_deleted": 0,
            "errors": []
        }

        # Process files in hot tier (output/)
        hot_path = self.project_root / self.TIERS["hot"]["path"]

        if not hot_path.exists():
            logger.warning(f"Hot tier path does not exist: {hot_path}")
            return stats

        all_extensions = self.VIDEO_EXTENSIONS | self.AUDIO_EXTENSIONS | self.IMAGE_EXTENSIONS

        for file_path in hot_path.rglob("*"):
            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in all_extensions:
                continue

            stats["files_processed"] += 1
            age_days = self.get_file_age_days(file_path)
            file_size = file_path.stat().st_size

            target_tier = self.get_target_tier(age_days)

            if target_tier == "warm":
                result = self.move_to_tier(file_path, age_days, dry_run)
                if result and result != file_path:
                    stats["files_moved_to_warm"] += 1
                    stats["bytes_archived"] += file_size
            elif target_tier == "cold":
                result = self.move_to_tier(file_path, age_days, dry_run)
                if result and result != file_path:
                    stats["files_moved_to_cold"] += 1
                    stats["bytes_archived"] += file_size
            elif target_tier is None:
                result = self.move_to_tier(file_path, age_days, dry_run)
                if result is None:
                    stats["files_deleted"] += 1
                    stats["bytes_deleted"] += file_size

        # Also process warm tier -> cold tier migration
        warm_path = self.project_root / self.TIERS["warm"]["path"]
        if warm_path.exists():
            for file_path in warm_path.rglob("*"):
                if not file_path.is_file():
                    continue

                if file_path.suffix.lower() not in all_extensions:
                    continue

                stats["files_processed"] += 1
                age_days = self.get_file_age_days(file_path)
                file_size = file_path.stat().st_size
                target_tier = self.get_target_tier(age_days)

                if target_tier == "cold":
                    result = self.move_to_tier(file_path, age_days, dry_run)
                    if result and result != file_path:
                        stats["files_moved_to_cold"] += 1
                        stats["bytes_archived"] += file_size
                elif target_tier is None:
                    result = self.move_to_tier(file_path, age_days, dry_run)
                    if result is None:
                        stats["files_deleted"] += 1
                        stats["bytes_deleted"] += file_size

        logger.info(
            f"Tier migration complete: "
            f"{stats['files_moved_to_warm']} to warm, "
            f"{stats['files_moved_to_cold']} to cold, "
            f"{stats['files_deleted']} deleted"
        )

        return stats

    def compress_archive_videos(
        self,
        input_file: Path,
        output_file: Optional[Path] = None,
        crf: int = 28,
        preset: str = "slow"
    ) -> Optional[Path]:
        """
        Compress a video file using HEVC (H.265) for archival.

        Typically achieves 50-70% size reduction with minimal quality loss.

        Args:
            input_file: Path to input video
            output_file: Path for compressed output (default: same with _hevc suffix)
            crf: Constant Rate Factor (18-28, higher = smaller file)
            preset: Encoding preset (ultrafast to veryslow)

        Returns:
            Path to compressed file or None on failure
        """
        input_file = Path(input_file)

        if not input_file.exists():
            logger.error(f"Input file not found: {input_file}")
            return None

        if output_file is None:
            output_file = input_file.with_suffix(".hevc.mp4")

        output_file = Path(output_file)

        cmd = [
            "ffmpeg", "-i", str(input_file),
            "-c:v", "libx265", "-crf", str(crf), "-preset", preset,
            "-c:a", "libopus", "-b:a", "64k",
            "-y",  # Overwrite output
            str(output_file)
        ]

        logger.info(f"Compressing {input_file.name} with HEVC (CRF {crf})...")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            if result.returncode != 0:
                logger.error(f"FFmpeg compression failed: {result.stderr[:500]}")
                return None

            # Log size reduction
            original_size = input_file.stat().st_size
            compressed_size = output_file.stat().st_size
            reduction = (1 - compressed_size / original_size) * 100

            logger.info(
                f"Compression complete: {input_file.name} "
                f"({original_size / 1024 / 1024:.1f} MB -> "
                f"{compressed_size / 1024 / 1024:.1f} MB, "
                f"{reduction:.1f}% reduction)"
            )

            return output_file

        except subprocess.TimeoutExpired:
            logger.error(f"Compression timed out for {input_file}")
            return None
        except FileNotFoundError:
            logger.error("FFmpeg not found. Please install FFmpeg.")
            return None
        except Exception as e:
            logger.error(f"Compression error: {e}")
            return None

    def compress_cold_tier(
        self,
        dry_run: bool = False,
        delete_originals: bool = True
    ) -> Dict[str, Any]:
        """
        Compress all uncompressed videos in cold storage tier.

        Args:
            dry_run: If True, only preview what would happen
            delete_originals: If True, delete originals after successful compression

        Returns:
            Compression statistics
        """
        stats = {
            "files_processed": 0,
            "files_compressed": 0,
            "bytes_saved": 0,
            "errors": []
        }

        cold_path = self.project_root / self.TIERS["cold"]["path"]

        if not cold_path.exists():
            return stats

        for video_file in cold_path.rglob("*"):
            if not video_file.is_file():
                continue

            # Skip already compressed files
            if ".hevc" in video_file.name.lower():
                continue

            if video_file.suffix.lower() not in self.VIDEO_EXTENSIONS:
                continue

            stats["files_processed"] += 1
            original_size = video_file.stat().st_size

            if dry_run:
                logger.info(f"Would compress: {video_file.name}")
                continue

            output_file = video_file.with_suffix(".hevc.mp4")
            result = self.compress_archive_videos(video_file, output_file)

            if result and result.exists():
                compressed_size = result.stat().st_size
                bytes_saved = original_size - compressed_size

                stats["files_compressed"] += 1
                stats["bytes_saved"] += bytes_saved

                if delete_originals:
                    try:
                        video_file.unlink()
                        logger.info(f"Deleted original: {video_file.name}")
                    except OSError as e:
                        stats["errors"].append(f"Could not delete {video_file}: {e}")
            else:
                stats["errors"].append(f"Failed to compress {video_file}")

        logger.info(
            f"Cold tier compression: {stats['files_compressed']} files, "
            f"{stats['bytes_saved'] / 1024 / 1024:.1f} MB saved"
        )

        return stats


def get_cleanup_directories() -> List[Path]:
    """Get list of directories to clean up."""
    return [
        PROJECT_ROOT / "output" / "videos",
        PROJECT_ROOT / "output" / "audio",
        PROJECT_ROOT / "output" / "thumbnails",
        PROJECT_ROOT / "output" / "shorts",
        PROJECT_ROOT / "data" / "stock_cache",
        PROJECT_ROOT / "cache",
        PROJECT_ROOT / "logs",
        Path(tempfile.gettempdir()) / "video_ultra",
        Path(tempfile.gettempdir()) / "video_shorts",
        Path(tempfile.gettempdir()) / "video_fast",
    ]


def cleanup_old_files(
    max_age_days: int = 30,
    dry_run: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Remove old files to free disk space.

    Args:
        max_age_days: Delete files older than this many days (default: 30)
        dry_run: If True, only preview what would be deleted without actually deleting
        verbose: If True, log each file being deleted

    Returns:
        dict with cleanup statistics:
            - files_deleted: Number of files deleted
            - space_freed_bytes: Total bytes freed
            - space_freed_gb: Total GB freed
            - dry_run: Whether this was a dry run
            - errors: List of any errors encountered
            - files_by_directory: Breakdown by directory
    """
    dirs_to_clean = get_cleanup_directories()

    total_freed = 0
    files_deleted = 0
    errors: List[str] = []
    files_by_directory: Dict[str, Dict[str, Any]] = {}
    deleted_files: List[Dict[str, Any]] = []

    now = datetime.now()

    for directory in dirs_to_clean:
        if not directory.exists():
            continue

        dir_files = 0
        dir_freed = 0
        dir_key = str(directory)

        try:
            for file in directory.rglob("*"):
                if not file.is_file():
                    continue

                try:
                    # Get file modification time
                    mtime = datetime.fromtimestamp(file.stat().st_mtime)
                    age_days = (now - mtime).days

                    if age_days > max_age_days:
                        file_size = file.stat().st_size

                        if verbose:
                            action = "Would delete" if dry_run else "Deleting"
                            logger.info(f"  {action}: {file} ({format_size(file_size)}, {age_days} days old)")

                        deleted_files.append({
                            "path": str(file),
                            "size": file_size,
                            "age_days": age_days
                        })

                        if not dry_run:
                            file.unlink()

                        total_freed += file_size
                        files_deleted += 1
                        dir_files += 1
                        dir_freed += file_size

                except PermissionError as e:
                    errors.append(f"Permission denied: {file}")
                    logger.warning(f"Permission denied: {file}")
                except Exception as e:
                    errors.append(f"Error processing {file}: {e}")
                    logger.warning(f"Error processing {file}: {e}")

        except Exception as e:
            errors.append(f"Error scanning {directory}: {e}")
            logger.warning(f"Error scanning {directory}: {e}")

        if dir_files > 0:
            files_by_directory[dir_key] = {
                "files_deleted": dir_files,
                "space_freed_bytes": dir_freed,
                "space_freed_gb": dir_freed / (1024**3)
            }

    # Clean empty directories after file cleanup
    if not dry_run:
        empty_dirs_removed = cleanup_empty_directories(dirs_to_clean, verbose)
    else:
        empty_dirs_removed = 0

    return {
        "files_deleted": files_deleted,
        "space_freed_bytes": total_freed,
        "space_freed_gb": total_freed / (1024**3),
        "dry_run": dry_run,
        "max_age_days": max_age_days,
        "errors": errors,
        "files_by_directory": files_by_directory,
        "empty_dirs_removed": empty_dirs_removed,
        "deleted_files": deleted_files if verbose else []
    }


def cleanup_empty_directories(directories: List[Path], verbose: bool = True) -> int:
    """
    Remove empty directories within the cleanup directories.

    Args:
        directories: List of root directories to check
        verbose: Log removed directories

    Returns:
        Number of empty directories removed
    """
    removed = 0

    for root_dir in directories:
        if not root_dir.exists():
            continue

        # Walk directories bottom-up to remove nested empty dirs
        for dirpath in sorted(root_dir.rglob("*"), reverse=True):
            if dirpath.is_dir():
                try:
                    # Check if directory is empty
                    if not any(dirpath.iterdir()):
                        if verbose:
                            logger.info(f"  Removing empty directory: {dirpath}")
                        dirpath.rmdir()
                        removed += 1
                except Exception:
                    pass  # Directory not empty or permission error

    return removed


def get_disk_usage() -> Dict[str, Any]:
    """
    Get current disk usage stats for project directories.

    Returns:
        dict with disk usage information:
            - total_bytes: Total bytes used by project
            - total_gb: Total GB used
            - directories: Breakdown by directory
            - system_disk: System disk usage info
    """
    dirs_to_check = get_cleanup_directories()

    # Add additional project directories
    dirs_to_check.extend([
        PROJECT_ROOT / "output",
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "assets",
        PROJECT_ROOT / "logs",
    ])

    total_bytes = 0
    directories: Dict[str, Dict[str, Any]] = {}

    for directory in dirs_to_check:
        if not directory.exists():
            continue

        dir_key = str(directory)
        if dir_key in directories:
            continue  # Skip duplicates

        dir_size = 0
        file_count = 0
        oldest_file: Optional[datetime] = None
        newest_file: Optional[datetime] = None

        try:
            for file in directory.rglob("*"):
                if file.is_file():
                    try:
                        size = file.stat().st_size
                        mtime = datetime.fromtimestamp(file.stat().st_mtime)

                        dir_size += size
                        file_count += 1

                        if oldest_file is None or mtime < oldest_file:
                            oldest_file = mtime
                        if newest_file is None or mtime > newest_file:
                            newest_file = mtime

                    except Exception:
                        pass

        except Exception:
            pass

        if file_count > 0:
            directories[dir_key] = {
                "size_bytes": dir_size,
                "size_gb": dir_size / (1024**3),
                "size_formatted": format_size(dir_size),
                "file_count": file_count,
                "oldest_file": oldest_file.isoformat() if oldest_file else None,
                "newest_file": newest_file.isoformat() if newest_file else None,
                "oldest_age_days": (datetime.now() - oldest_file).days if oldest_file else 0
            }
            total_bytes += dir_size

    # Get system disk usage
    system_disk = get_system_disk_usage()

    return {
        "total_bytes": total_bytes,
        "total_gb": total_bytes / (1024**3),
        "total_formatted": format_size(total_bytes),
        "directories": directories,
        "system_disk": system_disk
    }


def get_system_disk_usage() -> Dict[str, Any]:
    """Get system disk usage information."""
    try:
        # Get disk usage for the project root
        total, used, free = shutil.disk_usage(PROJECT_ROOT)

        return {
            "total_bytes": total,
            "total_gb": total / (1024**3),
            "total_formatted": format_size(total),
            "used_bytes": used,
            "used_gb": used / (1024**3),
            "used_formatted": format_size(used),
            "free_bytes": free,
            "free_gb": free / (1024**3),
            "free_formatted": format_size(free),
            "used_percent": (used / total) * 100
        }
    except Exception as e:
        logger.warning(f"Could not get system disk usage: {e}")
        return {}


def format_size(bytes_size: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(bytes_size) < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def should_run_cleanup(threshold_gb: float = 10.0) -> bool:
    """
    Check if cleanup should run based on available disk space.

    Args:
        threshold_gb: Run cleanup if free space is below this (default: 10 GB)

    Returns:
        True if cleanup should run
    """
    system_disk = get_system_disk_usage()
    if not system_disk:
        return True  # Run cleanup if we can't check disk space

    free_gb = system_disk.get("free_gb", 0)
    return free_gb < threshold_gb


def print_disk_usage_report():
    """Print a formatted disk usage report."""
    usage = get_disk_usage()

    print()
    print("=" * 60)
    print("  DISK USAGE REPORT")
    print("=" * 60)
    print()

    # System disk info
    system_disk = usage.get("system_disk", {})
    if system_disk:
        print("  System Disk:")
        print("  " + "-" * 30)
        print(f"    Total:  {system_disk.get('total_formatted', 'N/A')}")
        print(f"    Used:   {system_disk.get('used_formatted', 'N/A')} ({system_disk.get('used_percent', 0):.1f}%)")
        print(f"    Free:   {system_disk.get('free_formatted', 'N/A')}")
        print()

    # Project directories
    print("  Project Directories:")
    print("  " + "-" * 30)

    directories = usage.get("directories", {})
    if directories:
        # Sort by size descending
        sorted_dirs = sorted(
            directories.items(),
            key=lambda x: x[1].get("size_bytes", 0),
            reverse=True
        )

        for dir_path, info in sorted_dirs:
            # Shorten path for display
            short_path = dir_path.replace(str(PROJECT_ROOT), ".")
            size = info.get("size_formatted", "0 B")
            files = info.get("file_count", 0)
            oldest = info.get("oldest_age_days", 0)

            print(f"    {short_path}")
            print(f"      Size: {size} | Files: {files} | Oldest: {oldest} days")
    else:
        print("    No data found")

    print()
    print(f"  Total Project Size: {usage.get('total_formatted', '0 B')}")
    print()
    print("=" * 60)


def print_cleanup_report(result: Dict[str, Any]):
    """Print a formatted cleanup report."""
    print()
    print("=" * 60)
    if result.get("dry_run"):
        print("  CLEANUP PREVIEW (DRY RUN)")
    else:
        print("  CLEANUP COMPLETE")
    print("=" * 60)
    print()

    print(f"  Files older than: {result.get('max_age_days', 30)} days")
    print()

    action = "Would delete" if result.get("dry_run") else "Deleted"
    print(f"  {action}: {result.get('files_deleted', 0)} files")
    print(f"  Space freed: {format_size(result.get('space_freed_bytes', 0))}")

    if not result.get("dry_run"):
        print(f"  Empty directories removed: {result.get('empty_dirs_removed', 0)}")

    # Show breakdown by directory
    files_by_dir = result.get("files_by_directory", {})
    if files_by_dir:
        print()
        print("  Breakdown by Directory:")
        print("  " + "-" * 30)

        for dir_path, info in files_by_dir.items():
            short_path = dir_path.replace(str(PROJECT_ROOT), ".")
            files = info.get("files_deleted", 0)
            freed = info.get("space_freed_bytes", 0)
            print(f"    {short_path}: {files} files ({format_size(freed)})")

    # Show errors if any
    errors = result.get("errors", [])
    if errors:
        print()
        print("  Errors:")
        print("  " + "-" * 30)
        for error in errors[:10]:  # Show max 10 errors
            print(f"    - {error}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more errors")

    print()
    print("=" * 60)


def run_scheduled_cleanup():
    """
    Run cleanup as a scheduled task.

    This function is designed to be called by the scheduler.
    It cleans files older than 30 days and logs the results.
    """
    logger.info("Starting scheduled disk cleanup...")

    # Check if cleanup is needed
    if not should_run_cleanup(threshold_gb=10.0):
        system_disk = get_system_disk_usage()
        free_gb = system_disk.get("free_gb", 0)
        logger.info(f"Skipping cleanup - sufficient disk space ({free_gb:.1f} GB free)")
        return {
            "skipped": True,
            "reason": "sufficient_disk_space",
            "free_gb": free_gb
        }

    # Run cleanup
    result = cleanup_old_files(
        max_age_days=30,
        dry_run=False,
        verbose=False  # Less verbose for scheduled runs
    )

    logger.info(
        f"Cleanup complete: {result['files_deleted']} files deleted, "
        f"{result['space_freed_gb']:.2f} GB freed"
    )

    return result


def predictive_cleanup(
    aggressive_threshold_gb: float = 5.0,
    normal_threshold_gb: float = 10.0,
    light_threshold_gb: float = 20.0,
    dry_run: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Smart cleanup based on available disk space.

    Automatically adjusts cleanup aggressiveness based on free disk space:
    - <5 GB free: Aggressive cleanup (7 days max age)
    - <10 GB free: Normal cleanup (14 days max age)
    - <20 GB free: Light cleanup (30 days max age)
    - >=20 GB free: No cleanup needed

    Args:
        aggressive_threshold_gb: Free space threshold for aggressive cleanup
        normal_threshold_gb: Free space threshold for normal cleanup
        light_threshold_gb: Free space threshold for light cleanup
        dry_run: If True, only preview what would happen
        verbose: If True, log details

    Returns:
        Cleanup result dictionary with cleanup level and statistics
    """
    # Get current disk space
    system_disk = get_system_disk_usage()
    free_gb = system_disk.get("free_gb", float('inf'))

    result = {
        "free_gb_before": free_gb,
        "cleanup_level": "none",
        "max_age_days": None,
        "cleanup_result": None,
        "tier_migration": None,
        "compression_result": None
    }

    if free_gb < aggressive_threshold_gb:
        # Aggressive cleanup - disk is critically low
        result["cleanup_level"] = "aggressive"
        result["max_age_days"] = 7

        if verbose:
            logger.warning(
                f"CRITICAL: Only {free_gb:.1f} GB free! "
                f"Running aggressive cleanup (files >7 days)"
            )

        # Run aggressive cleanup
        result["cleanup_result"] = cleanup_old_files(
            max_age_days=7,
            dry_run=dry_run,
            verbose=verbose
        )

        # Also run tier migration to move files to archives
        if not dry_run:
            tiered = TieredStorage()
            result["tier_migration"] = tiered.run_migration(dry_run=dry_run)
            # Compress cold tier to save space
            result["compression_result"] = tiered.compress_cold_tier(dry_run=dry_run)

    elif free_gb < normal_threshold_gb:
        # Normal cleanup
        result["cleanup_level"] = "normal"
        result["max_age_days"] = 14

        if verbose:
            logger.warning(
                f"Low disk space: {free_gb:.1f} GB free. "
                f"Running normal cleanup (files >14 days)"
            )

        result["cleanup_result"] = cleanup_old_files(
            max_age_days=14,
            dry_run=dry_run,
            verbose=verbose
        )

        # Run tier migration
        if not dry_run:
            tiered = TieredStorage()
            result["tier_migration"] = tiered.run_migration(dry_run=dry_run)

    elif free_gb < light_threshold_gb:
        # Light cleanup
        result["cleanup_level"] = "light"
        result["max_age_days"] = 30

        if verbose:
            logger.info(
                f"Moderate disk space: {free_gb:.1f} GB free. "
                f"Running light cleanup (files >30 days)"
            )

        result["cleanup_result"] = cleanup_old_files(
            max_age_days=30,
            dry_run=dry_run,
            verbose=verbose
        )

    else:
        # No cleanup needed
        if verbose:
            logger.info(f"Sufficient disk space: {free_gb:.1f} GB free. No cleanup needed.")
        result["cleanup_level"] = "none"

    # Get final disk space
    system_disk_after = get_system_disk_usage()
    result["free_gb_after"] = system_disk_after.get("free_gb", free_gb)
    result["space_recovered_gb"] = result["free_gb_after"] - free_gb

    return result


class PostUploadCleaner:
    """
    Handles cleanup of source files after successful YouTube upload.

    Tracks uploaded videos and provides options to automatically or manually
    delete source files after upload confirmation.

    Usage:
        cleaner = PostUploadCleaner()

        # After successful upload
        cleaner.mark_uploaded(video_path, youtube_video_id, youtube_url)

        # Clean up uploaded files
        cleaner.cleanup_uploaded_sources(confirm=True)

        # Or auto-cleanup if enabled
        cleaner.auto_cleanup_after_upload(video_path, youtube_video_id, youtube_url)
    """

    def __init__(
        self,
        tracking_file: Optional[Path] = None,
        auto_delete: bool = False,
        retention_days: int = 3
    ):
        """
        Initialize post-upload cleaner.

        Args:
            tracking_file: Path to store upload tracking data
            auto_delete: If True, automatically delete after retention period
            retention_days: Days to keep source files after upload
        """
        self.tracking_file = tracking_file or (PROJECT_ROOT / "data" / "upload_tracking.json")
        self.auto_delete = auto_delete
        self.retention_days = retention_days

        # Ensure directory exists
        self.tracking_file.parent.mkdir(parents=True, exist_ok=True)

        # Load tracking data
        self.tracking_data: Dict[str, Dict[str, Any]] = self._load_tracking()

    def _load_tracking(self) -> Dict[str, Dict[str, Any]]:
        """Load upload tracking data from file."""
        if self.tracking_file.exists():
            try:
                import json
                with open(self.tracking_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load upload tracking: {e}")
        return {}

    def _save_tracking(self) -> None:
        """Save upload tracking data to file."""
        try:
            import json
            with open(self.tracking_file, 'w', encoding='utf-8') as f:
                json.dump(self.tracking_data, f, indent=2)
        except (IOError, OSError) as e:
            logger.error(f"Failed to save upload tracking: {e}")

    def mark_uploaded(
        self,
        video_path: str,
        youtube_video_id: str,
        youtube_url: str,
        related_files: Optional[List[str]] = None
    ) -> None:
        """
        Mark a video as successfully uploaded to YouTube.

        Args:
            video_path: Path to the source video file
            youtube_video_id: YouTube video ID
            youtube_url: Full YouTube URL
            related_files: Optional list of related files (audio, thumbnails)
        """
        video_path = str(Path(video_path).resolve())

        # Collect all related files if not provided
        if related_files is None:
            related_files = self._find_related_files(video_path)

        entry = {
            "video_path": video_path,
            "youtube_video_id": youtube_video_id,
            "youtube_url": youtube_url,
            "upload_time": datetime.now().isoformat(),
            "related_files": related_files,
            "deleted": False,
            "delete_time": None
        }

        self.tracking_data[youtube_video_id] = entry
        self._save_tracking()

        logger.info(
            f"Marked as uploaded: {Path(video_path).name} -> {youtube_url}"
        )

        # Auto-delete if enabled
        if self.auto_delete:
            self._schedule_delete(youtube_video_id)

    def _find_related_files(self, video_path: str) -> List[str]:
        """Find related files (audio, thumbnails) based on video filename."""
        video_path = Path(video_path)
        base_name = video_path.stem
        parent_dir = video_path.parent

        related = []

        # Common related file patterns
        patterns = [
            f"{base_name}.mp3",
            f"{base_name}.wav",
            f"{base_name}_audio.mp3",
            f"{base_name}_thumbnail.png",
            f"{base_name}_thumbnail.jpg",
            f"{base_name}_thumb.png",
            f"{base_name}_thumb.jpg",
        ]

        for pattern in patterns:
            potential_file = parent_dir / pattern
            if potential_file.exists():
                related.append(str(potential_file))

        # Check audio subdirectory
        audio_dir = parent_dir.parent / "audio"
        if audio_dir.exists():
            for audio_file in audio_dir.glob(f"{base_name}*"):
                if audio_file.is_file():
                    related.append(str(audio_file))

        # Check thumbnails subdirectory
        thumb_dir = parent_dir.parent / "thumbnails"
        if thumb_dir.exists():
            for thumb_file in thumb_dir.glob(f"{base_name}*"):
                if thumb_file.is_file():
                    related.append(str(thumb_file))

        return related

    def _schedule_delete(self, youtube_video_id: str) -> None:
        """Schedule deletion after retention period."""
        # This would integrate with a scheduler - for now just log
        logger.info(
            f"Scheduled deletion for {youtube_video_id} in {self.retention_days} days"
        )

    def cleanup_uploaded_sources(
        self,
        min_age_days: Optional[int] = None,
        confirm: bool = True,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Clean up source files for videos that have been uploaded.

        Args:
            min_age_days: Only delete files uploaded at least this many days ago
            confirm: If True, require confirmation before deleting
            dry_run: If True, only preview what would be deleted

        Returns:
            Cleanup statistics
        """
        if min_age_days is None:
            min_age_days = self.retention_days

        stats = {
            "videos_processed": 0,
            "files_deleted": 0,
            "bytes_freed": 0,
            "errors": []
        }

        now = datetime.now()
        files_to_delete: List[Tuple[str, str]] = []  # (video_id, file_path)

        for video_id, entry in self.tracking_data.items():
            if entry.get("deleted"):
                continue

            try:
                upload_time = datetime.fromisoformat(entry["upload_time"])
                age_days = (now - upload_time).days

                if age_days >= min_age_days:
                    # Add video file
                    video_path = entry.get("video_path")
                    if video_path and Path(video_path).exists():
                        files_to_delete.append((video_id, video_path))

                    # Add related files
                    for related_file in entry.get("related_files", []):
                        if Path(related_file).exists():
                            files_to_delete.append((video_id, related_file))

            except (ValueError, KeyError) as e:
                stats["errors"].append(f"Error processing {video_id}: {e}")

        if not files_to_delete:
            logger.info("No uploaded source files ready for cleanup")
            return stats

        # Show what will be deleted
        if confirm and not dry_run:
            print(f"\nFiles to delete ({len(files_to_delete)} files):")
            for video_id, file_path in files_to_delete[:10]:
                print(f"  - {Path(file_path).name}")
            if len(files_to_delete) > 10:
                print(f"  ... and {len(files_to_delete) - 10} more files")

            response = input("\nProceed with deletion? (yes/no): ")
            if response.lower() not in ("yes", "y"):
                print("Cancelled.")
                return stats

        # Delete files
        deleted_video_ids = set()

        for video_id, file_path in files_to_delete:
            file_path = Path(file_path)
            stats["videos_processed"] += 1

            if dry_run:
                logger.info(f"Would delete: {file_path}")
                stats["files_deleted"] += 1
                continue

            try:
                file_size = file_path.stat().st_size
                file_path.unlink()
                stats["files_deleted"] += 1
                stats["bytes_freed"] += file_size
                deleted_video_ids.add(video_id)
                logger.info(f"Deleted: {file_path.name}")
            except OSError as e:
                stats["errors"].append(f"Failed to delete {file_path}: {e}")

        # Update tracking data
        if not dry_run:
            for video_id in deleted_video_ids:
                if video_id in self.tracking_data:
                    self.tracking_data[video_id]["deleted"] = True
                    self.tracking_data[video_id]["delete_time"] = datetime.now().isoformat()
            self._save_tracking()

        logger.info(
            f"Post-upload cleanup: {stats['files_deleted']} files deleted, "
            f"{stats['bytes_freed'] / 1024 / 1024:.1f} MB freed"
        )

        return stats

    def auto_cleanup_after_upload(
        self,
        video_path: str,
        youtube_video_id: str,
        youtube_url: str
    ) -> None:
        """
        One-step method to mark upload and optionally auto-delete.

        Call this after a successful YouTube upload to track the file
        and schedule deletion if auto_delete is enabled.

        Args:
            video_path: Path to the source video file
            youtube_video_id: YouTube video ID
            youtube_url: Full YouTube URL
        """
        self.mark_uploaded(video_path, youtube_video_id, youtube_url)

        if self.auto_delete and self.retention_days == 0:
            # Immediate deletion
            logger.info(f"Auto-deleting source files for {youtube_video_id}")
            self.cleanup_uploaded_sources(
                min_age_days=0,
                confirm=False,
                dry_run=False
            )

    def get_pending_cleanups(self) -> List[Dict[str, Any]]:
        """Get list of uploaded videos waiting for cleanup."""
        pending = []

        now = datetime.now()

        for video_id, entry in self.tracking_data.items():
            if entry.get("deleted"):
                continue

            try:
                upload_time = datetime.fromisoformat(entry["upload_time"])
                age_days = (now - upload_time).days
                video_path = Path(entry.get("video_path", ""))

                pending.append({
                    "video_id": video_id,
                    "youtube_url": entry.get("youtube_url"),
                    "upload_age_days": age_days,
                    "video_exists": video_path.exists(),
                    "video_size_mb": video_path.stat().st_size / 1024 / 1024 if video_path.exists() else 0,
                    "ready_for_cleanup": age_days >= self.retention_days
                })

            except (ValueError, KeyError):
                pass

        return pending


def delete_source_after_upload(
    video_path: str,
    youtube_video_id: str,
    youtube_url: str,
    immediate: bool = False,
    retention_days: int = 3
) -> None:
    """
    Convenience function to handle post-upload cleanup.

    Args:
        video_path: Path to the source video file
        youtube_video_id: YouTube video ID
        youtube_url: Full YouTube URL
        immediate: If True, delete immediately after marking
        retention_days: Days to keep before deletion (if not immediate)
    """
    cleaner = PostUploadCleaner(
        auto_delete=immediate,
        retention_days=0 if immediate else retention_days
    )

    cleaner.auto_cleanup_after_upload(video_path, youtube_video_id, youtube_url)


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Disk Cleanup Utility")
    parser.add_argument("--days", type=int, default=30,
                        help="Delete files older than N days (default: 30)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be deleted without deleting")
    parser.add_argument("--usage", action="store_true",
                        help="Show disk usage report")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Less verbose output")
    args = parser.parse_args()

    if args.usage:
        print_disk_usage_report()
    else:
        result = cleanup_old_files(
            max_age_days=args.days,
            dry_run=args.dry_run,
            verbose=not args.quiet
        )
        print_cleanup_report(result)
