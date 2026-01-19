#!/usr/bin/env python3
"""
Disk Cleanup Utility - Automatic cleanup of old files to prevent disk space issues.

Usage:
    python run.py cleanup              # Clean files older than 30 days
    python run.py cleanup --days 7     # Clean files older than 7 days
    python run.py cleanup --dry-run    # Preview what would be deleted
    python run.py disk-usage           # Show disk usage stats
"""

import os
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any

from loguru import logger

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


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
