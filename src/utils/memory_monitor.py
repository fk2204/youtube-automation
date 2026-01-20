"""
Memory Monitoring Utilities

Tools for monitoring and managing memory usage during video processing.
Prevents crashes during long batch operations.

Features:
- Memory usage decorator
- Memory tracker class
- Memory-optimized context manager
- Batch processing with memory limits
"""

import asyncio
import functools
import gc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar

from loguru import logger

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - memory monitoring disabled")


T = TypeVar("T")


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    percent: float  # Percentage of total system memory
    available_mb: float  # Available system memory


def get_process_memory_mb() -> float:
    """
    Get current process memory usage in MB.

    Returns:
        RSS memory in megabytes
    """
    if not PSUTIL_AVAILABLE:
        return 0.0

    try:
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except Exception as e:
        logger.debug(f"Could not get memory: {e}")
        return 0.0


def get_memory_stats() -> MemoryStats:
    """
    Get detailed memory statistics.

    Returns:
        MemoryStats dataclass with current usage
    """
    if not PSUTIL_AVAILABLE:
        return MemoryStats(0.0, 0.0, 0.0, 0.0)

    try:
        process = psutil.Process()
        mem_info = process.memory_info()
        sys_mem = psutil.virtual_memory()

        return MemoryStats(
            rss_mb=mem_info.rss / (1024 * 1024),
            vms_mb=mem_info.vms / (1024 * 1024),
            percent=process.memory_percent(),
            available_mb=sys_mem.available / (1024 * 1024),
        )
    except Exception as e:
        logger.debug(f"Could not get memory stats: {e}")
        return MemoryStats(0.0, 0.0, 0.0, 0.0)


def log_memory_usage(context: str = "") -> None:
    """
    Log current memory usage.

    Args:
        context: Optional context string for the log message
    """
    stats = get_memory_stats()
    prefix = f"[{context}] " if context else ""
    logger.debug(
        f"{prefix}Memory: {stats.rss_mb:.1f} MB RSS, "
        f"{stats.percent:.1f}% of system, "
        f"{stats.available_mb:.0f} MB available"
    )


def monitor_memory(func: Callable) -> Callable:
    """
    Decorator to monitor memory usage of a function.

    Logs warning if memory increases by more than 100MB.

    Args:
        func: Function to monitor

    Returns:
        Wrapped function
    """
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        mem_before = get_process_memory_mb()

        try:
            result = func(*args, **kwargs)
        finally:
            mem_after = get_process_memory_mb()
            mem_delta = mem_after - mem_before

            if mem_delta > 100:
                logger.warning(
                    f"{func.__name__} increased memory by {mem_delta:.0f} MB "
                    f"({mem_before:.0f} -> {mem_after:.0f} MB)"
                )
            elif mem_delta > 50:
                logger.debug(
                    f"{func.__name__} memory delta: +{mem_delta:.0f} MB"
                )

        return result

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        mem_before = get_process_memory_mb()

        try:
            result = await func(*args, **kwargs)
        finally:
            mem_after = get_process_memory_mb()
            mem_delta = mem_after - mem_before

            if mem_delta > 100:
                logger.warning(
                    f"{func.__name__} increased memory by {mem_delta:.0f} MB "
                    f"({mem_before:.0f} -> {mem_after:.0f} MB)"
                )
            elif mem_delta > 50:
                logger.debug(
                    f"{func.__name__} memory delta: +{mem_delta:.0f} MB"
                )

        return result

    # Return appropriate wrapper based on function type
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


class MemoryTracker:
    """
    Track memory usage over time.

    Usage:
        tracker = MemoryTracker(threshold_mb=4096)

        if tracker.is_memory_critical():
            tracker.force_cleanup()

        stats = tracker.get_memory_stats()
    """

    def __init__(self, threshold_mb: float = 4096):
        """
        Initialize memory tracker.

        Args:
            threshold_mb: Memory threshold for critical warnings (default: 4GB)
        """
        self.threshold_mb = threshold_mb
        self.baseline_mb = get_process_memory_mb()
        self.peak_mb = self.baseline_mb
        self.cleanups_performed = 0

        logger.debug(f"MemoryTracker initialized: baseline={self.baseline_mb:.0f} MB")

    def check_memory(self) -> float:
        """
        Check current memory usage.

        Returns:
            Current RSS in MB
        """
        current = get_process_memory_mb()
        self.peak_mb = max(self.peak_mb, current)
        return current

    def is_memory_critical(self) -> bool:
        """
        Check if memory usage is above threshold.

        Returns:
            True if memory is critical
        """
        current = self.check_memory()
        is_critical = current > self.threshold_mb

        if is_critical:
            logger.warning(
                f"Memory critical: {current:.0f} MB > {self.threshold_mb:.0f} MB threshold"
            )

        return is_critical

    def force_cleanup(self) -> float:
        """
        Force garbage collection and return freed memory.

        Returns:
            MB of memory freed
        """
        before = get_process_memory_mb()

        # Run multiple GC passes
        gc.collect()
        gc.collect()
        gc.collect()

        after = get_process_memory_mb()
        freed = before - after
        self.cleanups_performed += 1

        logger.info(
            f"Memory cleanup #{self.cleanups_performed}: "
            f"freed {freed:.0f} MB ({before:.0f} -> {after:.0f} MB)"
        )

        return freed

    def get_stats(self) -> Dict[str, float]:
        """
        Get memory statistics.

        Returns:
            Dict with memory metrics
        """
        current = self.check_memory()
        return {
            "current_mb": current,
            "baseline_mb": self.baseline_mb,
            "peak_mb": self.peak_mb,
            "threshold_mb": self.threshold_mb,
            "delta_mb": current - self.baseline_mb,
            "usage_percent": (current / self.threshold_mb * 100) if self.threshold_mb > 0 else 0,
            "cleanups": self.cleanups_performed,
        }


@contextmanager
def memory_optimized_context(
    cleanup_on_exit: bool = True,
    log_usage: bool = True,
    context_name: str = ""
) -> Generator[MemoryTracker, None, None]:
    """
    Context manager for memory-optimized operations.

    Runs garbage collection on entry and exit.

    Usage:
        with memory_optimized_context("video_encoding") as tracker:
            # ... memory-intensive operations ...
            if tracker.is_memory_critical():
                tracker.force_cleanup()

    Args:
        cleanup_on_exit: Run GC on exit
        log_usage: Log memory delta
        context_name: Name for logging

    Yields:
        MemoryTracker instance
    """
    # Cleanup on entry
    gc.collect()

    tracker = MemoryTracker()
    start_mb = tracker.check_memory()

    if log_usage:
        log_memory_usage(f"{context_name} START" if context_name else "START")

    try:
        yield tracker
    finally:
        if cleanup_on_exit:
            gc.collect()

        end_mb = get_process_memory_mb()
        delta = end_mb - start_mb

        if log_usage:
            logger.debug(
                f"[{context_name or 'END'}] Memory delta: {delta:+.0f} MB "
                f"({start_mb:.0f} -> {end_mb:.0f} MB)"
            )


async def process_with_memory_limit(
    items: List[T],
    process_func: Callable[[T], Any],
    batch_size: int = 5,
    memory_threshold_mb: float = 4096,
    cleanup_between_batches: bool = True
) -> List[Any]:
    """
    Process items in batches with memory management.

    Pauses and cleans up if memory exceeds threshold.

    Args:
        items: Items to process
        process_func: Function to apply to each item
        batch_size: Items per batch
        memory_threshold_mb: Memory threshold for cleanup
        cleanup_between_batches: Force GC between batches

    Returns:
        List of results
    """
    results = []
    tracker = MemoryTracker(threshold_mb=memory_threshold_mb)

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(items) + batch_size - 1) // batch_size

        logger.debug(f"Processing batch {batch_num}/{total_batches}")

        # Check memory before batch
        if tracker.is_memory_critical():
            logger.warning(f"Memory critical before batch {batch_num}, cleaning up...")
            tracker.force_cleanup()
            await asyncio.sleep(0.5)  # Brief pause

        # Process batch
        for item in batch:
            try:
                if asyncio.iscoroutinefunction(process_func):
                    result = await process_func(item)
                else:
                    result = process_func(item)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing item: {e}")
                results.append(None)

        # Cleanup between batches
        if cleanup_between_batches:
            gc.collect()

        # Log progress
        logger.debug(
            f"Batch {batch_num}/{total_batches} complete, "
            f"memory: {tracker.check_memory():.0f} MB"
        )

    logger.info(
        f"Batch processing complete: {len(results)}/{len(items)} items, "
        f"peak memory: {tracker.peak_mb:.0f} MB, "
        f"cleanups: {tracker.cleanups_performed}"
    )

    return results


def process_with_memory_limit_sync(
    items: List[T],
    process_func: Callable[[T], Any],
    batch_size: int = 5,
    memory_threshold_mb: float = 4096
) -> List[Any]:
    """
    Synchronous version of process_with_memory_limit.

    Args:
        items: Items to process
        process_func: Function to apply to each item
        batch_size: Items per batch
        memory_threshold_mb: Memory threshold

    Returns:
        List of results
    """
    results = []
    tracker = MemoryTracker(threshold_mb=memory_threshold_mb)

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]

        if tracker.is_memory_critical():
            tracker.force_cleanup()

        for item in batch:
            try:
                result = process_func(item)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing item: {e}")
                results.append(None)

        gc.collect()

    return results
