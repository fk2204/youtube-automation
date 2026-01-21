"""
Performance Profiler for YouTube Automation.

Provides utilities for tracking execution time, memory usage, and performance
metrics across the application.

Features:
- Context manager for profiling code blocks
- Decorator for profiling functions
- Memory tracking with tracemalloc
- Aggregated performance reports
- Thread-safe operation tracking

Usage:
    from src.utils.profiler import Profiler

    # Context manager
    with Profiler.profile("video_generation"):
        create_video(...)

    # Decorator
    @Profiler.profile_func
    def my_function():
        ...

    # Generate report
    print(Profiler.get_report())

    # Clear results
    Profiler.clear()
"""

import time
import functools
import tracemalloc
import threading
import statistics
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path

from loguru import logger


@dataclass
class ProfileResult:
    """Result of a single profiling operation."""
    name: str
    duration_ms: float
    memory_peak_mb: float
    memory_current_mb: float
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "duration_ms": self.duration_ms,
            "memory_peak_mb": self.memory_peak_mb,
            "memory_current_mb": self.memory_current_mb,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class AggregatedStats:
    """Aggregated statistics for a profiled operation."""
    name: str
    count: int
    total_duration_ms: float
    avg_duration_ms: float
    min_duration_ms: float
    max_duration_ms: float
    std_dev_ms: float
    avg_memory_peak_mb: float
    total_memory_peak_mb: float

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "count": self.count,
            "total_duration_ms": self.total_duration_ms,
            "avg_duration_ms": self.avg_duration_ms,
            "min_duration_ms": self.min_duration_ms,
            "max_duration_ms": self.max_duration_ms,
            "std_dev_ms": self.std_dev_ms,
            "avg_memory_peak_mb": self.avg_memory_peak_mb,
            "total_memory_peak_mb": self.total_memory_peak_mb,
        }


class Profiler:
    """
    Performance profiler for tracking execution time and memory.

    Thread-safe class for profiling code blocks and functions.
    Results are stored globally and can be aggregated into reports.
    """

    # Class-level storage (thread-safe)
    _results: List[ProfileResult] = []
    _lock = threading.Lock()
    _enabled: bool = True
    _tracemalloc_started: bool = False

    @classmethod
    def enable(cls):
        """Enable profiling."""
        cls._enabled = True
        logger.debug("Profiler enabled")

    @classmethod
    def disable(cls):
        """Disable profiling."""
        cls._enabled = False
        logger.debug("Profiler disabled")

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if profiling is enabled."""
        return cls._enabled

    @classmethod
    @contextmanager
    def profile(cls, name: str, metadata: Dict[str, Any] = None):
        """
        Context manager for profiling a code block.

        Args:
            name: Name/identifier for this profile operation
            metadata: Optional metadata to attach to the result

        Yields:
            None

        Example:
            with Profiler.profile("video_generation"):
                create_video(audio_file, output_file)
        """
        if not cls._enabled:
            yield
            return

        # Start memory tracking
        was_tracing = tracemalloc.is_tracing()
        if not was_tracing:
            tracemalloc.start()

        start_time = time.perf_counter()

        try:
            yield
        finally:
            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Get memory stats
            current, peak = tracemalloc.get_traced_memory()

            # Stop tracemalloc if we started it
            if not was_tracing:
                tracemalloc.stop()

            # Create result
            result = ProfileResult(
                name=name,
                duration_ms=duration_ms,
                memory_peak_mb=peak / 1024 / 1024,
                memory_current_mb=current / 1024 / 1024,
                metadata=metadata or {}
            )

            # Store result (thread-safe)
            with cls._lock:
                cls._results.append(result)

            # Log the result
            logger.debug(
                f"[PROFILE] {name}: {duration_ms:.1f}ms, "
                f"{peak/1024/1024:.1f}MB peak"
            )

    @classmethod
    def profile_func(cls, func: Callable = None, *, name: str = None) -> Callable:
        """
        Decorator for profiling functions.

        Can be used with or without arguments:

        @Profiler.profile_func
        def my_function():
            ...

        @Profiler.profile_func(name="custom_name")
        def another_function():
            ...

        Args:
            func: Function to profile (when used without arguments)
            name: Optional custom name for the profile

        Returns:
            Decorated function
        """
        def decorator(f: Callable) -> Callable:
            profile_name = name or f.__name__

            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                if not cls._enabled:
                    return f(*args, **kwargs)

                with cls.profile(profile_name):
                    return f(*args, **kwargs)

            @functools.wraps(f)
            async def async_wrapper(*args, **kwargs):
                if not cls._enabled:
                    return await f(*args, **kwargs)

                with cls.profile(profile_name):
                    return await f(*args, **kwargs)

            # Return appropriate wrapper based on function type
            import asyncio
            if asyncio.iscoroutinefunction(f):
                return async_wrapper
            return wrapper

        # Handle both @profile_func and @profile_func(name="...")
        if func is not None:
            return decorator(func)
        return decorator

    @classmethod
    def get_results(cls) -> List[ProfileResult]:
        """
        Get all profiling results.

        Returns:
            List of ProfileResult objects
        """
        with cls._lock:
            return list(cls._results)

    @classmethod
    def get_results_by_name(cls, name: str) -> List[ProfileResult]:
        """
        Get all profiling results for a specific operation.

        Args:
            name: Operation name to filter by

        Returns:
            List of ProfileResult objects matching the name
        """
        with cls._lock:
            return [r for r in cls._results if r.name == name]

    @classmethod
    def get_aggregated_stats(cls) -> Dict[str, AggregatedStats]:
        """
        Get aggregated statistics for all profiled operations.

        Returns:
            Dictionary mapping operation names to AggregatedStats
        """
        with cls._lock:
            # Group results by name
            grouped: Dict[str, List[ProfileResult]] = {}
            for result in cls._results:
                if result.name not in grouped:
                    grouped[result.name] = []
                grouped[result.name].append(result)

        # Calculate aggregated stats
        stats = {}
        for name, results in grouped.items():
            durations = [r.duration_ms for r in results]
            memory_peaks = [r.memory_peak_mb for r in results]

            stats[name] = AggregatedStats(
                name=name,
                count=len(results),
                total_duration_ms=sum(durations),
                avg_duration_ms=statistics.mean(durations),
                min_duration_ms=min(durations),
                max_duration_ms=max(durations),
                std_dev_ms=statistics.stdev(durations) if len(durations) > 1 else 0,
                avg_memory_peak_mb=statistics.mean(memory_peaks),
                total_memory_peak_mb=sum(memory_peaks),
            )

        return stats

    @classmethod
    def get_report(cls, format: str = "text") -> str:
        """
        Generate a profiling report.

        Args:
            format: Output format ("text" or "markdown")

        Returns:
            Formatted report string
        """
        stats = cls.get_aggregated_stats()

        if not stats:
            return "No profiling data available."

        if format == "markdown":
            return cls._generate_markdown_report(stats)
        return cls._generate_text_report(stats)

    @classmethod
    def _generate_text_report(cls, stats: Dict[str, AggregatedStats]) -> str:
        """Generate a text-formatted report."""
        lines = [
            "=" * 80,
            "  PERFORMANCE PROFILING REPORT",
            "=" * 80,
            f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Total operations tracked: {sum(s.count for s in stats.values())}",
            "=" * 80,
            "",
            f"{'Operation':<30} {'Count':>8} {'Avg (ms)':>10} {'Min':>10} {'Max':>10} {'Memory (MB)':>12}",
            "-" * 80,
        ]

        # Sort by total duration (most time-consuming first)
        sorted_stats = sorted(
            stats.values(),
            key=lambda x: x.total_duration_ms,
            reverse=True
        )

        for s in sorted_stats:
            lines.append(
                f"{s.name:<30} {s.count:>8} {s.avg_duration_ms:>10.1f} "
                f"{s.min_duration_ms:>10.1f} {s.max_duration_ms:>10.1f} "
                f"{s.avg_memory_peak_mb:>12.1f}"
            )

        lines.extend([
            "-" * 80,
            "",
            "Top 5 Slowest Operations:",
        ])

        for i, s in enumerate(sorted_stats[:5], 1):
            lines.append(f"  {i}. {s.name}: {s.total_duration_ms:.1f}ms total ({s.count} calls)")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    @classmethod
    def _generate_markdown_report(cls, stats: Dict[str, AggregatedStats]) -> str:
        """Generate a markdown-formatted report."""
        lines = [
            "# Performance Profiling Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total operations tracked:** {sum(s.count for s in stats.values())}",
            "",
            "## Summary by Operation",
            "",
            "| Operation | Count | Avg (ms) | Min (ms) | Max (ms) | Memory (MB) |",
            "|-----------|-------|----------|----------|----------|-------------|",
        ]

        sorted_stats = sorted(
            stats.values(),
            key=lambda x: x.total_duration_ms,
            reverse=True
        )

        for s in sorted_stats:
            lines.append(
                f"| {s.name} | {s.count} | {s.avg_duration_ms:.1f} | "
                f"{s.min_duration_ms:.1f} | {s.max_duration_ms:.1f} | "
                f"{s.avg_memory_peak_mb:.1f} |"
            )

        lines.extend([
            "",
            "## Top 5 Slowest Operations",
            "",
        ])

        for i, s in enumerate(sorted_stats[:5], 1):
            lines.append(
                f"{i}. **{s.name}**: {s.total_duration_ms:.1f}ms total ({s.count} calls)"
            )

        return "\n".join(lines)

    @classmethod
    def clear(cls):
        """Clear all profiling results."""
        with cls._lock:
            cls._results = []
        logger.debug("Profiler results cleared")

    @classmethod
    def save_report(cls, filepath: str, format: str = "markdown"):
        """
        Save profiling report to a file.

        Args:
            filepath: Output file path
            format: Report format ("text" or "markdown")
        """
        report = cls.get_report(format=format)
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            f.write(report)

        logger.info(f"Profiling report saved to: {filepath}")

    @classmethod
    def get_total_time(cls) -> float:
        """
        Get total profiled time in milliseconds.

        Returns:
            Total duration of all profiled operations
        """
        with cls._lock:
            return sum(r.duration_ms for r in cls._results)

    @classmethod
    def get_operation_count(cls) -> int:
        """
        Get total number of profiled operations.

        Returns:
            Count of all profiled operations
        """
        with cls._lock:
            return len(cls._results)


class TimingContext:
    """
    Lightweight timing context manager without memory tracking.

    Use this for simple timing when memory tracking overhead is not needed.

    Example:
        with TimingContext() as timer:
            do_something()
        print(f"Took {timer.elapsed_ms:.1f}ms")
    """

    def __init__(self):
        self.start_time: float = 0
        self.end_time: float = 0
        self.elapsed_ms: float = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000
        return False


def timed(func: Callable) -> Callable:
    """
    Simple timing decorator that logs execution time.

    Unlike Profiler.profile_func, this doesn't track memory
    and is more lightweight.

    Example:
        @timed
        def my_function():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = (time.perf_counter() - start) * 1000
        logger.debug(f"[TIMED] {func.__name__}: {duration:.1f}ms")
        return result

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        duration = (time.perf_counter() - start) * 1000
        logger.debug(f"[TIMED] {func.__name__}: {duration:.1f}ms")
        return result

    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return wrapper


# Convenience functions
def profile(name: str, metadata: Dict[str, Any] = None):
    """Convenience function for Profiler.profile context manager."""
    return Profiler.profile(name, metadata)


def profile_func(func: Callable = None, *, name: str = None):
    """Convenience function for Profiler.profile_func decorator."""
    return Profiler.profile_func(func, name=name)


def get_report(format: str = "text") -> str:
    """Convenience function for Profiler.get_report."""
    return Profiler.get_report(format)


def clear():
    """Convenience function for Profiler.clear."""
    Profiler.clear()


# Example usage
if __name__ == "__main__":
    import random

    # Enable profiler
    Profiler.enable()

    # Example 1: Context manager
    with Profiler.profile("example_operation"):
        time.sleep(0.1)  # Simulate work
        data = [random.random() for _ in range(10000)]

    # Example 2: Decorated function
    @Profiler.profile_func
    def process_data(n: int) -> List[float]:
        """Process some data."""
        return sorted([random.random() for _ in range(n)])

    @Profiler.profile_func(name="custom_sort")
    def another_function():
        """Another function with custom profile name."""
        time.sleep(0.05)

    # Run profiled functions
    for _ in range(5):
        process_data(5000)
        another_function()

    # Example 3: Simple timing
    with TimingContext() as timer:
        time.sleep(0.02)
    print(f"Simple timing: {timer.elapsed_ms:.1f}ms")

    # Example 4: Lightweight timed decorator
    @timed
    def quick_function():
        return sum(range(1000))

    quick_function()

    # Generate and print report
    print("\n" + "=" * 60)
    print(Profiler.get_report(format="text"))

    # Save markdown report
    Profiler.save_report("output/profiler_report.md", format="markdown")

    # Get aggregated stats
    stats = Profiler.get_aggregated_stats()
    for name, stat in stats.items():
        print(f"\n{name}:")
        print(f"  Calls: {stat.count}")
        print(f"  Avg: {stat.avg_duration_ms:.1f}ms")
        print(f"  Memory: {stat.avg_memory_peak_mb:.1f}MB peak")

    # Clear results
    Profiler.clear()
    print(f"\nResults after clear: {Profiler.get_operation_count()}")
