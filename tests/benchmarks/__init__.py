"""
Performance Benchmarking Suite for YouTube Automation.

This module provides benchmarks for:
- Video generation (FFmpeg, MoviePy, effects)
- TTS generation (Edge-TTS, Fish Audio)
- API costs and token usage
- Storage operations (caching, compression)
- Full pipeline performance

Run benchmarks:
    pytest tests/benchmarks/ -v --benchmark-only
    python -m tests.benchmarks.bench_video_generation
    python -m tests.benchmarks.bench_api_costs
    python -m tests.benchmarks.bench_storage

Example:
    from tests.benchmarks import VideoBenchmark, TTSBenchmark

    bench = VideoBenchmark()
    results = bench.compare_methods()
    print(bench.generate_report())
"""

from .bench_video_generation import (
    VideoBenchmark,
    TTSBenchmark,
    PipelineBenchmark,
    BenchmarkResult,
    run_all_benchmarks,
)

from .bench_api_costs import (
    APICostBenchmark,
    TokenUsage,
    CostEstimate,
    PROVIDER_COSTS,
    TYPICAL_OPERATIONS,
    run_api_benchmarks,
)

from .bench_storage import (
    StorageBenchmark,
    StorageBenchmarkResult,
    CacheStats,
    CompressionStats,
    DeduplicationStats,
    run_storage_benchmarks,
)

__all__ = [
    # Video benchmarks
    "VideoBenchmark",
    "TTSBenchmark",
    "PipelineBenchmark",
    "BenchmarkResult",
    "run_all_benchmarks",
    # API cost benchmarks
    "APICostBenchmark",
    "TokenUsage",
    "CostEstimate",
    "PROVIDER_COSTS",
    "TYPICAL_OPERATIONS",
    "run_api_benchmarks",
    # Storage benchmarks
    "StorageBenchmark",
    "StorageBenchmarkResult",
    "CacheStats",
    "CompressionStats",
    "DeduplicationStats",
    "run_storage_benchmarks",
]
