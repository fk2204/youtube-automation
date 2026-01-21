"""
Storage Operations Benchmarks.

Benchmarks storage operations including:
- Cache hit/miss rates
- Deduplication savings
- Compression ratios
- Read/write performance

Run with: pytest tests/benchmarks/bench_storage.py -v
Or: python tests/benchmarks/bench_storage.py

Example:
    from tests.benchmarks.bench_storage import StorageBenchmark

    bench = StorageBenchmark()
    bench.benchmark_cache_performance()
    print(bench.generate_report())
"""

import os
import sys
import time
import json
import gzip
import shutil
import hashlib
import tempfile
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import random
import string

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from loguru import logger


@dataclass
class StorageBenchmarkResult:
    """Result of a storage benchmark."""
    name: str
    duration_seconds: float
    operations_count: int
    ops_per_second: float
    data_size_mb: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if self.duration_seconds > 0:
            self.ops_per_second = self.operations_count / self.duration_seconds

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CacheStats:
    """Cache performance statistics."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    hit_rate: float = 0.0
    avg_hit_time_ms: float = 0.0
    avg_miss_time_ms: float = 0.0
    total_size_mb: float = 0.0
    entry_count: int = 0

    def calculate_hit_rate(self):
        if self.total_requests > 0:
            self.hit_rate = self.cache_hits / self.total_requests * 100


@dataclass
class CompressionStats:
    """Compression benchmark statistics."""
    original_size_bytes: int = 0
    compressed_size_bytes: int = 0
    compression_ratio: float = 0.0
    compression_time_ms: float = 0.0
    decompression_time_ms: float = 0.0
    space_savings_percent: float = 0.0

    def calculate_ratio(self):
        if self.compressed_size_bytes > 0:
            self.compression_ratio = self.original_size_bytes / self.compressed_size_bytes
            self.space_savings_percent = (
                1 - (self.compressed_size_bytes / self.original_size_bytes)
            ) * 100


@dataclass
class DeduplicationStats:
    """Deduplication statistics."""
    total_files: int = 0
    unique_files: int = 0
    duplicate_files: int = 0
    total_size_bytes: int = 0
    unique_size_bytes: int = 0
    space_saved_bytes: int = 0
    space_saved_percent: float = 0.0

    def calculate_savings(self):
        if self.total_size_bytes > 0:
            self.space_saved_bytes = self.total_size_bytes - self.unique_size_bytes
            self.space_saved_percent = (self.space_saved_bytes / self.total_size_bytes) * 100


class StorageBenchmark:
    """Benchmark storage operations."""

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize storage benchmark.

        Args:
            output_dir: Directory for benchmark data. Uses temp dir if not specified.
        """
        self.results: List[StorageBenchmarkResult] = []
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Any] = {}
        logger.info(f"StorageBenchmark initialized, output dir: {self.output_dir}")

    def _generate_random_data(self, size_kb: int = 100) -> bytes:
        """Generate random data for testing."""
        return os.urandom(size_kb * 1024)

    def _generate_test_file(self, size_mb: float = 1.0) -> str:
        """Generate a test file with random data."""
        filepath = str(self.output_dir / f"test_{size_mb}mb_{time.time()}.bin")
        with open(filepath, 'wb') as f:
            # Write in chunks to avoid memory issues
            chunk_size = 1024 * 1024  # 1 MB chunks
            remaining = int(size_mb * 1024 * 1024)
            while remaining > 0:
                chunk = min(chunk_size, remaining)
                f.write(os.urandom(chunk))
                remaining -= chunk
        return filepath

    def _generate_cache_key(self, data: str) -> str:
        """Generate a cache key from data."""
        return hashlib.md5(data.encode()).hexdigest()

    def benchmark_cache_performance(
        self,
        num_operations: int = 1000,
        cache_size: int = 100
    ) -> CacheStats:
        """
        Benchmark cache hit/miss performance.

        Args:
            num_operations: Number of cache operations to perform
            cache_size: Number of unique items to cache

        Returns:
            CacheStats with hit/miss rates
        """
        logger.info(f"Benchmarking cache performance ({num_operations} operations)")

        stats = CacheStats()
        hit_times = []
        miss_times = []

        # Pre-populate cache with some items
        items = [f"item_{i}" for i in range(cache_size)]
        for item in items[:cache_size // 2]:  # Cache half the items
            key = self._generate_cache_key(item)
            self._cache[key] = {"data": item, "created": time.time()}

        # Run cache operations
        for _ in range(num_operations):
            item = random.choice(items)
            key = self._generate_cache_key(item)
            stats.total_requests += 1

            start = time.perf_counter()
            if key in self._cache:
                # Cache hit
                _ = self._cache[key]
                duration = (time.perf_counter() - start) * 1000
                hit_times.append(duration)
                stats.cache_hits += 1
            else:
                # Cache miss - simulate fetching data
                time.sleep(0.0001)  # Small delay for miss
                self._cache[key] = {"data": item, "created": time.time()}
                duration = (time.perf_counter() - start) * 1000
                miss_times.append(duration)
                stats.cache_misses += 1

        stats.calculate_hit_rate()
        stats.avg_hit_time_ms = sum(hit_times) / len(hit_times) if hit_times else 0
        stats.avg_miss_time_ms = sum(miss_times) / len(miss_times) if miss_times else 0
        stats.entry_count = len(self._cache)
        stats.total_size_mb = sum(
            len(str(v)) for v in self._cache.values()
        ) / 1024 / 1024

        result = StorageBenchmarkResult(
            name="cache_performance",
            duration_seconds=sum(hit_times + miss_times) / 1000,
            operations_count=num_operations,
            ops_per_second=0,
            data_size_mb=stats.total_size_mb,
            metadata={
                "hit_rate": stats.hit_rate,
                "avg_hit_time_ms": stats.avg_hit_time_ms,
                "avg_miss_time_ms": stats.avg_miss_time_ms,
            }
        )
        self.results.append(result)

        logger.info(f"Cache performance: {stats.hit_rate:.1f}% hit rate")
        logger.info(f"  Hit time: {stats.avg_hit_time_ms:.3f}ms, Miss time: {stats.avg_miss_time_ms:.3f}ms")

        return stats

    def benchmark_compression(
        self,
        data_sizes_kb: List[int] = None,
        compression_levels: List[int] = None
    ) -> List[CompressionStats]:
        """
        Benchmark compression ratios and performance.

        Args:
            data_sizes_kb: List of data sizes to test (in KB)
            compression_levels: List of gzip compression levels (1-9)

        Returns:
            List of CompressionStats for each test
        """
        if data_sizes_kb is None:
            data_sizes_kb = [10, 100, 1000, 5000]
        if compression_levels is None:
            compression_levels = [1, 6, 9]  # Fast, default, max

        logger.info(f"Benchmarking compression for sizes: {data_sizes_kb} KB")
        all_stats = []

        for size_kb in data_sizes_kb:
            # Generate test data - mix of random and repetitive for realistic compression
            random_part = os.urandom(size_kb * 512)  # Half random
            repetitive_part = (b"repetitive data pattern " * (size_kb * 512 // 25))[:size_kb * 512]
            test_data = random_part + repetitive_part

            for level in compression_levels:
                stats = CompressionStats()
                stats.original_size_bytes = len(test_data)

                # Benchmark compression
                start = time.perf_counter()
                compressed = gzip.compress(test_data, compresslevel=level)
                stats.compression_time_ms = (time.perf_counter() - start) * 1000
                stats.compressed_size_bytes = len(compressed)

                # Benchmark decompression
                start = time.perf_counter()
                _ = gzip.decompress(compressed)
                stats.decompression_time_ms = (time.perf_counter() - start) * 1000

                stats.calculate_ratio()
                all_stats.append(stats)

                logger.debug(
                    f"  {size_kb}KB level={level}: ratio={stats.compression_ratio:.2f}x, "
                    f"savings={stats.space_savings_percent:.1f}%"
                )

        # Create summary result
        avg_ratio = sum(s.compression_ratio for s in all_stats) / len(all_stats)
        avg_savings = sum(s.space_savings_percent for s in all_stats) / len(all_stats)

        result = StorageBenchmarkResult(
            name="compression",
            duration_seconds=sum(s.compression_time_ms + s.decompression_time_ms for s in all_stats) / 1000,
            operations_count=len(all_stats),
            ops_per_second=0,
            data_size_mb=sum(s.original_size_bytes for s in all_stats) / 1024 / 1024,
            metadata={
                "avg_compression_ratio": avg_ratio,
                "avg_space_savings_percent": avg_savings,
                "tested_sizes_kb": data_sizes_kb,
                "tested_levels": compression_levels,
            }
        )
        self.results.append(result)

        logger.info(f"Compression: avg ratio={avg_ratio:.2f}x, avg savings={avg_savings:.1f}%")

        return all_stats

    def benchmark_deduplication(
        self,
        num_files: int = 50,
        duplicate_ratio: float = 0.3
    ) -> DeduplicationStats:
        """
        Benchmark deduplication savings.

        Args:
            num_files: Number of files to create
            duplicate_ratio: Ratio of duplicate files (0.0 to 1.0)

        Returns:
            DeduplicationStats with savings analysis
        """
        logger.info(f"Benchmarking deduplication ({num_files} files, {duplicate_ratio*100:.0f}% duplicates)")

        stats = DeduplicationStats()
        file_hashes = {}
        dedup_dir = self.output_dir / "dedup_test"
        dedup_dir.mkdir(exist_ok=True)

        # Generate unique files
        num_unique = int(num_files * (1 - duplicate_ratio))
        unique_data = []
        for i in range(num_unique):
            data = self._generate_random_data(random.randint(10, 100))
            unique_data.append(data)

        # Create files (some unique, some duplicates)
        for i in range(num_files):
            if i < num_unique:
                data = unique_data[i]
            else:
                # Create duplicate
                data = random.choice(unique_data)

            filepath = dedup_dir / f"file_{i}.bin"
            with open(filepath, 'wb') as f:
                f.write(data)

            # Track hash
            file_hash = hashlib.md5(data).hexdigest()
            if file_hash not in file_hashes:
                file_hashes[file_hash] = {"size": len(data), "count": 1}
                stats.unique_files += 1
                stats.unique_size_bytes += len(data)
            else:
                file_hashes[file_hash]["count"] += 1
                stats.duplicate_files += 1

            stats.total_files += 1
            stats.total_size_bytes += len(data)

        stats.calculate_savings()

        result = StorageBenchmarkResult(
            name="deduplication",
            duration_seconds=0,
            operations_count=num_files,
            ops_per_second=0,
            data_size_mb=stats.total_size_bytes / 1024 / 1024,
            metadata={
                "unique_files": stats.unique_files,
                "duplicate_files": stats.duplicate_files,
                "space_saved_percent": stats.space_saved_percent,
            }
        )
        self.results.append(result)

        logger.info(
            f"Deduplication: {stats.duplicate_files}/{stats.total_files} duplicates, "
            f"{stats.space_saved_percent:.1f}% space saved"
        )

        # Cleanup
        shutil.rmtree(dedup_dir, ignore_errors=True)

        return stats

    def benchmark_read_write_performance(
        self,
        file_sizes_mb: List[float] = None,
        iterations: int = 3
    ) -> Dict[str, StorageBenchmarkResult]:
        """
        Benchmark file read/write performance.

        Args:
            file_sizes_mb: List of file sizes to test
            iterations: Number of iterations per size

        Returns:
            Dictionary with read and write benchmark results
        """
        if file_sizes_mb is None:
            file_sizes_mb = [0.1, 1.0, 10.0]

        logger.info(f"Benchmarking read/write for sizes: {file_sizes_mb} MB")
        results = {}

        write_times = []
        read_times = []
        total_size = 0

        for size_mb in file_sizes_mb:
            for _ in range(iterations):
                # Write benchmark
                data = self._generate_random_data(int(size_mb * 1024))
                filepath = str(self.output_dir / f"rw_test_{size_mb}mb.bin")

                start = time.perf_counter()
                with open(filepath, 'wb') as f:
                    f.write(data)
                write_time = time.perf_counter() - start
                write_times.append((size_mb, write_time))

                # Read benchmark
                start = time.perf_counter()
                with open(filepath, 'rb') as f:
                    _ = f.read()
                read_time = time.perf_counter() - start
                read_times.append((size_mb, read_time))

                total_size += size_mb

                # Cleanup
                os.remove(filepath)

        # Calculate write performance
        total_write_time = sum(t for _, t in write_times)
        write_throughput = total_size / total_write_time if total_write_time > 0 else 0

        results["write"] = StorageBenchmarkResult(
            name="file_write",
            duration_seconds=total_write_time,
            operations_count=len(write_times),
            ops_per_second=len(write_times) / total_write_time if total_write_time > 0 else 0,
            data_size_mb=total_size,
            metadata={
                "throughput_mb_per_sec": write_throughput,
                "avg_write_time_sec": total_write_time / len(write_times),
            }
        )
        self.results.append(results["write"])

        # Calculate read performance
        total_read_time = sum(t for _, t in read_times)
        read_throughput = total_size / total_read_time if total_read_time > 0 else 0

        results["read"] = StorageBenchmarkResult(
            name="file_read",
            duration_seconds=total_read_time,
            operations_count=len(read_times),
            ops_per_second=len(read_times) / total_read_time if total_read_time > 0 else 0,
            data_size_mb=total_size,
            metadata={
                "throughput_mb_per_sec": read_throughput,
                "avg_read_time_sec": total_read_time / len(read_times),
            }
        )
        self.results.append(results["read"])

        logger.info(f"Write throughput: {write_throughput:.2f} MB/s")
        logger.info(f"Read throughput: {read_throughput:.2f} MB/s")

        return results

    def benchmark_json_serialization(
        self,
        num_records: int = 1000,
        iterations: int = 5
    ) -> StorageBenchmarkResult:
        """
        Benchmark JSON serialization/deserialization performance.

        Args:
            num_records: Number of records in test data
            iterations: Number of test iterations

        Returns:
            StorageBenchmarkResult with serialization stats
        """
        logger.info(f"Benchmarking JSON serialization ({num_records} records)")

        # Generate test data
        test_data = [
            {
                "id": i,
                "name": f"item_{i}",
                "description": "".join(random.choices(string.ascii_letters, k=100)),
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "score": random.random(),
                    "tags": [f"tag_{j}" for j in range(5)],
                },
            }
            for i in range(num_records)
        ]

        serialize_times = []
        deserialize_times = []
        data_size = 0

        for _ in range(iterations):
            # Serialize
            start = time.perf_counter()
            json_str = json.dumps(test_data)
            serialize_time = time.perf_counter() - start
            serialize_times.append(serialize_time)
            data_size = len(json_str)

            # Deserialize
            start = time.perf_counter()
            _ = json.loads(json_str)
            deserialize_time = time.perf_counter() - start
            deserialize_times.append(deserialize_time)

        avg_serialize = sum(serialize_times) / len(serialize_times)
        avg_deserialize = sum(deserialize_times) / len(deserialize_times)

        result = StorageBenchmarkResult(
            name="json_serialization",
            duration_seconds=sum(serialize_times) + sum(deserialize_times),
            operations_count=iterations * 2,
            ops_per_second=0,
            data_size_mb=data_size / 1024 / 1024,
            metadata={
                "avg_serialize_time_ms": avg_serialize * 1000,
                "avg_deserialize_time_ms": avg_deserialize * 1000,
                "records_count": num_records,
            }
        )
        self.results.append(result)

        logger.info(
            f"JSON: serialize={avg_serialize*1000:.2f}ms, "
            f"deserialize={avg_deserialize*1000:.2f}ms"
        )

        return result

    def benchmark_stock_cache_simulation(
        self,
        num_queries: int = 100,
        hit_ratio: float = 0.7
    ) -> CacheStats:
        """
        Simulate stock footage cache performance.

        Args:
            num_queries: Number of cache queries to simulate
            hit_ratio: Expected cache hit ratio

        Returns:
            CacheStats with simulation results
        """
        logger.info(f"Simulating stock cache ({num_queries} queries, {hit_ratio*100:.0f}% hit ratio)")

        stats = CacheStats()
        hit_times = []
        miss_times = []

        # Simulate different query patterns
        queries = [
            "money", "finance", "success", "business", "technology",
            "nature", "city", "people", "abstract", "minimal",
        ]

        # Pre-populate cache
        cached_queries = queries[:int(len(queries) * hit_ratio)]
        cache = {q: f"cached_video_{i}.mp4" for i, q in enumerate(cached_queries)}

        for _ in range(num_queries):
            query = random.choice(queries)
            stats.total_requests += 1

            start = time.perf_counter()

            if query in cache:
                # Cache hit - fast retrieval
                _ = cache[query]
                duration = (time.perf_counter() - start) * 1000
                hit_times.append(duration)
                stats.cache_hits += 1
            else:
                # Cache miss - simulate download (slow)
                time.sleep(0.01)  # Simulate network delay
                cache[query] = f"downloaded_video_{query}.mp4"
                duration = (time.perf_counter() - start) * 1000
                miss_times.append(duration)
                stats.cache_misses += 1

        stats.calculate_hit_rate()
        stats.avg_hit_time_ms = sum(hit_times) / len(hit_times) if hit_times else 0
        stats.avg_miss_time_ms = sum(miss_times) / len(miss_times) if miss_times else 0

        result = StorageBenchmarkResult(
            name="stock_cache_simulation",
            duration_seconds=sum(hit_times + miss_times) / 1000,
            operations_count=num_queries,
            ops_per_second=0,
            data_size_mb=0,
            metadata={
                "hit_rate": stats.hit_rate,
                "avg_hit_time_ms": stats.avg_hit_time_ms,
                "avg_miss_time_ms": stats.avg_miss_time_ms,
                "time_saved_ms": stats.cache_hits * (stats.avg_miss_time_ms - stats.avg_hit_time_ms),
            }
        )
        self.results.append(result)

        time_saved = stats.cache_hits * (stats.avg_miss_time_ms - stats.avg_hit_time_ms) / 1000
        logger.info(f"Stock cache: {stats.hit_rate:.1f}% hit rate, {time_saved:.2f}s saved")

        return stats

    def generate_report(self) -> str:
        """
        Generate markdown storage benchmark report.

        Returns:
            Markdown-formatted report string
        """
        lines = [
            "# Storage Benchmark Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            f"Total benchmarks run: {len(self.results)}",
            "",
            "## Results",
            "",
            "| Benchmark | Duration | Operations | Data Size | Key Metric |",
            "|-----------|----------|------------|-----------|------------|",
        ]

        for r in self.results:
            key_metric = ""
            if "hit_rate" in r.metadata:
                key_metric = f"{r.metadata['hit_rate']:.1f}% hit rate"
            elif "throughput_mb_per_sec" in r.metadata:
                key_metric = f"{r.metadata['throughput_mb_per_sec']:.2f} MB/s"
            elif "avg_compression_ratio" in r.metadata:
                key_metric = f"{r.metadata['avg_compression_ratio']:.2f}x ratio"
            elif "space_saved_percent" in r.metadata:
                key_metric = f"{r.metadata['space_saved_percent']:.1f}% saved"

            lines.append(
                f"| {r.name} | {r.duration_seconds:.3f}s | {r.operations_count} | "
                f"{r.data_size_mb:.2f}MB | {key_metric} |"
            )

        # Recommendations
        lines.extend([
            "",
            "## Recommendations",
            "",
            "1. **Enable caching:** Stock footage caching can save 70%+ download time",
            "2. **Use compression:** gzip level 6 offers best speed/ratio balance",
            "3. **Deduplicate:** Regular deduplication can save 20-40% storage",
            "4. **JSON optimization:** Use `orjson` for 5-10x faster serialization",
            "",
        ])

        return "\n".join(lines)

    def cleanup(self):
        """Remove benchmark output files."""
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir, ignore_errors=True)
            logger.info(f"Cleaned up benchmark output: {self.output_dir}")


def run_storage_benchmarks():
    """Run all storage benchmarks."""
    logger.info("=" * 60)
    logger.info("  RUNNING STORAGE BENCHMARKS")
    logger.info("=" * 60)

    output_dir = Path("output/benchmarks/storage")
    output_dir.mkdir(parents=True, exist_ok=True)

    bench = StorageBenchmark(str(output_dir))

    # Run benchmarks
    logger.info("\n--- Cache Performance ---")
    bench.benchmark_cache_performance(num_operations=500)

    logger.info("\n--- Compression ---")
    bench.benchmark_compression([10, 100, 500], [1, 6, 9])

    logger.info("\n--- Deduplication ---")
    bench.benchmark_deduplication(num_files=30, duplicate_ratio=0.4)

    logger.info("\n--- Read/Write Performance ---")
    bench.benchmark_read_write_performance([0.1, 1.0, 5.0])

    logger.info("\n--- JSON Serialization ---")
    bench.benchmark_json_serialization(num_records=500)

    logger.info("\n--- Stock Cache Simulation ---")
    bench.benchmark_stock_cache_simulation(num_queries=100)

    # Generate report
    report = bench.generate_report()

    # Save report
    report_file = output_dir / "storage_benchmark_report.md"
    with open(report_file, "w") as f:
        f.write(report)

    logger.success(f"Storage report saved to: {report_file}")
    print(report)

    # Cleanup
    bench.cleanup()

    logger.info("=" * 60)
    logger.info("  STORAGE BENCHMARKS COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_storage_benchmarks()
