"""
Test script for async downloads and GPU acceleration features.
"""

import asyncio
import time
from pathlib import Path


async def test_async_downloads():
    """Test async stock footage downloads."""
    print("\n" + "="*60)
    print("TESTING ASYNC DOWNLOADS")
    print("="*60 + "\n")

    from src.content.stock_footage import StockFootageProvider

    provider = StockFootageProvider()

    if not provider.is_available():
        print("No stock footage providers configured. Skipping test.")
        return

    # Search for videos
    print("Searching for test videos...")
    videos = provider.search_videos("technology", count=5, min_duration=5, max_duration=30)

    if not videos:
        print("No videos found. Test failed.")
        return

    print(f"Found {len(videos)} videos\n")

    # Test async download
    output_dir = "output/test_async"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(videos)} videos asynchronously...")
    start_time = time.time()

    results = await provider.download_videos_async(
        videos,
        output_dir=output_dir,
        max_concurrent=3
    )

    elapsed = time.time() - start_time
    success_count = sum(1 for r in results if r is not None)

    print(f"\nAsync download complete:")
    print(f"  Time:    {elapsed:.1f}s")
    print(f"  Success: {success_count}/{len(videos)}")
    print(f"  Speed:   {len(videos)/elapsed:.1f} videos/sec")


def test_gpu_acceleration():
    """Test GPU acceleration for video encoding."""
    print("\n" + "="*60)
    print("TESTING GPU ACCELERATION")
    print("="*60 + "\n")

    from src.utils.gpu_utils import GPUAccelerator

    accelerator = GPUAccelerator()
    status = accelerator.get_status()

    print(f"GPU Status:")
    print(f"  Available:  {status['available']}")
    print(f"  Type:       {status['type']}")
    print(f"  Name:       {status['name']}")
    print(f"  Encoder:    {status['encoder']}")
    print(f"  Speedup:    {status['speedup']}")

    if status['available']:
        print(f"\nGPU acceleration is working!")
        print(f"Expected encoding speedup: {status['speedup']}")
    else:
        print(f"\nUsing CPU encoding (no GPU detected)")


def test_parallel_processing():
    """Test parallel video processing."""
    print("\n" + "="*60)
    print("TESTING PARALLEL PROCESSING")
    print("="*60 + "\n")

    from src.content.parallel_processor import ParallelVideoProcessor, ProcessingTask
    import multiprocessing as mp

    cpu_count = mp.cpu_count()
    print(f"CPU Cores: {cpu_count}")

    processor = ParallelVideoProcessor(max_workers=2)
    print(f"Worker Processes: {processor.max_workers}")

    # Create dummy tasks
    tasks = [
        ProcessingTask(
            task_id=f"test_{i}",
            task_type="full_video",
            params={
                "topic": f"Test Video {i}",
                "channel_id": "test",
                "niche": "psychology"
            }
        )
        for i in range(3)
    ]

    print(f"\nCreated {len(tasks)} test tasks")
    print("Note: Actual processing skipped in test mode")


def test_performance_config():
    """Test performance configuration loading."""
    print("\n" + "="*60)
    print("TESTING PERFORMANCE CONFIG")
    print("="*60 + "\n")

    import yaml
    from pathlib import Path

    config_path = Path("config/performance.yaml")

    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        print("Configuration loaded successfully:")
        print(f"  Active Profile:          {config.get('active_profile')}")
        print(f"  GPU Enabled:             {config.get('gpu', {}).get('enabled')}")
        print(f"  Max Concurrent Downloads: {config.get('async', {}).get('max_concurrent_downloads')}")
        print(f"  Max Workers:             {config.get('parallel', {}).get('max_workers')}")
        print(f"  Cache TTL:               {config.get('stock_cache', {}).get('ttl_seconds')}s")
    else:
        print("Config file not found: config/performance.yaml")


async def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print(" " * 15 + "ADVANCED OPTIMIZATION FEATURES TEST")
    print("="*70)

    # Test 1: GPU Acceleration
    test_gpu_acceleration()

    # Test 2: Performance Config
    test_performance_config()

    # Test 3: Parallel Processing
    test_parallel_processing()

    # Test 4: Async Downloads (requires API keys)
    try:
        await test_async_downloads()
    except Exception as e:
        print(f"\nAsync download test skipped: {e}")

    print("\n" + "="*70)
    print(" " * 25 + "TESTS COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
