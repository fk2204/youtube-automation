"""
Performance Benchmark Script

Measures video generation pipeline performance to track optimizations.

Usage:
    python scripts/benchmark_performance.py --video-type regular
    python scripts/benchmark_performance.py --video-type short
"""

import time
import argparse
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger


def benchmark_video_generation(video_type: str = "regular"):
    """Benchmark full video generation pipeline."""

    logger.info(f"Starting benchmark for {video_type} video...")
    logger.info("=" * 60)

    results = {}

    # Stage 1: Import timing
    start = time.time()
    try:
        from src.content.video_fast import FastVideoGenerator
        from src.content.audio_processor import AudioProcessor
        results['import_time'] = time.time() - start
        logger.info(f"Import time: {results['import_time']:.2f}s")
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        return

    # Stage 2: Audio processor initialization
    start = time.time()
    processor = AudioProcessor()
    results['audio_init'] = time.time() - start
    logger.info(f"Audio processor init: {results['audio_init']:.2f}s")

    # Stage 3: Video generator initialization
    start = time.time()
    generator = FastVideoGenerator(content_type=video_type)
    results['video_init'] = time.time() - start
    logger.info(f"Video generator init: {results['video_init']:.2f}s")

    # Summary
    total = sum(results.values())
    logger.info("=" * 60)
    logger.success(f"Benchmark Results ({video_type}):")
    for key, value in results.items():
        pct = (value / total) * 100 if total > 0 else 0
        logger.info(f"  {key}: {value:.2f}s ({pct:.1f}%)")
    logger.info(f"  Total initialization: {total:.2f}s")
    logger.info("=" * 60)

    # Check encoding preset
    if hasattr(generator, 'encoding_preset'):
        logger.info(f"Encoding preset: {generator.encoding_preset}")
        if generator.encoding_preset == "medium":
            logger.success("✓ Optimized preset detected (medium)")
        elif generator.encoding_preset == "slow":
            logger.warning("⚠ Slow preset detected - consider 'medium' for 3x speedup")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark video generation pipeline")
    parser.add_argument(
        "--video-type",
        choices=["regular", "short", "shorts"],
        default="regular",
        help="Type of video to benchmark"
    )
    args = parser.parse_args()

    # Normalize shorts
    video_type = "shorts" if args.video_type in ["short", "shorts"] else "regular"

    benchmark_video_generation(video_type)
