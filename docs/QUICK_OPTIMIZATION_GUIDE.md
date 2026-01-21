# Quick Optimization Guide

Fast reference for using the new performance features.

---

## Check Your System

```bash
# Check GPU status
python run.py gpu-status

# Run performance benchmark
python run.py benchmark

# Test async downloads
python run.py async-download-test
```

---

## 1. Enable GPU Acceleration (2-3x faster)

**Automatic** - GPU is auto-detected and enabled by default.

### Check if GPU is working:
```bash
python run.py gpu-status
```

### Manual configuration:
Edit `config/performance.yaml`:
```yaml
gpu:
  enabled: true           # Enable GPU
  prefer_quality: false   # false = speed, true = quality
  fallback_to_cpu: true   # Fallback if GPU unavailable
```

### In your code:
```python
from src.content.video_pro import ProVideoGenerator

# GPU enabled by default
generator = ProVideoGenerator()

# Or explicitly control
generator = ProVideoGenerator(
    use_gpu=True,
    prefer_quality=False  # Speed over quality
)
```

---

## 2. Use Async Downloads (5-10x faster)

### For multiple stock clips:

```python
import asyncio
from src.content.stock_footage import StockFootageProvider

async def download_clips():
    provider = StockFootageProvider()

    # Search
    videos = provider.search_videos("finance", count=10)

    # Download asynchronously
    results = await provider.download_videos_async(
        videos,
        output_dir="output/clips",
        max_concurrent=5  # Download 5 at once
    )

asyncio.run(download_clips())
```

### Configuration:
Edit `config/performance.yaml`:
```yaml
async:
  max_concurrent_downloads: 5  # Concurrent downloads
  download_timeout: 60         # Timeout per download
  max_retries: 3               # Retry failed downloads
```

---

## 3. Parallel Video Processing (3-4x faster)

### For creating multiple videos:

```python
from src.content.parallel_processor import ParallelVideoProcessor

# Initialize (auto-detects CPU cores)
processor = ParallelVideoProcessor(max_workers=4)

# Create multiple videos in parallel
results = processor.create_videos_parallel(
    topics=[
        "Passive Income Ideas",
        "Money Saving Tips",
        "Investment Basics"
    ],
    channel_id="money_blueprints",
    niche="finance"
)

# Check results
for result in results:
    print(f"{result.task_id}: {result.success}")
```

### Configuration:
Edit `config/performance.yaml`:
```yaml
parallel:
  max_workers: auto  # auto = CPU cores - 1
  chunk_size: 10     # Process 10 at a time
  enabled: true
```

---

## 4. Combined Example (15x faster)

```python
import asyncio
from src.content.stock_footage import StockFootageProvider
from src.content.parallel_processor import ParallelVideoProcessor

async def fast_video_creation():
    # 1. Download stock footage (async)
    provider = StockFootageProvider()
    videos = provider.search_videos("technology", count=15)
    await provider.download_videos_async(videos, "output/clips")

    # 2. Create videos (parallel + GPU)
    processor = ParallelVideoProcessor(max_workers=4)
    topics = ["AI Trends", "Tech News", "Gadget Reviews"]

    results = processor.create_videos_parallel(
        topics=topics,
        channel_id="tech_channel",
        niche="technology"
    )

    print(f"Created {len(results)} videos")

asyncio.run(fast_video_creation())
```

---

## Configuration Presets

### Fast Mode (Bulk Processing)
```yaml
active_profile: "production"

gpu:
  prefer_quality: false  # Speed over quality

async:
  max_concurrent_downloads: 10

parallel:
  max_workers: 6
```

### Quality Mode (Final Videos)
```yaml
active_profile: "high_quality"

gpu:
  prefer_quality: true  # Quality over speed

async:
  max_concurrent_downloads: 3

parallel:
  max_workers: 2
```

---

## Benchmarking

```bash
# Quick 30-second test
python run.py benchmark

# Comprehensive 60-second test
python run.py benchmark 60
```

Expected results:
- **NVIDIA GPU**: 2.5-3x speedup
- **AMD GPU**: 2-2.5x speedup
- **Intel GPU**: 1.5-2x speedup

---

## Troubleshooting

### GPU not working?
1. Check: `python run.py gpu-status`
2. Update GPU drivers
3. Ensure FFmpeg has GPU support
4. Set `fallback_to_cpu: true`

### Downloads failing?
1. Reduce `max_concurrent_downloads: 3`
2. Increase `download_timeout: 120`
3. Check API keys configured

### Out of memory?
1. Reduce `max_workers: 2`
2. Lower `max_video_cache_mb: 250`
3. Enable `cleanup_temp_files: true`

---

## Performance Cheat Sheet

| Feature | Command | Speedup | When to Use |
|---------|---------|---------|-------------|
| GPU | Auto-enabled | 2-3x | All video encoding |
| Async Downloads | `download_videos_async()` | 5-10x | 5+ stock clips |
| Parallel Processing | `ParallelVideoProcessor()` | 3-4x | Multiple videos |
| Combined | All together | 15x+ | Batch operations |

---

## Quick Tips

1. **GPU is automatic** - Just update and it works
2. **Async for downloads** - 5+ clips? Use async
3. **Parallel for batches** - Multiple videos? Go parallel
4. **Monitor with benchmarks** - Test your actual speedup
5. **Adjust config** - Tune for your hardware

---

## CLI Commands

```bash
# Diagnostic
python run.py gpu-status
python run.py benchmark
python run.py async-download-test

# Normal operations (now faster!)
python run.py video money_blueprints
python run.py short mind_unlocked
python run.py batch 10
```

---

## File Locations

- **Config**: `config/performance.yaml`
- **Full docs**: `docs/PERFORMANCE_OPTIMIZATION.md`
- **Summary**: `OPTIMIZATION_SUMMARY.md`
- **This guide**: `docs/QUICK_OPTIMIZATION_GUIDE.md`

---

## Need Help?

1. Read full docs: `docs/PERFORMANCE_OPTIMIZATION.md`
2. Check config: `config/performance.yaml`
3. Run diagnostics: `python run.py gpu-status`
4. View benchmark: `python run.py benchmark`

---

**Last Updated**: 2026-01-20
