"""
Parallel Stock Footage Downloader

Downloads multiple stock clips simultaneously using ThreadPoolExecutor.
Reduces download time from 30-60 seconds to 10-15 seconds.
"""

import os
import time
import asyncio
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class DownloadTask:
    url: str
    output_path: str
    clip_id: str
    timeout: int = 30


@dataclass
class DownloadResult:
    """Result of a single download operation."""
    clip_id: str
    success: bool
    file_path: Optional[str] = None
    error: Optional[str] = None
    download_time: float = 0.0


@dataclass
class BatchDownloadResult:
    """Result of a batch download operation."""
    total: int
    successful: int
    failed: int
    file_paths: List[str]
    elapsed_time: float


class ParallelDownloader:
    """Download multiple stock clips in parallel."""

    def __init__(self, max_workers: int = 4, cache_dir: str = "cache/stock_footage"):
        self.max_workers = max_workers
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_single(self, task: DownloadTask) -> Optional[str]:
        """Download a single clip."""
        try:
            response = requests.get(task.url, stream=True, timeout=task.timeout)
            response.raise_for_status()

            output_path = Path(task.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            if os.path.getsize(output_path) > 10000:
                logger.debug(f"Downloaded: {task.clip_id}")
                return str(output_path)

        except Exception as e:
            logger.warning(f"Download failed for {task.clip_id}: {e}")

        return None

    def download_batch(self, tasks: List[DownloadTask]) -> List[str]:
        """Download multiple clips in parallel."""
        if not tasks:
            return []

        logger.info(f"Starting parallel download: {len(tasks)} clips, {self.max_workers} workers")
        start_time = time.time()

        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(self.download_single, task): task
                for task in tasks
            }

            for future in as_completed(future_to_task):
                result = future.result()
                if result:
                    results.append(result)

        elapsed = time.time() - start_time
        logger.success(
            f"Parallel download complete: {len(results)}/{len(tasks)} succeeded "
            f"in {elapsed:.1f}s"
        )

        return results

    async def download_batch_async(self, tasks: List[DownloadTask]) -> BatchDownloadResult:
        """Download multiple clips in parallel using asyncio."""
        if not tasks:
            return BatchDownloadResult(total=0, successful=0, failed=0, file_paths=[], elapsed_time=0)

        logger.info(f"Starting async parallel download: {len(tasks)} clips")
        start_time = time.time()

        async def download_one(task: DownloadTask) -> DownloadResult:
            loop = asyncio.get_event_loop()
            start = time.time()
            try:
                result = await loop.run_in_executor(None, self.download_single, task)
                return DownloadResult(
                    clip_id=task.clip_id,
                    success=result is not None,
                    file_path=result,
                    download_time=time.time() - start
                )
            except Exception as e:
                return DownloadResult(
                    clip_id=task.clip_id,
                    success=False,
                    error=str(e),
                    download_time=time.time() - start
                )

        results = await asyncio.gather(*[download_one(t) for t in tasks])

        elapsed = time.time() - start_time
        successful = [r for r in results if r.success]

        logger.success(f"Async download complete: {len(successful)}/{len(tasks)} in {elapsed:.1f}s")

        return BatchDownloadResult(
            total=len(tasks),
            successful=len(successful),
            failed=len(tasks) - len(successful),
            file_paths=[r.file_path for r in successful if r.file_path],
            elapsed_time=elapsed
        )


def get_parallel_downloader(max_workers: int = 4) -> ParallelDownloader:
    """Get a parallel downloader instance."""
    return ParallelDownloader(max_workers=max_workers)
