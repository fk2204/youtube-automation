"""
Batch Automation Runner

Creates multiple videos across channels efficiently.
Designed to be called by Claude subagents.

Supports parallel processing for 3x throughput increase.

Usage:
    python -m src.automation.batch --videos 3
    python -m src.automation.batch --channel money_blueprints --videos 2
    python -m src.automation.batch --all-channels
    python -m src.automation.batch --parallel 3 --channels money_blueprints,mind_unlocked --videos 2
"""

import os
import sys
import json
import argparse
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / "config" / ".env")

from loguru import logger
import yaml

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")


# Progress tracking
@dataclass
class JobProgress:
    """Track progress of a single video job."""
    job_id: str
    channel_id: str
    status: str = "pending"  # pending, running, completed, failed
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict] = None
    error: Optional[str] = None

    def duration_seconds(self) -> float:
        """Get job duration in seconds."""
        if not self.started_at:
            return 0
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()


@dataclass
class BatchProgress:
    """Track progress of entire batch."""
    total_jobs: int
    jobs: Dict[str, JobProgress] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add_job(self, job: JobProgress):
        """Add a job to tracking."""
        with self._lock:
            self.jobs[job.job_id] = job

    def update_job(self, job_id: str, **kwargs):
        """Update job status."""
        with self._lock:
            if job_id in self.jobs:
                for key, value in kwargs.items():
                    setattr(self.jobs[job_id], key, value)

    @property
    def pending(self) -> int:
        return sum(1 for j in self.jobs.values() if j.status == "pending")

    @property
    def running(self) -> int:
        return sum(1 for j in self.jobs.values() if j.status == "running")

    @property
    def completed(self) -> int:
        return sum(1 for j in self.jobs.values() if j.status == "completed")

    @property
    def failed(self) -> int:
        return sum(1 for j in self.jobs.values() if j.status == "failed")

    def progress_str(self) -> str:
        """Get progress string for logging."""
        return f"[{self.completed + self.failed}/{self.total_jobs}] " \
               f"OK:{self.completed} FAIL:{self.failed} RUN:{self.running} WAIT:{self.pending}"

    def summary(self) -> Dict:
        """Get summary statistics."""
        return {
            "total": self.total_jobs,
            "completed": self.completed,
            "failed": self.failed,
            "success_rate": f"{(self.completed / self.total_jobs * 100):.1f}%" if self.total_jobs > 0 else "0%",
            "total_duration_seconds": (self.completed_at - self.started_at).total_seconds() if self.started_at and self.completed_at else 0,
            "avg_duration_seconds": sum(j.duration_seconds() for j in self.jobs.values() if j.status in ["completed", "failed"]) / max(self.completed + self.failed, 1)
        }


def get_channels() -> List[Dict]:
    """Load enabled channels from config."""
    with open(PROJECT_ROOT / "config" / "channels.yaml") as f:
        config = yaml.safe_load(f)
    return [ch for ch in config["channels"] if ch.get("enabled", True)]


def create_video_for_channel(channel_id: str, upload: bool = True) -> Dict:
    """Create and optionally upload a video for a channel."""
    from src.automation.runner import task_full_pipeline, task_upload

    logger.info(f"\n{'='*50}")
    logger.info(f"Creating video for: {channel_id}")
    logger.info(f"{'='*50}")

    # Run pipeline
    result = task_full_pipeline(channel_id)

    if not result["success"]:
        logger.error(f"Pipeline failed: {result.get('error')}")
        return result

    # Upload if requested
    if upload:
        logger.info("\nUploading to YouTube...")
        upload_result = task_upload(
            video_file=result["results"]["video_file"],
            channel_id=channel_id,
            title=result["results"]["title"],
            description=result["results"]["description"],
            tags=result["results"]["tags"]
        )

        result["results"]["upload"] = upload_result
        if upload_result["success"]:
            result["results"]["video_url"] = upload_result["video_url"]
            logger.success(f"Uploaded: {upload_result['video_url']}")

    return result


def run_batch(
    channels: List[str] = None,
    videos_per_channel: int = 1,
    upload: bool = True
) -> Dict:
    """
    Run batch video creation.

    Args:
        channels: List of channel IDs (None = all enabled)
        videos_per_channel: Videos to create per channel
        upload: Whether to upload to YouTube

    Returns:
        Summary of results
    """
    all_channels = get_channels()

    if channels:
        target_channels = [ch for ch in all_channels if ch["id"] in channels]
    else:
        target_channels = all_channels

    logger.info(f"\n{'='*60}")
    logger.info(f"BATCH VIDEO CREATION")
    logger.info(f"Channels: {len(target_channels)}")
    logger.info(f"Videos per channel: {videos_per_channel}")
    logger.info(f"Total videos: {len(target_channels) * videos_per_channel}")
    logger.info(f"Upload: {'Yes' if upload else 'No'}")
    logger.info(f"{'='*60}\n")

    results = {
        "started_at": datetime.now().isoformat(),
        "channels": [],
        "total_videos": 0,
        "successful": 0,
        "failed": 0,
        "uploaded": 0
    }

    for channel in target_channels:
        channel_id = channel["id"]
        channel_results = {
            "channel": channel_id,
            "name": channel["name"],
            "videos": []
        }

        for i in range(videos_per_channel):
            logger.info(f"\n[{channel_id}] Video {i+1}/{videos_per_channel}")

            result = create_video_for_channel(channel_id, upload=upload)
            results["total_videos"] += 1

            video_info = {
                "success": result["success"],
                "title": result.get("results", {}).get("title", "Unknown"),
                "video_file": result.get("results", {}).get("video_file"),
                "video_url": result.get("results", {}).get("video_url")
            }

            if result["success"]:
                results["successful"] += 1
                if result.get("results", {}).get("video_url"):
                    results["uploaded"] += 1
            else:
                results["failed"] += 1
                video_info["error"] = result.get("error")

            channel_results["videos"].append(video_info)

        results["channels"].append(channel_results)

    results["completed_at"] = datetime.now().isoformat()

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("BATCH COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total: {results['total_videos']}")
    logger.info(f"Successful: {results['successful']}")
    logger.info(f"Failed: {results['failed']}")
    logger.info(f"Uploaded: {results['uploaded']}")

    # List uploaded videos
    if results["uploaded"] > 0:
        logger.info("\nUploaded Videos:")
        for ch in results["channels"]:
            for v in ch["videos"]:
                if v.get("video_url"):
                    logger.info(f"  {v['video_url']}")
                    logger.info(f"    \"{v['title']}\"")

    return results


def _run_single_video_job(
    channel_id: str,
    job_id: str,
    progress: BatchProgress,
    upload: bool = True
) -> Tuple[str, Dict]:
    """
    Run a single video job. Called by executor threads.

    Returns:
        Tuple of (job_id, result_dict)
    """
    from src.automation.runner import task_full_with_upload, task_full_pipeline

    # Update status to running
    progress.update_job(job_id, status="running", started_at=datetime.now())
    logger.info(f"{progress.progress_str()} Starting job {job_id} for {channel_id}")

    try:
        # Run the pipeline
        if upload:
            result = task_full_with_upload(channel_id)
        else:
            result = task_full_pipeline(channel_id)

        # Update progress
        if result.get("success"):
            progress.update_job(
                job_id,
                status="completed",
                completed_at=datetime.now(),
                result=result
            )
            video_url = result.get("results", {}).get("video_url", "N/A")
            title = result.get("results", {}).get("title", "Unknown")
            logger.success(f"{progress.progress_str()} Completed {job_id}: {title}")
            if video_url != "N/A":
                logger.info(f"  -> {video_url}")
        else:
            error_msg = result.get("error", "Unknown error")
            progress.update_job(
                job_id,
                status="failed",
                completed_at=datetime.now(),
                error=error_msg,
                result=result
            )
            logger.error(f"{progress.progress_str()} Failed {job_id}: {error_msg}")

        return (job_id, result)

    except Exception as e:
        error_msg = str(e)
        progress.update_job(
            job_id,
            status="failed",
            completed_at=datetime.now(),
            error=error_msg
        )
        logger.error(f"{progress.progress_str()} Exception in {job_id}: {error_msg}")
        return (job_id, {"success": False, "error": error_msg})


def run_batch_parallel(
    channels: List[str] = None,
    videos_per_channel: int = 1,
    upload: bool = True,
    max_workers: int = 3
) -> Dict:
    """
    Run batch video creation with parallel processing.

    Uses ThreadPoolExecutor to process multiple videos concurrently,
    achieving up to 3x throughput increase.

    Args:
        channels: List of channel IDs (None = all enabled)
        videos_per_channel: Videos to create per channel
        upload: Whether to upload to YouTube
        max_workers: Maximum parallel workers (default: 3)

    Returns:
        Summary of results with detailed job information
    """
    all_channels_config = get_channels()

    if channels:
        target_channels = [ch for ch in all_channels_config if ch["id"] in channels]
    else:
        target_channels = all_channels_config

    total_videos = len(target_channels) * videos_per_channel

    logger.info(f"\n{'='*60}")
    logger.info(f"PARALLEL BATCH VIDEO CREATION")
    logger.info(f"{'='*60}")
    logger.info(f"Channels: {len(target_channels)}")
    logger.info(f"Videos per channel: {videos_per_channel}")
    logger.info(f"Total videos: {total_videos}")
    logger.info(f"Parallel workers: {max_workers}")
    logger.info(f"Upload: {'Yes' if upload else 'No'}")
    logger.info(f"{'='*60}\n")

    # Initialize progress tracking
    progress = BatchProgress(total_jobs=total_videos)
    progress.started_at = datetime.now()

    # Create job list
    jobs = []
    job_counter = 0
    for channel in target_channels:
        channel_id = channel["id"]
        for i in range(videos_per_channel):
            job_id = f"{channel_id}_{i+1}"
            job = JobProgress(job_id=job_id, channel_id=channel_id)
            progress.add_job(job)
            jobs.append((channel_id, job_id))
            job_counter += 1

    # Execute in parallel using ThreadPoolExecutor
    results_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_job = {
            executor.submit(
                _run_single_video_job,
                channel_id,
                job_id,
                progress,
                upload
            ): job_id
            for channel_id, job_id in jobs
        }

        # Collect results as they complete
        for future in as_completed(future_to_job):
            job_id = future_to_job[future]
            try:
                result_job_id, result = future.result()
                results_map[result_job_id] = result
            except Exception as e:
                logger.error(f"Exception getting result for {job_id}: {e}")
                results_map[job_id] = {"success": False, "error": str(e)}

    progress.completed_at = datetime.now()

    # Build final results
    results = {
        "started_at": progress.started_at.isoformat(),
        "completed_at": progress.completed_at.isoformat(),
        "parallel_workers": max_workers,
        "statistics": progress.summary(),
        "channels": [],
        "total_videos": total_videos,
        "successful": progress.completed,
        "failed": progress.failed,
        "uploaded": 0
    }

    # Organize results by channel
    for channel in target_channels:
        channel_id = channel["id"]
        channel_results = {
            "channel": channel_id,
            "name": channel["name"],
            "videos": []
        }

        for i in range(videos_per_channel):
            job_id = f"{channel_id}_{i+1}"
            job = progress.jobs.get(job_id)
            result = results_map.get(job_id, {})

            video_info = {
                "job_id": job_id,
                "success": result.get("success", False),
                "title": result.get("results", {}).get("title", "Unknown"),
                "video_file": result.get("results", {}).get("video_file"),
                "video_url": result.get("results", {}).get("video_url"),
                "duration_seconds": job.duration_seconds() if job else 0
            }

            if result.get("success") and result.get("results", {}).get("video_url"):
                results["uploaded"] += 1

            if not result.get("success"):
                video_info["error"] = result.get("error") or (job.error if job else "Unknown error")

            channel_results["videos"].append(video_info)

        results["channels"].append(channel_results)

    # Summary
    duration = (progress.completed_at - progress.started_at).total_seconds()
    logger.info(f"\n{'='*60}")
    logger.info(f"PARALLEL BATCH COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total: {results['total_videos']}")
    logger.info(f"Successful: {results['successful']}")
    logger.info(f"Failed: {results['failed']}")
    logger.info(f"Uploaded: {results['uploaded']}")
    logger.info(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")
    logger.info(f"Avg per video: {duration/max(total_videos, 1):.1f}s")
    logger.info(f"Throughput: {total_videos/max(duration/60, 0.1):.1f} videos/min")

    # List uploaded videos
    if results["uploaded"] > 0:
        logger.info("\nUploaded Videos:")
        for ch in results["channels"]:
            for v in ch["videos"]:
                if v.get("video_url"):
                    logger.info(f"  {v['video_url']}")
                    logger.info(f"    \"{v['title']}\"")

    # List failed jobs
    if results["failed"] > 0:
        logger.warning("\nFailed Jobs:")
        for ch in results["channels"]:
            for v in ch["videos"]:
                if not v.get("success"):
                    logger.warning(f"  {v['job_id']}: {v.get('error', 'Unknown error')}")

    return results


async def batch_create_videos(
    channels: List[str],
    count: int,
    max_workers: int = 3,
    upload: bool = True
) -> List[Dict]:
    """
    Process multiple videos in parallel using asyncio + ThreadPoolExecutor.

    This is the async interface for parallel batch processing.

    Args:
        channels: List of channel IDs to process
        count: Number of videos per channel
        max_workers: Maximum concurrent workers (default: 3)
        upload: Whether to upload to YouTube (default: True)

    Returns:
        List of results for each job (success or exception)
    """
    from src.automation.runner import task_full_with_upload, task_full_pipeline

    executor = ThreadPoolExecutor(max_workers=max_workers)
    loop = asyncio.get_event_loop()

    # Create task function
    def run_video_task(channel_id: str) -> Dict:
        """Run single video task in thread."""
        try:
            if upload:
                return task_full_with_upload(channel_id)
            else:
                return task_full_pipeline(channel_id)
        except Exception as e:
            return {"success": False, "error": str(e), "channel": channel_id}

    # Build task list
    tasks = []
    for channel in channels:
        for _ in range(count):
            task = loop.run_in_executor(executor, run_video_task, channel)
            tasks.append(task)

    # Run all tasks concurrently and gather results
    # return_exceptions=True ensures one failure doesn't stop others
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Clean up executor
    executor.shutdown(wait=False)

    # Convert exceptions to error dicts
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            processed_results.append({
                "success": False,
                "error": str(result),
                "exception_type": type(result).__name__
            })
        else:
            processed_results.append(result)

    return processed_results


def main():
    parser = argparse.ArgumentParser(description="Batch Video Creation")
    parser.add_argument("--channel", "-c", help="Specific channel ID")
    parser.add_argument("--channels", help="Comma-separated list of channel IDs")
    parser.add_argument("--videos", "-v", type=int, default=1, help="Videos per channel")
    parser.add_argument("--all-channels", "-a", action="store_true", help="Run for all channels")
    parser.add_argument("--no-upload", action="store_true", help="Don't upload to YouTube")
    parser.add_argument("--parallel", "-p", type=int, default=0, help="Parallel workers (0=sequential)")
    parser.add_argument("--output", "-o", help="Save results to JSON file")

    args = parser.parse_args()

    # Determine channels
    if args.channels:
        # Support comma-separated list
        channels = [ch.strip() for ch in args.channels.split(",")]
    elif args.channel:
        channels = [args.channel]
    elif args.all_channels:
        channels = None  # All channels
    else:
        channels = ["money_blueprints"]  # Default

    # Run batch (parallel or sequential)
    if args.parallel > 0:
        # Parallel processing
        results = run_batch_parallel(
            channels=channels,
            videos_per_channel=args.videos,
            upload=not args.no_upload,
            max_workers=args.parallel
        )
    else:
        # Sequential processing (original behavior)
        results = run_batch(
            channels=channels,
            videos_per_channel=args.videos,
            upload=not args.no_upload
        )

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nResults saved: {args.output}")

    print("\n" + json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
