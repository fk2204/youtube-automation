"""
Parallel Video Processing

Process multiple videos in parallel using multiprocessing for maximum performance.
Automatically manages worker pool and distributes tasks efficiently.

Usage:
    from src.content.parallel_processor import ParallelVideoProcessor

    processor = ParallelVideoProcessor(max_workers=4)
    results = processor.process_batch([
        {"topic": "video 1", "channel": "money_blueprints"},
        {"topic": "video 2", "channel": "mind_unlocked"}
    ])
"""

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass
from loguru import logger
import time


@dataclass
class ProcessingTask:
    """A video processing task."""
    task_id: str
    task_type: str  # 'full_video', 'short', 'thumbnail', 'subtitle'
    params: Dict[str, Any]
    priority: int = 0  # Higher = processed first


@dataclass
class ProcessingResult:
    """Result of a processing task."""
    task_id: str
    success: bool
    output_path: Optional[str] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0


class ParallelVideoProcessor:
    """
    Process multiple videos in parallel using multiprocessing.

    Features:
    - Automatic worker pool management
    - Task prioritization
    - Progress tracking
    - Error handling and recovery
    - Memory-efficient chunking
    """

    def __init__(self, max_workers: int = None, chunk_size: int = 10):
        """
        Initialize parallel processor.

        Args:
            max_workers: Maximum worker processes (default: CPU count - 1)
            chunk_size: Number of tasks per batch (default: 10)
        """
        if max_workers is None:
            # Leave one CPU free for system
            cpu_count = mp.cpu_count()
            max_workers = max(1, cpu_count - 1)

        self.max_workers = max_workers
        self.chunk_size = chunk_size
        logger.info(f"ParallelVideoProcessor initialized with {max_workers} workers")

    def process_batch(
        self,
        tasks: List[ProcessingTask],
        callback: Optional[Callable] = None
    ) -> List[ProcessingResult]:
        """
        Process multiple tasks in parallel.

        Args:
            tasks: List of ProcessingTask objects
            callback: Optional callback(result) called after each completion

        Returns:
            List of ProcessingResult objects
        """
        if not tasks:
            return []

        # Sort by priority (higher first)
        tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)

        logger.info(f"Processing {len(tasks)} tasks with {self.max_workers} workers...")
        results = []
        start_time = time.time()

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._process_single_task, task): task
                for task in tasks
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_task):
                task = future_to_task[future]

                try:
                    result = future.result()
                    results.append(result)

                    if result.success:
                        logger.success(f"[{completed+1}/{len(tasks)}] Completed: {task.task_id}")
                    else:
                        logger.error(f"[{completed+1}/{len(tasks)}] Failed: {task.task_id} - {result.error}")

                    # Call callback if provided
                    if callback:
                        try:
                            callback(result)
                        except Exception as e:
                            logger.warning(f"Callback error: {e}")

                except Exception as e:
                    logger.error(f"Task {task.task_id} raised exception: {e}")
                    results.append(ProcessingResult(
                        task_id=task.task_id,
                        success=False,
                        error=str(e)
                    ))

                completed += 1

        elapsed = time.time() - start_time
        success_count = sum(1 for r in results if r.success)
        logger.info(f"Batch complete: {success_count}/{len(tasks)} succeeded in {elapsed:.1f}s")

        return results

    @staticmethod
    def _process_single_task(task: ProcessingTask) -> ProcessingResult:
        """
        Process a single task (runs in worker process).

        This is a static method so it can be pickled for multiprocessing.
        """
        start_time = time.time()

        try:
            if task.task_type == "full_video":
                output_path = ParallelVideoProcessor._create_full_video(task.params)
            elif task.task_type == "short":
                output_path = ParallelVideoProcessor._create_short_video(task.params)
            elif task.task_type == "thumbnail":
                output_path = ParallelVideoProcessor._create_thumbnail(task.params)
            elif task.task_type == "subtitle":
                output_path = ParallelVideoProcessor._create_subtitles(task.params)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")

            duration = time.time() - start_time

            return ProcessingResult(
                task_id=task.task_id,
                success=True,
                output_path=output_path,
                duration_seconds=duration
            )

        except Exception as e:
            duration = time.time() - start_time
            return ProcessingResult(
                task_id=task.task_id,
                success=False,
                error=str(e),
                duration_seconds=duration
            )

    @staticmethod
    def _create_full_video(params: Dict) -> Optional[str]:
        """Create a full video (worker method)."""
        # Import here to avoid issues with multiprocessing
        from .video_ultra import UltraVideoGenerator

        generator = UltraVideoGenerator()

        # Extract params
        topic = params.get("topic")
        channel_id = params.get("channel_id")
        niche = params.get("niche", "general")
        output_file = params.get("output_file", f"output/videos/{topic[:30]}.mp4")

        # Create video
        result = generator.create_full_video(
            topic=topic,
            niche=niche,
            output_file=output_file
        )

        return result

    @staticmethod
    def _create_short_video(params: Dict) -> Optional[str]:
        """Create a short video (worker method)."""
        from .shorts_hybrid import HybridShortsGenerator

        generator = HybridShortsGenerator()

        topic = params.get("topic")
        output_file = params.get("output_file", f"output/shorts/{topic[:30]}.mp4")

        result = generator.create_short(
            topic=topic,
            output_file=output_file
        )

        return result

    @staticmethod
    def _create_thumbnail(params: Dict) -> Optional[str]:
        """Create a thumbnail (worker method)."""
        from .thumbnail_ai import ThumbnailGenerator

        generator = ThumbnailGenerator()

        title = params.get("title")
        output_file = params.get("output_file", f"output/thumbnails/{title[:30]}.png")
        use_ai = params.get("use_ai", True)

        if use_ai:
            result = generator.generate_ai_thumbnail(title, output_file)
        else:
            result = generator.generate_standard_thumbnail(title, output_file)

        return result

    @staticmethod
    def _create_subtitles(params: Dict) -> Optional[str]:
        """Create subtitles (worker method)."""
        from .subtitles import SubtitleGenerator

        generator = SubtitleGenerator()

        audio_file = params.get("audio_file")
        output_file = params.get("output_file", audio_file.replace(".mp3", ".srt"))

        track = generator.generate_subtitles_from_audio(audio_file)
        if track:
            generator.create_srt_file(track, output_file)
            return output_file

        return None

    def create_videos_parallel(
        self,
        topics: List[str],
        channel_id: str,
        niche: str = "general",
        task_type: str = "full_video"
    ) -> List[ProcessingResult]:
        """
        Create multiple videos in parallel.

        Args:
            topics: List of video topics
            channel_id: Channel identifier
            niche: Content niche
            task_type: Type of video to create

        Returns:
            List of ProcessingResult objects
        """
        tasks = []

        for i, topic in enumerate(topics):
            task = ProcessingTask(
                task_id=f"{channel_id}_{i}_{topic[:20]}",
                task_type=task_type,
                params={
                    "topic": topic,
                    "channel_id": channel_id,
                    "niche": niche,
                    "output_file": f"output/{task_type}/{channel_id}_{i}_{topic[:30]}.mp4"
                },
                priority=i  # First topics have higher priority
            )
            tasks.append(task)

        return self.process_batch(tasks)

    def create_thumbnails_parallel(
        self,
        titles: List[str],
        use_ai: bool = True
    ) -> List[ProcessingResult]:
        """
        Create multiple thumbnails in parallel.

        Args:
            titles: List of video titles
            use_ai: Whether to use AI generation

        Returns:
            List of ProcessingResult objects
        """
        tasks = []

        for i, title in enumerate(titles):
            task = ProcessingTask(
                task_id=f"thumb_{i}_{title[:20]}",
                task_type="thumbnail",
                params={
                    "title": title,
                    "use_ai": use_ai,
                    "output_file": f"output/thumbnails/thumb_{i}.png"
                }
            )
            tasks.append(task)

        return self.process_batch(tasks)

    def process_chunks(
        self,
        all_tasks: List[ProcessingTask],
        callback: Optional[Callable] = None
    ) -> List[ProcessingResult]:
        """
        Process tasks in chunks to manage memory usage.

        Args:
            all_tasks: List of all tasks
            callback: Optional callback for each result

        Returns:
            List of all ProcessingResult objects
        """
        all_results = []

        # Process in chunks
        for i in range(0, len(all_tasks), self.chunk_size):
            chunk = all_tasks[i:i + self.chunk_size]
            logger.info(f"Processing chunk {i // self.chunk_size + 1} ({len(chunk)} tasks)...")

            chunk_results = self.process_batch(chunk, callback)
            all_results.extend(chunk_results)

            # Small delay between chunks to allow garbage collection
            if i + self.chunk_size < len(all_tasks):
                time.sleep(1)

        return all_results


# Example usage
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  PARALLEL VIDEO PROCESSOR TEST")
    print("="*60 + "\n")

    processor = ParallelVideoProcessor(max_workers=3)

    # Create sample tasks
    tasks = [
        ProcessingTask(
            task_id=f"video_{i}",
            task_type="full_video",
            params={
                "topic": f"Test Video {i}",
                "channel_id": "test_channel",
                "niche": "psychology",
                "output_file": f"output/test_video_{i}.mp4"
            },
            priority=i
        )
        for i in range(5)
    ]

    print(f"Processing {len(tasks)} tasks with {processor.max_workers} workers...")

    # Process with callback
    def progress_callback(result: ProcessingResult):
        status = "SUCCESS" if result.success else "FAILED"
        print(f"  [{status}] {result.task_id} ({result.duration_seconds:.1f}s)")

    results = processor.process_batch(tasks, callback=progress_callback)

    print(f"\nResults:")
    print(f"  Total: {len(results)}")
    print(f"  Success: {sum(1 for r in results if r.success)}")
    print(f"  Failed: {sum(1 for r in results if not r.success)}")
