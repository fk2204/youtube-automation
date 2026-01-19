"""
YouTube Automation Module

Token-optimized automation with independent task runners.
Includes unified launcher for parallel execution.
"""

from .runner import (
    task_research,
    task_script,
    task_audio,
    task_video,
    task_upload,
    task_full_pipeline,
    task_full_with_upload
)

from .unified_launcher import (
    UnifiedLauncher,
    LaunchConfig,
    LaunchResult,
    quick_video,
    quick_short,
    daily_all,
    parallel_batch,
)

__all__ = [
    # Task runners
    "task_research",
    "task_script",
    "task_audio",
    "task_video",
    "task_upload",
    "task_full_pipeline",
    "task_full_with_upload",
    # Unified launcher
    "UnifiedLauncher",
    "LaunchConfig",
    "LaunchResult",
    "quick_video",
    "quick_short",
    "daily_all",
    "parallel_batch",
]
