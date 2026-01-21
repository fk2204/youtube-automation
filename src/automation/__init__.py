"""
YouTube Automation Module

Token-optimized automation with independent task runners.
Includes unified launcher for parallel execution.

Integrated (2026-01-20):
- IntegratedPipelineOrchestrator: Pipeline with Whisper, AI disclosure, viral hooks, metadata optimizer
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

# Pipeline Orchestrator with integrated modules (NEW)
try:
    from .pipeline_orchestrator import (
        PipelineOrchestrator,
        IntegratedPipelineOrchestrator,
        Pipeline,
        Task,
        TaskStatus,
        PipelineStatus,
    )
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False

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
    # Pipeline Orchestrator (NEW)
    "PipelineOrchestrator",
    "IntegratedPipelineOrchestrator",
    "Pipeline",
    "Task",
    "TaskStatus",
    "PipelineStatus",
]
