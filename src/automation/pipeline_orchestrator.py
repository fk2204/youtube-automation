"""
Pipeline Orchestrator - Parallel Pipeline Execution and Management

Provides robust pipeline orchestration for YouTube automation:
- ParallelPipelineRunner: Run multiple pipelines simultaneously
- DependencyResolver: Handle task dependencies
- FailureRecovery: Auto-retry with exponential backoff
- ProgressTracker: Real-time pipeline status
- ResourceManager: Manage CPU/memory/API limits
- NotificationSystem: Alerts on success/failure

Usage:
    from src.automation.pipeline_orchestrator import PipelineOrchestrator

    orchestrator = PipelineOrchestrator()

    # Run multiple pipelines
    results = await orchestrator.run_parallel([
        {"channel_id": "money_blueprints", "topic": "passive income"},
        {"channel_id": "mind_unlocked", "topic": "psychology tips"}
    ])

    # Track progress
    status = await orchestrator.get_progress("pipeline_123")

    # Configure notifications
    await orchestrator.configure_notifications(
        email="admin@example.com",
        webhook="https://hooks.slack.com/..."
    )
"""

import asyncio
import json
import sqlite3
import os
import time
import psutil
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
from loguru import logger
import aiohttp


class PipelineStatus(Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    PAUSED = "paused"


class TaskStatus(Enum):
    """Individual task status."""
    PENDING = "pending"
    WAITING = "waiting"  # Waiting for dependencies
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class NotificationType(Enum):
    """Types of notifications."""
    SUCCESS = "success"
    FAILURE = "failure"
    WARNING = "warning"
    PROGRESS = "progress"
    QUOTA_ALERT = "quota_alert"


@dataclass
class Task:
    """Represents a single task in a pipeline."""
    task_id: str
    name: str
    handler: Optional[str] = None  # Handler function name
    dependencies: List[str] = field(default_factory=list)
    status: str = TaskStatus.PENDING.value
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Pipeline:
    """Represents a complete pipeline."""
    pipeline_id: str
    name: str
    channel_id: str
    tasks: List[Task] = field(default_factory=list)
    status: str = PipelineStatus.PENDING.value
    progress_percent: float = 0.0
    current_task: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["tasks"] = [t.to_dict() if isinstance(t, Task) else t for t in self.tasks]
        return result


@dataclass
class ResourceLimits:
    """Resource limits for pipeline execution."""
    max_concurrent_pipelines: int = 3
    max_cpu_percent: float = 80.0
    max_memory_percent: float = 75.0
    max_api_calls_per_minute: int = 60
    max_disk_usage_percent: float = 90.0


@dataclass
class NotificationConfig:
    """Notification configuration."""
    email: Optional[str] = None
    webhook_url: Optional[str] = None
    slack_channel: Optional[str] = None
    discord_webhook: Optional[str] = None
    notify_on_success: bool = True
    notify_on_failure: bool = True
    notify_on_warning: bool = True
    notify_on_progress: bool = False
    progress_interval_percent: int = 25


class DependencyResolver:
    """
    Resolve task dependencies and determine execution order.
    Uses topological sort for DAG resolution.
    """

    def __init__(self):
        self._resolved_order: Dict[str, List[str]] = {}

    def resolve(self, tasks: List[Task]) -> List[List[Task]]:
        """
        Resolve dependencies and return tasks grouped by execution level.

        Args:
            tasks: List of tasks with dependencies

        Returns:
            List of task groups that can be executed in parallel
        """
        # Build dependency graph
        task_map = {t.task_id: t for t in tasks}
        in_degree: Dict[str, int] = {t.task_id: 0 for t in tasks}
        dependents: Dict[str, List[str]] = defaultdict(list)

        for task in tasks:
            for dep in task.dependencies:
                if dep in task_map:
                    dependents[dep].append(task.task_id)
                    in_degree[task.task_id] += 1

        # Topological sort with levels
        levels: List[List[Task]] = []
        ready = [tid for tid, degree in in_degree.items() if degree == 0]

        while ready:
            # All tasks with no remaining dependencies can run in parallel
            current_level = [task_map[tid] for tid in ready]
            levels.append(current_level)

            next_ready = []
            for tid in ready:
                for dependent in dependents[tid]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        next_ready.append(dependent)

            ready = next_ready

        # Check for cycles
        if sum(len(level) for level in levels) != len(tasks):
            remaining = [tid for tid, degree in in_degree.items() if degree > 0]
            logger.warning(f"Circular dependency detected in tasks: {remaining}")

        return levels

    def get_ready_tasks(
        self,
        tasks: List[Task],
        completed_tasks: Set[str]
    ) -> List[Task]:
        """
        Get tasks that are ready to execute (all dependencies satisfied).

        Args:
            tasks: All tasks
            completed_tasks: Set of completed task IDs

        Returns:
            List of tasks ready to execute
        """
        ready = []
        for task in tasks:
            if task.status != TaskStatus.PENDING.value:
                continue

            # Check all dependencies are completed
            deps_satisfied = all(
                dep in completed_tasks for dep in task.dependencies
            )
            if deps_satisfied:
                ready.append(task)

        return ready

    def validate_dependencies(self, tasks: List[Task]) -> List[str]:
        """
        Validate task dependencies.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        task_ids = {t.task_id for t in tasks}

        for task in tasks:
            for dep in task.dependencies:
                if dep not in task_ids:
                    errors.append(
                        f"Task '{task.task_id}' depends on unknown task '{dep}'"
                    )

        # Check for self-dependencies
        for task in tasks:
            if task.task_id in task.dependencies:
                errors.append(f"Task '{task.task_id}' depends on itself")

        return errors


class FailureRecovery:
    """
    Handle failure recovery with exponential backoff and retry strategies.
    """

    # Backoff configuration
    BASE_DELAY = 1.0
    MAX_DELAY = 60.0
    MULTIPLIER = 2.0
    JITTER = 0.1

    # Retryable error patterns
    RETRYABLE_ERRORS = [
        "rate limit",
        "timeout",
        "connection",
        "temporary",
        "429",
        "503",
        "502",
        "500",
        "network",
    ]

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path("data/pipeline.db")
        self._retry_counts: Dict[str, int] = {}
        self._init_db()

    def _init_db(self):
        """Initialize failure tracking database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS task_failures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    pipeline_id TEXT,
                    error_message TEXT,
                    retry_count INTEGER,
                    recovered BOOLEAN DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def is_retryable(self, error: str) -> bool:
        """Check if an error is retryable."""
        error_lower = error.lower()
        return any(pattern in error_lower for pattern in self.RETRYABLE_ERRORS)

    def calculate_delay(self, retry_count: int) -> float:
        """
        Calculate delay with exponential backoff and jitter.

        Args:
            retry_count: Number of previous retries

        Returns:
            Delay in seconds
        """
        delay = min(
            self.BASE_DELAY * (self.MULTIPLIER ** retry_count),
            self.MAX_DELAY
        )

        # Add jitter
        import random
        jitter = delay * self.JITTER * (random.random() * 2 - 1)
        return max(0, delay + jitter)

    async def handle_failure(
        self,
        task: Task,
        pipeline_id: str,
        error: str,
        handler: Optional[Callable] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Handle a task failure with retry logic.

        Args:
            task: Failed task
            pipeline_id: Pipeline identifier
            error: Error message
            handler: Retry handler function

        Returns:
            Tuple of (success, result)
        """
        # Record failure
        self._record_failure(task.task_id, pipeline_id, error, task.retries)

        # Check if retryable
        if not self.is_retryable(error):
            logger.warning(f"Non-retryable error for task {task.task_id}: {error[:50]}...")
            return False, None

        # Check retry limit
        if task.retries >= task.max_retries:
            logger.warning(f"Task {task.task_id} exceeded max retries ({task.max_retries})")
            return False, None

        # Calculate delay
        delay = self.calculate_delay(task.retries)
        logger.info(
            f"Retrying task {task.task_id} in {delay:.1f}s "
            f"(attempt {task.retries + 1}/{task.max_retries})"
        )

        await asyncio.sleep(delay)
        task.retries += 1

        # Retry
        if handler:
            try:
                result = await handler(task)
                self._mark_recovered(task.task_id, pipeline_id)
                return True, result
            except Exception as e:
                logger.error(f"Retry failed for task {task.task_id}: {e}")
                return await self.handle_failure(task, pipeline_id, str(e), handler)

        return False, None

    def _record_failure(
        self,
        task_id: str,
        pipeline_id: str,
        error: str,
        retry_count: int
    ):
        """Record a failure to the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO task_failures
                (task_id, pipeline_id, error_message, retry_count)
                VALUES (?, ?, ?, ?)
            """, (task_id, pipeline_id, error[:500], retry_count))

    def _mark_recovered(self, task_id: str, pipeline_id: str):
        """Mark a task as recovered."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE task_failures SET recovered = 1
                WHERE task_id = ? AND pipeline_id = ?
                AND id = (SELECT MAX(id) FROM task_failures
                          WHERE task_id = ? AND pipeline_id = ?)
            """, (task_id, pipeline_id, task_id, pipeline_id))

    async def get_failure_stats(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get failure statistics."""
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("""
                SELECT COUNT(*) FROM task_failures WHERE created_at > ?
            """, (cutoff,)).fetchone()[0]

            recovered = conn.execute("""
                SELECT COUNT(*) FROM task_failures
                WHERE created_at > ? AND recovered = 1
            """, (cutoff,)).fetchone()[0]

            by_task = conn.execute("""
                SELECT task_id, COUNT(*) as count
                FROM task_failures WHERE created_at > ?
                GROUP BY task_id ORDER BY count DESC LIMIT 10
            """, (cutoff,)).fetchall()

        return {
            "total_failures": total,
            "recovered": recovered,
            "recovery_rate": f"{(recovered/total*100):.1f}%" if total > 0 else "N/A",
            "top_failing_tasks": [{"task": t[0], "count": t[1]} for t in by_task]
        }


class ProgressTracker:
    """
    Track real-time pipeline progress with callbacks and persistence.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path("data/pipeline.db")
        self._callbacks: List[Callable] = []
        self._progress_cache: Dict[str, Dict[str, Any]] = {}
        self._init_db()

    def _init_db(self):
        """Initialize progress tracking database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_progress (
                    pipeline_id TEXT PRIMARY KEY,
                    status TEXT,
                    progress_percent REAL,
                    current_task TEXT,
                    tasks_completed INTEGER,
                    tasks_total INTEGER,
                    started_at TEXT,
                    updated_at TEXT,
                    metadata TEXT
                )
            """)

    def register_callback(self, callback: Callable):
        """Register a progress callback."""
        self._callbacks.append(callback)

    async def update_progress(
        self,
        pipeline: Pipeline,
        current_task: Optional[str] = None
    ):
        """
        Update pipeline progress.

        Args:
            pipeline: Pipeline to update
            current_task: Current task name
        """
        completed = sum(
            1 for t in pipeline.tasks
            if t.status == TaskStatus.COMPLETED.value
        )
        total = len(pipeline.tasks)
        progress = (completed / total * 100) if total > 0 else 0

        pipeline.progress_percent = progress
        pipeline.current_task = current_task

        # Update cache
        self._progress_cache[pipeline.pipeline_id] = {
            "pipeline_id": pipeline.pipeline_id,
            "status": pipeline.status,
            "progress_percent": progress,
            "current_task": current_task,
            "tasks_completed": completed,
            "tasks_total": total,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }

        # Persist
        await self._persist_progress(pipeline, completed, total)

        # Notify callbacks
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(pipeline, progress, current_task)
                else:
                    callback(pipeline, progress, current_task)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    async def _persist_progress(
        self,
        pipeline: Pipeline,
        completed: int,
        total: int
    ):
        """Persist progress to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO pipeline_progress
                (pipeline_id, status, progress_percent, current_task,
                 tasks_completed, tasks_total, started_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pipeline.pipeline_id, pipeline.status, pipeline.progress_percent,
                pipeline.current_task, completed, total, pipeline.started_at,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(pipeline.metadata)
            ))

    async def get_progress(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get current progress for a pipeline."""
        # Check cache first
        if pipeline_id in self._progress_cache:
            return self._progress_cache[pipeline_id]

        # Load from database
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("""
                SELECT * FROM pipeline_progress WHERE pipeline_id = ?
            """, (pipeline_id,)).fetchone()

        if row:
            return dict(row)
        return None

    async def get_active_pipelines(self) -> List[Dict[str, Any]]:
        """Get all active (running) pipelines."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM pipeline_progress
                WHERE status IN ('running', 'queued', 'retrying')
                ORDER BY updated_at DESC
            """).fetchall()

        return [dict(row) for row in rows]


class ResourceManager:
    """
    Manage system resources and API rate limits.
    """

    def __init__(self, limits: Optional[ResourceLimits] = None):
        self.limits = limits or ResourceLimits()
        self._api_calls: Dict[str, List[float]] = defaultdict(list)
        self._active_pipelines: Set[str] = set()
        self._resource_locks: Dict[str, asyncio.Lock] = {}
        self._semaphore = asyncio.Semaphore(self.limits.max_concurrent_pipelines)

    async def acquire_pipeline_slot(self, pipeline_id: str) -> bool:
        """
        Acquire a slot to run a pipeline.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            True if slot acquired, False if at capacity
        """
        # Check concurrent limit
        if len(self._active_pipelines) >= self.limits.max_concurrent_pipelines:
            logger.warning(f"Pipeline {pipeline_id} queued - at max concurrent limit")
            return False

        # Check system resources
        resources_ok = await self.check_resources()
        if not resources_ok:
            logger.warning(f"Pipeline {pipeline_id} queued - insufficient resources")
            return False

        await self._semaphore.acquire()
        self._active_pipelines.add(pipeline_id)
        return True

    async def release_pipeline_slot(self, pipeline_id: str):
        """Release a pipeline slot."""
        if pipeline_id in self._active_pipelines:
            self._active_pipelines.discard(pipeline_id)
            self._semaphore.release()

    async def check_resources(self) -> bool:
        """
        Check if system resources are within limits.

        Returns:
            True if resources are available
        """
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > self.limits.max_cpu_percent:
                logger.warning(f"CPU usage too high: {cpu_percent}%")
                return False

            # Memory usage
            memory = psutil.virtual_memory()
            if memory.percent > self.limits.max_memory_percent:
                logger.warning(f"Memory usage too high: {memory.percent}%")
                return False

            # Disk usage
            disk = psutil.disk_usage("/")
            if disk.percent > self.limits.max_disk_usage_percent:
                logger.warning(f"Disk usage too high: {disk.percent}%")
                return False

            return True

        except Exception as e:
            logger.error(f"Resource check failed: {e}")
            return True  # Allow on error

    async def check_api_rate_limit(self, api_name: str = "default") -> bool:
        """
        Check if API rate limit allows a call.

        Args:
            api_name: API identifier

        Returns:
            True if call is allowed
        """
        now = time.time()
        minute_ago = now - 60

        # Clean old calls
        self._api_calls[api_name] = [
            t for t in self._api_calls[api_name] if t > minute_ago
        ]

        # Check limit
        if len(self._api_calls[api_name]) >= self.limits.max_api_calls_per_minute:
            return False

        return True

    async def record_api_call(self, api_name: str = "default"):
        """Record an API call for rate limiting."""
        self._api_calls[api_name].append(time.time())

    async def wait_for_api_slot(
        self,
        api_name: str = "default",
        timeout: float = 60.0
    ) -> bool:
        """
        Wait for an API rate limit slot to become available.

        Args:
            api_name: API identifier
            timeout: Maximum wait time in seconds

        Returns:
            True if slot became available, False if timeout
        """
        start = time.time()

        while time.time() - start < timeout:
            if await self.check_api_rate_limit(api_name):
                await self.record_api_call(api_name)
                return True
            await asyncio.sleep(1)

        return False

    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status."""
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            return {
                "cpu_percent": cpu,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3),
                "active_pipelines": len(self._active_pipelines),
                "max_pipelines": self.limits.max_concurrent_pipelines,
                "api_calls_last_minute": {
                    api: len(calls) for api, calls in self._api_calls.items()
                }
            }
        except Exception as e:
            return {"error": str(e)}


class NotificationSystem:
    """
    Send notifications on pipeline events.
    """

    def __init__(self, config: Optional[NotificationConfig] = None):
        self.config = config or NotificationConfig()
        self._http_session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()
        return self._http_session

    async def close(self):
        """Close HTTP session."""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()

    async def notify(
        self,
        notification_type: NotificationType,
        title: str,
        message: str,
        pipeline: Optional[Pipeline] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Send a notification.

        Args:
            notification_type: Type of notification
            title: Notification title
            message: Notification message
            pipeline: Related pipeline (optional)
            metadata: Additional metadata
        """
        # Check if this type should be sent
        if notification_type == NotificationType.SUCCESS and not self.config.notify_on_success:
            return
        if notification_type == NotificationType.FAILURE and not self.config.notify_on_failure:
            return
        if notification_type == NotificationType.WARNING and not self.config.notify_on_warning:
            return
        if notification_type == NotificationType.PROGRESS and not self.config.notify_on_progress:
            return

        # Build notification payload
        payload = {
            "type": notification_type.value,
            "title": title,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {}
        }

        if pipeline:
            payload["pipeline"] = {
                "id": pipeline.pipeline_id,
                "name": pipeline.name,
                "channel": pipeline.channel_id,
                "status": pipeline.status,
                "progress": pipeline.progress_percent
            }

        # Send to configured channels
        tasks = []

        if self.config.webhook_url:
            tasks.append(self._send_webhook(self.config.webhook_url, payload))

        if self.config.slack_channel and self.config.webhook_url:
            tasks.append(self._send_slack(payload))

        if self.config.discord_webhook:
            tasks.append(self._send_discord(payload))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.info(f"Notification sent: [{notification_type.value}] {title}")

    async def _send_webhook(self, url: str, payload: Dict[str, Any]):
        """Send webhook notification."""
        try:
            session = await self._get_session()
            async with session.post(url, json=payload, timeout=10) as response:
                if response.status >= 400:
                    logger.warning(f"Webhook failed: {response.status}")
        except Exception as e:
            logger.error(f"Webhook error: {e}")

    async def _send_slack(self, payload: Dict[str, Any]):
        """Send Slack notification."""
        # Format for Slack
        color = {
            "success": "#36a64f",
            "failure": "#ff0000",
            "warning": "#ffcc00",
            "progress": "#3498db"
        }.get(payload["type"], "#808080")

        slack_payload = {
            "attachments": [{
                "color": color,
                "title": payload["title"],
                "text": payload["message"],
                "ts": int(datetime.now().timestamp())
            }]
        }

        if "pipeline" in payload:
            slack_payload["attachments"][0]["fields"] = [
                {"title": "Pipeline", "value": payload["pipeline"]["name"], "short": True},
                {"title": "Status", "value": payload["pipeline"]["status"], "short": True},
                {"title": "Progress", "value": f"{payload['pipeline']['progress']:.0f}%", "short": True}
            ]

        try:
            session = await self._get_session()
            # Slack uses the same webhook URL
            if self.config.webhook_url:
                async with session.post(
                    self.config.webhook_url,
                    json=slack_payload,
                    timeout=10
                ) as response:
                    if response.status >= 400:
                        logger.warning(f"Slack notification failed: {response.status}")
        except Exception as e:
            logger.error(f"Slack notification error: {e}")

    async def _send_discord(self, payload: Dict[str, Any]):
        """Send Discord notification."""
        color = {
            "success": 0x36a64f,
            "failure": 0xff0000,
            "warning": 0xffcc00,
            "progress": 0x3498db
        }.get(payload["type"], 0x808080)

        discord_payload = {
            "embeds": [{
                "title": payload["title"],
                "description": payload["message"],
                "color": color,
                "timestamp": payload["timestamp"]
            }]
        }

        if "pipeline" in payload:
            discord_payload["embeds"][0]["fields"] = [
                {"name": "Pipeline", "value": payload["pipeline"]["name"], "inline": True},
                {"name": "Status", "value": payload["pipeline"]["status"], "inline": True},
                {"name": "Progress", "value": f"{payload['pipeline']['progress']:.0f}%", "inline": True}
            ]

        try:
            session = await self._get_session()
            async with session.post(
                self.config.discord_webhook,
                json=discord_payload,
                timeout=10
            ) as response:
                if response.status >= 400:
                    logger.warning(f"Discord notification failed: {response.status}")
        except Exception as e:
            logger.error(f"Discord notification error: {e}")

    async def notify_success(self, pipeline: Pipeline, message: str = ""):
        """Send success notification."""
        await self.notify(
            NotificationType.SUCCESS,
            f"Pipeline Completed: {pipeline.name}",
            message or f"Successfully completed {len(pipeline.tasks)} tasks",
            pipeline
        )

    async def notify_failure(self, pipeline: Pipeline, error: str):
        """Send failure notification."""
        await self.notify(
            NotificationType.FAILURE,
            f"Pipeline Failed: {pipeline.name}",
            f"Error: {error}",
            pipeline
        )

    async def notify_progress(self, pipeline: Pipeline):
        """Send progress notification."""
        await self.notify(
            NotificationType.PROGRESS,
            f"Pipeline Progress: {pipeline.name}",
            f"Progress: {pipeline.progress_percent:.0f}% - {pipeline.current_task}",
            pipeline
        )


class ParallelPipelineRunner:
    """
    Execute pipelines with parallel task support.
    """

    def __init__(
        self,
        resource_manager: ResourceManager,
        progress_tracker: ProgressTracker,
        failure_recovery: FailureRecovery,
        notification_system: NotificationSystem
    ):
        self.resource_manager = resource_manager
        self.progress_tracker = progress_tracker
        self.failure_recovery = failure_recovery
        self.notification_system = notification_system
        self.dependency_resolver = DependencyResolver()
        self._task_handlers: Dict[str, Callable] = {}
        self._running_pipelines: Dict[str, asyncio.Task] = {}

    def register_handler(self, name: str, handler: Callable):
        """Register a task handler."""
        self._task_handlers[name] = handler

    async def run_pipeline(self, pipeline: Pipeline) -> Pipeline:
        """
        Execute a single pipeline.

        Args:
            pipeline: Pipeline to execute

        Returns:
            Updated pipeline with results
        """
        pipeline.started_at = datetime.now(timezone.utc).isoformat()
        pipeline.status = PipelineStatus.RUNNING.value

        try:
            # Acquire resources
            acquired = await self.resource_manager.acquire_pipeline_slot(pipeline.pipeline_id)
            if not acquired:
                pipeline.status = PipelineStatus.QUEUED.value
                await self.progress_tracker.update_progress(pipeline)

                # Wait for slot
                while not acquired:
                    await asyncio.sleep(5)
                    acquired = await self.resource_manager.acquire_pipeline_slot(
                        pipeline.pipeline_id
                    )

                pipeline.status = PipelineStatus.RUNNING.value

            # Validate dependencies
            errors = self.dependency_resolver.validate_dependencies(pipeline.tasks)
            if errors:
                raise ValueError(f"Dependency errors: {errors}")

            # Resolve execution order
            task_levels = self.dependency_resolver.resolve(pipeline.tasks)

            # Execute tasks level by level
            completed_tasks: Set[str] = set()
            last_progress = 0

            for level_idx, level_tasks in enumerate(task_levels):
                logger.info(
                    f"Pipeline {pipeline.pipeline_id}: "
                    f"Executing level {level_idx + 1}/{len(task_levels)} "
                    f"({len(level_tasks)} tasks)"
                )

                # Run tasks in parallel
                task_coroutines = [
                    self._execute_task(task, pipeline)
                    for task in level_tasks
                ]

                results = await asyncio.gather(*task_coroutines, return_exceptions=True)

                # Process results
                for task, result in zip(level_tasks, results):
                    if isinstance(result, Exception):
                        task.status = TaskStatus.FAILED.value
                        task.error = str(result)

                        # Attempt recovery
                        success, retry_result = await self.failure_recovery.handle_failure(
                            task,
                            pipeline.pipeline_id,
                            str(result),
                            self._task_handlers.get(task.handler)
                        )

                        if not success:
                            # Check if task is critical
                            if task.metadata.get("critical", True):
                                raise result
                    else:
                        completed_tasks.add(task.task_id)

                # Update progress
                await self.progress_tracker.update_progress(
                    pipeline,
                    f"Completed level {level_idx + 1}/{len(task_levels)}"
                )

                # Check if progress milestone reached for notification
                current_progress = int(pipeline.progress_percent)
                if (current_progress - last_progress) >= \
                   self.notification_system.config.progress_interval_percent:
                    await self.notification_system.notify_progress(pipeline)
                    last_progress = current_progress

            # Mark completed
            pipeline.status = PipelineStatus.COMPLETED.value
            pipeline.completed_at = datetime.now(timezone.utc).isoformat()
            pipeline.progress_percent = 100.0

            await self.progress_tracker.update_progress(pipeline, "Completed")
            await self.notification_system.notify_success(pipeline)

        except Exception as e:
            logger.error(f"Pipeline {pipeline.pipeline_id} failed: {e}")
            pipeline.status = PipelineStatus.FAILED.value
            pipeline.error = str(e)
            await self.progress_tracker.update_progress(pipeline, f"Failed: {str(e)[:50]}")
            await self.notification_system.notify_failure(pipeline, str(e))

        finally:
            await self.resource_manager.release_pipeline_slot(pipeline.pipeline_id)

        return pipeline

    async def _execute_task(self, task: Task, pipeline: Pipeline) -> Dict[str, Any]:
        """Execute a single task."""
        task.status = TaskStatus.RUNNING.value
        task.started_at = datetime.now(timezone.utc).isoformat()

        await self.progress_tracker.update_progress(pipeline, task.name)

        try:
            # Get handler
            handler = self._task_handlers.get(task.handler)
            if not handler:
                raise ValueError(f"No handler registered for: {task.handler}")

            # Check API rate limit if needed
            if task.metadata.get("requires_api"):
                api_name = task.metadata.get("api_name", "default")
                if not await self.resource_manager.wait_for_api_slot(api_name, 30):
                    raise TimeoutError(f"API rate limit timeout for {api_name}")

            # Execute with timeout
            result = await asyncio.wait_for(
                handler(task, pipeline),
                timeout=task.timeout_seconds
            )

            task.status = TaskStatus.COMPLETED.value
            task.result = result
            task.completed_at = datetime.now(timezone.utc).isoformat()

            return result

        except asyncio.TimeoutError:
            task.status = TaskStatus.FAILED.value
            task.error = f"Task timeout after {task.timeout_seconds}s"
            raise

        except Exception as e:
            task.status = TaskStatus.FAILED.value
            task.error = str(e)
            raise

    async def run_parallel(
        self,
        pipelines: List[Pipeline],
        max_concurrent: int = 3
    ) -> List[Pipeline]:
        """
        Run multiple pipelines in parallel.

        Args:
            pipelines: List of pipelines to execute
            max_concurrent: Maximum concurrent pipelines

        Returns:
            List of completed pipelines
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_with_semaphore(pipeline: Pipeline) -> Pipeline:
            async with semaphore:
                return await self.run_pipeline(pipeline)

        results = await asyncio.gather(
            *[run_with_semaphore(p) for p in pipelines],
            return_exceptions=True
        )

        return [
            r if isinstance(r, Pipeline) else pipelines[i]
            for i, r in enumerate(results)
        ]

    async def cancel_pipeline(self, pipeline_id: str) -> bool:
        """Cancel a running pipeline."""
        if pipeline_id in self._running_pipelines:
            self._running_pipelines[pipeline_id].cancel()
            return True
        return False


class PipelineOrchestrator:
    """
    Main orchestrator class combining all pipeline management features.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        resource_limits: Optional[ResourceLimits] = None,
        notification_config: Optional[NotificationConfig] = None
    ):
        self.db_path = db_path or Path("data/pipeline.db")

        # Initialize components
        self.resource_manager = ResourceManager(resource_limits)
        self.progress_tracker = ProgressTracker(self.db_path)
        self.failure_recovery = FailureRecovery(self.db_path)
        self.notification_system = NotificationSystem(notification_config)
        self.dependency_resolver = DependencyResolver()

        self.runner = ParallelPipelineRunner(
            self.resource_manager,
            self.progress_tracker,
            self.failure_recovery,
            self.notification_system
        )

        # Pipeline storage
        self._pipelines: Dict[str, Pipeline] = {}
        self._init_db()

        logger.info("PipelineOrchestrator initialized")

    def _init_db(self):
        """Initialize database tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pipelines (
                    pipeline_id TEXT PRIMARY KEY,
                    name TEXT,
                    channel_id TEXT,
                    status TEXT,
                    progress_percent REAL,
                    tasks TEXT,
                    metadata TEXT,
                    created_at TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    error TEXT
                )
            """)

    def _generate_pipeline_id(self, channel_id: str) -> str:
        """Generate unique pipeline ID."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        unique = hashlib.md5(f"{channel_id}{timestamp}".encode()).hexdigest()[:8]
        return f"pipeline_{channel_id}_{timestamp}_{unique}"

    def register_handler(self, name: str, handler: Callable):
        """Register a task handler."""
        self.runner.register_handler(name, handler)

    async def create_pipeline(
        self,
        channel_id: str,
        name: str,
        tasks: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Pipeline:
        """
        Create a new pipeline.

        Args:
            channel_id: Channel identifier
            name: Pipeline name
            tasks: List of task definitions
            metadata: Additional metadata

        Returns:
            Created pipeline
        """
        pipeline_id = self._generate_pipeline_id(channel_id)

        # Convert task dicts to Task objects
        task_objects = [
            Task(
                task_id=f"{pipeline_id}_task_{i}",
                name=t.get("name", f"Task {i}"),
                handler=t.get("handler"),
                dependencies=t.get("dependencies", []),
                max_retries=t.get("max_retries", 3),
                timeout_seconds=t.get("timeout", 300),
                metadata=t.get("metadata", {})
            )
            for i, t in enumerate(tasks)
        ]

        pipeline = Pipeline(
            pipeline_id=pipeline_id,
            name=name,
            channel_id=channel_id,
            tasks=task_objects,
            metadata=metadata or {}
        )

        self._pipelines[pipeline_id] = pipeline
        await self._save_pipeline(pipeline)

        return pipeline

    async def _save_pipeline(self, pipeline: Pipeline):
        """Save pipeline to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO pipelines
                (pipeline_id, name, channel_id, status, progress_percent,
                 tasks, metadata, created_at, started_at, completed_at, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pipeline.pipeline_id, pipeline.name, pipeline.channel_id,
                pipeline.status, pipeline.progress_percent,
                json.dumps([t.to_dict() for t in pipeline.tasks]),
                json.dumps(pipeline.metadata),
                pipeline.created_at, pipeline.started_at,
                pipeline.completed_at, pipeline.error
            ))

    async def run(self, pipeline_id: str) -> Pipeline:
        """
        Run a pipeline by ID.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            Completed pipeline
        """
        pipeline = self._pipelines.get(pipeline_id)
        if not pipeline:
            raise ValueError(f"Pipeline not found: {pipeline_id}")

        result = await self.runner.run_pipeline(pipeline)
        await self._save_pipeline(result)

        return result

    async def run_parallel(
        self,
        items: List[Dict[str, Any]],
        default_tasks: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Create and run multiple pipelines in parallel.

        Args:
            items: List of items with channel_id and topic
            default_tasks: Default task list for each pipeline

        Returns:
            Dict with results for each pipeline
        """
        # Create pipelines
        pipelines = []
        for item in items:
            channel_id = item.get("channel_id", "unknown")
            topic = item.get("topic", "Untitled")
            tasks = item.get("tasks", default_tasks or [])

            pipeline = await self.create_pipeline(
                channel_id=channel_id,
                name=f"Video: {topic[:30]}",
                tasks=tasks,
                metadata={"topic": topic, **item.get("metadata", {})}
            )
            pipelines.append(pipeline)

        # Run all pipelines
        results = await self.runner.run_parallel(pipelines)

        # Save results
        for pipeline in results:
            await self._save_pipeline(pipeline)

        return {
            "pipelines": [p.to_dict() for p in results],
            "total": len(results),
            "completed": sum(1 for p in results if p.status == PipelineStatus.COMPLETED.value),
            "failed": sum(1 for p in results if p.status == PipelineStatus.FAILED.value)
        }

    async def get_progress(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get pipeline progress."""
        return await self.progress_tracker.get_progress(pipeline_id)

    async def get_pipeline(self, pipeline_id: str) -> Optional[Pipeline]:
        """Get pipeline by ID."""
        return self._pipelines.get(pipeline_id)

    async def cancel(self, pipeline_id: str) -> bool:
        """Cancel a running pipeline."""
        success = await self.runner.cancel_pipeline(pipeline_id)
        if success and pipeline_id in self._pipelines:
            self._pipelines[pipeline_id].status = PipelineStatus.CANCELLED.value
            await self._save_pipeline(self._pipelines[pipeline_id])
        return success

    async def configure_notifications(
        self,
        email: Optional[str] = None,
        webhook: Optional[str] = None,
        slack_channel: Optional[str] = None,
        discord_webhook: Optional[str] = None
    ):
        """Configure notification settings."""
        if email:
            self.notification_system.config.email = email
        if webhook:
            self.notification_system.config.webhook_url = webhook
        if slack_channel:
            self.notification_system.config.slack_channel = slack_channel
        if discord_webhook:
            self.notification_system.config.discord_webhook = discord_webhook

    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status."""
        return self.resource_manager.get_resource_status()

    async def get_failure_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get failure statistics."""
        return await self.failure_recovery.get_failure_stats(hours)

    async def get_active_pipelines(self) -> List[Dict[str, Any]]:
        """Get all active pipelines."""
        return await self.progress_tracker.get_active_pipelines()

    async def close(self):
        """Clean up resources."""
        await self.notification_system.close()


# CLI entry point
if __name__ == "__main__":
    import sys

    async def main():
        print("\n" + "=" * 60)
        print("  PIPELINE ORCHESTRATOR")
        print("=" * 60 + "\n")

        orchestrator = PipelineOrchestrator()

        # Register example handlers
        async def research_handler(task: Task, pipeline: Pipeline) -> Dict[str, Any]:
            await asyncio.sleep(1)  # Simulate work
            return {"topic": pipeline.metadata.get("topic"), "ideas": ["idea1", "idea2"]}

        async def script_handler(task: Task, pipeline: Pipeline) -> Dict[str, Any]:
            await asyncio.sleep(2)
            return {"script": "Generated script content..."}

        orchestrator.register_handler("research", research_handler)
        orchestrator.register_handler("script", script_handler)

        if len(sys.argv) > 1:
            command = sys.argv[1]

            if command == "run":
                channel = sys.argv[2] if len(sys.argv) > 2 else "money_blueprints"
                topic = sys.argv[3] if len(sys.argv) > 3 else "passive income"

                tasks = [
                    {"name": "Research", "handler": "research"},
                    {"name": "Script", "handler": "script", "dependencies": ["research"]}
                ]

                pipeline = await orchestrator.create_pipeline(
                    channel_id=channel,
                    name=f"Video: {topic}",
                    tasks=tasks,
                    metadata={"topic": topic}
                )

                print(f"Created pipeline: {pipeline.pipeline_id}")
                result = await orchestrator.run(pipeline.pipeline_id)
                print(f"Status: {result.status}")
                print(f"Progress: {result.progress_percent:.0f}%")

            elif command == "status":
                status = orchestrator.get_resource_status()
                print("Resource Status:")
                print(f"  CPU: {status.get('cpu_percent', 'N/A')}%")
                print(f"  Memory: {status.get('memory_percent', 'N/A')}%")
                print(f"  Disk: {status.get('disk_percent', 'N/A')}%")
                print(f"  Active Pipelines: {status.get('active_pipelines', 0)}")

            elif command == "failures":
                hours = int(sys.argv[2]) if len(sys.argv) > 2 else 24
                stats = await orchestrator.get_failure_stats(hours)
                print(f"Failure Stats (last {hours}h):")
                print(f"  Total: {stats['total_failures']}")
                print(f"  Recovered: {stats['recovered']}")
                print(f"  Recovery Rate: {stats['recovery_rate']}")

            else:
                print(f"Unknown command: {command}")
        else:
            print("Usage:")
            print("  python -m src.automation.pipeline_orchestrator run [channel] [topic]")
            print("  python -m src.automation.pipeline_orchestrator status")
            print("  python -m src.automation.pipeline_orchestrator failures [hours]")

        await orchestrator.close()

    asyncio.run(main())
