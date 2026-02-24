"""
Async job queue for long-running content operations.

Content creation takes 5-15 minutes. This module provides:
- Job submission with immediate ID return
- Background execution via thread pool
- Status polling
- Optional webhook callback on completion
- SQLite-backed persistence across restarts
"""

import asyncio
import json
import sqlite3
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import aiohttp
from loguru import logger

from src.api.models import JobStatus, JobPriority


DB_PATH = Path("data/jobs.db")


@dataclass
class Job:
    """Represents an async content job."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    job_type: str = ""
    status: JobStatus = JobStatus.QUEUED
    priority: JobPriority = JobPriority.NORMAL
    progress: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    callback_url: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["priority"] = self.priority.value
        d["created_at"] = self.created_at.isoformat()
        d["started_at"] = self.started_at.isoformat() if self.started_at else None
        d["completed_at"] = self.completed_at.isoformat() if self.completed_at else None
        return d


class JobManager:
    """Manages async content jobs with SQLite persistence and thread pool execution."""

    def __init__(self, max_workers: int = 3, db_path: Optional[Path] = None):
        self._db_path = db_path or DB_PATH
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="job")
        self._jobs: Dict[str, Job] = {}
        self._handlers: Dict[str, Callable] = {}
        self._running_futures: Dict[str, asyncio.Future] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._init_db()
        self._load_incomplete_jobs()

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued',
                    priority TEXT NOT NULL DEFAULT 'normal',
                    progress REAL DEFAULT 0.0,
                    params TEXT DEFAULT '{}',
                    result TEXT,
                    error TEXT,
                    callback_url TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS ix_jobs_status ON jobs(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS ix_jobs_created ON jobs(created_at)")

    def _load_incomplete_jobs(self) -> None:
        """Load queued/running jobs from DB on startup (for crash recovery)."""
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM jobs WHERE status IN ('queued', 'running') ORDER BY created_at"
                ).fetchall()
                for row in rows:
                    job = self._row_to_job(row)
                    # Mark previously running jobs as queued for retry
                    if job.status == JobStatus.RUNNING:
                        job.status = JobStatus.QUEUED
                        job.started_at = None
                    self._jobs[job.id] = job
                if rows:
                    logger.info(f"Recovered {len(rows)} incomplete jobs from database")
        except Exception as e:
            logger.warning(f"Failed to load incomplete jobs: {e}")

    def _row_to_job(self, row: sqlite3.Row) -> Job:
        return Job(
            id=row["id"],
            job_type=row["job_type"],
            status=JobStatus(row["status"]),
            priority=JobPriority(row["priority"]),
            progress=row["progress"] or 0.0,
            params=json.loads(row["params"]) if row["params"] else {},
            result=json.loads(row["result"]) if row["result"] else None,
            error=row["error"],
            callback_url=row["callback_url"],
            created_at=datetime.fromisoformat(row["created_at"]),
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
        )

    def _persist_job(self, job: Job) -> None:
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO jobs
                    (id, job_type, status, priority, progress, params, result, error,
                     callback_url, created_at, started_at, completed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    job.id, job.job_type, job.status.value, job.priority.value,
                    job.progress,
                    json.dumps(job.params), json.dumps(job.result) if job.result else None,
                    job.error, job.callback_url,
                    job.created_at.isoformat(),
                    job.started_at.isoformat() if job.started_at else None,
                    job.completed_at.isoformat() if job.completed_at else None,
                ))
        except Exception as e:
            logger.error(f"Failed to persist job {job.id}: {e}")

    def register_handler(self, job_type: str, handler: Callable) -> None:
        """Register a handler function for a job type.

        Handler signature: async def handler(job: Job) -> Dict[str, Any]
        """
        self._handlers[job_type] = handler
        logger.debug(f"Registered handler for job type: {job_type}")

    def submit(self, job_type: str, params: Dict[str, Any],
               priority: JobPriority = JobPriority.NORMAL,
               callback_url: Optional[str] = None) -> Job:
        """Submit a new job. Returns immediately with job ID."""
        job = Job(
            job_type=job_type,
            params=params,
            priority=priority,
            callback_url=callback_url,
        )
        self._jobs[job.id] = job
        self._persist_job(job)
        logger.info(f"Job submitted: {job.id} type={job_type} priority={priority.value}")

        # Schedule execution
        if self._loop and self._loop.is_running():
            self._loop.create_task(self._execute_job(job))
        else:
            # Fallback: try to get running loop
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._execute_job(job))
            except RuntimeError:
                logger.warning(f"No event loop available, job {job.id} will be picked up on next poll")

        return job

    async def _execute_job(self, job: Job) -> None:
        """Execute a job in the thread pool."""
        handler = self._handlers.get(job.job_type)
        if not handler:
            job.status = JobStatus.FAILED
            job.error = f"No handler registered for job type: {job.job_type}"
            job.completed_at = datetime.utcnow()
            self._persist_job(job)
            logger.error(job.error)
            return

        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()
        self._persist_job(job)
        logger.info(f"Job started: {job.id}")

        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(job)
            else:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(self._executor, handler, job)

            job.status = JobStatus.COMPLETED
            job.result = result if isinstance(result, dict) else {"output": result}
            job.progress = 100.0
            job.completed_at = datetime.utcnow()
            logger.info(f"Job completed: {job.id}")

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = f"{type(e).__name__}: {str(e)}"
            job.completed_at = datetime.utcnow()
            logger.error(f"Job failed: {job.id} — {job.error}\n{traceback.format_exc()}")

        self._persist_job(job)
        await self._send_callback(job)

    async def _send_callback(self, job: Job) -> None:
        """POST job result to callback_url if configured."""
        if not job.callback_url:
            return

        payload = {
            "job_id": job.id,
            "status": job.status.value,
            "job_type": job.job_type,
            "result": job.result,
            "error": job.error,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        }

        for attempt in range(3):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        job.callback_url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        if resp.status < 400:
                            logger.info(f"Callback sent for job {job.id} → {resp.status}")
                            return
                        logger.warning(f"Callback returned {resp.status} for job {job.id}")
            except Exception as e:
                logger.warning(f"Callback attempt {attempt + 1} failed for job {job.id}: {e}")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID (memory first, then DB)."""
        if job_id in self._jobs:
            return self._jobs[job_id]

        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
                if row:
                    job = self._row_to_job(row)
                    self._jobs[job_id] = job
                    return job
        except Exception as e:
            logger.error(f"Failed to fetch job {job_id}: {e}")

        return None

    def update_progress(self, job_id: str, progress: float) -> None:
        """Update job progress percentage (0-100)."""
        job = self._jobs.get(job_id)
        if job:
            job.progress = min(max(progress, 0.0), 100.0)
            self._persist_job(job)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued job. Running jobs cannot be cancelled."""
        job = self._jobs.get(job_id)
        if not job:
            return False
        if job.status != JobStatus.QUEUED:
            return False

        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.utcnow()
        self._persist_job(job)
        logger.info(f"Job cancelled: {job_id}")
        return True

    def list_jobs(self, status: Optional[JobStatus] = None, limit: int = 50) -> List[Job]:
        """List jobs, optionally filtered by status."""
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.row_factory = sqlite3.Row
                if status:
                    rows = conn.execute(
                        "SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                        (status.value, limit)
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
                    ).fetchall()
                return [self._row_to_job(r) for r in rows]
        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            return []

    def get_stats(self) -> Dict[str, int]:
        """Get job queue statistics."""
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                row = conn.execute("""
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN status = 'queued' THEN 1 ELSE 0 END) as queued,
                        SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) as running,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
                    FROM jobs
                """).fetchone()
                return {
                    "total": row[0] or 0,
                    "queued": row[1] or 0,
                    "running": row[2] or 0,
                    "completed": row[3] or 0,
                    "failed": row[4] or 0,
                }
        except Exception:
            return {"total": 0, "queued": 0, "running": 0, "completed": 0, "failed": 0}

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the event loop for job scheduling."""
        self._loop = loop

    async def start_recovery(self) -> None:
        """Re-execute any recovered queued jobs on startup."""
        queued = [j for j in self._jobs.values() if j.status == JobStatus.QUEUED]
        for job in queued:
            logger.info(f"Re-executing recovered job: {job.id}")
            await self._execute_job(job)


# Singleton
_instance: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    global _instance
    if _instance is None:
        _instance = JobManager()
    return _instance
