"""
Scheduler Agent - Job Scheduling with APScheduler

Wraps APScheduler functionality for managing automated YouTube video creation
and upload schedules per channel configuration.

Usage:
    from src.agents.scheduler_agent import SchedulerAgent

    agent = SchedulerAgent()

    # Add a job
    result = agent.run(
        action="add",
        channel_id="money_blueprints",
        job_type="video",
        schedule_time="15:00"
    )

    # List all jobs
    result = agent.run(action="list")

    # Get next run time
    result = agent.get_next_run(channel_id="money_blueprints")
"""

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from loguru import logger
import yaml

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.date import DateTrigger
    from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
    from apscheduler.executors.pool import ThreadPoolExecutor
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False
    logger.warning("APScheduler not available. Install with: pip install apscheduler")


class JobStatus(Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    MISSED = "missed"
    PAUSED = "paused"


class JobType(Enum):
    """Types of scheduled jobs."""
    VIDEO = "video"
    SHORT = "short"
    CLEANUP = "cleanup"
    HEALTH_CHECK = "health_check"
    ANALYTICS = "analytics"


@dataclass
class ScheduleResult:
    """Result from scheduler agent operations."""
    job_id: str
    status: str
    next_run: Optional[str]
    action: str = ""
    channel_id: Optional[str] = None
    job_type: Optional[str] = None
    success: bool = True
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScheduledJob:
    """Represents a scheduled job."""
    job_id: str
    channel_id: str
    job_type: str
    schedule_time: str  # HH:MM format (UTC)
    posting_days: Optional[List[int]]  # 0=Mon, 6=Sun, None=everyday
    status: str = JobStatus.PENDING.value
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    run_count: int = 0
    fail_count: int = 0
    priority: int = 0  # Higher = more important
    catch_up: bool = True  # Run missed schedules
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseAgent:
    """Base class for all agents."""

    def __init__(self, name: str):
        self.name = name
        self.state_dir = Path("data")
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def run(self, **kwargs) -> Any:
        """Execute the agent's main function. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement run()")


class SchedulerAgent(BaseAgent):
    """
    Scheduler Agent - Job scheduling with APScheduler.

    Features:
    - Execute daily video creation jobs per channel config
    - Manage upload schedules per channel (posting_days, times)
    - Handle timezone conversions (store in UTC)
    - Manage job queue with priorities
    - Handle missed schedules (catch-up mode)
    """

    # Day name mapping for cron
    DAY_NAMES = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

    def __init__(self, blocking: bool = False, auto_start: bool = False):
        """
        Initialize the scheduler agent.

        Args:
            blocking: Use blocking scheduler (for main process)
            auto_start: Start scheduler immediately
        """
        super().__init__("SchedulerAgent")

        self.db_path = self.state_dir / "scheduler_jobs.db"
        self._init_db()

        self.jobs: Dict[str, ScheduledJob] = {}
        self._job_handlers: Dict[str, Callable] = {}
        self._load_jobs()

        # Initialize APScheduler if available
        self.scheduler = None
        if APSCHEDULER_AVAILABLE:
            jobstores = {
                'default': SQLAlchemyJobStore(url=f'sqlite:///{self.db_path}')
            }
            executors = {
                'default': ThreadPoolExecutor(3)
            }
            job_defaults = {
                'coalesce': True,  # Combine missed runs into one
                'max_instances': 1,
                'misfire_grace_time': 3600  # 1 hour grace period
            }

            if blocking:
                self.scheduler = BlockingScheduler(
                    jobstores=jobstores,
                    executors=executors,
                    job_defaults=job_defaults,
                    timezone=timezone.utc
                )
            else:
                self.scheduler = BackgroundScheduler(
                    jobstores=jobstores,
                    executors=executors,
                    job_defaults=job_defaults,
                    timezone=timezone.utc
                )

            if auto_start:
                self.start()

        logger.info(f"{self.name} initialized (APScheduler: {APSCHEDULER_AVAILABLE})")

    def _init_db(self):
        """Initialize job tracking database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scheduled_jobs (
                    job_id TEXT PRIMARY KEY,
                    channel_id TEXT,
                    job_type TEXT,
                    schedule_time TEXT,
                    posting_days TEXT,
                    status TEXT DEFAULT 'pending',
                    last_run TEXT,
                    next_run TEXT,
                    run_count INTEGER DEFAULT 0,
                    fail_count INTEGER DEFAULT 0,
                    priority INTEGER DEFAULT 0,
                    catch_up INTEGER DEFAULT 1,
                    created_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS job_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT,
                    channel_id TEXT,
                    job_type TEXT,
                    status TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    duration_seconds REAL,
                    error TEXT,
                    result TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_job_history_job_id
                ON job_history(job_id)
            """)

    def _load_jobs(self):
        """Load scheduled jobs from database."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM scheduled_jobs").fetchall()
            for row in rows:
                job = ScheduledJob(
                    job_id=row[0],
                    channel_id=row[1],
                    job_type=row[2],
                    schedule_time=row[3],
                    posting_days=json.loads(row[4]) if row[4] else None,
                    status=row[5],
                    last_run=row[6],
                    next_run=row[7],
                    run_count=row[8],
                    fail_count=row[9],
                    priority=row[10],
                    catch_up=bool(row[11]),
                    created_at=row[12]
                )
                self.jobs[job.job_id] = job

    def _save_job(self, job: ScheduledJob):
        """Save job to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO scheduled_jobs
                (job_id, channel_id, job_type, schedule_time, posting_days,
                 status, last_run, next_run, run_count, fail_count, priority,
                 catch_up, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job.job_id, job.channel_id, job.job_type, job.schedule_time,
                json.dumps(job.posting_days) if job.posting_days else None,
                job.status, job.last_run, job.next_run, job.run_count,
                job.fail_count, job.priority, int(job.catch_up), job.created_at
            ))
        self.jobs[job.job_id] = job

    def register_handler(self, job_type: str, handler: Callable):
        """Register a handler for a job type."""
        self._job_handlers[job_type] = handler
        logger.debug(f"Registered handler for job type: {job_type}")

    def run(
        self,
        action: str = "list",
        channel_id: str = None,
        job_type: str = None,
        schedule_time: str = None,
        posting_days: List[int] = None,
        job_id: str = None,
        **kwargs
    ) -> ScheduleResult:
        """
        Execute a scheduler action.

        Args:
            action: Action to perform (add, remove, list, pause, resume, run_now)
            channel_id: Channel for the job
            job_type: Type of job (video, short, cleanup)
            schedule_time: Time in HH:MM format (UTC)
            posting_days: Days to run (0=Mon, 6=Sun)
            job_id: Specific job ID (for remove/pause/resume)
            **kwargs: Additional parameters

        Returns:
            ScheduleResult with operation outcome
        """
        logger.info(f"[{self.name}] Action: {action}")

        if action == "add":
            return self.add_job(channel_id, job_type, schedule_time, posting_days, **kwargs)
        elif action == "remove":
            return self.remove_job(job_id)
        elif action == "list":
            return self.list_jobs(channel_id)
        elif action == "pause":
            return self.pause_job(job_id)
        elif action == "resume":
            return self.resume_job(job_id)
        elif action == "run_now":
            return self.run_job_now(job_id)
        elif action == "status":
            return self.get_status(job_id)
        elif action == "load_config":
            return self.load_from_config()
        else:
            return ScheduleResult(
                job_id="",
                status="error",
                next_run=None,
                action=action,
                success=False,
                message=f"Unknown action: {action}"
            )

    def add_job(
        self,
        channel_id: str,
        job_type: str,
        schedule_time: str,
        posting_days: List[int] = None,
        priority: int = 0,
        catch_up: bool = True
    ) -> ScheduleResult:
        """
        Add a new scheduled job.

        Args:
            channel_id: Channel ID
            job_type: Type of job (video, short)
            schedule_time: Time in HH:MM format (UTC)
            posting_days: Days to post (0=Mon, 6=Sun), None=everyday
            priority: Job priority (higher = more important)
            catch_up: Run missed schedules

        Returns:
            ScheduleResult with job details
        """
        # Generate job ID
        time_part = schedule_time.replace(":", "")
        job_id = f"{channel_id}_{job_type}_{time_part}"

        logger.info(f"[{self.name}] Adding job: {job_id}")

        # Calculate next run time
        next_run = self._calculate_next_run(schedule_time, posting_days)

        job = ScheduledJob(
            job_id=job_id,
            channel_id=channel_id,
            job_type=job_type,
            schedule_time=schedule_time,
            posting_days=posting_days,
            next_run=next_run.isoformat() if next_run else None,
            priority=priority,
            catch_up=catch_up
        )

        self._save_job(job)

        # Add to APScheduler if available and running
        if self.scheduler and self.scheduler.running:
            self._add_to_scheduler(job)

        return ScheduleResult(
            job_id=job_id,
            status=JobStatus.PENDING.value,
            next_run=job.next_run,
            action="add",
            channel_id=channel_id,
            job_type=job_type,
            success=True,
            message=f"Job {job_id} scheduled for {schedule_time} UTC"
        )

    def remove_job(self, job_id: str) -> ScheduleResult:
        """Remove a scheduled job."""
        if job_id not in self.jobs:
            return ScheduleResult(
                job_id=job_id,
                status="not_found",
                next_run=None,
                action="remove",
                success=False,
                message=f"Job {job_id} not found"
            )

        job = self.jobs[job_id]

        # Remove from APScheduler
        if self.scheduler:
            try:
                self.scheduler.remove_job(job_id)
            except:
                pass

        # Remove from database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM scheduled_jobs WHERE job_id = ?", (job_id,))

        del self.jobs[job_id]

        return ScheduleResult(
            job_id=job_id,
            status="removed",
            next_run=None,
            action="remove",
            channel_id=job.channel_id,
            job_type=job.job_type,
            success=True,
            message=f"Job {job_id} removed"
        )

    def pause_job(self, job_id: str) -> ScheduleResult:
        """Pause a scheduled job."""
        if job_id not in self.jobs:
            return ScheduleResult(
                job_id=job_id,
                status="not_found",
                next_run=None,
                action="pause",
                success=False,
                message=f"Job {job_id} not found"
            )

        job = self.jobs[job_id]
        job.status = JobStatus.PAUSED.value
        self._save_job(job)

        # Pause in APScheduler
        if self.scheduler:
            try:
                self.scheduler.pause_job(job_id)
            except:
                pass

        return ScheduleResult(
            job_id=job_id,
            status=JobStatus.PAUSED.value,
            next_run=job.next_run,
            action="pause",
            channel_id=job.channel_id,
            success=True,
            message=f"Job {job_id} paused"
        )

    def resume_job(self, job_id: str) -> ScheduleResult:
        """Resume a paused job."""
        if job_id not in self.jobs:
            return ScheduleResult(
                job_id=job_id,
                status="not_found",
                next_run=None,
                action="resume",
                success=False,
                message=f"Job {job_id} not found"
            )

        job = self.jobs[job_id]
        job.status = JobStatus.PENDING.value
        job.next_run = self._calculate_next_run(job.schedule_time, job.posting_days).isoformat()
        self._save_job(job)

        # Resume in APScheduler
        if self.scheduler:
            try:
                self.scheduler.resume_job(job_id)
            except:
                pass

        return ScheduleResult(
            job_id=job_id,
            status=JobStatus.PENDING.value,
            next_run=job.next_run,
            action="resume",
            channel_id=job.channel_id,
            success=True,
            message=f"Job {job_id} resumed"
        )

    def run_job_now(self, job_id: str) -> ScheduleResult:
        """Execute a job immediately."""
        if job_id not in self.jobs:
            return ScheduleResult(
                job_id=job_id,
                status="not_found",
                next_run=None,
                action="run_now",
                success=False,
                message=f"Job {job_id} not found"
            )

        job = self.jobs[job_id]
        logger.info(f"[{self.name}] Running job immediately: {job_id}")

        result = self._execute_job(job)

        return ScheduleResult(
            job_id=job_id,
            status=job.status,
            next_run=job.next_run,
            action="run_now",
            channel_id=job.channel_id,
            job_type=job.job_type,
            success=result.get("success", False),
            data=result,
            message=result.get("message", "Job executed")
        )

    def list_jobs(self, channel_id: str = None) -> ScheduleResult:
        """List all scheduled jobs, optionally filtered by channel."""
        jobs_list = []

        for jid, job in self.jobs.items():
            if channel_id and job.channel_id != channel_id:
                continue

            jobs_list.append({
                "job_id": job.job_id,
                "channel_id": job.channel_id,
                "job_type": job.job_type,
                "schedule_time": job.schedule_time,
                "posting_days": job.posting_days,
                "status": job.status,
                "next_run": job.next_run,
                "last_run": job.last_run,
                "run_count": job.run_count,
                "priority": job.priority
            })

        # Sort by priority (desc) then next_run
        jobs_list.sort(key=lambda x: (-x["priority"], x["next_run"] or ""))

        return ScheduleResult(
            job_id="",
            status="success",
            next_run=None,
            action="list",
            channel_id=channel_id,
            success=True,
            data={"jobs": jobs_list, "count": len(jobs_list)},
            message=f"Found {len(jobs_list)} jobs"
        )

    def get_status(self, job_id: str) -> ScheduleResult:
        """Get status of a specific job."""
        if job_id not in self.jobs:
            return ScheduleResult(
                job_id=job_id,
                status="not_found",
                next_run=None,
                action="status",
                success=False,
                message=f"Job {job_id} not found"
            )

        job = self.jobs[job_id]
        return ScheduleResult(
            job_id=job_id,
            status=job.status,
            next_run=job.next_run,
            action="status",
            channel_id=job.channel_id,
            job_type=job.job_type,
            success=True,
            data=job.to_dict()
        )

    def get_next_run(self, channel_id: str = None, job_type: str = None) -> Optional[str]:
        """Get the next scheduled run time, optionally filtered."""
        next_runs = []

        for job in self.jobs.values():
            if channel_id and job.channel_id != channel_id:
                continue
            if job_type and job.job_type != job_type:
                continue
            if job.status == JobStatus.PAUSED.value:
                continue
            if job.next_run:
                next_runs.append(job.next_run)

        if next_runs:
            return min(next_runs)
        return None

    def load_from_config(self) -> ScheduleResult:
        """
        Load schedule from channels.yaml configuration.

        Returns:
            ScheduleResult with loaded jobs
        """
        logger.info(f"[{self.name}] Loading schedule from config...")

        config_path = Path(__file__).parent.parent.parent / "config" / "channels.yaml"
        if not config_path.exists():
            return ScheduleResult(
                job_id="",
                status="error",
                next_run=None,
                action="load_config",
                success=False,
                message=f"Config file not found: {config_path}"
            )

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            jobs_added = 0

            # Load posting schedule from daily_scheduler module
            from ..scheduler.daily_scheduler import POSTING_SCHEDULE

            for channel_id, channel_config in POSTING_SCHEDULE.items():
                # Get posting days from channels.yaml
                posting_days = None
                for channel in config.get("channels", []):
                    if channel.get("id") == channel_id:
                        settings = channel.get("settings", {})
                        posting_days = settings.get("posting_days")
                        break

                # Add video jobs
                for schedule_time in channel_config.get("times", []):
                    result = self.add_job(
                        channel_id=channel_id,
                        job_type="video",
                        schedule_time=schedule_time,
                        posting_days=posting_days,
                        priority=1
                    )
                    if result.success:
                        jobs_added += 1

                # Add Shorts jobs from channel config
                for channel in config.get("channels", []):
                    if channel.get("id") == channel_id:
                        shorts_config = channel.get("shorts_schedule", {})
                        if shorts_config.get("enabled", True):
                            delay_hours = shorts_config.get("delay_hours", 2)

                            # Calculate Shorts times (2 hours after regular)
                            for schedule_time in channel_config.get("times", []):
                                hour, minute = map(int, schedule_time.split(":"))
                                short_hour = (hour + delay_hours) % 24
                                short_time = f"{short_hour:02d}:{minute:02d}"

                                result = self.add_job(
                                    channel_id=channel_id,
                                    job_type="short",
                                    schedule_time=short_time,
                                    posting_days=posting_days,
                                    priority=0
                                )
                                if result.success:
                                    jobs_added += 1

            # Add cleanup job (daily at 04:00 UTC)
            self.add_job(
                channel_id="_system",
                job_type="cleanup",
                schedule_time="04:00",
                posting_days=None,  # Everyday
                priority=-1
            )
            jobs_added += 1

            return ScheduleResult(
                job_id="",
                status="success",
                next_run=self.get_next_run(),
                action="load_config",
                success=True,
                data={"jobs_added": jobs_added},
                message=f"Loaded {jobs_added} jobs from config"
            )

        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return ScheduleResult(
                job_id="",
                status="error",
                next_run=None,
                action="load_config",
                success=False,
                message=str(e)
            )

    def _calculate_next_run(
        self,
        schedule_time: str,
        posting_days: List[int] = None
    ) -> datetime:
        """Calculate the next run time in UTC."""
        hour, minute = map(int, schedule_time.split(":"))
        now = datetime.now(timezone.utc)

        # Start with today at the scheduled time
        next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

        # If that time has passed, start tomorrow
        if next_run <= now:
            next_run += timedelta(days=1)

        # Find the next valid posting day
        if posting_days:
            while next_run.weekday() not in posting_days:
                next_run += timedelta(days=1)

        return next_run

    def _add_to_scheduler(self, job: ScheduledJob):
        """Add a job to the APScheduler."""
        if not self.scheduler:
            return

        hour, minute = map(int, job.schedule_time.split(":"))

        # Build day_of_week for cron
        if job.posting_days:
            day_of_week = ",".join([self.DAY_NAMES[d] for d in job.posting_days])
        else:
            day_of_week = "*"

        trigger = CronTrigger(
            hour=hour,
            minute=minute,
            day_of_week=day_of_week,
            timezone=timezone.utc
        )

        self.scheduler.add_job(
            func=self._execute_job,
            trigger=trigger,
            args=[job],
            id=job.job_id,
            name=f"{job.channel_id} {job.job_type} at {job.schedule_time}",
            misfire_grace_time=3600,  # 1 hour
            coalesce=True
        )

        logger.info(f"[{self.name}] Added to scheduler: {job.job_id}")

    def _execute_job(self, job: ScheduledJob) -> Dict[str, Any]:
        """Execute a scheduled job."""
        logger.info(f"[{self.name}] Executing job: {job.job_id}")

        job.status = JobStatus.RUNNING.value
        job.last_run = datetime.now(timezone.utc).isoformat()
        self._save_job(job)

        start_time = datetime.now()

        try:
            handler = self._job_handlers.get(job.job_type)
            if handler:
                result = handler(job.channel_id, job.job_type)
            else:
                # Default execution using existing functions
                result = self._default_job_handler(job)

            # Update job status
            job.status = JobStatus.COMPLETED.value
            job.run_count += 1
            job.next_run = self._calculate_next_run(
                job.schedule_time, job.posting_days
            ).isoformat()
            self._save_job(job)

            # Record history
            self._record_job_history(job, start_time, "completed", result)

            return {"success": True, "result": result}

        except Exception as e:
            logger.error(f"Job {job.job_id} failed: {e}")

            job.status = JobStatus.FAILED.value
            job.fail_count += 1
            job.next_run = self._calculate_next_run(
                job.schedule_time, job.posting_days
            ).isoformat()
            self._save_job(job)

            # Record history
            self._record_job_history(job, start_time, "failed", error=str(e))

            return {"success": False, "error": str(e)}

    def _default_job_handler(self, job: ScheduledJob) -> Dict[str, Any]:
        """Default handler using existing scheduler functions."""
        try:
            if job.job_type == "video":
                from ..scheduler.daily_scheduler import create_and_upload_video
                return create_and_upload_video(job.channel_id)

            elif job.job_type == "short":
                from ..scheduler.daily_scheduler import create_and_upload_short
                return create_and_upload_short(job.channel_id)

            elif job.job_type == "cleanup":
                from ..scheduler.daily_scheduler import run_scheduled_cleanup
                return run_scheduled_cleanup()

            else:
                return {"success": False, "error": f"Unknown job type: {job.job_type}"}

        except ImportError as e:
            return {"success": False, "error": f"Handler import failed: {e}"}

    def _record_job_history(
        self,
        job: ScheduledJob,
        start_time: datetime,
        status: str,
        result: Any = None,
        error: str = None
    ):
        """Record job execution history."""
        completed_at = datetime.now()
        duration = (completed_at - start_time).total_seconds()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO job_history
                (job_id, channel_id, job_type, status, started_at, completed_at,
                 duration_seconds, error, result)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job.job_id, job.channel_id, job.job_type, status,
                start_time.isoformat(), completed_at.isoformat(),
                duration, error, json.dumps(result) if result else None
            ))

    def start(self):
        """Start the scheduler."""
        if not self.scheduler:
            logger.error("APScheduler not available")
            return

        if self.scheduler.running:
            logger.warning("Scheduler already running")
            return

        # Add all jobs to scheduler
        for job in self.jobs.values():
            if job.status != JobStatus.PAUSED.value:
                self._add_to_scheduler(job)

        self.scheduler.start()
        logger.info(f"[{self.name}] Scheduler started with {len(self.jobs)} jobs")

    def stop(self):
        """Stop the scheduler."""
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown()
            logger.info(f"[{self.name}] Scheduler stopped")

    def check_missed_schedules(self) -> List[Dict[str, Any]]:
        """
        Check for and handle missed schedules.

        Returns:
            List of missed jobs that were executed
        """
        logger.info(f"[{self.name}] Checking for missed schedules...")

        executed = []
        now = datetime.now(timezone.utc)

        for job in self.jobs.values():
            if job.status == JobStatus.PAUSED.value:
                continue
            if not job.catch_up:
                continue
            if not job.next_run:
                continue

            next_run = datetime.fromisoformat(job.next_run.replace('Z', '+00:00'))
            if next_run.tzinfo is None:
                next_run = next_run.replace(tzinfo=timezone.utc)

            # Check if we missed a run (scheduled time passed but status is still pending)
            if next_run < now and job.status == JobStatus.PENDING.value:
                logger.info(f"[{self.name}] Running missed job: {job.job_id}")

                result = self._execute_job(job)
                executed.append({
                    "job_id": job.job_id,
                    "scheduled_for": job.next_run,
                    "executed_at": datetime.now().isoformat(),
                    "result": result
                })

        if executed:
            logger.info(f"[{self.name}] Executed {len(executed)} missed jobs")

        return executed

    def get_schedule_summary(self) -> Dict[str, Any]:
        """Get a summary of the current schedule."""
        summary = {
            "total_jobs": len(self.jobs),
            "by_status": {},
            "by_type": {},
            "by_channel": {},
            "next_run": self.get_next_run()
        }

        for job in self.jobs.values():
            # By status
            summary["by_status"][job.status] = summary["by_status"].get(job.status, 0) + 1

            # By type
            summary["by_type"][job.job_type] = summary["by_type"].get(job.job_type, 0) + 1

            # By channel
            summary["by_channel"][job.channel_id] = summary["by_channel"].get(job.channel_id, 0) + 1

        return summary


# CLI entry point
if __name__ == "__main__":
    import sys

    print("\n" + "=" * 60)
    print("  SCHEDULER AGENT")
    print("=" * 60 + "\n")

    agent = SchedulerAgent()

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "list":
            result = agent.run(action="list")
            print(f"Jobs ({result.data['count']}):\n")
            for job in result.data["jobs"]:
                status_icon = {
                    "pending": "[*]",
                    "running": "[>]",
                    "completed": "[v]",
                    "failed": "[x]",
                    "paused": "[-]"
                }.get(job["status"], "[?]")
                print(f"  {status_icon} {job['job_id']}")
                print(f"      Time: {job['schedule_time']} UTC, Next: {job['next_run']}")

        elif command == "load":
            result = agent.load_from_config()
            print(f"Result: {result.message}")
            print(f"Jobs added: {result.data.get('jobs_added', 0)}")

        elif command == "add" and len(sys.argv) >= 5:
            result = agent.add_job(
                channel_id=sys.argv[2],
                job_type=sys.argv[3],
                schedule_time=sys.argv[4]
            )
            print(f"Result: {result.message}")
            print(f"Next run: {result.next_run}")

        elif command == "remove" and len(sys.argv) > 2:
            result = agent.remove_job(sys.argv[2])
            print(f"Result: {result.message}")

        elif command == "run" and len(sys.argv) > 2:
            result = agent.run_job_now(sys.argv[2])
            print(f"Result: {result.message}")
            print(f"Success: {result.success}")

        elif command == "summary":
            summary = agent.get_schedule_summary()
            print(f"Total jobs: {summary['total_jobs']}")
            print(f"Next run: {summary['next_run']}")
            print(f"\nBy status: {summary['by_status']}")
            print(f"By type: {summary['by_type']}")
            print(f"By channel: {summary['by_channel']}")

        elif command == "missed":
            executed = agent.check_missed_schedules()
            print(f"Executed {len(executed)} missed jobs")
            for job in executed:
                print(f"  {job['job_id']}: {job['result'].get('success')}")

    else:
        print("Usage:")
        print("  python -m src.agents.scheduler_agent list           # List all jobs")
        print("  python -m src.agents.scheduler_agent load           # Load from config")
        print("  python -m src.agents.scheduler_agent add <ch> <type> <time>  # Add job")
        print("  python -m src.agents.scheduler_agent remove <job_id>  # Remove job")
        print("  python -m src.agents.scheduler_agent run <job_id>   # Run job now")
        print("  python -m src.agents.scheduler_agent summary        # Show summary")
        print("  python -m src.agents.scheduler_agent missed         # Check missed")
        print()
        print("Examples:")
        print("  python -m src.agents.scheduler_agent add money_blueprints video 15:00")
        print("  python -m src.agents.scheduler_agent run money_blueprints_video_1500")
