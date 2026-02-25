"""
Job Tracker - SQLite-backed async job registry
Tracks video creation, uploads, and other long-running tasks
"""

import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
import uuid
from pathlib import Path


class JobTracker:
    """SQLite-backed job tracker for async operations"""

    def __init__(self, db_path: str = "data/jobs.db"):
        """Initialize job tracker with database path"""
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    channel_id TEXT,
                    topic TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    progress INTEGER DEFAULT 0,
                    result TEXT,
                    error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_status ON jobs(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_channel ON jobs(channel_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_created ON jobs(created_at DESC)
            """)
            conn.commit()

    def create_job(self, job_type: str, channel_id: Optional[str] = None,
                   topic: Optional[str] = None) -> str:
        """
        Create a new job and return job_id

        Args:
            job_type: Type of job (video, short, batch, scheduler)
            channel_id: Optional channel ID
            topic: Optional topic

        Returns:
            job_id (UUID)
        """
        job_id = str(uuid.uuid4())

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO jobs (id, job_type, channel_id, topic, status)
                VALUES (?, ?, ?, ?, 'pending')
            """, (job_id, job_type, channel_id, topic))
            conn.commit()

        return job_id

    def update_job(self, job_id: str, status: str, progress: int = None,
                   result: Dict[str, Any] = None, error: str = None):
        """
        Update job status

        Args:
            job_id: Job ID
            status: New status (pending, running, completed, failed)
            progress: Progress percentage (0-100)
            result: Result data if completed
            error: Error message if failed
        """
        result_json = json.dumps(result) if result else None

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE jobs
                SET status = ?, progress = ?, result = ?, error = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (status, progress, result_json, error, job_id))
            conn.commit()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job by ID

        Args:
            job_id: Job ID

        Returns:
            Job dict or None
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, job_type, channel_id, topic, status, progress, result, error, created_at, updated_at
                FROM jobs
                WHERE id = ?
            """, (job_id,))
            row = cursor.fetchone()

            if not row:
                return None

            job = dict(row)
            if job['result']:
                job['result'] = json.loads(job['result'])
            return job

    def list_jobs(self, limit: int = 20, status: Optional[str] = None,
                  channel_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List recent jobs

        Args:
            limit: Max jobs to return
            status: Filter by status (optional)
            channel_id: Filter by channel (optional)

        Returns:
            List of jobs
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = "SELECT * FROM jobs WHERE 1=1"
            params = []

            if status:
                query += " AND status = ?"
                params.append(status)

            if channel_id:
                query += " AND channel_id = ?"
                params.append(channel_id)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            jobs = []
            for row in rows:
                job = dict(row)
                if job['result']:
                    job['result'] = json.loads(job['result'])
                jobs.append(job)

            return jobs

    def cleanup_old_jobs(self, days: int = 7):
        """
        Delete completed/failed jobs older than N days

        Args:
            days: Delete jobs older than this many days
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM jobs
                WHERE status IN ('completed', 'failed')
                AND created_at < datetime('now', '-' || ? || ' days')
            """, (days,))
            conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        """Get job statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Count by status
            cursor.execute("""
                SELECT status, COUNT(*) as count
                FROM jobs
                GROUP BY status
            """)
            stats_by_status = {row[0]: row[1] for row in cursor.fetchall()}

            # Count by type
            cursor.execute("""
                SELECT job_type, COUNT(*) as count
                FROM jobs
                GROUP BY job_type
            """)
            stats_by_type = {row[0]: row[1] for row in cursor.fetchall()}

            # Total jobs
            cursor.execute("SELECT COUNT(*) FROM jobs")
            total = cursor.fetchone()[0]

            return {
                "total_jobs": total,
                "by_status": stats_by_status,
                "by_type": stats_by_type
            }


# Global job tracker instance
_job_tracker = None


def get_job_tracker(db_path: str = "data/jobs.db") -> JobTracker:
    """Get or create job tracker instance"""
    global _job_tracker
    if _job_tracker is None:
        _job_tracker = JobTracker(db_path)
    return _job_tracker
