"""
Monitor Agent - System Health Monitoring

Monitors system health, API rate limits, token usage, and resource utilization
for the YouTube automation pipeline.

Usage:
    from src.agents.monitor_agent import MonitorAgent

    agent = MonitorAgent()

    # Get full health check
    result = agent.run()

    # Check specific components
    api_status = agent.check_api_limits()
    resources = agent.check_resources()
    jobs = agent.check_stuck_jobs()
"""

import os
import json
import shutil
import sqlite3

# psutil is optional - provides system resource monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from loguru import logger


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class HealthAlert:
    """Represents a health alert."""
    severity: str
    category: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class APIStatus:
    """API rate limit and usage status."""
    provider: str
    daily_limit: int
    used_today: int
    remaining: int
    reset_time: str
    status: str = HealthStatus.HEALTHY.value
    warning_threshold: float = 0.8

    @property
    def usage_percent(self) -> float:
        if self.daily_limit == 0:
            return 0
        return (self.used_today / self.daily_limit) * 100

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["usage_percent"] = self.usage_percent
        return data


@dataclass
class ResourceUsage:
    """System resource usage."""
    disk_total_gb: float
    disk_free_gb: float
    disk_used_percent: float
    memory_total_gb: float
    memory_available_gb: float
    memory_used_percent: float
    output_dir_size_gb: float
    data_dir_size_gb: float
    status: str = HealthStatus.HEALTHY.value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HealthResult:
    """Result from health monitoring."""
    status: str  # healthy, warning, critical
    api_status: Dict[str, Dict[str, Any]]
    resource_usage: Dict[str, Any]
    alerts: List[Dict[str, Any]]
    stuck_jobs: List[Dict[str, Any]] = field(default_factory=list)
    token_usage: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    summary: str = ""

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


class MonitorAgent(BaseAgent):
    """
    Monitor Agent - System health monitoring.

    Tracks:
    - API rate limits (YouTube 10k/day, Groq, etc.)
    - Token usage and daily costs
    - System resources (disk space, memory)
    - Stuck jobs (running > 30 minutes)
    """

    # API Rate Limits (daily quotas)
    API_LIMITS = {
        "youtube": {
            "daily_limit": 10000,  # YouTube Data API quota units
            "warning_threshold": 0.8,
            "description": "YouTube Data API"
        },
        "groq": {
            "daily_limit": 14400,  # Free tier: 30 requests/min * 60 * 8 hours
            "warning_threshold": 0.9,
            "description": "Groq API (free tier)"
        },
        "pexels": {
            "daily_limit": 200,  # Requests per hour * 24
            "warning_threshold": 0.8,
            "description": "Pexels Stock Footage"
        },
        "pixabay": {
            "daily_limit": 5000,
            "warning_threshold": 0.8,
            "description": "Pixabay Stock Footage"
        },
        "fish_audio": {
            "daily_limit": 1000,  # Estimated based on typical usage
            "warning_threshold": 0.9,
            "description": "Fish Audio TTS"
        }
    }

    # Resource thresholds
    DISK_WARNING_GB = 10  # Warn if less than 10 GB free
    DISK_CRITICAL_GB = 5  # Critical if less than 5 GB free
    MEMORY_WARNING_PERCENT = 85  # Warn if memory usage > 85%
    MEMORY_CRITICAL_PERCENT = 95  # Critical if > 95%
    OUTPUT_DIR_WARNING_GB = 50  # Warn if output dir > 50 GB

    # Job monitoring
    STUCK_JOB_THRESHOLD_MINUTES = 30

    def __init__(self):
        super().__init__("MonitorAgent")
        self.alerts: List[HealthAlert] = []
        self.alerts_file = self.state_dir / "health_alerts.json"
        self.api_usage_file = self.state_dir / "api_usage.json"
        logger.info(f"{self.name} initialized")

    def run(self, **kwargs) -> HealthResult:
        """
        Run a complete health check.

        Returns:
            HealthResult with overall status, API status, resource usage, and alerts
        """
        logger.info(f"[{self.name}] Running health check...")
        self.alerts = []

        # Check all components
        api_status = self.check_api_limits()
        resource_usage = self.check_resources()
        token_usage = self.check_token_usage()
        stuck_jobs = self.check_stuck_jobs()

        # Determine overall status
        overall_status = HealthStatus.HEALTHY.value

        # Check for critical alerts
        for alert in self.alerts:
            if alert.severity == AlertSeverity.CRITICAL.value:
                overall_status = HealthStatus.CRITICAL.value
                break
            elif alert.severity == AlertSeverity.WARNING.value:
                if overall_status != HealthStatus.CRITICAL.value:
                    overall_status = HealthStatus.WARNING.value

        # Generate summary
        summary_parts = []
        if overall_status == HealthStatus.HEALTHY.value:
            summary_parts.append("All systems operational")
        else:
            critical_count = sum(1 for a in self.alerts if a.severity == AlertSeverity.CRITICAL.value)
            warning_count = sum(1 for a in self.alerts if a.severity == AlertSeverity.WARNING.value)
            if critical_count:
                summary_parts.append(f"{critical_count} critical issues")
            if warning_count:
                summary_parts.append(f"{warning_count} warnings")

        if stuck_jobs:
            summary_parts.append(f"{len(stuck_jobs)} stuck jobs")

        summary = ". ".join(summary_parts) if summary_parts else "System healthy"

        # Save alerts
        self._save_alerts()

        result = HealthResult(
            status=overall_status,
            api_status={k: v.to_dict() for k, v in api_status.items()},
            resource_usage=resource_usage.to_dict(),
            alerts=[a.to_dict() for a in self.alerts],
            stuck_jobs=stuck_jobs,
            token_usage=token_usage,
            summary=summary
        )

        logger.info(f"[{self.name}] Health check complete: {overall_status}")
        return result

    def check_api_limits(self) -> Dict[str, APIStatus]:
        """
        Check API rate limits for all providers.

        Returns:
            Dictionary of provider -> APIStatus
        """
        logger.debug(f"[{self.name}] Checking API limits...")
        results = {}

        # Load API usage from tracking file
        api_usage = self._load_api_usage()
        today = datetime.now().strftime("%Y-%m-%d")

        for provider, config in self.API_LIMITS.items():
            usage_today = api_usage.get(provider, {}).get(today, 0)
            daily_limit = config["daily_limit"]
            remaining = max(0, daily_limit - usage_today)
            warning_threshold = config["warning_threshold"]

            # Determine status
            usage_percent = usage_today / daily_limit if daily_limit > 0 else 0
            if usage_percent >= 1.0:
                status = HealthStatus.CRITICAL.value
                self.alerts.append(HealthAlert(
                    severity=AlertSeverity.CRITICAL.value,
                    category="api_limit",
                    message=f"{provider} API limit exhausted",
                    details={"provider": provider, "used": usage_today, "limit": daily_limit}
                ))
            elif usage_percent >= warning_threshold:
                status = HealthStatus.WARNING.value
                self.alerts.append(HealthAlert(
                    severity=AlertSeverity.WARNING.value,
                    category="api_limit",
                    message=f"{provider} API at {usage_percent*100:.0f}% of daily limit",
                    details={"provider": provider, "used": usage_today, "limit": daily_limit}
                ))
            else:
                status = HealthStatus.HEALTHY.value

            # Calculate reset time (midnight UTC)
            tomorrow = (datetime.now() + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            reset_time = tomorrow.isoformat()

            results[provider] = APIStatus(
                provider=provider,
                daily_limit=daily_limit,
                used_today=usage_today,
                remaining=remaining,
                reset_time=reset_time,
                status=status,
                warning_threshold=warning_threshold
            )

        return results

    def check_resources(self) -> ResourceUsage:
        """
        Check system resources (disk, memory).

        Returns:
            ResourceUsage with current usage
        """
        logger.debug(f"[{self.name}] Checking resources...")

        project_root = Path(__file__).parent.parent.parent

        # Get disk usage - use psutil if available, otherwise use shutil
        if PSUTIL_AVAILABLE:
            disk = psutil.disk_usage(str(project_root))
            disk_total_gb = disk.total / (1024**3)
            disk_free_gb = disk.free / (1024**3)
            disk_used_percent = disk.percent
        else:
            # Fallback to shutil for disk usage
            total, used, free = shutil.disk_usage(str(project_root))
            disk_total_gb = total / (1024**3)
            disk_free_gb = free / (1024**3)
            disk_used_percent = (used / total) * 100 if total > 0 else 0

        # Get memory usage - requires psutil
        if PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            memory_total_gb = memory.total / (1024**3)
            memory_available_gb = memory.available / (1024**3)
            memory_used_percent = memory.percent
        else:
            # Default values when psutil not available
            memory_total_gb = 0
            memory_available_gb = 0
            memory_used_percent = 0
            logger.warning("psutil not available - memory monitoring disabled")

        # Get output directory size
        output_dir = project_root / "output"
        output_size_gb = self._get_dir_size(output_dir) / (1024**3)

        # Get data directory size
        data_dir = project_root / "data"
        data_size_gb = self._get_dir_size(data_dir) / (1024**3)

        # Determine status and generate alerts
        status = HealthStatus.HEALTHY.value

        # Disk space checks
        if disk_free_gb < self.DISK_CRITICAL_GB:
            status = HealthStatus.CRITICAL.value
            self.alerts.append(HealthAlert(
                severity=AlertSeverity.CRITICAL.value,
                category="disk_space",
                message=f"Critical: Only {disk_free_gb:.1f} GB disk space remaining",
                details={"free_gb": disk_free_gb, "threshold_gb": self.DISK_CRITICAL_GB}
            ))
        elif disk_free_gb < self.DISK_WARNING_GB:
            if status != HealthStatus.CRITICAL.value:
                status = HealthStatus.WARNING.value
            self.alerts.append(HealthAlert(
                severity=AlertSeverity.WARNING.value,
                category="disk_space",
                message=f"Low disk space: {disk_free_gb:.1f} GB remaining",
                details={"free_gb": disk_free_gb, "threshold_gb": self.DISK_WARNING_GB}
            ))

        # Memory checks (only if psutil is available)
        if PSUTIL_AVAILABLE:
            if memory_used_percent > self.MEMORY_CRITICAL_PERCENT:
                status = HealthStatus.CRITICAL.value
                self.alerts.append(HealthAlert(
                    severity=AlertSeverity.CRITICAL.value,
                    category="memory",
                    message=f"Critical: Memory usage at {memory_used_percent:.0f}%",
                    details={"used_percent": memory_used_percent, "available_gb": memory_available_gb}
                ))
            elif memory_used_percent > self.MEMORY_WARNING_PERCENT:
                if status != HealthStatus.CRITICAL.value:
                    status = HealthStatus.WARNING.value
                self.alerts.append(HealthAlert(
                    severity=AlertSeverity.WARNING.value,
                    category="memory",
                    message=f"High memory usage: {memory_used_percent:.0f}%",
                    details={"used_percent": memory_used_percent, "available_gb": memory_available_gb}
                ))

        # Output directory size check
        if output_size_gb > self.OUTPUT_DIR_WARNING_GB:
            self.alerts.append(HealthAlert(
                severity=AlertSeverity.WARNING.value,
                category="storage",
                message=f"Output directory is {output_size_gb:.1f} GB (recommend cleanup)",
                details={"size_gb": output_size_gb, "threshold_gb": self.OUTPUT_DIR_WARNING_GB}
            ))

        return ResourceUsage(
            disk_total_gb=disk_total_gb,
            disk_free_gb=disk_free_gb,
            disk_used_percent=disk_used_percent,
            memory_total_gb=memory_total_gb,
            memory_available_gb=memory_available_gb,
            memory_used_percent=memory_used_percent,
            output_dir_size_gb=output_size_gb,
            data_dir_size_gb=data_size_gb,
            status=status
        )

    def check_token_usage(self) -> Dict[str, Any]:
        """
        Check token usage and costs.

        Returns:
            Dictionary with token usage statistics
        """
        logger.debug(f"[{self.name}] Checking token usage...")

        try:
            from ..utils.token_manager import get_token_manager

            tracker = get_token_manager()
            daily = tracker.get_daily_usage()
            weekly = tracker.get_weekly_usage()
            budget = tracker.check_budget()

            # Generate alerts for budget
            if budget["exceeded"]:
                self.alerts.append(HealthAlert(
                    severity=AlertSeverity.CRITICAL.value,
                    category="budget",
                    message=f"Daily budget exceeded: ${budget['spent_today']:.2f} of ${budget['daily_budget']:.2f}",
                    details=budget
                ))
            elif budget["warning"]:
                self.alerts.append(HealthAlert(
                    severity=AlertSeverity.WARNING.value,
                    category="budget",
                    message=f"Budget at {budget['usage_percent']:.0f}%: ${budget['spent_today']:.2f} of ${budget['daily_budget']:.2f}",
                    details=budget
                ))

            return {
                "daily": daily,
                "weekly": weekly,
                "budget": budget,
                "cost_per_video": tracker.get_cost_per_video()
            }

        except Exception as e:
            logger.warning(f"[{self.name}] Failed to check token usage: {e}")
            return {"error": str(e)}

    def check_stuck_jobs(self) -> List[Dict[str, Any]]:
        """
        Check for stuck jobs (running > 30 minutes).

        Returns:
            List of stuck job details
        """
        logger.debug(f"[{self.name}] Checking for stuck jobs...")
        stuck_jobs = []

        # Check workflow states
        workflow_dir = self.state_dir / "workflow_states" / "workflows"
        if workflow_dir.exists():
            for filepath in workflow_dir.glob("*.json"):
                try:
                    with open(filepath) as f:
                        state = json.load(f)

                    if state.get("status") == "running":
                        updated_at = datetime.fromisoformat(state.get("updated_at", state.get("created_at")))
                        minutes_running = (datetime.now() - updated_at).total_seconds() / 60

                        if minutes_running > self.STUCK_JOB_THRESHOLD_MINUTES:
                            stuck_job = {
                                "workflow_id": state.get("workflow_id"),
                                "channel_id": state.get("channel_id"),
                                "topic": state.get("topic"),
                                "current_step": state.get("current_step"),
                                "minutes_running": round(minutes_running, 1),
                                "started_at": state.get("created_at")
                            }
                            stuck_jobs.append(stuck_job)

                            self.alerts.append(HealthAlert(
                                severity=AlertSeverity.WARNING.value,
                                category="stuck_job",
                                message=f"Workflow {state.get('workflow_id')} stuck at {state.get('current_step')} for {minutes_running:.0f} minutes",
                                details=stuck_job
                            ))

                except Exception as e:
                    logger.warning(f"Failed to check workflow {filepath}: {e}")

        # Check scheduler jobs from database
        try:
            db_path = self.state_dir / "scheduler_jobs.db"
            if db_path.exists():
                with sqlite3.connect(db_path) as conn:
                    rows = conn.execute("""
                        SELECT job_id, channel_id, job_type, started_at, status
                        FROM scheduled_jobs
                        WHERE status = 'running'
                    """).fetchall()

                    for row in rows:
                        job_id, channel_id, job_type, started_at, status = row
                        started = datetime.fromisoformat(started_at)
                        minutes_running = (datetime.now() - started).total_seconds() / 60

                        if minutes_running > self.STUCK_JOB_THRESHOLD_MINUTES:
                            stuck_job = {
                                "job_id": job_id,
                                "channel_id": channel_id,
                                "job_type": job_type,
                                "minutes_running": round(minutes_running, 1),
                                "started_at": started_at
                            }
                            stuck_jobs.append(stuck_job)

        except Exception as e:
            logger.debug(f"No scheduler database or error: {e}")

        if stuck_jobs:
            logger.warning(f"[{self.name}] Found {len(stuck_jobs)} stuck jobs")

        return stuck_jobs

    def record_api_usage(self, provider: str, count: int = 1):
        """
        Record API usage for a provider.

        Args:
            provider: API provider name
            count: Number of requests/units to record
        """
        api_usage = self._load_api_usage()
        today = datetime.now().strftime("%Y-%m-%d")

        if provider not in api_usage:
            api_usage[provider] = {}

        api_usage[provider][today] = api_usage[provider].get(today, 0) + count
        self._save_api_usage(api_usage)

    def _load_api_usage(self) -> Dict[str, Dict[str, int]]:
        """Load API usage tracking data."""
        if self.api_usage_file.exists():
            try:
                with open(self.api_usage_file) as f:
                    return json.load(f)
            except:
                pass
        return {}

    def _save_api_usage(self, usage: Dict[str, Dict[str, int]]):
        """Save API usage tracking data."""
        with open(self.api_usage_file, "w") as f:
            json.dump(usage, f, indent=2)

    def _save_alerts(self):
        """Save alerts to file."""
        # Load existing alerts
        existing = []
        if self.alerts_file.exists():
            try:
                with open(self.alerts_file) as f:
                    existing = json.load(f)
            except:
                pass

        # Add new alerts
        for alert in self.alerts:
            existing.append(alert.to_dict())

        # Keep last 500 alerts
        existing = existing[-500:]

        with open(self.alerts_file, "w") as f:
            json.dump(existing, f, indent=2)

    def _get_dir_size(self, path: Path) -> int:
        """Get total size of a directory in bytes."""
        total = 0
        if path.exists():
            for entry in path.rglob("*"):
                if entry.is_file():
                    try:
                        total += entry.stat().st_size
                    except:
                        pass
        return total

    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get alert history from the last N hours.

        Args:
            hours: Number of hours to look back

        Returns:
            List of alerts
        """
        if not self.alerts_file.exists():
            return []

        try:
            with open(self.alerts_file) as f:
                all_alerts = json.load(f)

            cutoff = datetime.now() - timedelta(hours=hours)
            recent = []

            for alert in all_alerts:
                try:
                    timestamp = datetime.fromisoformat(alert.get("timestamp", ""))
                    if timestamp > cutoff:
                        recent.append(alert)
                except:
                    pass

            return recent

        except Exception as e:
            logger.error(f"Failed to load alert history: {e}")
            return []

    def get_summary(self) -> str:
        """Get a quick summary of system health."""
        result = self.run()
        return f"Status: {result.status.upper()} - {result.summary}"


# CLI entry point
if __name__ == "__main__":
    import sys

    print("\n" + "=" * 60)
    print("  MONITOR AGENT - HEALTH CHECK")
    print("=" * 60 + "\n")

    agent = MonitorAgent()

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "api":
            api_status = agent.check_api_limits()
            print("API Status:\n")
            for provider, status in api_status.items():
                print(f"  {provider}: {status.used_today}/{status.daily_limit} ({status.usage_percent:.0f}%) - {status.status}")

        elif command == "resources":
            resources = agent.check_resources()
            print("Resource Usage:\n")
            print(f"  Disk: {resources.disk_free_gb:.1f} GB free ({resources.disk_used_percent:.0f}% used)")
            print(f"  Memory: {resources.memory_available_gb:.1f} GB available ({resources.memory_used_percent:.0f}% used)")
            print(f"  Output dir: {resources.output_dir_size_gb:.2f} GB")

        elif command == "alerts":
            alerts = agent.get_alert_history(24)
            print(f"Alerts (last 24h): {len(alerts)}\n")
            for alert in alerts[-10:]:
                print(f"  [{alert['severity'].upper()}] {alert['category']}: {alert['message']}")

        elif command == "stuck":
            stuck = agent.check_stuck_jobs()
            print(f"Stuck Jobs: {len(stuck)}\n")
            for job in stuck:
                print(f"  {job.get('workflow_id', job.get('job_id'))}: {job.get('minutes_running', 0):.0f} min")

    else:
        result = agent.run()

        print(f"Overall Status: {result.status.upper()}")
        print(f"Summary: {result.summary}")
        print()

        print("API Status:")
        for provider, status in result.api_status.items():
            print(f"  {provider}: {status['status']}")

        print()
        print("Resources:")
        print(f"  Disk Free: {result.resource_usage['disk_free_gb']:.1f} GB")
        print(f"  Memory Used: {result.resource_usage['memory_used_percent']:.0f}%")

        if result.alerts:
            print()
            print(f"Alerts ({len(result.alerts)}):")
            for alert in result.alerts[:5]:
                print(f"  [{alert['severity'].upper()}] {alert['message']}")

        if result.stuck_jobs:
            print()
            print(f"Stuck Jobs ({len(result.stuck_jobs)}):")
            for job in result.stuck_jobs:
                print(f"  {job.get('workflow_id', job.get('job_id'))}")

    print()
    print("Usage:")
    print("  python -m src.agents.monitor_agent          # Full health check")
    print("  python -m src.agents.monitor_agent api      # API limits only")
    print("  python -m src.agents.monitor_agent resources  # Resources only")
    print("  python -m src.agents.monitor_agent alerts   # View alert history")
    print("  python -m src.agents.monitor_agent stuck    # Check stuck jobs")
