"""
Error monitoring and alerting system.

Tracks errors, sends alerts, and provides dashboards.
"""

import functools
import hashlib
import os
import sqlite3
import json
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from enum import Enum

from loguru import logger


class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    API = "api"
    VIDEO = "video"
    UPLOAD = "upload"
    DATABASE = "database"
    NETWORK = "network"
    TTS = "tts"
    SCRIPT = "script"
    SCHEDULER = "scheduler"
    UNKNOWN = "unknown"


@dataclass
class ErrorEvent:
    """Represents a tracked error event."""
    id: str
    message: str
    severity: ErrorSeverity
    category: ErrorCategory
    stack_trace: str
    context: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    resolved: bool = False
    resolution_notes: str = ""
    occurrence_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "stack_trace": self.stack_trace,
            "context": self.context,
            "timestamp": self.timestamp,
            "resolved": self.resolved,
            "resolution_notes": self.resolution_notes,
            "occurrence_count": self.occurrence_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ErrorEvent":
        """Create ErrorEvent from dictionary."""
        return cls(
            id=data["id"],
            message=data["message"],
            severity=ErrorSeverity(data["severity"]),
            category=ErrorCategory(data["category"]),
            stack_trace=data["stack_trace"],
            context=data.get("context", {}),
            timestamp=data["timestamp"],
            resolved=data.get("resolved", False),
            resolution_notes=data.get("resolution_notes", ""),
            occurrence_count=data.get("occurrence_count", 1),
        )


class ErrorMonitor:
    """Monitor and track errors across the application."""

    def __init__(self, db_path: str = "data/errors.db"):
        """
        Initialize the error monitor.

        Args:
            db_path: Path to SQLite database for error storage
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._alert_handlers: List[Callable[[ErrorEvent], None]] = []
        self._error_counts: Dict[str, int] = {}  # For rate limiting alerts
        self._last_alert_time: Dict[str, datetime] = {}
        logger.info(f"ErrorMonitor initialized with database at {self.db_path}")

    def _init_db(self):
        """Initialize error tracking database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create errors table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS errors (
                    id TEXT PRIMARY KEY,
                    message TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    category TEXT NOT NULL,
                    stack_trace TEXT,
                    context TEXT,
                    timestamp TEXT NOT NULL,
                    resolved INTEGER DEFAULT 0,
                    resolution_notes TEXT DEFAULT '',
                    occurrence_count INTEGER DEFAULT 1,
                    error_hash TEXT
                )
            """)

            # Create index for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_errors_timestamp
                ON errors(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_errors_severity
                ON errors(severity)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_errors_category
                ON errors(category)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_errors_hash
                ON errors(error_hash)
            """)

            conn.commit()
            logger.debug("Error database initialized")

    def _generate_error_id(self) -> str:
        """Generate a unique error ID."""
        timestamp = datetime.now().isoformat()
        unique_str = f"{timestamp}-{os.urandom(4).hex()}"
        return hashlib.sha256(unique_str.encode()).hexdigest()[:12]

    def _generate_error_hash(self, error: Exception, category: ErrorCategory) -> str:
        """Generate a hash for deduplication based on error type and message."""
        error_str = f"{type(error).__name__}:{str(error)[:100]}:{category.value}"
        return hashlib.md5(error_str.encode()).hexdigest()

    def record_error(
        self,
        error: Exception,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorEvent:
        """
        Record an error event.

        Args:
            error: The exception that occurred
            severity: Error severity level
            category: Error category for classification
            context: Additional context about the error

        Returns:
            The created ErrorEvent
        """
        error_id = self._generate_error_id()
        error_hash = self._generate_error_hash(error, category)
        stack_trace = traceback.format_exc()
        context = context or {}

        # Add common context
        context.update({
            "error_type": type(error).__name__,
            "error_args": [str(arg) for arg in error.args] if error.args else [],
            "recorded_at": datetime.now().isoformat(),
        })

        event = ErrorEvent(
            id=error_id,
            message=str(error),
            severity=severity,
            category=category,
            stack_trace=stack_trace,
            context=context,
        )

        # Check for duplicate errors (same hash within last hour)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Check for recent duplicate
            one_hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
            cursor.execute("""
                SELECT id, occurrence_count FROM errors
                WHERE error_hash = ? AND timestamp > ? AND resolved = 0
                ORDER BY timestamp DESC LIMIT 1
            """, (error_hash, one_hour_ago))

            existing = cursor.fetchone()

            if existing:
                # Increment occurrence count for existing error
                existing_id, count = existing
                cursor.execute("""
                    UPDATE errors SET occurrence_count = ?, timestamp = ?
                    WHERE id = ?
                """, (count + 1, event.timestamp, existing_id))
                event.id = existing_id
                event.occurrence_count = count + 1
                logger.debug(f"Incremented error count for {existing_id}: {count + 1}")
            else:
                # Insert new error
                cursor.execute("""
                    INSERT INTO errors
                    (id, message, severity, category, stack_trace, context,
                     timestamp, resolved, resolution_notes, occurrence_count, error_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.id,
                    event.message,
                    event.severity.value,
                    event.category.value,
                    event.stack_trace,
                    json.dumps(event.context),
                    event.timestamp,
                    0,
                    "",
                    1,
                    error_hash
                ))

            conn.commit()

        # Log the error
        log_level = {
            ErrorSeverity.LOW: "DEBUG",
            ErrorSeverity.MEDIUM: "WARNING",
            ErrorSeverity.HIGH: "ERROR",
            ErrorSeverity.CRITICAL: "CRITICAL",
        }.get(severity, "ERROR")

        logger.log(
            log_level,
            f"[{category.value.upper()}] {event.message} (ID: {event.id})"
        )

        # Trigger alert handlers
        self._trigger_alerts(event)

        return event

    def add_alert_handler(self, handler: Callable[[ErrorEvent], None]):
        """
        Add a handler for error alerts.

        Args:
            handler: Callable that receives ErrorEvent when alert triggers
        """
        self._alert_handlers.append(handler)
        logger.debug(f"Added alert handler: {handler.__name__}")

    def _trigger_alerts(self, event: ErrorEvent):
        """Trigger all registered alert handlers."""
        for handler in self._alert_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Alert handler {handler.__name__} failed: {e}")

    def get_error_by_id(self, error_id: str) -> Optional[ErrorEvent]:
        """Get a specific error by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, message, severity, category, stack_trace, context,
                       timestamp, resolved, resolution_notes, occurrence_count
                FROM errors WHERE id = ?
            """, (error_id,))

            row = cursor.fetchone()
            if row:
                return ErrorEvent(
                    id=row[0],
                    message=row[1],
                    severity=ErrorSeverity(row[2]),
                    category=ErrorCategory(row[3]),
                    stack_trace=row[4],
                    context=json.loads(row[5]) if row[5] else {},
                    timestamp=row[6],
                    resolved=bool(row[7]),
                    resolution_notes=row[8] or "",
                    occurrence_count=row[9],
                )
        return None

    def resolve_error(self, error_id: str, notes: str = "") -> bool:
        """
        Mark an error as resolved.

        Args:
            error_id: The error ID to resolve
            notes: Optional resolution notes

        Returns:
            True if error was found and resolved
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE errors SET resolved = 1, resolution_notes = ?
                WHERE id = ?
            """, (notes, error_id))
            conn.commit()

            if cursor.rowcount > 0:
                logger.info(f"Error {error_id} marked as resolved")
                return True
        return False

    def get_recent_errors(
        self,
        hours: int = 24,
        severity: Optional[ErrorSeverity] = None,
        category: Optional[ErrorCategory] = None,
        include_resolved: bool = False
    ) -> List[ErrorEvent]:
        """
        Get errors from the last N hours.

        Args:
            hours: Number of hours to look back
            severity: Filter by severity level
            category: Filter by category
            include_resolved: Whether to include resolved errors

        Returns:
            List of ErrorEvent objects
        """
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

        query = """
            SELECT id, message, severity, category, stack_trace, context,
                   timestamp, resolved, resolution_notes, occurrence_count
            FROM errors WHERE timestamp > ?
        """
        params = [cutoff]

        if severity:
            query += " AND severity = ?"
            params.append(severity.value)

        if category:
            query += " AND category = ?"
            params.append(category.value)

        if not include_resolved:
            query += " AND resolved = 0"

        query += " ORDER BY timestamp DESC"

        errors = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)

            for row in cursor.fetchall():
                errors.append(ErrorEvent(
                    id=row[0],
                    message=row[1],
                    severity=ErrorSeverity(row[2]),
                    category=ErrorCategory(row[3]),
                    stack_trace=row[4],
                    context=json.loads(row[5]) if row[5] else {},
                    timestamp=row[6],
                    resolved=bool(row[7]),
                    resolution_notes=row[8] or "",
                    occurrence_count=row[9],
                ))

        return errors

    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get error summary for time period.

        Args:
            hours: Number of hours to summarize

        Returns:
            Dictionary with error counts and statistics
        """
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

        summary = {
            "period_hours": hours,
            "total_errors": 0,
            "total_occurrences": 0,
            "by_severity": {s.value: 0 for s in ErrorSeverity},
            "by_category": {c.value: 0 for c in ErrorCategory},
            "unresolved": 0,
            "resolved": 0,
            "top_errors": [],
            "recent_critical": [],
        }

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get total counts
            cursor.execute("""
                SELECT COUNT(*), SUM(occurrence_count)
                FROM errors WHERE timestamp > ?
            """, (cutoff,))
            row = cursor.fetchone()
            summary["total_errors"] = row[0] or 0
            summary["total_occurrences"] = row[1] or 0

            # Get counts by severity
            cursor.execute("""
                SELECT severity, COUNT(*), SUM(occurrence_count)
                FROM errors WHERE timestamp > ?
                GROUP BY severity
            """, (cutoff,))
            for row in cursor.fetchall():
                if row[0] in summary["by_severity"]:
                    summary["by_severity"][row[0]] = row[2] or row[1]

            # Get counts by category
            cursor.execute("""
                SELECT category, COUNT(*), SUM(occurrence_count)
                FROM errors WHERE timestamp > ?
                GROUP BY category
            """, (cutoff,))
            for row in cursor.fetchall():
                if row[0] in summary["by_category"]:
                    summary["by_category"][row[0]] = row[2] or row[1]

            # Get resolved/unresolved counts
            cursor.execute("""
                SELECT resolved, COUNT(*)
                FROM errors WHERE timestamp > ?
                GROUP BY resolved
            """, (cutoff,))
            for row in cursor.fetchall():
                if row[0] == 0:
                    summary["unresolved"] = row[1]
                else:
                    summary["resolved"] = row[1]

            # Get top errors by occurrence
            cursor.execute("""
                SELECT message, category, severity, SUM(occurrence_count) as total
                FROM errors WHERE timestamp > ?
                GROUP BY message, category
                ORDER BY total DESC
                LIMIT 5
            """, (cutoff,))
            for row in cursor.fetchall():
                summary["top_errors"].append({
                    "message": row[0][:100],
                    "category": row[1],
                    "severity": row[2],
                    "occurrences": row[3],
                })

            # Get recent critical errors
            cursor.execute("""
                SELECT id, message, timestamp, occurrence_count
                FROM errors
                WHERE timestamp > ? AND severity = 'critical' AND resolved = 0
                ORDER BY timestamp DESC
                LIMIT 5
            """, (cutoff,))
            for row in cursor.fetchall():
                summary["recent_critical"].append({
                    "id": row[0],
                    "message": row[1][:100],
                    "timestamp": row[2],
                    "occurrences": row[3],
                })

        return summary

    def get_error_trends(self, days: int = 7) -> Dict[str, List[int]]:
        """
        Get error trends over time.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with daily error counts by severity
        """
        trends = {
            "dates": [],
            "total": [],
            "critical": [],
            "high": [],
            "medium": [],
            "low": [],
        }

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            for i in range(days - 1, -1, -1):
                date = datetime.now() - timedelta(days=i)
                date_str = date.strftime("%Y-%m-%d")
                trends["dates"].append(date_str)

                start = date.replace(hour=0, minute=0, second=0).isoformat()
                end = date.replace(hour=23, minute=59, second=59).isoformat()

                # Get total for the day
                cursor.execute("""
                    SELECT SUM(occurrence_count) FROM errors
                    WHERE timestamp BETWEEN ? AND ?
                """, (start, end))
                total = cursor.fetchone()[0] or 0
                trends["total"].append(total)

                # Get by severity
                for severity in ["critical", "high", "medium", "low"]:
                    cursor.execute("""
                        SELECT SUM(occurrence_count) FROM errors
                        WHERE timestamp BETWEEN ? AND ? AND severity = ?
                    """, (start, end, severity))
                    count = cursor.fetchone()[0] or 0
                    trends[severity].append(count)

        return trends

    def check_health(self) -> Dict[str, Any]:
        """
        Check system health based on recent errors.

        Returns:
            Health status with recommendations
        """
        summary = self.get_error_summary(hours=1)  # Last hour
        trends = self.get_error_trends(days=1)

        health = {
            "status": "healthy",
            "score": 100,
            "issues": [],
            "recommendations": [],
            "last_check": datetime.now().isoformat(),
        }

        # Check critical errors
        critical_count = summary["by_severity"].get("critical", 0)
        if critical_count > 0:
            health["status"] = "critical"
            health["score"] -= 50
            health["issues"].append(f"{critical_count} critical errors in last hour")
            health["recommendations"].append("Investigate critical errors immediately")

        # Check high errors
        high_count = summary["by_severity"].get("high", 0)
        if high_count > 5:
            health["status"] = min(health["status"], "degraded", key=lambda x: ["healthy", "degraded", "critical"].index(x))
            health["score"] -= 20
            health["issues"].append(f"{high_count} high severity errors in last hour")

        # Check error rate
        if summary["total_occurrences"] > 50:
            health["status"] = "degraded" if health["status"] == "healthy" else health["status"]
            health["score"] -= 15
            health["issues"].append(f"High error rate: {summary['total_occurrences']} errors in last hour")
            health["recommendations"].append("Review error patterns and consider rate limiting")

        # Check unresolved errors
        if summary["unresolved"] > 20:
            health["score"] -= 10
            health["issues"].append(f"{summary['unresolved']} unresolved errors")
            health["recommendations"].append("Review and resolve accumulated errors")

        # Check API errors specifically
        api_errors = summary["by_category"].get("api", 0)
        if api_errors > 10:
            health["issues"].append(f"{api_errors} API errors - check external service status")
            health["recommendations"].append("Verify API keys and service availability")

        # Ensure score doesn't go below 0
        health["score"] = max(0, health["score"])

        return health

    def cleanup_old_errors(self, days: int = 30) -> int:
        """
        Remove errors older than specified days.

        Args:
            days: Number of days to keep

        Returns:
            Number of errors deleted
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM errors WHERE timestamp < ?", (cutoff,))
            deleted = cursor.rowcount
            conn.commit()

        logger.info(f"Cleaned up {deleted} errors older than {days} days")
        return deleted

    def export_errors(self, output_path: str, hours: int = 24) -> str:
        """
        Export errors to JSON file.

        Args:
            output_path: Path for output file
            hours: Number of hours of errors to export

        Returns:
            Path to exported file
        """
        errors = self.get_recent_errors(hours=hours, include_resolved=True)
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "period_hours": hours,
            "error_count": len(errors),
            "errors": [e.to_dict() for e in errors],
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported {len(errors)} errors to {output_path}")
        return str(output_path)


class AlertManager:
    """Manage error alerts and notifications."""

    def __init__(self):
        """Initialize the alert manager."""
        self.thresholds = {
            ErrorSeverity.CRITICAL: 1,   # Alert on first critical
            ErrorSeverity.HIGH: 3,       # Alert after 3 high errors
            ErrorSeverity.MEDIUM: 10,    # Alert after 10 medium errors
            ErrorSeverity.LOW: 50,       # Alert after 50 low errors
        }
        self._error_counts: Dict[str, int] = {}
        self._last_alert: Dict[str, datetime] = {}
        self._alert_cooldown = timedelta(minutes=15)  # Minimum time between same alerts
        logger.debug("AlertManager initialized")

    def should_alert(self, severity: ErrorSeverity, count: int) -> bool:
        """
        Check if alert should be sent.

        Args:
            severity: Error severity level
            count: Current error count

        Returns:
            True if alert should be sent
        """
        threshold = self.thresholds.get(severity, 10)
        return count >= threshold

    def check_cooldown(self, alert_key: str) -> bool:
        """
        Check if alert is in cooldown period.

        Args:
            alert_key: Unique key for the alert type

        Returns:
            True if alert can be sent (not in cooldown)
        """
        last_time = self._last_alert.get(alert_key)
        if last_time is None:
            return True
        return datetime.now() - last_time > self._alert_cooldown

    def record_alert(self, alert_key: str):
        """Record that an alert was sent."""
        self._last_alert[alert_key] = datetime.now()

    def send_console_alert(self, event: ErrorEvent):
        """
        Send alert to console/logs.

        Args:
            event: The error event to alert on
        """
        alert_key = f"console:{event.category.value}:{event.severity.value}"

        if not self.check_cooldown(alert_key):
            logger.debug(f"Alert {alert_key} in cooldown, skipping")
            return

        severity_prefix = {
            ErrorSeverity.CRITICAL: "CRITICAL ALERT",
            ErrorSeverity.HIGH: "HIGH ALERT",
            ErrorSeverity.MEDIUM: "ALERT",
            ErrorSeverity.LOW: "NOTICE",
        }

        prefix = severity_prefix.get(event.severity, "ALERT")
        message = f"""
{'='*60}
{prefix}: {event.category.value.upper()} ERROR
{'='*60}
ID: {event.id}
Time: {event.timestamp}
Occurrences: {event.occurrence_count}

Message: {event.message}

Context: {json.dumps(event.context, indent=2)}

Stack Trace:
{event.stack_trace[:500]}{'...' if len(event.stack_trace) > 500 else ''}
{'='*60}
"""

        log_level = {
            ErrorSeverity.CRITICAL: "CRITICAL",
            ErrorSeverity.HIGH: "ERROR",
            ErrorSeverity.MEDIUM: "WARNING",
            ErrorSeverity.LOW: "INFO",
        }.get(event.severity, "WARNING")

        logger.log(log_level, message)
        self.record_alert(alert_key)

    def send_file_alert(self, event: ErrorEvent, alert_file: str = "logs/alerts.log"):
        """
        Write alert to file.

        Args:
            event: The error event to write
            alert_file: Path to alert log file
        """
        alert_path = Path(alert_file)
        alert_path.parent.mkdir(parents=True, exist_ok=True)

        alert_data = {
            "timestamp": datetime.now().isoformat(),
            "event": event.to_dict(),
        }

        with open(alert_path, "a") as f:
            f.write(json.dumps(alert_data) + "\n")

        logger.debug(f"Alert written to {alert_file}")

    def create_alert_handler(self, alert_file: Optional[str] = None) -> Callable[[ErrorEvent], None]:
        """
        Create an alert handler function.

        Args:
            alert_file: Optional file path for file alerts

        Returns:
            Alert handler function
        """
        def handler(event: ErrorEvent):
            # Check if we should alert based on threshold
            count_key = f"{event.category.value}:{event.severity.value}"
            self._error_counts[count_key] = self._error_counts.get(count_key, 0) + event.occurrence_count

            if self.should_alert(event.severity, self._error_counts[count_key]):
                # Always send console alert for critical/high
                if event.severity in (ErrorSeverity.CRITICAL, ErrorSeverity.HIGH):
                    self.send_console_alert(event)

                # Write to file if specified
                if alert_file:
                    self.send_file_alert(event, alert_file)

                # Reset counter after alert
                self._error_counts[count_key] = 0

        return handler


# Global error monitor instance
_error_monitor: Optional[ErrorMonitor] = None
_alert_manager: Optional[AlertManager] = None


def get_error_monitor() -> ErrorMonitor:
    """Get global error monitor instance."""
    global _error_monitor
    if _error_monitor is None:
        _error_monitor = ErrorMonitor()
        # Setup default alert handler
        alert_manager = get_alert_manager()
        _error_monitor.add_alert_handler(
            alert_manager.create_alert_handler("logs/alerts.log")
        )
    return _error_monitor


def get_alert_manager() -> AlertManager:
    """Get global alert manager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


def monitor_errors(category: ErrorCategory = ErrorCategory.UNKNOWN):
    """
    Decorator to automatically monitor function errors.

    Args:
        category: Error category for classification

    Returns:
        Decorated function that records errors
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Determine severity based on exception type
                severity = ErrorSeverity.MEDIUM
                if isinstance(e, (ConnectionError, TimeoutError)):
                    severity = ErrorSeverity.HIGH
                elif isinstance(e, (KeyError, ValueError, TypeError)):
                    severity = ErrorSeverity.LOW

                # Record the error
                get_error_monitor().record_error(
                    e,
                    severity=severity,
                    category=category,
                    context={
                        "function": func.__name__,
                        "module": func.__module__,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys()),
                    }
                )
                raise
        return wrapper
    return decorator


def monitor_errors_async(category: ErrorCategory = ErrorCategory.UNKNOWN):
    """
    Async decorator to automatically monitor function errors.

    Args:
        category: Error category for classification

    Returns:
        Decorated async function that records errors
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Determine severity based on exception type
                severity = ErrorSeverity.MEDIUM
                if isinstance(e, (ConnectionError, TimeoutError)):
                    severity = ErrorSeverity.HIGH
                elif isinstance(e, (KeyError, ValueError, TypeError)):
                    severity = ErrorSeverity.LOW

                # Record the error
                get_error_monitor().record_error(
                    e,
                    severity=severity,
                    category=category,
                    context={
                        "function": func.__name__,
                        "module": func.__module__,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys()),
                    }
                )
                raise
        return wrapper
    return decorator


def quick_record_error(
    error: Exception,
    category: str = "unknown",
    severity: str = "medium",
    context: Optional[Dict[str, Any]] = None
) -> ErrorEvent:
    """
    Quick helper to record an error with string parameters.

    Args:
        error: The exception
        category: Category name as string
        severity: Severity name as string
        context: Additional context

    Returns:
        The recorded ErrorEvent
    """
    try:
        cat = ErrorCategory(category.lower())
    except ValueError:
        cat = ErrorCategory.UNKNOWN

    try:
        sev = ErrorSeverity(severity.lower())
    except ValueError:
        sev = ErrorSeverity.MEDIUM

    return get_error_monitor().record_error(
        error,
        severity=sev,
        category=cat,
        context=context
    )


if __name__ == "__main__":
    # Example usage and self-test
    print("Error Monitor - Self Test")
    print("=" * 60)

    # Initialize monitor
    monitor = get_error_monitor()

    # Record some test errors
    try:
        raise ValueError("Test error 1: Invalid input")
    except Exception as e:
        monitor.record_error(e, ErrorSeverity.LOW, ErrorCategory.SCRIPT)

    try:
        raise ConnectionError("Test error 2: API connection failed")
    except Exception as e:
        monitor.record_error(e, ErrorSeverity.HIGH, ErrorCategory.API)

    try:
        raise RuntimeError("Test error 3: Critical failure")
    except Exception as e:
        monitor.record_error(e, ErrorSeverity.CRITICAL, ErrorCategory.VIDEO)

    # Get summary
    summary = monitor.get_error_summary(hours=1)
    print("\nError Summary (Last Hour):")
    print(f"  Total Errors: {summary['total_errors']}")
    print(f"  By Severity: {summary['by_severity']}")
    print(f"  By Category: {summary['by_category']}")

    # Check health
    health = monitor.check_health()
    print(f"\nSystem Health:")
    print(f"  Status: {health['status']}")
    print(f"  Score: {health['score']}/100")
    print(f"  Issues: {health['issues']}")

    # Get trends
    trends = monitor.get_error_trends(days=1)
    print(f"\nError Trends (Today):")
    print(f"  Total: {trends['total']}")

    print("\nSelf-test complete!")
