"""
Monitoring module for YouTube automation.

Includes performance monitoring and error tracking.
"""

from .performance_monitor import PerformanceAlert, PerformanceMonitor
from .error_monitor import (
    ErrorSeverity,
    ErrorCategory,
    ErrorEvent,
    ErrorMonitor,
    AlertManager,
    get_error_monitor,
    get_alert_manager,
    monitor_errors,
    monitor_errors_async,
    quick_record_error,
)

__all__ = [
    # Performance monitoring
    "PerformanceAlert",
    "PerformanceMonitor",
    # Error monitoring
    "ErrorSeverity",
    "ErrorCategory",
    "ErrorEvent",
    "ErrorMonitor",
    "AlertManager",
    "get_error_monitor",
    "get_alert_manager",
    "monitor_errors",
    "monitor_errors_async",
    "quick_record_error",
]
