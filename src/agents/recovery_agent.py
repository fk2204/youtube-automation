"""
Recovery Agent - Error Recovery and Fallback Management

Handles error recovery with exponential backoff, fallback provider strategies,
partial result recovery, and failure pattern detection.

Usage:
    from src.agents.recovery_agent import RecoveryAgent

    agent = RecoveryAgent()

    # Retry a failed operation
    result = agent.run(
        operation="tts_generate",
        error="Rate limit exceeded",
        context={"text": "Hello world", "voice": "en-US-GuyNeural"}
    )

    # Get recovery strategy for an error
    strategy = agent.get_recovery_strategy(error_type="api_rate_limit")

    # Clean up failed workflow
    result = agent.cleanup_failed_workflow(workflow_id="wf_20260119_123456")
"""

import json
import time
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from loguru import logger


class ErrorCategory(Enum):
    """Categories of errors for recovery strategies."""
    API_RATE_LIMIT = "api_rate_limit"
    API_AUTH = "api_auth"
    API_ERROR = "api_error"
    NETWORK = "network"
    TIMEOUT = "timeout"
    FILE_NOT_FOUND = "file_not_found"
    DISK_FULL = "disk_full"
    MEMORY = "memory"
    INVALID_INPUT = "invalid_input"
    PROVIDER_DOWN = "provider_down"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    RETRY_IMMEDIATE = "retry_immediate"
    RETRY_BACKOFF = "retry_backoff"
    FALLBACK_PROVIDER = "fallback_provider"
    SKIP = "skip"
    MANUAL = "manual"
    CLEANUP_RETRY = "cleanup_retry"


@dataclass
class RecoveryResult:
    """Result from recovery agent operations."""
    original_error: str
    error_category: str
    recovery_strategy: str
    success: bool
    fallback_used: Optional[str] = None
    retry_count: int = 0
    total_wait_seconds: float = 0
    result_data: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FailureRecord:
    """Record of a failure for pattern detection."""
    error_type: str
    error_message: str
    operation: str
    provider: str
    context: Dict[str, Any]
    timestamp: str
    recovered: bool = False
    recovery_strategy: Optional[str] = None

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


class RecoveryAgent(BaseAgent):
    """
    Recovery Agent - Error recovery and fallback management.

    Features:
    - Retry failed operations with exponential backoff (1s, 2s, 4s, 8s)
    - Fallback provider strategies (Groq -> Ollama, Fish -> Edge-TTS)
    - Recover partial results from failed workflows
    - Clean up failed jobs and temporary files
    - Maintain failure database for pattern detection
    """

    # Exponential backoff configuration
    BACKOFF_BASE_SECONDS = 1
    BACKOFF_MAX_SECONDS = 60
    BACKOFF_MULTIPLIER = 2
    MAX_RETRIES = 4  # 1s, 2s, 4s, 8s = 4 retries

    # Fallback provider chains
    FALLBACK_PROVIDERS = {
        # AI providers: Groq (free) -> Ollama (local) -> Gemini
        "groq": ["ollama", "gemini"],
        "ollama": ["groq", "gemini"],
        "gemini": ["groq", "ollama"],
        "claude": ["gemini", "groq", "ollama"],
        "openai": ["gemini", "groq", "ollama"],

        # TTS providers: Fish -> Edge-TTS
        "fish_audio": ["edge_tts"],
        "edge_tts": ["fish_audio"],

        # Stock footage: Pexels -> Pixabay -> Coverr
        "pexels": ["pixabay", "coverr"],
        "pixabay": ["pexels", "coverr"],
    }

    # Error category detection patterns
    ERROR_PATTERNS = {
        ErrorCategory.API_RATE_LIMIT: [
            "rate limit", "rate_limit", "429", "too many requests",
            "quota exceeded", "limit exceeded"
        ],
        ErrorCategory.API_AUTH: [
            "unauthorized", "401", "403", "forbidden", "invalid api key",
            "authentication failed", "invalid credentials"
        ],
        ErrorCategory.NETWORK: [
            "connection", "network", "dns", "timeout", "unreachable",
            "connection refused", "connection reset"
        ],
        ErrorCategory.TIMEOUT: [
            "timeout", "timed out", "request timeout"
        ],
        ErrorCategory.FILE_NOT_FOUND: [
            "file not found", "filenotfounderror", "no such file",
            "path not found"
        ],
        ErrorCategory.DISK_FULL: [
            "disk full", "no space", "out of disk", "storage full"
        ],
        ErrorCategory.MEMORY: [
            "out of memory", "memory error", "oom", "memoryerror"
        ],
        ErrorCategory.PROVIDER_DOWN: [
            "service unavailable", "503", "500", "internal server error",
            "bad gateway", "502"
        ],
    }

    # Recovery strategies by error category
    CATEGORY_STRATEGIES = {
        ErrorCategory.API_RATE_LIMIT: RecoveryStrategy.RETRY_BACKOFF,
        ErrorCategory.API_AUTH: RecoveryStrategy.MANUAL,
        ErrorCategory.API_ERROR: RecoveryStrategy.FALLBACK_PROVIDER,
        ErrorCategory.NETWORK: RecoveryStrategy.RETRY_BACKOFF,
        ErrorCategory.TIMEOUT: RecoveryStrategy.RETRY_BACKOFF,
        ErrorCategory.FILE_NOT_FOUND: RecoveryStrategy.MANUAL,
        ErrorCategory.DISK_FULL: RecoveryStrategy.CLEANUP_RETRY,
        ErrorCategory.MEMORY: RecoveryStrategy.RETRY_BACKOFF,
        ErrorCategory.PROVIDER_DOWN: RecoveryStrategy.FALLBACK_PROVIDER,
        ErrorCategory.UNKNOWN: RecoveryStrategy.RETRY_BACKOFF,
    }

    def __init__(self):
        super().__init__("RecoveryAgent")
        self.db_path = self.state_dir / "failure_database.db"
        self._init_db()
        self._operation_handlers: Dict[str, Callable] = {}
        logger.info(f"{self.name} initialized")

    def _init_db(self):
        """Initialize failure tracking database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS failures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    error_type TEXT NOT NULL,
                    error_message TEXT,
                    operation TEXT,
                    provider TEXT,
                    context TEXT,
                    recovered INTEGER DEFAULT 0,
                    recovery_strategy TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_failures_type
                ON failures(error_type, timestamp)
            """)

    def register_handler(self, operation: str, handler: Callable):
        """Register a retry handler for an operation type."""
        self._operation_handlers[operation] = handler
        logger.debug(f"Registered handler for operation: {operation}")

    def run(
        self,
        operation: str,
        error: str,
        context: Dict[str, Any] = None,
        provider: str = None,
        **kwargs
    ) -> RecoveryResult:
        """
        Attempt to recover from a failed operation.

        Args:
            operation: Name of the failed operation (e.g., "tts_generate", "api_call")
            error: Error message from the failure
            context: Context/parameters for the operation
            provider: Provider that failed (for fallback selection)
            **kwargs: Additional parameters

        Returns:
            RecoveryResult with recovery outcome
        """
        context = context or {}
        logger.info(f"[{self.name}] Attempting recovery for: {operation}")
        logger.info(f"  Error: {error[:100]}...")
        if provider:
            logger.info(f"  Provider: {provider}")

        # Categorize the error
        error_category = self._categorize_error(error)
        logger.info(f"  Category: {error_category.value}")

        # Record the failure
        self._record_failure(error_category.value, error, operation, provider, context)

        # Get recovery strategy
        strategy = self.CATEGORY_STRATEGIES.get(error_category, RecoveryStrategy.RETRY_BACKOFF)
        logger.info(f"  Strategy: {strategy.value}")

        # Execute recovery based on strategy
        if strategy == RecoveryStrategy.RETRY_BACKOFF:
            return self._retry_with_backoff(operation, context, error, provider, **kwargs)

        elif strategy == RecoveryStrategy.FALLBACK_PROVIDER:
            return self._try_fallback_providers(operation, context, error, provider, **kwargs)

        elif strategy == RecoveryStrategy.CLEANUP_RETRY:
            return self._cleanup_and_retry(operation, context, error, provider, **kwargs)

        elif strategy == RecoveryStrategy.MANUAL:
            return RecoveryResult(
                original_error=error,
                error_category=error_category.value,
                recovery_strategy=strategy.value,
                success=False,
                message="Manual intervention required"
            )

        else:  # SKIP
            return RecoveryResult(
                original_error=error,
                error_category=error_category.value,
                recovery_strategy=strategy.value,
                success=False,
                message="Operation skipped due to unrecoverable error"
            )

    def _categorize_error(self, error: str) -> ErrorCategory:
        """Categorize an error based on its message."""
        error_lower = error.lower()

        for category, patterns in self.ERROR_PATTERNS.items():
            for pattern in patterns:
                if pattern in error_lower:
                    return category

        return ErrorCategory.UNKNOWN

    def _retry_with_backoff(
        self,
        operation: str,
        context: Dict[str, Any],
        original_error: str,
        provider: str = None,
        **kwargs
    ) -> RecoveryResult:
        """
        Retry operation with exponential backoff.

        Backoff sequence: 1s, 2s, 4s, 8s
        """
        handler = self._operation_handlers.get(operation)
        if not handler:
            logger.warning(f"No handler registered for operation: {operation}")
            # Try a generic retry if we have a callable in kwargs
            handler = kwargs.get("retry_func")

        if not handler:
            return RecoveryResult(
                original_error=original_error,
                error_category=self._categorize_error(original_error).value,
                recovery_strategy=RecoveryStrategy.RETRY_BACKOFF.value,
                success=False,
                message=f"No handler available for operation: {operation}"
            )

        total_wait = 0
        last_error = original_error

        for attempt in range(self.MAX_RETRIES):
            wait_time = min(
                self.BACKOFF_BASE_SECONDS * (self.BACKOFF_MULTIPLIER ** attempt),
                self.BACKOFF_MAX_SECONDS
            )

            logger.info(f"[{self.name}] Retry {attempt + 1}/{self.MAX_RETRIES} after {wait_time}s...")
            time.sleep(wait_time)
            total_wait += wait_time

            try:
                result = handler(**context)

                # Record successful recovery
                self._update_failure_recovered(operation, RecoveryStrategy.RETRY_BACKOFF.value)

                return RecoveryResult(
                    original_error=original_error,
                    error_category=self._categorize_error(original_error).value,
                    recovery_strategy=RecoveryStrategy.RETRY_BACKOFF.value,
                    success=True,
                    retry_count=attempt + 1,
                    total_wait_seconds=total_wait,
                    result_data=result if isinstance(result, dict) else {"result": result},
                    message=f"Succeeded after {attempt + 1} retries"
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(f"  Retry {attempt + 1} failed: {last_error[:50]}...")

        return RecoveryResult(
            original_error=original_error,
            error_category=self._categorize_error(original_error).value,
            recovery_strategy=RecoveryStrategy.RETRY_BACKOFF.value,
            success=False,
            retry_count=self.MAX_RETRIES,
            total_wait_seconds=total_wait,
            message=f"All {self.MAX_RETRIES} retries failed. Last error: {last_error}"
        )

    def _try_fallback_providers(
        self,
        operation: str,
        context: Dict[str, Any],
        original_error: str,
        provider: str = None,
        **kwargs
    ) -> RecoveryResult:
        """Try fallback providers when primary provider fails."""
        if not provider:
            # No provider specified, fall back to retry
            return self._retry_with_backoff(operation, context, original_error, **kwargs)

        fallbacks = self.FALLBACK_PROVIDERS.get(provider, [])
        if not fallbacks:
            logger.warning(f"No fallbacks configured for provider: {provider}")
            return self._retry_with_backoff(operation, context, original_error, provider, **kwargs)

        handler = self._operation_handlers.get(operation)
        fallback_func = kwargs.get("fallback_func")

        for fallback_provider in fallbacks:
            logger.info(f"[{self.name}] Trying fallback provider: {fallback_provider}")

            try:
                # Update context with new provider
                fallback_context = context.copy()
                fallback_context["provider"] = fallback_provider

                if fallback_func:
                    result = fallback_func(fallback_provider, **fallback_context)
                elif handler:
                    result = handler(**fallback_context)
                else:
                    continue

                # Record successful recovery
                self._update_failure_recovered(operation, RecoveryStrategy.FALLBACK_PROVIDER.value)

                return RecoveryResult(
                    original_error=original_error,
                    error_category=self._categorize_error(original_error).value,
                    recovery_strategy=RecoveryStrategy.FALLBACK_PROVIDER.value,
                    success=True,
                    fallback_used=fallback_provider,
                    result_data=result if isinstance(result, dict) else {"result": result},
                    message=f"Succeeded with fallback provider: {fallback_provider}"
                )

            except Exception as e:
                logger.warning(f"  Fallback {fallback_provider} failed: {str(e)[:50]}...")
                continue

        return RecoveryResult(
            original_error=original_error,
            error_category=self._categorize_error(original_error).value,
            recovery_strategy=RecoveryStrategy.FALLBACK_PROVIDER.value,
            success=False,
            message=f"All fallback providers failed: {fallbacks}"
        )

    def _cleanup_and_retry(
        self,
        operation: str,
        context: Dict[str, Any],
        original_error: str,
        provider: str = None,
        **kwargs
    ) -> RecoveryResult:
        """Clean up temporary files and retry."""
        logger.info(f"[{self.name}] Running cleanup before retry...")

        # Run cleanup
        cleaned = self._cleanup_temp_files()
        logger.info(f"  Cleaned {cleaned} temporary files")

        # Now retry with backoff
        return self._retry_with_backoff(operation, context, original_error, provider, **kwargs)

    def _cleanup_temp_files(self) -> int:
        """Clean up temporary files to free disk space."""
        import tempfile
        import shutil

        cleaned = 0
        temp_dirs = [
            Path(tempfile.gettempdir()) / "video_ultra",
            Path(tempfile.gettempdir()) / "video_shorts",
            Path(tempfile.gettempdir()) / "video_fast",
        ]

        for temp_dir in temp_dirs:
            if temp_dir.exists():
                try:
                    for file in temp_dir.glob("*"):
                        if file.is_file():
                            file.unlink()
                            cleaned += 1
                        elif file.is_dir():
                            shutil.rmtree(file, ignore_errors=True)
                            cleaned += 1
                except Exception as e:
                    logger.warning(f"Failed to clean {temp_dir}: {e}")

        return cleaned

    def _record_failure(
        self,
        error_type: str,
        error_message: str,
        operation: str,
        provider: str,
        context: Dict[str, Any]
    ):
        """Record a failure to the database for pattern detection."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO failures (error_type, error_message, operation, provider, context)
                VALUES (?, ?, ?, ?, ?)
            """, (
                error_type,
                error_message[:500],  # Truncate long messages
                operation,
                provider or "",
                json.dumps(context)[:1000]
            ))

    def _update_failure_recovered(self, operation: str, strategy: str):
        """Mark the most recent failure for an operation as recovered."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE failures
                SET recovered = 1, recovery_strategy = ?
                WHERE operation = ?
                AND id = (SELECT MAX(id) FROM failures WHERE operation = ?)
            """, (strategy, operation, operation))

    def get_recovery_strategy(self, error_type: str = None, error_message: str = None) -> Dict[str, Any]:
        """
        Get the recommended recovery strategy for an error.

        Args:
            error_type: Error category (if known)
            error_message: Error message (for categorization)

        Returns:
            Dictionary with strategy and recommendations
        """
        if error_type:
            try:
                category = ErrorCategory(error_type)
            except ValueError:
                category = ErrorCategory.UNKNOWN
        elif error_message:
            category = self._categorize_error(error_message)
        else:
            category = ErrorCategory.UNKNOWN

        strategy = self.CATEGORY_STRATEGIES.get(category, RecoveryStrategy.RETRY_BACKOFF)

        return {
            "error_category": category.value,
            "recommended_strategy": strategy.value,
            "backoff_sequence": [
                self.BACKOFF_BASE_SECONDS * (self.BACKOFF_MULTIPLIER ** i)
                for i in range(self.MAX_RETRIES)
            ],
            "max_retries": self.MAX_RETRIES,
            "description": self._get_strategy_description(strategy)
        }

    def _get_strategy_description(self, strategy: RecoveryStrategy) -> str:
        """Get human-readable description of a recovery strategy."""
        descriptions = {
            RecoveryStrategy.RETRY_IMMEDIATE: "Retry immediately without waiting",
            RecoveryStrategy.RETRY_BACKOFF: f"Retry with exponential backoff: {self.BACKOFF_BASE_SECONDS}s, {self.BACKOFF_BASE_SECONDS*2}s, {self.BACKOFF_BASE_SECONDS*4}s, {self.BACKOFF_BASE_SECONDS*8}s",
            RecoveryStrategy.FALLBACK_PROVIDER: "Try alternative providers in sequence",
            RecoveryStrategy.SKIP: "Skip this operation and continue",
            RecoveryStrategy.MANUAL: "Manual intervention required - check logs",
            RecoveryStrategy.CLEANUP_RETRY: "Clean up temp files and retry",
        }
        return descriptions.get(strategy, "Unknown strategy")

    def cleanup_failed_workflow(self, workflow_id: str) -> RecoveryResult:
        """
        Clean up resources from a failed workflow.

        Args:
            workflow_id: Workflow to clean up

        Returns:
            RecoveryResult indicating cleanup success
        """
        logger.info(f"[{self.name}] Cleaning up failed workflow: {workflow_id}")

        cleaned_files = []
        errors = []

        # Find workflow state
        workflow_dir = self.state_dir / "workflow_states" / "workflows"
        workflow_file = workflow_dir / f"{workflow_id}.json"

        if workflow_file.exists():
            try:
                with open(workflow_file) as f:
                    state = json.load(f)

                # Clean up partial outputs
                for step, result in state.get("step_results", {}).items():
                    if isinstance(result, dict):
                        for key, value in result.items():
                            if isinstance(value, str) and Path(value).exists():
                                try:
                                    Path(value).unlink()
                                    cleaned_files.append(value)
                                except Exception as e:
                                    errors.append(f"Failed to delete {value}: {e}")

                # Mark workflow as cleaned
                state["status"] = "cleaned"
                state["cleaned_at"] = datetime.now().isoformat()
                with open(workflow_file, "w") as f:
                    json.dump(state, f, indent=2)

            except Exception as e:
                errors.append(f"Failed to process workflow state: {e}")

        # Clean up temp files
        temp_cleaned = self._cleanup_temp_files()

        success = len(errors) == 0
        message = f"Cleaned {len(cleaned_files)} output files and {temp_cleaned} temp files"
        if errors:
            message += f". Errors: {len(errors)}"

        return RecoveryResult(
            original_error="workflow_failed",
            error_category="cleanup",
            recovery_strategy="cleanup",
            success=success,
            result_data={
                "cleaned_files": cleaned_files,
                "temp_files_cleaned": temp_cleaned,
                "errors": errors
            },
            message=message
        )

    def recover_partial_results(self, workflow_id: str) -> Dict[str, Any]:
        """
        Recover partial results from a failed workflow.

        Args:
            workflow_id: Workflow to recover from

        Returns:
            Dictionary with recovered outputs
        """
        logger.info(f"[{self.name}] Recovering partial results from: {workflow_id}")

        workflow_dir = self.state_dir / "workflow_states" / "workflows"
        workflow_file = workflow_dir / f"{workflow_id}.json"

        if not workflow_file.exists():
            return {"success": False, "error": f"Workflow {workflow_id} not found"}

        try:
            with open(workflow_file) as f:
                state = json.load(f)

            recovered = {
                "workflow_id": workflow_id,
                "topic": state.get("topic"),
                "channel_id": state.get("channel_id"),
                "completed_steps": [],
                "outputs": {}
            }

            for step, result in state.get("step_results", {}).items():
                if isinstance(result, dict) and result.get("success"):
                    recovered["completed_steps"].append(step)

                    # Extract output files
                    for key, value in result.items():
                        if isinstance(value, str) and Path(value).exists():
                            recovered["outputs"][key] = value

            return {
                "success": True,
                "recovered": recovered
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_failure_patterns(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Analyze failure patterns from the database.

        Args:
            hours: Look back period in hours

        Returns:
            List of failure patterns with counts
        """
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            # Get failure counts by type
            type_counts = conn.execute("""
                SELECT error_type, COUNT(*) as count,
                       SUM(recovered) as recovered_count
                FROM failures
                WHERE timestamp > ?
                GROUP BY error_type
                ORDER BY count DESC
            """, (cutoff,)).fetchall()

            # Get failure counts by provider
            provider_counts = conn.execute("""
                SELECT provider, COUNT(*) as count
                FROM failures
                WHERE timestamp > ? AND provider != ''
                GROUP BY provider
                ORDER BY count DESC
            """, (cutoff,)).fetchall()

            # Get failure counts by operation
            operation_counts = conn.execute("""
                SELECT operation, COUNT(*) as count
                FROM failures
                WHERE timestamp > ?
                GROUP BY operation
                ORDER BY count DESC
            """, (cutoff,)).fetchall()

        patterns = []

        for error_type, count, recovered in type_counts:
            recovery_rate = (recovered / count * 100) if count > 0 else 0
            patterns.append({
                "category": "error_type",
                "value": error_type,
                "count": count,
                "recovered": recovered,
                "recovery_rate": f"{recovery_rate:.0f}%"
            })

        for provider, count in provider_counts:
            patterns.append({
                "category": "provider",
                "value": provider,
                "count": count
            })

        for operation, count in operation_counts:
            patterns.append({
                "category": "operation",
                "value": operation,
                "count": count
            })

        return patterns


# CLI entry point
if __name__ == "__main__":
    import sys

    print("\n" + "=" * 60)
    print("  RECOVERY AGENT")
    print("=" * 60 + "\n")

    agent = RecoveryAgent()

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "strategy":
            error = sys.argv[2] if len(sys.argv) > 2 else "rate limit exceeded"
            strategy = agent.get_recovery_strategy(error_message=error)
            print(f"Error: {error}")
            print(f"Category: {strategy['error_category']}")
            print(f"Strategy: {strategy['recommended_strategy']}")
            print(f"Description: {strategy['description']}")
            print(f"Backoff: {strategy['backoff_sequence']}")

        elif command == "patterns":
            hours = int(sys.argv[2]) if len(sys.argv) > 2 else 24
            patterns = agent.get_failure_patterns(hours)
            print(f"Failure Patterns (last {hours}h):\n")
            for p in patterns:
                if p["category"] == "error_type":
                    print(f"  {p['value']}: {p['count']} failures ({p['recovery_rate']} recovered)")
                else:
                    print(f"  {p['category']}={p['value']}: {p['count']} failures")

        elif command == "cleanup" and len(sys.argv) > 2:
            result = agent.cleanup_failed_workflow(sys.argv[2])
            print(f"Cleanup: {result.message}")
            print(f"Success: {result.success}")

        elif command == "recover" and len(sys.argv) > 2:
            result = agent.recover_partial_results(sys.argv[2])
            if result["success"]:
                print(f"Recovered from: {result['recovered']['workflow_id']}")
                print(f"Completed steps: {result['recovered']['completed_steps']}")
                print(f"Outputs: {result['recovered']['outputs']}")
            else:
                print(f"Error: {result['error']}")

    else:
        print("Usage:")
        print("  python -m src.agents.recovery_agent strategy <error_message>")
        print("  python -m src.agents.recovery_agent patterns [hours]")
        print("  python -m src.agents.recovery_agent cleanup <workflow_id>")
        print("  python -m src.agents.recovery_agent recover <workflow_id>")
        print()
        print("Fallback Providers:")
        for provider, fallbacks in agent.FALLBACK_PROVIDERS.items():
            print(f"  {provider} -> {' -> '.join(fallbacks)}")
