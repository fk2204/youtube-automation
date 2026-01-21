"""
Base Agent Infrastructure for YouTube Automation
Provides abstract base class and utilities for all agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from datetime import datetime
from uuid import uuid4
from loguru import logger
import time
import functools

# Import existing utilities
import sys
sys.path.insert(0, 'C:/Users/fkozi/youtube-automation')


@dataclass
class AgentResult:
    """
    Unified standard result from all agent operations.

    Used by all 19+ agents for consistent return format across the system.

    Fields:
        success: Operation success status
        operation: Name of the operation performed (e.g., "script_generation", "seo_analysis")
        data: Operation-specific result data (dict, list, str, etc.)
        error: Error message if operation failed
        timestamp: When the operation completed
        agent_name: Name of the agent that produced this result
        tokens_used: Total tokens consumed (input + output)
        cost: Estimated cost in USD for the operation
        duration_seconds: How long the operation took
        metadata: Additional operation-specific metadata
    """
    success: bool
    operation: str
    data: Any = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    agent_name: Optional[str] = None
    tokens_used: int = 0
    cost: float = 0.0
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert datetime to ISO string
        if isinstance(result.get('timestamp'), datetime):
            result['timestamp'] = result['timestamp'].isoformat()
        return result

    def __str__(self) -> str:
        """Human-readable representation."""
        status = "SUCCESS" if self.success else "FAILURE"
        details = []
        if self.agent_name:
            details.append(f"agent={self.agent_name}")
        if self.tokens_used:
            details.append(f"tokens={self.tokens_used}")
        if self.cost:
            details.append(f"cost=${self.cost:.4f}")
        if self.duration_seconds:
            details.append(f"duration={self.duration_seconds:.1f}s")

        detail_str = ", ".join(details) if details else "no details"
        return f"[{status}] {self.operation} ({detail_str})"


@dataclass
class AgentMessage:
    """Standard message format for agent communication."""
    sender: str
    recipient: str  # Agent name or "broadcast"
    message_type: str  # request, response, event, error
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5  # 1 (highest) to 10 (lowest)
    message_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class AgentError(Exception):
    """Base exception for agent errors."""
    pass


class TokenBudgetExceeded(AgentError):
    """Daily token budget exceeded."""
    pass


class APIRateLimitError(AgentError):
    """API rate limit hit."""
    pass


class QualityError(AgentError):
    """Content failed quality checks."""
    pass


class SafetyError(AgentError):
    """Content flagged for safety concerns."""
    pass


def handle_agent_errors(func):
    """Decorator for graceful error handling."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TokenBudgetExceeded as e:
            logger.warning(f"Budget exceeded: {e}, switching to free provider")
            kwargs['provider'] = 'ollama'
            return func(*args, **kwargs)
        except APIRateLimitError:
            logger.warning("Rate limited, waiting 60s and retrying")
            time.sleep(60)
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Agent error: {e}")
            raise
    return wrapper


class BaseAgent(ABC):
    """Base class for all YouTube automation agents."""

    def __init__(self, provider: str = "groq", api_key: str = None):
        self.name = self.__class__.__name__
        self.provider = provider
        self.api_key = api_key
        self._start_time = None

        # Try to import utilities
        try:
            from src.utils.token_manager import get_token_manager
            self.tracker = get_token_manager()
        except:
            self.tracker = None

        logger.info(f"{self.name} initialized with provider={provider}")

    @abstractmethod
    def run(self, **kwargs) -> AgentResult:
        """Main entry point - must be implemented by subclasses."""
        pass

    def handle_message(self, message: AgentMessage) -> AgentMessage:
        """Handle incoming message from another agent."""
        self._start_time = time.time()
        try:
            result = self.run(**message.payload)
            duration = time.time() - self._start_time
            return AgentMessage(
                sender=self.name,
                recipient=message.sender,
                message_type="response",
                payload={"result": result, "duration": duration},
                correlation_id=message.message_id
            )
        except Exception as e:
            logger.error(f"{self.name} error handling message: {e}")
            return AgentMessage(
                sender=self.name,
                recipient=message.sender,
                message_type="error",
                payload={"error": str(e), "error_type": type(e).__name__},
                correlation_id=message.message_id
            )

    def create_result(
        self,
        success: bool,
        operation: str,
        data: Any = None,
        error: Optional[str] = None,
        tokens_used: int = 0,
        cost: float = 0.0,
        duration_seconds: float = 0.0,
        **metadata
    ) -> AgentResult:
        """
        Convenience method to create a standardized AgentResult.

        Args:
            success: Whether the operation succeeded
            operation: Name of the operation (e.g., "script_generation")
            data: Operation result data
            error: Error message if failed
            tokens_used: Total tokens consumed
            cost: Estimated cost in USD
            duration_seconds: Operation duration
            **metadata: Additional metadata as keyword arguments

        Returns:
            AgentResult instance
        """
        return AgentResult(
            success=success,
            operation=operation,
            data=data,
            error=error,
            timestamp=datetime.now(),
            agent_name=self.name,
            tokens_used=tokens_used,
            cost=cost,
            duration_seconds=duration_seconds,
            metadata=metadata
        )

    def log_operation(self, operation: str, tokens: int = 0, cost: float = 0.0):
        """Log operation for tracking."""
        if self.tracker:
            self.tracker.record_usage(
                provider=self.provider,
                input_tokens=tokens // 2,
                output_tokens=tokens // 2,
                operation=f"{self.name}_{operation}"
            )
        logger.bind(
            agent=self.name,
            operation=operation,
            tokens=tokens,
            cost=cost
        ).info(f"{self.name} completed {operation}")

    def _timed_operation(self, operation_name: str):
        """Context manager for timing operations."""
        class Timer:
            def __init__(self, agent, name):
                self.agent = agent
                self.name = name
                self.start = None
            def __enter__(self):
                self.start = time.time()
                return self
            def __exit__(self, *args):
                duration = time.time() - self.start
                logger.debug(f"{self.agent.name}.{self.name} took {duration:.2f}s")
        return Timer(self, operation_name)
