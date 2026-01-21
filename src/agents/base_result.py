"""
Unified Agent Result System

Provides a standardized AgentResult dataclass used by all 19+ agents
for consistent return values and cross-agent communication.

Usage:
    from src.agents.base_result import AgentResult

    # Create a result
    result = AgentResult(
        success=True,
        operation="script_generation",
        data={"title": "My Video", "sections": [...]},
        agent_name="ScriptAgent",
        tokens_used=1500,
        cost=0.002
    )

    # Convert to dict for JSON
    result_dict = result.to_dict()

    # Print human-readable
    print(result)  # [SUCCESS] script_generation (agent=ScriptAgent, tokens=1500, cost=$0.0020)
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Dict
from datetime import datetime


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

    Example:
        >>> result = AgentResult(
        ...     success=True,
        ...     operation="video_analysis",
        ...     data={"score": 85, "issues": []},
        ...     agent_name="QualityAgent",
        ...     tokens_used=500,
        ...     cost=0.001,
        ...     duration_seconds=2.5
        ... )
        >>> print(result)
        [SUCCESS] video_analysis (agent=QualityAgent, tokens=500, cost=$0.0010, duration=2.5s)
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
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all fields, datetime converted to ISO format
        """
        result = asdict(self)
        # Convert datetime to ISO string for JSON serialization
        if isinstance(result.get('timestamp'), datetime):
            result['timestamp'] = result['timestamp'].isoformat()
        return result

    def __str__(self) -> str:
        """
        Human-readable representation.

        Returns:
            String like "[SUCCESS] operation_name (agent=Name, tokens=100, cost=$0.0010)"
        """
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

    def __repr__(self) -> str:
        """Technical representation for debugging."""
        return (
            f"AgentResult(success={self.success}, operation='{self.operation}', "
            f"agent_name='{self.agent_name}', tokens={self.tokens_used})"
        )


# Convenience function for backward compatibility
def create_result(
    success: bool,
    operation: str,
    agent_name: str,
    data: Any = None,
    error: Optional[str] = None,
    tokens_used: int = 0,
    cost: float = 0.0,
    duration_seconds: float = 0.0,
    **metadata
) -> AgentResult:
    """
    Convenience factory function to create an AgentResult.

    Args:
        success: Whether the operation succeeded
        operation: Name of the operation (e.g., "script_generation")
        agent_name: Name of the agent performing the operation
        data: Operation result data
        error: Error message if failed
        tokens_used: Total tokens consumed
        cost: Estimated cost in USD
        duration_seconds: Operation duration
        **metadata: Additional metadata as keyword arguments

    Returns:
        AgentResult instance

    Example:
        >>> from src.agents.base_result import create_result
        >>> result = create_result(
        ...     success=True,
        ...     operation="seo_optimization",
        ...     agent_name="SEOAgent",
        ...     data={"keywords": ["AI", "automation"]},
        ...     tokens_used=200,
        ...     cost=0.0004
        ... )
    """
    return AgentResult(
        success=success,
        operation=operation,
        data=data,
        error=error,
        timestamp=datetime.now(),
        agent_name=agent_name,
        tokens_used=tokens_used,
        cost=cost,
        duration_seconds=duration_seconds,
        metadata=metadata
    )
