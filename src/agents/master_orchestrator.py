"""
Master Orchestrator - Unified Agent Coordination System

A production-ready orchestration system that coordinates all 19 agents efficiently
with parallel execution, dependency management, load balancing, and result aggregation.

Components:
- AgentRegistry: Central registry of all agents with capabilities
- TaskRouter: Route tasks to appropriate agents
- ParallelExecutor: Run independent agents simultaneously
- DependencyGraph: Handle agent dependencies
- ResultAggregator: Combine results from multiple agents
- LoadBalancer: Distribute work evenly

Usage:
    from src.agents.master_orchestrator import (
        MasterOrchestrator,
        get_master_orchestrator
    )

    orchestrator = get_master_orchestrator()

    # Register agents automatically
    orchestrator.auto_register_agents()

    # Execute a workflow
    result = await orchestrator.execute_workflow(
        workflow_type="full_video",
        channel_id="money_blueprints",
        topic="passive income"
    )

    # Run parallel tasks
    results = await orchestrator.run_parallel([
        ("ResearchAgent", "find_topics", {"niche": "finance"}),
        ("AnalyticsAgent", "analyze_channel", {"channel": "money_blueprints"}),
    ])
"""

import asyncio
import time
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)
from uuid import uuid4

from loguru import logger


# ============================================================================
# Data Classes and Enums
# ============================================================================


class AgentCapability(Enum):
    """Agent capabilities for task routing."""
    RESEARCH = "research"
    CONTENT_GENERATION = "content_generation"
    VIDEO_PRODUCTION = "video_production"
    AUDIO_PROCESSING = "audio_processing"
    QUALITY_CHECK = "quality_check"
    SEO_OPTIMIZATION = "seo_optimization"
    ANALYTICS = "analytics"
    SCHEDULING = "scheduling"
    MONITORING = "monitoring"
    RECOVERY = "recovery"
    VALIDATION = "validation"
    COMPLIANCE = "compliance"
    THUMBNAILS = "thumbnails"
    UPLOAD = "upload"
    WORKFLOW = "workflow"


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class ExecutionStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class AgentInfo:
    """Information about a registered agent."""
    name: str
    agent_class: Type
    instance: Optional[Any] = None
    capabilities: List[AgentCapability] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    max_concurrent: int = 1
    current_load: int = 0
    avg_execution_time: float = 0.0
    success_rate: float = 1.0
    total_executions: int = 0
    last_execution: Optional[datetime] = None
    is_healthy: bool = True
    priority_weight: float = 1.0


@dataclass
class TaskResult:
    """Result from task execution."""
    task_id: str
    agent_name: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowResult:
    """Result from workflow execution."""
    workflow_id: str
    workflow_type: str
    status: ExecutionStatus
    success: bool
    results: Dict[str, TaskResult] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    total_time: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DependencyNode:
    """Node in the dependency graph."""
    agent_name: str
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    resolved: bool = False
    in_progress: bool = False


# ============================================================================
# Agent Registry
# ============================================================================


class AgentRegistry:
    """
    Central registry of all agents with capabilities tracking.

    Features:
    - Register agents with their capabilities
    - Query agents by capability
    - Track agent health and performance
    - Lazy initialization of agent instances
    """

    # Agent capability mappings
    AGENT_CAPABILITIES = {
        "ResearchAgent": [AgentCapability.RESEARCH, AgentCapability.CONTENT_GENERATION],
        "QualityAgent": [AgentCapability.QUALITY_CHECK, AgentCapability.VALIDATION],
        "AnalyticsAgent": [AgentCapability.ANALYTICS, AgentCapability.MONITORING],
        "ThumbnailAgent": [AgentCapability.THUMBNAILS, AgentCapability.VIDEO_PRODUCTION],
        "RetentionOptimizerAgent": [AgentCapability.CONTENT_GENERATION, AgentCapability.QUALITY_CHECK],
        "ValidatorAgent": [AgentCapability.VALIDATION, AgentCapability.QUALITY_CHECK],
        "WorkflowAgent": [AgentCapability.WORKFLOW, AgentCapability.SCHEDULING],
        "MonitorAgent": [AgentCapability.MONITORING, AgentCapability.RECOVERY],
        "RecoveryAgent": [AgentCapability.RECOVERY, AgentCapability.MONITORING],
        "SchedulerAgent": [AgentCapability.SCHEDULING, AgentCapability.WORKFLOW],
        "ComplianceAgent": [AgentCapability.COMPLIANCE, AgentCapability.VALIDATION],
        "ContentSafetyAgent": [AgentCapability.COMPLIANCE, AgentCapability.QUALITY_CHECK],
        "AudioQualityAgent": [AgentCapability.AUDIO_PROCESSING, AgentCapability.QUALITY_CHECK],
        "VideoQualityAgent": [AgentCapability.VIDEO_PRODUCTION, AgentCapability.QUALITY_CHECK],
        "AccessibilityAgent": [AgentCapability.COMPLIANCE, AgentCapability.QUALITY_CHECK],
        "SEOAgent": [AgentCapability.SEO_OPTIMIZATION, AgentCapability.CONTENT_GENERATION],
        "SEOStrategist": [AgentCapability.SEO_OPTIMIZATION, AgentCapability.ANALYTICS],
        "InsightAgent": [AgentCapability.ANALYTICS, AgentCapability.RESEARCH],
        "ContentStrategyAgent": [AgentCapability.CONTENT_GENERATION, AgentCapability.ANALYTICS],
    }

    # Agent dependencies (which agents must run before)
    AGENT_DEPENDENCIES = {
        "ValidatorAgent": ["QualityAgent", "VideoQualityAgent", "AudioQualityAgent"],
        "SEOAgent": ["ResearchAgent"],
        "ThumbnailAgent": ["ResearchAgent"],
        "VideoQualityAgent": ["AudioQualityAgent"],
        "ContentSafetyAgent": ["QualityAgent"],
    }

    def __init__(self):
        self._agents: Dict[str, AgentInfo] = {}
        self._capability_index: Dict[AgentCapability, List[str]] = {
            cap: [] for cap in AgentCapability
        }
        self._lock = threading.RLock()
        logger.info("AgentRegistry initialized")

    def register(
        self,
        name: str,
        agent_class: Type,
        capabilities: Optional[List[AgentCapability]] = None,
        dependencies: Optional[List[str]] = None,
        max_concurrent: int = 1,
        priority_weight: float = 1.0,
    ) -> None:
        """Register an agent with the registry."""
        with self._lock:
            if capabilities is None:
                capabilities = self.AGENT_CAPABILITIES.get(name, [])

            if dependencies is None:
                dependencies = self.AGENT_DEPENDENCIES.get(name, [])

            agent_info = AgentInfo(
                name=name,
                agent_class=agent_class,
                capabilities=capabilities,
                dependencies=dependencies,
                max_concurrent=max_concurrent,
                priority_weight=priority_weight,
            )
            self._agents[name] = agent_info

            # Update capability index
            for cap in capabilities:
                if name not in self._capability_index[cap]:
                    self._capability_index[cap].append(name)

            logger.debug(f"Registered agent: {name} with capabilities {[c.value for c in capabilities]}")

    def get_agent(self, name: str, create_instance: bool = True) -> Optional[AgentInfo]:
        """Get agent info, optionally creating instance."""
        with self._lock:
            agent_info = self._agents.get(name)
            if agent_info and create_instance and agent_info.instance is None:
                try:
                    agent_info.instance = agent_info.agent_class()
                    logger.debug(f"Created instance of {name}")
                except Exception as e:
                    logger.error(f"Failed to create instance of {name}: {e}")
            return agent_info

    def get_agents_by_capability(
        self,
        capability: AgentCapability,
        healthy_only: bool = True,
    ) -> List[AgentInfo]:
        """Get all agents with a specific capability."""
        with self._lock:
            agent_names = self._capability_index.get(capability, [])
            agents = [self._agents[name] for name in agent_names if name in self._agents]

            if healthy_only:
                agents = [a for a in agents if a.is_healthy]

            return agents

    def get_all_agents(self) -> Dict[str, AgentInfo]:
        """Get all registered agents."""
        with self._lock:
            return dict(self._agents)

    def update_stats(
        self,
        name: str,
        execution_time: float,
        success: bool,
    ) -> None:
        """Update agent execution statistics."""
        with self._lock:
            agent_info = self._agents.get(name)
            if agent_info:
                total = agent_info.total_executions
                agent_info.total_executions = total + 1

                # Update average execution time
                agent_info.avg_execution_time = (
                    (agent_info.avg_execution_time * total + execution_time) /
                    (total + 1)
                )

                # Update success rate
                successes = agent_info.success_rate * total
                agent_info.success_rate = (successes + (1 if success else 0)) / (total + 1)

                agent_info.last_execution = datetime.now()

    def set_health(self, name: str, is_healthy: bool) -> None:
        """Update agent health status."""
        with self._lock:
            if name in self._agents:
                self._agents[name].is_healthy = is_healthy

    def get_load_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current load status of all agents."""
        with self._lock:
            return {
                name: {
                    "current_load": info.current_load,
                    "max_concurrent": info.max_concurrent,
                    "utilization": info.current_load / max(info.max_concurrent, 1),
                    "is_healthy": info.is_healthy,
                }
                for name, info in self._agents.items()
            }


# ============================================================================
# Task Router
# ============================================================================


class TaskRouter:
    """
    Routes tasks to appropriate agents based on capabilities and load.

    Features:
    - Capability-based routing
    - Load-aware agent selection
    - Fallback handling
    - Priority-based routing
    """

    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self._routing_rules: Dict[str, Callable] = {}
        logger.info("TaskRouter initialized")

    def add_routing_rule(
        self,
        task_type: str,
        router: Callable[[Dict[str, Any]], str],
    ) -> None:
        """Add custom routing rule for a task type."""
        self._routing_rules[task_type] = router

    def route(
        self,
        task_type: str,
        capability: Optional[AgentCapability] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Route a task to the best available agent.

        Args:
            task_type: Type of task (e.g., "research", "quality_check")
            capability: Required capability
            params: Task parameters for custom routing

        Returns:
            Name of the selected agent or None
        """
        # Check custom routing rules first
        if task_type in self._routing_rules:
            return self._routing_rules[task_type](params or {})

        # Get agents with required capability
        if capability:
            agents = self.registry.get_agents_by_capability(capability)
        else:
            # Map task type to capability
            capability_map = {
                "research": AgentCapability.RESEARCH,
                "content": AgentCapability.CONTENT_GENERATION,
                "video": AgentCapability.VIDEO_PRODUCTION,
                "audio": AgentCapability.AUDIO_PROCESSING,
                "quality": AgentCapability.QUALITY_CHECK,
                "seo": AgentCapability.SEO_OPTIMIZATION,
                "analytics": AgentCapability.ANALYTICS,
                "schedule": AgentCapability.SCHEDULING,
                "monitor": AgentCapability.MONITORING,
                "recover": AgentCapability.RECOVERY,
                "validate": AgentCapability.VALIDATION,
                "compliance": AgentCapability.COMPLIANCE,
                "thumbnail": AgentCapability.THUMBNAILS,
            }
            cap = capability_map.get(task_type)
            agents = self.registry.get_agents_by_capability(cap) if cap else []

        if not agents:
            logger.warning(f"No agents available for task type: {task_type}")
            return None

        # Select best agent based on load and performance
        best_agent = self._select_best_agent(agents)
        return best_agent.name if best_agent else None

    def _select_best_agent(self, agents: List[AgentInfo]) -> Optional[AgentInfo]:
        """Select the best agent based on load, performance, and priority."""
        if not agents:
            return None

        # Score each agent (lower is better)
        def score_agent(agent: AgentInfo) -> float:
            load_score = agent.current_load / max(agent.max_concurrent, 1)
            perf_score = agent.avg_execution_time / 100  # Normalize
            success_penalty = 1 - agent.success_rate
            return (load_score * 0.5 + perf_score * 0.3 + success_penalty * 0.2) / agent.priority_weight

        agents.sort(key=score_agent)
        return agents[0]

    def get_fallback(self, agent_name: str) -> Optional[str]:
        """Get fallback agent for a failed agent."""
        agent_info = self.registry.get_agent(agent_name, create_instance=False)
        if not agent_info:
            return None

        # Find other agents with same capabilities
        for cap in agent_info.capabilities:
            agents = self.registry.get_agents_by_capability(cap)
            for agent in agents:
                if agent.name != agent_name and agent.is_healthy:
                    return agent.name
        return None


# ============================================================================
# Dependency Graph
# ============================================================================


class DependencyGraph:
    """
    Manages agent dependencies for proper execution order.

    Features:
    - Build dependency graph from agent registry
    - Topological sort for execution order
    - Parallel execution of independent agents
    - Cycle detection
    """

    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self._nodes: Dict[str, DependencyNode] = {}
        logger.info("DependencyGraph initialized")

    def build_graph(self, agent_names: List[str]) -> None:
        """Build dependency graph for specified agents."""
        self._nodes.clear()

        # Create nodes for all agents
        for name in agent_names:
            agent_info = self.registry.get_agent(name, create_instance=False)
            if agent_info:
                deps = set(agent_info.dependencies) & set(agent_names)
                self._nodes[name] = DependencyNode(
                    agent_name=name,
                    dependencies=deps,
                )

        # Build dependents (reverse dependencies)
        for name, node in self._nodes.items():
            for dep in node.dependencies:
                if dep in self._nodes:
                    self._nodes[dep].dependents.add(name)

    def get_execution_order(self) -> List[List[str]]:
        """
        Get topologically sorted execution order.

        Returns:
            List of execution levels (agents at same level can run in parallel)
        """
        if not self._nodes:
            return []

        # Reset state
        for node in self._nodes.values():
            node.resolved = False
            node.in_progress = False

        levels: List[List[str]] = []
        remaining = set(self._nodes.keys())

        while remaining:
            # Find all agents with resolved dependencies
            ready = []
            for name in remaining:
                node = self._nodes[name]
                if all(
                    self._nodes[dep].resolved
                    for dep in node.dependencies
                    if dep in self._nodes
                ):
                    ready.append(name)

            if not ready:
                # Cycle detected
                logger.error(f"Dependency cycle detected among: {remaining}")
                # Break cycle by picking first remaining
                ready = [next(iter(remaining))]

            levels.append(ready)
            for name in ready:
                self._nodes[name].resolved = True
                remaining.discard(name)

        return levels

    def get_parallel_groups(self, agents: List[str]) -> List[List[str]]:
        """Get groups of agents that can run in parallel."""
        self.build_graph(agents)
        return self.get_execution_order()

    def has_cycle(self) -> bool:
        """Check if the dependency graph has a cycle."""
        visited = set()
        rec_stack = set()

        def dfs(name: str) -> bool:
            visited.add(name)
            rec_stack.add(name)

            node = self._nodes.get(name)
            if node:
                for dep in node.dependencies:
                    if dep not in visited:
                        if dfs(dep):
                            return True
                    elif dep in rec_stack:
                        return True

            rec_stack.discard(name)
            return False

        return any(dfs(name) for name in self._nodes if name not in visited)


# ============================================================================
# Parallel Executor
# ============================================================================


class ParallelExecutor:
    """
    Executes multiple agents in parallel with thread pooling.

    Features:
    - Async execution with asyncio
    - Thread pool for blocking operations
    - Timeout handling
    - Result collection
    """

    def __init__(
        self,
        registry: AgentRegistry,
        max_workers: int = 8,
        default_timeout: float = 300.0,
    ):
        self.registry = registry
        self.max_workers = max_workers
        self.default_timeout = default_timeout
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        logger.info(f"ParallelExecutor initialized with {max_workers} workers")

    async def execute_parallel(
        self,
        tasks: List[Tuple[str, str, Dict[str, Any]]],
        timeout: Optional[float] = None,
    ) -> Dict[str, TaskResult]:
        """
        Execute multiple agent tasks in parallel.

        Args:
            tasks: List of (agent_name, method_name, params) tuples
            timeout: Maximum execution time per task

        Returns:
            Dict mapping task_id to TaskResult
        """
        timeout = timeout or self.default_timeout
        results: Dict[str, TaskResult] = {}

        async def execute_task(
            task_id: str,
            agent_name: str,
            method_name: str,
            params: Dict[str, Any],
        ) -> TaskResult:
            start_time = time.time()

            try:
                agent_info = self.registry.get_agent(agent_name)
                if not agent_info or not agent_info.instance:
                    return TaskResult(
                        task_id=task_id,
                        agent_name=agent_name,
                        success=False,
                        error=f"Agent {agent_name} not available",
                    )

                with self._lock:
                    agent_info.current_load += 1

                try:
                    # Get the method
                    method = getattr(agent_info.instance, method_name, None)
                    if method is None:
                        method = getattr(agent_info.instance, "run", None)
                        params = {"command": method_name, **params}

                    if method is None:
                        return TaskResult(
                            task_id=task_id,
                            agent_name=agent_name,
                            success=False,
                            error=f"Method {method_name} not found on {agent_name}",
                        )

                    # Execute in thread pool
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(
                            self._executor,
                            lambda: method(**params),
                        ),
                        timeout=timeout,
                    )

                    execution_time = time.time() - start_time
                    self.registry.update_stats(agent_name, execution_time, True)

                    # Extract success/data from result
                    success = True
                    data = result
                    error = None

                    if hasattr(result, "success"):
                        success = result.success
                    if hasattr(result, "data"):
                        data = result.data if hasattr(result, "data") else result
                    if hasattr(result, "error"):
                        error = result.error

                    return TaskResult(
                        task_id=task_id,
                        agent_name=agent_name,
                        success=success,
                        data=data,
                        error=error,
                        execution_time=execution_time,
                    )

                finally:
                    with self._lock:
                        agent_info.current_load -= 1

            except asyncio.TimeoutError:
                execution_time = time.time() - start_time
                self.registry.update_stats(agent_name, execution_time, False)
                return TaskResult(
                    task_id=task_id,
                    agent_name=agent_name,
                    success=False,
                    error=f"Task timed out after {timeout}s",
                    execution_time=execution_time,
                )

            except Exception as e:
                execution_time = time.time() - start_time
                self.registry.update_stats(agent_name, execution_time, False)
                logger.error(f"Task {task_id} failed: {e}")
                return TaskResult(
                    task_id=task_id,
                    agent_name=agent_name,
                    success=False,
                    error=str(e),
                    execution_time=execution_time,
                )

        # Create tasks
        coroutines = []
        for agent_name, method_name, params in tasks:
            task_id = f"{agent_name}_{method_name}_{uuid4().hex[:8]}"
            coroutines.append(execute_task(task_id, agent_name, method_name, params))

        # Execute all in parallel
        task_results = await asyncio.gather(*coroutines, return_exceptions=True)

        for result in task_results:
            if isinstance(result, Exception):
                logger.error(f"Task exception: {result}")
            elif isinstance(result, TaskResult):
                results[result.task_id] = result

        return results

    def execute_sync(
        self,
        tasks: List[Tuple[str, str, Dict[str, Any]]],
        timeout: Optional[float] = None,
    ) -> Dict[str, TaskResult]:
        """Synchronous wrapper for parallel execution."""
        return asyncio.run(self.execute_parallel(tasks, timeout))

    def shutdown(self):
        """Shutdown the executor."""
        self._executor.shutdown(wait=True)
        logger.info("ParallelExecutor shutdown complete")


# ============================================================================
# Result Aggregator
# ============================================================================


class ResultAggregator:
    """
    Aggregates and combines results from multiple agents.

    Features:
    - Merge results from parallel executions
    - Weighted aggregation
    - Conflict resolution
    - Summary generation
    """

    def __init__(self):
        self._aggregation_rules: Dict[str, Callable] = {}
        logger.info("ResultAggregator initialized")

    def add_rule(
        self,
        result_type: str,
        aggregator: Callable[[List[TaskResult]], Any],
    ) -> None:
        """Add custom aggregation rule."""
        self._aggregation_rules[result_type] = aggregator

    def aggregate(
        self,
        results: Dict[str, TaskResult],
        strategy: str = "merge",
    ) -> Dict[str, Any]:
        """
        Aggregate multiple task results.

        Args:
            results: Dict of task results
            strategy: Aggregation strategy (merge, first_success, vote, weighted)

        Returns:
            Aggregated result
        """
        if not results:
            return {"success": False, "error": "No results to aggregate"}

        successful = {k: v for k, v in results.items() if v.success}
        failed = {k: v for k, v in results.items() if not v.success}

        if strategy == "merge":
            return self._merge_results(successful, failed)
        elif strategy == "first_success":
            return self._first_success(successful, failed)
        elif strategy == "vote":
            return self._vote_results(successful)
        elif strategy == "weighted":
            return self._weighted_aggregate(successful)
        else:
            return self._merge_results(successful, failed)

    def _merge_results(
        self,
        successful: Dict[str, TaskResult],
        failed: Dict[str, TaskResult],
    ) -> Dict[str, Any]:
        """Merge all successful results."""
        merged_data = {}
        errors = []

        for task_id, result in successful.items():
            if isinstance(result.data, dict):
                merged_data.update(result.data)
            else:
                merged_data[task_id] = result.data

        for task_id, result in failed.items():
            if result.error:
                errors.append(f"{task_id}: {result.error}")

        return {
            "success": len(successful) > 0,
            "data": merged_data,
            "errors": errors,
            "successful_count": len(successful),
            "failed_count": len(failed),
            "total_execution_time": sum(r.execution_time for r in {**successful, **failed}.values()),
        }

    def _first_success(
        self,
        successful: Dict[str, TaskResult],
        failed: Dict[str, TaskResult],
    ) -> Dict[str, Any]:
        """Return first successful result."""
        if successful:
            first = next(iter(successful.values()))
            return {
                "success": True,
                "data": first.data,
                "agent": first.agent_name,
                "execution_time": first.execution_time,
            }

        errors = [f"{r.agent_name}: {r.error}" for r in failed.values()]
        return {
            "success": False,
            "errors": errors,
        }

    def _vote_results(self, successful: Dict[str, TaskResult]) -> Dict[str, Any]:
        """Return most common result (voting)."""
        if not successful:
            return {"success": False, "error": "No successful results"}

        # Simple voting - count occurrences of each result
        votes: Dict[str, int] = {}
        for result in successful.values():
            key = str(result.data)
            votes[key] = votes.get(key, 0) + 1

        winner = max(votes.items(), key=lambda x: x[1])
        return {
            "success": True,
            "data": winner[0],
            "vote_count": winner[1],
            "total_votes": len(successful),
        }

    def _weighted_aggregate(self, successful: Dict[str, TaskResult]) -> Dict[str, Any]:
        """Weighted aggregation based on agent performance."""
        if not successful:
            return {"success": False, "error": "No successful results"}

        # Weight by execution time (faster is better)
        total_weight = sum(1 / max(r.execution_time, 0.1) for r in successful.values())

        weighted_data = {}
        for result in successful.values():
            weight = (1 / max(result.execution_time, 0.1)) / total_weight
            weighted_data[result.agent_name] = {
                "data": result.data,
                "weight": weight,
            }

        return {
            "success": True,
            "weighted_results": weighted_data,
            "total_weight": total_weight,
        }

    def generate_summary(self, results: Dict[str, TaskResult]) -> str:
        """Generate human-readable summary of results."""
        successful = sum(1 for r in results.values() if r.success)
        failed = len(results) - successful
        total_time = sum(r.execution_time for r in results.values())

        lines = [
            f"Execution Summary",
            f"================",
            f"Total Tasks: {len(results)}",
            f"Successful: {successful}",
            f"Failed: {failed}",
            f"Total Time: {total_time:.2f}s",
            "",
        ]

        if failed > 0:
            lines.append("Failures:")
            for task_id, result in results.items():
                if not result.success:
                    lines.append(f"  - {result.agent_name}: {result.error}")

        return "\n".join(lines)


# ============================================================================
# Load Balancer
# ============================================================================


class LoadBalancer:
    """
    Distributes work evenly across available agents.

    Features:
    - Round-robin distribution
    - Least-connections algorithm
    - Weighted distribution
    - Health-aware routing
    """

    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self._round_robin_index: Dict[AgentCapability, int] = {}
        self._lock = threading.Lock()
        logger.info("LoadBalancer initialized")

    def get_next_agent(
        self,
        capability: AgentCapability,
        strategy: str = "least_loaded",
    ) -> Optional[str]:
        """
        Get the next agent for a task.

        Args:
            capability: Required capability
            strategy: Load balancing strategy (round_robin, least_loaded, weighted)

        Returns:
            Agent name or None
        """
        agents = self.registry.get_agents_by_capability(capability)
        if not agents:
            return None

        if strategy == "round_robin":
            return self._round_robin(capability, agents)
        elif strategy == "least_loaded":
            return self._least_loaded(agents)
        elif strategy == "weighted":
            return self._weighted(agents)
        else:
            return self._least_loaded(agents)

    def _round_robin(
        self,
        capability: AgentCapability,
        agents: List[AgentInfo],
    ) -> str:
        """Round-robin selection."""
        with self._lock:
            idx = self._round_robin_index.get(capability, 0)
            agent = agents[idx % len(agents)]
            self._round_robin_index[capability] = idx + 1
            return agent.name

    def _least_loaded(self, agents: List[AgentInfo]) -> str:
        """Select least loaded agent."""
        def load_score(agent: AgentInfo) -> float:
            return agent.current_load / max(agent.max_concurrent, 1)

        agents.sort(key=load_score)
        return agents[0].name

    def _weighted(self, agents: List[AgentInfo]) -> str:
        """Weighted selection based on priority and performance."""
        def weight_score(agent: AgentInfo) -> float:
            load_factor = 1 - (agent.current_load / max(agent.max_concurrent, 1))
            perf_factor = agent.success_rate
            return load_factor * perf_factor * agent.priority_weight

        agents.sort(key=weight_score, reverse=True)
        return agents[0].name

    def get_distribution_stats(self) -> Dict[str, Any]:
        """Get current load distribution statistics."""
        load_status = self.registry.get_load_status()
        total_load = sum(s["current_load"] for s in load_status.values())
        total_capacity = sum(s["max_concurrent"] for s in load_status.values())

        return {
            "total_load": total_load,
            "total_capacity": total_capacity,
            "utilization": total_load / max(total_capacity, 1),
            "by_agent": load_status,
        }


# ============================================================================
# Master Orchestrator
# ============================================================================


class MasterOrchestrator:
    """
    Unified orchestration system that coordinates all agents.

    Combines:
    - AgentRegistry for agent management
    - TaskRouter for intelligent routing
    - DependencyGraph for execution order
    - ParallelExecutor for concurrent execution
    - ResultAggregator for result combination
    - LoadBalancer for work distribution
    """

    def __init__(
        self,
        max_workers: int = 8,
        default_timeout: float = 300.0,
    ):
        self.registry = AgentRegistry()
        self.router = TaskRouter(self.registry)
        self.dependency_graph = DependencyGraph(self.registry)
        self.executor = ParallelExecutor(
            self.registry,
            max_workers=max_workers,
            default_timeout=default_timeout,
        )
        self.aggregator = ResultAggregator()
        self.load_balancer = LoadBalancer(self.registry)

        self._workflows: Dict[str, WorkflowResult] = {}
        self._lock = threading.Lock()

        logger.info("MasterOrchestrator initialized")

    def auto_register_agents(self) -> int:
        """
        Automatically register all available agents.

        Returns:
            Number of agents registered
        """
        count = 0

        try:
            # Import all agents
            from . import (
                ResearchAgent,
                QualityAgent,
                AnalyticsAgent,
                ThumbnailAgent,
                RetentionOptimizerAgent,
                ValidatorAgent,
                WorkflowAgent,
                MonitorAgent,
                RecoveryAgent,
                SchedulerAgent,
                ComplianceAgent,
                ContentSafetyAgent,
                AudioQualityAgent,
                VideoQualityAgent,
                AccessibilityAgent,
                SEOAgent,
                SEOStrategist,
            )

            agents = [
                ("ResearchAgent", ResearchAgent),
                ("QualityAgent", QualityAgent),
                ("AnalyticsAgent", AnalyticsAgent),
                ("ThumbnailAgent", ThumbnailAgent),
                ("RetentionOptimizerAgent", RetentionOptimizerAgent),
                ("ValidatorAgent", ValidatorAgent),
                ("WorkflowAgent", WorkflowAgent),
                ("MonitorAgent", MonitorAgent),
                ("RecoveryAgent", RecoveryAgent),
                ("SchedulerAgent", SchedulerAgent),
                ("ComplianceAgent", ComplianceAgent),
                ("ContentSafetyAgent", ContentSafetyAgent),
                ("AudioQualityAgent", AudioQualityAgent),
                ("VideoQualityAgent", VideoQualityAgent),
                ("AccessibilityAgent", AccessibilityAgent),
                ("SEOAgent", SEOAgent),
                ("SEOStrategist", SEOStrategist),
            ]

            # Try to import additional agents
            try:
                from . import InsightAgent
                agents.append(("InsightAgent", InsightAgent))
            except ImportError:
                pass

            try:
                from . import ContentStrategyAgent
                agents.append(("ContentStrategyAgent", ContentStrategyAgent))
            except ImportError:
                pass

            for name, agent_class in agents:
                try:
                    self.registry.register(name, agent_class)
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to register {name}: {e}")

            logger.info(f"Auto-registered {count} agents")

        except ImportError as e:
            logger.error(f"Failed to import agents: {e}")

        return count

    async def execute_workflow(
        self,
        workflow_type: str,
        **params,
    ) -> WorkflowResult:
        """
        Execute a workflow with coordinated agent execution.

        Args:
            workflow_type: Type of workflow (full_video, short_video, research_only, etc.)
            **params: Workflow parameters

        Returns:
            WorkflowResult with execution details
        """
        workflow_id = f"wf_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"

        workflow_result = WorkflowResult(
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            status=ExecutionStatus.RUNNING,
            success=False,
            started_at=datetime.now(),
        )

        with self._lock:
            self._workflows[workflow_id] = workflow_result

        logger.info(f"Starting workflow: {workflow_id} ({workflow_type})")
        start_time = time.time()

        try:
            # Get workflow definition
            workflow_def = self._get_workflow_definition(workflow_type)
            if not workflow_def:
                workflow_result.status = ExecutionStatus.FAILED
                workflow_result.errors.append(f"Unknown workflow type: {workflow_type}")
                return workflow_result

            # Build dependency graph
            agent_names = [step["agent"] for step in workflow_def]
            self.dependency_graph.build_graph(agent_names)

            # Get execution levels
            levels = self.dependency_graph.get_execution_order()

            # Execute each level
            for level_idx, level_agents in enumerate(levels):
                logger.info(f"Executing level {level_idx + 1}: {level_agents}")

                # Get tasks for this level
                tasks = []
                for agent_name in level_agents:
                    step_def = next(
                        (s for s in workflow_def if s["agent"] == agent_name),
                        None
                    )
                    if step_def:
                        step_params = {**params, **step_def.get("params", {})}
                        tasks.append((
                            agent_name,
                            step_def.get("method", "run"),
                            step_params,
                        ))

                # Execute level in parallel
                level_results = await self.executor.execute_parallel(tasks)

                # Check for failures
                for task_id, result in level_results.items():
                    workflow_result.results[task_id] = result
                    if not result.success:
                        workflow_result.errors.append(
                            f"{result.agent_name}: {result.error}"
                        )

                        # Check if this is a critical failure
                        step_def = next(
                            (s for s in workflow_def if s["agent"] == result.agent_name),
                            None
                        )
                        if step_def and step_def.get("required", True):
                            workflow_result.status = ExecutionStatus.FAILED
                            break

                if workflow_result.status == ExecutionStatus.FAILED:
                    break

            # Finalize
            if workflow_result.status != ExecutionStatus.FAILED:
                workflow_result.status = ExecutionStatus.COMPLETED
                workflow_result.success = True

        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {e}")
            workflow_result.status = ExecutionStatus.FAILED
            workflow_result.errors.append(str(e))

        workflow_result.completed_at = datetime.now()
        workflow_result.total_time = time.time() - start_time

        logger.info(
            f"Workflow {workflow_id} completed: {workflow_result.status.value} "
            f"({workflow_result.total_time:.2f}s)"
        )

        return workflow_result

    def _get_workflow_definition(self, workflow_type: str) -> Optional[List[Dict]]:
        """Get workflow definition by type."""
        workflows = {
            "full_video": [
                {"agent": "ResearchAgent", "method": "find_topics", "required": True},
                {"agent": "SEOAgent", "method": "optimize", "required": True},
                {"agent": "QualityAgent", "method": "quick_check", "required": True},
                {"agent": "ThumbnailAgent", "method": "generate", "required": True},
                {"agent": "AudioQualityAgent", "method": "check", "required": True},
                {"agent": "VideoQualityAgent", "method": "check", "required": True},
                {"agent": "ValidatorAgent", "method": "run", "required": True},
            ],
            "short_video": [
                {"agent": "ResearchAgent", "method": "generate_viral_ideas", "required": True},
                {"agent": "QualityAgent", "method": "quick_check", "required": True},
                {"agent": "ValidatorAgent", "method": "run", "required": True},
            ],
            "research_only": [
                {"agent": "ResearchAgent", "method": "find_topics", "required": True},
                {"agent": "AnalyticsAgent", "method": "find_patterns", "required": False},
            ],
            "quality_check": [
                {"agent": "QualityAgent", "method": "full_analysis", "required": True},
                {"agent": "ContentSafetyAgent", "method": "check", "required": True},
                {"agent": "ComplianceAgent", "method": "check", "required": True},
                {"agent": "AccessibilityAgent", "method": "check", "required": False},
            ],
            "analytics": [
                {"agent": "AnalyticsAgent", "method": "analyze_channel", "required": True},
                {"agent": "SEOStrategist", "method": "analyze", "required": False},
            ],
        }
        return workflows.get(workflow_type)

    async def run_parallel(
        self,
        tasks: List[Tuple[str, str, Dict[str, Any]]],
        timeout: Optional[float] = None,
    ) -> Dict[str, TaskResult]:
        """
        Run multiple agent tasks in parallel.

        Args:
            tasks: List of (agent_name, method_name, params) tuples
            timeout: Maximum execution time per task

        Returns:
            Dict of task results
        """
        return await self.executor.execute_parallel(tasks, timeout)

    def run_parallel_sync(
        self,
        tasks: List[Tuple[str, str, Dict[str, Any]]],
        timeout: Optional[float] = None,
    ) -> Dict[str, TaskResult]:
        """Synchronous wrapper for parallel execution."""
        return self.executor.execute_sync(tasks, timeout)

    def route_task(
        self,
        task_type: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Route a task to the best available agent."""
        return self.router.route(task_type, params=params)

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            "registered_agents": len(self.registry.get_all_agents()),
            "active_workflows": len([
                w for w in self._workflows.values()
                if w.status == ExecutionStatus.RUNNING
            ]),
            "completed_workflows": len([
                w for w in self._workflows.values()
                if w.status == ExecutionStatus.COMPLETED
            ]),
            "failed_workflows": len([
                w for w in self._workflows.values()
                if w.status == ExecutionStatus.FAILED
            ]),
            "load_distribution": self.load_balancer.get_distribution_stats(),
        }

    def get_workflow(self, workflow_id: str) -> Optional[WorkflowResult]:
        """Get workflow by ID."""
        return self._workflows.get(workflow_id)

    def shutdown(self):
        """Shutdown the orchestrator."""
        self.executor.shutdown()
        logger.info("MasterOrchestrator shutdown complete")


# ============================================================================
# Singleton Instance
# ============================================================================


_master_orchestrator: Optional[MasterOrchestrator] = None


def get_master_orchestrator(
    max_workers: int = 8,
    default_timeout: float = 300.0,
) -> MasterOrchestrator:
    """Get or create the global MasterOrchestrator instance."""
    global _master_orchestrator
    if _master_orchestrator is None:
        _master_orchestrator = MasterOrchestrator(
            max_workers=max_workers,
            default_timeout=default_timeout,
        )
    return _master_orchestrator


# ============================================================================
# CLI Entry Point
# ============================================================================


def main():
    """CLI entry point for master orchestrator."""
    import sys

    print("\n" + "=" * 60)
    print("  MASTER ORCHESTRATOR")
    print("=" * 60 + "\n")

    orchestrator = get_master_orchestrator()
    count = orchestrator.auto_register_agents()
    print(f"Registered {count} agents\n")

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "status":
            status = orchestrator.get_status()
            print(f"Registered Agents: {status['registered_agents']}")
            print(f"Active Workflows: {status['active_workflows']}")
            print(f"Completed Workflows: {status['completed_workflows']}")
            print(f"Failed Workflows: {status['failed_workflows']}")
            print(f"\nLoad Distribution:")
            for agent, load in status['load_distribution']['by_agent'].items():
                print(f"  {agent}: {load['current_load']}/{load['max_concurrent']}")

        elif command == "workflow" and len(sys.argv) > 2:
            workflow_type = sys.argv[2]
            channel_id = sys.argv[3] if len(sys.argv) > 3 else "money_blueprints"

            async def run_workflow():
                result = await orchestrator.execute_workflow(
                    workflow_type=workflow_type,
                    channel_id=channel_id,
                )
                return result

            result = asyncio.run(run_workflow())
            print(f"Workflow: {result.workflow_id}")
            print(f"Status: {result.status.value}")
            print(f"Success: {result.success}")
            print(f"Total Time: {result.total_time:.2f}s")
            if result.errors:
                print(f"Errors: {result.errors}")

        elif command == "agents":
            agents = orchestrator.registry.get_all_agents()
            print(f"Registered Agents ({len(agents)}):\n")
            for name, info in agents.items():
                caps = [c.value for c in info.capabilities]
                print(f"  {name}")
                print(f"    Capabilities: {', '.join(caps)}")
                print(f"    Dependencies: {', '.join(info.dependencies) or 'None'}")
                print()

    else:
        print("Usage:")
        print("  python -m src.agents.master_orchestrator status")
        print("  python -m src.agents.master_orchestrator agents")
        print("  python -m src.agents.master_orchestrator workflow <type> [channel_id]")
        print()
        print("Workflow types:")
        print("  full_video     - Complete video creation workflow")
        print("  short_video    - YouTube Shorts workflow")
        print("  research_only  - Research and analysis only")
        print("  quality_check  - Content quality validation")
        print("  analytics      - Performance analysis")

    orchestrator.shutdown()


if __name__ == "__main__":
    main()
