"""
Workflow Agent - Pipeline Orchestration

Manages pipeline state machine and coordinates agent handoffs
for the YouTube video automation workflow.

State Machine:
    pending -> running -> completed/failed

Usage:
    from src.agents.workflow_agent import WorkflowAgent

    agent = WorkflowAgent()

    # Start a new workflow
    result = agent.run(channel_id="money_blueprints", topic="passive income")

    # Check workflow status
    result = agent.get_status(workflow_id="wf_20260119_123456")

    # Resume a failed workflow
    result = agent.resume(workflow_id="wf_20260119_123456")
"""

import json
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from loguru import logger


class WorkflowStatus(Enum):
    """Workflow status states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class WorkflowStep(Enum):
    """Pipeline steps in order."""
    RESEARCH = "research"
    SCRIPT = "script"
    AUDIO = "audio"
    VIDEO = "video"
    THUMBNAIL = "thumbnail"
    QUALITY_CHECK = "quality_check"
    UPLOAD = "upload"


# Step order for progress calculation
STEP_ORDER = [
    WorkflowStep.RESEARCH,
    WorkflowStep.SCRIPT,
    WorkflowStep.AUDIO,
    WorkflowStep.VIDEO,
    WorkflowStep.THUMBNAIL,
    WorkflowStep.QUALITY_CHECK,
    WorkflowStep.UPLOAD,
]


@dataclass
class WorkflowResult:
    """Result from workflow agent operations."""
    workflow_id: str
    status: str
    current_step: str
    progress_percent: float
    success: bool = True
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: float = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkflowState:
    """Persistent workflow state."""
    workflow_id: str
    channel_id: str
    topic: str
    status: str = WorkflowStatus.PENDING.value
    current_step: str = WorkflowStep.RESEARCH.value
    progress_percent: float = 0.0

    # Step outputs
    step_results: Dict[str, Any] = field(default_factory=dict)
    step_timestamps: Dict[str, str] = field(default_factory=dict)

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    errors: List[str] = field(default_factory=list)

    # Dependencies
    parallel_tasks: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseAgent:
    """Base class for all agents."""

    def __init__(self, name: str):
        self.name = name
        self.state_dir = Path("data/workflow_states")
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def run(self, **kwargs) -> Any:
        """Execute the agent's main function. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement run()")

    def _save_state(self, state: Dict[str, Any], filename: str):
        """Save state to JSON file."""
        filepath = self.state_dir / filename
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def _load_state(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load state from JSON file."""
        filepath = self.state_dir / filename
        if filepath.exists():
            with open(filepath) as f:
                return json.load(f)
        return None


class WorkflowAgent(BaseAgent):
    """
    Workflow Agent - Pipeline orchestration.

    Manages the state machine for video production workflows:
    - Tracks workflow progress with percentage completion
    - Coordinates agent handoffs between steps
    - Handles parallel processing where possible
    - Manages dependencies between agents
    - Gracefully handles failures with rollback
    """

    # Dependency map: step -> list of required previous steps
    STEP_DEPENDENCIES = {
        WorkflowStep.RESEARCH: [],
        WorkflowStep.SCRIPT: [WorkflowStep.RESEARCH],
        WorkflowStep.AUDIO: [WorkflowStep.SCRIPT],
        WorkflowStep.VIDEO: [WorkflowStep.AUDIO],
        WorkflowStep.THUMBNAIL: [WorkflowStep.SCRIPT],  # Can run parallel to video
        WorkflowStep.QUALITY_CHECK: [WorkflowStep.VIDEO, WorkflowStep.THUMBNAIL],
        WorkflowStep.UPLOAD: [WorkflowStep.QUALITY_CHECK],
    }

    # Parallel groups: steps that can run simultaneously
    PARALLEL_GROUPS = [
        [WorkflowStep.VIDEO, WorkflowStep.THUMBNAIL],  # These can run in parallel
    ]

    def __init__(self):
        super().__init__("WorkflowAgent")
        self.workflows: Dict[str, WorkflowState] = {}
        self._load_workflows()
        self._step_handlers: Dict[WorkflowStep, Callable] = {}
        self._register_default_handlers()
        logger.info(f"{self.name} initialized")

    def _load_workflows(self):
        """Load existing workflows from disk."""
        workflow_dir = self.state_dir / "workflows"
        workflow_dir.mkdir(exist_ok=True)

        for filepath in workflow_dir.glob("*.json"):
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    state = WorkflowState(**data)
                    self.workflows[state.workflow_id] = state
            except Exception as e:
                logger.warning(f"Failed to load workflow {filepath}: {e}")

    def _save_workflow(self, state: WorkflowState):
        """Save workflow state to disk."""
        workflow_dir = self.state_dir / "workflows"
        workflow_dir.mkdir(exist_ok=True)
        filepath = workflow_dir / f"{state.workflow_id}.json"
        with open(filepath, "w") as f:
            json.dump(state.to_dict(), f, indent=2, default=str)

    def _register_default_handlers(self):
        """Register default step handlers."""
        # Default handlers that just mark completion
        # Real implementations should be registered via register_handler()
        for step in WorkflowStep:
            self._step_handlers[step] = self._default_handler

    def _default_handler(self, state: WorkflowState, **kwargs) -> Dict[str, Any]:
        """Default step handler (placeholder)."""
        return {"success": True, "message": "Step completed"}

    def register_handler(self, step: WorkflowStep, handler: Callable):
        """Register a custom handler for a workflow step."""
        self._step_handlers[step] = handler
        logger.debug(f"Registered handler for step: {step.value}")

    def _generate_workflow_id(self) -> str:
        """Generate a unique workflow ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"wf_{timestamp}"

    def _calculate_progress(self, current_step: WorkflowStep, completed_steps: List[str]) -> float:
        """Calculate progress percentage based on completed steps."""
        total_steps = len(STEP_ORDER)
        completed_count = len(completed_steps)

        # Add partial progress for current step
        current_idx = STEP_ORDER.index(current_step)
        progress = (completed_count / total_steps) * 100

        return min(progress, 100.0)

    def _check_dependencies(self, step: WorkflowStep, state: WorkflowState) -> bool:
        """Check if all dependencies for a step are satisfied."""
        deps = self.STEP_DEPENDENCIES.get(step, [])
        for dep in deps:
            if dep.value not in state.step_results:
                return False
            if not state.step_results[dep.value].get("success", False):
                return False
        return True

    def _get_parallel_steps(self, current_step: WorkflowStep, state: WorkflowState) -> List[WorkflowStep]:
        """Get steps that can run in parallel with the current step."""
        parallel_steps = []
        for group in self.PARALLEL_GROUPS:
            if current_step in group:
                for step in group:
                    if step != current_step and self._check_dependencies(step, state):
                        if step.value not in state.step_results:
                            parallel_steps.append(step)
        return parallel_steps

    def run(
        self,
        channel_id: str,
        topic: str,
        upload: bool = False,
        **kwargs
    ) -> WorkflowResult:
        """
        Execute a complete workflow for video creation.

        Args:
            channel_id: YouTube channel ID
            topic: Video topic/niche
            upload: Whether to upload to YouTube
            **kwargs: Additional parameters

        Returns:
            WorkflowResult with status and progress
        """
        # Create new workflow state
        workflow_id = self._generate_workflow_id()
        state = WorkflowState(
            workflow_id=workflow_id,
            channel_id=channel_id,
            topic=topic,
        )
        self.workflows[workflow_id] = state
        self._save_workflow(state)

        logger.info(f"[{self.name}] Starting workflow: {workflow_id}")
        logger.info(f"  Channel: {channel_id}")
        logger.info(f"  Topic: {topic}")

        start_time = time.time()
        state.status = WorkflowStatus.RUNNING.value
        state.updated_at = datetime.now().isoformat()

        try:
            # Execute each step in order
            for step in STEP_ORDER:
                # Skip upload if not requested
                if step == WorkflowStep.UPLOAD and not upload:
                    state.step_results[step.value] = {
                        "success": True,
                        "skipped": True,
                        "reason": "upload=False"
                    }
                    continue

                # Check dependencies
                if not self._check_dependencies(step, state):
                    dep_names = [d.value for d in self.STEP_DEPENDENCIES.get(step, [])]
                    state.errors.append(f"Dependencies not met for {step.value}: {dep_names}")
                    state.status = WorkflowStatus.FAILED.value
                    break

                # Update current step
                state.current_step = step.value
                completed_steps = [s for s in state.step_results if state.step_results[s].get("success")]
                state.progress_percent = self._calculate_progress(step, completed_steps)
                state.updated_at = datetime.now().isoformat()
                self._save_workflow(state)

                logger.info(f"[{self.name}] Step: {step.value} ({state.progress_percent:.0f}%)")

                # Check for parallel tasks
                parallel = self._get_parallel_steps(step, state)
                if parallel:
                    logger.info(f"  Parallel tasks available: {[s.value for s in parallel]}")
                    # Execute parallel tasks in threads
                    threads = []
                    results = {}
                    for pstep in parallel:
                        t = threading.Thread(
                            target=self._execute_step_thread,
                            args=(pstep, state, results, kwargs)
                        )
                        threads.append(t)
                        t.start()

                    # Execute current step in main thread
                    result = self._execute_step(step, state, **kwargs)
                    state.step_results[step.value] = result
                    state.step_timestamps[step.value] = datetime.now().isoformat()

                    # Wait for parallel tasks
                    for t in threads:
                        t.join()

                    # Merge parallel results
                    state.step_results.update(results)
                else:
                    # Execute step
                    result = self._execute_step(step, state, **kwargs)
                    state.step_results[step.value] = result
                    state.step_timestamps[step.value] = datetime.now().isoformat()

                # Check for failure
                if not result.get("success", False):
                    state.errors.append(f"Step {step.value} failed: {result.get('error', 'Unknown')}")
                    state.status = WorkflowStatus.FAILED.value
                    break

            # Mark completion
            if state.status != WorkflowStatus.FAILED.value:
                state.status = WorkflowStatus.COMPLETED.value
                state.progress_percent = 100.0

            state.completed_at = datetime.now().isoformat()

        except Exception as e:
            logger.error(f"[{self.name}] Workflow failed: {e}")
            state.status = WorkflowStatus.FAILED.value
            state.errors.append(str(e))

        duration = time.time() - start_time
        state.updated_at = datetime.now().isoformat()
        self._save_workflow(state)

        success = state.status == WorkflowStatus.COMPLETED.value
        logger.info(f"[{self.name}] Workflow {workflow_id}: {state.status} ({duration:.1f}s)")

        return WorkflowResult(
            workflow_id=workflow_id,
            status=state.status,
            current_step=state.current_step,
            progress_percent=state.progress_percent,
            success=success,
            data=state.step_results,
            errors=state.errors,
            started_at=state.created_at,
            completed_at=state.completed_at,
            duration_seconds=duration
        )

    def _execute_step(self, step: WorkflowStep, state: WorkflowState, **kwargs) -> Dict[str, Any]:
        """Execute a single workflow step."""
        handler = self._step_handlers.get(step)
        if handler:
            try:
                return handler(state, **kwargs)
            except Exception as e:
                logger.error(f"Step {step.value} handler error: {e}")
                return {"success": False, "error": str(e)}
        return {"success": False, "error": f"No handler for step {step.value}"}

    def _execute_step_thread(
        self,
        step: WorkflowStep,
        state: WorkflowState,
        results: Dict[str, Any],
        kwargs: Dict[str, Any]
    ):
        """Execute a step in a thread for parallel processing."""
        result = self._execute_step(step, state, **kwargs)
        results[step.value] = result

    def get_status(self, workflow_id: str) -> WorkflowResult:
        """
        Get the current status of a workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            WorkflowResult with current status
        """
        state = self.workflows.get(workflow_id)
        if not state:
            return WorkflowResult(
                workflow_id=workflow_id,
                status="not_found",
                current_step="",
                progress_percent=0,
                success=False,
                errors=[f"Workflow {workflow_id} not found"]
            )

        return WorkflowResult(
            workflow_id=workflow_id,
            status=state.status,
            current_step=state.current_step,
            progress_percent=state.progress_percent,
            success=state.status == WorkflowStatus.COMPLETED.value,
            data=state.step_results,
            errors=state.errors,
            started_at=state.created_at,
            completed_at=state.completed_at
        )

    def resume(self, workflow_id: str, **kwargs) -> WorkflowResult:
        """
        Resume a failed or paused workflow.

        Args:
            workflow_id: Workflow to resume
            **kwargs: Additional parameters

        Returns:
            WorkflowResult with completion status
        """
        state = self.workflows.get(workflow_id)
        if not state:
            return WorkflowResult(
                workflow_id=workflow_id,
                status="not_found",
                current_step="",
                progress_percent=0,
                success=False,
                errors=[f"Workflow {workflow_id} not found"]
            )

        if state.status not in [WorkflowStatus.FAILED.value, WorkflowStatus.PAUSED.value]:
            return WorkflowResult(
                workflow_id=workflow_id,
                status=state.status,
                current_step=state.current_step,
                progress_percent=state.progress_percent,
                success=False,
                errors=[f"Workflow is {state.status}, cannot resume"]
            )

        logger.info(f"[{self.name}] Resuming workflow: {workflow_id} from step {state.current_step}")

        # Find the step to resume from
        current_step_enum = WorkflowStep(state.current_step)
        start_idx = STEP_ORDER.index(current_step_enum)

        state.status = WorkflowStatus.RUNNING.value
        state.errors = []  # Clear previous errors
        state.updated_at = datetime.now().isoformat()

        start_time = time.time()

        try:
            for step in STEP_ORDER[start_idx:]:
                if not self._check_dependencies(step, state):
                    continue

                state.current_step = step.value
                completed_steps = [s for s in state.step_results if state.step_results[s].get("success")]
                state.progress_percent = self._calculate_progress(step, completed_steps)
                state.updated_at = datetime.now().isoformat()
                self._save_workflow(state)

                logger.info(f"[{self.name}] Resuming step: {step.value}")

                result = self._execute_step(step, state, **kwargs)
                state.step_results[step.value] = result
                state.step_timestamps[step.value] = datetime.now().isoformat()

                if not result.get("success", False):
                    state.errors.append(f"Step {step.value} failed: {result.get('error')}")
                    state.status = WorkflowStatus.FAILED.value
                    break

            if state.status != WorkflowStatus.FAILED.value:
                state.status = WorkflowStatus.COMPLETED.value
                state.progress_percent = 100.0

            state.completed_at = datetime.now().isoformat()

        except Exception as e:
            state.status = WorkflowStatus.FAILED.value
            state.errors.append(str(e))

        duration = time.time() - start_time
        self._save_workflow(state)

        return WorkflowResult(
            workflow_id=workflow_id,
            status=state.status,
            current_step=state.current_step,
            progress_percent=state.progress_percent,
            success=state.status == WorkflowStatus.COMPLETED.value,
            data=state.step_results,
            errors=state.errors,
            duration_seconds=duration
        )

    def pause(self, workflow_id: str) -> WorkflowResult:
        """Pause a running workflow."""
        state = self.workflows.get(workflow_id)
        if not state:
            return WorkflowResult(
                workflow_id=workflow_id,
                status="not_found",
                current_step="",
                progress_percent=0,
                success=False,
                errors=[f"Workflow {workflow_id} not found"]
            )

        if state.status == WorkflowStatus.RUNNING.value:
            state.status = WorkflowStatus.PAUSED.value
            state.updated_at = datetime.now().isoformat()
            self._save_workflow(state)
            logger.info(f"[{self.name}] Paused workflow: {workflow_id}")

        return self.get_status(workflow_id)

    def list_workflows(
        self,
        status: Optional[str] = None,
        channel_id: Optional[str] = None
    ) -> List[WorkflowResult]:
        """
        List all workflows, optionally filtered by status or channel.

        Args:
            status: Filter by status (pending, running, completed, failed)
            channel_id: Filter by channel

        Returns:
            List of WorkflowResult
        """
        results = []
        for wf_id, state in self.workflows.items():
            if status and state.status != status:
                continue
            if channel_id and state.channel_id != channel_id:
                continue
            results.append(self.get_status(wf_id))
        return results

    def cleanup_old_workflows(self, days: int = 30) -> int:
        """
        Clean up workflows older than specified days.

        Args:
            days: Remove workflows older than this many days

        Returns:
            Number of workflows removed
        """
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=days)
        removed = 0

        for wf_id in list(self.workflows.keys()):
            state = self.workflows[wf_id]
            created = datetime.fromisoformat(state.created_at)
            if created < cutoff:
                del self.workflows[wf_id]
                filepath = self.state_dir / "workflows" / f"{wf_id}.json"
                if filepath.exists():
                    filepath.unlink()
                removed += 1

        logger.info(f"[{self.name}] Cleaned up {removed} old workflows")
        return removed


# CLI entry point
if __name__ == "__main__":
    import sys

    print("\n" + "=" * 60)
    print("  WORKFLOW AGENT TEST")
    print("=" * 60 + "\n")

    agent = WorkflowAgent()

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "list":
            workflows = agent.list_workflows()
            print(f"Found {len(workflows)} workflows:\n")
            for wf in workflows:
                print(f"  {wf.workflow_id}: {wf.status} ({wf.progress_percent:.0f}%)")

        elif command == "status" and len(sys.argv) > 2:
            result = agent.get_status(sys.argv[2])
            print(f"Workflow: {result.workflow_id}")
            print(f"Status: {result.status}")
            print(f"Progress: {result.progress_percent:.0f}%")
            print(f"Current Step: {result.current_step}")

        elif command == "run":
            channel = sys.argv[2] if len(sys.argv) > 2 else "money_blueprints"
            topic = sys.argv[3] if len(sys.argv) > 3 else "passive income"
            result = agent.run(channel_id=channel, topic=topic)
            print(f"\nResult: {result.status}")
            print(f"Progress: {result.progress_percent:.0f}%")

    else:
        print("Usage:")
        print("  python -m src.agents.workflow_agent list")
        print("  python -m src.agents.workflow_agent status <workflow_id>")
        print("  python -m src.agents.workflow_agent run [channel] [topic]")
