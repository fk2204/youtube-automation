"""
Agent Orchestration System for YouTube Automation
Manages agent communication, pipelines, and workflows.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from loguru import logger
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

from src.agents.base_agent import (
    BaseAgent, AgentMessage, AgentResult, AgentError
)


@dataclass
class PipelineStep:
    """A step in the agent pipeline."""
    agent_name: str
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    required: bool = True
    timeout_seconds: int = 300


@dataclass
class WorkflowState:
    """State tracking for workflows."""
    workflow_id: str
    status: str = "pending"  # pending, running, completed, failed
    current_step: int = 0
    steps_completed: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def progress_percent(self) -> float:
        total = len(self.steps_completed) + (1 if self.status == "running" else 0)
        return (len(self.steps_completed) / total * 100) if total > 0 else 0


class AgentOrchestrator:
    """Orchestrates communication and workflows between agents."""

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.workflows: Dict[str, WorkflowState] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        logger.info("AgentOrchestrator initialized")

    def register_agent(self, agent: BaseAgent):
        """Register an agent with the orchestrator."""
        self.agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get a registered agent by name."""
        return self.agents.get(name)

    def send_message(self, message: AgentMessage) -> AgentMessage:
        """Send a message to an agent and get response."""
        if message.recipient == "broadcast":
            return self._broadcast(message)

        agent = self.agents.get(message.recipient)
        if not agent:
            return AgentMessage(
                sender="Orchestrator",
                recipient=message.sender,
                message_type="error",
                payload={"error": f"Agent {message.recipient} not found"},
                correlation_id=message.message_id
            )

        return agent.handle_message(message)

    def _broadcast(self, message: AgentMessage) -> AgentMessage:
        """Broadcast message to all agents."""
        responses = []
        for name, agent in self.agents.items():
            if name != message.sender:
                response = agent.handle_message(message)
                responses.append(response)
        return AgentMessage(
            sender="Orchestrator",
            recipient=message.sender,
            message_type="response",
            payload={"responses": [r.to_dict() for r in responses]},
            correlation_id=message.message_id
        )

    def run_pipeline(self, pipeline: List[PipelineStep], initial_data: Dict = None) -> WorkflowState:
        """Run a sequential pipeline of agent operations."""
        from uuid import uuid4
        workflow_id = str(uuid4())[:8]
        state = WorkflowState(workflow_id=workflow_id)
        state.status = "running"
        state.started_at = datetime.now()
        self.workflows[workflow_id] = state

        data = initial_data or {}

        for i, step in enumerate(pipeline):
            state.current_step = i
            logger.info(f"Pipeline step {i+1}/{len(pipeline)}: {step.agent_name}.{step.action}")

            agent = self.agents.get(step.agent_name)
            if not agent:
                if step.required:
                    state.errors.append(f"Agent {step.agent_name} not found")
                    state.status = "failed"
                    break
                continue

            try:
                # Merge step params with accumulated data
                params = {**data, **step.params}
                result = agent.run(**params)

                if result.success:
                    state.steps_completed.append(f"{step.agent_name}.{step.action}")
                    # Accumulate data for next step
                    if isinstance(result.data, dict):
                        data.update(result.data)
                    else:
                        data[step.action] = result.data
                    state.results[step.agent_name] = result.data
                else:
                    if step.required:
                        state.errors.append(f"{step.agent_name} failed: {result.error}")
                        state.status = "failed"
                        break
            except Exception as e:
                logger.error(f"Pipeline error at {step.agent_name}: {e}")
                if step.required:
                    state.errors.append(str(e))
                    state.status = "failed"
                    break

        if state.status != "failed":
            state.status = "completed"
        state.completed_at = datetime.now()

        return state

    async def run_parallel(self, agents_tasks: List[tuple]) -> Dict[str, AgentResult]:
        """Run multiple agent tasks in parallel."""
        async def run_agent_task(agent_name: str, params: Dict) -> tuple:
            agent = self.agents.get(agent_name)
            if not agent:
                return agent_name, AgentResult(success=False, error=f"Agent not found")

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                lambda: agent.run(**params)
            )
            return agent_name, result

        tasks = [run_agent_task(name, params) for name, params in agents_tasks]
        results = await asyncio.gather(*tasks)
        return dict(results)

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all registered agents."""
        return {
            "total_agents": len(self.agents),
            "agents": list(self.agents.keys()),
            "active_workflows": len([w for w in self.workflows.values() if w.status == "running"]),
            "completed_workflows": len([w for w in self.workflows.values() if w.status == "completed"]),
            "failed_workflows": len([w for w in self.workflows.values() if w.status == "failed"])
        }


# Pre-defined pipelines
class VideoPipelines:
    """Pre-defined video production pipelines."""

    @staticmethod
    def full_video_pipeline(niche: str, channel: str) -> List[PipelineStep]:
        """Full video creation pipeline."""
        return [
            PipelineStep("ResearchAgent", "find_topics", {"niche": niche, "count": 5}),
            PipelineStep("ScriptAgent", "generate", {"niche": niche}),
            PipelineStep("SEOAgent", "optimize", {}),
            PipelineStep("ProductionAgent", "produce", {"channel": channel}),
            PipelineStep("ThumbnailAgent", "generate", {}),
            PipelineStep("ValidatorAgent", "validate", {}),
            PipelineStep("UploadAgent", "upload", {"channel": channel}),
        ]

    @staticmethod
    def short_video_pipeline(niche: str, channel: str) -> List[PipelineStep]:
        """YouTube Shorts pipeline."""
        return [
            PipelineStep("ResearchAgent", "find_topics", {"niche": niche, "type": "short"}),
            PipelineStep("ScriptAgent", "generate_short", {"niche": niche}),
            PipelineStep("ProductionAgent", "produce_short", {"channel": channel}),
            PipelineStep("ValidatorAgent", "validate", {}),
            PipelineStep("UploadAgent", "upload_short", {"channel": channel}),
        ]


# Singleton orchestrator
_orchestrator = None


def get_orchestrator() -> AgentOrchestrator:
    """Get or create the global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator
