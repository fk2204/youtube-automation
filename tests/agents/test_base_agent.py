"""Comprehensive tests for src/agents/base_agent.py"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch
from src.agents.base_agent import (
    BaseAgent, AgentResult, AgentMessage, AgentError,
    TokenBudgetExceeded, APIRateLimitError, QualityError, SafetyError,
    handle_agent_errors,
)


@pytest.fixture
def concrete_agent():
    """Test implementation of BaseAgent."""
    class TestAgent(BaseAgent):
        def run(self):
            return AgentResult(
                success=True, operation="test",
                data={"test": "data"}, agent_name=self.name
            )
        def handle_message(self, message):
            return message
    return TestAgent(name="TestAgent")


class TestAgentResult:
    """Tests for AgentResult dataclass."""

    def test_creation_minimal(self):
        result = AgentResult(success=True, operation="test")
        assert result.success is True
        assert result.operation == "test"

    def test_creation_full(self):
        result = AgentResult(
            success=True, operation="test_op", data={"key": "value"},
            agent_name="TestAgent", tokens_used=100, cost=0.01,
            duration_seconds=1.5
        )
        assert result.agent_name == "TestAgent"

    def test_to_dict(self):
        result = AgentResult(success=True, operation="test", agent_name="TestAgent")
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["success"] is True


class TestAgentMessage:
    """Tests for AgentMessage dataclass."""

    def test_creation(self):
        msg = AgentMessage(
            sender="Agent1", recipient="Agent2", message_type="request"
        )
        assert msg.sender == "Agent1"
        assert msg.message_type == "request"


class TestBaseAgent:
    """Tests for BaseAgent abstract class."""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseAgent()

    def test_concrete_agent_creation(self, concrete_agent):
        assert concrete_agent.name == "TestAgent"
        assert concrete_agent.provider == "groq"

    def test_run_method(self, concrete_agent):
        result = concrete_agent.run()
        assert isinstance(result, AgentResult)
        assert result.success is True

    def test_handle_message(self, concrete_agent):
        msg = AgentMessage(sender="A", recipient="B", message_type="request")
        result_msg = concrete_agent.handle_message(msg)
        assert result_msg == msg

    @pytest.mark.parametrize("success", [True, False])
    def test_result_success_flag(self, success):
        result = AgentResult(success=success, operation="test")
        assert result.success == success
