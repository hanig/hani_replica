"""Tests for multi-agent architecture."""

from unittest.mock import MagicMock, patch

import pytest

from src.bot.agents.base import BaseAgent, AgentType, AgentResult, AgentStreamEvent
from src.bot.agents.calendar_agent import CalendarAgent
from src.bot.agents.email_agent import EmailAgent
from src.bot.agents.github_agent import GitHubAgent
from src.bot.agents.research_agent import ResearchAgent
from src.bot.agents.orchestrator import Orchestrator, TaskPlan
from src.bot.conversation import ConversationContext


# Patch the base module's ANTHROPIC_API_KEY for all tests
@pytest.fixture(autouse=True)
def mock_api_key():
    """Mock API key for all tests."""
    with patch("src.bot.agents.base.ANTHROPIC_API_KEY", "test-api-key"):
        yield


class TestAgentResult:
    """Tests for AgentResult class."""

    def test_default_values(self):
        """Test AgentResult default values."""
        result = AgentResult(
            response="Test response",
            agent_type=AgentType.CALENDAR,
        )
        assert result.response == "Test response"
        assert result.agent_type == AgentType.CALENDAR
        assert result.tool_calls == []
        assert result.iterations == 0
        assert result.success is True
        assert result.error is None

    def test_to_dict(self):
        """Test AgentResult to_dict conversion."""
        result = AgentResult(
            response="Response",
            agent_type=AgentType.EMAIL,
            tool_calls=[{"tool": "search", "input": {}}],
            iterations=2,
            metadata={"key": "value"},
        )
        d = result.to_dict()
        assert d["response"] == "Response"
        assert d["agent_type"] == "email"
        assert len(d["tool_calls"]) == 1
        assert d["iterations"] == 2
        assert d["metadata"]["key"] == "value"

    def test_with_error(self):
        """Test AgentResult with error."""
        result = AgentResult(
            response="Error occurred",
            agent_type=AgentType.RESEARCH,
            success=False,
            error="Connection failed",
        )
        assert result.success is False
        assert result.error == "Connection failed"


class TestAgentStreamEvent:
    """Tests for AgentStreamEvent class."""

    def test_text_delta_event(self):
        """Test text delta event."""
        event = AgentStreamEvent(
            event_type="text_delta",
            data="Hello",
            agent_type=AgentType.CALENDAR,
            iteration=1,
        )
        assert event.event_type == "text_delta"
        assert event.data == "Hello"
        assert event.agent_type == AgentType.CALENDAR

    def test_tool_event(self):
        """Test tool event."""
        event = AgentStreamEvent(
            event_type="tool_done",
            tool_name="SearchEmailsTool",
            tool_input={"query": "test"},
            tool_result="Found 5 results",
            iteration=1,
        )
        assert event.tool_name == "SearchEmailsTool"
        assert event.tool_input == {"query": "test"}
        assert event.tool_result == "Found 5 results"


class TestAgentType:
    """Tests for AgentType enum."""

    def test_agent_types(self):
        """Test all agent types exist."""
        assert AgentType.CALENDAR == "calendar"
        assert AgentType.EMAIL == "email"
        assert AgentType.GITHUB == "github"
        assert AgentType.RESEARCH == "research"
        assert AgentType.ORCHESTRATOR == "orchestrator"


class TestCalendarAgent:
    """Tests for CalendarAgent."""

    @pytest.fixture
    def context(self):
        """Create a mock context."""
        return ConversationContext(
            user_id="U123",
            channel_id="C456",
            thread_ts="123.456",
        )

    def test_agent_type(self):
        """Test agent type is correct."""
        agent = CalendarAgent(api_key="test-key")
        assert agent.AGENT_TYPE == AgentType.CALENDAR

    def test_tool_names(self):
        """Test calendar-specific tools."""
        agent = CalendarAgent(api_key="test-key")
        tools = agent.tool_names
        assert "GetCalendarEventsTool" in tools
        assert "CheckAvailabilityTool" in tools
        assert "RespondToUserTool" in tools
        assert "SearchEmailsTool" not in tools

    def test_can_handle_calendar_message(self, context):
        """Test can_handle for calendar messages."""
        agent = CalendarAgent(api_key="test-key")

        # Calendar-related messages should score high
        assert agent.can_handle("what's on my calendar today", context) > 0.3
        assert agent.can_handle("check my schedule for tomorrow", context) > 0.3
        assert agent.can_handle("when is my next meeting", context) > 0.3

        # Non-calendar messages should score low
        assert agent.can_handle("find emails from John", context) < 0.3

    def test_system_prompt_contains_date_placeholder(self):
        """Test system prompt has date placeholder."""
        agent = CalendarAgent(api_key="test-key")
        assert "{current_date}" in agent.system_prompt

    def test_description(self):
        """Test agent description."""
        agent = CalendarAgent(api_key="test-key")
        assert "calendar" in agent.description.lower()


class TestEmailAgent:
    """Tests for EmailAgent."""

    @pytest.fixture
    def context(self):
        """Create a mock context."""
        return ConversationContext(
            user_id="U123",
            channel_id="C456",
            thread_ts="123.456",
        )

    def test_agent_type(self):
        """Test agent type is correct."""
        agent = EmailAgent(api_key="test-key")
        assert agent.AGENT_TYPE == AgentType.EMAIL

    def test_tool_names(self):
        """Test email-specific tools."""
        agent = EmailAgent(api_key="test-key")
        tools = agent.tool_names
        assert "SearchEmailsTool" in tools
        assert "GetUnreadCountsTool" in tools
        assert "CreateEmailDraftTool" in tools
        assert "GetCalendarEventsTool" not in tools

    def test_can_handle_email_message(self, context):
        """Test can_handle for email messages."""
        agent = EmailAgent(api_key="test-key")

        # Email-related messages should score high
        assert agent.can_handle("find emails from John", context) > 0.3
        assert agent.can_handle("check my inbox", context) > 0.3
        assert agent.can_handle("draft an email to jane@example.com", context) > 0.3

        # Non-email messages should score low
        assert agent.can_handle("what's on my calendar", context) < 0.3


class TestGitHubAgent:
    """Tests for GitHubAgent."""

    @pytest.fixture
    def context(self):
        """Create a mock context."""
        return ConversationContext(
            user_id="U123",
            channel_id="C456",
            thread_ts="123.456",
        )

    def test_agent_type(self):
        """Test agent type is correct."""
        agent = GitHubAgent(api_key="test-key")
        assert agent.AGENT_TYPE == AgentType.GITHUB

    def test_tool_names(self):
        """Test GitHub-specific tools."""
        agent = GitHubAgent(api_key="test-key")
        tools = agent.tool_names
        assert "GetGitHubPRsTool" in tools
        assert "GetGitHubIssuesTool" in tools
        assert "SearchGitHubCodeTool" in tools
        assert "CreateGitHubIssueTool" in tools
        assert "SearchEmailsTool" not in tools

    def test_can_handle_github_message(self, context):
        """Test can_handle for GitHub messages."""
        agent = GitHubAgent(api_key="test-key")

        # GitHub-related messages should score high
        assert agent.can_handle("show my open PRs", context) > 0.3
        assert agent.can_handle("list issues on the repo", context) > 0.3
        assert agent.can_handle("search code for def main", context) > 0.3

        # Non-GitHub messages should score low
        assert agent.can_handle("check my calendar", context) < 0.3


class TestResearchAgent:
    """Tests for ResearchAgent."""

    @pytest.fixture
    def context(self):
        """Create a mock context."""
        return ConversationContext(
            user_id="U123",
            channel_id="C456",
            thread_ts="123.456",
        )

    def test_agent_type(self):
        """Test agent type is correct."""
        agent = ResearchAgent(api_key="test-key")
        assert agent.AGENT_TYPE == AgentType.RESEARCH

    def test_tool_names(self):
        """Test research-specific tools."""
        agent = ResearchAgent(api_key="test-key")
        tools = agent.tool_names
        assert "SemanticSearchTool" in tools
        assert "SearchDriveTool" in tools
        assert "FindPersonTool" in tools
        assert "GetDailyBriefingTool" in tools

    def test_can_handle_research_message(self, context):
        """Test can_handle for research messages."""
        agent = ResearchAgent(api_key="test-key")

        # Research-related messages should score reasonably
        assert agent.can_handle("find documents about project X", context) > 0.1
        assert agent.can_handle("give me my daily briefing overview", context) > 0.3
        assert agent.can_handle("who is John Smith", context) > 0.1


class TestOrchestrator:
    """Tests for Orchestrator."""

    @pytest.fixture
    def context(self):
        """Create a mock context."""
        return ConversationContext(
            user_id="U123",
            channel_id="C456",
            thread_ts="123.456",
        )

    def test_init_creates_specialists(self):
        """Test orchestrator initializes specialists."""
        orchestrator = Orchestrator(api_key="test-key")

        assert AgentType.CALENDAR in orchestrator.specialists
        assert AgentType.EMAIL in orchestrator.specialists
        assert AgentType.GITHUB in orchestrator.specialists
        assert AgentType.RESEARCH in orchestrator.specialists

    def test_get_available_specialists(self):
        """Test listing available specialists."""
        orchestrator = Orchestrator(api_key="test-key")
        specialists = orchestrator.get_available_specialists()

        assert "calendar" in specialists
        assert "email" in specialists
        assert "github" in specialists
        assert "research" in specialists

    def test_is_conversational_greetings(self):
        """Test conversational detection for greetings."""
        orchestrator = Orchestrator(api_key="test-key")

        assert orchestrator._is_conversational("hi")
        assert orchestrator._is_conversational("hello")
        assert orchestrator._is_conversational("hey")
        assert orchestrator._is_conversational("good morning")
        assert orchestrator._is_conversational("thanks")
        assert orchestrator._is_conversational("how are you")

    def test_is_conversational_not_conversational(self):
        """Test conversational detection for task messages."""
        orchestrator = Orchestrator(api_key="test-key")

        assert not orchestrator._is_conversational("check my calendar")
        assert not orchestrator._is_conversational("find emails from John")
        assert not orchestrator._is_conversational("show my open PRs")

    def test_select_specialist_calendar(self, context):
        """Test specialist selection for calendar messages."""
        orchestrator = Orchestrator(api_key="test-key")

        result = orchestrator._select_specialist("what's on my calendar today", context)
        assert result == AgentType.CALENDAR

    def test_select_specialist_email(self, context):
        """Test specialist selection for email messages."""
        orchestrator = Orchestrator(api_key="test-key")

        result = orchestrator._select_specialist("find emails from John", context)
        assert result == AgentType.EMAIL

    def test_select_specialist_github(self, context):
        """Test specialist selection for GitHub messages."""
        orchestrator = Orchestrator(api_key="test-key")

        result = orchestrator._select_specialist("show my open pull requests", context)
        assert result == AgentType.GITHUB

    def test_select_specialist_conversational_returns_none(self, context):
        """Test specialist selection returns None for conversational."""
        orchestrator = Orchestrator(api_key="test-key")

        result = orchestrator._select_specialist("hi there", context)
        assert result is None

    def test_plan_task_conversational(self, context):
        """Test task planning for conversational messages."""
        orchestrator = Orchestrator(api_key="test-key")

        plan = orchestrator._plan_task("hi there", context)
        assert plan.is_conversational is True
        assert plan.needs_specialist is False

    def test_plan_task_single_domain(self, context):
        """Test task planning for single domain messages."""
        orchestrator = Orchestrator(api_key="test-key")

        plan = orchestrator._plan_task("check my calendar for today", context)
        assert plan.needs_specialist is True
        assert AgentType.CALENDAR in plan.specialist_types

    def test_can_handle_always_returns_1(self, context):
        """Test orchestrator can_handle returns 1.0 for everything."""
        orchestrator = Orchestrator(api_key="test-key")

        assert orchestrator.can_handle("anything", context) == 1.0


class TestBaseAgentRun:
    """Tests for BaseAgent.run method."""

    @pytest.fixture
    def context(self):
        """Create a mock context."""
        return ConversationContext(
            user_id="U123",
            channel_id="C456",
            thread_ts="123.456",
        )

    @patch("src.bot.agents.base.Anthropic")
    def test_run_direct_response(self, mock_anthropic, context):
        """Test run with direct text response."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        # Mock response
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_text = MagicMock()
        mock_text.type = "text"
        mock_text.text = "Here are your events for today."
        mock_response.content = [mock_text]
        mock_client.messages.create.return_value = mock_response

        agent = CalendarAgent(api_key="test-key")
        result = agent.run("what's on my calendar", context)

        assert result.success is True
        assert result.response == "Here are your events for today."
        assert result.agent_type == AgentType.CALENDAR

    @patch("src.bot.agents.base.Anthropic")
    def test_run_with_respond_tool(self, mock_anthropic, context):
        """Test run when RespondToUserTool is used."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        # Mock response with tool use
        mock_response = MagicMock()
        mock_response.stop_reason = "tool_use"
        mock_tool = MagicMock()
        mock_tool.type = "tool_use"
        mock_tool.name = "RespondToUserTool"
        mock_tool.input = {"message": "You have 3 meetings today."}
        mock_tool.id = "tool_123"
        mock_response.content = [mock_tool]
        mock_client.messages.create.return_value = mock_response

        agent = CalendarAgent(api_key="test-key")
        result = agent.run("check my calendar", context)

        assert result.success is True
        assert result.response == "You have 3 meetings today."

    @patch("src.bot.agents.base.Anthropic")
    def test_run_handles_api_error(self, mock_anthropic, context):
        """Test run handles API errors gracefully."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API Error")

        agent = CalendarAgent(api_key="test-key")
        result = agent.run("check my calendar", context)

        assert result.success is False
        assert "error" in result.response.lower()


class TestOrchestratorRun:
    """Tests for Orchestrator.run method."""

    @pytest.fixture
    def context(self):
        """Create a mock context."""
        return ConversationContext(
            user_id="U123",
            channel_id="C456",
            thread_ts="123.456",
        )

    @patch("src.bot.agents.base.Anthropic")
    def test_run_conversational(self, mock_anthropic, context):
        """Test run with conversational message."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        # Mock response
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_text = MagicMock()
        mock_text.type = "text"
        mock_text.text = "Hello! How can I help you today?"
        mock_response.content = [mock_text]
        mock_client.messages.create.return_value = mock_response

        orchestrator = Orchestrator(api_key="test-key")
        result = orchestrator.run("hi", context)

        assert result.success is True
        assert "hello" in result.response.lower() or "help" in result.response.lower()

    @patch("src.bot.agents.base.Anthropic")
    def test_run_routes_to_specialist(self, mock_anthropic, context):
        """Test run routes to specialist for domain messages."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        # Mock specialist response
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_text = MagicMock()
        mock_text.type = "text"
        mock_text.text = "Your calendar shows 3 meetings."
        mock_response.content = [mock_text]
        mock_client.messages.create.return_value = mock_response

        orchestrator = Orchestrator(api_key="test-key")
        result = orchestrator.run("check my calendar for today", context)

        assert result.success is True
        # Result should come from calendar specialist
        assert result.agent_type == AgentType.CALENDAR


class TestTaskPlan:
    """Tests for TaskPlan dataclass."""

    def test_task_plan_defaults(self):
        """Test TaskPlan default values."""
        plan = TaskPlan(needs_specialist=False)
        assert plan.needs_specialist is False
        assert plan.specialist_types == []
        assert plan.subtasks == []
        assert plan.is_conversational is False
        assert plan.reasoning == ""

    def test_task_plan_with_specialists(self):
        """Test TaskPlan with specialist types."""
        plan = TaskPlan(
            needs_specialist=True,
            specialist_types=[AgentType.CALENDAR, AgentType.EMAIL],
            reasoning="Multi-domain request",
        )
        assert len(plan.specialist_types) == 2
        assert AgentType.CALENDAR in plan.specialist_types
        assert AgentType.EMAIL in plan.specialist_types
