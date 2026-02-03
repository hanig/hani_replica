"""Tests for agent executor."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from src.bot.executor import (
    AgentExecutor,
    ToolExecutor,
    ExecutionResult,
    StreamEvent,
    StreamEventType,
    MAX_ITERATIONS,
    SYSTEM_PROMPT,
)
from src.bot.tools import ToolResult
from src.bot.conversation import ConversationContext


class TestToolExecutor:
    """Tests for ToolExecutor class."""

    @pytest.fixture
    def executor(self):
        """Create a ToolExecutor instance."""
        return ToolExecutor()

    def test_execute_unknown_tool(self, executor):
        """Test executing unknown tool returns error."""
        result = executor.execute("UnknownTool", {})
        assert result.success is False
        assert "Unknown tool" in result.error

    @patch("src.bot.executor.ToolExecutor.semantic_indexer", new_callable=MagicMock)
    def test_execute_semantic_search(self, mock_indexer, executor):
        """Test executing semantic search tool."""
        mock_indexer.search.return_value = [
            {"id": "1", "text": "result 1"},
            {"id": "2", "text": "result 2"},
        ]

        result = executor.execute("SemanticSearchTool", {
            "query": "test query",
            "max_results": 5,
        })

        assert result.success is True
        assert result.data["query"] == "test query"
        assert result.data["result_count"] == 2

    @patch("src.bot.executor.ToolExecutor.multi_google", new_callable=MagicMock)
    def test_execute_search_emails(self, mock_google, executor):
        """Test executing email search tool."""
        mock_google.search_mail_tiered.return_value = [
            {"id": "email1", "subject": "Test Email"}
        ]

        result = executor.execute("SearchEmailsTool", {
            "query": "from:test@example.com",
        })

        assert result.success is True
        assert result.data["result_count"] == 1

    @patch("src.bot.executor.ToolExecutor.multi_google", new_callable=MagicMock)
    def test_execute_search_drive(self, mock_google, executor):
        """Test executing Drive search tool."""
        mock_google.search_drive_tiered.return_value = [
            {"id": "file1", "name": "test.pdf"}
        ]

        result = executor.execute("SearchDriveTool", {"query": "project plan"})

        assert result.success is True
        assert result.data["result_count"] == 1

    @patch("src.bot.executor.ToolExecutor.multi_google", new_callable=MagicMock)
    def test_execute_get_calendar_events(self, mock_google, executor):
        """Test executing calendar events tool."""
        mock_google.get_all_calendars_for_date.return_value = [
            {"id": "event1", "summary": "Team Meeting"}
        ]

        result = executor.execute("GetCalendarEventsTool", {"date": "today"})

        assert result.success is True
        assert result.data["event_count"] == 1

    @patch("src.bot.executor.ToolExecutor.multi_google", new_callable=MagicMock)
    def test_execute_check_availability(self, mock_google, executor):
        """Test executing availability check tool."""
        mock_google.check_availability.return_value = [
            {"start": "09:00", "end": "10:00"},
            {"start": "14:00", "end": "16:00"},
        ]

        result = executor.execute("CheckAvailabilityTool", {
            "date": "tomorrow",
            "duration_minutes": 60,
        })

        assert result.success is True
        assert result.data["free_slot_count"] == 2

    @patch("src.bot.executor.ToolExecutor.multi_google", new_callable=MagicMock)
    def test_execute_get_unread_counts(self, mock_google, executor):
        """Test executing unread counts tool."""
        mock_google.get_unread_counts.return_value = {
            "arc": 5,
            "personal": 10,
        }

        result = executor.execute("GetUnreadCountsTool", {})

        assert result.success is True
        assert result.data["total_unread"] == 15
        assert result.data["by_account"]["arc"] == 5

    @patch("src.bot.executor.ToolExecutor.multi_google", new_callable=MagicMock)
    def test_execute_create_email_draft(self, mock_google, executor):
        """Test executing create email draft tool."""
        mock_google.create_draft.return_value = {"id": "draft123"}

        result = executor.execute("CreateEmailDraftTool", {
            "to": "test@example.com",
            "subject": "Test Subject",
            "body": "Test body",
            "account": "arc",
        })

        assert result.success is True
        assert result.data["draft_id"] == "draft123"

    @patch("src.bot.executor.ToolExecutor.github_client", new_callable=MagicMock)
    def test_execute_get_github_prs(self, mock_github, executor):
        """Test executing GitHub PRs tool."""
        mock_github.get_my_prs.return_value = [
            {"id": 1, "title": "Feature PR"}
        ]

        result = executor.execute("GetGitHubPRsTool", {"state": "open"})

        assert result.success is True
        assert result.data["pr_count"] == 1

    @patch("src.bot.executor.ToolExecutor.github_client", new_callable=MagicMock)
    def test_execute_get_github_issues(self, mock_github, executor):
        """Test executing GitHub issues tool."""
        mock_github.get_my_issues.return_value = [
            {"id": 1, "title": "Bug Report"}
        ]

        result = executor.execute("GetGitHubIssuesTool", {})

        assert result.success is True
        assert result.data["issue_count"] == 1

    @patch("src.bot.executor.ToolExecutor.github_client", new_callable=MagicMock)
    def test_execute_search_github_code(self, mock_github, executor):
        """Test executing GitHub code search tool."""
        mock_github.search_code.return_value = [
            {"path": "src/main.py", "repo": "owner/repo"}
        ]

        result = executor.execute("SearchGitHubCodeTool", {"query": "def main"})

        assert result.success is True
        assert result.data["result_count"] == 1

    @patch("src.bot.executor.ToolExecutor.github_client", new_callable=MagicMock)
    def test_execute_search_github_code_in_repo(self, mock_github, executor):
        """Test executing GitHub code search in specific repo."""
        mock_github.search_code_in_repo.return_value = []

        result = executor.execute("SearchGitHubCodeTool", {
            "query": "class Handler",
            "repo": "owner/repo",
        })

        mock_github.search_code_in_repo.assert_called_once()
        assert result.success is True

    @patch("src.bot.executor.ToolExecutor.github_client", new_callable=MagicMock)
    def test_execute_create_github_issue(self, mock_github, executor):
        """Test executing create GitHub issue tool."""
        mock_github.create_issue.return_value = {
            "number": 42,
            "html_url": "https://github.com/owner/repo/issues/42",
        }

        result = executor.execute("CreateGitHubIssueTool", {
            "repo": "owner/repo",
            "title": "New Issue",
            "body": "Issue description",
            "labels": ["bug"],
        })

        assert result.success is True
        assert result.data["issue_number"] == 42

    @patch("src.bot.executor.ToolExecutor.query_engine", new_callable=MagicMock)
    def test_execute_find_person(self, mock_engine, executor):
        """Test executing find person tool."""
        mock_engine.find_person.return_value = [
            {"id": "p1", "name": "John Doe", "email": "john@example.com"}
        ]

        result = executor.execute("FindPersonTool", {"query": "John"})

        assert result.success is True
        assert result.data["result_count"] == 1

    @patch("src.bot.executor.ToolExecutor.query_engine", new_callable=MagicMock)
    def test_execute_get_person_activity(self, mock_engine, executor):
        """Test executing get person activity tool."""
        mock_engine.get_person_activity.return_value = [
            {"type": "email", "title": "Meeting notes"}
        ]

        result = executor.execute("GetPersonActivityTool", {"person_id": "p1"})

        assert result.success is True
        assert result.data["activity_count"] == 1

    def test_execute_respond_to_user(self, executor):
        """Test executing respond to user tool."""
        result = executor.execute("RespondToUserTool", {
            "message": "Hello! How can I help?"
        })

        assert result.success is True
        assert result.data["message"] == "Hello! How can I help?"

    @patch("src.bot.executor.ToolExecutor.multi_google", new_callable=MagicMock)
    @patch("src.bot.executor.ToolExecutor.github_client", new_callable=MagicMock)
    def test_execute_get_daily_briefing(self, mock_github, mock_google, executor):
        """Test executing daily briefing tool."""
        mock_google.get_all_calendars_today.return_value = [{"summary": "Meeting"}]
        mock_google.get_unread_counts.return_value = {"arc": 5}
        mock_github.get_my_prs.return_value = [{"title": "PR"}]
        mock_github.get_my_issues.return_value = []

        result = executor.execute("GetDailyBriefingTool", {})

        assert result.success is True
        assert "date" in result.data
        assert len(result.data["events"]) == 1

    def test_execute_handles_exception(self, executor):
        """Test that execute handles exceptions gracefully."""
        with patch.object(executor, "_execute_semantic_search", side_effect=Exception("API Error")):
            result = executor.execute("SemanticSearchTool", {"query": "test"})
            assert result.success is False
            assert "API Error" in result.error


class TestExecutionResult:
    """Tests for ExecutionResult class."""

    def test_default_values(self):
        """Test ExecutionResult default values."""
        result = ExecutionResult(response="Test response")
        assert result.response == "Test response"
        assert result.tool_calls == []
        assert result.iterations == 0
        assert result.success is True
        assert result.error is None

    def test_with_tool_calls(self):
        """Test ExecutionResult with tool calls."""
        result = ExecutionResult(
            response="Done",
            tool_calls=[{"tool": "search", "input": {"query": "test"}}],
            iterations=2,
        )
        assert len(result.tool_calls) == 1
        assert result.iterations == 2

    def test_with_error(self):
        """Test ExecutionResult with error."""
        result = ExecutionResult(
            response="Error occurred",
            success=False,
            error="Connection failed",
        )
        assert result.success is False
        assert result.error == "Connection failed"


class TestAgentExecutor:
    """Tests for AgentExecutor class."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock conversation context."""
        return ConversationContext(
            user_id="U123",
            channel_id="C456",
            thread_ts="123.456",
        )

    def test_init_without_api_key_raises_error(self):
        """Test initialization without API key raises error."""
        with patch("src.bot.executor.ANTHROPIC_API_KEY", None):
            with pytest.raises(ValueError, match="API key is required"):
                AgentExecutor(api_key=None)

    @patch("src.bot.executor.Anthropic")
    def test_init_with_api_key(self, mock_anthropic):
        """Test initialization with API key."""
        executor = AgentExecutor(api_key="test-key")
        assert executor.api_key == "test-key"
        mock_anthropic.assert_called_once_with(api_key="test-key")

    @patch("src.bot.executor.Anthropic")
    def test_build_messages_empty_history(self, mock_anthropic, mock_context):
        """Test message building with empty history."""
        executor = AgentExecutor(api_key="test-key")
        messages = executor._build_messages(mock_context, "Hello")

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"

    @patch("src.bot.executor.Anthropic")
    def test_build_messages_with_history(self, mock_anthropic, mock_context):
        """Test message building includes history."""
        mock_context.add_message("user", "Previous message")
        mock_context.add_message("assistant", "Previous response")

        executor = AgentExecutor(api_key="test-key")
        messages = executor._build_messages(mock_context, "New message")

        assert len(messages) == 3
        assert messages[-1]["content"] == "New message"

    @patch("src.bot.executor.Anthropic")
    def test_run_with_direct_response(self, mock_anthropic, mock_context):
        """Test run when Claude responds directly without tools."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        # Create mock response with end_turn
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_content = MagicMock()
        mock_content.type = "text"
        mock_content.text = "Hello! How can I help you today?"
        mock_response.content = [mock_content]

        mock_client.messages.create.return_value = mock_response

        executor = AgentExecutor(api_key="test-key")
        result = executor.run("Hi", mock_context)

        assert result.success is True
        assert result.response == "Hello! How can I help you today?"
        assert result.iterations == 1

    @patch("src.bot.executor.Anthropic")
    def test_run_with_respond_to_user_tool(self, mock_anthropic, mock_context):
        """Test run when Claude uses RespondToUserTool."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        # Create mock response with tool_use for RespondToUserTool
        mock_response = MagicMock()
        mock_response.stop_reason = "tool_use"
        mock_tool_use = MagicMock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "RespondToUserTool"
        mock_tool_use.input = {"message": "Hello from tool!"}
        mock_tool_use.id = "tool_123"
        mock_response.content = [mock_tool_use]

        mock_client.messages.create.return_value = mock_response

        executor = AgentExecutor(api_key="test-key")
        result = executor.run("Hi", mock_context)

        assert result.success is True
        assert result.response == "Hello from tool!"

    @patch("src.bot.executor.ToolExecutor")
    @patch("src.bot.executor.Anthropic")
    def test_run_with_tool_use_and_follow_up(
        self, mock_anthropic, mock_tool_executor, mock_context
    ):
        """Test run with tool use followed by final response."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        # First response: tool_use for search
        mock_tool_use = MagicMock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "SemanticSearchTool"
        mock_tool_use.input = {"query": "test"}
        mock_tool_use.id = "tool_123"

        mock_response1 = MagicMock()
        mock_response1.stop_reason = "tool_use"
        mock_response1.content = [mock_tool_use]

        # Second response: final text
        mock_text = MagicMock()
        mock_text.type = "text"
        mock_text.text = "Found 2 results."

        mock_response2 = MagicMock()
        mock_response2.stop_reason = "end_turn"
        mock_response2.content = [mock_text]

        mock_client.messages.create.side_effect = [mock_response1, mock_response2]

        # Mock tool executor
        mock_executor_instance = MagicMock()
        mock_executor_instance.execute.return_value = ToolResult(
            data={"results": [{"id": 1}, {"id": 2}]}
        )
        mock_tool_executor.return_value = mock_executor_instance

        executor = AgentExecutor(api_key="test-key")
        result = executor.run("Search for test", mock_context)

        assert result.success is True
        assert result.response == "Found 2 results."
        assert result.iterations == 2
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["tool"] == "SemanticSearchTool"

    @patch("src.bot.executor.Anthropic")
    def test_run_max_iterations(self, mock_anthropic, mock_context):
        """Test run stops at max iterations."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        # Always return tool_use to force max iterations
        mock_tool_use = MagicMock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "SemanticSearchTool"
        mock_tool_use.input = {"query": "test"}
        mock_tool_use.id = "tool_123"

        mock_response = MagicMock()
        mock_response.stop_reason = "tool_use"
        mock_response.content = [mock_tool_use]

        mock_client.messages.create.return_value = mock_response

        executor = AgentExecutor(api_key="test-key")

        with patch.object(executor._tool_executor, "execute") as mock_execute:
            mock_execute.return_value = ToolResult(data={"results": []})
            result = executor.run("Endless search", mock_context, max_iterations=3)

        assert result.success is False
        assert "Max iterations" in result.error
        assert result.iterations == 3

    @patch("src.bot.executor.Anthropic")
    def test_run_handles_api_error(self, mock_anthropic, mock_context):
        """Test run handles API errors gracefully."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API Error")

        executor = AgentExecutor(api_key="test-key")
        result = executor.run("Test", mock_context)

        assert result.success is False
        assert "API Error" in result.error

    @patch("src.bot.executor.Anthropic")
    def test_run_with_user_memory(self, mock_anthropic, mock_context):
        """Test run injects user memory context."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_text = MagicMock()
        mock_text.type = "text"
        mock_text.text = "Response with context"
        mock_response.content = [mock_text]

        mock_client.messages.create.return_value = mock_response

        # Mock user memory
        mock_memory = MagicMock()
        mock_memory.get_context_summary.return_value = "User prefers Arc email"

        executor = AgentExecutor(api_key="test-key", user_memory=mock_memory)
        result = executor.run("Check my email", mock_context)

        # Verify memory was queried
        mock_memory.get_context_summary.assert_called_once_with(mock_context.user_id)

        # Verify system prompt includes user context
        call_args = mock_client.messages.create.call_args
        system_prompt = call_args.kwargs["system"]
        assert "User prefers Arc email" in system_prompt

    @patch("src.bot.executor.Anthropic")
    def test_extract_text_response(self, mock_anthropic):
        """Test extracting text from response."""
        executor = AgentExecutor(api_key="test-key")

        mock_response = MagicMock()
        mock_text = MagicMock()
        mock_text.type = "text"
        mock_text.text = "Extracted text"
        mock_response.content = [mock_text]

        result = executor._extract_text_response(mock_response)
        assert result == "Extracted text"

    @patch("src.bot.executor.Anthropic")
    def test_extract_text_response_no_text(self, mock_anthropic):
        """Test extracting text when no text content."""
        executor = AgentExecutor(api_key="test-key")

        mock_response = MagicMock()
        mock_tool = MagicMock()
        mock_tool.type = "tool_use"
        mock_response.content = [mock_tool]

        result = executor._extract_text_response(mock_response)
        assert result == ""


class TestSystemPrompt:
    """Tests for system prompt configuration."""

    def test_system_prompt_contains_guidelines(self):
        """Test system prompt contains usage guidelines."""
        assert "conversational" in SYSTEM_PROMPT.lower()
        assert "tool" in SYSTEM_PROMPT.lower()
        assert "RespondToUserTool" in SYSTEM_PROMPT

    def test_system_prompt_has_date_placeholder(self):
        """Test system prompt has date placeholder."""
        assert "{current_date}" in SYSTEM_PROMPT

    def test_max_iterations_is_reasonable(self):
        """Test max iterations is set to a reasonable value."""
        assert MAX_ITERATIONS >= 5
        assert MAX_ITERATIONS <= 20


class TestStreamEvent:
    """Tests for StreamEvent class."""

    def test_text_delta_event(self):
        """Test creating text delta event."""
        event = StreamEvent(
            event_type=StreamEventType.TEXT_DELTA,
            data="Hello ",
            iteration=1,
        )
        assert event.event_type == StreamEventType.TEXT_DELTA
        assert event.data == "Hello "
        assert event.iteration == 1

    def test_tool_start_event(self):
        """Test creating tool start event."""
        event = StreamEvent(
            event_type=StreamEventType.TOOL_START,
            tool_name="SemanticSearchTool",
            iteration=1,
        )
        assert event.event_type == StreamEventType.TOOL_START
        assert event.tool_name == "SemanticSearchTool"

    def test_tool_done_event(self):
        """Test creating tool done event."""
        event = StreamEvent(
            event_type=StreamEventType.TOOL_DONE,
            tool_name="SemanticSearchTool",
            tool_input={"query": "test"},
            tool_result="Found 5 results",
            iteration=1,
        )
        assert event.event_type == StreamEventType.TOOL_DONE
        assert event.tool_input == {"query": "test"}
        assert event.tool_result == "Found 5 results"

    def test_error_event(self):
        """Test creating error event."""
        event = StreamEvent(
            event_type=StreamEventType.ERROR,
            error="API connection failed",
            iteration=1,
        )
        assert event.event_type == StreamEventType.ERROR
        assert event.error == "API connection failed"

    def test_done_event(self):
        """Test creating done event."""
        event = StreamEvent(
            event_type=StreamEventType.DONE,
            data="Final response text",
            iteration=2,
        )
        assert event.event_type == StreamEventType.DONE
        assert event.data == "Final response text"


class TestStreamEventType:
    """Tests for StreamEventType enum."""

    def test_event_types_exist(self):
        """Test all expected event types exist."""
        assert StreamEventType.TEXT_DELTA == "text_delta"
        assert StreamEventType.TEXT_DONE == "text_done"
        assert StreamEventType.TOOL_START == "tool_start"
        assert StreamEventType.TOOL_DONE == "tool_done"
        assert StreamEventType.THINKING == "thinking"
        assert StreamEventType.ERROR == "error"
        assert StreamEventType.DONE == "done"


def collect_generator_events(gen):
    """Helper to collect events from a generator and capture its return value.

    Returns:
        Tuple of (list of events, return value)
    """
    events = []
    result = None
    try:
        while True:
            event = next(gen)
            events.append(event)
    except StopIteration as e:
        result = e.value
    return events, result


class TestAgentExecutorStreaming:
    """Tests for AgentExecutor streaming functionality."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock conversation context."""
        return ConversationContext(
            user_id="U123",
            channel_id="C456",
            thread_ts="123.456",
        )

    @patch("src.bot.executor.Anthropic")
    def test_run_streaming_direct_response(self, mock_anthropic, mock_context):
        """Test streaming with direct text response (no tools)."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        # Create a mock stream context manager
        mock_stream = MagicMock()

        # Mock streaming events
        mock_text_start = MagicMock()
        mock_text_start.type = "content_block_start"
        mock_text_start.content_block = MagicMock()
        mock_text_start.content_block.type = "text"

        mock_text_delta = MagicMock()
        mock_text_delta.type = "content_block_delta"
        mock_text_delta.delta = MagicMock()
        mock_text_delta.delta.type = "text_delta"
        mock_text_delta.delta.text = "Hello!"

        mock_text_stop = MagicMock()
        mock_text_stop.type = "content_block_stop"

        mock_stream.__iter__ = MagicMock(
            return_value=iter([mock_text_start, mock_text_delta, mock_text_stop])
        )

        # Mock final message
        mock_final = MagicMock()
        mock_final.stop_reason = "end_turn"
        mock_final_text = MagicMock()
        mock_final_text.type = "text"
        mock_final_text.text = "Hello!"
        mock_final.content = [mock_final_text]
        mock_stream.get_final_message.return_value = mock_final

        mock_stream_cm = MagicMock()
        mock_stream_cm.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream_cm.__exit__ = MagicMock(return_value=None)

        mock_client.messages.stream.return_value = mock_stream_cm

        executor = AgentExecutor(api_key="test-key")

        # Collect events from generator using helper
        gen = executor.run_streaming("Hi", mock_context)
        events, result = collect_generator_events(gen)

        # Verify events were generated
        assert len(events) > 0

        # Verify result
        assert result is not None
        assert result.response == "Hello!"

    @patch("src.bot.executor.Anthropic")
    def test_run_streaming_with_respond_tool(self, mock_anthropic, mock_context):
        """Test streaming when RespondToUserTool is used."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        # Create mock stream
        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter([]))

        # Mock final message with RespondToUserTool
        mock_final = MagicMock()
        mock_final.stop_reason = "tool_use"
        mock_tool_use = MagicMock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "RespondToUserTool"
        mock_tool_use.input = {"message": "Hello from tool!"}
        mock_tool_use.id = "tool_123"
        mock_final.content = [mock_tool_use]
        mock_stream.get_final_message.return_value = mock_final

        mock_stream_cm = MagicMock()
        mock_stream_cm.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream_cm.__exit__ = MagicMock(return_value=None)

        mock_client.messages.stream.return_value = mock_stream_cm

        executor = AgentExecutor(api_key="test-key")

        # Collect events using helper
        gen = executor.run_streaming("Hi", mock_context)
        events, result = collect_generator_events(gen)

        # Verify result
        assert result is not None
        assert result.response == "Hello from tool!"

    @patch("src.bot.executor.Anthropic")
    def test_run_streaming_handles_error(self, mock_anthropic, mock_context):
        """Test streaming handles errors gracefully."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        # Make stream raise an error
        mock_client.messages.stream.side_effect = Exception("Connection error")

        executor = AgentExecutor(api_key="test-key")

        # Collect events using helper
        gen = executor.run_streaming("Test", mock_context)
        events, result = collect_generator_events(gen)

        # Verify error event was emitted
        error_events = [e for e in events if e.event_type == StreamEventType.ERROR]
        assert len(error_events) > 0
        assert "Connection error" in error_events[0].error

        # Verify result indicates failure
        assert result is not None
        assert result.success is False

    @patch("src.bot.executor.ToolExecutor")
    @patch("src.bot.executor.Anthropic")
    def test_run_streaming_with_tool_use(
        self, mock_anthropic, mock_tool_executor, mock_context
    ):
        """Test streaming with tool use and follow-up response."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        # First iteration: tool use
        mock_stream1 = MagicMock()
        mock_stream1.__iter__ = MagicMock(return_value=iter([]))

        mock_final1 = MagicMock()
        mock_final1.stop_reason = "tool_use"
        mock_tool_use = MagicMock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "SemanticSearchTool"
        mock_tool_use.input = {"query": "test"}
        mock_tool_use.id = "tool_123"
        mock_final1.content = [mock_tool_use]
        mock_stream1.get_final_message.return_value = mock_final1

        mock_stream_cm1 = MagicMock()
        mock_stream_cm1.__enter__ = MagicMock(return_value=mock_stream1)
        mock_stream_cm1.__exit__ = MagicMock(return_value=None)

        # Second iteration: final response
        mock_stream2 = MagicMock()
        mock_stream2.__iter__ = MagicMock(return_value=iter([]))

        mock_final2 = MagicMock()
        mock_final2.stop_reason = "end_turn"
        mock_text = MagicMock()
        mock_text.type = "text"
        mock_text.text = "Found results!"
        mock_final2.content = [mock_text]
        mock_stream2.get_final_message.return_value = mock_final2

        mock_stream_cm2 = MagicMock()
        mock_stream_cm2.__enter__ = MagicMock(return_value=mock_stream2)
        mock_stream_cm2.__exit__ = MagicMock(return_value=None)

        mock_client.messages.stream.side_effect = [mock_stream_cm1, mock_stream_cm2]

        # Mock tool executor
        mock_executor_instance = MagicMock()
        mock_executor_instance.execute.return_value = ToolResult(
            data={"results": ["result1", "result2"]}
        )
        mock_tool_executor.return_value = mock_executor_instance

        executor = AgentExecutor(api_key="test-key")

        # Collect events using helper
        gen = executor.run_streaming("Search for test", mock_context)
        events, result = collect_generator_events(gen)

        # Verify tool events were emitted
        # Note: TOOL_START is emitted during streaming (content_block_start),
        # THINKING is emitted when processing tool calls from final message
        thinking_events = [e for e in events if e.event_type == StreamEventType.THINKING]
        tool_done_events = [e for e in events if e.event_type == StreamEventType.TOOL_DONE]
        assert len(thinking_events) > 0, f"Expected THINKING events, got: {[e.event_type for e in events]}"
        assert len(tool_done_events) > 0, f"Expected TOOL_DONE events, got: {[e.event_type for e in events]}"

        # Verify result
        assert result is not None
        assert result.response == "Found results!"
        assert len(result.tool_calls) == 1
