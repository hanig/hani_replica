"""Tests for tool definitions."""

from datetime import datetime, timedelta, timezone

import pytest

from src.bot.tools import (
    ALL_TOOLS,
    TOOL_NAME_MAP,
    ToolResult,
    get_tool_schemas,
    parse_date_reference,
    SemanticSearchTool,
    SearchEmailsTool,
    SearchDriveTool,
    GetCalendarEventsTool,
    CheckAvailabilityTool,
    GetUnreadCountsTool,
    CreateEmailDraftTool,
    GetGitHubPRsTool,
    GetGitHubIssuesTool,
    SearchGitHubCodeTool,
    CreateGitHubIssueTool,
    FindPersonTool,
    GetPersonActivityTool,
    GetDailyBriefingTool,
    RespondToUserTool,
)


class TestToolResult:
    """Tests for ToolResult class."""

    def test_success_with_string_data(self):
        """Test ToolResult with string data."""
        result = ToolResult(data="Hello, world!")
        content = result.to_content()
        assert content == "Hello, world!"

    def test_success_with_dict_data(self):
        """Test ToolResult with dict data."""
        result = ToolResult(data={"key": "value", "count": 42})
        content = result.to_content()
        assert "key" in content
        assert "value" in content
        assert "42" in content

    def test_error_result(self):
        """Test ToolResult with error."""
        result = ToolResult(success=False, error="Something went wrong")
        content = result.to_content()
        assert "Error:" in content
        assert "Something went wrong" in content

    def test_default_success(self):
        """Test ToolResult default success is True."""
        result = ToolResult()
        assert result.success is True


class TestParseDateReference:
    """Tests for date parsing utility."""

    def test_parse_today(self):
        """Test parsing 'today'."""
        result = parse_date_reference("today")
        now = datetime.now(timezone.utc)
        assert result.date() == now.date()

    def test_parse_tomorrow(self):
        """Test parsing 'tomorrow'."""
        result = parse_date_reference("tomorrow")
        expected = datetime.now(timezone.utc) + timedelta(days=1)
        assert result.date() == expected.date()

    def test_parse_yesterday(self):
        """Test parsing 'yesterday'."""
        result = parse_date_reference("yesterday")
        expected = datetime.now(timezone.utc) - timedelta(days=1)
        assert result.date() == expected.date()

    def test_parse_next_week(self):
        """Test parsing 'next week'."""
        result = parse_date_reference("next week")
        expected = datetime.now(timezone.utc) + timedelta(days=7)
        assert result.date() == expected.date()

    def test_parse_iso_format(self):
        """Test parsing ISO date format."""
        result = parse_date_reference("2024-06-15")
        assert result.year == 2024
        assert result.month == 6
        assert result.day == 15

    def test_parse_invalid_returns_now(self):
        """Test invalid date returns current date."""
        result = parse_date_reference("invalid-date")
        now = datetime.now(timezone.utc)
        assert result.date() == now.date()

    def test_parse_case_insensitive(self):
        """Test parsing is case insensitive."""
        result1 = parse_date_reference("TODAY")
        result2 = parse_date_reference("Today")
        result3 = parse_date_reference("today")
        assert result1.date() == result2.date() == result3.date()


class TestToolSchemas:
    """Tests for tool schema generation."""

    def test_get_tool_schemas_returns_list(self):
        """Test that get_tool_schemas returns a list."""
        schemas = get_tool_schemas()
        assert isinstance(schemas, list)
        assert len(schemas) == len(ALL_TOOLS)

    def test_all_tools_have_name(self):
        """Test all tool schemas have a name."""
        schemas = get_tool_schemas()
        for schema in schemas:
            assert "name" in schema
            assert schema["name"]

    def test_all_tools_have_description(self):
        """Test all tool schemas have a description."""
        schemas = get_tool_schemas()
        for schema in schemas:
            assert "description" in schema
            assert schema["description"]

    def test_all_tools_have_input_schema(self):
        """Test all tool schemas have input_schema."""
        schemas = get_tool_schemas()
        for schema in schemas:
            assert "input_schema" in schema
            assert schema["input_schema"]["type"] == "object"

    def test_semantic_search_schema(self):
        """Test SemanticSearchTool schema."""
        schemas = get_tool_schemas()
        search_schema = next(s for s in schemas if s["name"] == "SemanticSearchTool")

        assert "query" in search_schema["input_schema"]["properties"]
        assert "query" in search_schema["input_schema"]["required"]

    def test_search_emails_schema(self):
        """Test SearchEmailsTool schema."""
        schemas = get_tool_schemas()
        schema = next(s for s in schemas if s["name"] == "SearchEmailsTool")

        props = schema["input_schema"]["properties"]
        assert "query" in props
        assert "account" in props
        assert "tier1_only" in props

    def test_create_email_draft_schema(self):
        """Test CreateEmailDraftTool schema."""
        schemas = get_tool_schemas()
        schema = next(s for s in schemas if s["name"] == "CreateEmailDraftTool")

        props = schema["input_schema"]["properties"]
        assert "to" in props
        assert "subject" in props
        assert "body" in props
        assert "account" in props

    def test_respond_to_user_schema(self):
        """Test RespondToUserTool schema."""
        schemas = get_tool_schemas()
        schema = next(s for s in schemas if s["name"] == "RespondToUserTool")

        assert "message" in schema["input_schema"]["properties"]
        assert "message" in schema["input_schema"]["required"]


class TestToolModels:
    """Tests for individual tool Pydantic models."""

    def test_semantic_search_tool_validation(self):
        """Test SemanticSearchTool model validation."""
        tool = SemanticSearchTool(query="test query")
        assert tool.query == "test query"
        assert tool.max_results == 10  # default

    def test_semantic_search_tool_with_filters(self):
        """Test SemanticSearchTool with content type filters."""
        tool = SemanticSearchTool(
            query="test",
            content_types=["email", "file"],
            sources=["gmail"],
            max_results=5,
        )
        assert tool.content_types == ["email", "file"]
        assert tool.sources == ["gmail"]
        assert tool.max_results == 5

    def test_search_emails_tool_defaults(self):
        """Test SearchEmailsTool default values."""
        tool = SearchEmailsTool(query="from:test@example.com")
        assert tool.account is None
        assert tool.max_results == 20
        assert tool.tier1_only is False

    def test_get_calendar_events_tool_default_date(self):
        """Test GetCalendarEventsTool default date."""
        tool = GetCalendarEventsTool()
        assert tool.date == "today"

    def test_check_availability_tool_defaults(self):
        """Test CheckAvailabilityTool default values."""
        tool = CheckAvailabilityTool()
        assert tool.date == "today"
        assert tool.duration_minutes == 30
        assert tool.working_hours_start == 9
        assert tool.working_hours_end == 18

    def test_create_email_draft_tool_required_fields(self):
        """Test CreateEmailDraftTool required fields."""
        tool = CreateEmailDraftTool(
            to="test@example.com",
            subject="Test Subject",
            body="Test body content",
        )
        assert tool.to == "test@example.com"
        assert tool.subject == "Test Subject"
        assert tool.body == "Test body content"
        assert tool.account is None  # default (resolved at execution time)

    def test_get_github_prs_tool_defaults(self):
        """Test GetGitHubPRsTool default values."""
        tool = GetGitHubPRsTool()
        assert tool.state == "open"
        assert tool.max_results == 10

    def test_create_github_issue_tool(self):
        """Test CreateGitHubIssueTool model."""
        tool = CreateGitHubIssueTool(
            repo="owner/repo",
            title="Bug Report",
            body="Description",
            labels=["bug", "high-priority"],
        )
        assert tool.repo == "owner/repo"
        assert tool.title == "Bug Report"
        assert tool.labels == ["bug", "high-priority"]

    def test_respond_to_user_tool(self):
        """Test RespondToUserTool model."""
        tool = RespondToUserTool(message="Hello, how can I help?")
        assert tool.message == "Hello, how can I help?"


class TestToolNameMap:
    """Tests for tool name mapping."""

    def test_all_tools_mapped(self):
        """Test all tools have a handler mapping."""
        for tool_class in ALL_TOOLS:
            assert tool_class.__name__ in TOOL_NAME_MAP

    def test_unique_handler_names(self):
        """Test handler names are unique."""
        handler_names = list(TOOL_NAME_MAP.values())
        assert len(handler_names) == len(set(handler_names))

    def test_expected_mappings(self):
        """Test expected tool-to-handler mappings."""
        assert TOOL_NAME_MAP["SemanticSearchTool"] == "semantic_search"
        assert TOOL_NAME_MAP["SearchEmailsTool"] == "search_emails"
        assert TOOL_NAME_MAP["GetCalendarEventsTool"] == "get_calendar_events"
        assert TOOL_NAME_MAP["RespondToUserTool"] == "respond_to_user"
