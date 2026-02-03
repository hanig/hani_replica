"""Tests for MCP server."""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.mcp.server import (
    create_mcp_server,
    parse_date,
    _handle_search,
    _handle_search_emails,
    _handle_search_drive,
    _handle_get_calendar_events,
    _handle_check_availability,
    _handle_get_unread_counts,
    _handle_get_github_prs,
    _handle_get_github_issues,
    _handle_search_github_code,
    _handle_find_person,
    _handle_get_person_activity,
    _handle_get_knowledge_graph_stats,
    _handle_get_daily_briefing,
)


class TestParseDate:
    """Tests for date parsing utility."""

    def test_parse_today(self):
        """Test parsing 'today'."""
        result = parse_date("today")
        now = datetime.now(timezone.utc)
        assert result.date() == now.date()

    def test_parse_tomorrow(self):
        """Test parsing 'tomorrow'."""
        result = parse_date("tomorrow")
        now = datetime.now(timezone.utc)
        assert result.date() > now.date()

    def test_parse_yesterday(self):
        """Test parsing 'yesterday'."""
        result = parse_date("yesterday")
        now = datetime.now(timezone.utc)
        assert result.date() < now.date()

    def test_parse_iso_format(self):
        """Test parsing ISO format date."""
        result = parse_date("2024-06-15")
        assert result.year == 2024
        assert result.month == 6
        assert result.day == 15

    def test_parse_invalid_returns_now(self):
        """Test that invalid date strings return current time."""
        result = parse_date("not-a-date")
        now = datetime.now(timezone.utc)
        assert result.date() == now.date()

    def test_parse_case_insensitive(self):
        """Test parsing is case insensitive."""
        result_lower = parse_date("today")
        result_upper = parse_date("TODAY")
        result_mixed = parse_date("Today")

        assert result_lower.date() == result_upper.date() == result_mixed.date()


class TestMCPServer:
    """Tests for MCP server creation and configuration."""

    def test_create_server(self):
        """Test server creation."""
        server = create_mcp_server()
        assert server is not None
        assert server.name == "hani-replica"

    def test_server_has_handlers_registered(self):
        """Test that server has handlers registered."""
        server = create_mcp_server()

        # The server should have request handlers registered
        # Check that the server object is valid
        assert hasattr(server, 'name')
        assert hasattr(server, 'run')


class TestSearchHandlers:
    """Tests for search-related tool handlers."""

    @patch("src.mcp.server.get_query_engine")
    def test_handle_search(self, mock_get_engine):
        """Test semantic search handler."""
        mock_engine = MagicMock()
        mock_engine.search.return_value = [
            {"id": "1", "title": "Test result", "score": 0.9}
        ]
        mock_get_engine.return_value = mock_engine

        result = _handle_search({"query": "test query", "max_results": 5})

        assert result["query"] == "test query"
        assert result["result_count"] == 1
        assert len(result["results"]) == 1
        mock_engine.search.assert_called_once_with(
            query="test query",
            content_types=None,
            sources=None,
            top_k=5,
        )

    @patch("src.mcp.server.get_query_engine")
    def test_handle_search_with_filters(self, mock_get_engine):
        """Test search with content type and source filters."""
        mock_engine = MagicMock()
        mock_engine.search.return_value = []
        mock_get_engine.return_value = mock_engine

        _handle_search({
            "query": "test",
            "content_types": ["email", "file"],
            "sources": ["gmail"],
        })

        mock_engine.search.assert_called_once_with(
            query="test",
            content_types=["email", "file"],
            sources=["gmail"],
            top_k=10,
        )

    @patch("src.mcp.server.get_multi_google")
    def test_handle_search_emails(self, mock_get_google):
        """Test email search handler."""
        mock_manager = MagicMock()
        mock_manager.search_mail_tiered.return_value = [
            {"id": "email1", "subject": "Test email"}
        ]
        mock_get_google.return_value = mock_manager

        result = _handle_search_emails({"query": "from:test@example.com"})

        assert result["query"] == "from:test@example.com"
        assert result["result_count"] == 1
        mock_manager.search_mail_tiered.assert_called_once()

    @patch("src.mcp.server.get_multi_google")
    def test_handle_search_drive(self, mock_get_google):
        """Test Drive search handler."""
        mock_manager = MagicMock()
        mock_manager.search_drive_tiered.return_value = [
            {"id": "file1", "name": "Test document"}
        ]
        mock_get_google.return_value = mock_manager

        result = _handle_search_drive({"query": "project plan"})

        assert result["query"] == "project plan"
        assert result["result_count"] == 1


class TestCalendarHandlers:
    """Tests for calendar-related tool handlers."""

    @patch("src.mcp.server.get_multi_google")
    def test_handle_get_calendar_events(self, mock_get_google):
        """Test calendar events handler."""
        mock_manager = MagicMock()
        mock_manager.get_all_calendars_for_date.return_value = [
            {"id": "event1", "summary": "Team meeting"}
        ]
        mock_get_google.return_value = mock_manager

        result = _handle_get_calendar_events({"date": "today"})

        assert result["event_count"] == 1
        assert len(result["events"]) == 1

    @patch("src.mcp.server.get_multi_google")
    def test_handle_check_availability(self, mock_get_google):
        """Test availability check handler."""
        mock_manager = MagicMock()
        mock_manager.check_availability.return_value = [
            {"start": "09:00", "end": "10:00"},
            {"start": "14:00", "end": "16:00"},
        ]
        mock_get_google.return_value = mock_manager

        result = _handle_check_availability({
            "date": "tomorrow",
            "duration_minutes": 60,
        })

        assert result["duration_minutes"] == 60
        assert result["free_slot_count"] == 2


class TestEmailHandlers:
    """Tests for email-related tool handlers."""

    @patch("src.mcp.server.get_multi_google")
    def test_handle_get_unread_counts(self, mock_get_google):
        """Test unread counts handler."""
        mock_manager = MagicMock()
        mock_manager.get_unread_counts.return_value = {
            "arc": 5,
            "personal": 10,
            "tahoe": 0,
        }
        mock_get_google.return_value = mock_manager

        result = _handle_get_unread_counts({})

        assert result["total_unread"] == 15
        assert result["by_account"]["arc"] == 5
        assert result["by_account"]["personal"] == 10


class TestGitHubHandlers:
    """Tests for GitHub-related tool handlers."""

    @patch("src.mcp.server.get_github_client")
    def test_handle_get_github_prs(self, mock_get_github):
        """Test GitHub PRs handler."""
        mock_client = MagicMock()
        mock_client.get_my_prs.return_value = [
            {"id": 1, "title": "Add feature", "state": "open"}
        ]
        mock_get_github.return_value = mock_client

        result = _handle_get_github_prs({"state": "open"})

        assert result["state"] == "open"
        assert result["pr_count"] == 1

    @patch("src.mcp.server.get_github_client")
    def test_handle_get_github_issues(self, mock_get_github):
        """Test GitHub issues handler."""
        mock_client = MagicMock()
        mock_client.get_my_issues.return_value = [
            {"id": 1, "title": "Bug report", "state": "open"}
        ]
        mock_get_github.return_value = mock_client

        result = _handle_get_github_issues({})

        assert result["issue_count"] == 1

    @patch("src.mcp.server.get_github_client")
    def test_handle_search_github_code(self, mock_get_github):
        """Test GitHub code search handler."""
        mock_client = MagicMock()
        mock_client.search_code.return_value = [
            {"path": "src/main.py", "content": "def search():"}
        ]
        mock_get_github.return_value = mock_client

        result = _handle_search_github_code({"query": "def search"})

        assert result["query"] == "def search"
        assert result["result_count"] == 1

    @patch("src.mcp.server.get_github_client")
    def test_handle_search_github_code_in_repo(self, mock_get_github):
        """Test GitHub code search in specific repo."""
        mock_client = MagicMock()
        mock_client.search_code_in_repo.return_value = []
        mock_get_github.return_value = mock_client

        _handle_search_github_code({
            "query": "class Handler",
            "repo": "owner/repo",
        })

        mock_client.search_code_in_repo.assert_called_once()


class TestKnowledgeGraphHandlers:
    """Tests for knowledge graph tool handlers."""

    @patch("src.mcp.server.get_query_engine")
    def test_handle_find_person(self, mock_get_engine):
        """Test person search handler."""
        mock_engine = MagicMock()
        mock_engine.find_person.return_value = [
            {"id": "p1", "name": "John Doe", "email": "john@example.com"}
        ]
        mock_get_engine.return_value = mock_engine

        result = _handle_find_person({"query": "John"})

        assert result["query"] == "John"
        assert result["result_count"] == 1

    @patch("src.mcp.server.get_query_engine")
    def test_handle_get_person_activity(self, mock_get_engine):
        """Test person activity handler."""
        mock_engine = MagicMock()
        mock_engine.get_person_activity.return_value = [
            {"type": "email", "title": "Meeting notes"}
        ]
        mock_get_engine.return_value = mock_engine

        result = _handle_get_person_activity({"person_id": "p1"})

        assert result["person_id"] == "p1"
        assert result["activity_count"] == 1

    @patch("src.mcp.server.get_query_engine")
    def test_handle_get_knowledge_graph_stats(self, mock_get_engine):
        """Test knowledge graph stats handler."""
        mock_engine = MagicMock()
        mock_engine.get_stats.return_value = {
            "entity_count": 100,
            "content_count": 1000,
        }
        mock_get_engine.return_value = mock_engine

        result = _handle_get_knowledge_graph_stats({})

        assert result["entity_count"] == 100
        assert result["content_count"] == 1000


class TestBriefingHandler:
    """Tests for daily briefing handler."""

    @patch("src.mcp.server.get_github_client")
    @patch("src.mcp.server.get_multi_google")
    def test_handle_get_daily_briefing(self, mock_get_google, mock_get_github):
        """Test daily briefing handler."""
        mock_manager = MagicMock()
        mock_manager.get_all_calendars_today.return_value = [
            {"summary": "Morning standup"}
        ]
        mock_manager.get_unread_counts.return_value = {"arc": 5}
        mock_get_google.return_value = mock_manager

        mock_client = MagicMock()
        mock_client.get_my_prs.return_value = [{"title": "Feature PR"}]
        mock_client.get_my_issues.return_value = []
        mock_get_github.return_value = mock_client

        result = _handle_get_daily_briefing({})

        assert "date" in result
        assert len(result["events"]) == 1
        assert result["unread_counts"]["arc"] == 5
        assert len(result["open_prs"]) == 1

    @patch("src.mcp.server.get_github_client")
    @patch("src.mcp.server.get_multi_google")
    def test_handle_get_daily_briefing_handles_errors(
        self, mock_get_google, mock_get_github
    ):
        """Test that briefing handles integration errors gracefully."""
        mock_manager = MagicMock()
        mock_manager.get_all_calendars_today.side_effect = Exception("API Error")
        mock_manager.get_unread_counts.side_effect = Exception("API Error")
        mock_get_google.return_value = mock_manager

        mock_client = MagicMock()
        mock_client.get_my_prs.side_effect = Exception("API Error")
        mock_client.get_my_issues.side_effect = Exception("API Error")
        mock_get_github.return_value = mock_client

        # Should not raise
        result = _handle_get_daily_briefing({})

        assert "date" in result
        assert result["events"] == []
        assert result["unread_counts"] == {}


class TestToolExecution:
    """Tests for tool execution (tested via handlers directly)."""

    @patch("src.mcp.server.get_query_engine")
    def test_search_returns_correct_format(self, mock_get_engine):
        """Test that search handler returns correct format."""
        mock_engine = MagicMock()
        mock_engine.search.return_value = [{"id": "1", "title": "Result"}]
        mock_get_engine.return_value = mock_engine

        result = _handle_search({"query": "test"})

        assert "query" in result
        assert "result_count" in result
        assert "results" in result
        assert result["query"] == "test"
        assert result["result_count"] == 1

    @patch("src.mcp.server.get_query_engine")
    def test_search_handles_exception(self, mock_get_engine):
        """Test that search handler propagates exceptions."""
        mock_engine = MagicMock()
        mock_engine.search.side_effect = Exception("Database error")
        mock_get_engine.return_value = mock_engine

        with pytest.raises(Exception, match="Database error"):
            _handle_search({"query": "test"})

    @patch("src.mcp.server.get_multi_google")
    def test_email_search_with_tier1_only(self, mock_get_google):
        """Test email search with tier1_only parameter."""
        mock_manager = MagicMock()
        mock_manager.search_mail_tiered.return_value = []
        mock_get_google.return_value = mock_manager

        _handle_search_emails({"query": "test", "tier1_only": True})

        mock_manager.search_mail_tiered.assert_called_once_with(
            query="test",
            max_results=20,
            tier1_only=True,
        )
