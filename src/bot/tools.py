"""Tool definitions for Claude's native tool calling.

This module defines all tools available to the agent using Pydantic models.
These schemas are converted to Claude's tool format for native tool calling.
"""

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Standard result format for tool execution."""

    success: bool = True
    data: Any = None
    error: str | None = None

    def to_content(self) -> str:
        """Convert to string content for Claude."""
        if self.error:
            return f"Error: {self.error}"
        if isinstance(self.data, str):
            return self.data
        import json
        return json.dumps(self.data, default=str, indent=2)


# --- Search Tools ---

class SemanticSearchTool(BaseModel):
    """Search across all indexed data including emails, documents, calendar events, Slack messages, and GitHub content."""

    query: str = Field(description="The search query text")
    content_types: list[str] | None = Field(
        default=None,
        description="Filter by content types: email, file, event, message, issue, pr",
    )
    sources: list[str] | None = Field(
        default=None,
        description="Filter by sources: gmail, drive, calendar, github, slack",
    )
    max_results: int = Field(default=10, description="Maximum number of results")


class SearchEmailsTool(BaseModel):
    """Search emails across all Google accounts with tiered priority."""

    query: str = Field(description="Gmail search query (supports Gmail search syntax)")
    account: str | None = Field(
        default=None,
        description="Specific account to search: arc, personal, tahoe, therna, amplify",
    )
    max_results: int = Field(default=20, description="Maximum number of results")
    tier1_only: bool = Field(
        default=False,
        description="Only search primary accounts (Arc, Personal)",
    )


class SearchDriveTool(BaseModel):
    """Search Google Drive files across all accounts."""

    query: str = Field(description="Search text for file names and content")
    account: str | None = Field(
        default=None,
        description="Specific account to search",
    )
    max_results: int = Field(default=20, description="Maximum number of results")


# --- Calendar Tools ---

class GetCalendarEventsTool(BaseModel):
    """Get calendar events for a specific date from all Google calendars."""

    date: str = Field(
        default="today",
        description="Date reference: 'today', 'tomorrow', 'yesterday', or ISO format (YYYY-MM-DD)",
    )


class CheckAvailabilityTool(BaseModel):
    """Find available time slots across all calendars for scheduling meetings."""

    date: str = Field(
        default="today",
        description="Date to check: 'today', 'tomorrow', or ISO format",
    )
    duration_minutes: int = Field(
        default=30,
        description="Minimum slot duration in minutes",
    )
    working_hours_start: int = Field(
        default=9,
        description="Working hours start (24h format)",
    )
    working_hours_end: int = Field(
        default=18,
        description="Working hours end (24h format)",
    )


# --- Email Tools ---

class GetUnreadCountsTool(BaseModel):
    """Get unread email counts for all Google accounts."""

    pass  # No parameters needed


class CreateEmailDraftTool(BaseModel):
    """Create an email draft (never sends automatically)."""

    to: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email body content")
    account: str = Field(
        default="arc",
        description="Account to create draft in: arc, personal, tahoe, therna, amplify",
    )


class SendEmailTool(BaseModel):
    """Send an email immediately. Use with caution - this actually sends the email."""

    to: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email body content")
    account: str = Field(
        default="arc",
        description="Account to send from: arc, personal, tahoe, therna, amplify",
    )
    cc: str | None = Field(default=None, description="CC recipients (comma-separated)")
    bcc: str | None = Field(default=None, description="BCC recipients (comma-separated)")


# --- GitHub Tools ---

class GetGitHubPRsTool(BaseModel):
    """Get pull requests from GitHub."""

    state: Literal["open", "closed", "all"] = Field(
        default="open",
        description="PR state filter",
    )
    max_results: int = Field(default=10, description="Maximum number of results")


class GetGitHubIssuesTool(BaseModel):
    """Get GitHub issues assigned to the user."""

    state: Literal["open", "closed", "all"] = Field(
        default="open",
        description="Issue state filter",
    )
    max_results: int = Field(default=10, description="Maximum number of results")


class SearchGitHubCodeTool(BaseModel):
    """Search code in GitHub repositories."""

    query: str = Field(description="Code search query")
    repo: str | None = Field(
        default=None,
        description="Limit to specific repository (e.g., 'owner/repo')",
    )
    max_results: int = Field(default=20, description="Maximum number of results")


class CreateGitHubIssueTool(BaseModel):
    """Create a new GitHub issue."""

    repo: str = Field(description="Repository in format 'owner/repo'")
    title: str = Field(description="Issue title")
    body: str = Field(default="", description="Issue description/body")
    labels: list[str] = Field(default_factory=list, description="Labels to apply")


# --- Knowledge Graph Tools ---

class FindPersonTool(BaseModel):
    """Find people in the knowledge graph by name or email."""

    query: str = Field(description="Name or email to search for")


class GetPersonActivityTool(BaseModel):
    """Get recent activity involving a specific person."""

    person_id: str = Field(description="Person entity ID from the knowledge graph")
    content_types: list[str] | None = Field(
        default=None,
        description="Filter by content types: email, event, message",
    )
    max_results: int = Field(default=20, description="Maximum number of results")


# --- Briefing Tool ---

class GetDailyBriefingTool(BaseModel):
    """Get a daily briefing summary including calendar, emails, and GitHub status."""

    pass  # No parameters needed


# --- Response Tool ---

class RespondToUserTool(BaseModel):
    """Send a response to the user. Use this when you want to reply conversationally or don't need to use any other tools."""

    message: str = Field(description="The message to send to the user")


# All available tools
ALL_TOOLS: list[type[BaseModel]] = [
    SemanticSearchTool,
    SearchEmailsTool,
    SearchDriveTool,
    GetCalendarEventsTool,
    CheckAvailabilityTool,
    GetUnreadCountsTool,
    CreateEmailDraftTool,
    SendEmailTool,
    GetGitHubPRsTool,
    GetGitHubIssuesTool,
    SearchGitHubCodeTool,
    CreateGitHubIssueTool,
    FindPersonTool,
    GetPersonActivityTool,
    GetDailyBriefingTool,
    RespondToUserTool,
]

# Map tool class names to handler functions (set up in executor)
TOOL_NAME_MAP = {
    "SemanticSearchTool": "semantic_search",
    "SearchEmailsTool": "search_emails",
    "SearchDriveTool": "search_drive",
    "GetCalendarEventsTool": "get_calendar_events",
    "CheckAvailabilityTool": "check_availability",
    "GetUnreadCountsTool": "get_unread_counts",
    "CreateEmailDraftTool": "create_email_draft",
    "SendEmailTool": "send_email",
    "GetGitHubPRsTool": "get_github_prs",
    "GetGitHubIssuesTool": "get_github_issues",
    "SearchGitHubCodeTool": "search_github_code",
    "CreateGitHubIssueTool": "create_github_issue",
    "FindPersonTool": "find_person",
    "GetPersonActivityTool": "get_person_activity",
    "GetDailyBriefingTool": "get_daily_briefing",
    "RespondToUserTool": "respond_to_user",
}


def get_tool_schemas() -> list[dict[str, Any]]:
    """Get all tool schemas in Claude's tool format.

    Returns:
        List of tool definitions for Claude's API.
    """
    schemas = []

    for tool_class in ALL_TOOLS:
        # Get the JSON schema from Pydantic
        json_schema = tool_class.model_json_schema()

        # Extract description from docstring
        description = tool_class.__doc__ or ""

        # Build Claude's tool format
        tool_def = {
            "name": tool_class.__name__,
            "description": description.strip(),
            "input_schema": {
                "type": "object",
                "properties": json_schema.get("properties", {}),
                "required": json_schema.get("required", []),
            },
        }

        schemas.append(tool_def)

    return schemas


def parse_date_reference(date_ref: str) -> datetime:
    """Parse a date reference string into a datetime.

    Args:
        date_ref: Date reference like "today", "tomorrow", etc.

    Returns:
        datetime object.
    """
    now = datetime.now(timezone.utc)
    date_ref_lower = date_ref.lower().strip()

    if date_ref_lower == "today":
        return now
    elif date_ref_lower == "tomorrow":
        return now + timedelta(days=1)
    elif date_ref_lower == "yesterday":
        return now - timedelta(days=1)
    elif date_ref_lower == "next week":
        return now + timedelta(days=7)
    elif date_ref_lower == "this week":
        return now
    else:
        try:
            return datetime.fromisoformat(date_ref)
        except ValueError:
            return now
