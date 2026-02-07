"""Tool definitions for Claude's native tool calling.

This module defines all tools available to the agent using Pydantic models.
These schemas are converted to Claude's tool format for native tool calling.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..config import get_user_timezone, get_accounts_description, PRIMARY_ACCOUNT, ZOTERO_DEFAULT_COLLECTION

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
        description=get_accounts_description(),
    )
    max_results: int = Field(default=20, description="Maximum number of results")
    tier1_only: bool = Field(
        default=False,
        description="Only search primary/tier-1 accounts",
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


class CreateCalendarEventTool(BaseModel):
    """Create a calendar event and optionally send invites to attendees."""

    title: str = Field(description="Event title/summary")
    date: str = Field(
        description="Date for the event: 'today', 'tomorrow', day name (e.g., 'Monday'), or ISO format (YYYY-MM-DD)",
    )
    time: str = Field(
        description="Start time: 'noon', '2pm', '14:00', etc.",
    )
    duration_minutes: int = Field(
        default=60,
        description="Event duration in minutes (default 60)",
    )
    attendees: list[str] = Field(
        default_factory=list,
        description="List of attendee email addresses (will receive calendar invites)",
    )
    location: str = Field(
        default="",
        description="Event location (optional)",
    )
    description: str = Field(
        default="",
        description="Event description (optional)",
    )
    account: str | None = Field(
        default=None,
        description="Google account to create event in. " + get_accounts_description(),
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
    account: str | None = Field(
        default=None,
        description="Account to create draft in. " + get_accounts_description(),
    )


class SendEmailTool(BaseModel):
    """Send an email immediately. Use with caution - this actually sends the email."""

    to: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email body content")
    account: str | None = Field(
        default=None,
        description="Account to send from. " + get_accounts_description(),
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


# --- Todoist Tools ---

class GetTodoistTasksTool(BaseModel):
    """Get active tasks from Todoist."""

    project: str | None = Field(
        default=None,
        description="Filter by project name (optional)",
    )
    filter: str | None = Field(
        default=None,
        description="Todoist filter string like 'today', 'overdue', '@label' (optional)",
    )


class CreateTodoistTaskTool(BaseModel):
    """Create a new task in Todoist."""

    content: str = Field(description="Task title/content")
    description: str | None = Field(default=None, description="Task description")
    due: str | None = Field(
        default=None,
        description="Due date in natural language (e.g., 'tomorrow', 'next monday', 'jan 15')",
    )
    project: str | None = Field(
        default=None,
        description="Project name to add task to (defaults to Inbox)",
    )
    priority: int = Field(
        default=1,
        description="Priority 1-4 where 4 is urgent (default 1)",
    )
    labels: list[str] | None = Field(
        default=None,
        description="Labels to add to the task",
    )


class CompleteTodoistTaskTool(BaseModel):
    """Mark a Todoist task as complete."""

    task_id: str = Field(description="The task ID to complete")


# --- Notion Tools ---

class SearchNotionTool(BaseModel):
    """Search Notion pages and databases."""

    query: str = Field(description="Search query text")
    max_results: int = Field(default=10, description="Maximum number of results")


class CreateNotionPageTool(BaseModel):
    """Create a new page in a Notion database."""

    database_id: str = Field(description="Target database ID")
    title: str = Field(description="Page title")
    properties: dict = Field(
        default_factory=dict,
        description="Additional properties matching the database schema",
    )


class AddNotionCommentTool(BaseModel):
    """Add a comment to a Notion page."""

    page_id: str = Field(description="Page ID to comment on")
    content: str = Field(description="Comment text")


# --- Zotero Tools ---

class SearchZoteroPapersTool(BaseModel):
    """Search papers in Zotero library by title, author, tag, or keyword."""

    query: str = Field(description="Search query (title, author, keyword)")
    max_results: int = Field(default=10, description="Maximum number of results")


class GetZoteroPaperTool(BaseModel):
    """Get full details of a Zotero paper including abstract, metadata, and notes."""

    item_key: str = Field(description="Zotero item key")


class ListRecentPapersTool(BaseModel):
    """List papers recently added to Zotero library."""

    days: int = Field(default=7, description="Look back N days")
    max_results: int = Field(default=20, description="Maximum number of results")


class SearchPapersByTagTool(BaseModel):
    """Search Zotero papers by tag."""

    tag: str = Field(description="Tag to search for")
    max_results: int = Field(default=20, description="Maximum number of results")


class GetZoteroCollectionTool(BaseModel):
    """Get papers in a specific Zotero collection/folder."""

    collection_name: str = Field(description="Collection name")
    max_results: int = Field(default=50, description="Maximum number of results")


class AddZoteroPaperTool(BaseModel):
    """Add a paper to Zotero library by DOI or URL."""

    identifier: str = Field(
        description="DOI (e.g., 10.1038/xxx) or URL to paper"
    )
    collection: str = Field(
        default=ZOTERO_DEFAULT_COLLECTION or "",
        description="Collection name to add to",
    )


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
    CreateCalendarEventTool,
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
    GetTodoistTasksTool,
    CreateTodoistTaskTool,
    CompleteTodoistTaskTool,
    SearchNotionTool,
    CreateNotionPageTool,
    AddNotionCommentTool,
    SearchZoteroPapersTool,
    GetZoteroPaperTool,
    ListRecentPapersTool,
    SearchPapersByTagTool,
    GetZoteroCollectionTool,
    AddZoteroPaperTool,
    RespondToUserTool,
]

# Map tool class names to handler functions (set up in executor)
TOOL_NAME_MAP = {
    "SemanticSearchTool": "semantic_search",
    "SearchEmailsTool": "search_emails",
    "SearchDriveTool": "search_drive",
    "GetCalendarEventsTool": "get_calendar_events",
    "CheckAvailabilityTool": "check_availability",
    "CreateCalendarEventTool": "create_calendar_event",
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
    "GetTodoistTasksTool": "get_todoist_tasks",
    "CreateTodoistTaskTool": "create_todoist_task",
    "CompleteTodoistTaskTool": "complete_todoist_task",
    "SearchNotionTool": "search_notion",
    "CreateNotionPageTool": "create_notion_page",
    "AddNotionCommentTool": "add_notion_comment",
    "SearchZoteroPapersTool": "search_zotero_papers",
    "GetZoteroPaperTool": "get_zotero_paper",
    "ListRecentPapersTool": "list_recent_papers",
    "SearchPapersByTagTool": "search_papers_by_tag",
    "GetZoteroCollectionTool": "get_zotero_collection",
    "AddZoteroPaperTool": "add_zotero_paper",
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


def _parse_iso_datetime(value: str) -> datetime | None:
    """Parse ISO datetime/date, handling 'Z' suffix."""
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def parse_date_reference(date_ref: str) -> datetime:
    """Parse a date reference string into a datetime.

    Args:
        date_ref: Date reference like "today", "tomorrow", etc.

    Returns:
        datetime object.
    """
    tz = get_user_timezone()
    now = datetime.now(tz)
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
        parsed = _parse_iso_datetime(date_ref)
        if parsed:
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=tz)
            return parsed
        return now
