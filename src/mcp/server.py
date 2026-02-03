"""MCP Server exposing Hani Replica capabilities as tools.

This server exposes the knowledge graph, calendar, email, and search
capabilities via the Model Context Protocol, allowing Claude Desktop,
Cursor, and other MCP-compatible tools to access Hani's indexed data.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    ListToolsResult,
    TextContent,
    Tool,
)

logger = logging.getLogger(__name__)

# Global instances (lazy-loaded)
_query_engine = None
_multi_google = None
_github_client = None
_knowledge_graph = None


def get_query_engine():
    """Get or create the query engine instance."""
    global _query_engine
    if _query_engine is None:
        from ..query.engine import QueryEngine
        _query_engine = QueryEngine()
    return _query_engine


def get_multi_google():
    """Get or create the multi-Google manager instance."""
    global _multi_google
    if _multi_google is None:
        from ..integrations.google_multi import MultiGoogleManager
        _multi_google = MultiGoogleManager()
    return _multi_google


def get_github_client():
    """Get or create the GitHub client instance."""
    global _github_client
    if _github_client is None:
        from ..integrations.github_client import GitHubClient
        _github_client = GitHubClient()
    return _github_client


def get_knowledge_graph():
    """Get or create the knowledge graph instance."""
    global _knowledge_graph
    if _knowledge_graph is None:
        from ..knowledge_graph import KnowledgeGraph
        _knowledge_graph = KnowledgeGraph()
    return _knowledge_graph


def parse_date(date_str: str) -> datetime:
    """Parse a date string into a datetime object.

    Args:
        date_str: Date string like "today", "tomorrow", "2024-01-15".

    Returns:
        datetime object.
    """
    now = datetime.now(timezone.utc)
    date_lower = date_str.lower().strip()

    if date_lower == "today":
        return now
    elif date_lower == "tomorrow":
        return now + timedelta(days=1)
    elif date_lower == "yesterday":
        return now - timedelta(days=1)
    elif date_lower == "next week":
        return now + timedelta(days=7)
    elif date_lower == "this week":
        return now
    else:
        try:
            return datetime.fromisoformat(date_str)
        except ValueError:
            return now


def create_mcp_server() -> Server:
    """Create and configure the MCP server.

    Returns:
        Configured MCP Server instance.
    """
    server = Server("hani-replica")

    @server.list_tools()
    async def list_tools() -> ListToolsResult:
        """List all available tools."""
        return ListToolsResult(tools=[
            # Search tools
            Tool(
                name="search",
                description="Semantic search across all indexed data including emails, documents, calendar events, Slack messages, and GitHub content.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query text",
                        },
                        "content_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by content types: email, file, event, message, issue, pr",
                        },
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by sources: gmail, drive, calendar, github, slack",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 10)",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="search_emails",
                description="Search emails across all Google accounts with tiered priority (Arc and Personal accounts searched first).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Gmail search query (supports Gmail search syntax)",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 20)",
                            "default": 20,
                        },
                        "tier1_only": {
                            "type": "boolean",
                            "description": "Only search primary accounts (Arc, Personal)",
                            "default": False,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="search_drive",
                description="Search Google Drive files across all accounts.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search text for file names and content",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 20)",
                            "default": 20,
                        },
                    },
                    "required": ["query"],
                },
            ),

            # Calendar tools
            Tool(
                name="get_calendar_events",
                description="Get calendar events for a specific date from all Google calendars.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "description": "Date reference: 'today', 'tomorrow', 'yesterday', or ISO format (YYYY-MM-DD)",
                            "default": "today",
                        },
                    },
                },
            ),
            Tool(
                name="check_availability",
                description="Find available time slots across all calendars for scheduling meetings.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "description": "Date to check: 'today', 'tomorrow', or ISO format",
                            "default": "today",
                        },
                        "duration_minutes": {
                            "type": "integer",
                            "description": "Minimum slot duration in minutes (default: 30)",
                            "default": 30,
                        },
                        "working_hours_start": {
                            "type": "integer",
                            "description": "Working hours start (24h format, default: 9)",
                            "default": 9,
                        },
                        "working_hours_end": {
                            "type": "integer",
                            "description": "Working hours end (24h format, default: 18)",
                            "default": 18,
                        },
                    },
                },
            ),

            # Email tools
            Tool(
                name="get_unread_counts",
                description="Get unread email counts for all Google accounts.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),

            # GitHub tools
            Tool(
                name="get_github_prs",
                description="Get open pull requests from GitHub.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "state": {
                            "type": "string",
                            "description": "PR state: 'open', 'closed', or 'all'",
                            "default": "open",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 10)",
                            "default": 10,
                        },
                    },
                },
            ),
            Tool(
                name="get_github_issues",
                description="Get GitHub issues assigned to the user.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "state": {
                            "type": "string",
                            "description": "Issue state: 'open', 'closed', or 'all'",
                            "default": "open",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 10)",
                            "default": 10,
                        },
                    },
                },
            ),
            Tool(
                name="search_github_code",
                description="Search code in GitHub repositories.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Code search query",
                        },
                        "repo": {
                            "type": "string",
                            "description": "Optional: limit to specific repository",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 20)",
                            "default": 20,
                        },
                    },
                    "required": ["query"],
                },
            ),

            # Knowledge graph tools
            Tool(
                name="find_person",
                description="Find people in the knowledge graph by name or email.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Name or email to search for",
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="get_person_activity",
                description="Get recent activity involving a specific person (emails, meetings, mentions).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "person_id": {
                            "type": "string",
                            "description": "Person entity ID from the knowledge graph",
                        },
                        "content_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by content types",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 20)",
                            "default": 20,
                        },
                    },
                    "required": ["person_id"],
                },
            ),
            Tool(
                name="get_knowledge_graph_stats",
                description="Get statistics about the indexed knowledge graph (entity counts, content counts, sync status).",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),

            # Briefing tool
            Tool(
                name="get_daily_briefing",
                description="Get a daily briefing summary including calendar events, unread emails, and GitHub status.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
        ])

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
        """Execute a tool and return results."""
        try:
            if name == "search":
                result = _handle_search(arguments)
            elif name == "search_emails":
                result = _handle_search_emails(arguments)
            elif name == "search_drive":
                result = _handle_search_drive(arguments)
            elif name == "get_calendar_events":
                result = _handle_get_calendar_events(arguments)
            elif name == "check_availability":
                result = _handle_check_availability(arguments)
            elif name == "get_unread_counts":
                result = _handle_get_unread_counts(arguments)
            elif name == "get_github_prs":
                result = _handle_get_github_prs(arguments)
            elif name == "get_github_issues":
                result = _handle_get_github_issues(arguments)
            elif name == "search_github_code":
                result = _handle_search_github_code(arguments)
            elif name == "find_person":
                result = _handle_find_person(arguments)
            elif name == "get_person_activity":
                result = _handle_get_person_activity(arguments)
            elif name == "get_knowledge_graph_stats":
                result = _handle_get_knowledge_graph_stats(arguments)
            elif name == "get_daily_briefing":
                result = _handle_get_daily_briefing(arguments)
            else:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Unknown tool: {name}")]
                )

            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, default=str, indent=2))]
            )

        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error: {str(e)}")]
            )

    return server


# Tool handlers

def _handle_search(arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle semantic search."""
    query = arguments["query"]
    content_types = arguments.get("content_types")
    sources = arguments.get("sources")
    max_results = arguments.get("max_results", 10)

    engine = get_query_engine()
    results = engine.search(
        query=query,
        content_types=content_types,
        sources=sources,
        top_k=max_results,
    )

    return {
        "query": query,
        "result_count": len(results),
        "results": results,
    }


def _handle_search_emails(arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle email search."""
    query = arguments["query"]
    max_results = arguments.get("max_results", 20)
    tier1_only = arguments.get("tier1_only", False)

    manager = get_multi_google()
    emails = manager.search_mail_tiered(
        query=query,
        max_results=max_results,
        tier1_only=tier1_only,
    )

    return {
        "query": query,
        "result_count": len(emails),
        "emails": emails,
    }


def _handle_search_drive(arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle Drive search."""
    query = arguments["query"]
    max_results = arguments.get("max_results", 20)

    manager = get_multi_google()
    files = manager.search_drive_tiered(
        query=query,
        max_results=max_results,
    )

    return {
        "query": query,
        "result_count": len(files),
        "files": files,
    }


def _handle_get_calendar_events(arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle calendar events request."""
    date_str = arguments.get("date", "today")
    target_date = parse_date(date_str)

    manager = get_multi_google()
    events = manager.get_all_calendars_for_date(target_date)

    return {
        "date": target_date.strftime("%Y-%m-%d"),
        "event_count": len(events),
        "events": events,
    }


def _handle_check_availability(arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle availability check."""
    date_str = arguments.get("date", "today")
    duration = arguments.get("duration_minutes", 30)
    work_start = arguments.get("working_hours_start", 9)
    work_end = arguments.get("working_hours_end", 18)

    target_date = parse_date(date_str)

    manager = get_multi_google()
    free_slots = manager.check_availability(
        date=target_date,
        duration_minutes=duration,
        working_hours=(work_start, work_end),
    )

    return {
        "date": target_date.strftime("%Y-%m-%d"),
        "duration_minutes": duration,
        "working_hours": f"{work_start}:00 - {work_end}:00",
        "free_slot_count": len(free_slots),
        "free_slots": free_slots,
    }


def _handle_get_unread_counts(arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle unread counts request."""
    manager = get_multi_google()
    counts = manager.get_unread_counts()

    total = sum(c for c in counts.values() if c >= 0)

    return {
        "total_unread": total,
        "by_account": counts,
    }


def _handle_get_github_prs(arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle GitHub PRs request."""
    state = arguments.get("state", "open")
    max_results = arguments.get("max_results", 10)

    client = get_github_client()
    prs = client.get_my_prs(state=state, max_results=max_results)

    return {
        "state": state,
        "pr_count": len(prs),
        "pull_requests": prs,
    }


def _handle_get_github_issues(arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle GitHub issues request."""
    state = arguments.get("state", "open")
    max_results = arguments.get("max_results", 10)

    client = get_github_client()
    issues = client.get_my_issues(state=state, max_results=max_results)

    return {
        "state": state,
        "issue_count": len(issues),
        "issues": issues,
    }


def _handle_search_github_code(arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle GitHub code search."""
    query = arguments["query"]
    repo = arguments.get("repo")
    max_results = arguments.get("max_results", 20)

    client = get_github_client()

    if repo:
        results = client.search_code_in_repo(repo, query, max_results=max_results)
    else:
        results = client.search_code(query, max_results=max_results)

    return {
        "query": query,
        "repo": repo,
        "result_count": len(results),
        "results": results,
    }


def _handle_find_person(arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle person search."""
    query = arguments["query"]

    engine = get_query_engine()
    people = engine.find_person(query)

    return {
        "query": query,
        "result_count": len(people),
        "people": people,
    }


def _handle_get_person_activity(arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle person activity request."""
    person_id = arguments["person_id"]
    content_types = arguments.get("content_types")
    max_results = arguments.get("max_results", 20)

    engine = get_query_engine()
    activity = engine.get_person_activity(
        person_id=person_id,
        content_types=content_types,
        limit=max_results,
    )

    return {
        "person_id": person_id,
        "activity_count": len(activity),
        "activity": activity,
    }


def _handle_get_knowledge_graph_stats(arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle knowledge graph stats request."""
    engine = get_query_engine()
    return engine.get_stats()


def _handle_get_daily_briefing(arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle daily briefing request."""
    manager = get_multi_google()
    github = get_github_client()

    briefing = {
        "date": datetime.now(timezone.utc).strftime("%A, %B %d, %Y"),
        "events": [],
        "unread_counts": {},
        "open_prs": [],
        "open_issues": [],
    }

    try:
        briefing["events"] = manager.get_all_calendars_today()
    except Exception as e:
        logger.warning(f"Error getting calendar: {e}")

    try:
        briefing["unread_counts"] = manager.get_unread_counts()
    except Exception as e:
        logger.warning(f"Error getting unread counts: {e}")

    try:
        briefing["open_prs"] = github.get_my_prs(state="open", max_results=10)
    except Exception as e:
        logger.warning(f"Error getting PRs: {e}")

    try:
        briefing["open_issues"] = github.get_my_issues(state="open", max_results=10)
    except Exception as e:
        logger.warning(f"Error getting issues: {e}")

    return briefing


async def run_server():
    """Run the MCP server with stdio transport."""
    server = create_mcp_server()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )
