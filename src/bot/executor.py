"""Agent executor for multi-step tool calling with Claude.

This module implements an agentic loop that uses Claude's native tool calling
to dynamically select and execute tools until a final response is generated.
Supports both synchronous and streaming execution modes.
"""

import logging
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from anthropic import Anthropic

from ..config import (
    AGENT_MODEL,
    ANTHROPIC_API_KEY,
    ENABLE_DIRECT_EMAIL_SEND,
    PRIMARY_ACCOUNT,
    ZOTERO_DEFAULT_COLLECTION,
    get_user_timezone,
)
from .tools import (
    TOOL_NAME_MAP,
    ToolResult,
    get_tool_schemas,
    parse_date_reference,
)
from .tracing import get_trace_logger, model_usage

if TYPE_CHECKING:
    from .actions.confirmable import PendingAction
    from .conversation import ConversationContext
    from .user_memory import UserMemory

logger = logging.getLogger(__name__)

# Maximum number of tool calling iterations to prevent infinite loops
MAX_ITERATIONS = 10

SYSTEM_PROMPT = """You are a personal AI assistant with access to tools for managing emails, calendar, GitHub, Todoist tasks, and searching a personal knowledge graph.

You have access to tools that let you:
- Search across all indexed data (emails, documents, calendar events, Slack messages, GitHub)
- Search and manage emails across multiple Google accounts
- Manage outbound email
- Check calendar events and availability
- Create calendar events and send meeting invites to attendees
- Update or cancel calendar events when the user provides an event ID or enough context to identify one
- Search GitHub code, issues, and PRs
- Create GitHub issues
- Get, create, update, comment on, reopen, and complete Todoist tasks
- Search Notion pages and databases
- Add, reply to, and resolve Google Doc comments
- Read and update proactive notification settings
- Get daily briefings

Guidelines:
1. Be conversational and helpful. You can chat naturally without using tools for greetings and simple questions.
2. Use the RespondToUserTool when you want to reply directly to the user.
3. For data requests, use the appropriate tool to fetch real information.
4. If a task requires multiple steps, execute them in sequence.
5. Always provide clear, concise responses.
6. Never make up information - only use data from tools.
7. For actions like creating issues, drafts, or sending emails, confirm the details before executing.
8. {email_send_policy}
9. When the user asks about "tasks" or "to-dos", use the Todoist tools (GetTodoistTasksTool, CreateTodoistTaskTool).

Current local date/time: {current_date}
"""


class StreamEventType(str, Enum):
    """Types of streaming events."""

    TEXT_DELTA = "text_delta"  # Incremental text chunk
    TEXT_DONE = "text_done"  # Text block completed
    TOOL_START = "tool_start"  # Tool execution starting
    TOOL_DONE = "tool_done"  # Tool execution completed
    THINKING = "thinking"  # Status update (e.g., "Searching...")
    ERROR = "error"  # Error occurred
    DONE = "done"  # Streaming complete


@dataclass
class StreamEvent:
    """Event emitted during streaming execution."""

    event_type: StreamEventType
    data: str = ""
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_result: str | None = None
    error: str | None = None
    iteration: int = 0


@dataclass
class ExecutionResult:
    """Result of agent execution."""

    response: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    success: bool = True
    error: str | None = None
    response_blocks: list[dict[str, Any]] | None = None


class ToolExecutor:
    """Executes individual tools and returns results."""

    def __init__(self):
        """Initialize the tool executor with lazy-loaded integrations."""
        self._semantic_indexer = None
        self._multi_google = None
        self._github_client = None
        self._query_engine = None
        self._notion_client = None
        self._todoist_client = None
        self._zotero_client = None
        self._proactive_settings = None

    @property
    def semantic_indexer(self):
        """Lazy load semantic indexer."""
        if self._semantic_indexer is None:
            from ..semantic.semantic_indexer import SemanticIndexer
            self._semantic_indexer = SemanticIndexer()
        return self._semantic_indexer

    @property
    def multi_google(self):
        """Lazy load multi-Google manager."""
        if self._multi_google is None:
            from ..integrations.google_multi import MultiGoogleManager
            self._multi_google = MultiGoogleManager()
        return self._multi_google

    @property
    def github_client(self):
        """Lazy load GitHub client."""
        if self._github_client is None:
            from ..integrations.github_client import GitHubClient
            self._github_client = GitHubClient()
        return self._github_client

    @property
    def query_engine(self):
        """Lazy load query engine."""
        if self._query_engine is None:
            from ..query.engine import QueryEngine
            self._query_engine = QueryEngine()
        return self._query_engine

    @property
    def notion_client(self):
        """Lazy load Notion client."""
        if self._notion_client is None:
            from ..integrations.notion_client import NotionClient
            self._notion_client = NotionClient()
        return self._notion_client

    @property
    def todoist_client(self):
        """Lazy load Todoist client."""
        if self._todoist_client is None:
            from ..integrations.todoist_client import TodoistClient
            self._todoist_client = TodoistClient()
        return self._todoist_client

    @property
    def zotero_client(self):
        """Lazy load Zotero client."""
        if self._zotero_client is None:
            from ..integrations.zotero_client import ZoteroClient
            self._zotero_client = ZoteroClient()
        return self._zotero_client

    @property
    def proactive_settings(self):
        """Lazy load proactive settings store."""
        if self._proactive_settings is None:
            from .proactive_settings import ProactiveSettingsStore
            self._proactive_settings = ProactiveSettingsStore()
        return self._proactive_settings

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: "ConversationContext | None" = None,
    ) -> ToolResult:
        """Execute a tool and return the result.

        Args:
            tool_name: Name of the tool class.
            arguments: Tool arguments.

        Returns:
            ToolResult with data or error.
        """
        try:
            handler_name = TOOL_NAME_MAP.get(tool_name)
            if not handler_name:
                return ToolResult(success=False, error=f"Unknown tool: {tool_name}")

            handler = getattr(self, f"_execute_{handler_name}", None)
            if not handler:
                return ToolResult(success=False, error=f"No handler for: {handler_name}")

            if handler_name == "send_email":
                return handler(arguments, context=context)
            if handler_name in {
                "create_calendar_event",
                "update_calendar_event",
                "delete_calendar_event",
                "create_email_draft",
                "create_github_issue",
                "create_todoist_task",
                "complete_todoist_task",
                "update_todoist_task",
                "add_todoist_comment",
                "reopen_todoist_task",
                "create_notion_page",
                "add_notion_comment",
                "add_google_doc_comment",
                "reply_google_doc_comment",
                "resolve_google_doc_comment",
                "get_proactive_settings",
                "update_proactive_settings",
                "add_zotero_paper",
            }:
                return handler(arguments, context=context)
            return handler(arguments)

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
            return ToolResult(success=False, error=str(e))

    def _queue_confirmation(
        self,
        action: "PendingAction",
        context: "ConversationContext | None",
        message: str,
    ) -> ToolResult:
        """Queue a write action behind Slack button confirmation."""
        if context is None:
            return ToolResult(
                success=False,
                error="Missing conversation context for confirmation-gated action.",
            )

        context.pending_action = action
        return ToolResult(data={
            "requires_confirmation": True,
            "message": message,
            "confirmation": action.get_confirmation_prompt(),
        })

    def _execute_semantic_search(self, args: dict) -> ToolResult:
        """Execute semantic search."""
        filters = {}
        sources = args.get("sources")
        if sources and len(sources) == 1:
            filters["source"] = sources[0]
        elif sources:
            # Chroma metadata filters do not support the simple list shape here;
            # filter after search when multiple sources are requested.
            filters = None

        results = self.semantic_indexer.search(
            query=args["query"],
            content_types=args.get("content_types"),
            top_k=args.get("max_results", 10),
            filters=filters or None,
        )
        if sources and len(sources) > 1:
            results = [
                r for r in results
                if (r.get("metadata") or {}).get("source") in sources
            ]
        return ToolResult(data={
            "query": args["query"],
            "result_count": len(results),
            "results": results[:args.get("max_results", 10)],
        })

    def _execute_search_emails(self, args: dict) -> ToolResult:
        """Execute email search."""
        results = self.multi_google.search_mail_tiered(
            query=args["query"],
            max_results=args.get("max_results", 20),
            tier1_only=args.get("tier1_only", False),
        )
        return ToolResult(data={
            "query": args["query"],
            "result_count": len(results),
            "emails": results,
        })

    def _execute_search_drive(self, args: dict) -> ToolResult:
        """Execute Drive search."""
        results = self.multi_google.search_drive_tiered(
            query=args["query"],
            max_results=args.get("max_results", 20),
        )
        return ToolResult(data={
            "query": args["query"],
            "result_count": len(results),
            "files": results,
        })

    def _execute_get_calendar_events(self, args: dict) -> ToolResult:
        """Get calendar events."""
        target_date = parse_date_reference(args.get("date", "today"))
        tz = get_user_timezone()
        now = datetime.now(tz)
        events = self.multi_google.get_all_calendars_for_date(target_date)
        target_local_date = target_date.astimezone(tz).date()
        today_local_date = now.date()
        upcoming_events = (
            self._filter_upcoming_events(events, now)
            if target_local_date == today_local_date
            else events
        )

        return ToolResult(data={
            "date": target_date.strftime("%Y-%m-%d"),
            "current_time": now.isoformat(),
            "timezone": str(tz),
            "event_count": len(events),
            "upcoming_event_count": len(upcoming_events),
            "next_event": upcoming_events[0] if upcoming_events else None,
            "events": events,
            "upcoming_events": upcoming_events,
        })

    @staticmethod
    def _coerce_event_datetime(value: Any, tz) -> datetime | None:
        """Normalize event datetime values from integrations or mocks."""
        if value is None:
            return None
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, str):
            raw = value[:-1] + "+00:00" if value.endswith("Z") else value
            try:
                dt = datetime.fromisoformat(raw)
            except ValueError:
                return None
        else:
            return None

        if dt.tzinfo is None:
            return dt.replace(tzinfo=tz)
        return dt.astimezone(tz)

    @classmethod
    def _filter_upcoming_events(
        cls,
        events: list[dict[str, Any]],
        now: datetime,
    ) -> list[dict[str, Any]]:
        """Return events that have not ended yet, sorted by start time."""
        tz = now.tzinfo
        upcoming = []
        for event in events:
            start = cls._coerce_event_datetime(event.get("start"), tz)
            end = cls._coerce_event_datetime(event.get("end"), tz)
            if end is not None:
                if end > now:
                    upcoming.append(event)
            elif start is None or start >= now:
                upcoming.append(event)

        latest = datetime.max.replace(tzinfo=tz)
        return sorted(
            upcoming,
            key=lambda event: cls._coerce_event_datetime(event.get("start"), tz)
            or latest,
        )

    def _execute_check_availability(self, args: dict) -> ToolResult:
        """Check availability."""
        target_date = parse_date_reference(args.get("date", "today"))
        free_slots = self.multi_google.check_availability(
            date=target_date,
            duration_minutes=args.get("duration_minutes", 30),
            working_hours=(
                args.get("working_hours_start", 9),
                args.get("working_hours_end", 18),
            ),
        )
        return ToolResult(data={
            "date": target_date.strftime("%Y-%m-%d"),
            "duration_minutes": args.get("duration_minutes", 30),
            "free_slot_count": len(free_slots),
            "free_slots": free_slots,
        })

    def _execute_create_calendar_event(
        self,
        args: dict,
        context: "ConversationContext | None" = None,
    ) -> ToolResult:
        """Queue calendar event creation for confirmation."""
        from datetime import timedelta

        from .actions.confirmable import ConfirmableAction

        # Parse date and time
        start_dt = self._parse_event_datetime(
            args.get("date", "today"),
            args.get("time", "12:00"),
        )
        duration = args.get("duration_minutes", 60)
        end_dt = start_dt + timedelta(minutes=duration)

        # Get optional fields
        attendees = args.get("attendees", [])
        location = args.get("location", "")
        description = args.get("description", "")
        account = args.get("account") or PRIMARY_ACCOUNT

        preview = (
            f"*Event:* {args['title']}\n"
            f"*When:* {start_dt.strftime('%Y-%m-%d %I:%M %p')} "
            f"({duration} min)\n"
            f"*Account:* {account}"
        )
        if location:
            preview += f"\n*Location:* {location}"
        if attendees:
            preview += f"\n*Attendees:* {', '.join(attendees)}"
            preview += "\n_(Calendar invites will be sent.)_"
        if description:
            desc_preview = description[:200] + ("..." if len(description) > 200 else "")
            preview += f"\n*Description:* {desc_preview}"

        def execute_event() -> dict[str, Any]:
            event = self.multi_google.create_calendar_event(
                account=account,
                summary=args["title"],
                start=start_dt,
                end=end_dt,
                description=description or None,
                attendees=attendees if attendees else None,
                location=location or None,
                send_notifications=True,
            )
            attendee_msg = ""
            if attendees:
                attendee_msg = f" Calendar invites sent to {len(attendees)} attendee(s)."
            return {
                "success": True,
                "event_id": event.get("id"),
                "html_link": event.get("htmlLink"),
                "message": (
                    f"Created event '{args['title']}' on {start_dt.strftime('%Y-%m-%d')} "
                    f"at {start_dt.strftime('%I:%M %p')}.{attendee_msg}"
                ),
            }

        return self._queue_confirmation(
            ConfirmableAction("Create Calendar Event", preview, execute_event),
            context,
            "Please confirm creating this calendar event.",
        )

    def _execute_update_calendar_event(
        self,
        args: dict,
        context: "ConversationContext | None" = None,
    ) -> ToolResult:
        """Queue calendar event update for confirmation."""
        from datetime import timedelta

        from .actions.confirmable import ConfirmableAction

        account = args.get("account") or PRIMARY_ACCOUNT
        calendar_id = args.get("calendar_id", "primary")
        updates: dict[str, Any] = {}
        preview_lines = [
            f"*Event ID:* {args['event_id']}",
            f"*Account:* {account}",
            f"*Calendar:* {calendar_id}",
        ]

        if args.get("title") is not None:
            updates["summary"] = args["title"]
            preview_lines.append(f"*New title:* {args['title']}")
        if args.get("location") is not None:
            updates["location"] = args["location"]
            preview_lines.append(f"*New location:* {args['location'] or '(blank)'}")
        if args.get("description") is not None:
            updates["description"] = args["description"]
            desc = args["description"][:200] + ("..." if len(args["description"]) > 200 else "")
            preview_lines.append(f"*New description:* {desc or '(blank)'}")
        if args.get("attendees") is not None:
            updates["attendees"] = args["attendees"]
            preview_lines.append(f"*New attendees:* {', '.join(args['attendees']) or '(none)'}")

        if args.get("date") or args.get("time"):
            start_dt = self._parse_event_datetime(
                args.get("date") or "today",
                args.get("time") or "12:00",
            )
            duration = args.get("duration_minutes") or 60
            updates["start"] = start_dt
            updates["end"] = start_dt + timedelta(minutes=duration)
            preview_lines.append(
                f"*New time:* {start_dt.strftime('%Y-%m-%d %I:%M %p')} ({duration} min)"
            )

        if not updates:
            return ToolResult(success=False, error="No calendar event updates were provided.")

        def execute_update() -> dict[str, Any]:
            event = self.multi_google.update_calendar_event(
                account=account,
                event_id=args["event_id"],
                calendar_id=calendar_id,
                send_notifications=args.get("send_notifications", True),
                **updates,
            )
            return {
                "success": True,
                "event_id": event.get("id", args["event_id"]),
                "html_link": event.get("htmlLink"),
                "message": f"Updated calendar event: {event.get('summary', args['event_id'])}",
            }

        return self._queue_confirmation(
            ConfirmableAction("Update Calendar Event", "\n".join(preview_lines), execute_update),
            context,
            "Please confirm updating this calendar event.",
        )

    def _execute_delete_calendar_event(
        self,
        args: dict,
        context: "ConversationContext | None" = None,
    ) -> ToolResult:
        """Queue calendar event deletion for confirmation."""
        from .actions.confirmable import ConfirmableAction

        account = args.get("account") or PRIMARY_ACCOUNT
        calendar_id = args.get("calendar_id", "primary")
        preview = (
            f"*Event ID:* {args['event_id']}\n"
            f"*Account:* {account}\n"
            f"*Calendar:* {calendar_id}\n"
            f"*Send cancellation notifications:* {args.get('send_notifications', True)}"
        )

        def execute_delete() -> dict[str, Any]:
            self.multi_google.delete_calendar_event(
                account=account,
                event_id=args["event_id"],
                calendar_id=calendar_id,
                send_notifications=args.get("send_notifications", True),
            )
            return {
                "success": True,
                "event_id": args["event_id"],
                "message": f"Cancelled calendar event: {args['event_id']}",
            }

        return self._queue_confirmation(
            ConfirmableAction("Cancel Calendar Event", preview, execute_delete),
            context,
            "Please confirm cancelling this calendar event.",
        )

    def _parse_event_datetime(self, date_str: str, time_str: str) -> datetime:
        """Parse date and time strings into a datetime."""
        from .datetime_utils import parse_event_datetime

        return parse_event_datetime(date_str, time_str)

    def _execute_get_unread_counts(self, args: dict) -> ToolResult:
        """Get unread email counts."""
        counts = self.multi_google.get_unread_counts()
        total = sum(c for c in counts.values() if c >= 0)
        return ToolResult(data={
            "total_unread": total,
            "by_account": counts,
        })

    def _execute_create_email_draft(
        self,
        args: dict,
        context: "ConversationContext | None" = None,
    ) -> ToolResult:
        """Queue email draft creation for confirmation."""
        from .actions.email_actions import CreateDraftAction

        account = args.get("account") or PRIMARY_ACCOUNT
        action = CreateDraftAction(
            to=args["to"],
            subject=args["subject"],
            body=args["body"],
            account=account,
        )
        return self._queue_confirmation(
            action,
            context,
            "Please confirm creating this email draft.",
        )

    def _execute_send_email(
        self,
        args: dict,
        context: "ConversationContext | None" = None,
    ) -> ToolResult:
        """Queue an email send behind explicit Slack confirmation."""
        if not ENABLE_DIRECT_EMAIL_SEND:
            return ToolResult(
                success=False,
                error=(
                    "Direct sending is disabled. Use CreateEmailDraftTool and ask the user "
                    "to send from their mailbox after review."
                ),
            )

        if context is None:
            return ToolResult(
                success=False,
                error="Missing conversation context for confirmation-gated send.",
            )

        from .actions.email_actions import SendEmailAction

        action = SendEmailAction(
            to=args["to"],
            subject=args["subject"],
            body=args["body"],
            account=args.get("account") or PRIMARY_ACCOUNT,
            cc=(args.get("cc") or "").strip(),
            bcc=(args.get("bcc") or "").strip(),
        )
        context.pending_action = action
        prompt = action.get_confirmation_prompt()
        return ToolResult(data={
            "requires_confirmation": True,
            "message": "Please confirm sending this email.",
            "confirmation": prompt,
        })

    def _execute_get_github_prs(self, args: dict) -> ToolResult:
        """Get GitHub PRs."""
        prs = self.github_client.get_my_prs(
            state=args.get("state", "open"),
            max_results=args.get("max_results", 10),
        )
        return ToolResult(data={
            "state": args.get("state", "open"),
            "pr_count": len(prs),
            "pull_requests": prs,
        })

    def _execute_get_github_issues(self, args: dict) -> ToolResult:
        """Get GitHub issues."""
        issues = self.github_client.get_my_issues(
            state=args.get("state", "open"),
            max_results=args.get("max_results", 10),
        )
        return ToolResult(data={
            "state": args.get("state", "open"),
            "issue_count": len(issues),
            "issues": issues,
        })

    def _execute_search_github_code(self, args: dict) -> ToolResult:
        """Search GitHub code."""
        repo = args.get("repo")
        if repo:
            results = self.github_client.search_code_in_repo(
                repo=repo,
                query=args["query"],
                max_results=args.get("max_results", 20),
            )
        else:
            results = self.github_client.search_code(
                query=args["query"],
                max_results=args.get("max_results", 20),
            )
        return ToolResult(data={
            "query": args["query"],
            "repo": repo,
            "result_count": len(results),
            "results": results,
        })

    def _execute_create_github_issue(
        self,
        args: dict,
        context: "ConversationContext | None" = None,
    ) -> ToolResult:
        """Queue GitHub issue creation for confirmation."""
        from .actions.github_actions import CreateIssueAction

        action = CreateIssueAction(
            repo=args["repo"],
            title=args["title"],
            body=args.get("body", ""),
            labels=args.get("labels", []),
        )
        return self._queue_confirmation(
            action,
            context,
            "Please confirm creating this GitHub issue.",
        )

    def _execute_find_person(self, args: dict) -> ToolResult:
        """Find person in knowledge graph."""
        people = self.query_engine.find_person(args["query"])
        return ToolResult(data={
            "query": args["query"],
            "result_count": len(people),
            "people": people,
        })

    def _execute_get_person_activity(self, args: dict) -> ToolResult:
        """Get person activity."""
        activity = self.query_engine.get_person_activity(
            person_id=args["person_id"],
            content_types=args.get("content_types"),
            limit=args.get("max_results", 20),
        )
        return ToolResult(data={
            "person_id": args["person_id"],
            "activity_count": len(activity),
            "activity": activity,
        })

    def _execute_get_daily_briefing(self, args: dict) -> ToolResult:
        """Get daily briefing."""
        from ..config import get_user_timezone
        briefing = {
            "date": datetime.now(get_user_timezone()).strftime("%A, %B %d, %Y"),
            "events": [],
            "unread_counts": {},
            "open_prs": [],
            "open_issues": [],
            "overdue_tasks": [],
        }

        try:
            briefing["events"] = self.multi_google.get_all_calendars_today()
        except Exception as e:
            logger.warning(f"Error getting calendar: {e}")

        try:
            briefing["unread_counts"] = self.multi_google.get_unread_counts()
        except Exception as e:
            logger.warning(f"Error getting unread counts: {e}")

        try:
            briefing["open_prs"] = self.github_client.get_my_prs(state="open", max_results=10)
        except Exception as e:
            logger.warning(f"Error getting PRs: {e}")

        try:
            briefing["open_issues"] = self.github_client.get_my_issues(state="open", max_results=10)
        except Exception as e:
            logger.warning(f"Error getting issues: {e}")

        try:
            briefing["overdue_tasks"] = self.todoist_client.list_tasks(filter="overdue")
        except Exception as e:
            logger.error(f"Error getting Todoist overdue tasks: {e}", exc_info=True)

        return ToolResult(data=briefing)

    def _execute_respond_to_user(self, args: dict) -> ToolResult:
        """Handle direct response to user (special case - not really a tool)."""
        return ToolResult(data={"message": args["message"]})

    def _execute_get_todoist_tasks(self, args: dict) -> ToolResult:
        """Get active tasks from Todoist."""
        # Get project ID if project name provided
        project_id = None
        project_name = args.get("project")
        if project_name:
            projects = self.todoist_client.list_projects()
            for p in projects:
                if p["name"].lower() == project_name.lower():
                    project_id = p["id"]
                    break

        tasks = self.todoist_client.list_tasks(
            project_id=project_id,
            filter=args.get("filter"),
        )

        # Get project names for context
        projects = self.todoist_client.list_projects()
        project_map = {p["id"]: p["name"] for p in projects}

        # Format tasks for display
        formatted = []
        for task in tasks:
            proj_name = project_map.get(task.get("project_id"), "Inbox")
            due_str = None
            if task.get("due"):
                due_str = task["due"].get("string") or task["due"].get("date")

            formatted.append({
                "id": task["id"],
                "content": task["content"],
                "project": proj_name,
                "due": due_str,
                "priority": task.get("priority", 1),
                "labels": task.get("labels", []),
                "url": task.get("url"),
            })

        return ToolResult(data={
            "task_count": len(formatted),
            "tasks": formatted,
        })

    def _execute_create_todoist_task(
        self,
        args: dict,
        context: "ConversationContext | None" = None,
    ) -> ToolResult:
        """Queue Todoist task creation for confirmation."""
        from .actions.confirmable import ConfirmableAction

        # Find project ID if project name provided
        project_id = None
        project_name = args.get("project")
        if project_name:
            projects = self.todoist_client.list_projects()
            for p in projects:
                if p["name"].lower() == project_name.lower():
                    project_id = p["id"]
                    break

        preview = f"*Task:* {args['content']}"
        if project_name:
            preview += f"\n*Project:* {project_name}"
        if args.get("due"):
            preview += f"\n*Due:* {args['due']}"
        if args.get("description"):
            desc = args["description"][:200]
            if len(args["description"]) > 200:
                desc += "..."
            preview += f"\n*Description:* {desc}"
        if args.get("labels"):
            preview += f"\n*Labels:* {', '.join(args['labels'])}"

        def execute_task() -> dict[str, Any]:
            task = self.todoist_client.create_task(
                content=args["content"],
                description=args.get("description"),
                project_id=project_id,
                due_string=args.get("due"),
                priority=args.get("priority", 1),
                labels=args.get("labels"),
            )
            return {
                "success": True,
                "task_id": task["id"],
                "content": task["content"],
                "url": task.get("url"),
                "message": f"Task created: {task['content']}",
            }

        return self._queue_confirmation(
            ConfirmableAction("Create Todoist Task", preview, execute_task),
            context,
            "Please confirm creating this Todoist task.",
        )

    def _execute_complete_todoist_task(
        self,
        args: dict,
        context: "ConversationContext | None" = None,
    ) -> ToolResult:
        """Queue Todoist task completion for confirmation."""
        from .actions.confirmable import ConfirmableAction

        task_id = args["task_id"]

        # Get task info first for confirmation message
        try:
            task = self.todoist_client.get_task(task_id)
            task_content = task.get("content", "Unknown task")
        except Exception:
            task_content = "Unknown task"

        def execute_complete() -> dict[str, Any]:
            self.todoist_client.complete_task(task_id)
            return {
                "success": True,
                "task_id": task_id,
                "message": f"Completed: {task_content}",
            }

        return self._queue_confirmation(
            ConfirmableAction(
                "Complete Todoist Task",
                f"*Task:* {task_content}\n*ID:* {task_id}",
                execute_complete,
            ),
            context,
            "Please confirm completing this Todoist task.",
        )

    def _execute_update_todoist_task(
        self,
        args: dict,
        context: "ConversationContext | None" = None,
    ) -> ToolResult:
        """Queue Todoist task update for confirmation."""
        from .actions.confirmable import ConfirmableAction

        updates = {
            "content": args.get("content"),
            "description": args.get("description"),
            "due_string": args.get("due"),
            "priority": args.get("priority"),
            "labels": args.get("labels"),
        }
        updates = {k: v for k, v in updates.items() if v is not None}
        if not updates:
            return ToolResult(success=False, error="No Todoist task updates were provided.")

        try:
            task = self.todoist_client.get_task(args["task_id"])
            task_content = task.get("content", "Unknown task")
        except Exception:
            task_content = "Unknown task"

        preview_lines = [f"*Task:* {task_content}", f"*ID:* {args['task_id']}"]
        if args.get("content") is not None:
            preview_lines.append(f"*New content:* {args['content']}")
        if args.get("description") is not None:
            desc = args["description"][:200] + ("..." if len(args["description"]) > 200 else "")
            preview_lines.append(f"*New description:* {desc or '(blank)'}")
        if args.get("due") is not None:
            preview_lines.append(f"*New due:* {args['due']}")
        if args.get("priority") is not None:
            preview_lines.append(f"*New priority:* {args['priority']}")
        if args.get("labels") is not None:
            preview_lines.append(f"*New labels:* {', '.join(args['labels']) or '(none)'}")

        def execute_update() -> dict[str, Any]:
            task = self.todoist_client.update_task(args["task_id"], **updates)
            return {
                "success": True,
                "task_id": task["id"],
                "content": task["content"],
                "url": task.get("url"),
                "message": f"Updated Todoist task: {task['content']}",
            }

        return self._queue_confirmation(
            ConfirmableAction("Update Todoist Task", "\n".join(preview_lines), execute_update),
            context,
            "Please confirm updating this Todoist task.",
        )

    def _execute_add_todoist_comment(
        self,
        args: dict,
        context: "ConversationContext | None" = None,
    ) -> ToolResult:
        """Queue Todoist task comment for confirmation."""
        from .actions.confirmable import ConfirmableAction

        comment = args["content"][:300] + ("..." if len(args["content"]) > 300 else "")

        def execute_comment() -> dict[str, Any]:
            created = self.todoist_client.add_comment(args["task_id"], args["content"])
            return {
                "success": True,
                "comment_id": created["id"],
                "task_id": args["task_id"],
                "message": "Added Todoist task comment.",
            }

        return self._queue_confirmation(
            ConfirmableAction(
                "Add Todoist Comment",
                f"*Task ID:* {args['task_id']}\n*Comment:*\n{comment}",
                execute_comment,
            ),
            context,
            "Please confirm adding this Todoist comment.",
        )

    def _execute_reopen_todoist_task(
        self,
        args: dict,
        context: "ConversationContext | None" = None,
    ) -> ToolResult:
        """Queue reopening a Todoist task for confirmation."""
        from .actions.confirmable import ConfirmableAction

        def execute_reopen() -> dict[str, Any]:
            self.todoist_client.reopen_task(args["task_id"])
            return {
                "success": True,
                "task_id": args["task_id"],
                "message": f"Reopened Todoist task: {args['task_id']}",
            }

        return self._queue_confirmation(
            ConfirmableAction(
                "Reopen Todoist Task",
                f"*Task ID:* {args['task_id']}",
                execute_reopen,
            ),
            context,
            "Please confirm reopening this Todoist task.",
        )

    def _execute_search_notion(self, args: dict) -> ToolResult:
        """Search Notion pages and databases."""
        results = self.notion_client.search(
            query=args["query"],
            max_results=args.get("max_results", 10),
        )

        # Format results for display
        formatted = []
        for item in results:
            formatted.append({
                "id": item["id"],
                "type": item.get("object", "page"),
                "title": item.get("title", "Untitled"),
                "url": item.get("url"),
                "last_edited": item.get("last_edited_time"),
            })

        return ToolResult(data={
            "query": args["query"],
            "result_count": len(formatted),
            "results": formatted,
        })

    def _execute_create_notion_page(
        self,
        args: dict,
        context: "ConversationContext | None" = None,
    ) -> ToolResult:
        """Queue Notion page creation for confirmation."""
        from .actions.confirmable import ConfirmableAction

        # Build properties with title
        properties = args.get("properties", {})
        # Add title property (Notion databases typically use "Name" or "Title")
        properties["Name"] = {
            "title": [{"text": {"content": args["title"]}}]
        }

        def execute_page() -> dict[str, Any]:
            page = self.notion_client.create_page(
                database_id=args["database_id"],
                properties=properties,
            )
            return {
                "success": True,
                "page_id": page["id"],
                "url": page.get("url"),
                "title": args["title"],
                "message": f"Page created: {page.get('url', page['id'])}",
            }

        return self._queue_confirmation(
            ConfirmableAction(
                "Create Notion Page",
                f"*Title:* {args['title']}\n*Database:* {args['database_id']}",
                execute_page,
            ),
            context,
            "Please confirm creating this Notion page.",
        )

    def _execute_add_notion_comment(
        self,
        args: dict,
        context: "ConversationContext | None" = None,
    ) -> ToolResult:
        """Queue Notion comment creation for confirmation."""
        from .actions.confirmable import ConfirmableAction

        comment_preview = args["content"][:300]
        if len(args["content"]) > 300:
            comment_preview += "..."

        def execute_comment() -> dict[str, Any]:
            comment = self.notion_client.add_comment(
                page_id=args["page_id"],
                content=args["content"],
            )
            return {
                "success": True,
                "comment_id": comment["id"],
                "page_id": args["page_id"],
                "message": "Comment added successfully",
            }

        return self._queue_confirmation(
            ConfirmableAction(
                "Add Notion Comment",
                f"*Page:* {args['page_id']}\n*Comment:*\n{comment_preview}",
                execute_comment,
            ),
            context,
            "Please confirm adding this Notion comment.",
        )

    def _execute_add_google_doc_comment(
        self,
        args: dict,
        context: "ConversationContext | None" = None,
    ) -> ToolResult:
        """Queue adding a Google Doc comment for confirmation."""
        from .actions.confirmable import ConfirmableAction

        account = args.get("account") or PRIMARY_ACCOUNT
        comment_preview = args["content"][:300] + ("..." if len(args["content"]) > 300 else "")
        preview = (
            f"*Document:* {args['document_id']}\n"
            f"*Account:* {account}\n"
            f"*Comment:*\n{comment_preview}"
        )
        if args.get("quoted_text"):
            preview += f"\n*Anchor text:* {args['quoted_text'][:200]}"

        def execute_comment() -> dict[str, Any]:
            comment = self.multi_google.add_doc_comment(
                account=account,
                document_id=args["document_id"],
                content=args["content"],
                quoted_text=args.get("quoted_text"),
            )
            return {
                "success": True,
                "comment_id": comment["id"],
                "document_id": args["document_id"],
                "message": "Added Google Doc comment.",
            }

        return self._queue_confirmation(
            ConfirmableAction("Add Google Doc Comment", preview, execute_comment),
            context,
            "Please confirm adding this Google Doc comment.",
        )

    def _execute_reply_google_doc_comment(
        self,
        args: dict,
        context: "ConversationContext | None" = None,
    ) -> ToolResult:
        """Queue replying to a Google Doc comment for confirmation."""
        from .actions.confirmable import ConfirmableAction

        account = args.get("account") or PRIMARY_ACCOUNT
        reply_preview = args["content"][:300] + ("..." if len(args["content"]) > 300 else "")

        def execute_reply() -> dict[str, Any]:
            reply = self.multi_google.reply_to_doc_comment(
                account=account,
                document_id=args["document_id"],
                comment_id=args["comment_id"],
                content=args["content"],
            )
            return {
                "success": True,
                "reply_id": reply["id"],
                "comment_id": args["comment_id"],
                "message": "Replied to Google Doc comment.",
            }

        return self._queue_confirmation(
            ConfirmableAction(
                "Reply Google Doc Comment",
                (
                    f"*Document:* {args['document_id']}\n"
                    f"*Comment:* {args['comment_id']}\n"
                    f"*Account:* {account}\n"
                    f"*Reply:*\n{reply_preview}"
                ),
                execute_reply,
            ),
            context,
            "Please confirm replying to this Google Doc comment.",
        )

    def _execute_resolve_google_doc_comment(
        self,
        args: dict,
        context: "ConversationContext | None" = None,
    ) -> ToolResult:
        """Queue resolving a Google Doc comment for confirmation."""
        from .actions.confirmable import ConfirmableAction

        account = args.get("account") or PRIMARY_ACCOUNT

        def execute_resolve() -> dict[str, Any]:
            comment = self.multi_google.resolve_doc_comment(
                account=account,
                document_id=args["document_id"],
                comment_id=args["comment_id"],
            )
            return {
                "success": True,
                "comment_id": comment["id"],
                "message": "Resolved Google Doc comment.",
            }

        return self._queue_confirmation(
            ConfirmableAction(
                "Resolve Google Doc Comment",
                (
                    f"*Document:* {args['document_id']}\n"
                    f"*Comment:* {args['comment_id']}\n"
                    f"*Account:* {account}"
                ),
                execute_resolve,
            ),
            context,
            "Please confirm resolving this Google Doc comment.",
        )

    def _execute_get_proactive_settings(
        self,
        args: dict,
        context: "ConversationContext | None" = None,
    ) -> ToolResult:
        """Get proactive settings for the current Slack user."""
        if context is None:
            return ToolResult(success=False, error="Missing conversation context.")
        settings = self.proactive_settings.get(context.user_id)
        return ToolResult(data=settings.to_dict())

    def _execute_update_proactive_settings(
        self,
        args: dict,
        context: "ConversationContext | None" = None,
    ) -> ToolResult:
        """Queue proactive settings update for confirmation."""
        from .actions.confirmable import ConfirmableAction

        if context is None:
            return ToolResult(success=False, error="Missing conversation context.")

        allowed = {
            "calendar_reminders_enabled",
            "email_alerts_enabled",
            "daily_briefing_enabled",
            "reminder_minutes_before",
            "briefing_hour",
            "briefing_minute",
            "briefing_days",
            "important_contacts",
            "alert_keywords",
            "quiet_hours_start",
            "quiet_hours_end",
        }
        updates = {k: v for k, v in args.items() if k in allowed and v is not None}
        if not updates:
            return ToolResult(success=False, error="No proactive setting updates were provided.")

        def _validate_hour(name: str) -> None:
            if name in updates and not 0 <= int(updates[name]) <= 23:
                raise ValueError(f"{name} must be between 0 and 23")

        try:
            _validate_hour("briefing_hour")
            _validate_hour("quiet_hours_start")
            _validate_hour("quiet_hours_end")
            if "briefing_minute" in updates and not 0 <= int(updates["briefing_minute"]) <= 59:
                raise ValueError("briefing_minute must be between 0 and 59")
            if "reminder_minutes_before" in updates and int(updates["reminder_minutes_before"]) < 0:
                raise ValueError("reminder_minutes_before must be non-negative")
            if "briefing_days" in updates and any(day < 0 or day > 6 for day in updates["briefing_days"]):
                raise ValueError("briefing_days values must be between 0 and 6")
        except ValueError as e:
            return ToolResult(success=False, error=str(e))

        preview = "\n".join(f"*{key}:* {value}" for key, value in updates.items())

        def execute_update() -> dict[str, Any]:
            settings = self.proactive_settings.get(context.user_id)
            for key, value in updates.items():
                setattr(settings, key, value)
            self.proactive_settings.save(settings)
            return {
                "success": True,
                "settings": settings.to_dict(),
                "message": "Updated proactive notification settings.",
            }

        return self._queue_confirmation(
            ConfirmableAction("Update Proactive Settings", preview, execute_update),
            context,
            "Please confirm updating your proactive notification settings.",
        )

    def _execute_search_zotero_papers(self, args: dict) -> ToolResult:
        """Search papers in Zotero library."""
        results = self.zotero_client.search_items(
            query=args["query"],
            max_results=args.get("max_results", 10),
        )

        # Format results for display
        formatted = []
        for item in results:
            formatted.append({
                "key": item["key"],
                "title": item["title"],
                "authors": item.get("authors", []),
                "year": item.get("year"),
                "journal": item.get("journal"),
                "doi": item.get("doi"),
                "tags": item.get("tags", []),
            })

        return ToolResult(data={
            "query": args["query"],
            "result_count": len(formatted),
            "papers": formatted,
        })

    def _execute_get_zotero_paper(self, args: dict) -> ToolResult:
        """Get full details of a Zotero paper."""
        item = self.zotero_client.get_item(args["item_key"])

        # Get notes for this item
        notes = self.zotero_client.get_item_notes(args["item_key"])
        note_texts = [n.get("note", "") for n in notes]

        return ToolResult(data={
            "key": item["key"],
            "title": item["title"],
            "abstract": item.get("abstract", ""),
            "authors": item.get("authors", []),
            "year": item.get("year"),
            "journal": item.get("journal"),
            "volume": item.get("volume"),
            "issue": item.get("issue"),
            "pages": item.get("pages"),
            "doi": item.get("doi"),
            "url": item.get("url"),
            "tags": item.get("tags", []),
            "notes": note_texts,
            "date_added": item.get("date_added"),
        })

    def _execute_list_recent_papers(self, args: dict) -> ToolResult:
        """List recently added papers."""
        items = self.zotero_client.get_recent_items(days=args.get("days", 7))

        # Limit results
        max_results = args.get("max_results", 20)
        items = items[:max_results]

        formatted = []
        for item in items:
            formatted.append({
                "key": item["key"],
                "title": item["title"],
                "authors": item.get("authors", []),
                "year": item.get("year"),
                "date_added": item.get("date_added"),
                "tags": item.get("tags", []),
            })

        return ToolResult(data={
            "days": args.get("days", 7),
            "paper_count": len(formatted),
            "papers": formatted,
        })

    def _execute_search_papers_by_tag(self, args: dict) -> ToolResult:
        """Search papers by tag."""
        items = self.zotero_client.get_items_by_tag(
            tag=args["tag"],
            max_results=args.get("max_results", 20),
        )

        formatted = []
        for item in items:
            formatted.append({
                "key": item["key"],
                "title": item["title"],
                "authors": item.get("authors", []),
                "year": item.get("year"),
                "tags": item.get("tags", []),
            })

        return ToolResult(data={
            "tag": args["tag"],
            "paper_count": len(formatted),
            "papers": formatted,
        })

    def _execute_get_zotero_collection(self, args: dict) -> ToolResult:
        """Get papers in a Zotero collection."""
        collection = self.zotero_client.get_collection_by_name(args["collection_name"])

        if not collection:
            return ToolResult(
                success=False,
                error=f"Collection '{args['collection_name']}' not found",
            )

        items = self.zotero_client.get_collection_items(
            collection_key=collection["key"],
            max_results=args.get("max_results", 50),
        )

        formatted = []
        for item in items:
            formatted.append({
                "key": item["key"],
                "title": item["title"],
                "authors": item.get("authors", []),
                "year": item.get("year"),
                "tags": item.get("tags", []),
            })

        return ToolResult(data={
            "collection": collection["name"],
            "paper_count": len(formatted),
            "papers": formatted,
        })

    def _execute_add_zotero_paper(
        self,
        args: dict,
        context: "ConversationContext | None" = None,
    ) -> ToolResult:
        """Queue adding a paper to Zotero for confirmation."""
        from .actions.confirmable import ConfirmableAction

        identifier = args["identifier"].strip()
        collection = args.get("collection") or ZOTERO_DEFAULT_COLLECTION

        # Determine if it's a DOI or URL
        is_doi = (
            identifier.startswith("10.") or
            "doi.org" in identifier or
            identifier.lower().startswith("doi:")
        )

        def execute_add() -> dict[str, Any]:
            if is_doi:
                item = self.zotero_client.add_item_by_doi(identifier, collection)
            else:
                item = self.zotero_client.add_item_by_url(identifier, collection)

            return {
                "success": True,
                "key": item["key"],
                "title": item["title"],
                "collection": collection,
                "message": f"Paper added to Zotero: {item['title']}",
            }

        return self._queue_confirmation(
            ConfirmableAction(
                "Add Zotero Paper",
                f"*Identifier:* {identifier}\n*Collection:* {collection or '(default)'}",
                execute_add,
            ),
            context,
            "Please confirm adding this paper to Zotero.",
        )


class AgentExecutor:
    """Executes agent loop with Claude's native tool calling."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        user_memory: "UserMemory | None" = None,
    ):
        """Initialize the agent executor.

        Args:
            api_key: Anthropic API key.
            model: Model to use for the agent.
            user_memory: Optional user memory for context injection.
        """
        self.api_key = api_key or ANTHROPIC_API_KEY
        self.model = model or AGENT_MODEL
        self.user_memory = user_memory

        if not self.api_key:
            raise ValueError("Anthropic API key is required")

        self._client = Anthropic(api_key=self.api_key)
        self._tool_executor = ToolExecutor()
        self._tool_schemas = get_tool_schemas()
        self.trace_logger = get_trace_logger()

    def _create_message(self, **kwargs):
        """Call Anthropic messages.create with structured tracing."""
        started = time.perf_counter()
        try:
            response = self._client.messages.create(**kwargs)
            self.trace_logger.log_model_call(
                caller="agent.executor",
                model=kwargs.get("model", self.model),
                operation="messages.create",
                duration_ms=(time.perf_counter() - started) * 1000,
                success=True,
                usage=model_usage(response),
                stop_reason=getattr(response, "stop_reason", None),
            )
            return response
        except Exception as e:
            self.trace_logger.log_model_call(
                caller="agent.executor",
                model=kwargs.get("model", self.model),
                operation="messages.create",
                duration_ms=(time.perf_counter() - started) * 1000,
                success=False,
                error=e,
            )
            raise

    def _execute_tool_traced(self, tool_name: str, tool_input: dict[str, Any], context):
        """Execute a tool with structured tracing."""
        started = time.perf_counter()
        try:
            result = self._tool_executor.execute(
                tool_name,
                tool_input,
                context=context,
            )
            self.trace_logger.log_tool_call(
                caller="agent.executor",
                tool_name=tool_name,
                duration_ms=(time.perf_counter() - started) * 1000,
                success=result.success,
                input_keys=list(tool_input.keys()) if isinstance(tool_input, dict) else [],
                result_preview=result.to_content()[:300],
            )
            return result
        except Exception as e:
            self.trace_logger.log_tool_call(
                caller="agent.executor",
                tool_name=tool_name,
                duration_ms=(time.perf_counter() - started) * 1000,
                success=False,
                input_keys=list(tool_input.keys()) if isinstance(tool_input, dict) else [],
                error=e,
            )
            raise

    def run(
        self,
        message: str,
        context: "ConversationContext",
        max_iterations: int = MAX_ITERATIONS,
    ) -> ExecutionResult:
        """Run the agent loop until a response is generated.

        Args:
            message: User message.
            context: Conversation context.
            max_iterations: Maximum number of tool-calling iterations.

        Returns:
            ExecutionResult with response and tool call history.
        """
        # Build system prompt with current date and user context
        current_date = datetime.now(get_user_timezone()).strftime(
            "%A, %B %d, %Y %I:%M %p %Z"
        )
        email_send_policy = (
            "If you need to send an email, use SendEmailTool which requires explicit Slack confirmation."
            if ENABLE_DIRECT_EMAIL_SEND
            else "Email sending is disabled in this runtime. Create drafts for review instead."
        )
        system = SYSTEM_PROMPT.format(
            current_date=current_date,
            email_send_policy=email_send_policy,
        )

        # Inject user memory context if available
        if self.user_memory:
            try:
                # General user context
                user_context = self.user_memory.get_context_summary(context.user_id)

                # Also search for memories relevant to this specific message
                relevant_memories = self.user_memory.search_memories(
                    context.user_id,
                    message[:200],  # First 200 chars of message
                    limit=3
                )
                if relevant_memories and relevant_memories.get("results"):
                    if user_context:
                        user_context += "\n\nRelevant past context:"
                    else:
                        user_context = "Relevant past context:"
                    for mem in relevant_memories["results"]:
                        memory_text = mem.get("memory", "")
                        if memory_text:
                            user_context += f"\n- {memory_text}"

                if user_context:
                    system += f"\n\nUser context:\n{user_context}"
            except Exception as e:
                logger.warning(f"Failed to get user context: {e}")

        # Build messages from conversation history
        messages = self._build_messages(context, message)

        tool_calls_history = []
        iterations = 0

        while iterations < max_iterations:
            iterations += 1

            try:
                # Call Claude with tools
                response = self._create_message(
                    model=self.model,
                    max_tokens=4096,
                    system=system,
                    tools=self._tool_schemas,
                    messages=messages,
                )

                # Check stop reason
                if response.stop_reason == "end_turn":
                    # Extract text response
                    text_response = self._extract_text_response(response)
                    # Extract memories from conversation
                    self._extract_memories(context, message, text_response)
                    return ExecutionResult(
                        response=text_response,
                        tool_calls=tool_calls_history,
                        iterations=iterations,
                    )

                elif response.stop_reason == "tool_use":
                    # Process tool calls
                    tool_results = []

                    for content in response.content:
                        if content.type == "tool_use":
                            tool_name = content.name
                            tool_input = content.input
                            tool_id = content.id

                            logger.info(f"Executing tool: {tool_name} with {tool_input}")

                            # Check for RespondToUserTool (special case)
                            if tool_name == "RespondToUserTool":
                                final_response = tool_input.get("message", "")
                                # Extract memories from conversation
                                self._extract_memories(context, message, final_response)
                                return ExecutionResult(
                                    response=final_response,
                                    tool_calls=tool_calls_history,
                                    iterations=iterations,
                                )

                            # Execute the tool
                            result = self._execute_tool_traced(
                                tool_name,
                                tool_input,
                                context=context,
                            )

                            # Record tool call
                            tool_calls_history.append({
                                "tool": tool_name,
                                "input": tool_input,
                                "result": result.to_content()[:500],  # Truncate for history
                                "success": result.success,
                            })

                            if (
                                result.success
                                and isinstance(result.data, dict)
                                and result.data.get("requires_confirmation")
                            ):
                                confirmation = result.data.get("confirmation", {})
                                return ExecutionResult(
                                    response=confirmation.get(
                                        "text", result.data.get(
                                            "message", "Please confirm this action."
                                        )
                                    ),
                                    tool_calls=tool_calls_history,
                                    iterations=iterations,
                                    response_blocks=confirmation.get("blocks"),
                                )

                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": result.to_content(),
                            })

                    # Add assistant response and tool results to messages
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": tool_results})

                else:
                    # Unexpected stop reason
                    text_response = self._extract_text_response(response)
                    return ExecutionResult(
                        response=text_response or "I encountered an issue processing your request.",
                        tool_calls=tool_calls_history,
                        iterations=iterations,
                    )

            except Exception as e:
                logger.error(f"Error in agent loop: {e}", exc_info=True)
                return ExecutionResult(
                    response=f"I encountered an error: {str(e)}",
                    tool_calls=tool_calls_history,
                    iterations=iterations,
                    success=False,
                    error=str(e),
                )

        # Max iterations reached
        return ExecutionResult(
            response="I reached the maximum number of steps. Here's what I found so far.",
            tool_calls=tool_calls_history,
            iterations=iterations,
            success=False,
            error="Max iterations reached",
        )

    def run_streaming(
        self,
        message: str,
        context: "ConversationContext",
        max_iterations: int = MAX_ITERATIONS,
    ) -> Generator[StreamEvent, None, ExecutionResult]:
        """Run the agent loop with streaming, yielding events as they occur.

        Args:
            message: User message.
            context: Conversation context.
            max_iterations: Maximum number of tool-calling iterations.

        Yields:
            StreamEvent objects for text chunks, tool executions, and status updates.

        Returns:
            ExecutionResult with final response and tool call history.
        """
        # Build system prompt with current date and user context
        current_date = datetime.now(get_user_timezone()).strftime(
            "%A, %B %d, %Y %I:%M %p %Z"
        )
        email_send_policy = (
            "If you need to send an email, use SendEmailTool which requires explicit Slack confirmation."
            if ENABLE_DIRECT_EMAIL_SEND
            else "Email sending is disabled in this runtime. Create drafts for review instead."
        )
        system = SYSTEM_PROMPT.format(
            current_date=current_date,
            email_send_policy=email_send_policy,
        )

        # Inject user memory context if available
        if self.user_memory:
            try:
                # General user context
                user_context = self.user_memory.get_context_summary(context.user_id)

                # Also search for memories relevant to this specific message
                relevant_memories = self.user_memory.search_memories(
                    context.user_id,
                    message[:200],  # First 200 chars of message
                    limit=3
                )
                if relevant_memories and relevant_memories.get("results"):
                    if user_context:
                        user_context += "\n\nRelevant past context:"
                    else:
                        user_context = "Relevant past context:"
                    for mem in relevant_memories["results"]:
                        memory_text = mem.get("memory", "")
                        if memory_text:
                            user_context += f"\n- {memory_text}"

                if user_context:
                    system += f"\n\nUser context:\n{user_context}"
            except Exception as e:
                logger.warning(f"Failed to get user context: {e}")

        # Build messages from conversation history
        messages = self._build_messages(context, message)

        tool_calls_history = []
        iterations = 0
        accumulated_text = ""

        while iterations < max_iterations:
            iterations += 1

            try:
                # Use streaming API
                model_call_started = time.perf_counter()
                with self._client.messages.stream(
                    model=self.model,
                    max_tokens=4096,
                    system=system,
                    tools=self._tool_schemas,
                    messages=messages,
                ) as stream:
                    current_text = ""
                    tool_uses = []

                    for event in stream:
                        # Handle different event types
                        if event.type == "content_block_start":
                            if hasattr(event, "content_block"):
                                block = event.content_block
                                if block.type == "tool_use":
                                    # Tool use starting
                                    tool_uses.append({
                                        "id": block.id,
                                        "name": block.name,
                                        "input": {},
                                    })
                                    yield StreamEvent(
                                        event_type=StreamEventType.TOOL_START,
                                        tool_name=block.name,
                                        iteration=iterations,
                                    )

                        elif event.type == "content_block_delta":
                            if hasattr(event, "delta"):
                                delta = event.delta
                                if delta.type == "text_delta":
                                    # Text chunk received
                                    text_chunk = delta.text
                                    current_text += text_chunk
                                    accumulated_text += text_chunk
                                    yield StreamEvent(
                                        event_type=StreamEventType.TEXT_DELTA,
                                        data=text_chunk,
                                        iteration=iterations,
                                    )
                                elif delta.type == "input_json_delta":
                                    # Tool input JSON chunk
                                    if tool_uses:
                                        # Accumulate input JSON
                                        pass  # Input is accumulated by the SDK

                        elif event.type == "content_block_stop":
                            if current_text:
                                yield StreamEvent(
                                    event_type=StreamEventType.TEXT_DONE,
                                    data=current_text,
                                    iteration=iterations,
                                )

                    # Get the final message
                    response = stream.get_final_message()
                self.trace_logger.log_model_call(
                    caller="agent.executor",
                    model=self.model,
                    operation="messages.stream",
                    duration_ms=(time.perf_counter() - model_call_started) * 1000,
                    success=True,
                    usage=model_usage(response),
                    stop_reason=getattr(response, "stop_reason", None),
                )
                model_call_started = None

                # Process the complete response
                if response.stop_reason == "end_turn":
                    # Extract final text
                    final_text = self._extract_text_response(response)
                    # Extract memories from conversation
                    self._extract_memories(context, message, final_text)
                    yield StreamEvent(
                        event_type=StreamEventType.DONE,
                        data=final_text,
                        iteration=iterations,
                    )
                    return ExecutionResult(
                        response=final_text,
                        tool_calls=tool_calls_history,
                        iterations=iterations,
                    )

                elif response.stop_reason == "tool_use":
                    # Process tool calls
                    tool_results = []

                    for content in response.content:
                        if content.type == "tool_use":
                            tool_name = content.name
                            tool_input = content.input
                            tool_id = content.id

                            logger.info(f"Executing tool: {tool_name}")

                            # Check for RespondToUserTool (special case)
                            if tool_name == "RespondToUserTool":
                                response_text = tool_input.get("message", "")
                                # Extract memories from conversation
                                self._extract_memories(context, message, response_text)
                                yield StreamEvent(
                                    event_type=StreamEventType.DONE,
                                    data=response_text,
                                    iteration=iterations,
                                )
                                return ExecutionResult(
                                    response=response_text,
                                    tool_calls=tool_calls_history,
                                    iterations=iterations,
                                )

                            # Yield thinking status
                            yield StreamEvent(
                                event_type=StreamEventType.THINKING,
                                data=f"Using {tool_name}...",
                                tool_name=tool_name,
                                iteration=iterations,
                            )

                            # Execute the tool
                            result = self._execute_tool_traced(
                                tool_name,
                                tool_input,
                                context=context,
                            )

                            # Record tool call
                            tool_calls_history.append({
                                "tool": tool_name,
                                "input": tool_input,
                                "result": result.to_content()[:500],
                                "success": result.success,
                            })

                            if (
                                result.success
                                and isinstance(result.data, dict)
                                and result.data.get("requires_confirmation")
                            ):
                                confirmation = result.data.get("confirmation", {})
                                response_text = confirmation.get(
                                    "text", result.data.get(
                                        "message", "Please confirm this action."
                                    )
                                )
                                yield StreamEvent(
                                    event_type=StreamEventType.DONE,
                                    data=response_text,
                                    iteration=iterations,
                                )
                                return ExecutionResult(
                                    response=response_text,
                                    tool_calls=tool_calls_history,
                                    iterations=iterations,
                                    response_blocks=confirmation.get("blocks"),
                                )

                            # Yield tool completion
                            yield StreamEvent(
                                event_type=StreamEventType.TOOL_DONE,
                                tool_name=tool_name,
                                tool_input=tool_input,
                                tool_result=result.to_content()[:200],
                                iteration=iterations,
                            )

                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": result.to_content(),
                            })

                    # Add assistant response and tool results to messages
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": tool_results})

                else:
                    # Unexpected stop reason
                    final_text = self._extract_text_response(response)
                    yield StreamEvent(
                        event_type=StreamEventType.DONE,
                        data=final_text or "I encountered an issue.",
                        iteration=iterations,
                    )
                    return ExecutionResult(
                        response=final_text or "I encountered an issue processing your request.",
                        tool_calls=tool_calls_history,
                        iterations=iterations,
                    )

            except Exception as e:
                if "model_call_started" in locals() and model_call_started is not None:
                    self.trace_logger.log_model_call(
                        caller="agent.executor",
                        model=self.model,
                        operation="messages.stream",
                        duration_ms=(time.perf_counter() - model_call_started) * 1000,
                        success=False,
                        error=e,
                    )
                logger.error(f"Error in streaming agent loop: {e}", exc_info=True)
                yield StreamEvent(
                    event_type=StreamEventType.ERROR,
                    error=str(e),
                    iteration=iterations,
                )
                return ExecutionResult(
                    response=f"I encountered an error: {str(e)}",
                    tool_calls=tool_calls_history,
                    iterations=iterations,
                    success=False,
                    error=str(e),
                )

        # Max iterations reached
        yield StreamEvent(
            event_type=StreamEventType.ERROR,
            error="Max iterations reached",
            iteration=iterations,
        )
        return ExecutionResult(
            response="I reached the maximum number of steps. Here's what I found so far.",
            tool_calls=tool_calls_history,
            iterations=iterations,
            success=False,
            error="Max iterations reached",
        )

    def _build_messages(
        self,
        context: "ConversationContext",
        current_message: str,
    ) -> list[dict]:
        """Build message list from context and current message.

        Args:
            context: Conversation context with history.
            current_message: Current user message.

        Returns:
            List of messages for Claude API.
        """
        messages = []

        # Add conversation history (limit to recent exchanges)
        if context.history:
            for msg in context.history[-6:]:  # Last 3 exchanges
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})

        # Add current message
        if not messages or messages[-1] != {"role": "user", "content": current_message}:
            messages.append({"role": "user", "content": current_message})

        return messages

    def _extract_text_response(self, response) -> str:
        """Extract text content from Claude response.

        Args:
            response: Claude API response.

        Returns:
            Text content as string.
        """
        for content in response.content:
            if content.type == "text":
                return content.text
        return ""

    def _extract_memories(
        self,
        context: "ConversationContext",
        user_message: str,
        assistant_response: str,
    ) -> None:
        """Extract and store memories from the conversation.

        Uses Mem0 to automatically identify and store relevant memories
        from the user message and assistant response.

        Args:
            context: Conversation context.
            user_message: The user's message.
            assistant_response: The assistant's response.
        """
        if not self.user_memory:
            return

        try:
            # Build messages for memory extraction
            messages = []

            # Include recent history for context (last 10 turns)
            if context.history:
                for msg in context.history[-10:]:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role in ("user", "assistant") and content:
                        messages.append({"role": role, "content": content})

            # Add current exchange
            if not messages or messages[-1] != {"role": "user", "content": user_message}:
                messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": assistant_response})

            # Auto-extract memories via Mem0
            self.user_memory.add_from_conversation(context.user_id, messages)
        except Exception as e:
            logger.debug(f"Memory extraction failed: {e}")
