"""Agent executor for multi-step tool calling with Claude.

This module implements an agentic loop that uses Claude's native tool calling
to dynamically select and execute tools until a final response is generated.
Supports both synchronous and streaming execution modes.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Generator, Iterator

from anthropic import Anthropic

from ..config import ANTHROPIC_API_KEY
from .tools import (
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
    SendEmailTool,
    GetGitHubPRsTool,
    GetGitHubIssuesTool,
    SearchGitHubCodeTool,
    CreateGitHubIssueTool,
    FindPersonTool,
    GetPersonActivityTool,
    GetDailyBriefingTool,
    RespondToUserTool,
)

if TYPE_CHECKING:
    from .conversation import ConversationContext
    from .user_memory import UserMemory

logger = logging.getLogger(__name__)

# Maximum number of tool calling iterations to prevent infinite loops
MAX_ITERATIONS = 10

# Model to use for agent
AGENT_MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = """You are Hani's personal AI assistant with access to tools for managing emails, calendar, GitHub, Todoist tasks, and searching a personal knowledge graph.

You have access to tools that let you:
- Search across all indexed data (emails, documents, calendar events, Slack messages, GitHub)
- Search and manage emails across multiple Google accounts
- Send emails and create drafts
- Check calendar events and availability
- Search GitHub code, issues, and PRs
- Create GitHub issues
- Get and create Todoist tasks, mark tasks complete
- Search Notion pages and databases
- Get daily briefings

Guidelines:
1. Be conversational and helpful. You can chat naturally without using tools for greetings and simple questions.
2. Use the RespondToUserTool when you want to reply directly to the user.
3. For data requests, use the appropriate tool to fetch real information.
4. If a task requires multiple steps, execute them in sequence.
5. Always provide clear, concise responses.
6. Never make up information - only use data from tools.
7. For actions like creating issues, drafts, or sending emails, confirm the details before executing.
8. ALWAYS confirm with the user before sending an email - show them the content first.
9. When the user asks about "tasks" or "to-dos", use the Todoist tools (GetTodoistTasksTool, CreateTodoistTaskTool).

Current date: {current_date}
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

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
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

            return handler(arguments)

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
            return ToolResult(success=False, error=str(e))

    def _execute_semantic_search(self, args: dict) -> ToolResult:
        """Execute semantic search."""
        results = self.semantic_indexer.search(
            query=args["query"],
            top_k=args.get("max_results", 10),
        )
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
        events = self.multi_google.get_all_calendars_for_date(target_date)
        return ToolResult(data={
            "date": target_date.strftime("%Y-%m-%d"),
            "event_count": len(events),
            "events": events,
        })

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

    def _execute_get_unread_counts(self, args: dict) -> ToolResult:
        """Get unread email counts."""
        counts = self.multi_google.get_unread_counts()
        total = sum(c for c in counts.values() if c >= 0)
        return ToolResult(data={
            "total_unread": total,
            "by_account": counts,
        })

    def _execute_create_email_draft(self, args: dict) -> ToolResult:
        """Create email draft."""
        account = args.get("account", "arc")
        draft = self.multi_google.create_draft(
            account=account,
            to=args["to"],
            subject=args["subject"],
            body=args["body"],
        )
        return ToolResult(data={
            "draft_id": draft.get("id"),
            "account": account,
            "to": args["to"],
            "subject": args["subject"],
            "message": f"Draft created in {account} account",
        })

    def _execute_send_email(self, args: dict) -> ToolResult:
        """Send an email."""
        account = args.get("account", "arc")
        result = self.multi_google.send_email(
            account=account,
            to=args["to"],
            subject=args["subject"],
            body=args["body"],
            cc=args.get("cc"),
            bcc=args.get("bcc"),
        )
        return ToolResult(data={
            "message_id": result.get("id"),
            "thread_id": result.get("threadId"),
            "account": account,
            "to": args["to"],
            "subject": args["subject"],
            "message": f"Email sent successfully from {account} account",
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

    def _execute_create_github_issue(self, args: dict) -> ToolResult:
        """Create GitHub issue."""
        issue = self.github_client.create_issue(
            repo=args["repo"],
            title=args["title"],
            body=args.get("body", ""),
            labels=args.get("labels", []),
        )
        return ToolResult(data={
            "issue_number": issue.get("number"),
            "url": issue.get("html_url"),
            "title": args["title"],
            "message": f"Issue created: {issue.get('html_url')}",
        })

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
        briefing = {
            "date": datetime.now(timezone.utc).strftime("%A, %B %d, %Y"),
            "events": [],
            "unread_counts": {},
            "open_prs": [],
            "open_issues": [],
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

    def _execute_create_todoist_task(self, args: dict) -> ToolResult:
        """Create a new task in Todoist."""
        # Find project ID if project name provided
        project_id = None
        project_name = args.get("project")
        if project_name:
            projects = self.todoist_client.list_projects()
            for p in projects:
                if p["name"].lower() == project_name.lower():
                    project_id = p["id"]
                    break

        task = self.todoist_client.create_task(
            content=args["content"],
            description=args.get("description"),
            project_id=project_id,
            due_string=args.get("due"),
            priority=args.get("priority", 1),
            labels=args.get("labels"),
        )

        return ToolResult(data={
            "task_id": task["id"],
            "content": task["content"],
            "url": task.get("url"),
            "message": f"Task created: {task['content']}",
        })

    def _execute_complete_todoist_task(self, args: dict) -> ToolResult:
        """Mark a Todoist task as complete."""
        task_id = args["task_id"]

        # Get task info first for confirmation message
        try:
            task = self.todoist_client.get_task(task_id)
            task_content = task.get("content", "Unknown task")
        except Exception:
            task_content = "Unknown task"

        self.todoist_client.complete_task(task_id)

        return ToolResult(data={
            "task_id": task_id,
            "message": f"Completed: {task_content}",
        })

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

    def _execute_create_notion_page(self, args: dict) -> ToolResult:
        """Create a new page in a Notion database."""
        # Build properties with title
        properties = args.get("properties", {})
        # Add title property (Notion databases typically use "Name" or "Title")
        properties["Name"] = {
            "title": [{"text": {"content": args["title"]}}]
        }

        page = self.notion_client.create_page(
            database_id=args["database_id"],
            properties=properties,
        )

        return ToolResult(data={
            "page_id": page["id"],
            "url": page.get("url"),
            "title": args["title"],
            "message": f"Page created: {page.get('url', page['id'])}",
        })

    def _execute_add_notion_comment(self, args: dict) -> ToolResult:
        """Add a comment to a Notion page."""
        comment = self.notion_client.add_comment(
            page_id=args["page_id"],
            content=args["content"],
        )

        return ToolResult(data={
            "comment_id": comment["id"],
            "page_id": args["page_id"],
            "message": "Comment added successfully",
        })

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

    def _execute_add_zotero_paper(self, args: dict) -> ToolResult:
        """Add a paper to Zotero by DOI or URL."""
        identifier = args["identifier"].strip()
        collection = args.get("collection", "GoodarziLab")

        # Determine if it's a DOI or URL
        is_doi = (
            identifier.startswith("10.") or
            "doi.org" in identifier or
            identifier.lower().startswith("doi:")
        )

        try:
            if is_doi:
                item = self.zotero_client.add_item_by_doi(identifier, collection)
            else:
                item = self.zotero_client.add_item_by_url(identifier, collection)

            return ToolResult(data={
                "key": item["key"],
                "title": item["title"],
                "collection": collection,
                "message": f"Paper added to Zotero: {item['title']}",
            })
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to add paper: {str(e)}",
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
        current_date = datetime.now(timezone.utc).strftime("%A, %B %d, %Y")
        system = SYSTEM_PROMPT.format(current_date=current_date)

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
                response = self._client.messages.create(
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
                            result = self._tool_executor.execute(tool_name, tool_input)

                            # Record tool call
                            tool_calls_history.append({
                                "tool": tool_name,
                                "input": tool_input,
                                "result": result.to_content()[:500],  # Truncate for history
                                "success": result.success,
                            })

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
        current_date = datetime.now(timezone.utc).strftime("%A, %B %d, %Y")
        system = SYSTEM_PROMPT.format(current_date=current_date)

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
                            result = self._tool_executor.execute(tool_name, tool_input)

                            # Record tool call
                            tool_calls_history.append({
                                "tool": tool_name,
                                "input": tool_input,
                                "result": result.to_content()[:500],
                                "success": result.success,
                            })

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
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": assistant_response})

            # Auto-extract memories via Mem0
            self.user_memory.add_from_conversation(context.user_id, messages)
        except Exception as e:
            logger.debug(f"Memory extraction failed: {e}")
