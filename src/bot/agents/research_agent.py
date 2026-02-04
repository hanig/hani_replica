"""Research specialist agent."""

import logging
from typing import Any

from .base import BaseAgent, AgentType
from ..conversation import ConversationContext

logger = logging.getLogger(__name__)


# Research-related keywords for routing
RESEARCH_KEYWORDS = {
    "search", "find", "look", "what", "where", "who",
    "information", "about", "related", "document", "file",
    "drive", "note", "summary", "briefing", "overview",
    "task", "tasks", "todoist", "todo", "to-do",
    "notion", "page", "database",
}


class ResearchAgent(BaseAgent):
    """Specialist agent for research and information retrieval.

    Handles:
    - Semantic search across the knowledge graph
    - Google Drive file search
    - Finding information about people
    - Daily briefings and summaries
    - General information retrieval
    """

    AGENT_TYPE = AgentType.RESEARCH
    MAX_ITERATIONS = 6

    @property
    def tool_names(self) -> list[str]:
        """Research-specific tools."""
        return [
            "SemanticSearchTool",
            "SearchDriveTool",
            "FindPersonTool",
            "GetPersonActivityTool",
            "GetDailyBriefingTool",
            "GetTodoistTasksTool",
            "CreateTodoistTaskTool",
            "CompleteTodoistTaskTool",
            "SearchNotionTool",
            "CreateNotionPageTool",
            "AddNotionCommentTool",
            "RespondToUserTool",
        ]

    @property
    def system_prompt(self) -> str:
        """Research-focused system prompt."""
        return """You are a research and information retrieval specialist for Hani's personal assistant.

Your expertise is finding information across the personal knowledge graph.

Today's date: {current_date}

KNOWLEDGE GRAPH SOURCES:
- Emails (indexed from 6 Google accounts)
- Calendar events
- Google Drive documents
- GitHub activity
- Slack conversations
- Todoist tasks
- Notion pages and databases

CAPABILITIES:
- Semantic search across all indexed content
- Google Drive file search
- Find people and their contact information
- Get activity history for specific people
- Generate daily briefings
- Get, create, and complete Todoist tasks
- Search and create Notion pages

SEARCH STRATEGIES:
1. Start with semantic search for broad queries
2. Use specific source filters when you know the type
3. For people queries, use FindPersonTool first
4. For recent activity, check GetDailyBriefingTool

TODOIST TASKS:
- Use GetTodoistTasksTool when user asks about "my tasks", "to-do", "what do I need to do"
- Use CreateTodoistTaskTool to add new tasks (supports natural language due dates like "tomorrow")
- Use CompleteTodoistTaskTool to mark tasks done

NOTION:
- Use SearchNotionTool for Notion page/database queries
- Use CreateNotionPageTool to add pages to databases

GUIDELINES:
1. Summarize findings concisely
2. Include source/type information for traceability
3. For document searches, show title and relevant excerpt
4. For people, show their recent interactions
5. Use RespondToUserTool to send your final response

SEMANTIC SEARCH TIPS:
- Use content_types filter: email, calendar_event, drive_file, github, slack
- Use source filter for specific Google accounts
- Limit results to avoid overwhelming responses

DAILY BRIEFING:
- Use for "what's happening" or overview queries
- Combines calendar, email, and GitHub status"""

    @property
    def description(self) -> str:
        return "Research expert: semantic search, documents, people lookup, briefings"

    def can_handle(self, message: str, context: ConversationContext) -> float:
        """Estimate relevance for research tasks."""
        message_lower = message.lower()
        words = set(message_lower.split())

        # High confidence for Todoist queries
        if any(kw in message_lower for kw in ["task", "tasks", "todoist", "todo", "to-do", "to do"]):
            return 0.9

        # High confidence for Notion queries
        if "notion" in message_lower:
            return 0.9

        # Check for research keywords
        matches = words & RESEARCH_KEYWORDS
        if matches:
            # More matches = higher confidence
            return min(0.2 + (len(matches) * 0.1), 0.7)

        # Check for question patterns
        question_words = ["what", "where", "who", "how", "why", "when"]
        if any(message_lower.startswith(q) for q in question_words):
            return 0.3

        # Check for briefing patterns
        if "briefing" in message_lower or "overview" in message_lower:
            return 0.8

        # Default low confidence - research is often a fallback
        return 0.1
