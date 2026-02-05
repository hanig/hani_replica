"""Briefing handler for daily summaries."""

import logging
from datetime import datetime, timezone
from typing import Any

from ..conversation import ConversationContext
from ..formatters import format_briefing
from ..intent_router import Intent
from ...config import get_user_timezone
from .base import BaseHandler

logger = logging.getLogger(__name__)


class BriefingHandler(BaseHandler):
    """Handler for daily briefing/summary requests."""

    def __init__(self):
        """Initialize the briefing handler."""
        self._multi_google = None
        self._github_client = None
        self._todoist_client = None

    @property
    def multi_google(self):
        """Lazy load multi-Google manager."""
        if self._multi_google is None:
            from ...integrations.google_multi import MultiGoogleManager
            self._multi_google = MultiGoogleManager()
        return self._multi_google

    @property
    def github_client(self):
        """Lazy load GitHub client."""
        if self._github_client is None:
            from ...integrations.github_client import GitHubClient
            self._github_client = GitHubClient()
        return self._github_client

    @property
    def todoist_client(self):
        """Lazy load Todoist client."""
        if self._todoist_client is None:
            from ...integrations.todoist_client import TodoistClient
            self._todoist_client = TodoistClient()
        return self._todoist_client

    def handle(self, intent: Intent, context: ConversationContext) -> dict[str, Any]:
        """Handle a briefing intent.

        Args:
            intent: Classified intent with entities.
            context: Conversation context.

        Returns:
            Response dictionary with briefing information.
        """
        try:
            briefing = self._generate_briefing()
            return format_briefing(briefing)

        except Exception as e:
            logger.error(f"Error generating briefing: {e}")
            return {"text": f"Error generating briefing: {str(e)}"}

    def _generate_briefing(self) -> dict[str, Any]:
        """Generate a daily briefing.

        Returns:
            Dictionary with briefing data.
        """
        briefing = {
            "date": datetime.now(get_user_timezone()).strftime("%A, %B %d, %Y"),
            "events": [],
            "unread_counts": {},
            "open_prs": [],
            "open_issues": [],
            "overdue_tasks": [],
        }

        # Get today's calendar events
        try:
            briefing["events"] = self.multi_google.get_all_calendars_today()
        except Exception as e:
            logger.warning(f"Error getting calendar for briefing: {e}")

        # Get unread email counts
        try:
            briefing["unread_counts"] = self.multi_google.get_unread_counts()
        except Exception as e:
            logger.warning(f"Error getting unread counts: {e}")

        # Get GitHub data
        try:
            briefing["open_prs"] = self.github_client.get_my_prs(state="open", max_results=10)
        except Exception as e:
            logger.warning(f"Error getting PRs for briefing: {e}")

        try:
            briefing["open_issues"] = self.github_client.get_my_issues(state="open", max_results=10)
        except Exception as e:
            logger.warning(f"Error getting issues for briefing: {e}")

        # Get Todoist overdue tasks
        try:
            briefing["overdue_tasks"] = self.todoist_client.list_tasks(filter="overdue")
        except Exception as e:
            logger.warning(f"Error getting Todoist overdue tasks for briefing: {e}")

        return briefing
