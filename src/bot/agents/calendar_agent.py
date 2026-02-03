"""Calendar specialist agent."""

import logging
from typing import Any

from .base import BaseAgent, AgentType
from ..conversation import ConversationContext

logger = logging.getLogger(__name__)


# Calendar-related keywords for routing
CALENDAR_KEYWORDS = {
    "calendar", "schedule", "meeting", "event", "appointment",
    "availability", "free", "busy", "slot", "when", "tomorrow",
    "today", "next week", "this week", "morning", "afternoon",
    "evening", "book", "scheduled", "upcoming", "agenda",
}


class CalendarAgent(BaseAgent):
    """Specialist agent for calendar-related tasks.

    Handles:
    - Checking calendar events for specific dates
    - Finding available time slots
    - Understanding meeting schedules
    - Date-based queries
    """

    AGENT_TYPE = AgentType.CALENDAR
    MAX_ITERATIONS = 4

    @property
    def tool_names(self) -> list[str]:
        """Calendar-specific tools."""
        return [
            "GetCalendarEventsTool",
            "CheckAvailabilityTool",
            "RespondToUserTool",
        ]

    @property
    def system_prompt(self) -> str:
        """Calendar-focused system prompt."""
        return """You are a calendar management specialist for Hani's personal assistant.

Your expertise is managing calendar events and scheduling across multiple Google accounts.

Today's date: {current_date}

CAPABILITIES:
- Check calendar events for any date (today, tomorrow, specific dates)
- Find available time slots for scheduling
- Understand meeting patterns and schedules
- Work with multiple Google calendars (Arc Institute, personal, Tahoe Bio, etc.)

GUIDELINES:
1. Always specify the date context clearly in your responses
2. When showing events, organize by time of day
3. For availability checks, suggest the best slots based on typical patterns
4. Be concise but include key details (time, meeting name, location if available)
5. Use RespondToUserTool to send your final response

RESPONSE FORMAT:
- For event listings: Group by morning/afternoon/evening
- For availability: List free slots with durations
- Always mention which calendar/account events are from when relevant

When the user's request is unclear, ask for clarification about the date or type of information needed."""

    @property
    def description(self) -> str:
        return "Calendar expert: checking events, finding availability, scheduling queries"

    def can_handle(self, message: str, context: ConversationContext) -> float:
        """Estimate relevance for calendar tasks."""
        message_lower = message.lower()
        words = set(message_lower.split())

        # Check for calendar keywords
        matches = words & CALENDAR_KEYWORDS
        if matches:
            # More matches = higher confidence
            return min(0.3 + (len(matches) * 0.15), 0.95)

        # Check for date patterns
        date_indicators = ["today", "tomorrow", "monday", "tuesday", "wednesday",
                          "thursday", "friday", "saturday", "sunday", "next",
                          "this week", "next week"]
        for indicator in date_indicators:
            if indicator in message_lower:
                return 0.4

        return 0.0
