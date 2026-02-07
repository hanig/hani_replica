"""Calendar actions that require confirmation."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from .confirmable import PendingAction
from ...config import PRIMARY_ACCOUNT

logger = logging.getLogger(__name__)


@dataclass
class CreateEventAction(PendingAction):
    """Action to create a calendar event with optional attendees."""

    title: str = ""
    date_str: str = ""  # Natural language date like "tomorrow", "Monday"
    time_str: str = ""  # Natural language time like "2pm", "14:00"
    duration_minutes: int = 60  # Default 1 hour
    attendees: list[str] = field(default_factory=list)
    location: str = ""
    description: str = ""
    account: str = ""  # Which Google account to use (resolved to PRIMARY_ACCOUNT if empty)
    _state: str = "title"  # What we're asking for: title, date, time, attendees, account

    def __post_init__(self):
        if not self.account:
            self.account = PRIMARY_ACCOUNT

    def is_ready(self) -> bool:
        """Check if we have enough info to create the event."""
        return bool(self.title and self.date_str and self.time_str)

    def get_next_prompt(self) -> str:
        """Get the prompt for the next required field."""
        if not self.title:
            return "What should the event be called?"
        if not self.date_str:
            return "What date should this event be on? (e.g., tomorrow, Monday, 2024-01-15)"
        if not self.time_str:
            return "What time should it start? (e.g., 2pm, 14:00, noon)"
        return ""

    def update_from_input(self, text: str) -> None:
        """Update action fields from user input."""
        text = text.strip()

        if not self.title:
            self.title = text
            self._state = "date"
        elif not self.date_str:
            self.date_str = text
            self._state = "time"
        elif not self.time_str:
            self.time_str = text
            self._state = "done"

    def get_preview(self) -> str:
        """Get a preview of the event."""
        preview = f"*Event:* {self.title}\n*When:* {self.date_str} at {self.time_str}"

        if self.duration_minutes != 60:
            preview += f" ({self.duration_minutes} min)"

        if self.location:
            preview += f"\n*Location:* {self.location}"

        if self.attendees:
            preview += f"\n*Attendees:* {', '.join(self.attendees)}"
            preview += "\n_(Calendar invites will be sent to attendees)_"

        if self.description:
            desc_preview = self.description[:100]
            if len(self.description) > 100:
                desc_preview += "..."
            preview += f"\n*Description:* {desc_preview}"

        preview += f"\n*Account:* {self.account}"

        return preview

    def execute(self) -> dict[str, Any]:
        """Create the calendar event."""
        from ...integrations.gcalendar import CalendarClient
        from ...config import get_user_timezone

        try:
            # Parse the date and time
            start_dt = self._parse_datetime()
            end_dt = start_dt + timedelta(minutes=self.duration_minutes)

            # Create the event
            client = CalendarClient(self.account)
            event = client.create_event(
                summary=self.title,
                start=start_dt,
                end=end_dt,
                description=self.description or None,
                attendees=self.attendees if self.attendees else None,
                location=self.location or None,
                send_notifications=True,  # Always send invites if attendees
            )

            # Format response
            event_link = event.get("htmlLink", "")
            attendee_msg = ""
            if self.attendees:
                attendee_msg = f" Invites sent to {len(self.attendees)} attendee(s)."

            return {
                "success": True,
                "message": f"Created event '{self.title}' on {self.date_str} at {self.time_str}.{attendee_msg}\n{event_link}",
                "event": event,
            }

        except Exception as e:
            logger.error(f"Error creating calendar event: {e}")
            return {
                "success": False,
                "message": f"Failed to create event: {str(e)}",
            }

    def _parse_datetime(self) -> datetime:
        """Parse date_str and time_str into a datetime."""
        from ...config import get_user_timezone

        tz = get_user_timezone()
        now = datetime.now(tz)

        # Parse date
        date_lower = self.date_str.lower()
        if date_lower == "today":
            target_date = now.date()
        elif date_lower == "tomorrow":
            target_date = (now + timedelta(days=1)).date()
        elif date_lower == "yesterday":
            target_date = (now - timedelta(days=1)).date()
        else:
            # Try day names (next occurrence)
            day_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            if date_lower in day_names:
                target_weekday = day_names.index(date_lower)
                days_ahead = target_weekday - now.weekday()
                if days_ahead <= 0:  # Target day already happened this week
                    days_ahead += 7
                target_date = (now + timedelta(days=days_ahead)).date()
            else:
                # Try ISO format
                try:
                    target_date = datetime.fromisoformat(self.date_str).date()
                except ValueError:
                    # Fall back to today
                    target_date = now.date()

        # Parse time
        time_lower = self.time_str.lower().strip()
        hour = 12  # Default to noon
        minute = 0

        if time_lower == "noon":
            hour, minute = 12, 0
        elif time_lower == "midnight":
            hour, minute = 0, 0
        elif ":" in time_lower:
            # Format like "14:00" or "2:30pm"
            time_part = time_lower.replace("am", "").replace("pm", "").strip()
            parts = time_part.split(":")
            hour = int(parts[0])
            minute = int(parts[1]) if len(parts) > 1 else 0
            if "pm" in time_lower and hour < 12:
                hour += 12
            elif "am" in time_lower and hour == 12:
                hour = 0
        else:
            # Format like "2pm" or "14"
            time_clean = time_lower.replace("am", "").replace("pm", "").strip()
            try:
                hour = int(time_clean)
                if "pm" in time_lower and hour < 12:
                    hour += 12
                elif "am" in time_lower and hour == 12:
                    hour = 0
            except ValueError:
                pass

        # Combine date and time
        return datetime(
            year=target_date.year,
            month=target_date.month,
            day=target_date.day,
            hour=hour,
            minute=minute,
            tzinfo=tz,
        )

    def get_action_type(self) -> str:
        return "Create Calendar Event"
