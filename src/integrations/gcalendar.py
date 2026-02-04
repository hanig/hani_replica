"""Google Calendar API client."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from .google_auth import get_credentials

logger = logging.getLogger(__name__)


class CalendarClient:
    """Client for interacting with Google Calendar API."""

    def __init__(self, account: str):
        """Initialize Calendar client for a specific account.

        Args:
            account: Account identifier (e.g., "arc", "personal").
        """
        self.account = account
        self._service = None

    @property
    def service(self):
        """Lazily initialize the Calendar service."""
        if self._service is None:
            creds = get_credentials(self.account)
            if not creds:
                raise RuntimeError(f"No valid credentials for account '{self.account}'")
            self._service = build("calendar", "v3", credentials=creds)
        return self._service

    def list_calendars(self) -> list[dict]:
        """List all calendars the user has access to."""
        try:
            response = self.service.calendarList().list().execute()
            return response.get("items", [])
        except HttpError as e:
            logger.error(f"Error listing calendars: {e}")
            raise

    def get_primary_calendar(self) -> dict | None:
        """Get the user's primary calendar."""
        try:
            return self.service.calendars().get(calendarId="primary").execute()
        except HttpError as e:
            logger.error(f"Error getting primary calendar: {e}")
            return None

    def list_events(
        self,
        calendar_id: str = "primary",
        time_min: datetime | None = None,
        time_max: datetime | None = None,
        max_results: int = 250,
        page_token: str | None = None,
        single_events: bool = True,
        order_by: str = "startTime",
    ) -> dict[str, Any]:
        """List events from a calendar.

        Args:
            calendar_id: Calendar ID (use "primary" for main calendar).
            time_min: Lower bound for events (inclusive).
            time_max: Upper bound for events (exclusive).
            max_results: Maximum number of events to return.
            page_token: Token for pagination.
            single_events: Whether to expand recurring events.
            order_by: Sort order ("startTime" or "updated").

        Returns:
            Dictionary with 'items' list and optional 'nextPageToken'.
        """
        try:
            params = {
                "calendarId": calendar_id,
                "maxResults": min(max_results, 2500),
                "singleEvents": single_events,
                "orderBy": order_by,
            }

            if time_min:
                params["timeMin"] = time_min.isoformat()
            if time_max:
                params["timeMax"] = time_max.isoformat()
            if page_token:
                params["pageToken"] = page_token

            return self.service.events().list(**params).execute()
        except HttpError as e:
            logger.error(f"Error listing events: {e}")
            raise

    def get_event(self, event_id: str, calendar_id: str = "primary") -> dict | None:
        """Get a specific event by ID.

        Args:
            event_id: The event ID.
            calendar_id: Calendar ID.

        Returns:
            Event data or None if not found.
        """
        try:
            return (
                self.service.events()
                .get(calendarId=calendar_id, eventId=event_id)
                .execute()
            )
        except HttpError as e:
            if e.resp.status == 404:
                return None
            logger.error(f"Error getting event {event_id}: {e}")
            raise

    def get_today_events(self, calendar_id: str = "primary") -> list[dict]:
        """Get all events for today.

        Args:
            calendar_id: Calendar ID.

        Returns:
            List of events for today.
        """
        now = datetime.now(timezone.utc)
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)

        events = []
        page_token = None

        while True:
            response = self.list_events(
                calendar_id=calendar_id,
                time_min=start_of_day,
                time_max=end_of_day,
                page_token=page_token,
            )

            events.extend(response.get("items", []))

            page_token = response.get("nextPageToken")
            if not page_token:
                break

        return events

    def get_events_for_date(
        self, date: datetime, calendar_id: str = "primary"
    ) -> list[dict]:
        """Get all events for a specific date.

        Args:
            date: The date to get events for.
            calendar_id: Calendar ID.

        Returns:
            List of events for the specified date.
        """
        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        if start_of_day.tzinfo is None:
            start_of_day = start_of_day.replace(tzinfo=timezone.utc)
        end_of_day = start_of_day + timedelta(days=1)

        events = []
        page_token = None

        while True:
            response = self.list_events(
                calendar_id=calendar_id,
                time_min=start_of_day,
                time_max=end_of_day,
                page_token=page_token,
            )

            events.extend(response.get("items", []))

            page_token = response.get("nextPageToken")
            if not page_token:
                break

        return events

    def get_upcoming_events(
        self,
        calendar_id: str = "primary",
        max_results: int = 10,
        days_ahead: int = 7,
    ) -> list[dict]:
        """Get upcoming events within a time window.

        Args:
            calendar_id: Calendar ID.
            max_results: Maximum number of events.
            days_ahead: Number of days to look ahead.

        Returns:
            List of upcoming events.
        """
        now = datetime.now(timezone.utc)
        future = now + timedelta(days=days_ahead)

        response = self.list_events(
            calendar_id=calendar_id,
            time_min=now,
            time_max=future,
            max_results=max_results,
        )

        return response.get("items", [])

    def get_free_busy(
        self,
        time_min: datetime,
        time_max: datetime,
        calendar_ids: list[str] | None = None,
    ) -> dict[str, list[dict]]:
        """Get free/busy information for calendars.

        Args:
            time_min: Start of time range.
            time_max: End of time range.
            calendar_ids: List of calendar IDs. Uses "primary" if not specified.

        Returns:
            Dictionary mapping calendar ID to list of busy periods.
        """
        if calendar_ids is None:
            calendar_ids = ["primary"]

        body = {
            "timeMin": time_min.isoformat(),
            "timeMax": time_max.isoformat(),
            "items": [{"id": cal_id} for cal_id in calendar_ids],
        }

        try:
            response = self.service.freebusy().query(body=body).execute()
            return {
                cal_id: info.get("busy", [])
                for cal_id, info in response.get("calendars", {}).items()
            }
        except HttpError as e:
            logger.error(f"Error getting free/busy: {e}")
            raise

    def list_changes(
        self,
        calendar_id: str = "primary",
        sync_token: str | None = None,
        max_results: int = 250,
    ) -> dict[str, Any]:
        """List changes since last sync (incremental sync).

        Args:
            calendar_id: Calendar ID.
            sync_token: Token from previous sync.
            max_results: Maximum number of events.

        Returns:
            Dictionary with 'items' and 'nextSyncToken'.
        """
        try:
            params = {
                "calendarId": calendar_id,
                "maxResults": max_results,
                "showDeleted": True,
            }

            if sync_token:
                params["syncToken"] = sync_token
            else:
                # First sync - get events from 1 year ago
                time_min = datetime.now(timezone.utc) - timedelta(days=365)
                params["timeMin"] = time_min.isoformat()

            return self.service.events().list(**params).execute()
        except HttpError as e:
            if e.resp.status == 410:
                # Sync token expired, do full sync
                logger.warning("Sync token expired, performing full sync")
                time_min = datetime.now(timezone.utc) - timedelta(days=365)
                return (
                    self.service.events()
                    .list(
                        calendarId=calendar_id,
                        maxResults=max_results,
                        timeMin=time_min.isoformat(),
                        showDeleted=True,
                    )
                    .execute()
                )
            logger.error(f"Error listing changes: {e}")
            raise

    def create_event(
        self,
        summary: str,
        start: datetime,
        end: datetime,
        description: str | None = None,
        attendees: list[str] | None = None,
        location: str | None = None,
        calendar_id: str = "primary",
        send_notifications: bool = True,
    ) -> dict[str, Any]:
        """Create a calendar event.

        Args:
            summary: Event title.
            start: Event start time.
            end: Event end time.
            description: Event description.
            attendees: List of attendee email addresses.
            location: Event location.
            calendar_id: Calendar ID (default "primary").
            send_notifications: Whether to send email notifications to attendees.

        Returns:
            Created event data.
        """
        # Ensure times have timezone info
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        event_body: dict[str, Any] = {
            "summary": summary,
            "start": {"dateTime": start.isoformat()},
            "end": {"dateTime": end.isoformat()},
        }

        if description:
            event_body["description"] = description
        if location:
            event_body["location"] = location
        if attendees:
            event_body["attendees"] = [{"email": email} for email in attendees]

        try:
            return (
                self.service.events()
                .insert(
                    calendarId=calendar_id,
                    body=event_body,
                    sendUpdates="all" if send_notifications else "none",
                )
                .execute()
            )
        except HttpError as e:
            logger.error(f"Error creating event: {e}")
            raise

    def update_event(
        self,
        event_id: str,
        calendar_id: str = "primary",
        send_notifications: bool = True,
        **updates: Any,
    ) -> dict[str, Any]:
        """Update an existing calendar event.

        Args:
            event_id: The event ID to update.
            calendar_id: Calendar ID (default "primary").
            send_notifications: Whether to send email notifications about the update.
            **updates: Fields to update (summary, description, location, start, end, attendees).

        Returns:
            Updated event data.
        """
        # First get the existing event
        event = self.get_event(event_id, calendar_id)
        if not event:
            raise ValueError(f"Event {event_id} not found")

        # Apply updates
        for key, value in updates.items():
            if key in ("start", "end") and isinstance(value, datetime):
                if value.tzinfo is None:
                    value = value.replace(tzinfo=timezone.utc)
                event[key] = {"dateTime": value.isoformat()}
            elif key == "attendees" and isinstance(value, list):
                event[key] = [{"email": email} for email in value]
            else:
                event[key] = value

        try:
            return (
                self.service.events()
                .update(
                    calendarId=calendar_id,
                    eventId=event_id,
                    body=event,
                    sendUpdates="all" if send_notifications else "none",
                )
                .execute()
            )
        except HttpError as e:
            logger.error(f"Error updating event {event_id}: {e}")
            raise

    def delete_event(
        self,
        event_id: str,
        calendar_id: str = "primary",
        send_notifications: bool = True,
    ) -> None:
        """Delete a calendar event.

        Args:
            event_id: The event ID to delete.
            calendar_id: Calendar ID (default "primary").
            send_notifications: Whether to send cancellation notifications.
        """
        try:
            self.service.events().delete(
                calendarId=calendar_id,
                eventId=event_id,
                sendUpdates="all" if send_notifications else "none",
            ).execute()
        except HttpError as e:
            if e.resp.status == 404:
                logger.warning(f"Event {event_id} not found, may already be deleted")
                return
            logger.error(f"Error deleting event {event_id}: {e}")
            raise

    def create_quick_event(
        self,
        text: str,
        calendar_id: str = "primary",
        send_notifications: bool = True,
    ) -> dict[str, Any]:
        """Create an event from natural language text.

        Uses Google's natural language processing to parse event details.

        Args:
            text: Natural language description (e.g., "Lunch with John tomorrow at noon").
            calendar_id: Calendar ID (default "primary").
            send_notifications: Whether to send email notifications.

        Returns:
            Created event data.
        """
        try:
            return (
                self.service.events()
                .quickAdd(
                    calendarId=calendar_id,
                    text=text,
                    sendUpdates="all" if send_notifications else "none",
                )
                .execute()
            )
        except HttpError as e:
            logger.error(f"Error creating quick event: {e}")
            raise

    @staticmethod
    def parse_event(event: dict) -> dict[str, Any]:
        """Parse a Calendar event into a structured format.

        Args:
            event: Raw event from Calendar API.

        Returns:
            Parsed event metadata.
        """
        result = {
            "id": event["id"],
            "summary": event.get("summary", "(No title)"),
            "description": event.get("description", ""),
            "location": event.get("location", ""),
            "status": event.get("status", "confirmed"),
            "html_link": event.get("htmlLink", ""),
            "organizer": None,
            "attendees": [],
            "is_all_day": False,
            "start": None,
            "end": None,
            "cancelled": event.get("status") == "cancelled",
        }

        # Parse start/end times
        start = event.get("start", {})
        end = event.get("end", {})

        if "dateTime" in start:
            result["start"] = datetime.fromisoformat(start["dateTime"])
            result["end"] = datetime.fromisoformat(end.get("dateTime", start["dateTime"]))
            result["is_all_day"] = False
        elif "date" in start:
            # All-day events: parse as date and add UTC timezone
            start_dt = datetime.fromisoformat(start["date"])
            end_dt = datetime.fromisoformat(end.get("date", start["date"]))
            result["start"] = start_dt.replace(tzinfo=timezone.utc)
            result["end"] = end_dt.replace(tzinfo=timezone.utc)
            result["is_all_day"] = True

        # Parse organizer
        if "organizer" in event:
            result["organizer"] = {
                "email": event["organizer"].get("email"),
                "name": event["organizer"].get("displayName"),
                "self": event["organizer"].get("self", False),
            }

        # Parse attendees
        if "attendees" in event:
            result["attendees"] = [
                {
                    "email": att.get("email"),
                    "name": att.get("displayName"),
                    "response_status": att.get("responseStatus"),
                    "self": att.get("self", False),
                    "optional": att.get("optional", False),
                }
                for att in event["attendees"]
            ]

        # Conference data (Zoom, Meet, etc.)
        if "conferenceData" in event:
            conf = event["conferenceData"]
            entry_points = conf.get("entryPoints", [])
            for ep in entry_points:
                if ep.get("entryPointType") == "video":
                    result["conference_url"] = ep.get("uri")
                    break

        return result
