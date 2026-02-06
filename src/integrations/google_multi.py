"""Multi-account Google manager with tiered search."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any

from ..config import GOOGLE_ACCOUNTS, GOOGLE_EMAILS, GOOGLE_TIER1, GOOGLE_TIER2, get_user_timezone
from .gcalendar import CalendarClient
from .gdrive import DriveClient
from .gmail import GmailClient
from .google_auth import check_all_accounts, get_credentials

logger = logging.getLogger(__name__)


class MultiGoogleManager:
    """Manager for multi-account Google services with tiered search."""

    def __init__(self):
        """Initialize the multi-account manager."""
        self._gmail_clients: dict[str, GmailClient] = {}
        self._drive_clients: dict[str, DriveClient] = {}
        self._calendar_clients: dict[str, CalendarClient] = {}

    def get_gmail_client(self, account: str) -> GmailClient | None:
        """Get or create a Gmail client for an account."""
        if account not in self._gmail_clients:
            if get_credentials(account):
                self._gmail_clients[account] = GmailClient(account)
            else:
                return None
        return self._gmail_clients.get(account)

    def get_drive_client(self, account: str) -> DriveClient | None:
        """Get or create a Drive client for an account."""
        if account not in self._drive_clients:
            if get_credentials(account):
                self._drive_clients[account] = DriveClient(account)
            else:
                return None
        return self._drive_clients.get(account)

    def get_calendar_client(self, account: str) -> CalendarClient | None:
        """Get or create a Calendar client for an account."""
        if account not in self._calendar_clients:
            if get_credentials(account):
                self._calendar_clients[account] = CalendarClient(account)
            else:
                return None
        return self._calendar_clients.get(account)

    def get_authenticated_accounts(self) -> list[str]:
        """Get list of accounts with valid credentials."""
        status = check_all_accounts()
        return [account for account, valid in status.items() if valid]

    def search_mail_tiered(
        self,
        query: str,
        max_results: int = 20,
        tier1_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Search mail across accounts with tiered priority.

        Searches Tier 1 accounts first. If no results, searches Tier 2.

        Args:
            query: Gmail search query.
            max_results: Maximum total results.
            tier1_only: Only search Tier 1 accounts.

        Returns:
            List of messages with account metadata.
        """
        results = []

        # Search Tier 1 first
        tier1_results = self._search_mail_accounts(
            GOOGLE_TIER1, query, max_results
        )
        results.extend(tier1_results)

        # If we have enough results or tier1_only, return
        if len(results) >= max_results or tier1_only:
            return results[:max_results]

        # Search Tier 2 for remaining quota
        remaining = max_results - len(results)
        tier2_results = self._search_mail_accounts(
            GOOGLE_TIER2, query, remaining
        )
        results.extend(tier2_results)

        return results[:max_results]

    def _search_mail_accounts(
        self, accounts: list[str], query: str, max_results: int
    ) -> list[dict[str, Any]]:
        """Search mail across specific accounts in parallel."""
        if not accounts:
            return []
        results = []

        with ThreadPoolExecutor(max_workers=len(accounts)) as executor:
            futures = {}
            for account in accounts:
                client = self.get_gmail_client(account)
                if client:
                    future = executor.submit(
                        self._search_mail_single,
                        client,
                        account,
                        query,
                        max_results,
                    )
                    futures[future] = account

            for future in as_completed(futures):
                account = futures[future]
                try:
                    account_results = future.result()
                    results.extend(account_results)
                except Exception as e:
                    logger.error(f"Error searching mail for {account}: {e}")

        # Sort by timestamp, newest first
        results.sort(
            key=lambda x: x.get("timestamp") or datetime.min,
            reverse=True,
        )

        return results

    def _search_mail_single(
        self,
        client: GmailClient,
        account: str,
        query: str,
        max_results: int,
    ) -> list[dict[str, Any]]:
        """Search mail for a single account."""
        messages = client.search_messages(query, max_results=max_results)
        results = []

        for msg in messages:
            parsed = GmailClient.parse_message(msg)
            parsed["account"] = account
            parsed["account_email"] = GOOGLE_EMAILS.get(account, "")
            results.append(parsed)

        return results

    def search_drive_tiered(
        self,
        query: str,
        max_results: int = 20,
        tier1_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Search Drive across accounts with tiered priority.

        Args:
            query: Search text.
            max_results: Maximum total results.
            tier1_only: Only search Tier 1 accounts.

        Returns:
            List of files with account metadata.
        """
        results = []

        # Search Tier 1 first
        tier1_results = self._search_drive_accounts(
            GOOGLE_TIER1, query, max_results
        )
        results.extend(tier1_results)

        if len(results) >= max_results or tier1_only:
            return results[:max_results]

        # Search Tier 2
        remaining = max_results - len(results)
        tier2_results = self._search_drive_accounts(
            GOOGLE_TIER2, query, remaining
        )
        results.extend(tier2_results)

        return results[:max_results]

    def _search_drive_accounts(
        self, accounts: list[str], query: str, max_results: int
    ) -> list[dict[str, Any]]:
        """Search Drive across specific accounts in parallel."""
        if not accounts:
            return []
        results = []

        with ThreadPoolExecutor(max_workers=len(accounts)) as executor:
            futures = {}
            for account in accounts:
                client = self.get_drive_client(account)
                if client:
                    future = executor.submit(
                        self._search_drive_single,
                        client,
                        account,
                        query,
                        max_results,
                    )
                    futures[future] = account

            for future in as_completed(futures):
                account = futures[future]
                try:
                    account_results = future.result()
                    results.extend(account_results)
                except Exception as e:
                    logger.error(f"Error searching drive for {account}: {e}")

        # Sort by modified time, newest first
        results.sort(
            key=lambda x: x.get("modified_time") or datetime.min,
            reverse=True,
        )

        return results

    def _search_drive_single(
        self,
        client: DriveClient,
        account: str,
        query: str,
        max_results: int,
    ) -> list[dict[str, Any]]:
        """Search Drive for a single account."""
        files = client.search_files(query, max_results=max_results)
        results = []

        for file in files:
            parsed = DriveClient.parse_file(file)
            parsed["account"] = account
            parsed["account_email"] = GOOGLE_EMAILS.get(account, "")
            results.append(parsed)

        return results

    def get_all_calendars_today(self) -> list[dict[str, Any]]:
        """Get today's events from all calendars.

        Returns:
            List of events with account metadata, sorted by start time.
        """
        return self.get_all_calendars_for_date(datetime.now(get_user_timezone()))

    def get_all_calendars_for_date(
        self, date: datetime
    ) -> list[dict[str, Any]]:
        """Get events for a specific date from all calendars.

        Args:
            date: The date to get events for.

        Returns:
            List of events with account metadata, sorted by start time.
        """
        if not GOOGLE_ACCOUNTS:
            return []
        all_events = []

        with ThreadPoolExecutor(max_workers=len(GOOGLE_ACCOUNTS)) as executor:
            futures = {}
            for account in GOOGLE_ACCOUNTS:
                client = self.get_calendar_client(account)
                if client:
                    future = executor.submit(
                        self._get_calendar_events,
                        client,
                        account,
                        date,
                    )
                    futures[future] = account

            for future in as_completed(futures):
                account = futures[future]
                try:
                    events = future.result()
                    all_events.extend(events)
                except Exception as e:
                    logger.error(f"Error getting calendar for {account}: {e}")

        # Sort by start time
        all_events.sort(key=lambda x: x.get("start") or datetime.min)

        return all_events

    def _get_calendar_events(
        self,
        client: CalendarClient,
        account: str,
        date: datetime,
    ) -> list[dict[str, Any]]:
        """Get calendar events for a single account."""
        events = client.get_events_for_date(date)
        results = []

        for event in events:
            parsed = CalendarClient.parse_event(event)
            parsed["account"] = account
            parsed["account_email"] = GOOGLE_EMAILS.get(account, "")
            results.append(parsed)

        return results

    def check_availability(
        self,
        date: datetime,
        duration_minutes: int = 30,
        working_hours: tuple[int, int] = (9, 18),
    ) -> list[dict[str, Any]]:
        """Find available time slots across all calendars.

        Args:
            date: The date to check.
            duration_minutes: Minimum slot duration.
            working_hours: (start_hour, end_hour) in local time.

        Returns:
            List of available time slots.
        """
        tz = get_user_timezone()
        if date.tzinfo is None:
            date = date.replace(tzinfo=tz)

        # Get all events for the date
        events = self.get_all_calendars_for_date(date)

        # Filter to only confirmed events (not cancelled or declined)
        busy_periods = []
        for event in events:
            if event.get("cancelled"):
                continue
            if event.get("start") and event.get("end"):
                busy_periods.append((event["start"], event["end"]))

        # Sort and merge overlapping periods
        busy_periods.sort(key=lambda x: x[0])
        merged = []
        for start, end in busy_periods:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        # Find free slots within working hours
        start_hour, end_hour = working_hours
        day_start = date.astimezone(tz).replace(
            hour=start_hour, minute=0, second=0, microsecond=0
        )
        day_end = date.astimezone(tz).replace(
            hour=end_hour, minute=0, second=0, microsecond=0
        )

        free_slots = []
        current = day_start

        for busy_start, busy_end in merged:
            # Ensure timezone-aware comparison
            if busy_start.tzinfo is None:
                busy_start = busy_start.replace(tzinfo=tz)
            if busy_end.tzinfo is None:
                busy_end = busy_end.replace(tzinfo=tz)
            busy_start = busy_start.astimezone(tz)
            busy_end = busy_end.astimezone(tz)

            if current < busy_start:
                slot_duration = (busy_start - current).total_seconds() / 60
                if slot_duration >= duration_minutes:
                    free_slots.append({
                        "start": current,
                        "end": busy_start,
                        "duration_minutes": int(slot_duration),
                    })
            current = max(current, busy_end)

        # Check for free time after last meeting
        if current < day_end:
            slot_duration = (day_end - current).total_seconds() / 60
            if slot_duration >= duration_minutes:
                free_slots.append({
                    "start": current,
                    "end": day_end,
                    "duration_minutes": int(slot_duration),
                })

        return free_slots

    def get_unread_counts(self) -> dict[str, int]:
        """Get unread email counts for all accounts.

        Returns:
            Dictionary mapping account to unread count.
        """
        if not GOOGLE_ACCOUNTS:
            return {}
        counts = {}

        with ThreadPoolExecutor(max_workers=len(GOOGLE_ACCOUNTS)) as executor:
            futures = {}
            for account in GOOGLE_ACCOUNTS:
                client = self.get_gmail_client(account)
                if client:
                    future = executor.submit(client.get_unread_count)
                    futures[future] = account

            for future in as_completed(futures):
                account = futures[future]
                try:
                    counts[account] = future.result()
                except Exception as e:
                    logger.error(f"Error getting unread count for {account}: {e}")
                    counts[account] = -1

        return counts

    def create_draft(
        self,
        account: str,
        to: str,
        subject: str,
        body: str,
        cc: str | None = None,
        bcc: str | None = None,
    ) -> dict[str, Any]:
        """Create an email draft in a specific account.

        Args:
            account: Account to create draft in.
            to: Recipient email address.
            subject: Email subject.
            body: Email body.
            cc: CC recipients.
            bcc: BCC recipients.

        Returns:
            Created draft data.
        """
        client = self.get_gmail_client(account)
        if not client:
            raise ValueError(f"No Gmail client available for account: {account}")

        return client.create_draft(
            to=to,
            subject=subject,
            body=body,
            cc=cc,
            bcc=bcc,
        )

    def send_email(
        self,
        account: str,
        to: str,
        subject: str,
        body: str,
        cc: str | None = None,
        bcc: str | None = None,
    ) -> dict[str, Any]:
        """Send an email from a specific account.

        Args:
            account: Account to send from.
            to: Recipient email address.
            subject: Email subject.
            body: Email body.
            cc: CC recipients.
            bcc: BCC recipients.

        Returns:
            Sent message data.
        """
        client = self.get_gmail_client(account)
        if not client:
            raise ValueError(f"No Gmail client available for account: {account}")

        return client.send_message(
            to=to,
            subject=subject,
            body=body,
            cc=cc,
            bcc=bcc,
        )

    def create_calendar_event(
        self,
        account: str,
        summary: str,
        start: datetime,
        end: datetime,
        description: str | None = None,
        attendees: list[str] | None = None,
        location: str | None = None,
        send_notifications: bool = True,
    ) -> dict[str, Any]:
        """Create a calendar event in a specific account.

        Args:
            account: Account to create event in.
            summary: Event title.
            start: Event start time.
            end: Event end time.
            description: Event description.
            attendees: List of attendee email addresses (will send invites).
            location: Event location.
            send_notifications: Whether to send email notifications to attendees.

        Returns:
            Created event data.
        """
        client = self.get_calendar_client(account)
        if not client:
            raise ValueError(f"No Calendar client available for account: {account}")

        return client.create_event(
            summary=summary,
            start=start,
            end=end,
            description=description,
            attendees=attendees,
            location=location,
            send_notifications=send_notifications,
        )
