"""Heartbeat system for proactive notifications."""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from ..config import SLACK_AUTHORIZED_USERS
from .formatters import format_briefing, format_calendar_events
from .proactive_settings import ProactiveSettingsStore, UserProactiveSettings

if TYPE_CHECKING:
    from ..integrations.google_multi import MultiGoogleManager
    from ..integrations.github_client import GitHubClient

logger = logging.getLogger(__name__)


class HeartbeatManager:
    """Manages proactive notifications and scheduled tasks.

    Features:
    - Calendar reminders (N minutes before meetings)
    - Important email alerts
    - Daily morning briefings
    """

    def __init__(
        self,
        slack_client: WebClient,
        settings_store: ProactiveSettingsStore | None = None,
    ):
        """Initialize the heartbeat manager.

        Args:
            slack_client: Slack WebClient for sending messages.
            settings_store: ProactiveSettingsStore for user settings.
        """
        self.slack_client = slack_client
        self.settings_store = settings_store or ProactiveSettingsStore()

        # Lazy-loaded integrations
        self._multi_google: "MultiGoogleManager | None" = None
        self._github_client: "GitHubClient | None" = None

        # Track recently sent notifications to avoid duplicates
        self._recent_reminders: set[str] = set()

    @property
    def multi_google(self) -> "MultiGoogleManager":
        """Lazy load multi-Google manager."""
        if self._multi_google is None:
            from ..integrations.google_multi import MultiGoogleManager
            self._multi_google = MultiGoogleManager()
        return self._multi_google

    @property
    def github_client(self) -> "GitHubClient":
        """Lazy load GitHub client."""
        if self._github_client is None:
            from ..integrations.github_client import GitHubClient
            self._github_client = GitHubClient()
        return self._github_client

    def check_calendar_reminders(self) -> int:
        """Check for upcoming meetings and send reminders.

        Returns:
            Number of reminders sent.
        """
        logger.debug("Checking calendar reminders...")
        reminders_sent = 0

        # Get users with calendar reminders enabled
        users = self.settings_store.get_all_enabled_users("calendar_reminders")

        if not users:
            # If no users have settings, check for authorized users
            users = [
                UserProactiveSettings(user_id=uid)
                for uid in SLACK_AUTHORIZED_USERS
            ]

        for settings in users:
            try:
                sent = self._check_calendar_for_user(settings)
                reminders_sent += sent
            except Exception as e:
                logger.error(f"Error checking calendar for {settings.user_id}: {e}")

        if reminders_sent > 0:
            logger.info(f"Sent {reminders_sent} calendar reminders")

        return reminders_sent

    def _check_calendar_for_user(self, settings: UserProactiveSettings) -> int:
        """Check calendar and send reminders for a specific user.

        Args:
            settings: User's proactive settings.

        Returns:
            Number of reminders sent.
        """
        if not settings.calendar_reminders_enabled:
            return 0

        # Check quiet hours
        if self._is_quiet_hours(settings):
            return 0

        reminders_sent = 0
        now = datetime.now(timezone.utc)
        reminder_window_start = now
        reminder_window_end = now + timedelta(minutes=settings.reminder_minutes_before + 5)

        try:
            # Get upcoming events
            events = self.multi_google.get_all_calendars_for_date(now)

            for event in events:
                event_start = event.get("start")
                if not event_start:
                    continue

                # Skip all-day events unless configured
                if event.get("is_all_day") and not settings.remind_for_all_day_events:
                    continue

                # Convert to datetime if needed
                if isinstance(event_start, str):
                    try:
                        event_start = datetime.fromisoformat(event_start.replace("Z", "+00:00"))
                    except ValueError:
                        continue

                # Check if event is within reminder window
                if reminder_window_start <= event_start <= reminder_window_end:
                    event_id = event.get("id", str(event_start))
                    notification_key = f"{event_id}_{event_start.isoformat()}"

                    # Check if we already sent this reminder
                    if self.settings_store.was_notification_sent(
                        settings.user_id, "calendar_reminder", notification_key
                    ):
                        continue

                    # Send reminder
                    if self._send_calendar_reminder(settings, event):
                        self.settings_store.mark_notification_sent(
                            settings.user_id,
                            "calendar_reminder",
                            notification_key,
                            time.time(),
                        )
                        reminders_sent += 1

        except Exception as e:
            logger.error(f"Error checking calendar: {e}")

        return reminders_sent

    def _send_calendar_reminder(
        self, settings: UserProactiveSettings, event: dict
    ) -> bool:
        """Send a calendar reminder to a user.

        Args:
            settings: User's proactive settings.
            event: Calendar event dictionary.

        Returns:
            True if sent successfully, False otherwise.
        """
        try:
            # Get or open DM channel
            channel_id = self._get_dm_channel(settings.user_id)
            if not channel_id:
                return False

            # Format the reminder
            title = event.get("summary", "Untitled Event")
            start = event.get("start")
            location = event.get("location", "")

            if isinstance(start, datetime):
                time_str = start.strftime("%I:%M %p")
            else:
                time_str = str(start)

            # Calculate time until event
            if isinstance(start, datetime):
                now = datetime.now(timezone.utc)
                minutes_until = int((start - now).total_seconds() / 60)
                time_until_str = f"{minutes_until} minutes"
            else:
                time_until_str = "soon"

            # Build message blocks
            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f":bell: *Upcoming Meeting in {time_until_str}*",
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*{title}*\n:clock3: {time_str}",
                    },
                },
            ]

            if location:
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f":round_pushpin: {location}",
                    },
                })

            # Add meeting link if available
            meeting_link = event.get("hangout_link") or event.get("meet_link")
            if meeting_link:
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f":video_camera: <{meeting_link}|Join Meeting>",
                    },
                })

            # Send the message
            self.slack_client.chat_postMessage(
                channel=channel_id,
                text=f"Reminder: {title} in {time_until_str}",
                blocks=blocks,
            )

            logger.info(f"Sent calendar reminder to {settings.user_id}: {title}")
            return True

        except SlackApiError as e:
            logger.error(f"Slack API error sending reminder: {e}")
            return False
        except Exception as e:
            logger.error(f"Error sending calendar reminder: {e}")
            return False

    def check_important_emails(self) -> int:
        """Check for important emails and send alerts.

        Returns:
            Number of alerts sent.
        """
        logger.debug("Checking important emails...")
        alerts_sent = 0

        # Get users with email alerts enabled
        users = self.settings_store.get_all_enabled_users("email_alerts")

        for settings in users:
            if not settings.important_contacts and not settings.alert_keywords:
                continue

            try:
                sent = self._check_emails_for_user(settings)
                alerts_sent += sent
            except Exception as e:
                logger.error(f"Error checking emails for {settings.user_id}: {e}")

        if alerts_sent > 0:
            logger.info(f"Sent {alerts_sent} email alerts")

        return alerts_sent

    def _check_emails_for_user(self, settings: UserProactiveSettings) -> int:
        """Check emails for a specific user and send alerts.

        Args:
            settings: User's proactive settings.

        Returns:
            Number of alerts sent.
        """
        if not settings.email_alerts_enabled:
            return 0

        if self._is_quiet_hours(settings):
            return 0

        alerts_sent = 0

        try:
            # Get recent emails (last hour)
            from datetime import datetime, timedelta

            since = datetime.now(timezone.utc) - timedelta(hours=1)
            emails = self.multi_google.search_emails_all_accounts(
                query="is:unread",
                max_results=20,
                since=since,
            )

            for email in emails:
                # Check if email matches important criteria
                sender = email.get("from", "").lower()
                subject = email.get("subject", "").lower()
                email_id = email.get("id", "")

                is_important = False

                # Check important contacts
                for contact in settings.important_contacts:
                    if contact.lower() in sender:
                        is_important = True
                        break

                # Check keywords
                if not is_important:
                    for keyword in settings.alert_keywords:
                        if keyword.lower() in subject:
                            is_important = True
                            break

                if is_important:
                    notification_key = f"email_{email_id}"

                    if not self.settings_store.was_notification_sent(
                        settings.user_id, "email_alert", notification_key
                    ):
                        if self._send_email_alert(settings, email):
                            self.settings_store.mark_notification_sent(
                                settings.user_id,
                                "email_alert",
                                notification_key,
                                time.time(),
                            )
                            alerts_sent += 1

        except Exception as e:
            logger.error(f"Error checking emails: {e}")

        return alerts_sent

    def _send_email_alert(
        self, settings: UserProactiveSettings, email: dict
    ) -> bool:
        """Send an email alert to a user.

        Args:
            settings: User's proactive settings.
            email: Email dictionary.

        Returns:
            True if sent successfully, False otherwise.
        """
        try:
            channel_id = self._get_dm_channel(settings.user_id)
            if not channel_id:
                return False

            sender = email.get("from", "Unknown")
            subject = email.get("subject", "(no subject)")
            snippet = email.get("snippet", "")[:200]
            account = email.get("account", "")

            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": ":email: *Important Email*",
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*From:* {sender}\n*Subject:* {subject}",
                    },
                },
            ]

            if snippet:
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f">{snippet}...",
                    },
                })

            if account:
                blocks.append({
                    "type": "context",
                    "elements": [
                        {"type": "mrkdwn", "text": f"_Account: {account}_"},
                    ],
                })

            self.slack_client.chat_postMessage(
                channel=channel_id,
                text=f"Important email from {sender}: {subject}",
                blocks=blocks,
            )

            logger.info(f"Sent email alert to {settings.user_id}: {subject}")
            return True

        except SlackApiError as e:
            logger.error(f"Slack API error sending email alert: {e}")
            return False
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
            return False

    def send_daily_briefings(self) -> int:
        """Send daily briefings to users who have them enabled.

        This should be called by a cron job at the appropriate times.

        Returns:
            Number of briefings sent.
        """
        logger.info("Checking for daily briefings to send...")
        briefings_sent = 0

        users = self.settings_store.get_all_enabled_users("daily_briefing")

        if not users:
            # Send to all authorized users if no settings configured
            users = [
                UserProactiveSettings(user_id=uid)
                for uid in SLACK_AUTHORIZED_USERS
            ]

        today = datetime.now().strftime("%Y-%m-%d")

        for settings in users:
            try:
                # Check if briefing already sent today
                if settings.last_briefing_sent == today:
                    continue

                # Check if it's the right day of week
                current_day = datetime.now().weekday()
                if current_day not in settings.briefing_days:
                    continue

                if self._send_daily_briefing(settings):
                    # Update last briefing sent
                    settings.last_briefing_sent = today
                    self.settings_store.save(settings)
                    briefings_sent += 1

            except Exception as e:
                logger.error(f"Error sending briefing to {settings.user_id}: {e}")

        if briefings_sent > 0:
            logger.info(f"Sent {briefings_sent} daily briefings")

        return briefings_sent

    def _send_daily_briefing(self, settings: UserProactiveSettings) -> bool:
        """Send a daily briefing to a user.

        Args:
            settings: User's proactive settings.

        Returns:
            True if sent successfully, False otherwise.
        """
        try:
            channel_id = self._get_dm_channel(settings.user_id)
            if not channel_id:
                return False

            # Generate briefing data
            briefing = self._generate_briefing()

            # Format the briefing
            formatted = format_briefing(briefing)

            # Add greeting
            hour = datetime.now().hour
            if hour < 12:
                greeting = "Good morning!"
            elif hour < 17:
                greeting = "Good afternoon!"
            else:
                greeting = "Good evening!"

            greeting_block = {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f":wave: *{greeting}* Here's your daily briefing:",
                },
            }

            blocks = [greeting_block]
            if "blocks" in formatted:
                blocks.extend(formatted["blocks"])

            self.slack_client.chat_postMessage(
                channel=channel_id,
                text=f"{greeting} Here's your daily briefing.",
                blocks=blocks,
            )

            logger.info(f"Sent daily briefing to {settings.user_id}")
            return True

        except SlackApiError as e:
            logger.error(f"Slack API error sending briefing: {e}")
            return False
        except Exception as e:
            logger.error(f"Error sending daily briefing: {e}")
            return False

    def _generate_briefing(self) -> dict[str, Any]:
        """Generate briefing data.

        Returns:
            Dictionary with briefing data.
        """
        briefing = {
            "date": datetime.now(timezone.utc).strftime("%A, %B %d, %Y"),
            "events": [],
            "unread_counts": {},
            "open_prs": [],
            "open_issues": [],
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

        return briefing

    def _get_dm_channel(self, user_id: str) -> str | None:
        """Get or open a DM channel with a user.

        Args:
            user_id: Slack user ID.

        Returns:
            Channel ID or None if failed.
        """
        try:
            response = self.slack_client.conversations_open(users=[user_id])
            return response["channel"]["id"]
        except SlackApiError as e:
            logger.error(f"Error opening DM with {user_id}: {e}")
            return None

    def _is_quiet_hours(self, settings: UserProactiveSettings) -> bool:
        """Check if current time is within quiet hours.

        Args:
            settings: User's proactive settings.

        Returns:
            True if in quiet hours, False otherwise.
        """
        if settings.quiet_hours_start is None or settings.quiet_hours_end is None:
            return False

        current_hour = datetime.now().hour

        start = settings.quiet_hours_start
        end = settings.quiet_hours_end

        # Handle overnight quiet hours (e.g., 22:00 - 07:00)
        if start > end:
            return current_hour >= start or current_hour < end
        else:
            return start <= current_hour < end

    def run_all_checks(self) -> dict[str, int]:
        """Run all proactive checks.

        Returns:
            Dictionary with counts of notifications sent.
        """
        return {
            "calendar_reminders": self.check_calendar_reminders(),
            "email_alerts": self.check_important_emails(),
        }

    def cleanup(self) -> None:
        """Clean up old notification records."""
        deleted = self.settings_store.cleanup_old_notifications(max_age_days=7)
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old notification records")
