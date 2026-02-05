"""Proactive settings for user notification preferences."""

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

from ..config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "proactive_settings.db"


@dataclass
class UserProactiveSettings:
    """Proactive notification settings for a user."""

    user_id: str

    # Feature toggles
    calendar_reminders_enabled: bool = True
    email_alerts_enabled: bool = True
    daily_briefing_enabled: bool = True

    # Calendar reminder settings
    reminder_minutes_before: int = 15  # Minutes before meeting
    remind_for_all_day_events: bool = False

    # Daily briefing settings
    briefing_hour: int = 7  # 24-hour format, local time
    briefing_minute: int = 0
    briefing_timezone: str = "America/Los_Angeles"
    briefing_days: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon-Fri

    # Email alert settings
    important_contacts: list[str] = field(default_factory=list)
    alert_keywords: list[str] = field(default_factory=list)
    min_email_priority: str = "high"  # "high", "medium", "low"

    # DM channel for notifications
    dm_channel_id: str | None = None

    # Quiet hours (no notifications)
    quiet_hours_start: int | None = None  # 24-hour format
    quiet_hours_end: int | None = None

    # Last notification tracking (to avoid duplicates)
    last_calendar_check: float = 0.0
    last_email_check: float = 0.0
    last_briefing_sent: str = ""  # Date string YYYY-MM-DD

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "user_id": self.user_id,
            "calendar_reminders_enabled": self.calendar_reminders_enabled,
            "email_alerts_enabled": self.email_alerts_enabled,
            "daily_briefing_enabled": self.daily_briefing_enabled,
            "reminder_minutes_before": self.reminder_minutes_before,
            "remind_for_all_day_events": self.remind_for_all_day_events,
            "briefing_hour": self.briefing_hour,
            "briefing_minute": self.briefing_minute,
            "briefing_timezone": self.briefing_timezone,
            "briefing_days": self.briefing_days,
            "important_contacts": self.important_contacts,
            "alert_keywords": self.alert_keywords,
            "min_email_priority": self.min_email_priority,
            "dm_channel_id": self.dm_channel_id,
            "quiet_hours_start": self.quiet_hours_start,
            "quiet_hours_end": self.quiet_hours_end,
            "last_calendar_check": self.last_calendar_check,
            "last_email_check": self.last_email_check,
            "last_briefing_sent": self.last_briefing_sent,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserProactiveSettings":
        """Create from dictionary."""
        return cls(
            user_id=data["user_id"],
            calendar_reminders_enabled=data.get("calendar_reminders_enabled", True),
            email_alerts_enabled=data.get("email_alerts_enabled", True),
            daily_briefing_enabled=data.get("daily_briefing_enabled", True),
            reminder_minutes_before=data.get("reminder_minutes_before", 15),
            remind_for_all_day_events=data.get("remind_for_all_day_events", False),
            briefing_hour=data.get("briefing_hour", 7),
            briefing_minute=data.get("briefing_minute", 0),
            briefing_timezone=data.get("briefing_timezone", "America/Los_Angeles"),
            briefing_days=data.get("briefing_days", [0, 1, 2, 3, 4]),
            important_contacts=data.get("important_contacts", []),
            alert_keywords=data.get("alert_keywords", []),
            min_email_priority=data.get("min_email_priority", "high"),
            dm_channel_id=data.get("dm_channel_id"),
            quiet_hours_start=data.get("quiet_hours_start"),
            quiet_hours_end=data.get("quiet_hours_end"),
            last_calendar_check=data.get("last_calendar_check", 0.0),
            last_email_check=data.get("last_email_check", 0.0),
            last_briefing_sent=data.get("last_briefing_sent", ""),
        )


class ProactiveSettingsStore:
    """SQLite-backed storage for proactive settings."""

    def __init__(self, db_path: Path | str | None = None):
        """Initialize the proactive settings store.

        Args:
            db_path: Path to SQLite database.
        """
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self._connection() as conn:
            conn.executescript("""
                -- User proactive settings
                CREATE TABLE IF NOT EXISTS proactive_settings (
                    user_id TEXT PRIMARY KEY,
                    settings TEXT NOT NULL
                );

                -- Sent notification tracking (to avoid duplicates)
                CREATE TABLE IF NOT EXISTS sent_notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    notification_type TEXT NOT NULL,
                    notification_key TEXT NOT NULL,
                    sent_at REAL NOT NULL,
                    UNIQUE(user_id, notification_type, notification_key)
                );
                CREATE INDEX IF NOT EXISTS idx_sent_user
                    ON sent_notifications(user_id);
                CREATE INDEX IF NOT EXISTS idx_sent_time
                    ON sent_notifications(sent_at);
            """)

    def get(self, user_id: str) -> UserProactiveSettings:
        """Get settings for a user, creating defaults if not exist.

        Args:
            user_id: Slack user ID.

        Returns:
            UserProactiveSettings instance.
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT settings FROM proactive_settings WHERE user_id = ?",
                (user_id,),
            ).fetchone()

            if row:
                try:
                    data = json.loads(row["settings"])
                    data["user_id"] = user_id
                    return UserProactiveSettings.from_dict(data)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Error loading settings for {user_id}: {e}")

            # Return defaults
            return UserProactiveSettings(user_id=user_id)

    def save(self, settings: UserProactiveSettings) -> None:
        """Save settings for a user.

        Args:
            settings: UserProactiveSettings to save.
        """
        with self._connection() as conn:
            settings_json = json.dumps(settings.to_dict())
            conn.execute(
                """
                INSERT INTO proactive_settings (user_id, settings)
                VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET settings = excluded.settings
                """,
                (settings.user_id, settings_json),
            )

    def delete(self, user_id: str) -> bool:
        """Delete settings for a user.

        Args:
            user_id: Slack user ID.

        Returns:
            True if deleted, False if not found.
        """
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM proactive_settings WHERE user_id = ?",
                (user_id,),
            )
            return cursor.rowcount > 0

    def get_all_enabled_users(self, feature: str) -> list[UserProactiveSettings]:
        """Get all users with a specific feature enabled.

        Args:
            feature: Feature name ("calendar_reminders", "email_alerts", "daily_briefing").

        Returns:
            List of UserProactiveSettings with the feature enabled.
        """
        enabled_users = []

        with self._connection() as conn:
            rows = conn.execute("SELECT user_id, settings FROM proactive_settings").fetchall()

            for row in rows:
                try:
                    data = json.loads(row["settings"])
                    data["user_id"] = row["user_id"]
                    settings = UserProactiveSettings.from_dict(data)

                    # Check if feature is enabled
                    feature_enabled = {
                        "calendar_reminders": settings.calendar_reminders_enabled,
                        "email_alerts": settings.email_alerts_enabled,
                        "daily_briefing": settings.daily_briefing_enabled,
                    }.get(feature, False)

                    if feature_enabled:
                        enabled_users.append(settings)

                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Error loading settings: {e}")

        return enabled_users

    def mark_notification_sent(
        self,
        user_id: str,
        notification_type: str,
        notification_key: str,
        sent_at: float,
    ) -> bool:
        """Mark a notification as sent to avoid duplicates.

        Args:
            user_id: Slack user ID.
            notification_type: Type of notification (e.g., "calendar_reminder").
            notification_key: Unique key for the notification (e.g., event_id).
            sent_at: Timestamp when sent.

        Returns:
            True if marked (new), False if already existed.
        """
        with self._connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO sent_notifications
                    (user_id, notification_type, notification_key, sent_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (user_id, notification_type, notification_key, sent_at),
                )
                return True
            except sqlite3.IntegrityError:
                return False

    def was_notification_sent(
        self,
        user_id: str,
        notification_type: str,
        notification_key: str,
    ) -> bool:
        """Check if a notification was already sent.

        Args:
            user_id: Slack user ID.
            notification_type: Type of notification.
            notification_key: Unique key for the notification.

        Returns:
            True if already sent, False otherwise.
        """
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM sent_notifications
                WHERE user_id = ? AND notification_type = ? AND notification_key = ?
                """,
                (user_id, notification_type, notification_key),
            ).fetchone()
            return row is not None

    def cleanup_old_notifications(self, max_age_days: int = 7) -> int:
        """Clean up old sent notification records.

        Args:
            max_age_days: Maximum age in days.

        Returns:
            Number of records deleted.
        """
        import time

        cutoff = time.time() - (max_age_days * 24 * 60 * 60)

        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM sent_notifications WHERE sent_at < ?",
                (cutoff,),
            )
            return cursor.rowcount

    def has_any_settings(self) -> bool:
        """Check if any users have saved settings.

        Returns:
            True if at least one user has settings, False otherwise.
        """
        with self._connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM proactive_settings"
            ).fetchone()[0]
            return count > 0

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about proactive settings.

        Returns:
            Dictionary with stats.
        """
        with self._connection() as conn:
            total_users = conn.execute(
                "SELECT COUNT(*) FROM proactive_settings"
            ).fetchone()[0]

            # Count enabled features
            rows = conn.execute("SELECT settings FROM proactive_settings").fetchall()

            calendar_enabled = 0
            email_enabled = 0
            briefing_enabled = 0

            for row in rows:
                try:
                    data = json.loads(row["settings"])
                    if data.get("calendar_reminders_enabled", True):
                        calendar_enabled += 1
                    if data.get("email_alerts_enabled", True):
                        email_enabled += 1
                    if data.get("daily_briefing_enabled", True):
                        briefing_enabled += 1
                except json.JSONDecodeError:
                    pass

            return {
                "total_users": total_users,
                "calendar_reminders_enabled": calendar_enabled,
                "email_alerts_enabled": email_enabled,
                "daily_briefing_enabled": briefing_enabled,
            }
