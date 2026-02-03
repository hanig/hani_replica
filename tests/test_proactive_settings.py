"""Tests for proactive settings."""

import tempfile
import time
from pathlib import Path

import pytest

from src.bot.proactive_settings import (
    ProactiveSettingsStore,
    UserProactiveSettings,
)


class TestUserProactiveSettings:
    """Tests for UserProactiveSettings dataclass."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = UserProactiveSettings(user_id="U123")

        assert settings.user_id == "U123"
        assert settings.calendar_reminders_enabled is True
        assert settings.email_alerts_enabled is True
        assert settings.daily_briefing_enabled is True
        assert settings.reminder_minutes_before == 15
        assert settings.briefing_hour == 7
        assert settings.briefing_days == [0, 1, 2, 3, 4]

    def test_to_dict(self):
        """Test conversion to dictionary."""
        settings = UserProactiveSettings(
            user_id="U123",
            calendar_reminders_enabled=False,
            reminder_minutes_before=30,
        )

        data = settings.to_dict()

        assert data["user_id"] == "U123"
        assert data["calendar_reminders_enabled"] is False
        assert data["reminder_minutes_before"] == 30

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "user_id": "U123",
            "calendar_reminders_enabled": False,
            "briefing_hour": 8,
            "important_contacts": ["john@example.com"],
        }

        settings = UserProactiveSettings.from_dict(data)

        assert settings.user_id == "U123"
        assert settings.calendar_reminders_enabled is False
        assert settings.briefing_hour == 8
        assert settings.important_contacts == ["john@example.com"]

    def test_roundtrip(self):
        """Test to_dict and from_dict roundtrip."""
        original = UserProactiveSettings(
            user_id="U123",
            calendar_reminders_enabled=False,
            email_alerts_enabled=True,
            daily_briefing_enabled=False,
            reminder_minutes_before=30,
            briefing_hour=9,
            important_contacts=["alice@example.com", "bob@example.com"],
            alert_keywords=["urgent", "important"],
        )

        data = original.to_dict()
        restored = UserProactiveSettings.from_dict(data)

        assert restored.user_id == original.user_id
        assert restored.calendar_reminders_enabled == original.calendar_reminders_enabled
        assert restored.reminder_minutes_before == original.reminder_minutes_before
        assert restored.important_contacts == original.important_contacts
        assert restored.alert_keywords == original.alert_keywords


class TestProactiveSettingsStore:
    """Tests for ProactiveSettingsStore class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield Path(f.name)

    @pytest.fixture
    def store(self, temp_db):
        """Create a ProactiveSettingsStore instance."""
        return ProactiveSettingsStore(temp_db)

    def test_get_default_settings(self, store):
        """Test getting default settings for new user."""
        settings = store.get("U123")

        assert settings.user_id == "U123"
        assert settings.calendar_reminders_enabled is True  # Default

    def test_save_and_get(self, store):
        """Test saving and retrieving settings."""
        settings = UserProactiveSettings(
            user_id="U123",
            calendar_reminders_enabled=False,
            reminder_minutes_before=30,
        )

        store.save(settings)
        loaded = store.get("U123")

        assert loaded.calendar_reminders_enabled is False
        assert loaded.reminder_minutes_before == 30

    def test_update_settings(self, store):
        """Test updating existing settings."""
        settings = UserProactiveSettings(user_id="U123")
        store.save(settings)

        settings.reminder_minutes_before = 20
        settings.important_contacts = ["test@example.com"]
        store.save(settings)

        loaded = store.get("U123")
        assert loaded.reminder_minutes_before == 20
        assert loaded.important_contacts == ["test@example.com"]

    def test_delete(self, store):
        """Test deleting settings."""
        settings = UserProactiveSettings(user_id="U123")
        store.save(settings)

        deleted = store.delete("U123")
        assert deleted is True

        # Should get defaults after delete
        loaded = store.get("U123")
        assert loaded.calendar_reminders_enabled is True

    def test_delete_nonexistent(self, store):
        """Test deleting non-existent settings."""
        deleted = store.delete("U999")
        assert deleted is False

    def test_get_all_enabled_users(self, store):
        """Test getting users with a feature enabled."""
        # User 1: all enabled
        settings1 = UserProactiveSettings(user_id="U1")
        store.save(settings1)

        # User 2: calendar disabled
        settings2 = UserProactiveSettings(
            user_id="U2",
            calendar_reminders_enabled=False,
        )
        store.save(settings2)

        # User 3: all enabled
        settings3 = UserProactiveSettings(user_id="U3")
        store.save(settings3)

        enabled = store.get_all_enabled_users("calendar_reminders")
        user_ids = [s.user_id for s in enabled]

        assert "U1" in user_ids
        assert "U2" not in user_ids
        assert "U3" in user_ids

    def test_mark_notification_sent(self, store):
        """Test marking a notification as sent."""
        result = store.mark_notification_sent(
            user_id="U123",
            notification_type="calendar_reminder",
            notification_key="event123",
            sent_at=time.time(),
        )

        assert result is True

        # Duplicate should return False
        result2 = store.mark_notification_sent(
            user_id="U123",
            notification_type="calendar_reminder",
            notification_key="event123",
            sent_at=time.time(),
        )

        assert result2 is False

    def test_was_notification_sent(self, store):
        """Test checking if notification was sent."""
        # Should be False initially
        assert not store.was_notification_sent("U123", "calendar_reminder", "event123")

        # Mark as sent
        store.mark_notification_sent("U123", "calendar_reminder", "event123", time.time())

        # Should be True now
        assert store.was_notification_sent("U123", "calendar_reminder", "event123")

        # Different notification should be False
        assert not store.was_notification_sent("U123", "calendar_reminder", "event456")

    def test_cleanup_old_notifications(self, store):
        """Test cleaning up old notifications."""
        # Add some notifications
        old_time = time.time() - (30 * 24 * 60 * 60)  # 30 days ago
        store.mark_notification_sent("U123", "test", "old1", old_time)
        store.mark_notification_sent("U123", "test", "new1", time.time())

        # Clean up with 7 days
        deleted = store.cleanup_old_notifications(max_age_days=7)

        assert deleted == 1
        assert not store.was_notification_sent("U123", "test", "old1")
        assert store.was_notification_sent("U123", "test", "new1")

    def test_get_stats(self, store):
        """Test getting statistics."""
        settings1 = UserProactiveSettings(user_id="U1")
        settings2 = UserProactiveSettings(
            user_id="U2",
            calendar_reminders_enabled=False,
        )

        store.save(settings1)
        store.save(settings2)

        stats = store.get_stats()

        assert stats["total_users"] == 2
        assert stats["calendar_reminders_enabled"] == 1
        assert stats["email_alerts_enabled"] == 2
        assert stats["daily_briefing_enabled"] == 2
