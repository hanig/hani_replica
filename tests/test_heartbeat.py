"""Tests for heartbeat manager."""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.bot.heartbeat import HeartbeatManager
from src.bot.proactive_settings import ProactiveSettingsStore, UserProactiveSettings


class TestHeartbeatManager:
    """Tests for HeartbeatManager class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield Path(f.name)

    @pytest.fixture
    def mock_slack_client(self):
        """Create a mock Slack client."""
        client = MagicMock()
        client.conversations_open.return_value = {"channel": {"id": "D123"}}
        client.chat_postMessage.return_value = {"ok": True}
        return client

    @pytest.fixture
    def heartbeat(self, temp_db, mock_slack_client):
        """Create a HeartbeatManager instance."""
        store = ProactiveSettingsStore(temp_db)
        return HeartbeatManager(
            slack_client=mock_slack_client,
            settings_store=store,
        )

    def test_init(self, heartbeat):
        """Test heartbeat manager initialization."""
        assert heartbeat.slack_client is not None
        assert heartbeat.settings_store is not None

    def test_is_quiet_hours_no_config(self, heartbeat):
        """Test quiet hours check when not configured."""
        settings = UserProactiveSettings(user_id="U123")

        result = heartbeat._is_quiet_hours(settings)

        assert result is False

    def test_is_quiet_hours_normal_range(self, heartbeat):
        """Test quiet hours with normal time range."""
        settings = UserProactiveSettings(
            user_id="U123",
            quiet_hours_start=22,
            quiet_hours_end=7,
        )

        # Mock current time
        with patch("src.bot.heartbeat.datetime") as mock_datetime:
            # Test during quiet hours (23:00)
            mock_datetime.now.return_value = datetime(2024, 1, 1, 23, 0)
            assert heartbeat._is_quiet_hours(settings) is True

            # Test outside quiet hours (10:00)
            mock_datetime.now.return_value = datetime(2024, 1, 1, 10, 0)
            assert heartbeat._is_quiet_hours(settings) is False

    def test_get_dm_channel(self, heartbeat, mock_slack_client):
        """Test getting DM channel."""
        channel = heartbeat._get_dm_channel("U123")

        assert channel == "D123"
        mock_slack_client.conversations_open.assert_called_once_with(users=["U123"])

    def test_get_dm_channel_error(self, heartbeat, mock_slack_client):
        """Test getting DM channel with error."""
        from slack_sdk.errors import SlackApiError

        mock_slack_client.conversations_open.side_effect = SlackApiError(
            message="Error", response={"error": "user_not_found"}
        )

        channel = heartbeat._get_dm_channel("U123")

        assert channel is None

    def test_send_calendar_reminder(self, heartbeat, mock_slack_client):
        """Test sending a calendar reminder."""
        settings = UserProactiveSettings(user_id="U123")
        event = {
            "id": "event123",
            "summary": "Team Meeting",
            "start": datetime.now(timezone.utc) + timedelta(minutes=15),
            "location": "Conference Room A",
        }

        result = heartbeat._send_calendar_reminder(settings, event)

        assert result is True
        mock_slack_client.chat_postMessage.assert_called_once()

        # Check message content
        call_args = mock_slack_client.chat_postMessage.call_args
        assert call_args.kwargs["channel"] == "D123"
        assert "Team Meeting" in call_args.kwargs["text"]

    def test_send_calendar_reminder_with_meeting_link(self, heartbeat, mock_slack_client):
        """Test calendar reminder includes meeting link."""
        settings = UserProactiveSettings(user_id="U123")
        event = {
            "id": "event123",
            "summary": "Video Call",
            "start": datetime.now(timezone.utc) + timedelta(minutes=15),
            "hangout_link": "https://meet.google.com/abc-def-ghi",
        }

        heartbeat._send_calendar_reminder(settings, event)

        call_args = mock_slack_client.chat_postMessage.call_args
        blocks = call_args.kwargs.get("blocks", [])

        # Check that meeting link is included in blocks
        block_text = str(blocks)
        assert "Join Meeting" in block_text or "meet.google.com" in block_text

    def test_send_email_alert(self, heartbeat, mock_slack_client):
        """Test sending an email alert."""
        settings = UserProactiveSettings(user_id="U123")
        email = {
            "id": "email123",
            "from": "important@example.com",
            "subject": "Urgent: Please Review",
            "snippet": "This is an important email that needs your attention...",
            "account": "arc",
        }

        result = heartbeat._send_email_alert(settings, email)

        assert result is True
        mock_slack_client.chat_postMessage.assert_called_once()

        call_args = mock_slack_client.chat_postMessage.call_args
        assert "Urgent: Please Review" in call_args.kwargs["text"]

    def test_send_daily_briefing(self, heartbeat, mock_slack_client):
        """Test sending daily briefing."""
        settings = UserProactiveSettings(user_id="U123")

        # Mock the multi_google and github_client
        heartbeat._multi_google = MagicMock()
        heartbeat._multi_google.get_all_calendars_today.return_value = [
            {"summary": "Morning Meeting", "start": "10:00 AM"},
        ]
        heartbeat._multi_google.get_unread_counts.return_value = {"arc": 5, "personal": 2}

        heartbeat._github_client = MagicMock()
        heartbeat._github_client.get_my_prs.return_value = []
        heartbeat._github_client.get_my_issues.return_value = []

        result = heartbeat._send_daily_briefing(settings)

        assert result is True
        mock_slack_client.chat_postMessage.assert_called_once()

    def test_check_calendar_reminders_no_users(self, heartbeat):
        """Test calendar reminder check with no users."""
        # With no authorized users and no settings, should return 0
        with patch.object(heartbeat.settings_store, "get_all_enabled_users", return_value=[]):
            with patch("src.bot.heartbeat.SLACK_AUTHORIZED_USERS", []):
                result = heartbeat.check_calendar_reminders()
                assert result == 0

    def test_check_calendar_reminders_disabled(self, heartbeat):
        """Test calendar reminder check when disabled."""
        settings = UserProactiveSettings(
            user_id="U123",
            calendar_reminders_enabled=False,
        )
        heartbeat.settings_store.save(settings)

        result = heartbeat._check_calendar_for_user(settings)

        assert result == 0

    def test_run_all_checks(self, heartbeat):
        """Test running all proactive checks."""
        # Mock the check methods
        heartbeat.check_calendar_reminders = MagicMock(return_value=2)
        heartbeat.check_important_emails = MagicMock(return_value=1)

        result = heartbeat.run_all_checks()

        assert result["calendar_reminders"] == 2
        assert result["email_alerts"] == 1

    def test_cleanup(self, heartbeat):
        """Test cleanup method."""
        # Add some old notifications
        import time

        old_time = time.time() - (30 * 24 * 60 * 60)  # 30 days ago
        heartbeat.settings_store.mark_notification_sent(
            "U123", "test", "old1", old_time
        )

        heartbeat.cleanup()

        # Old notification should be deleted
        assert not heartbeat.settings_store.was_notification_sent("U123", "test", "old1")

    def test_generate_briefing(self, heartbeat):
        """Test briefing generation."""
        # Mock the integrations
        heartbeat._multi_google = MagicMock()
        heartbeat._multi_google.get_all_calendars_today.return_value = [
            {"summary": "Meeting 1"},
            {"summary": "Meeting 2"},
        ]
        heartbeat._multi_google.get_unread_counts.return_value = {"arc": 10}

        heartbeat._github_client = MagicMock()
        heartbeat._github_client.get_my_prs.return_value = [{"title": "PR 1"}]
        heartbeat._github_client.get_my_issues.return_value = []

        briefing = heartbeat._generate_briefing()

        assert "date" in briefing
        assert len(briefing["events"]) == 2
        assert briefing["unread_counts"]["arc"] == 10
        assert len(briefing["open_prs"]) == 1

    def test_generate_briefing_handles_errors(self, heartbeat):
        """Test that briefing generation handles integration errors."""
        # Mock integrations to raise errors
        heartbeat._multi_google = MagicMock()
        heartbeat._multi_google.get_all_calendars_today.side_effect = Exception("API Error")
        heartbeat._multi_google.get_unread_counts.side_effect = Exception("API Error")

        heartbeat._github_client = MagicMock()
        heartbeat._github_client.get_my_prs.side_effect = Exception("API Error")
        heartbeat._github_client.get_my_issues.side_effect = Exception("API Error")

        # Should not raise, just return empty data
        briefing = heartbeat._generate_briefing()

        assert "date" in briefing
        assert briefing["events"] == []
        assert briefing["unread_counts"] == {}
