"""Tests for feedback loop and interaction learning."""

import tempfile
from pathlib import Path

import pytest

from src.bot.feedback_loop import FeedbackLoop, FeedbackEvent, FeedbackType


class TestFeedbackLoop:
    """Tests for FeedbackLoop class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield Path(f.name)

    @pytest.fixture
    def feedback(self, temp_db):
        """Create a FeedbackLoop instance."""
        return FeedbackLoop(temp_db)

    def test_record_feedback(self, feedback):
        """Test recording a feedback event."""
        event = FeedbackEvent(
            user_id="U1",
            query="search for emails",
            feedback_type=FeedbackType.RESULT_CLICK,
            result_id="email123",
            result_source="email",
        )

        feedback.record_feedback(event)

        stats = feedback.get_feedback_stats("U1")
        assert stats["total_feedback_events"] == 1

    def test_record_result_click(self, feedback):
        """Test convenience method for recording result clicks."""
        feedback.record_result_click(
            user_id="U1",
            query="calendar today",
            result_id="event123",
            result_source="calendar",
        )

        stats = feedback.get_feedback_stats("U1")
        assert stats["by_type"].get("result_click", 0) == 1

    def test_record_correction(self, feedback):
        """Test recording a user correction."""
        feedback.record_correction(
            user_id="U1",
            original_query="email John",
            corrected_value="john.smith@arc.org",
            correction_type="contact",
            original_result="john.doe@gmail.com",
        )

        corrections = feedback.get_corrections("U1")
        assert len(corrections) == 1
        assert corrections[0]["corrected_value"] == "john.smith@arc.org"

    def test_get_corrections_with_type(self, feedback):
        """Test getting corrections filtered by type."""
        feedback.record_correction("U1", "q1", "v1", "contact")
        feedback.record_correction("U1", "q2", "v2", "date")
        feedback.record_correction("U1", "q3", "v3", "contact")

        contact_corrections = feedback.get_corrections("U1", "contact")

        assert len(contact_corrections) == 2

    def test_record_query_pattern(self, feedback):
        """Test recording query patterns."""
        feedback.record_query_pattern("U1", "calendar today", "calendar_check", True)
        feedback.record_query_pattern("U1", "calendar today", "calendar_check", True)
        feedback.record_query_pattern("U1", "search emails", "email_search", True)

        patterns = feedback.get_common_patterns("U1")

        assert len(patterns) == 2
        # "calendar today" should be first (count=2)
        assert patterns[0]["pattern"] == "calendar today"
        assert patterns[0]["count"] == 2

    def test_relevance_scores(self, feedback):
        """Test relevance score calculation."""
        # Click on email results multiple times
        for _ in range(5):
            feedback.record_result_click("U1", "query", "r1", "email")

        # Click on calendar once
        feedback.record_result_click("U1", "query", "r2", "calendar")

        scores = feedback.get_relevance_scores("U1")

        assert "email" in scores
        assert "calendar" in scores
        assert scores["email"] > scores["calendar"]

    def test_source_ranking(self, feedback):
        """Test source ranking by relevance."""
        # Build up different relevance for sources
        for _ in range(5):
            feedback.record_result_click("U1", "q", "r", "github")
        for _ in range(3):
            feedback.record_result_click("U1", "q", "r", "email")
        for _ in range(1):
            feedback.record_result_click("U1", "q", "r", "calendar")

        ranking = feedback.get_source_ranking("U1")

        assert ranking[0] == "github"  # Most clicks
        assert ranking[1] == "email"
        assert ranking[2] == "calendar"

    def test_boost_results(self, feedback):
        """Test boosting search results by relevance."""
        # Build relevance: email > calendar > slack
        for _ in range(5):
            feedback.record_result_click("U1", "q", "r", "email")
        for _ in range(2):
            feedback.record_result_click("U1", "q", "r", "calendar")

        results = [
            {"id": "1", "source": "slack"},
            {"id": "2", "source": "calendar"},
            {"id": "3", "source": "email"},
        ]

        boosted = feedback.boost_results("U1", results)

        # Email should be first (highest relevance)
        assert boosted[0]["source"] == "email"

    def test_positive_negative_feedback(self, feedback):
        """Test that positive and negative feedback affect scores."""
        # Positive feedback for email
        feedback.record_feedback(FeedbackEvent(
            user_id="U1",
            query="q",
            feedback_type=FeedbackType.EXPLICIT_POSITIVE,
            result_source="email",
        ))

        # Negative feedback for calendar
        feedback.record_feedback(FeedbackEvent(
            user_id="U1",
            query="q",
            feedback_type=FeedbackType.EXPLICIT_NEGATIVE,
            result_source="calendar",
        ))

        scores = feedback.get_relevance_scores("U1")

        assert scores.get("email", 0) > scores.get("calendar", 1)

    def test_get_feedback_stats(self, feedback):
        """Test getting feedback statistics."""
        feedback.record_result_click("U1", "q1", "r1", "email")
        feedback.record_result_click("U1", "q2", "r2", "calendar")
        feedback.record_correction("U1", "q", "v", "contact")
        feedback.record_query_pattern("U1", "pattern", "intent", True)

        stats = feedback.get_feedback_stats("U1")

        assert stats["total_feedback_events"] >= 2
        assert stats["total_corrections"] == 1
        assert stats["total_patterns"] == 1

    def test_feedback_stats_global(self, feedback):
        """Test getting global feedback statistics."""
        feedback.record_result_click("U1", "q", "r", "email")
        feedback.record_result_click("U2", "q", "r", "calendar")

        global_stats = feedback.get_feedback_stats()

        assert global_stats["total_feedback_events"] == 2

    def test_cleanup_old_events(self, feedback):
        """Test cleaning up old feedback events."""
        # Record an event
        feedback.record_result_click("U1", "q", "r", "email")

        # Cleanup with 0 days should remove everything
        deleted = feedback.cleanup_old_events(max_age_days=0)

        assert deleted >= 1

    def test_query_pattern_success_rate(self, feedback):
        """Test that success rate is tracked for patterns."""
        # Record same pattern with different success
        feedback.record_query_pattern("U1", "test query", "search", True)
        feedback.record_query_pattern("U1", "test query", "search", True)
        feedback.record_query_pattern("U1", "test query", "search", False)

        patterns = feedback.get_common_patterns("U1")

        assert len(patterns) == 1
        assert patterns[0]["count"] == 3
        # Success rate should be between 0 and 1
        assert 0 <= patterns[0]["success_rate"] <= 1
