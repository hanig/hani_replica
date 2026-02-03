"""Tests for user memory storage."""

import tempfile
from pathlib import Path

import pytest

from src.bot.user_memory import UserMemory, MemoryType, Memory


class TestUserMemory:
    """Tests for UserMemory class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield Path(f.name)

    @pytest.fixture
    def memory(self, temp_db):
        """Create a UserMemory instance."""
        return UserMemory(temp_db)

    def test_remember_and_recall(self, memory):
        """Test basic remember and recall."""
        memory.remember(
            user_id="U1",
            key="preferred_calendar",
            value="arc",
            memory_type=MemoryType.PREFERENCE,
            source="user",
        )

        result = memory.recall("U1", "preferred_calendar")

        assert result == "arc"

    def test_remember_updates_existing(self, memory):
        """Test that remember updates existing values."""
        memory.remember("U1", "key", "value1", MemoryType.PREFERENCE, "test")
        memory.remember("U1", "key", "value2", MemoryType.PREFERENCE, "test")

        result = memory.recall("U1", "key")

        assert result == "value2"

    def test_recall_nonexistent(self, memory):
        """Test recalling non-existent memory."""
        result = memory.recall("U1", "nonexistent")

        assert result is None

    def test_recall_with_type_filter(self, memory):
        """Test recalling with memory type filter."""
        memory.remember("U1", "test", "pref_value", MemoryType.PREFERENCE, "test")
        memory.remember("U1", "test", "fact_value", MemoryType.FACT, "test")

        pref_result = memory.recall("U1", "test", MemoryType.PREFERENCE)
        fact_result = memory.recall("U1", "test", MemoryType.FACT)

        assert pref_result == "pref_value"
        assert fact_result == "fact_value"

    def test_recall_all(self, memory):
        """Test recalling all memories for a user."""
        memory.remember("U1", "key1", "value1", MemoryType.PREFERENCE, "test")
        memory.remember("U1", "key2", "value2", MemoryType.FACT, "test")
        memory.remember("U2", "key3", "value3", MemoryType.PREFERENCE, "test")

        all_memories = memory.recall_all("U1")

        assert len(all_memories) == 2

    def test_recall_all_with_type(self, memory):
        """Test recalling all memories of a specific type."""
        memory.remember("U1", "pref1", "v1", MemoryType.PREFERENCE, "test")
        memory.remember("U1", "pref2", "v2", MemoryType.PREFERENCE, "test")
        memory.remember("U1", "fact1", "v3", MemoryType.FACT, "test")

        prefs = memory.recall_all("U1", MemoryType.PREFERENCE)

        assert len(prefs) == 2

    def test_forget(self, memory):
        """Test forgetting a memory."""
        memory.remember("U1", "key", "value", MemoryType.PREFERENCE, "test")

        forgotten = memory.forget("U1", "key")

        assert forgotten is True
        assert memory.recall("U1", "key") is None

    def test_forget_nonexistent(self, memory):
        """Test forgetting non-existent memory."""
        forgotten = memory.forget("U1", "nonexistent")

        assert forgotten is False

    def test_forget_all(self, memory):
        """Test forgetting all memories for a user."""
        memory.remember("U1", "key1", "v1", MemoryType.PREFERENCE, "test")
        memory.remember("U1", "key2", "v2", MemoryType.FACT, "test")

        count = memory.forget_all("U1")

        assert count == 2
        assert len(memory.recall_all("U1")) == 0

    def test_json_serialization(self, memory):
        """Test that complex values are JSON serialized."""
        complex_value = {"nested": {"key": "value"}, "list": [1, 2, 3]}
        memory.remember("U1", "complex", complex_value, MemoryType.FACT, "test")

        result = memory.recall("U1", "complex")

        assert result == complex_value

    # Contact alias tests
    def test_add_contact_alias(self, memory):
        """Test adding a contact alias."""
        memory.add_contact_alias("U1", "John", "john.smith@example.com", "John Smith")

        contact = memory.resolve_contact("U1", "John")

        assert contact is not None
        assert contact["email"] == "john.smith@example.com"
        assert contact["name"] == "John Smith"

    def test_resolve_contact_case_insensitive(self, memory):
        """Test that contact resolution is case insensitive."""
        memory.add_contact_alias("U1", "John", "john@example.com")

        contact = memory.resolve_contact("U1", "JOHN")

        assert contact is not None
        assert contact["email"] == "john@example.com"

    def test_resolve_nonexistent_contact(self, memory):
        """Test resolving non-existent contact."""
        contact = memory.resolve_contact("U1", "Nobody")

        assert contact is None

    def test_get_frequent_contacts(self, memory):
        """Test getting frequently used contacts."""
        memory.add_contact_alias("U1", "alice", "alice@example.com")
        memory.add_contact_alias("U1", "bob", "bob@example.com")

        # Use alice multiple times
        memory.resolve_contact("U1", "alice")
        memory.resolve_contact("U1", "alice")
        memory.resolve_contact("U1", "alice")

        contacts = memory.get_frequent_contacts("U1")

        assert len(contacts) == 2
        assert contacts[0]["alias"] == "alice"  # Most used

    def test_get_context_summary_empty(self, memory):
        """Test context summary with no memories."""
        summary = memory.get_context_summary("U1")

        assert summary == ""

    def test_get_context_summary_with_data(self, memory):
        """Test context summary with memories."""
        memory.remember("U1", "preferred_calendar", "arc", MemoryType.PREFERENCE, "user")
        # For facts, the value should be the fact itself
        memory.remember("U1", "employer", "Works at Arc Institute", MemoryType.FACT, "user")
        memory.add_contact_alias("U1", "John", "john@arc.org", "John Smith")

        summary = memory.get_context_summary("U1")

        assert "preferred_calendar" in summary
        assert "Arc Institute" in summary
        assert "john" in summary.lower()

    def test_get_stats(self, memory):
        """Test getting memory statistics."""
        memory.remember("U1", "key1", "v1", MemoryType.PREFERENCE, "test")
        memory.remember("U1", "key2", "v2", MemoryType.FACT, "test")
        memory.add_contact_alias("U1", "test", "test@example.com")

        stats = memory.get_stats("U1")

        assert stats["total_memories"] == 2
        assert stats["contact_aliases"] == 1

    def test_string_memory_type(self, memory):
        """Test using string memory type."""
        memory.remember("U1", "key", "value", "preference", "test")

        result = memory.recall("U1", "key", "preference")

        assert result == "value"
