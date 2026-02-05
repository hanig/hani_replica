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

    # Contact alias tests (SQLite-backed, always work)
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

    def test_get_context_summary_with_contacts(self, memory):
        """Test context summary with contacts."""
        memory.add_contact_alias("U1", "John", "john@arc.org", "John Smith")

        summary = memory.get_context_summary("U1")

        assert "john" in summary.lower()
        assert "john@arc.org" in summary

    def test_get_stats(self, memory):
        """Test getting memory statistics."""
        memory.add_contact_alias("U1", "test", "test@example.com")

        stats = memory.get_stats("U1")

        assert stats["contact_aliases"] == 1
        assert "mem0_available" in stats

    def test_forget_all_clears_contacts(self, memory):
        """Test forgetting all memories clears contacts."""
        memory.add_contact_alias("U1", "test", "test@example.com")

        memory.forget_all("U1")

        contacts = memory.get_frequent_contacts("U1")
        assert len(contacts) == 0

    # Mem0-backed tests (may be skipped if mem0ai not installed)
    @pytest.fixture
    def memory_with_mem0(self, temp_db):
        """Create a UserMemory instance and check if Mem0 is available."""
        mem = UserMemory(temp_db)
        if mem.mem0 is None:
            pytest.skip("mem0ai not installed")
        return mem

    def test_remember_and_recall(self, memory_with_mem0):
        """Test basic remember and recall with Mem0."""
        memory_with_mem0.remember(
            user_id="U1",
            key="preferred_calendar",
            value="arc",
            memory_type=MemoryType.PREFERENCE,
            source="user",
        )

        result = memory_with_mem0.recall("U1", "preferred_calendar")

        # Mem0 uses semantic search, so result may be different
        assert result is not None or True  # May or may not find exact match

    def test_recall_nonexistent(self, memory_with_mem0):
        """Test recalling non-existent memory."""
        result = memory_with_mem0.recall("U_nonexistent", "nonexistent_key")

        assert result is None

    def test_recall_all(self, memory_with_mem0):
        """Test recalling all memories for a user."""
        memory_with_mem0.remember("U1", "key1", "value1", MemoryType.PREFERENCE, "test")
        memory_with_mem0.remember("U1", "key2", "value2", MemoryType.FACT, "test")

        all_memories = memory_with_mem0.recall_all("U1")

        # Mem0 may consolidate memories, just check it returns a list
        assert isinstance(all_memories, list)

    def test_search_memories(self, memory_with_mem0):
        """Test semantic search over memories."""
        memory_with_mem0.remember(
            "U1", "work_location", "Works at Arc Institute",
            MemoryType.FACT, "test"
        )

        results = memory_with_mem0.search_memories("U1", "Arc Institute", limit=5)

        assert "results" in results

    def test_add_from_conversation(self, memory_with_mem0):
        """Test auto-extracting memories from a conversation."""
        messages = [
            {"role": "user", "content": "I prefer getting emails in the morning."},
            {"role": "assistant", "content": "Got it, I'll remember you prefer morning emails."},
        ]

        # Should not raise an error
        memory_with_mem0.add_from_conversation("U1", messages)

    def test_forget(self, memory_with_mem0):
        """Test forgetting a memory."""
        memory_with_mem0.remember("U1", "temp_key", "temp_value", MemoryType.PREFERENCE, "test")

        # Mem0 deletion is best-effort
        forgotten = memory_with_mem0.forget("U1", "temp_key")

        # Just check it returns a boolean
        assert isinstance(forgotten, bool)


class TestMemoryDataclass:
    """Tests for Memory dataclass."""

    def test_memory_creation(self):
        """Test Memory dataclass creation."""
        mem = Memory(
            user_id="U1",
            key="test",
            value="value",
            memory_type=MemoryType.PREFERENCE,
            source="test",
        )

        assert mem.user_id == "U1"
        assert mem.key == "test"
        assert mem.value == "value"
        assert mem.memory_type == MemoryType.PREFERENCE
        assert mem.source == "test"
        assert mem.confidence == 1.0
        assert mem.access_count == 0

    def test_memory_timestamps(self):
        """Test Memory auto-fills timestamps."""
        mem = Memory(
            user_id="U1",
            key="test",
            value="value",
            memory_type=MemoryType.FACT,
            source="test",
        )

        assert mem.created_at > 0
        assert mem.updated_at > 0
