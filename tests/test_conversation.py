"""Tests for conversation management."""

import tempfile
import time
from pathlib import Path

import pytest

from src.bot.conversation import (
    ConversationContext,
    ConversationManager,
    ConversationStore,
)


class TestConversationContext:
    """Tests for ConversationContext class."""

    def test_create_context(self):
        """Test creating a conversation context."""
        context = ConversationContext(
            user_id="U123",
            channel_id="C456",
            thread_ts="123.456",
        )

        assert context.user_id == "U123"
        assert context.channel_id == "C456"
        assert context.thread_ts == "123.456"

    def test_add_message(self):
        """Test adding messages to history."""
        context = ConversationContext("U1", "C1")

        context.add_message("user", "Hello")
        context.add_message("assistant", "Hi there!")

        assert len(context.history) == 2
        assert context.history[0]["role"] == "user"
        assert context.history[1]["role"] == "assistant"

    def test_history_trimming(self):
        """Test that history is trimmed when too long."""
        context = ConversationContext("U1", "C1")

        # Add more than MAX_HISTORY_LENGTH messages
        for i in range(30):
            context.add_message("user", f"Message {i}")

        assert len(context.history) <= 20  # MAX_HISTORY_LENGTH

    def test_get_recent_history(self):
        """Test getting recent history."""
        context = ConversationContext("U1", "C1")

        for i in range(10):
            context.add_message("user", f"Message {i}")

        recent = context.get_recent_history(3)

        assert len(recent) == 3
        assert recent[-1]["content"] == "Message 9"

    def test_context_key(self):
        """Test context key generation."""
        context = ConversationContext("U1", "C1", "123.456")

        assert context.key == "U1:C1:123.456"

    def test_context_key_no_thread(self):
        """Test context key without thread."""
        context = ConversationContext("U1", "C1")

        assert context.key == "U1:C1:main"

    def test_metadata(self):
        """Test metadata storage."""
        context = ConversationContext("U1", "C1")

        context.set_metadata("key", "value")
        assert context.get_metadata("key") == "value"
        assert context.get_metadata("missing", "default") == "default"


class TestConversationManager:
    """Tests for ConversationManager class."""

    def test_get_or_create(self):
        """Test getting or creating a conversation."""
        manager = ConversationManager(persist=False)

        context = manager.get_or_create("U1", "C1")

        assert context is not None
        assert context.user_id == "U1"

    def test_get_existing(self):
        """Test getting an existing conversation."""
        manager = ConversationManager(persist=False)

        ctx1 = manager.get_or_create("U1", "C1")
        ctx1.add_message("user", "Hello")

        ctx2 = manager.get_or_create("U1", "C1")

        assert ctx1 is ctx2
        assert len(ctx2.history) == 1

    def test_get_nonexistent(self):
        """Test getting a non-existent conversation."""
        manager = ConversationManager(persist=False)

        context = manager.get("U1", "C1")

        assert context is None

    def test_delete(self):
        """Test deleting a conversation."""
        manager = ConversationManager(persist=False)

        manager.get_or_create("U1", "C1")
        deleted = manager.delete("U1", "C1")

        assert deleted is True
        assert manager.get("U1", "C1") is None

    def test_delete_nonexistent(self):
        """Test deleting non-existent conversation."""
        manager = ConversationManager(persist=False)

        deleted = manager.delete("U1", "C1")

        assert deleted is False

    def test_different_threads(self):
        """Test that different threads have separate contexts."""
        manager = ConversationManager(persist=False)

        ctx1 = manager.get_or_create("U1", "C1", "thread1")
        ctx2 = manager.get_or_create("U1", "C1", "thread2")

        assert ctx1 is not ctx2

    def test_get_stats(self):
        """Test getting manager statistics."""
        manager = ConversationManager(persist=False)

        manager.get_or_create("U1", "C1")
        manager.get_or_create("U2", "C2")

        stats = manager.get_stats()

        assert stats["active_conversations"] == 2


class TestConversationStore:
    """Tests for ConversationStore persistence."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield Path(f.name)

    def test_save_and_load(self, temp_db):
        """Test saving and loading a conversation."""
        store = ConversationStore(temp_db)

        context = ConversationContext("U1", "C1", "thread1")
        context.add_message("user", "Hello")
        context.add_message("assistant", "Hi!")

        store.save(context)

        loaded = store.load(context.key)

        assert loaded is not None
        assert loaded.user_id == "U1"
        assert loaded.channel_id == "C1"
        assert len(loaded.history) == 2

    def test_load_nonexistent(self, temp_db):
        """Test loading a non-existent conversation."""
        store = ConversationStore(temp_db)

        loaded = store.load("nonexistent:key:here")

        assert loaded is None

    def test_save_update(self, temp_db):
        """Test updating an existing conversation."""
        store = ConversationStore(temp_db)

        context = ConversationContext("U1", "C1")
        context.add_message("user", "Hello")
        store.save(context)

        context.add_message("assistant", "Hi!")
        store.save(context)

        loaded = store.load(context.key)

        assert len(loaded.history) == 2

    def test_load_all(self, temp_db):
        """Test loading all conversations."""
        store = ConversationStore(temp_db)

        ctx1 = ConversationContext("U1", "C1")
        ctx2 = ConversationContext("U2", "C2")

        store.save(ctx1)
        store.save(ctx2)

        all_convos = store.load_all()

        assert len(all_convos) == 2

    def test_load_for_user(self, temp_db):
        """Test loading conversations for a specific user."""
        store = ConversationStore(temp_db)

        ctx1 = ConversationContext("U1", "C1")
        ctx2 = ConversationContext("U1", "C2")
        ctx3 = ConversationContext("U2", "C3")

        store.save(ctx1)
        store.save(ctx2)
        store.save(ctx3)

        user_convos = store.load_for_user("U1")

        assert len(user_convos) == 2

    def test_delete(self, temp_db):
        """Test deleting a conversation."""
        store = ConversationStore(temp_db)

        context = ConversationContext("U1", "C1")
        store.save(context)

        deleted = store.delete(context.key)

        assert deleted is True
        assert store.load(context.key) is None

    def test_get_stats(self, temp_db):
        """Test getting store statistics."""
        store = ConversationStore(temp_db)

        ctx1 = ConversationContext("U1", "C1")
        ctx2 = ConversationContext("U1", "C2")
        ctx3 = ConversationContext("U2", "C3")

        store.save(ctx1)
        store.save(ctx2)
        store.save(ctx3)

        stats = store.get_stats()

        assert stats["total_conversations"] == 3
        assert stats["unique_users"] == 2


class TestConversationManagerPersistence:
    """Tests for ConversationManager with persistence."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield Path(f.name)

    def test_persist_on_update(self, temp_db):
        """Test that conversations are persisted on update."""
        manager = ConversationManager(db_path=temp_db, persist=True)

        context = manager.get_or_create("U1", "C1")
        context.add_message("user", "Hello")
        manager.update(context)

        # Force persist
        manager.persist_all()

        # Create new manager to load from DB
        manager2 = ConversationManager(db_path=temp_db, persist=True)
        loaded = manager2.get("U1", "C1")

        assert loaded is not None
        assert len(loaded.history) == 1

    def test_load_on_startup(self, temp_db):
        """Test that conversations are loaded on startup."""
        # Create and save
        manager1 = ConversationManager(db_path=temp_db, persist=True)
        context = manager1.get_or_create("U1", "C1")
        context.add_message("user", "Test message")
        manager1.persist_all()

        # Create new manager - should load existing
        manager2 = ConversationManager(db_path=temp_db, persist=True)

        assert manager2.get_stats()["active_conversations"] >= 1

    def test_user_history(self, temp_db):
        """Test getting user conversation history."""
        manager = ConversationManager(db_path=temp_db, persist=True)

        ctx1 = manager.get_or_create("U1", "C1")
        ctx1.add_message("user", "Message 1")

        ctx2 = manager.get_or_create("U1", "C2")
        ctx2.add_message("user", "Message 2")

        manager.persist_all()

        history = manager.get_user_history("U1")

        assert len(history) == 2
