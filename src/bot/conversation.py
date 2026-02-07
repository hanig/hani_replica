"""Conversation state management for the Slack bot with persistence."""

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

from ..config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# Conversation TTL in seconds (30 minutes for active, 7 days for persisted)
CONVERSATION_TTL = 30 * 60
PERSISTED_TTL = 7 * 24 * 60 * 60  # 7 days

# Maximum history length
MAX_HISTORY_LENGTH = 20

# Default database path
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "conversations.db"


@dataclass
class ConversationContext:
    """State for a single conversation."""

    user_id: str
    channel_id: str
    thread_ts: str | None = None
    history: list[dict] = field(default_factory=list)
    pending_action: Any = None  # PendingAction instance (not persisted)
    metadata: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history.

        Args:
            role: Message role ('user' or 'assistant').
            content: Message content.
        """
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
        })

        # Trim history if too long
        if len(self.history) > MAX_HISTORY_LENGTH:
            self.history = self.history[-MAX_HISTORY_LENGTH:]

        self.last_activity = time.time()

    def get_recent_history(self, count: int = 6) -> list[dict]:
        """Get recent conversation history.

        Args:
            count: Number of recent messages to return.

        Returns:
            List of recent messages.
        """
        return self.history[-count:]

    def is_expired(self, ttl: int | None = None) -> bool:
        """Check if the conversation has expired.

        Args:
            ttl: Custom TTL in seconds. Defaults to CONVERSATION_TTL.

        Returns:
            True if conversation is older than TTL.
        """
        ttl = ttl if ttl is not None else CONVERSATION_TTL
        return time.time() - self.last_activity > ttl

    def clear_pending_action(self) -> None:
        """Clear any pending action."""
        self.pending_action = None

    def set_metadata(self, key: str, value: Any) -> None:
        """Set a metadata value.

        Args:
            key: Metadata key.
            value: Metadata value.
        """
        self.metadata[key] = value
        self.last_activity = time.time()

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get a metadata value.

        Args:
            key: Metadata key.
            default: Default value if key not found.

        Returns:
            Metadata value or default.
        """
        return self.metadata.get(key, default)

    @property
    def key(self) -> str:
        """Get the unique key for this conversation."""
        return f"{self.user_id}:{self.channel_id}:{self.thread_ts or 'main'}"

    def to_dict(self) -> dict:
        """Convert to dictionary for persistence.

        Note: pending_action is not included as it's a runtime object.
        """
        return {
            "user_id": self.user_id,
            "channel_id": self.channel_id,
            "thread_ts": self.thread_ts,
            "history": self.history,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationContext":
        """Create from dictionary."""
        return cls(
            user_id=data["user_id"],
            channel_id=data["channel_id"],
            thread_ts=data.get("thread_ts"),
            history=data.get("history", []),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", time.time()),
            last_activity=data.get("last_activity", time.time()),
        )


class ConversationStore:
    """SQLite-backed storage for conversations."""

    def __init__(self, db_path: Path | str | None = None):
        """Initialize the conversation store.

        Args:
            db_path: Path to SQLite database. Defaults to data/conversations.db.
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
                -- Conversations table
                CREATE TABLE IF NOT EXISTS conversations (
                    key TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    channel_id TEXT NOT NULL,
                    thread_ts TEXT,
                    history TEXT NOT NULL,
                    metadata TEXT,
                    created_at REAL NOT NULL,
                    last_activity REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id);
                CREATE INDEX IF NOT EXISTS idx_conv_activity ON conversations(last_activity);
            """)

    def save(self, context: ConversationContext) -> None:
        """Save a conversation to the database.

        Args:
            context: Conversation context to save.
        """
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO conversations
                (key, user_id, channel_id, thread_ts, history, metadata, created_at, last_activity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    history = excluded.history,
                    metadata = excluded.metadata,
                    last_activity = excluded.last_activity
                """,
                (
                    context.key,
                    context.user_id,
                    context.channel_id,
                    context.thread_ts,
                    json.dumps(context.history),
                    json.dumps(context.metadata),
                    context.created_at,
                    context.last_activity,
                ),
            )

    def load(self, key: str) -> ConversationContext | None:
        """Load a conversation from the database.

        Args:
            key: Conversation key.

        Returns:
            ConversationContext or None if not found.
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE key = ?", (key,)
            ).fetchone()

            if row:
                return ConversationContext(
                    user_id=row["user_id"],
                    channel_id=row["channel_id"],
                    thread_ts=row["thread_ts"],
                    history=json.loads(row["history"]),
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    created_at=row["created_at"],
                    last_activity=row["last_activity"],
                )
            return None

    def load_all(self, max_age: float | None = None) -> list[ConversationContext]:
        """Load all conversations from the database.

        Args:
            max_age: Maximum age in seconds. Only load conversations newer than this.

        Returns:
            List of ConversationContext objects.
        """
        with self._connection() as conn:
            if max_age:
                min_time = time.time() - max_age
                rows = conn.execute(
                    "SELECT * FROM conversations WHERE last_activity > ?",
                    (min_time,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM conversations").fetchall()

            conversations = []
            for row in rows:
                try:
                    ctx = ConversationContext(
                        user_id=row["user_id"],
                        channel_id=row["channel_id"],
                        thread_ts=row["thread_ts"],
                        history=json.loads(row["history"]),
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                        created_at=row["created_at"],
                        last_activity=row["last_activity"],
                    )
                    conversations.append(ctx)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to load conversation {row['key']}: {e}")

            return conversations

    def load_for_user(self, user_id: str, limit: int = 10) -> list[ConversationContext]:
        """Load recent conversations for a user.

        Args:
            user_id: Slack user ID.
            limit: Maximum number of conversations to load.

        Returns:
            List of ConversationContext objects, most recent first.
        """
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM conversations
                WHERE user_id = ?
                ORDER BY last_activity DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()

            conversations = []
            for row in rows:
                try:
                    ctx = ConversationContext(
                        user_id=row["user_id"],
                        channel_id=row["channel_id"],
                        thread_ts=row["thread_ts"],
                        history=json.loads(row["history"]),
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                        created_at=row["created_at"],
                        last_activity=row["last_activity"],
                    )
                    conversations.append(ctx)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to load conversation: {e}")

            return conversations

    def delete(self, key: str) -> bool:
        """Delete a conversation from the database.

        Args:
            key: Conversation key.

        Returns:
            True if deleted, False if not found.
        """
        with self._connection() as conn:
            cursor = conn.execute("DELETE FROM conversations WHERE key = ?", (key,))
            return cursor.rowcount > 0

    def cleanup_old(self, max_age: float) -> int:
        """Delete conversations older than max_age.

        Args:
            max_age: Maximum age in seconds.

        Returns:
            Number of conversations deleted.
        """
        with self._connection() as conn:
            min_time = time.time() - max_age
            cursor = conn.execute(
                "DELETE FROM conversations WHERE last_activity < ?", (min_time,)
            )
            deleted = cursor.rowcount
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old conversations from database")
            return deleted

    def get_stats(self) -> dict:
        """Get statistics about stored conversations."""
        with self._connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            users = conn.execute(
                "SELECT COUNT(DISTINCT user_id) FROM conversations"
            ).fetchone()[0]
            return {
                "total_conversations": total,
                "unique_users": users,
            }


class ConversationManager:
    """Manages conversation contexts across users and channels with persistence."""

    def __init__(
        self,
        ttl: int = CONVERSATION_TTL,
        db_path: Path | str | None = None,
        persist: bool = True,
    ):
        """Initialize the conversation manager.

        Args:
            ttl: Time-to-live for active conversations in seconds.
            db_path: Path to SQLite database for persistence.
            persist: Whether to persist conversations to disk.
        """
        self.ttl = ttl
        self.persist = persist
        self._conversations: dict[str, ConversationContext] = {}
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 minutes
        self._last_persist = time.time()
        self._persist_interval = 60  # Persist every minute

        # Initialize store if persistence is enabled
        self._store: ConversationStore | None = None
        if persist:
            self._store = ConversationStore(db_path)
            self._load_recent_conversations()

    def _load_recent_conversations(self) -> None:
        """Load recent conversations from persistent storage."""
        if not self._store:
            return

        try:
            # Load conversations from last 7 days
            conversations = self._store.load_all(max_age=PERSISTED_TTL)
            for ctx in conversations:
                self._conversations[ctx.key] = ctx
            logger.info(f"Loaded {len(conversations)} conversations from storage")
        except Exception as e:
            logger.error(f"Failed to load conversations: {e}")

    def get(
        self,
        user_id: str,
        channel_id: str,
        thread_ts: str | None = None,
    ) -> ConversationContext | None:
        """Get a conversation context if it exists.

        Args:
            user_id: Slack user ID.
            channel_id: Slack channel ID.
            thread_ts: Thread timestamp (optional).

        Returns:
            ConversationContext or None if not found/expired.
        """
        self._maybe_cleanup()

        key = self._make_key(user_id, channel_id, thread_ts)
        context = self._conversations.get(key)

        if context and not context.is_expired():
            context.last_activity = time.time()
            return context

        # Check persistent storage if not in memory
        if context is None and self._store:
            context = self._store.load(key)
            if context and not context.is_expired(PERSISTED_TTL):
                # Refresh the conversation
                context.last_activity = time.time()
                self._conversations[key] = context
                return context

        # Remove expired context
        if context:
            del self._conversations[key]

        return None

    def get_or_create(
        self,
        user_id: str,
        channel_id: str,
        thread_ts: str | None = None,
    ) -> ConversationContext:
        """Get or create a conversation context.

        Args:
            user_id: Slack user ID.
            channel_id: Slack channel ID.
            thread_ts: Thread timestamp (optional).

        Returns:
            ConversationContext instance.
        """
        context = self.get(user_id, channel_id, thread_ts)

        if context is None:
            context = ConversationContext(
                user_id=user_id,
                channel_id=channel_id,
                thread_ts=thread_ts,
            )
            self._conversations[context.key] = context
            logger.debug(f"Created new conversation: {context.key}")

            # Persist new conversation
            self._persist_conversation(context)

        return context

    def update(self, context: ConversationContext) -> None:
        """Update a conversation and persist changes.

        Args:
            context: Conversation context to update.
        """
        self._conversations[context.key] = context
        self._maybe_persist()

    def delete(
        self,
        user_id: str,
        channel_id: str,
        thread_ts: str | None = None,
    ) -> bool:
        """Delete a conversation context.

        Args:
            user_id: Slack user ID.
            channel_id: Slack channel ID.
            thread_ts: Thread timestamp (optional).

        Returns:
            True if deleted, False if not found.
        """
        key = self._make_key(user_id, channel_id, thread_ts)
        deleted = False

        if key in self._conversations:
            del self._conversations[key]
            deleted = True

        if self._store:
            deleted = self._store.delete(key) or deleted

        return deleted

    def get_user_history(self, user_id: str, limit: int = 5) -> list[ConversationContext]:
        """Get recent conversations for a user.

        Args:
            user_id: Slack user ID.
            limit: Maximum number of conversations.

        Returns:
            List of recent conversations.
        """
        # Get from memory
        user_convos = [
            ctx for ctx in self._conversations.values()
            if ctx.user_id == user_id
        ]

        # Also check persistent storage
        if self._store:
            stored = self._store.load_for_user(user_id, limit)
            for ctx in stored:
                if ctx.key not in self._conversations:
                    user_convos.append(ctx)

        # Sort by last activity and limit
        user_convos.sort(key=lambda c: c.last_activity, reverse=True)
        return user_convos[:limit]

    def find_pending_action_context(
        self,
        user_id: str,
        channel_id: str,
        action_id: str | None = None,
    ) -> ConversationContext | None:
        """Find an in-memory context with a pending action.

        Args:
            user_id: Slack user ID.
            channel_id: Slack channel ID.
            action_id: Optional pending action ID to match.

        Returns:
            Matching ConversationContext or None.
        """
        self._maybe_cleanup()

        matches = [
            ctx for ctx in self._conversations.values()
            if ctx.user_id == user_id and ctx.channel_id == channel_id and ctx.pending_action
        ]
        if not matches:
            return None

        if action_id:
            exact = [
                ctx for ctx in matches
                if getattr(ctx.pending_action, "action_id", "") == action_id
            ]
            if exact:
                exact.sort(key=lambda c: c.last_activity, reverse=True)
                return exact[0]

        matches.sort(key=lambda c: c.last_activity, reverse=True)
        return matches[0]

    def _make_key(
        self,
        user_id: str,
        channel_id: str,
        thread_ts: str | None,
    ) -> str:
        """Create a unique key for a conversation."""
        return f"{user_id}:{channel_id}:{thread_ts or 'main'}"

    def _persist_conversation(self, context: ConversationContext) -> None:
        """Persist a single conversation."""
        if self._store:
            try:
                self._store.save(context)
            except Exception as e:
                logger.error(f"Failed to persist conversation: {e}")

    def _maybe_persist(self) -> None:
        """Periodically persist all active conversations."""
        if not self._store:
            return

        if time.time() - self._last_persist < self._persist_interval:
            return

        self._last_persist = time.time()
        persisted = 0

        for context in self._conversations.values():
            try:
                self._store.save(context)
                persisted += 1
            except Exception as e:
                logger.error(f"Failed to persist conversation {context.key}: {e}")

        if persisted > 0:
            logger.debug(f"Persisted {persisted} conversations")

    def _maybe_cleanup(self) -> None:
        """Periodically clean up expired conversations."""
        if time.time() - self._last_cleanup < self._cleanup_interval:
            return

        self._last_cleanup = time.time()
        expired = []

        for key, context in self._conversations.items():
            if context.is_expired():
                expired.append(key)

        for key in expired:
            # Persist before removing from memory
            context = self._conversations[key]
            self._persist_conversation(context)
            del self._conversations[key]

        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired conversations from memory")

        # Also clean up old persisted conversations
        if self._store:
            self._store.cleanup_old(PERSISTED_TTL)

    def persist_all(self) -> None:
        """Force persist all conversations (call on shutdown)."""
        if not self._store:
            return

        for context in self._conversations.values():
            try:
                self._store.save(context)
            except Exception as e:
                logger.error(f"Failed to persist conversation {context.key}: {e}")

        logger.info(f"Persisted {len(self._conversations)} conversations on shutdown")

    def get_stats(self) -> dict:
        """Get statistics about active conversations.

        Returns:
            Dictionary with conversation stats.
        """
        self._maybe_cleanup()

        stats = {
            "active_conversations": len(self._conversations),
            "ttl_seconds": self.ttl,
            "persistence_enabled": self.persist,
        }

        if self._store:
            stats.update(self._store.get_stats())

        return stats
