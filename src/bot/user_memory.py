"""Long-term memory storage for user preferences and context."""

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Generator

from ..config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "user_memory.db"


class MemoryType(str, Enum):
    """Types of memories that can be stored."""

    PREFERENCE = "preference"  # User preferences (e.g., preferred calendar)
    CONTACT = "contact"  # Contact mappings (e.g., "John" -> "john.smith@arc.org")
    CORRECTION = "correction"  # User corrections to bot responses
    FACT = "fact"  # Facts about the user (e.g., "works at Arc Institute")
    PATTERN = "pattern"  # Usage patterns (e.g., common search terms)


@dataclass
class Memory:
    """A single memory entry."""

    user_id: str
    key: str
    value: Any
    memory_type: MemoryType
    source: str  # Where this memory came from
    confidence: float = 1.0  # How confident we are in this memory
    created_at: float = 0.0
    updated_at: float = 0.0
    access_count: int = 0

    def __post_init__(self):
        now = time.time()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now


class UserMemory:
    """Long-term memory storage for user preferences and context.

    Stores:
    - Preferred calendars and accounts
    - Contact name mappings (e.g., "John" -> full email)
    - User corrections to bot responses
    - Facts learned about the user
    - Usage patterns and common queries
    """

    def __init__(self, db_path: Path | str | None = None):
        """Initialize user memory storage.

        Args:
            db_path: Path to SQLite database. Defaults to data/user_memory.db.
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
                -- User memories table
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    UNIQUE(user_id, key, memory_type)
                );
                CREATE INDEX IF NOT EXISTS idx_mem_user ON memories(user_id);
                CREATE INDEX IF NOT EXISTS idx_mem_type ON memories(memory_type);
                CREATE INDEX IF NOT EXISTS idx_mem_key ON memories(user_id, key);

                -- Contact aliases for quick lookup
                CREATE TABLE IF NOT EXISTS contact_aliases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    alias TEXT NOT NULL,
                    email TEXT NOT NULL,
                    name TEXT,
                    source TEXT NOT NULL,
                    use_count INTEGER DEFAULT 0,
                    created_at REAL NOT NULL,
                    UNIQUE(user_id, alias)
                );
                CREATE INDEX IF NOT EXISTS idx_alias_user ON contact_aliases(user_id);
                CREATE INDEX IF NOT EXISTS idx_alias_lookup ON contact_aliases(user_id, alias);
            """)

    def remember(
        self,
        user_id: str,
        key: str,
        value: Any,
        memory_type: MemoryType | str,
        source: str,
        confidence: float = 1.0,
    ) -> None:
        """Store or update a memory.

        Args:
            user_id: Slack user ID.
            key: Memory key (e.g., "preferred_calendar", "John").
            value: Value to store (will be JSON-serialized if not a string).
            memory_type: Type of memory.
            source: Source of this memory (e.g., "user_correction", "inference").
            confidence: Confidence level (0.0 to 1.0).
        """
        if isinstance(memory_type, str):
            memory_type = MemoryType(memory_type)

        value_str = json.dumps(value) if not isinstance(value, str) else value
        now = time.time()

        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO memories
                (user_id, key, value, memory_type, source, confidence, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, key, memory_type) DO UPDATE SET
                    value = excluded.value,
                    source = excluded.source,
                    confidence = excluded.confidence,
                    updated_at = excluded.updated_at,
                    access_count = access_count + 1
                """,
                (user_id, key, value_str, memory_type.value, source, confidence, now, now),
            )

        logger.debug(f"Remembered {memory_type.value}:{key} for user {user_id}")

    def recall(
        self,
        user_id: str,
        key: str,
        memory_type: MemoryType | str | None = None,
    ) -> Any | None:
        """Recall a specific memory.

        Args:
            user_id: Slack user ID.
            key: Memory key.
            memory_type: Optional type filter.

        Returns:
            Memory value or None if not found.
        """
        with self._connection() as conn:
            if memory_type:
                if isinstance(memory_type, str):
                    memory_type = MemoryType(memory_type)
                row = conn.execute(
                    """
                    SELECT value FROM memories
                    WHERE user_id = ? AND key = ? AND memory_type = ?
                    """,
                    (user_id, key, memory_type.value),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT value FROM memories
                    WHERE user_id = ? AND key = ?
                    ORDER BY confidence DESC, updated_at DESC
                    LIMIT 1
                    """,
                    (user_id, key),
                ).fetchone()

            if row:
                # Update access count
                conn.execute(
                    """
                    UPDATE memories SET access_count = access_count + 1
                    WHERE user_id = ? AND key = ?
                    """,
                    (user_id, key),
                )

                try:
                    return json.loads(row["value"])
                except json.JSONDecodeError:
                    return row["value"]

            return None

    def recall_all(
        self,
        user_id: str,
        memory_type: MemoryType | str | None = None,
        limit: int = 100,
    ) -> list[Memory]:
        """Recall all memories for a user.

        Args:
            user_id: Slack user ID.
            memory_type: Optional type filter.
            limit: Maximum number of memories to return.

        Returns:
            List of Memory objects.
        """
        with self._connection() as conn:
            if memory_type:
                if isinstance(memory_type, str):
                    memory_type = MemoryType(memory_type)
                rows = conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE user_id = ? AND memory_type = ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (user_id, memory_type.value, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE user_id = ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (user_id, limit),
                ).fetchall()

            memories = []
            for row in rows:
                try:
                    value = json.loads(row["value"])
                except json.JSONDecodeError:
                    value = row["value"]

                memories.append(Memory(
                    user_id=row["user_id"],
                    key=row["key"],
                    value=value,
                    memory_type=MemoryType(row["memory_type"]),
                    source=row["source"],
                    confidence=row["confidence"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    access_count=row["access_count"],
                ))

            return memories

    def forget(
        self,
        user_id: str,
        key: str,
        memory_type: MemoryType | str | None = None,
    ) -> bool:
        """Forget a specific memory.

        Args:
            user_id: Slack user ID.
            key: Memory key.
            memory_type: Optional type filter.

        Returns:
            True if forgotten, False if not found.
        """
        with self._connection() as conn:
            if memory_type:
                if isinstance(memory_type, str):
                    memory_type = MemoryType(memory_type)
                cursor = conn.execute(
                    "DELETE FROM memories WHERE user_id = ? AND key = ? AND memory_type = ?",
                    (user_id, key, memory_type.value),
                )
            else:
                cursor = conn.execute(
                    "DELETE FROM memories WHERE user_id = ? AND key = ?",
                    (user_id, key),
                )

            return cursor.rowcount > 0

    def forget_all(self, user_id: str) -> int:
        """Forget all memories for a user.

        Args:
            user_id: Slack user ID.

        Returns:
            Number of memories forgotten.
        """
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE user_id = ?", (user_id,)
            )
            count = cursor.rowcount
            conn.execute(
                "DELETE FROM contact_aliases WHERE user_id = ?", (user_id,)
            )
            return count

    # Contact alias helpers
    def add_contact_alias(
        self,
        user_id: str,
        alias: str,
        email: str,
        name: str | None = None,
        source: str = "user",
    ) -> None:
        """Add a contact alias mapping.

        Args:
            user_id: Slack user ID.
            alias: Short name/alias (e.g., "John").
            email: Full email address.
            name: Full name (optional).
            source: Source of the mapping.
        """
        now = time.time()
        alias_lower = alias.lower().strip()

        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO contact_aliases (user_id, alias, email, name, source, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, alias) DO UPDATE SET
                    email = excluded.email,
                    name = excluded.name,
                    use_count = use_count + 1
                """,
                (user_id, alias_lower, email, name, source, now),
            )

    def resolve_contact(self, user_id: str, alias: str) -> dict | None:
        """Resolve a contact alias to full details.

        Args:
            user_id: Slack user ID.
            alias: Short name/alias.

        Returns:
            Dictionary with email and name, or None if not found.
        """
        alias_lower = alias.lower().strip()

        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT email, name FROM contact_aliases
                WHERE user_id = ? AND alias = ?
                """,
                (user_id, alias_lower),
            ).fetchone()

            if row:
                # Update use count
                conn.execute(
                    """
                    UPDATE contact_aliases SET use_count = use_count + 1
                    WHERE user_id = ? AND alias = ?
                    """,
                    (user_id, alias_lower),
                )
                return {"email": row["email"], "name": row["name"]}

            return None

    def get_frequent_contacts(self, user_id: str, limit: int = 10) -> list[dict]:
        """Get frequently used contacts for a user.

        Args:
            user_id: Slack user ID.
            limit: Maximum number of contacts.

        Returns:
            List of contact dictionaries.
        """
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT alias, email, name, use_count FROM contact_aliases
                WHERE user_id = ?
                ORDER BY use_count DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()

            return [
                {
                    "alias": row["alias"],
                    "email": row["email"],
                    "name": row["name"],
                    "use_count": row["use_count"],
                }
                for row in rows
            ]

    def get_context_summary(self, user_id: str, max_items: int = 10) -> str:
        """Generate a context summary for LLM injection.

        Creates a natural language summary of the user's preferences
        and frequently used items.

        Args:
            user_id: Slack user ID.
            max_items: Maximum number of items per category.

        Returns:
            Summary string for LLM context.
        """
        lines = []

        # Get preferences
        preferences = self.recall_all(user_id, MemoryType.PREFERENCE, max_items)
        if preferences:
            pref_lines = []
            for mem in preferences:
                pref_lines.append(f"- {mem.key}: {mem.value}")
            if pref_lines:
                lines.append("User preferences:")
                lines.extend(pref_lines)

        # Get contact aliases
        contacts = self.get_frequent_contacts(user_id, max_items)
        if contacts:
            contact_lines = []
            for c in contacts[:5]:  # Top 5 contacts
                name_part = f" ({c['name']})" if c["name"] else ""
                contact_lines.append(f"- \"{c['alias']}\" refers to {c['email']}{name_part}")
            if contact_lines:
                lines.append("\nKnown contacts:")
                lines.extend(contact_lines)

        # Get facts about the user
        facts = self.recall_all(user_id, MemoryType.FACT, max_items)
        if facts:
            fact_lines = []
            for mem in facts:
                fact_lines.append(f"- {mem.value}")
            if fact_lines:
                lines.append("\nKnown facts:")
                lines.extend(fact_lines)

        # Get recent corrections
        corrections = self.recall_all(user_id, MemoryType.CORRECTION, 3)
        if corrections:
            corr_lines = []
            for mem in corrections:
                corr_lines.append(f"- {mem.key}: {mem.value}")
            if corr_lines:
                lines.append("\nRecent corrections:")
                lines.extend(corr_lines)

        return "\n".join(lines) if lines else ""

    def get_stats(self, user_id: str | None = None) -> dict:
        """Get statistics about stored memories.

        Args:
            user_id: Optional user ID to filter by.

        Returns:
            Dictionary with memory stats.
        """
        with self._connection() as conn:
            if user_id:
                total = conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE user_id = ?", (user_id,)
                ).fetchone()[0]
                by_type = conn.execute(
                    """
                    SELECT memory_type, COUNT(*) as count
                    FROM memories WHERE user_id = ?
                    GROUP BY memory_type
                    """,
                    (user_id,),
                ).fetchall()
                contacts = conn.execute(
                    "SELECT COUNT(*) FROM contact_aliases WHERE user_id = ?",
                    (user_id,),
                ).fetchone()[0]
            else:
                total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                by_type = conn.execute(
                    """
                    SELECT memory_type, COUNT(*) as count
                    FROM memories GROUP BY memory_type
                    """
                ).fetchall()
                contacts = conn.execute(
                    "SELECT COUNT(*) FROM contact_aliases"
                ).fetchone()[0]

            return {
                "total_memories": total,
                "by_type": {row["memory_type"]: row["count"] for row in by_type},
                "contact_aliases": contacts,
            }
