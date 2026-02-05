"""Long-term memory storage for user preferences and context.

Uses Mem0 for semantic memory storage with automatic extraction and
consolidation, while maintaining SQLite for contact aliases (key->value lookups).
"""

import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Generator

from ..config import PROJECT_ROOT, MEM0_CHROMA_PATH

logger = logging.getLogger(__name__)

# Default database path for contact aliases
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
    """A single memory entry (for compatibility with existing code)."""

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

    Uses Mem0 for semantic memory storage (preferences, facts, corrections)
    and SQLite for contact aliases (key->value lookups).

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
            db_path: Path to SQLite database for contact aliases.
                     Defaults to data/user_memory.db.
        """
        # Keep SQLite for contact_aliases (Mem0 doesn't handle key->value well)
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

        # Initialize Mem0 with OpenAI embeddings + local ChromaDB storage
        self._mem0 = None
        self._mem0_init_error = None

    @property
    def mem0(self):
        """Lazy-load Mem0 to avoid import errors if not installed."""
        if self._mem0 is None and self._mem0_init_error is None:
            try:
                from mem0 import Memory

                # Ensure Mem0 storage directory exists
                MEM0_CHROMA_PATH.mkdir(parents=True, exist_ok=True)

                config = {
                    "llm": {
                        "provider": "openai",
                        "config": {
                            "model": "gpt-4o-mini",
                            "api_key": os.getenv("OPENAI_API_KEY"),
                            "temperature": 0.1,
                        }
                    },
                    "embedder": {
                        "provider": "openai",
                        "config": {
                            "model": "text-embedding-3-large",
                            "api_key": os.getenv("OPENAI_API_KEY"),
                        }
                    },
                    "vector_store": {
                        "provider": "chroma",
                        "config": {
                            "collection_name": "user_memories",
                            "path": str(MEM0_CHROMA_PATH),
                        }
                    },
                }
                self._mem0 = Memory.from_config(config)
                logger.info("Mem0 initialized successfully")
            except Exception as e:
                self._mem0_init_error = str(e)
                logger.warning(f"Failed to initialize Mem0: {e}. Falling back to SQLite.")

        return self._mem0

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
        """Initialize database schema for contact aliases."""
        with self._connection() as conn:
            conn.executescript("""
                -- Contact aliases for quick lookup (kept in SQLite)
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
        """Store or update a memory using Mem0.

        Args:
            user_id: Slack user ID.
            key: Memory key (e.g., "preferred_calendar", "John").
            value: Value to store.
            memory_type: Type of memory.
            source: Source of this memory (e.g., "user_correction", "inference").
            confidence: Confidence level (0.0 to 1.0).
        """
        if isinstance(memory_type, str):
            memory_type = MemoryType(memory_type)

        # Format memory text for Mem0
        memory_text = f"{memory_type.value}: {key} = {value}"

        if self.mem0:
            try:
                self.mem0.add(
                    messages=[{"role": "system", "content": memory_text}],
                    user_id=user_id,
                    metadata={
                        "type": memory_type.value,
                        "key": key,
                        "source": source,
                        "confidence": confidence,
                    }
                )
                logger.debug(f"Remembered {memory_type.value}:{key} for user {user_id} via Mem0")
            except Exception as e:
                logger.warning(f"Mem0 remember failed: {e}")
        else:
            logger.debug(f"Mem0 not available, memory not stored: {memory_type.value}:{key}")

    def recall(
        self,
        user_id: str,
        key: str,
        memory_type: MemoryType | str | None = None,
    ) -> Any | None:
        """Recall a specific memory using semantic search.

        Args:
            user_id: Slack user ID.
            key: Memory key.
            memory_type: Optional type filter.

        Returns:
            Memory value or None if not found.
        """
        if not self.mem0:
            return None

        try:
            # Build search query
            query = f"{memory_type.value if memory_type else ''} {key}".strip()
            results = self.mem0.search(query, user_id=user_id, limit=1)

            if results and results.get("results"):
                return results["results"][0].get("memory")
        except Exception as e:
            logger.warning(f"Mem0 recall failed: {e}")

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
        if not self.mem0:
            return []

        memories = []
        try:
            all_memories = self.mem0.get_all(user_id=user_id, limit=limit)

            if all_memories and all_memories.get("results"):
                if isinstance(memory_type, str):
                    memory_type = MemoryType(memory_type)

                for mem in all_memories["results"]:
                    metadata = mem.get("metadata", {})
                    mem_type_str = metadata.get("type", "fact")

                    # Filter by type if specified
                    if memory_type and mem_type_str != memory_type.value:
                        continue

                    try:
                        mem_type = MemoryType(mem_type_str)
                    except ValueError:
                        mem_type = MemoryType.FACT

                    memories.append(Memory(
                        user_id=user_id,
                        key=metadata.get("key", ""),
                        value=mem.get("memory", ""),
                        memory_type=mem_type,
                        source=metadata.get("source", "mem0"),
                        confidence=metadata.get("confidence", 1.0),
                    ))
        except Exception as e:
            logger.warning(f"Mem0 recall_all failed: {e}")

        return memories

    def search_memories(self, user_id: str, query: str, limit: int = 5) -> dict:
        """Semantic search over all memories.

        Args:
            user_id: Slack user ID.
            query: Search query.
            limit: Maximum number of results.

        Returns:
            Dictionary with 'results' list containing matching memories.
        """
        if not self.mem0:
            return {"results": []}

        try:
            return self.mem0.search(query, user_id=user_id, limit=limit)
        except Exception as e:
            logger.warning(f"Mem0 search_memories failed: {e}")
            return {"results": []}

    def add_from_conversation(self, user_id: str, messages: list[dict]) -> None:
        """Auto-extract memories from a conversation.

        Mem0 will automatically identify and extract relevant memories
        from the conversation.

        Args:
            user_id: Slack user ID.
            messages: List of message dicts with 'role' and 'content' keys.
        """
        if not self.mem0:
            return

        if not messages:
            return

        try:
            self.mem0.add(messages, user_id=user_id)
            logger.debug(f"Extracted memories from {len(messages)} messages for user {user_id}")
        except Exception as e:
            logger.debug(f"Memory extraction failed: {e}")

    def forget(
        self,
        user_id: str,
        key: str,
        memory_type: MemoryType | str | None = None,
    ) -> bool:
        """Forget a specific memory.

        Note: Mem0 doesn't support direct key-based deletion, so this searches
        for matching memories and deletes them.

        Args:
            user_id: Slack user ID.
            key: Memory key.
            memory_type: Optional type filter.

        Returns:
            True if forgotten, False if not found.
        """
        if not self.mem0:
            return False

        try:
            # Search for the memory
            query = f"{memory_type.value if memory_type else ''} {key}".strip()
            results = self.mem0.search(query, user_id=user_id, limit=5)

            if results and results.get("results"):
                deleted = False
                for mem in results["results"]:
                    if key.lower() in mem.get("memory", "").lower():
                        mem_id = mem.get("id")
                        if mem_id:
                            self.mem0.delete(mem_id)
                            deleted = True
                return deleted
        except Exception as e:
            logger.warning(f"Mem0 forget failed: {e}")

        return False

    def forget_all(self, user_id: str) -> int:
        """Forget all memories for a user.

        Args:
            user_id: Slack user ID.

        Returns:
            Number of memories forgotten.
        """
        count = 0

        # Delete Mem0 memories
        if self.mem0:
            try:
                self.mem0.delete_all(user_id=user_id)
                logger.info(f"Deleted all Mem0 memories for user {user_id}")
            except Exception as e:
                logger.warning(f"Mem0 forget_all failed: {e}")

        # Also clear contact aliases from SQLite
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM contact_aliases WHERE user_id = ?", (user_id,)
            )
            count = cursor.rowcount

        return count

    # Contact alias helpers - kept in SQLite for fast key->value lookup
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

        # Get all memories from Mem0
        if self.mem0:
            try:
                all_memories = self.mem0.get_all(user_id=user_id, limit=max_items)

                if all_memories and all_memories.get("results"):
                    lines.append("What I know about this user:")
                    for mem in all_memories["results"]:
                        memory_text = mem.get("memory", "")
                        if memory_text:
                            lines.append(f"- {memory_text}")
            except Exception as e:
                logger.debug(f"Failed to get Mem0 memories for context: {e}")

        # Still include contact aliases from SQLite
        contacts = self.get_frequent_contacts(user_id, 5)
        if contacts:
            lines.append("\nKnown contacts:")
            for c in contacts[:5]:  # Top 5 contacts
                name_part = f" ({c['name']})" if c.get("name") else ""
                lines.append(f"- \"{c['alias']}\" refers to {c['email']}{name_part}")

        return "\n".join(lines) if lines else ""

    def get_stats(self, user_id: str | None = None) -> dict:
        """Get statistics about stored memories.

        Args:
            user_id: Optional user ID to filter by.

        Returns:
            Dictionary with memory stats.
        """
        stats = {
            "total_memories": 0,
            "by_type": {},
            "contact_aliases": 0,
            "mem0_available": self.mem0 is not None,
        }

        # Get Mem0 stats
        if self.mem0:
            try:
                if user_id:
                    all_mems = self.mem0.get_all(user_id=user_id)
                else:
                    all_mems = self.mem0.get_all()

                if all_mems and all_mems.get("results"):
                    stats["total_memories"] = len(all_mems["results"])

                    # Count by type
                    for mem in all_mems["results"]:
                        mem_type = mem.get("metadata", {}).get("type", "unknown")
                        stats["by_type"][mem_type] = stats["by_type"].get(mem_type, 0) + 1
            except Exception as e:
                logger.debug(f"Failed to get Mem0 stats: {e}")

        # Get contact alias count from SQLite
        with self._connection() as conn:
            if user_id:
                contacts = conn.execute(
                    "SELECT COUNT(*) FROM contact_aliases WHERE user_id = ?",
                    (user_id,),
                ).fetchone()[0]
            else:
                contacts = conn.execute(
                    "SELECT COUNT(*) FROM contact_aliases"
                ).fetchone()[0]

            stats["contact_aliases"] = contacts

        return stats
