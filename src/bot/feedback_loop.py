"""Feedback loop for learning from user interactions."""

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Generator

from ..config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "feedback.db"


class FeedbackType(str, Enum):
    """Types of feedback signals."""

    RESULT_CLICK = "result_click"  # User clicked/engaged with a result
    RESULT_SKIP = "result_skip"  # User skipped/ignored a result
    EXPLICIT_POSITIVE = "explicit_positive"  # User gave explicit positive feedback
    EXPLICIT_NEGATIVE = "explicit_negative"  # User gave explicit negative feedback
    CORRECTION = "correction"  # User corrected the bot's response
    REFINEMENT = "refinement"  # User refined their query


@dataclass
class FeedbackEvent:
    """A single feedback event."""

    user_id: str
    query: str
    feedback_type: FeedbackType
    result_id: str | None = None  # ID of the result that received feedback
    result_source: str | None = None  # Source of the result (email, calendar, etc.)
    metadata: dict | None = None
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


class FeedbackLoop:
    """Tracks user interactions to improve result relevance.

    Features:
    - Track which search results users engage with
    - Remember user corrections
    - Build per-user relevance scoring
    - Learn common query patterns
    """

    def __init__(self, db_path: Path | str | None = None):
        """Initialize feedback loop storage.

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
                -- Raw feedback events
                CREATE TABLE IF NOT EXISTS feedback_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    result_id TEXT,
                    result_source TEXT,
                    metadata TEXT,
                    timestamp REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_fb_user ON feedback_events(user_id);
                CREATE INDEX IF NOT EXISTS idx_fb_query ON feedback_events(query);
                CREATE INDEX IF NOT EXISTS idx_fb_time ON feedback_events(timestamp);

                -- Aggregated relevance scores per user/source
                CREATE TABLE IF NOT EXISTS relevance_scores (
                    user_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    positive_count INTEGER DEFAULT 0,
                    negative_count INTEGER DEFAULT 0,
                    score REAL DEFAULT 0.5,
                    last_updated REAL NOT NULL,
                    PRIMARY KEY (user_id, source)
                );

                -- Query pattern tracking
                CREATE TABLE IF NOT EXISTS query_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    pattern TEXT NOT NULL,
                    intent TEXT,
                    count INTEGER DEFAULT 1,
                    last_used REAL NOT NULL,
                    success_rate REAL DEFAULT 0.5,
                    UNIQUE(user_id, pattern)
                );
                CREATE INDEX IF NOT EXISTS idx_qp_user ON query_patterns(user_id);

                -- Correction history
                CREATE TABLE IF NOT EXISTS corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    original_query TEXT NOT NULL,
                    original_result TEXT,
                    corrected_value TEXT NOT NULL,
                    correction_type TEXT NOT NULL,
                    timestamp REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_corr_user ON corrections(user_id);
            """)

    def record_feedback(self, event: FeedbackEvent) -> None:
        """Record a feedback event.

        Args:
            event: Feedback event to record.
        """
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO feedback_events
                (user_id, query, feedback_type, result_id, result_source, metadata, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.user_id,
                    event.query,
                    event.feedback_type.value,
                    event.result_id,
                    event.result_source,
                    json.dumps(event.metadata) if event.metadata else None,
                    event.timestamp,
                ),
            )

            # Update relevance scores if we have a source
            if event.result_source:
                self._update_relevance_score(
                    conn, event.user_id, event.result_source, event.feedback_type
                )

        logger.debug(
            f"Recorded {event.feedback_type.value} feedback for user {event.user_id}"
        )

    def record_result_click(
        self,
        user_id: str,
        query: str,
        result_id: str,
        result_source: str,
        metadata: dict | None = None,
    ) -> None:
        """Convenience method to record a result click.

        Args:
            user_id: Slack user ID.
            query: Search query.
            result_id: ID of the clicked result.
            result_source: Source of the result.
            metadata: Additional metadata.
        """
        self.record_feedback(FeedbackEvent(
            user_id=user_id,
            query=query,
            feedback_type=FeedbackType.RESULT_CLICK,
            result_id=result_id,
            result_source=result_source,
            metadata=metadata,
        ))

    def record_correction(
        self,
        user_id: str,
        original_query: str,
        corrected_value: str,
        correction_type: str,
        original_result: str | None = None,
    ) -> None:
        """Record a user correction.

        Args:
            user_id: Slack user ID.
            original_query: Original query that produced wrong result.
            corrected_value: The correct value provided by user.
            correction_type: Type of correction (e.g., "contact", "date", "intent").
            original_result: The original wrong result (optional).
        """
        now = time.time()

        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO corrections
                (user_id, original_query, original_result, corrected_value, correction_type, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, original_query, original_result, corrected_value, correction_type, now),
            )

        # Also record as feedback event
        self.record_feedback(FeedbackEvent(
            user_id=user_id,
            query=original_query,
            feedback_type=FeedbackType.CORRECTION,
            metadata={
                "correction_type": correction_type,
                "corrected_value": corrected_value,
            },
        ))

        logger.info(f"Recorded correction from user {user_id}: {correction_type}")

    def record_query_pattern(
        self,
        user_id: str,
        pattern: str,
        intent: str,
        success: bool = True,
    ) -> None:
        """Record a query pattern for the user.

        Args:
            user_id: Slack user ID.
            pattern: Normalized query pattern.
            intent: Classified intent.
            success: Whether the query was successful.
        """
        now = time.time()

        with self._connection() as conn:
            # Upsert the pattern
            existing = conn.execute(
                "SELECT count, success_rate FROM query_patterns WHERE user_id = ? AND pattern = ?",
                (user_id, pattern),
            ).fetchone()

            if existing:
                # Update existing pattern
                old_count = existing["count"]
                old_rate = existing["success_rate"]
                new_count = old_count + 1
                # Exponential moving average for success rate
                new_rate = old_rate * 0.8 + (1.0 if success else 0.0) * 0.2

                conn.execute(
                    """
                    UPDATE query_patterns
                    SET count = ?, success_rate = ?, last_used = ?, intent = ?
                    WHERE user_id = ? AND pattern = ?
                    """,
                    (new_count, new_rate, now, intent, user_id, pattern),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO query_patterns (user_id, pattern, intent, count, last_used, success_rate)
                    VALUES (?, ?, ?, 1, ?, ?)
                    """,
                    (user_id, pattern, intent, now, 1.0 if success else 0.0),
                )

    def _update_relevance_score(
        self,
        conn: sqlite3.Connection,
        user_id: str,
        source: str,
        feedback_type: FeedbackType,
    ) -> None:
        """Update relevance scores based on feedback.

        Args:
            conn: Database connection.
            user_id: Slack user ID.
            source: Source of the result.
            feedback_type: Type of feedback.
        """
        is_positive = feedback_type in (
            FeedbackType.RESULT_CLICK,
            FeedbackType.EXPLICIT_POSITIVE,
        )

        now = time.time()

        # Get existing scores
        row = conn.execute(
            "SELECT positive_count, negative_count FROM relevance_scores WHERE user_id = ? AND source = ?",
            (user_id, source),
        ).fetchone()

        if row:
            pos = row["positive_count"] + (1 if is_positive else 0)
            neg = row["negative_count"] + (0 if is_positive else 1)
        else:
            pos = 1 if is_positive else 0
            neg = 0 if is_positive else 1

        # Calculate score (simple ratio with smoothing)
        score = (pos + 1) / (pos + neg + 2)  # Laplace smoothing

        conn.execute(
            """
            INSERT INTO relevance_scores (user_id, source, positive_count, negative_count, score, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, source) DO UPDATE SET
                positive_count = excluded.positive_count,
                negative_count = excluded.negative_count,
                score = excluded.score,
                last_updated = excluded.last_updated
            """,
            (user_id, source, pos, neg, score, now),
        )

    def get_relevance_scores(self, user_id: str) -> dict[str, float]:
        """Get relevance scores for all sources for a user.

        Args:
            user_id: Slack user ID.

        Returns:
            Dictionary mapping source to relevance score.
        """
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT source, score FROM relevance_scores WHERE user_id = ?",
                (user_id,),
            ).fetchall()

            return {row["source"]: row["score"] for row in rows}

    def get_source_ranking(self, user_id: str) -> list[str]:
        """Get sources ranked by relevance for a user.

        Args:
            user_id: Slack user ID.

        Returns:
            List of sources sorted by relevance (highest first).
        """
        scores = self.get_relevance_scores(user_id)
        return sorted(scores.keys(), key=lambda s: scores[s], reverse=True)

    def get_corrections(
        self,
        user_id: str,
        correction_type: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Get recent corrections for a user.

        Args:
            user_id: Slack user ID.
            correction_type: Optional type filter.
            limit: Maximum number of corrections.

        Returns:
            List of correction dictionaries.
        """
        with self._connection() as conn:
            if correction_type:
                rows = conn.execute(
                    """
                    SELECT * FROM corrections
                    WHERE user_id = ? AND correction_type = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (user_id, correction_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM corrections
                    WHERE user_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (user_id, limit),
                ).fetchall()

            return [dict(row) for row in rows]

    def get_common_patterns(self, user_id: str, limit: int = 10) -> list[dict]:
        """Get common query patterns for a user.

        Args:
            user_id: Slack user ID.
            limit: Maximum number of patterns.

        Returns:
            List of pattern dictionaries.
        """
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT pattern, intent, count, success_rate
                FROM query_patterns
                WHERE user_id = ?
                ORDER BY count DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()

            return [dict(row) for row in rows]

    def boost_results(
        self,
        user_id: str,
        results: list[dict],
        source_key: str = "source",
    ) -> list[dict]:
        """Boost search results based on user's relevance scores.

        Args:
            user_id: Slack user ID.
            results: List of result dictionaries.
            source_key: Key in result dict that contains the source.

        Returns:
            Results sorted by boosted relevance.
        """
        scores = self.get_relevance_scores(user_id)

        def get_boost(result: dict) -> float:
            source = result.get(source_key, "")
            return scores.get(source, 0.5)

        # Sort by original order but boost by relevance
        # Higher score = earlier in results
        boosted = []
        for i, result in enumerate(results):
            boost = get_boost(result)
            # Combine position (lower is better) with boost (higher is better)
            combined_score = boost - (i * 0.01)  # Small position penalty
            boosted.append((combined_score, i, result))

        boosted.sort(key=lambda x: x[0], reverse=True)
        return [item[2] for item in boosted]

    def get_feedback_stats(self, user_id: str | None = None) -> dict:
        """Get feedback statistics.

        Args:
            user_id: Optional user ID filter.

        Returns:
            Dictionary with feedback stats.
        """
        with self._connection() as conn:
            if user_id:
                total_events = conn.execute(
                    "SELECT COUNT(*) FROM feedback_events WHERE user_id = ?",
                    (user_id,),
                ).fetchone()[0]
                by_type = conn.execute(
                    """
                    SELECT feedback_type, COUNT(*) as count
                    FROM feedback_events WHERE user_id = ?
                    GROUP BY feedback_type
                    """,
                    (user_id,),
                ).fetchall()
                corrections = conn.execute(
                    "SELECT COUNT(*) FROM corrections WHERE user_id = ?",
                    (user_id,),
                ).fetchone()[0]
                patterns = conn.execute(
                    "SELECT COUNT(*) FROM query_patterns WHERE user_id = ?",
                    (user_id,),
                ).fetchone()[0]
            else:
                total_events = conn.execute(
                    "SELECT COUNT(*) FROM feedback_events"
                ).fetchone()[0]
                by_type = conn.execute(
                    """
                    SELECT feedback_type, COUNT(*) as count
                    FROM feedback_events GROUP BY feedback_type
                    """
                ).fetchall()
                corrections = conn.execute(
                    "SELECT COUNT(*) FROM corrections"
                ).fetchone()[0]
                patterns = conn.execute(
                    "SELECT COUNT(*) FROM query_patterns"
                ).fetchone()[0]

            return {
                "total_feedback_events": total_events,
                "by_type": {row["feedback_type"]: row["count"] for row in by_type},
                "total_corrections": corrections,
                "total_patterns": patterns,
            }

    def cleanup_old_events(self, max_age_days: int = 90) -> int:
        """Clean up old feedback events.

        Args:
            max_age_days: Maximum age in days.

        Returns:
            Number of events deleted.
        """
        cutoff = time.time() - (max_age_days * 24 * 60 * 60)

        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM feedback_events WHERE timestamp < ?", (cutoff,)
            )
            return cursor.rowcount
