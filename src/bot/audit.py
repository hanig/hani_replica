"""Audit logging for the Hani Replica bot.

Provides comprehensive logging of all bot interactions, tool executions,
and security events for compliance and debugging.
"""

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Generator

from ..config import PROJECT_ROOT, AUDIT_LOG_MESSAGES

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events."""
    # User interactions
    MESSAGE_RECEIVED = "message_received"
    MESSAGE_SENT = "message_sent"
    MENTION_RECEIVED = "mention_received"

    # Tool and agent operations
    TOOL_EXECUTED = "tool_executed"
    AGENT_INVOKED = "agent_invoked"
    AGENT_COMPLETED = "agent_completed"

    # Actions
    ACTION_REQUESTED = "action_requested"
    ACTION_CONFIRMED = "action_confirmed"
    ACTION_CANCELLED = "action_cancelled"
    ACTION_EXECUTED = "action_executed"

    # Security events
    SECURITY_WARNING = "security_warning"
    SECURITY_BLOCKED = "security_blocked"
    RATE_LIMITED = "rate_limited"
    UNAUTHORIZED = "unauthorized"

    # System events
    BOT_STARTED = "bot_started"
    BOT_STOPPED = "bot_stopped"
    ERROR = "error"


@dataclass
class AuditEvent:
    """An audit log entry."""
    event_type: AuditEventType
    timestamp: datetime
    user_id: str | None = None
    channel_id: str | None = None
    thread_ts: str | None = None
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    duration_ms: int | None = None
    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "channel_id": self.channel_id,
            "thread_ts": self.thread_ts,
            "message": self.message[:500] if self.message else None,  # Truncate long messages
            "details": self.details,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class AuditLogger:
    """Comprehensive audit logger for bot interactions.

    Logs to both SQLite database and standard logging.
    Provides queryable audit trail for security review and debugging.
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        retention_days: int = 90,
        enable_db: bool = True,
    ):
        """Initialize the audit logger.

        Args:
            db_path: Path to SQLite database. Defaults to data/audit.db.
            retention_days: Days to retain audit logs.
            enable_db: Whether to enable database logging.
        """
        self.retention_days = retention_days
        self.enable_db = enable_db
        self._lock = threading.Lock()

        if enable_db:
            if db_path is None:
                db_path = PROJECT_ROOT / "data" / "audit.db"

            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

        logger.info(f"AuditLogger initialized (db={enable_db}, retention={retention_days}d)")

    def _init_db(self) -> None:
        """Initialize the SQLite database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    user_id TEXT,
                    channel_id TEXT,
                    thread_ts TEXT,
                    message TEXT,
                    details TEXT,
                    duration_ms INTEGER,
                    success INTEGER DEFAULT 1,
                    error TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp
                ON audit_log(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_user
                ON audit_log(user_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_type
                ON audit_log(event_type)
            """)

            conn.commit()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with thread safety."""
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def log(self, event: AuditEvent) -> None:
        """Log an audit event.

        Args:
            event: Event to log.
        """
        # Log to standard logger
        log_msg = (
            f"[AUDIT] {event.event_type.value} | "
            f"user={event.user_id} | "
            f"channel={event.channel_id} | "
            f"success={event.success}"
        )
        if event.error:
            log_msg += f" | error={event.error}"
        if event.duration_ms:
            log_msg += f" | duration={event.duration_ms}ms"

        if event.success:
            logger.info(log_msg)
        else:
            logger.warning(log_msg)

        # Log to database
        if self.enable_db:
            self._log_to_db(event)

    def _log_to_db(self, event: AuditEvent) -> None:
        """Log event to SQLite database."""
        try:
            with self._lock:
                with self._get_connection() as conn:
                    conn.execute(
                        """
                        INSERT INTO audit_log
                        (event_type, timestamp, user_id, channel_id, thread_ts,
                         message, details, duration_ms, success, error)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            event.event_type.value,
                            event.timestamp.isoformat(),
                            event.user_id,
                            event.channel_id,
                            event.thread_ts,
                            event.message[:500] if event.message else None,
                            json.dumps(event.details) if event.details else None,
                            event.duration_ms,
                            1 if event.success else 0,
                            event.error,
                        ),
                    )
                    conn.commit()
        except Exception as e:
            logger.error(f"Failed to write audit log to database: {e}")

    def log_message_received(
        self,
        user_id: str,
        channel_id: str,
        message: str,
        thread_ts: str | None = None,
        is_mention: bool = False,
    ) -> None:
        """Log a received message.

        Args:
            user_id: User who sent the message.
            channel_id: Channel where message was received.
            message: Message content.
            thread_ts: Thread timestamp if in thread.
            is_mention: Whether this was an @mention.
        """
        safe_message = message if AUDIT_LOG_MESSAGES else "[redacted]"
        self.log(AuditEvent(
            event_type=(
                AuditEventType.MENTION_RECEIVED if is_mention
                else AuditEventType.MESSAGE_RECEIVED
            ),
            timestamp=datetime.now(),
            user_id=user_id,
            channel_id=channel_id,
            thread_ts=thread_ts,
            message=safe_message,
        ))

    def log_message_sent(
        self,
        channel_id: str,
        message: str,
        thread_ts: str | None = None,
        user_id: str | None = None,
    ) -> None:
        """Log a sent message.

        Args:
            channel_id: Channel where message was sent.
            message: Message content.
            thread_ts: Thread timestamp if in thread.
            user_id: User message was sent to (if DM).
        """
        safe_message = message if AUDIT_LOG_MESSAGES else "[redacted]"
        self.log(AuditEvent(
            event_type=AuditEventType.MESSAGE_SENT,
            timestamp=datetime.now(),
            user_id=user_id,
            channel_id=channel_id,
            thread_ts=thread_ts,
            message=safe_message,
        ))

    def log_tool_execution(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        result: Any,
        duration_ms: int,
        success: bool = True,
        error: str | None = None,
        user_id: str | None = None,
    ) -> None:
        """Log a tool execution.

        Args:
            tool_name: Name of the tool executed.
            tool_input: Input parameters.
            result: Execution result.
            duration_ms: Execution duration in milliseconds.
            success: Whether execution succeeded.
            error: Error message if failed.
            user_id: User who triggered the tool.
        """
        # Sanitize input for logging (remove potentially sensitive data)
        sanitized_input = {
            k: v for k, v in tool_input.items()
            if k not in ("password", "token", "secret", "key", "body")
        }
        if "body" in tool_input:
            sanitized_input["body"] = f"[{len(tool_input['body'])} chars]"

        self.log(AuditEvent(
            event_type=AuditEventType.TOOL_EXECUTED,
            timestamp=datetime.now(),
            user_id=user_id,
            message=f"Tool: {tool_name}",
            details={
                "tool_name": tool_name,
                "input": sanitized_input,
                "result_type": type(result).__name__ if result else None,
            },
            duration_ms=duration_ms,
            success=success,
            error=error,
        ))

    def log_agent_invoked(
        self,
        agent_type: str,
        user_id: str,
        message: str,
        channel_id: str | None = None,
    ) -> None:
        """Log an agent invocation.

        Args:
            agent_type: Type of agent invoked.
            user_id: User who triggered the agent.
            message: User message that triggered agent.
            channel_id: Channel ID.
        """
        self.log(AuditEvent(
            event_type=AuditEventType.AGENT_INVOKED,
            timestamp=datetime.now(),
            user_id=user_id,
            channel_id=channel_id,
            message=message,
            details={"agent_type": agent_type},
        ))

    def log_agent_completed(
        self,
        agent_type: str,
        user_id: str,
        iterations: int,
        tool_count: int,
        duration_ms: int,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """Log agent completion.

        Args:
            agent_type: Type of agent.
            user_id: User who triggered the agent.
            iterations: Number of iterations.
            tool_count: Number of tools used.
            duration_ms: Total duration.
            success: Whether agent completed successfully.
            error: Error message if failed.
        """
        self.log(AuditEvent(
            event_type=AuditEventType.AGENT_COMPLETED,
            timestamp=datetime.now(),
            user_id=user_id,
            details={
                "agent_type": agent_type,
                "iterations": iterations,
                "tool_count": tool_count,
            },
            duration_ms=duration_ms,
            success=success,
            error=error,
        ))

    def log_action(
        self,
        action_type: str,
        event_type: AuditEventType,
        user_id: str,
        details: dict[str, Any] | None = None,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """Log an action event.

        Args:
            action_type: Type of action.
            event_type: Audit event type.
            user_id: User who performed action.
            details: Additional details.
            success: Whether action succeeded.
            error: Error message if failed.
        """
        self.log(AuditEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            user_id=user_id,
            message=f"Action: {action_type}",
            details=details or {},
            success=success,
            error=error,
        ))

    def log_security_event(
        self,
        event_type: AuditEventType,
        user_id: str,
        description: str,
        details: dict[str, Any] | None = None,
        blocked: bool = False,
    ) -> None:
        """Log a security-related event.

        Args:
            event_type: Type of security event.
            user_id: User involved.
            description: Event description.
            details: Additional details.
            blocked: Whether the action was blocked.
        """
        self.log(AuditEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            user_id=user_id,
            message=description,
            details=details or {},
            success=not blocked,
        ))

    def log_error(
        self,
        error: str,
        user_id: str | None = None,
        channel_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log an error event.

        Args:
            error: Error message.
            user_id: User involved if any.
            channel_id: Channel if applicable.
            details: Additional context.
        """
        self.log(AuditEvent(
            event_type=AuditEventType.ERROR,
            timestamp=datetime.now(),
            user_id=user_id,
            channel_id=channel_id,
            message=error,
            details=details or {},
            success=False,
            error=error,
        ))

    def query(
        self,
        event_type: AuditEventType | None = None,
        user_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Query audit logs.

        Args:
            event_type: Filter by event type.
            user_id: Filter by user.
            start_time: Filter by start time.
            end_time: Filter by end time.
            limit: Maximum results.
            offset: Result offset for pagination.

        Returns:
            List of matching audit events.
        """
        if not self.enable_db:
            return []

        conditions = []
        params = []

        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type.value)
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time.isoformat())
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time.isoformat())

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with self._get_connection() as conn:
            cursor = conn.execute(
                f"""
                SELECT * FROM audit_log
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                params + [limit, offset],
            )
            rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def get_user_activity(
        self,
        user_id: str,
        days: int = 7,
    ) -> dict[str, Any]:
        """Get activity summary for a user.

        Args:
            user_id: User identifier.
            days: Number of days to analyze.

        Returns:
            Activity summary dictionary.
        """
        if not self.enable_db:
            return {"user_id": user_id, "db_disabled": True}

        start_time = datetime.now() - timedelta(days=days)

        with self._get_connection() as conn:
            # Total events
            total = conn.execute(
                "SELECT COUNT(*) FROM audit_log WHERE user_id = ? AND timestamp >= ?",
                (user_id, start_time.isoformat()),
            ).fetchone()[0]

            # Events by type
            by_type = conn.execute(
                """
                SELECT event_type, COUNT(*) as count
                FROM audit_log
                WHERE user_id = ? AND timestamp >= ?
                GROUP BY event_type
                """,
                (user_id, start_time.isoformat()),
            ).fetchall()

            # Error count
            errors = conn.execute(
                "SELECT COUNT(*) FROM audit_log WHERE user_id = ? AND success = 0 AND timestamp >= ?",
                (user_id, start_time.isoformat()),
            ).fetchone()[0]

        return {
            "user_id": user_id,
            "period_days": days,
            "total_events": total,
            "events_by_type": {row["event_type"]: row["count"] for row in by_type},
            "error_count": errors,
        }

    def cleanup_old_logs(self) -> int:
        """Remove audit logs older than retention period.

        Returns:
            Number of logs deleted.
        """
        if not self.enable_db:
            return 0

        cutoff = datetime.now() - timedelta(days=self.retention_days)

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM audit_log WHERE timestamp < ?",
                    (cutoff.isoformat(),),
                )
                deleted = cursor.rowcount
                conn.commit()

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} audit logs older than {self.retention_days} days")

        return deleted

    def get_stats(self) -> dict[str, Any]:
        """Get audit log statistics.

        Returns:
            Statistics dictionary.
        """
        if not self.enable_db:
            return {"db_enabled": False}

        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]

            oldest = conn.execute(
                "SELECT MIN(timestamp) FROM audit_log"
            ).fetchone()[0]

            newest = conn.execute(
                "SELECT MAX(timestamp) FROM audit_log"
            ).fetchone()[0]

            by_type = conn.execute(
                """
                SELECT event_type, COUNT(*) as count
                FROM audit_log
                GROUP BY event_type
                ORDER BY count DESC
                """
            ).fetchall()

        return {
            "total_events": total,
            "oldest_event": oldest,
            "newest_event": newest,
            "events_by_type": {row["event_type"]: row["count"] for row in by_type},
            "retention_days": self.retention_days,
            "db_path": str(self.db_path) if self.enable_db else None,
        }


# Singleton instance
_audit_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    """Get the global AuditLogger instance.

    Returns:
        AuditLogger singleton.
    """
    global _audit_logger
    if _audit_logger is None:
        from ..config import AUDIT_LOG_PATH, AUDIT_RETENTION_DAYS, ENABLE_AUDIT_LOG
        _audit_logger = AuditLogger(
            db_path=AUDIT_LOG_PATH,
            retention_days=AUDIT_RETENTION_DAYS,
            enable_db=ENABLE_AUDIT_LOG,
        )
    return _audit_logger


def reset_audit_logger() -> None:
    """Reset the global AuditLogger instance (for testing)."""
    global _audit_logger
    _audit_logger = None
