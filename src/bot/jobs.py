"""Background job state tracking for long-running Slack requests."""

import json
import sqlite3
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import JOB_DB_PATH

ACTIVE_JOB_STATUSES = {"queued", "running"}
TERMINAL_JOB_STATUSES = {"succeeded", "failed", "cancelled"}


@dataclass(frozen=True)
class BackgroundJob:
    """Persisted background job metadata."""

    job_id: str
    status: str
    user_id: str
    channel_id: str
    thread_ts: str
    message: str
    message_ts: str | None = None
    result_preview: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class JobStore:
    """SQLite-backed store for background job status."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else JOB_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
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
        with self._connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    channel_id TEXT NOT NULL,
                    thread_ts TEXT NOT NULL,
                    message TEXT NOT NULL,
                    message_ts TEXT,
                    result_preview TEXT,
                    error TEXT,
                    metadata TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
                CREATE INDEX IF NOT EXISTS idx_jobs_user_updated ON jobs(user_id, updated_at);
            """)

    def create(
        self,
        *,
        user_id: str,
        channel_id: str,
        thread_ts: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> BackgroundJob:
        """Create a queued background job."""
        now = time.time()
        job = BackgroundJob(
            job_id=uuid.uuid4().hex[:12],
            status="queued",
            user_id=user_id,
            channel_id=channel_id,
            thread_ts=thread_ts,
            message=message,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, status, user_id, channel_id, thread_ts, message,
                    message_ts, result_preview, error, metadata, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.job_id,
                    job.status,
                    job.user_id,
                    job.channel_id,
                    job.thread_ts,
                    job.message,
                    job.message_ts,
                    job.result_preview,
                    job.error,
                    json.dumps(job.metadata),
                    job.created_at,
                    job.updated_at,
                ),
            )
        return job

    def update(
        self,
        job_id: str,
        *,
        status: str | None = None,
        message_ts: str | None = None,
        result_preview: str | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update mutable job fields."""
        job = self.get(job_id)
        if job is None:
            raise ValueError(f"Unknown background job: {job_id}")

        next_metadata = job.metadata.copy()
        if metadata:
            next_metadata.update(metadata)

        with self._connection() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, message_ts = ?, result_preview = ?, error = ?,
                    metadata = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (
                    status or job.status,
                    message_ts if message_ts is not None else job.message_ts,
                    result_preview if result_preview is not None else job.result_preview,
                    error if error is not None else job.error,
                    json.dumps(next_metadata),
                    time.time(),
                    job_id,
                ),
            )

    def get_for_user(self, job_id: str, user_id: str) -> BackgroundJob | None:
        """Load a job only if it belongs to the requested user."""
        job = self.get(job_id)
        if job is None or job.user_id != user_id:
            return None
        return job

    def cancel(self, job_id: str, user_id: str) -> BackgroundJob | None:
        """Cancel a queued or running job owned by a user."""
        job = self.get_for_user(job_id, user_id)
        if job is None:
            return None
        if job.status in TERMINAL_JOB_STATUSES:
            return job

        self.update(
            job_id,
            status="cancelled",
            error="Cancelled by user",
            metadata={"cancelled_at": time.time()},
        )
        return self.get(job_id)

    def is_cancelled(self, job_id: str) -> bool:
        """Return whether a job has been cancelled."""
        job = self.get(job_id)
        return job is not None and job.status == "cancelled"

    def get(self, job_id: str) -> BackgroundJob | None:
        """Load a job by id."""
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return _row_to_job(row) if row else None

    def list_recent(self, user_id: str | None = None, limit: int = 20) -> list[BackgroundJob]:
        """List recently updated jobs, optionally for one user."""
        with self._connection() as conn:
            if user_id:
                rows = conn.execute(
                    """
                    SELECT * FROM jobs
                    WHERE user_id = ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (user_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM jobs ORDER BY updated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [_row_to_job(row) for row in rows]


def _row_to_job(row: sqlite3.Row) -> BackgroundJob:
    return BackgroundJob(
        job_id=row["job_id"],
        status=row["status"],
        user_id=row["user_id"],
        channel_id=row["channel_id"],
        thread_ts=row["thread_ts"],
        message=row["message"],
        message_ts=row["message_ts"],
        result_preview=row["result_preview"] or "",
        error=row["error"] or "",
        metadata=json.loads(row["metadata"] or "{}"),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
