"""Tests for background job state tracking."""

from src.bot.event_handlers import (
    _should_run_background_job,
    format_job_cancel_result,
    format_job_status,
    format_recent_jobs,
    job_action_blocks,
    parse_job_command,
)
from src.bot.jobs import JobStore


def test_job_store_create_update_get(tmp_path):
    """JobStore persists job lifecycle fields."""
    store = JobStore(tmp_path / "jobs.db")

    job = store.create(
        user_id="U123",
        channel_id="C123",
        thread_ts="111.222",
        message="give me daily briefing",
    )
    assert job.status == "queued"
    assert len(job.job_id) == 12

    store.update(
        job.job_id,
        status="succeeded",
        message_ts="333.444",
        result_preview="done",
        metadata={"mode": "multi_agent"},
    )

    loaded = store.get(job.job_id)
    assert loaded is not None
    assert loaded.status == "succeeded"
    assert loaded.message_ts == "333.444"
    assert loaded.result_preview == "done"
    assert loaded.metadata["mode"] == "multi_agent"


def test_job_store_list_recent_filters_by_user(tmp_path):
    """Recent job listing can be filtered to one user."""
    store = JobStore(tmp_path / "jobs.db")
    first = store.create(user_id="U1", channel_id="C", thread_ts="1", message="daily briefing")
    store.create(user_id="U2", channel_id="C", thread_ts="2", message="deep research")

    recent = store.list_recent(user_id="U1")
    assert [job.job_id for job in recent] == [first.job_id]


def test_job_store_cancel_only_user_jobs(tmp_path):
    """Users can cancel their own active jobs but not another user's job."""
    store = JobStore(tmp_path / "jobs.db")
    job = store.create(user_id="U1", channel_id="C", thread_ts="1", message="daily briefing")

    assert store.cancel(job.job_id, "U2") is None
    cancelled = store.cancel(job.job_id, "U1")

    assert cancelled is not None
    assert cancelled.status == "cancelled"
    assert store.is_cancelled(job.job_id)


def test_job_store_cleanup_removes_old_terminal_jobs(tmp_path):
    """Cleanup removes old finished jobs while keeping active jobs."""
    store = JobStore(tmp_path / "jobs.db")
    old = store.create(user_id="U1", channel_id="C", thread_ts="1", message="old")
    active = store.create(user_id="U1", channel_id="C", thread_ts="2", message="active")
    store.update(old.job_id, status="succeeded")
    store.update(active.job_id, status="running")
    old_timestamp = 1
    with store._connection() as conn:
        conn.execute(
            "UPDATE jobs SET updated_at = ? WHERE job_id IN (?, ?)",
            (old_timestamp, old.job_id, active.job_id),
        )

    assert store.cleanup(max_age_days=1) == 1
    assert store.get(old.job_id) is None
    assert store.get(active.job_id) is not None


def test_parse_job_commands():
    """Slack job commands parse without invoking the LLM."""
    assert parse_job_command("jobs").action == "list"
    assert parse_job_command("job status abcdef123456").job_id == "abcdef123456"
    assert parse_job_command("status job abcdef123456").action == "status"
    assert parse_job_command("job cancel abcdef123456").action == "cancel"
    assert parse_job_command("retry job abcdef123456").action == "retry"
    assert parse_job_command("what is my next meeting?") is None


def test_job_formatters(tmp_path):
    """Job status text includes useful status without failing on missing jobs."""
    store = JobStore(tmp_path / "jobs.db")
    job = store.create(user_id="U1", channel_id="C", thread_ts="1", message="daily briefing")
    store.update(job.job_id, status="succeeded", result_preview="done")
    loaded = store.get(job.job_id)

    assert loaded is not None
    assert "Recent background jobs" in format_recent_jobs([loaded])
    assert f"Job `{job.job_id}` is *succeeded*" in format_job_status(loaded)
    assert "could not find" in format_job_status(None)
    assert "already finished" in format_job_cancel_result(loaded, job.job_id)


def test_job_action_blocks_switch_from_cancel_to_retry(tmp_path):
    """Active jobs get cancel buttons; terminal jobs get retry buttons."""
    store = JobStore(tmp_path / "jobs.db")
    job = store.create(user_id="U1", channel_id="C", thread_ts="1", message="daily briefing")

    active_actions = job_action_blocks(job)[1]["elements"]
    assert [button["text"]["text"] for button in active_actions] == ["Status", "Cancel"]

    store.update(job.job_id, status="failed", error="boom")
    failed = store.get(job.job_id)
    assert failed is not None
    failed_actions = job_action_blocks(failed)[1]["elements"]
    assert [button["text"]["text"] for button in failed_actions] == ["Status", "Retry"]


def test_should_run_background_job():
    """Only likely long-running requests should be queued."""
    assert _should_run_background_job("give me daily briefing")
    assert _should_run_background_job("please do deep research on this")
    assert not _should_run_background_job("what is my next meeting?")
