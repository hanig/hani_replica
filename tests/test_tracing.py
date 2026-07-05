"""Tests for structured trace logging."""

import json
from types import SimpleNamespace

from src.bot.tracing import TraceLogger, model_usage


def test_trace_logger_writes_jsonl(tmp_path):
    """TraceLogger writes compact JSONL records."""
    path = tmp_path / "traces.jsonl"
    logger = TraceLogger(path=path, enabled=True)

    logger.log_model_call(
        caller="test",
        model="claude-sonnet-5",
        operation="messages.create",
        duration_ms=12.345,
        success=True,
        usage={"input_tokens": 1, "output_tokens": 2},
        stop_reason="end_turn",
    )

    record = json.loads(path.read_text().strip())
    assert record["event_type"] == "model_call"
    assert record["caller"] == "test"
    assert record["model"] == "claude-sonnet-5"
    assert record["duration_ms"] == 12.35
    assert record["usage"]["input_tokens"] == 1


def test_tool_trace_logs_input_keys_not_values(tmp_path):
    """Tool traces should store input keys, not raw input payloads."""
    path = tmp_path / "traces.jsonl"
    logger = TraceLogger(path=path, enabled=True)

    logger.log_tool_call(
        caller="agent.test",
        tool_name="SearchEmailsTool",
        duration_ms=5,
        success=True,
        input_keys=["query", "account"],
        result_preview="ok",
    )

    record = json.loads(path.read_text().strip())
    assert record["event_type"] == "tool_call"
    assert record["input_keys"] == ["account", "query"]
    assert record["result_preview_chars"] == 2
    assert "result_preview" not in record
    assert "query text" not in record


def test_model_usage_extracts_known_fields():
    """Anthropic usage metadata is normalized."""
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=20,
            cache_creation_input_tokens=3,
            cache_read_input_tokens=4,
        )
    )

    assert model_usage(response) == {
        "input_tokens": 10,
        "output_tokens": 20,
        "cache_creation_input_tokens": 3,
        "cache_read_input_tokens": 4,
    }
