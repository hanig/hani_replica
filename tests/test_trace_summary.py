"""Tests for trace summary reporting."""

import json

from src.trace_summary import (
    evaluate_trace_thresholds,
    format_trace_summary,
    format_trace_warnings,
    load_trace_records,
    summarize_trace_records,
)


def test_trace_summary_aggregates_models_tools_and_failures(tmp_path):
    """Trace summaries report counts, latency, tokens, and failures."""
    path = tmp_path / "traces.jsonl"
    records = [
        {
            "event_type": "model_call",
            "model": "claude-sonnet-5",
            "caller": "agent.executor",
            "duration_ms": 100,
            "success": True,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
        {
            "event_type": "model_call",
            "model": "claude-sonnet-5",
            "caller": "agent.executor",
            "duration_ms": 200,
            "success": False,
            "usage": {"input_tokens": "<mock>", "output_tokens": None},
            "error_type": "RuntimeError",
        },
        {
            "event_type": "tool_call",
            "tool_name": "GetCalendarEventsTool",
            "caller": "agent.calendar",
            "duration_ms": 50,
            "success": True,
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records))

    summary = summarize_trace_records(load_trace_records(path))
    formatted = format_trace_summary(summary)

    assert summary["events"] == 3
    assert summary["model_calls"] == 2
    assert summary["tool_calls"] == 1
    assert summary["failures"] == 1
    assert summary["models"]["claude-sonnet-5"]["input_tokens"] == 10
    assert "claude-sonnet-5" in formatted
    assert "GetCalendarEventsTool" in formatted
    assert "RuntimeError" in formatted

    warnings = evaluate_trace_thresholds(
        summary,
        {
            "max_model_p95_ms": 150,
            "max_tool_p95_ms": 25,
            "max_tool_failure_rate": 0,
            "max_model_tokens": 12,
        },
    )
    warning_text = format_trace_warnings(warnings)
    assert "failed calls" in warning_text
    assert "p95 latency" in warning_text
    assert "token usage" in warning_text
