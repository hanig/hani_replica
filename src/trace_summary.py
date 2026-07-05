"""Summaries for metadata-only runtime trace logs."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_THRESHOLDS = {
    "max_model_p95_ms": 20_000,
    "max_tool_p95_ms": 10_000,
    "max_tool_failure_rate": 0.05,
    "max_model_tokens": 250_000,
}


def load_trace_records(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL trace records from disk."""
    records: list[dict[str, Any]] = []
    path = Path(path)
    if not path.exists():
        return records

    with path.open() as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {e}") from e
    return records


def summarize_trace_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate trace records by event type, model, and tool."""
    model_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tool_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failures: list[dict[str, Any]] = []

    for record in records:
        if record.get("event_type") == "model_call":
            model_groups[str(record.get("model", "unknown"))].append(record)
        elif record.get("event_type") == "tool_call":
            tool_groups[str(record.get("tool_name", "unknown"))].append(record)

        if record.get("success") is False:
            failures.append(_failure_record(record))

    return {
        "events": len(records),
        "model_calls": sum(len(group) for group in model_groups.values()),
        "tool_calls": sum(len(group) for group in tool_groups.values()),
        "failures": len(failures),
        "models": {
            model: _summarize_calls(group)
            for model, group in sorted(model_groups.items())
        },
        "tools": {
            tool: _summarize_calls(group)
            for tool, group in sorted(tool_groups.items())
        },
        "recent_failures": failures[-10:],
    }


def format_trace_summary(summary: dict[str, Any]) -> str:
    """Format a trace summary as readable text."""
    lines = [
        "Trace summary",
        f"Events: {summary['events']}",
        f"Model calls: {summary['model_calls']}",
        f"Tool calls: {summary['tool_calls']}",
        f"Failures: {summary['failures']}",
    ]

    if summary["models"]:
        lines.append("")
        lines.append("Models:")
        for model, stats in summary["models"].items():
            lines.append(_format_stats_line(model, stats, include_tokens=True))

    if summary["tools"]:
        lines.append("")
        lines.append("Tools:")
        for tool, stats in summary["tools"].items():
            lines.append(_format_stats_line(tool, stats, include_tokens=False))

    if summary["recent_failures"]:
        lines.append("")
        lines.append("Recent failures:")
        for failure in summary["recent_failures"]:
            target = failure.get("model") or failure.get("tool_name") or failure.get("caller", "unknown")
            error = failure.get("error_type") or failure.get("error") or "unknown error"
            lines.append(f"- {failure.get('event_type', 'event')} {target}: {error}")

    return "\n".join(lines)


def evaluate_trace_thresholds(
    summary: dict[str, Any],
    thresholds: dict[str, float | int] | None = None,
) -> list[str]:
    """Return warning messages for trace metrics above thresholds."""
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    warnings: list[str] = []

    for model, stats in summary.get("models", {}).items():
        if stats["failures"]:
            warnings.append(f"Model {model} had {stats['failures']} failed calls")
        if stats["p95_ms"] > thresholds["max_model_p95_ms"]:
            warnings.append(
                f"Model {model} p95 latency {stats['p95_ms']}ms exceeds "
                f"{thresholds['max_model_p95_ms']}ms"
            )
        total_tokens = stats["input_tokens"] + stats["output_tokens"]
        if total_tokens > thresholds["max_model_tokens"]:
            warnings.append(
                f"Model {model} token usage {total_tokens} exceeds "
                f"{thresholds['max_model_tokens']}"
            )

    for tool, stats in summary.get("tools", {}).items():
        if stats["p95_ms"] > thresholds["max_tool_p95_ms"]:
            warnings.append(
                f"Tool {tool} p95 latency {stats['p95_ms']}ms exceeds "
                f"{thresholds['max_tool_p95_ms']}ms"
            )
        failure_rate = stats["failures"] / stats["calls"] if stats["calls"] else 0
        if failure_rate > thresholds["max_tool_failure_rate"]:
            warnings.append(
                f"Tool {tool} failure rate {failure_rate:.1%} exceeds "
                f"{thresholds['max_tool_failure_rate']:.1%}"
            )

    return warnings


def format_trace_warnings(warnings: list[str]) -> str:
    """Format trace threshold warnings."""
    if not warnings:
        return "No trace threshold warnings."
    return "Trace threshold warnings:\n" + "\n".join(f"- {warning}" for warning in warnings)


def _summarize_calls(records: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [
        float(record.get("duration_ms", 0))
        for record in records
        if record.get("duration_ms") is not None
    ]
    usage = [
        record.get("usage", {})
        for record in records
        if isinstance(record.get("usage"), dict)
    ]
    return {
        "calls": len(records),
        "failures": sum(1 for record in records if record.get("success") is False),
        "avg_ms": round(sum(durations) / len(durations), 2) if durations else 0,
        "p95_ms": round(_percentile(durations, 0.95), 2) if durations else 0,
        "input_tokens": sum(_safe_int(item.get("input_tokens")) for item in usage),
        "output_tokens": sum(_safe_int(item.get("output_tokens")) for item in usage),
    }


def _format_stats_line(name: str, stats: dict[str, Any], *, include_tokens: bool) -> str:
    line = (
        f"- {name}: calls={stats['calls']} failures={stats['failures']} "
        f"avg={stats['avg_ms']}ms p95={stats['p95_ms']}ms"
    )
    if include_tokens:
        line += f" input={stats['input_tokens']} output={stats['output_tokens']}"
    return line


def _failure_record(record: dict[str, Any]) -> dict[str, Any]:
    keys = ["event_type", "caller", "model", "tool_name", "error_type", "error"]
    return {
        key: record[key]
        for key in keys
        if key in record
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * quantile)))
    return ordered[index]


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
