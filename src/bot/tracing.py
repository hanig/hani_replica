"""Structured metadata traces for model and tool calls."""

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import ENABLE_TRACE_LOG, TRACE_LOG_PATH

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TraceEvent:
    """A structured trace record."""

    event_type: str
    payload: dict[str, Any]


class TraceLogger:
    """Append-only JSONL trace logger."""

    def __init__(self, path: str | Path | None = None, enabled: bool | None = None):
        self.path = Path(path) if path else TRACE_LOG_PATH
        self.enabled = ENABLE_TRACE_LOG if enabled is None else enabled
        self._lock = threading.Lock()
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, **payload: Any) -> None:
        """Write a trace event if tracing is enabled."""
        if not self.enabled:
            return

        record = {
            "trace_id": uuid.uuid4().hex,
            "timestamp": time.time(),
            "event_type": event_type,
            **_sanitize_payload(payload),
        }
        try:
            with self._lock:
                with self.path.open("a") as f:
                    f.write(json.dumps(record, sort_keys=True) + "\n")
        except Exception as e:
            logger.warning("Failed to write trace event: %s", e)

    def log_model_call(
        self,
        *,
        caller: str,
        model: str,
        operation: str,
        duration_ms: float,
        success: bool,
        usage: dict[str, Any] | None = None,
        stop_reason: str | None = None,
        error: Exception | str | None = None,
    ) -> None:
        """Record a model API call."""
        self.log(
            "model_call",
            caller=caller,
            model=model,
            operation=operation,
            duration_ms=round(duration_ms, 2),
            success=success,
            usage=usage or {},
            stop_reason=stop_reason,
            error_type=type(error).__name__ if isinstance(error, Exception) else None,
            error=str(error)[:300] if error else None,
        )

    def log_tool_call(
        self,
        *,
        caller: str,
        tool_name: str,
        duration_ms: float,
        success: bool,
        input_keys: list[str] | None = None,
        result_preview: str = "",
        error: Exception | str | None = None,
    ) -> None:
        """Record a tool call without storing raw tool input or output values."""
        self.log(
            "tool_call",
            caller=caller,
            tool_name=tool_name,
            duration_ms=round(duration_ms, 2),
            success=success,
            input_keys=sorted(input_keys or []),
            result_preview_chars=len(result_preview),
            error_type=type(error).__name__ if isinstance(error, Exception) else None,
            error=str(error)[:300] if error else None,
        )


_trace_logger: TraceLogger | None = None


def get_trace_logger() -> TraceLogger:
    """Return the process-global trace logger."""
    global _trace_logger
    if _trace_logger is None:
        _trace_logger = TraceLogger()
    return _trace_logger


def reset_trace_logger() -> None:
    """Reset the process-global trace logger. Intended for tests."""
    global _trace_logger
    _trace_logger = None


def model_usage(response: Any) -> dict[str, Any]:
    """Extract token usage metadata from an Anthropic response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
    }


def _sanitize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop empty values and recursively keep JSON-safe metadata."""
    return {
        key: _json_safe(value)
        for key, value in payload.items()
        if value not in (None, "", [], {})
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
