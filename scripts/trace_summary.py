#!/usr/bin/env python3
"""Summarize Engram metadata trace logs."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import TRACE_LOG_PATH
from src.trace_summary import (
    DEFAULT_THRESHOLDS,
    evaluate_trace_thresholds,
    format_trace_summary,
    format_trace_warnings,
    load_trace_records,
    summarize_trace_records,
)


def main() -> int:
    """Print a trace summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        default=str(TRACE_LOG_PATH),
        help="Trace JSONL path. Defaults to configured TRACE_LOG_PATH.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON.",
    )
    parser.add_argument(
        "--max-model-p95-ms",
        type=float,
        default=DEFAULT_THRESHOLDS["max_model_p95_ms"],
        help="Warn when a model's p95 latency exceeds this threshold.",
    )
    parser.add_argument(
        "--max-tool-p95-ms",
        type=float,
        default=DEFAULT_THRESHOLDS["max_tool_p95_ms"],
        help="Warn when a tool's p95 latency exceeds this threshold.",
    )
    parser.add_argument(
        "--max-tool-failure-rate",
        type=float,
        default=DEFAULT_THRESHOLDS["max_tool_failure_rate"],
        help="Warn when a tool's failure rate exceeds this decimal threshold.",
    )
    parser.add_argument(
        "--max-model-tokens",
        type=int,
        default=DEFAULT_THRESHOLDS["max_model_tokens"],
        help="Warn when a model's total input+output tokens exceed this threshold.",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Exit with status 1 when any threshold warning is produced.",
    )
    args = parser.parse_args()

    records = load_trace_records(args.path)
    summary = summarize_trace_records(records)
    warnings = evaluate_trace_thresholds(
        summary,
        {
            "max_model_p95_ms": args.max_model_p95_ms,
            "max_tool_p95_ms": args.max_tool_p95_ms,
            "max_tool_failure_rate": args.max_tool_failure_rate,
            "max_model_tokens": args.max_model_tokens,
        },
    )
    if args.json:
        summary["warnings"] = warnings
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(format_trace_summary(summary))
        print()
        print(format_trace_warnings(warnings))
    return 1 if args.fail_on_warning and warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
