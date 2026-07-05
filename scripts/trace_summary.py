#!/usr/bin/env python3
"""Summarize Engram metadata trace logs."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import TRACE_LOG_PATH
from src.trace_summary import format_trace_summary, load_trace_records, summarize_trace_records


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
    args = parser.parse_args()

    records = load_trace_records(args.path)
    summary = summarize_trace_records(records)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(format_trace_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
