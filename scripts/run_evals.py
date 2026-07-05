#!/usr/bin/env python3
"""Validate and score Engram Slack workflow eval cases."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evals import (
    generate_dry_run_prediction,
    load_eval_cases,
    load_eval_predictions,
    score_predictions,
)


def main() -> int:
    """Run eval validation or score saved responses."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        default="evals/slack_workflows.example.jsonl",
        help="JSONL eval case file.",
    )
    parser.add_argument(
        "--responses",
        help="Optional JSONL prediction file with id, response, route, safety, and model fields.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print loaded case ids and messages.",
    )
    parser.add_argument(
        "--generate-predictions",
        metavar="PATH",
        help="Write dry-run routing/safety predictions to PATH, or '-' for stdout.",
    )
    args = parser.parse_args()

    cases = load_eval_cases(args.cases)
    print(f"Loaded {len(cases)} eval cases from {args.cases}")

    if args.list:
        for case in cases:
            route = case.expected_agent or case.expected_intent or "unspecified"
            flags = []
            if case.expected_background is not None:
                flags.append(f"background={case.expected_background}")
            if case.expected_confirmation_required is not None:
                flags.append(f"confirm={case.expected_confirmation_required}")
            if case.expected_model_profile:
                flags.append(f"model={case.expected_model_profile}")
            suffix = f" ({', '.join(flags)})" if flags else ""
            print(f"- {case.id} [{route}]{suffix}: {case.message}")

    if args.generate_predictions:
        predictions = {
            case.id: generate_dry_run_prediction(case)
            for case in cases
        }
        lines = [
            json.dumps(prediction.__dict__, sort_keys=True)
            for prediction in predictions.values()
        ]
        if args.generate_predictions == "-":
            print("\n".join(lines))
        else:
            path = Path(args.generate_predictions)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(lines) + "\n")
            print(f"Wrote {len(lines)} predictions to {path}")

        if not args.responses:
            scores = score_predictions(cases, predictions)
            return _print_scores(scores)

    if not args.responses:
        return 0

    predictions = load_eval_predictions(args.responses)
    scores = score_predictions(cases, predictions)
    return _print_scores(scores)


def _print_scores(scores) -> int:
    """Print score details and return an exit code."""
    passed = sum(1 for score in scores if score.passed)

    for score in scores:
        status = "PASS" if score.passed else "FAIL"
        details = []
        if score.missing_terms:
            details.append(f"missing={score.missing_terms}")
        if score.forbidden_terms:
            details.append(f"forbidden={score.forbidden_terms}")
        if score.mismatches:
            details.append(f"mismatches={score.mismatches}")
        suffix = f" ({'; '.join(details)})" if details else ""
        print(f"{status} {score.case_id}{suffix}")

    print(f"{passed}/{len(scores)} passed")
    return 0 if passed == len(scores) else 1


if __name__ == "__main__":
    raise SystemExit(main())
