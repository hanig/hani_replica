#!/usr/bin/env python3
"""Smoke-test configured Anthropic model profiles."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from anthropic import Anthropic

from src.config import ANTHROPIC_API_KEY, MODEL_REGISTRY, get_model_profile


def _profile_names(args: argparse.Namespace) -> list[str]:
    if args.all:
        return sorted(MODEL_REGISTRY)
    return args.profile


def main() -> int:
    """Run a tiny request against one or more configured model profiles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "profile",
        nargs="*",
        default=["agent"],
        help="Model profile(s) to test. Defaults to 'agent'.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test every configured profile.",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with OK.",
        help="Prompt to send for the smoke test.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2,
        help="Maximum output tokens for each smoke request.",
    )
    args = parser.parse_args()

    if not ANTHROPIC_API_KEY:
        print("ANTHROPIC_API_KEY is not configured", file=sys.stderr)
        return 2

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    failures = 0
    tested_models: set[str] = set()

    for profile_name in _profile_names(args):
        profile = get_model_profile(profile_name)
        if profile.model in tested_models and not args.profile:
            continue
        tested_models.add(profile.model)

        try:
            client.messages.create(
                model=profile.model,
                max_tokens=args.max_tokens,
                messages=[{"role": "user", "content": args.prompt}],
            )
            print(f"OK {profile.name}: {profile.model}")
        except Exception as e:
            failures += 1
            print(f"FAIL {profile.name}: {profile.model}: {e}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
