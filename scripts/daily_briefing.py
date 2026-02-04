#!/usr/bin/env python3
"""Generate a daily briefing summary."""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.integrations.github_client import GitHubClient
from src.integrations.google_multi import MultiGoogleManager
from src.integrations.todoist_client import TodoistClient
from src.integrations.slack import SlackClient
from src.config import SLACK_AUTHORIZED_USERS

logger = logging.getLogger(__name__)


def generate_briefing(date: datetime | None = None) -> str:
    """Generate a daily briefing.

    Args:
        date: Date to generate briefing for. Defaults to today.

    Returns:
        Formatted briefing text.
    """
    if date is None:
        date = datetime.now(timezone.utc)

    date_str = date.strftime("%A, %B %d, %Y")
    lines = [
        f"",
        f"{'=' * 60}",
        f"  DAILY BRIEFING - {date_str}",
        f"{'=' * 60}",
        f"",
    ]

    # Calendar
    lines.append("📅 TODAY'S CALENDAR")
    lines.append("-" * 40)

    try:
        mg = MultiGoogleManager()
        events = mg.get_all_calendars_for_date(date)

        if events:
            for event in events:
                start = event.get("start")
                summary = event.get("summary", "Untitled")
                account = event.get("account", "")

                if event.get("is_all_day"):
                    time_str = "All day"
                elif start:
                    time_str = start.strftime("%I:%M %p")
                else:
                    time_str = ""

                lines.append(f"  {time_str:12} {summary} ({account})")
        else:
            lines.append("  No events scheduled")
    except Exception as e:
        lines.append(f"  Error loading calendar: {e}")

    lines.append("")

    # Unread emails
    lines.append("📧 UNREAD EMAILS")
    lines.append("-" * 40)

    try:
        mg = MultiGoogleManager()
        counts = mg.get_unread_counts()
        total = sum(counts.values())

        if total > 0:
            for account, count in counts.items():
                if count > 0:
                    lines.append(f"  {account:15} {count:5} unread")
            lines.append(f"  {'-' * 20}")
            lines.append(f"  {'TOTAL':15} {total:5} unread")
        else:
            lines.append("  Inbox Zero! 🎉")
    except Exception as e:
        lines.append(f"  Error loading email counts: {e}")

    lines.append("")

    # GitHub
    lines.append("🐙 GITHUB")
    lines.append("-" * 40)

    try:
        gh = GitHubClient()

        # Open PRs
        prs = gh.get_my_prs(state="open", max_results=10)
        if prs:
            lines.append(f"  Open PRs: {len(prs)}")
            for pr in prs[:3]:
                lines.append(f"    • #{pr['number']}: {pr['title'][:40]}")
            if len(prs) > 3:
                lines.append(f"    ... and {len(prs) - 3} more")
        else:
            lines.append("  No open PRs")

        lines.append("")

        # Assigned issues
        issues = gh.get_my_issues(state="open", max_results=10)
        if issues:
            lines.append(f"  Assigned Issues: {len(issues)}")
            for issue in issues[:3]:
                lines.append(f"    • #{issue['number']}: {issue['title'][:40]}")
            if len(issues) > 3:
                lines.append(f"    ... and {len(issues) - 3} more")
        else:
            lines.append("  No assigned issues")

    except Exception as e:
        lines.append(f"  Error loading GitHub: {e}")

    lines.append("")

    # Todoist Tasks
    lines.append("✅ TODOIST TASKS")
    lines.append("-" * 40)

    try:
        todoist = TodoistClient()
        projects = todoist.list_projects()
        project_map = {p["id"]: p["name"] for p in projects}

        # Get tasks due today or overdue
        all_tasks = todoist.list_tasks()

        # Separate by priority and due date
        overdue = []
        due_today = []
        upcoming = []
        no_date = []

        today_str = date.strftime("%Y-%m-%d")

        for task in all_tasks:
            due = task.get("due")
            if due:
                due_date = due.get("date", "")
                if due_date < today_str:
                    overdue.append(task)
                elif due_date == today_str:
                    due_today.append(task)
                else:
                    upcoming.append(task)
            else:
                no_date.append(task)

        # Show overdue (high priority)
        if overdue:
            lines.append(f"  ⚠️  OVERDUE ({len(overdue)}):")
            for task in overdue[:5]:
                proj = project_map.get(task.get("project_id"), "Inbox")
                due_str = task.get("due", {}).get("date", "")
                lines.append(f"    • [{proj}] {task['content'][:35]} (due {due_str})")
            if len(overdue) > 5:
                lines.append(f"    ... and {len(overdue) - 5} more overdue")
            lines.append("")

        # Show due today
        if due_today:
            lines.append(f"  📌 DUE TODAY ({len(due_today)}):")
            for task in due_today[:5]:
                proj = project_map.get(task.get("project_id"), "Inbox")
                lines.append(f"    • [{proj}] {task['content'][:40]}")
            if len(due_today) > 5:
                lines.append(f"    ... and {len(due_today) - 5} more")
        else:
            lines.append("  No tasks due today")

        # Summary
        lines.append("")
        lines.append(f"  Total active tasks: {len(all_tasks)}")

    except Exception as e:
        lines.append(f"  Error loading Todoist: {e}")

    lines.append("")

    # Availability
    lines.append("🟢 AVAILABILITY")
    lines.append("-" * 40)

    try:
        mg = MultiGoogleManager()
        slots = mg.check_availability(date, duration_minutes=30)

        if slots:
            lines.append(f"  {len(slots)} free slots today:")
            for slot in slots[:5]:
                start = slot["start"]
                end = slot["end"]
                duration = slot["duration_minutes"]

                if hasattr(start, "strftime"):
                    time_str = f"{start.strftime('%I:%M %p')} - {end.strftime('%I:%M %p')}"
                else:
                    time_str = f"{start} - {end}"

                lines.append(f"    {time_str} ({duration} min)")

            if len(slots) > 5:
                lines.append(f"    ... and {len(slots) - 5} more slots")
        else:
            lines.append("  No available slots today")

    except Exception as e:
        lines.append(f"  Error checking availability: {e}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("")

    return "\n".join(lines)


def send_to_slack(briefing: str, user_id: str | None = None) -> bool:
    """Send briefing to Slack.

    Args:
        briefing: The briefing text.
        user_id: Slack user ID to DM. Defaults to first authorized user.

    Returns:
        True if sent successfully.
    """
    try:
        slack = SlackClient()

        # Default to first authorized user
        if not user_id:
            if SLACK_AUTHORIZED_USERS:
                user_id = SLACK_AUTHORIZED_USERS[0]
            else:
                logger.error("No authorized Slack users configured")
                return False

        # Open DM channel with user
        response = slack._client.conversations_open(users=[user_id])
        channel_id = response["channel"]["id"]

        # Send the briefing
        slack._client.chat_postMessage(
            channel=channel_id,
            text=f"```{briefing}```",
            mrkdwn=True,
        )

        logger.info(f"Briefing sent to Slack user {user_id}")
        return True

    except Exception as e:
        logger.error(f"Error sending briefing to Slack: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate daily briefing")
    parser.add_argument(
        "--output",
        type=str,
        help="Output file (default: print to stdout)",
    )
    parser.add_argument(
        "--slack",
        action="store_true",
        help="Send briefing to Slack DM",
    )
    parser.add_argument(
        "--slack-user",
        type=str,
        help="Slack user ID to send to (default: first authorized user)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Don't print to stdout (useful with --slack)",
    )

    args = parser.parse_args()

    briefing = generate_briefing()

    # Send to Slack if requested
    if args.slack:
        success = send_to_slack(briefing, args.slack_user)
        if success:
            print("Briefing sent to Slack")
        else:
            print("Failed to send briefing to Slack")
            sys.exit(1)

    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            f.write(briefing)
        print(f"Briefing saved to {args.output}")

    # Print to stdout unless quiet
    if not args.quiet:
        print(briefing)


if __name__ == "__main__":
    main()
