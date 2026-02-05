"""Slack Block Kit formatters for bot responses."""

import re
from datetime import datetime
from typing import Any


def markdown_to_slack(text: str) -> str:
    """Convert standard markdown to Slack mrkdwn format.

    Args:
        text: Text with standard markdown formatting.

    Returns:
        Text with Slack mrkdwn formatting.
    """
    if not text:
        return text

    # Convert bold: **text** or __text__ -> *text*
    text = re.sub(r'\*\*(.+?)\*\*', r'*\1*', text)
    text = re.sub(r'__(.+?)__', r'*\1*', text)

    # Convert italic: *text* (single) -> _text_ (but not if already bold)
    # This is tricky - we need to avoid converting *bold* to _bold_
    # Only convert single asterisks that aren't part of double asterisks
    # Actually, after converting **bold** to *bold*, single * for italic becomes ambiguous
    # So we only convert _text_ style italic (underscore) which is less common in markdown

    # Convert links: [text](url) -> <url|text>
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<\2|\1>', text)

    # Convert inline code: `code` stays the same (Slack supports this)

    # Convert headers: # Header -> *Header*
    text = re.sub(r'^#{1,6}\s+(.+)$', r'*\1*', text, flags=re.MULTILINE)

    # Convert strikethrough: ~~text~~ -> ~text~
    text = re.sub(r'~~(.+?)~~', r'~\1~', text)

    return text


def format_help_message() -> str:
    """Format help message."""
    return """*Here's what I can help you with:*

*Calendar:*
• "What's on my calendar today?"
• "When am I free tomorrow afternoon?"
• "Show my meetings for next week"

*Email:*
• "Search emails about quarterly report"
• "Find emails from john@example.com"
• "Draft an email to team about the meeting"

*Documents:*
• "Search for documents about project X"
• "Find files mentioning budget 2024"

*GitHub:*
• "Show my open PRs"
• "Search code for authentication"
• "Create an issue in repo-name"

*General:*
• "What did I miss yesterday?" (daily briefing)
• "Search for anything about machine learning"

Just message me or @mention me with your question!"""


def format_error_message(error: str) -> str:
    """Format an error message.

    Args:
        error: Error description.

    Returns:
        Formatted error message.
    """
    return f"Sorry, something went wrong: {error}"


def format_search_results(results: list[dict], query: str) -> dict[str, Any]:
    """Format semantic search results.

    Args:
        results: List of search results.
        query: Original search query.

    Returns:
        Dictionary with 'text' and 'blocks'.
    """
    if not results:
        return {"text": f"No results found for '{query}'"}

    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Search results for:* _{query}_",
            },
        },
        {"type": "divider"},
    ]

    for i, result in enumerate(results[:5], 1):
        score = result.get("score", 0)
        title = result.get("metadata", {}).get("title", "Untitled")
        source_type = result.get("collection", result.get("metadata", {}).get("source_type", ""))
        snippet = result.get("text", "")[:200]

        # Add ellipsis if truncated
        if len(result.get("text", "")) > 200:
            snippet += "..."

        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*{i}. {title}*\n_{source_type}_ • Score: {score:.2f}\n>{snippet}",
            },
        })

    return {
        "text": f"Found {len(results)} results for '{query}'",
        "blocks": blocks,
    }


def format_calendar_events(events: list[dict], date_str: str) -> dict[str, Any]:
    """Format calendar events.

    Args:
        events: List of calendar events.
        date_str: Date description (e.g., "today", "tomorrow").

    Returns:
        Dictionary with 'text' and 'blocks'.
    """
    if not events:
        return {"text": f"No events scheduled for {date_str}."}

    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Calendar for {date_str}:*",
            },
        },
        {"type": "divider"},
    ]

    for event in events:
        start = event.get("start")
        end = event.get("end")
        title = event.get("summary", "Untitled")
        location = event.get("location", "")
        account = event.get("account", "")

        # Format time
        if event.get("is_all_day"):
            time_str = "All day"
        elif start:
            if isinstance(start, datetime):
                time_str = start.strftime("%I:%M %p")
                if end and isinstance(end, datetime):
                    time_str += f" - {end.strftime('%I:%M %p')}"
            else:
                time_str = str(start)
        else:
            time_str = ""

        # Build event text
        event_text = f"*{time_str}* - {title}"
        if location:
            event_text += f"\n   :round_pushpin: {location}"
        if account:
            event_text += f"\n   _{account}_"

        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": event_text},
        })

    return {
        "text": f"{len(events)} events for {date_str}",
        "blocks": blocks,
    }


def format_availability(free_slots: list[dict], date_str: str) -> dict[str, Any]:
    """Format availability slots.

    Args:
        free_slots: List of free time slots.
        date_str: Date description.

    Returns:
        Dictionary with 'text' and 'blocks'.
    """
    if not free_slots:
        return {"text": f"No available time slots for {date_str}."}

    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Available times for {date_str}:*",
            },
        },
        {"type": "divider"},
    ]

    for slot in free_slots:
        start = slot.get("start")
        end = slot.get("end")
        duration = slot.get("duration_minutes", 0)

        if isinstance(start, datetime) and isinstance(end, datetime):
            time_str = f"{start.strftime('%I:%M %p')} - {end.strftime('%I:%M %p')}"
        else:
            time_str = f"{start} - {end}"

        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f":white_check_mark: {time_str} ({duration} min)",
            },
        })

    return {
        "text": f"{len(free_slots)} available time slots",
        "blocks": blocks,
    }


def format_email_results(emails: list[dict], query: str) -> dict[str, Any]:
    """Format email search results.

    Args:
        emails: List of email results.
        query: Search query.

    Returns:
        Dictionary with 'text' and 'blocks'.
    """
    if not emails:
        return {"text": f"No emails found for '{query}'"}

    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Emails matching:* _{query}_",
            },
        },
        {"type": "divider"},
    ]

    for email in emails[:5]:
        subject = email.get("subject", "(no subject)")
        sender = email.get("from", email.get("account_email", "unknown"))
        timestamp = email.get("timestamp")
        account = email.get("account", "")

        # Format date
        if isinstance(timestamp, datetime):
            date_str = timestamp.strftime("%b %d, %Y")
        else:
            date_str = str(timestamp) if timestamp else ""

        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*{subject}*\nFrom: {sender}\n{date_str} • _{account}_",
            },
        })

    return {
        "text": f"Found {len(emails)} emails",
        "blocks": blocks,
    }


def format_github_prs(prs: list[dict]) -> dict[str, Any]:
    """Format GitHub PR list.

    Args:
        prs: List of pull requests.

    Returns:
        Dictionary with 'text' and 'blocks'.
    """
    if not prs:
        return {"text": "No open pull requests found."}

    blocks = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*Your Open Pull Requests:*"},
        },
        {"type": "divider"},
    ]

    for pr in prs[:10]:
        title = pr.get("title", "Untitled")
        repo = pr.get("repo", "unknown")
        number = pr.get("number", "")
        url = pr.get("url", "")
        state = pr.get("state", "open")

        status_emoji = ":large_green_circle:" if state == "open" else ":red_circle:"

        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"{status_emoji} <{url}|#{number}: {title}>\n_{repo}_",
            },
        })

    return {
        "text": f"{len(prs)} open pull requests",
        "blocks": blocks,
    }


def format_github_issues(issues: list[dict], query: str | None = None) -> dict[str, Any]:
    """Format GitHub issues list.

    Args:
        issues: List of issues.
        query: Optional search query.

    Returns:
        Dictionary with 'text' and 'blocks'.
    """
    if not issues:
        msg = f"No issues found for '{query}'" if query else "No issues found."
        return {"text": msg}

    header = f"*Issues matching:* _{query}_" if query else "*Your Issues:*"

    blocks = [
        {"type": "section", "text": {"type": "mrkdwn", "text": header}},
        {"type": "divider"},
    ]

    for issue in issues[:10]:
        title = issue.get("title", "Untitled")
        repo = issue.get("repo", "unknown")
        number = issue.get("number", "")
        url = issue.get("url", "")
        state = issue.get("state", "open")
        labels = issue.get("labels", [])

        status_emoji = ":large_green_circle:" if state == "open" else ":white_check_mark:"
        label_str = " ".join(f"`{l}`" for l in labels[:3]) if labels else ""

        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"{status_emoji} <{url}|#{number}: {title}>\n_{repo}_ {label_str}",
            },
        })

    return {
        "text": f"{len(issues)} issues",
        "blocks": blocks,
    }


def format_briefing(briefing: dict) -> dict[str, Any]:
    """Format daily briefing.

    Args:
        briefing: Briefing data.

    Returns:
        Dictionary with 'text' and 'blocks'.
    """
    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": ":sunrise: Daily Briefing"},
        },
        {"type": "divider"},
    ]

    # Calendar section
    events = briefing.get("events", [])
    events_text = f"*:calendar: Calendar:* {len(events)} events today"
    if events:
        first_events = [e.get("summary", "Event") for e in events[:3]]
        events_text += f"\n• {chr(10).join('• ' + e for e in first_events)}"

    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": events_text},
    })

    # Unread emails section
    unread = briefing.get("unread_counts", {})
    total_unread = sum(unread.values())
    email_text = f"*:email: Unread Emails:* {total_unread} total"
    if unread:
        account_counts = [f"{acc}: {count}" for acc, count in unread.items() if count > 0]
        if account_counts:
            email_text += f"\n• {', '.join(account_counts[:4])}"

    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": email_text},
    })

    # GitHub section
    prs = briefing.get("open_prs", [])
    issues = briefing.get("open_issues", [])
    github_text = f"*:octocat: GitHub:* {len(prs)} open PRs, {len(issues)} assigned issues"

    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": github_text},
    })

    # Todoist overdue tasks section
    overdue_tasks = briefing.get("overdue_tasks", [])
    overdue_text = f"*:white_check_mark: Todoist Overdue:* {len(overdue_tasks)} tasks"
    if overdue_tasks:
        first_tasks = [t.get("content", "Task") for t in overdue_tasks[:3]]
        overdue_text += f"\n• {chr(10).join('• ' + t for t in first_tasks)}"

    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": overdue_text},
    })

    return {
        "text": "Daily Briefing",
        "blocks": blocks,
    }


def format_confirmation(
    action_type: str,
    preview: str,
    action_id: str,
) -> dict[str, Any]:
    """Format action confirmation prompt.

    Args:
        action_type: Type of action (e.g., "Create GitHub Issue").
        preview: Preview of what will be created/done.
        action_id: Unique action ID for buttons.

    Returns:
        Dictionary with 'text' and 'blocks'.
    """
    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*{action_type}*\n\n{preview}",
            },
        },
        {"type": "divider"},
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Confirm"},
                    "style": "primary",
                    "action_id": f"confirm_action:{action_id}",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Cancel"},
                    "style": "danger",
                    "action_id": f"cancel_action:{action_id}",
                },
            ],
        },
    ]

    return {
        "text": f"Confirm: {action_type}",
        "blocks": blocks,
    }
