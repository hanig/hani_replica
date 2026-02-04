"""Integration modules for external services."""

from .google_auth import get_credentials, run_oauth_flow
from .google_multi import MultiGoogleManager
from .gmail import GmailClient
from .gdrive import DriveClient
from .gdocs import DocsClient
from .gcalendar import CalendarClient
from .github_client import GitHubClient
from .notion_client import NotionClient
from .slack import SlackClient
from .todoist_client import TodoistClient

__all__ = [
    "get_credentials",
    "run_oauth_flow",
    "MultiGoogleManager",
    "GmailClient",
    "DriveClient",
    "DocsClient",
    "CalendarClient",
    "GitHubClient",
    "NotionClient",
    "SlackClient",
    "TodoistClient",
]
