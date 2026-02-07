"""GitHub specialist agent."""

import logging
from typing import Any

from .base import BaseAgent, AgentType
from ..conversation import ConversationContext
from ...config import GITHUB_USERNAME, GITHUB_ORG

logger = logging.getLogger(__name__)


# GitHub-related keywords for routing
GITHUB_KEYWORDS = {
    "github", "git", "repo", "repository", "pr", "prs", "pull", "request",
    "issue", "issues", "commit", "branch", "merge", "code", "review",
    "fork", "clone", "push", "bug", "feature",
}
# Dynamically add org name as keyword if configured
if GITHUB_ORG:
    GITHUB_KEYWORDS.add(GITHUB_ORG.lower())


class GitHubAgent(BaseAgent):
    """Specialist agent for GitHub-related tasks.

    Handles:
    - Checking pull request status
    - Viewing issues
    - Searching code across repositories
    - Creating new issues
    """

    AGENT_TYPE = AgentType.GITHUB
    MAX_ITERATIONS = 5

    @property
    def tool_names(self) -> list[str]:
        """GitHub-specific tools."""
        return [
            "GetGitHubPRsTool",
            "GetGitHubIssuesTool",
            "SearchGitHubCodeTool",
            "CreateGitHubIssueTool",
            "RespondToUserTool",
        ]

    @property
    def system_prompt(self) -> str:
        """GitHub-focused system prompt."""
        username_info = GITHUB_USERNAME or "not configured"
        org_info = GITHUB_ORG or "not configured"

        return f"""You are a GitHub specialist, a personal assistant.

Your expertise is managing GitHub repositories, PRs, and issues.

Today's date: {{current_date}}

CONFIGURATION:
- Primary username: {username_info}
- Primary organization: {org_info}

CAPABILITIES:
- List pull requests (open, closed, or all)
- List issues assigned or created by user
- Search code across repositories
- Create new issues with labels

GUIDELINES:
1. For PR status, show state, title, and review status
2. For issues, include priority labels if present
3. For code search, show file paths and relevant snippets
4. When creating issues, ask for confirmation first
5. Use RespondToUserTool to send your final response

CODE SEARCH TIPS:
- Search across all accessible repos by default
- Use repo parameter to narrow to specific repository
- Show context around matches

ISSUE CREATION:
- Always include clear title and description
- Suggest appropriate labels based on content
- Never create without user confirmation

RESPONSE FORMAT:
- PRs: Title, number, state, author, review status
- Issues: Title, number, labels, assignee
- Code: File path, match context, repository"""

    @property
    def description(self) -> str:
        return "GitHub expert: PRs, issues, code search, repository management"

    def can_handle(self, message: str, context: ConversationContext) -> float:
        """Estimate relevance for GitHub tasks."""
        message_lower = message.lower()
        words = set(message_lower.split())

        # Check for GitHub keywords
        matches = words & GITHUB_KEYWORDS
        if matches:
            # More matches = higher confidence
            return min(0.3 + (len(matches) * 0.2), 0.95)

        # Check for repository patterns
        if "/" in message and any(c.isalnum() for c in message):
            # Could be owner/repo pattern
            return 0.3

        # Check for PR/issue number patterns
        import re
        if re.search(r"#\d+", message):
            return 0.5

        return 0.0
