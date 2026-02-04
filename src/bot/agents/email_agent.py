"""Email specialist agent."""

import logging
from typing import Any

from .base import BaseAgent, AgentType
from ..conversation import ConversationContext

logger = logging.getLogger(__name__)


# Email-related keywords for routing
EMAIL_KEYWORDS = {
    "email", "mail", "inbox", "unread", "message", "send",
    "reply", "draft", "from", "to", "subject", "attachment",
    "gmail", "sent", "received", "forward", "cc", "bcc",
}


class EmailAgent(BaseAgent):
    """Specialist agent for email-related tasks.

    Handles:
    - Searching emails across multiple accounts
    - Checking unread counts
    - Creating email drafts
    - Finding emails from specific people or about topics
    """

    AGENT_TYPE = AgentType.EMAIL
    MAX_ITERATIONS = 5

    @property
    def tool_names(self) -> list[str]:
        """Email-specific tools."""
        return [
            "SearchEmailsTool",
            "GetUnreadCountsTool",
            "CreateEmailDraftTool",
            "SendEmailTool",
            "FindPersonTool",  # Useful for resolving contact names
            "RespondToUserTool",
        ]

    @property
    def system_prompt(self) -> str:
        """Email-focused system prompt."""
        return """You are an email management specialist for Hani's personal assistant.

Your expertise is managing emails across multiple Google accounts.

Today's date: {current_date}

ACCOUNTS (in priority order):
- Tier 1 (searched first): arc (arcinstitute.org), personal (gmail.com)
- Tier 2: tahoe (tahoebio.ai), therna, amplify

AVAILABLE TOOLS:
- SearchEmailsTool: Search emails with Gmail query syntax
- GetUnreadCountsTool: Check unread counts across all accounts
- CreateEmailDraftTool: Create email drafts
- SendEmailTool: Send emails (use account param: "arc", "personal", "tahoe", etc.)
- FindPersonTool: Find contacts in the knowledge graph
- RespondToUserTool: Send your final response to the user

IMPORTANT: You MUST use these actual tools. Do NOT generate fake XML or pretend to call functions.
When asked to send an email, use the SendEmailTool with parameters: to, subject, body, account.

SEARCH TIPS:
- Use "from:" for sender, "to:" for recipient
- Use "subject:" for subject line
- Use "after:" and "before:" for date ranges
- Combine with AND/OR for complex queries

GUIDELINES:
1. When searching, start with Tier 1 accounts
2. Summarize results concisely (subject, sender, date)
3. For sending emails, confirm content with user first, then use SendEmailTool
4. Use FindPersonTool to resolve nicknames to email addresses
5. Use RespondToUserTool to send your final response

SENDING EMAILS:
- You CAN send emails using SendEmailTool
- Set "account" to specify which account to send from (e.g., "personal", "arc")
- Always confirm with user before sending
- Example: SendEmailTool(to="user@example.com", subject="Hello", body="Message", account="personal")"""

    @property
    def description(self) -> str:
        return "Email expert: searching mail, checking inbox, creating drafts"

    def can_handle(self, message: str, context: ConversationContext) -> float:
        """Estimate relevance for email tasks."""
        message_lower = message.lower()
        words = set(message_lower.split())

        # Check for email keywords
        matches = words & EMAIL_KEYWORDS
        if matches:
            # More matches = higher confidence
            return min(0.3 + (len(matches) * 0.15), 0.95)

        # Check for email-like patterns
        if "@" in message or "draft" in message_lower:
            return 0.6

        # Check for inbox/unread patterns
        if "unread" in message_lower or "inbox" in message_lower:
            return 0.7

        return 0.0
