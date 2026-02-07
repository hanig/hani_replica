"""Email actions that require confirmation."""

import logging
from dataclasses import dataclass
from typing import Any

from .confirmable import PendingAction
from ...config import PRIMARY_ACCOUNT

logger = logging.getLogger(__name__)


@dataclass
class CreateDraftAction(PendingAction):
    """Action to create an email draft (NEVER sends)."""

    to: str = ""
    subject: str = ""
    body: str = ""
    cc: str = ""
    account: str = ""  # Which account to create draft in (resolved to PRIMARY_ACCOUNT if empty)
    subject_hint: str = ""  # Hint for generating subject
    _state: str = "to"

    def __post_init__(self):
        if not self.account:
            self.account = PRIMARY_ACCOUNT

    def is_ready(self) -> bool:
        """Check if we have enough info to create the draft."""
        return bool(self.to and self.subject and self.body)

    def get_next_prompt(self) -> str:
        """Get the prompt for the next required field."""
        if not self.to:
            return "Who should I address this email to? (email address)"
        if not self.subject:
            if self.subject_hint:
                return f"What should the subject line be? (suggested: '{self.subject_hint}')"
            return "What should the subject line be?"
        if not self.body:
            return "What should the email say?"
        return ""

    def update_from_input(self, text: str) -> None:
        """Update action fields from user input."""
        text = text.strip()

        if not self.to:
            self.to = text
            self._state = "subject"
        elif not self.subject:
            self.subject = text
            self._state = "body"
        elif not self.body:
            self.body = text
            self._state = "done"

    def get_preview(self) -> str:
        """Get a preview of the email draft."""
        preview = f"*To:* {self.to}\n*Subject:* {self.subject}"
        if self.cc:
            preview += f"\n*CC:* {self.cc}"
        preview += f"\n*Account:* {self.account}"

        body_preview = self.body[:300]
        if len(self.body) > 300:
            body_preview += "..."
        preview += f"\n\n*Body:*\n{body_preview}"

        preview += "\n\n_This will create a draft - it will NOT be sent automatically._"

        return preview

    def execute(self) -> dict[str, Any]:
        """Create the email draft."""
        from ...integrations.gmail import GmailClient

        try:
            client = GmailClient(self.account)
            draft = client.create_draft(
                to=self.to,
                subject=self.subject,
                body=self.body,
                cc=self.cc if self.cc else None,
            )

            draft_id = draft.get("id", "unknown")

            return {
                "success": True,
                "message": f"Created draft (ID: {draft_id}). Open Gmail to review and send.",
                "draft": draft,
            }

        except Exception as e:
            logger.error(f"Error creating draft: {e}")
            return {
                "success": False,
                "message": f"Failed to create draft: {str(e)}",
            }

    def get_action_type(self) -> str:
        return "Create Email Draft"
