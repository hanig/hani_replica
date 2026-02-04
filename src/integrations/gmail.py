"""Gmail API client."""

import base64
import logging
from datetime import datetime
from email.mime.text import MIMEText
from typing import Any

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from .google_auth import get_credentials

logger = logging.getLogger(__name__)


class GmailClient:
    """Client for interacting with Gmail API."""

    def __init__(self, account: str):
        """Initialize Gmail client for a specific account.

        Args:
            account: Account identifier (e.g., "arc", "personal").
        """
        self.account = account
        self._service = None

    @property
    def service(self):
        """Lazily initialize the Gmail service."""
        if self._service is None:
            creds = get_credentials(self.account)
            if not creds:
                raise RuntimeError(f"No valid credentials for account '{self.account}'")
            self._service = build("gmail", "v1", credentials=creds)
        return self._service

    def get_profile(self) -> dict:
        """Get user profile information."""
        return self.service.users().getProfile(userId="me").execute()

    def list_messages(
        self,
        query: str = "",
        max_results: int = 100,
        page_token: str | None = None,
        label_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """List messages matching a query.

        Args:
            query: Gmail search query (same syntax as Gmail search box).
            max_results: Maximum number of messages to return.
            page_token: Token for pagination.
            label_ids: Filter by label IDs (e.g., ["INBOX", "UNREAD"]).

        Returns:
            Dictionary with 'messages' list and optional 'nextPageToken'.
        """
        try:
            params = {
                "userId": "me",
                "maxResults": min(max_results, 500),
                "q": query,
            }
            if page_token:
                params["pageToken"] = page_token
            if label_ids:
                params["labelIds"] = label_ids

            return self.service.users().messages().list(**params).execute()
        except HttpError as e:
            logger.error(f"Error listing messages: {e}")
            raise

    def get_message(
        self, message_id: str, format: str = "full"
    ) -> dict[str, Any] | None:
        """Get a specific message by ID.

        Args:
            message_id: The message ID.
            format: Response format ("minimal", "full", "raw", "metadata").

        Returns:
            Message data or None if not found.
        """
        try:
            return (
                self.service.users()
                .messages()
                .get(userId="me", id=message_id, format=format)
                .execute()
            )
        except HttpError as e:
            if e.resp.status == 404:
                return None
            logger.error(f"Error getting message {message_id}: {e}")
            raise

    def get_thread(self, thread_id: str) -> dict[str, Any] | None:
        """Get a thread by ID.

        Args:
            thread_id: The thread ID.

        Returns:
            Thread data with all messages or None if not found.
        """
        try:
            return (
                self.service.users()
                .threads()
                .get(userId="me", id=thread_id, format="full")
                .execute()
            )
        except HttpError as e:
            if e.resp.status == 404:
                return None
            logger.error(f"Error getting thread {thread_id}: {e}")
            raise

    def search_messages(
        self,
        query: str,
        max_results: int = 100,
        include_spam_trash: bool = False,
    ) -> list[dict[str, Any]]:
        """Search messages and return full message data.

        Args:
            query: Gmail search query.
            max_results: Maximum number of messages to return.
            include_spam_trash: Whether to include spam/trash.

        Returns:
            List of message dictionaries with full content.
        """
        messages = []
        page_token = None

        while len(messages) < max_results:
            response = self.list_messages(
                query=query,
                max_results=min(max_results - len(messages), 100),
                page_token=page_token,
            )

            if "messages" not in response:
                break

            for msg in response["messages"]:
                full_msg = self.get_message(msg["id"])
                if full_msg:
                    messages.append(full_msg)

            page_token = response.get("nextPageToken")
            if not page_token:
                break

        return messages[:max_results]

    def get_unread_count(self, label_id: str = "INBOX") -> int:
        """Get count of unread messages in a label."""
        try:
            label = (
                self.service.users()
                .labels()
                .get(userId="me", id=label_id)
                .execute()
            )
            return label.get("messagesUnread", 0)
        except HttpError as e:
            logger.error(f"Error getting unread count: {e}")
            return 0

    def list_labels(self) -> list[dict]:
        """List all labels."""
        try:
            response = self.service.users().labels().list(userId="me").execute()
            return response.get("labels", [])
        except HttpError as e:
            logger.error(f"Error listing labels: {e}")
            return []

    def _build_message(
        self,
        to: str,
        subject: str,
        body: str,
        cc: str | None = None,
        bcc: str | None = None,
    ) -> str:
        """Build an email message and return base64-encoded raw format.

        Args:
            to: Recipient email address.
            subject: Email subject.
            body: Email body (plain text).
            cc: CC recipients (comma-separated).
            bcc: BCC recipients (comma-separated).

        Returns:
            Base64 URL-safe encoded message string.
        """
        message = MIMEText(body)
        message["to"] = to
        message["subject"] = subject
        if cc:
            message["cc"] = cc
        if bcc:
            message["bcc"] = bcc

        return base64.urlsafe_b64encode(message.as_bytes()).decode()

    def create_draft(
        self,
        to: str,
        subject: str,
        body: str,
        cc: str | None = None,
        bcc: str | None = None,
        reply_to_message_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a draft email (does NOT send).

        Args:
            to: Recipient email address.
            subject: Email subject.
            body: Email body (plain text).
            cc: CC recipients (comma-separated).
            bcc: BCC recipients (comma-separated).
            reply_to_message_id: Message ID to reply to.

        Returns:
            Created draft data.
        """
        raw = self._build_message(to, subject, body, cc, bcc)
        draft_body = {"message": {"raw": raw}}

        if reply_to_message_id:
            # Get the original message to get thread ID
            original = self.get_message(reply_to_message_id, format="metadata")
            if original:
                draft_body["message"]["threadId"] = original.get("threadId")

        try:
            return (
                self.service.users()
                .drafts()
                .create(userId="me", body=draft_body)
                .execute()
            )
        except HttpError as e:
            logger.error(f"Error creating draft: {e}")
            raise

    def send_message(
        self,
        to: str,
        subject: str,
        body: str,
        cc: str | None = None,
        bcc: str | None = None,
        reply_to_message_id: str | None = None,
    ) -> dict[str, Any]:
        """Send an email.

        Args:
            to: Recipient email address.
            subject: Email subject.
            body: Email body (plain text).
            cc: CC recipients (comma-separated).
            bcc: BCC recipients (comma-separated).
            reply_to_message_id: Message ID to reply to (will add to same thread).

        Returns:
            Sent message data including id and threadId.
        """
        raw = self._build_message(to, subject, body, cc, bcc)
        message_body: dict[str, Any] = {"raw": raw}

        if reply_to_message_id:
            # Get the original message to get thread ID
            original = self.get_message(reply_to_message_id, format="metadata")
            if original:
                message_body["threadId"] = original.get("threadId")

        try:
            return (
                self.service.users()
                .messages()
                .send(userId="me", body=message_body)
                .execute()
            )
        except HttpError as e:
            logger.error(f"Error sending message: {e}")
            raise

    def list_history(
        self,
        start_history_id: str,
        history_types: list[str] | None = None,
        max_results: int = 100,
    ) -> dict[str, Any]:
        """List history changes since a given history ID.

        Useful for incremental sync.

        Args:
            start_history_id: History ID to start from.
            history_types: Types of changes to include
                          (e.g., ["messageAdded", "messageDeleted"]).
            max_results: Maximum number of history records.

        Returns:
            History response with changes.
        """
        try:
            params = {
                "userId": "me",
                "startHistoryId": start_history_id,
                "maxResults": max_results,
            }
            if history_types:
                params["historyTypes"] = history_types

            return self.service.users().history().list(**params).execute()
        except HttpError as e:
            logger.error(f"Error listing history: {e}")
            raise

    @staticmethod
    def parse_message(message: dict) -> dict[str, Any]:
        """Parse a Gmail message into a structured format.

        Args:
            message: Raw message from Gmail API.

        Returns:
            Parsed message with headers, body, and attachments info.
        """
        result = {
            "id": message["id"],
            "thread_id": message.get("threadId"),
            "label_ids": message.get("labelIds", []),
            "snippet": message.get("snippet", ""),
        }

        # Parse internal date
        if "internalDate" in message:
            timestamp = int(message["internalDate"]) / 1000
            result["timestamp"] = datetime.fromtimestamp(timestamp)

        # Parse headers
        headers = {}
        if "payload" in message and "headers" in message["payload"]:
            for header in message["payload"]["headers"]:
                name = header["name"].lower()
                headers[name] = header["value"]

        result["subject"] = headers.get("subject", "(no subject)")
        result["from"] = headers.get("from", "")
        result["to"] = headers.get("to", "")
        result["cc"] = headers.get("cc", "")
        result["date"] = headers.get("date", "")
        result["message_id"] = headers.get("message-id", "")

        # Parse body
        result["body"] = GmailClient._extract_body(message.get("payload", {}))

        # Check for attachments
        result["has_attachments"] = GmailClient._has_attachments(
            message.get("payload", {})
        )

        return result

    @staticmethod
    def _extract_body(payload: dict) -> str:
        """Extract body text from message payload."""
        if "body" in payload and payload["body"].get("data"):
            return base64.urlsafe_b64decode(payload["body"]["data"]).decode(
                "utf-8", errors="ignore"
            )

        if "parts" in payload:
            for part in payload["parts"]:
                mime_type = part.get("mimeType", "")
                if mime_type == "text/plain":
                    if "body" in part and part["body"].get("data"):
                        return base64.urlsafe_b64decode(part["body"]["data"]).decode(
                            "utf-8", errors="ignore"
                        )
                elif mime_type == "text/html":
                    # Fall back to HTML if no plain text
                    if "body" in part and part["body"].get("data"):
                        html = base64.urlsafe_b64decode(part["body"]["data"]).decode(
                            "utf-8", errors="ignore"
                        )
                        # Basic HTML stripping (consider using BeautifulSoup for better results)
                        import re
                        text = re.sub(r"<[^>]+>", "", html)
                        return text
                elif "parts" in part:
                    # Recursive for multipart messages
                    body = GmailClient._extract_body(part)
                    if body:
                        return body

        return ""

    @staticmethod
    def _has_attachments(payload: dict) -> bool:
        """Check if message has attachments."""
        if "parts" in payload:
            for part in payload["parts"]:
                if part.get("filename"):
                    return True
                if "parts" in part and GmailClient._has_attachments(part):
                    return True
        return False
