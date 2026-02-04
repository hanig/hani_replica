"""Google Docs/Drive comments API client."""

import logging
from typing import Any

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from .google_auth import get_credentials

logger = logging.getLogger(__name__)


class DocsClient:
    """Client for interacting with Google Docs and Drive comments.

    Note: Comments are managed via the Drive API, not the Docs API.
    The Docs API is used for reading document content.
    """

    def __init__(self, account: str):
        """Initialize Docs client for a specific account.

        Args:
            account: Account identifier (e.g., "arc", "personal").
        """
        self.account = account
        self._drive_service = None
        self._docs_service = None

    @property
    def drive_service(self):
        """Lazily initialize the Drive service (for comments)."""
        if self._drive_service is None:
            creds = get_credentials(self.account)
            if not creds:
                raise RuntimeError(f"No valid credentials for account '{self.account}'")
            self._drive_service = build("drive", "v3", credentials=creds)
        return self._drive_service

    @property
    def docs_service(self):
        """Lazily initialize the Docs service (for document content)."""
        if self._docs_service is None:
            creds = get_credentials(self.account)
            if not creds:
                raise RuntimeError(f"No valid credentials for account '{self.account}'")
            self._docs_service = build("docs", "v1", credentials=creds)
        return self._docs_service

    def get_document(self, document_id: str) -> dict[str, Any]:
        """Get a Google Doc by ID.

        Args:
            document_id: The document ID (from the URL).

        Returns:
            Document data including title and content structure.
        """
        try:
            return self.docs_service.documents().get(documentId=document_id).execute()
        except HttpError as e:
            logger.error(f"Error getting document {document_id}: {e}")
            raise

    def add_comment(
        self,
        document_id: str,
        content: str,
        quoted_text: str | None = None,
    ) -> dict[str, Any]:
        """Add a comment to a Google Doc.

        Args:
            document_id: The document ID.
            content: The comment text.
            quoted_text: Optional text from the document to anchor the comment to.
                        If provided, the comment will be attached to this text.

        Returns:
            Created comment data.
        """
        comment_body: dict[str, Any] = {"content": content}

        if quoted_text:
            comment_body["quotedFileContent"] = {
                "value": quoted_text,
            }

        try:
            return (
                self.drive_service.comments()
                .create(
                    fileId=document_id,
                    body=comment_body,
                    fields="id,content,author,createdTime,quotedFileContent,resolved",
                )
                .execute()
            )
        except HttpError as e:
            logger.error(f"Error adding comment to {document_id}: {e}")
            raise

    def reply_to_comment(
        self,
        document_id: str,
        comment_id: str,
        content: str,
    ) -> dict[str, Any]:
        """Reply to an existing comment.

        Args:
            document_id: The document ID.
            comment_id: The comment ID to reply to.
            content: The reply text.

        Returns:
            Created reply data.
        """
        try:
            return (
                self.drive_service.replies()
                .create(
                    fileId=document_id,
                    commentId=comment_id,
                    body={"content": content},
                    fields="id,content,author,createdTime",
                )
                .execute()
            )
        except HttpError as e:
            logger.error(f"Error replying to comment {comment_id}: {e}")
            raise

    def list_comments(
        self,
        document_id: str,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        """List all comments on a document.

        Args:
            document_id: The document ID.
            include_deleted: Whether to include deleted comments.

        Returns:
            List of comments with their replies.
        """
        comments = []
        page_token = None

        try:
            while True:
                params: dict[str, Any] = {
                    "fileId": document_id,
                    "fields": "comments(id,content,author,createdTime,modifiedTime,"
                    "quotedFileContent,resolved,replies(id,content,author,createdTime))",
                    "includeDeleted": include_deleted,
                }
                if page_token:
                    params["pageToken"] = page_token

                response = self.drive_service.comments().list(**params).execute()
                comments.extend(response.get("comments", []))

                page_token = response.get("nextPageToken")
                if not page_token:
                    break

            return comments
        except HttpError as e:
            logger.error(f"Error listing comments for {document_id}: {e}")
            raise

    def resolve_comment(
        self,
        document_id: str,
        comment_id: str,
    ) -> dict[str, Any]:
        """Resolve (close) a comment.

        Args:
            document_id: The document ID.
            comment_id: The comment ID to resolve.

        Returns:
            Updated comment data.
        """
        try:
            return (
                self.drive_service.comments()
                .update(
                    fileId=document_id,
                    commentId=comment_id,
                    body={"resolved": True},
                    fields="id,content,resolved",
                )
                .execute()
            )
        except HttpError as e:
            logger.error(f"Error resolving comment {comment_id}: {e}")
            raise

    def reopen_comment(
        self,
        document_id: str,
        comment_id: str,
    ) -> dict[str, Any]:
        """Reopen a resolved comment.

        Args:
            document_id: The document ID.
            comment_id: The comment ID to reopen.

        Returns:
            Updated comment data.
        """
        try:
            return (
                self.drive_service.comments()
                .update(
                    fileId=document_id,
                    commentId=comment_id,
                    body={"resolved": False},
                    fields="id,content,resolved",
                )
                .execute()
            )
        except HttpError as e:
            logger.error(f"Error reopening comment {comment_id}: {e}")
            raise

    def delete_comment(
        self,
        document_id: str,
        comment_id: str,
    ) -> None:
        """Delete a comment.

        Args:
            document_id: The document ID.
            comment_id: The comment ID to delete.
        """
        try:
            self.drive_service.comments().delete(
                fileId=document_id,
                commentId=comment_id,
            ).execute()
        except HttpError as e:
            logger.error(f"Error deleting comment {comment_id}: {e}")
            raise
