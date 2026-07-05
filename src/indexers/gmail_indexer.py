"""Gmail content indexer."""

import logging
import re
from datetime import UTC, datetime
from typing import Any

from googleapiclient.errors import HttpError

from ..config import (
    GMAIL_STALE_HISTORY_FULL_SYNC_LIMIT,
    GOOGLE_ACCOUNTS,
    SYNC_BATCH_SIZE,
)
from ..integrations.gmail import GmailClient
from ..knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


class GmailIndexer:
    """Indexer for Gmail messages across multiple accounts."""

    def __init__(self, kg: KnowledgeGraph | None = None):
        """Initialize the Gmail indexer.

        Args:
            kg: Knowledge graph instance. Creates new one if not provided.
        """
        self.kg = kg or KnowledgeGraph()

    def index_all(
        self,
        account: str,
        max_messages: int = 10000,
        labels: list[str] | None = None,
    ) -> dict[str, Any]:
        """Index all messages for an account.

        Args:
            account: Google account identifier.
            max_messages: Maximum number of messages to index.
            labels: Only index messages with these labels.

        Returns:
            Statistics about the indexing operation.
        """
        if account not in GOOGLE_ACCOUNTS:
            raise ValueError(f"Unknown account: {account}")

        logger.info(f"Starting full Gmail index for account '{account}'")

        client = GmailClient(account)
        stats = {
            "messages_processed": 0,
            "messages_indexed": 0,
            "people_extracted": 0,
            "errors": 0,
        }

        # Build query
        query_parts = []
        if labels:
            for label in labels:
                query_parts.append(f"label:{label}")

        query = " OR ".join(query_parts) if query_parts else ""
        page_token = None

        while stats["messages_processed"] < max_messages:
            try:
                response = client.list_messages(
                    query=query,
                    max_results=min(SYNC_BATCH_SIZE, max_messages - stats["messages_processed"]),
                    page_token=page_token,
                )

                if "messages" not in response:
                    break

                for msg_stub in response["messages"]:
                    try:
                        message = client.get_message(msg_stub["id"])
                        if message:
                            self._index_message(account, message, stats)
                            stats["messages_processed"] += 1
                    except Exception as e:
                        logger.error(f"Error indexing message {msg_stub['id']}: {e}")
                        stats["errors"] += 1

                page_token = response.get("nextPageToken")
                if not page_token:
                    break

                logger.info(f"Processed {stats['messages_processed']} messages...")

            except Exception as e:
                logger.error(f"Error listing messages: {e}")
                stats["errors"] += 1
                break

        # Update sync state
        profile = client.get_profile()
        history_id = profile.get("historyId")

        self.kg.set_last_sync(
            source="gmail",
            account=account,
            last_sync=datetime.now(UTC),
            sync_token=history_id,
            metadata={"type": "full", "stats": stats},
        )

        logger.info(f"Gmail indexing complete for '{account}': {stats}")
        return stats

    def index_delta(
        self,
        account: str,
        since: datetime | None = None,
    ) -> dict[str, Any]:
        """Index new/changed messages since last sync.

        Args:
            account: Google account identifier.
            since: Only index messages since this time. Uses last sync if not provided.

        Returns:
            Statistics about the indexing operation.
        """
        if account not in GOOGLE_ACCOUNTS:
            raise ValueError(f"Unknown account: {account}")

        logger.info(f"Starting delta Gmail sync for account '{account}'")

        client = GmailClient(account)
        stats = {
            "messages_added": 0,
            "messages_deleted": 0,
            "messages_indexed": 0,
            "people_extracted": 0,
            "errors": 0,
        }

        # Get last sync state
        sync_state = self.kg.get_last_sync("gmail", account)

        if not sync_state or not sync_state.get("last_sync_token"):
            logger.warning(f"No previous sync state for '{account}', doing full sync")
            return self.index_all(account)

        start_history_id = sync_state["last_sync_token"]

        try:
            try:
                response = client.list_history(
                    start_history_id=start_history_id,
                    history_types=["messageAdded", "messageDeleted"],
                )
            except Exception as e:
                if _is_stale_history_error(e):
                    logger.warning(
                        "Gmail history token for '%s' is stale; running bounded full sync "
                        "for %s messages",
                        account,
                        GMAIL_STALE_HISTORY_FULL_SYNC_LIMIT,
                    )
                    recovered_stats = self.index_all(
                        account,
                        max_messages=GMAIL_STALE_HISTORY_FULL_SYNC_LIMIT,
                    )
                    recovered_stats["recovered_from_stale_history"] = True
                    recovered_stats["stale_history_id"] = start_history_id
                    return recovered_stats
                raise

            # Process history records
            for history in response.get("history", []):
                # Handle added messages
                for added in history.get("messagesAdded", []):
                    msg_id = added["message"]["id"]
                    try:
                        message = client.get_message(msg_id)
                        if message:
                            self._index_message(account, message, stats)
                            stats["messages_added"] += 1
                    except Exception as e:
                        logger.error(f"Error indexing added message {msg_id}: {e}")
                        stats["errors"] += 1

                # Handle deleted messages
                for deleted in history.get("messagesDeleted", []):
                    msg_id = deleted["message"]["id"]
                    content_id = f"gmail:{account}:{msg_id}"
                    if self.kg.delete_content(content_id):
                        stats["messages_deleted"] += 1

            # Update sync state with new history ID
            new_history_id = response.get("historyId", start_history_id)
            self.kg.set_last_sync(
                source="gmail",
                account=account,
                last_sync=datetime.now(UTC),
                sync_token=new_history_id,
                metadata={"type": "delta", "stats": stats},
            )

        except Exception as e:
            logger.error(f"Error in delta sync: {e}")
            stats["errors"] += 1

        logger.info(f"Gmail delta sync complete for '{account}': {stats}")
        return stats

    def _index_message(
        self,
        account: str,
        message: dict,
        stats: dict[str, int],
    ) -> None:
        """Index a single message and extract entities."""
        parsed = GmailClient.parse_message(message)

        # Create content ID
        content_id = f"gmail:{account}:{parsed['id']}"

        # Extract and index people
        people = self._extract_people(parsed)
        for person in people:
            person_id = f"person:{person['email']}"
            is_new = self.kg.upsert_entity(
                entity_id=person_id,
                entity_type="person",
                name=person["name"] or person["email"],
                email=person["email"],
                source="gmail",
                source_account=account,
            )
            if is_new:
                stats["people_extracted"] += 1

            # Add relationship between message and person
            self.kg.add_relationship(
                from_id=content_id,
                from_type="email",
                to_id=person_id,
                to_type="person",
                relation=person["relation"],
            )

        # Index the message content
        body_preview = parsed["body"][:5000] if parsed["body"] else ""

        is_new = self.kg.upsert_content(
            content_id=content_id,
            content_type="email",
            source="gmail",
            source_account=account,
            title=parsed["subject"],
            body=body_preview,
            source_id=parsed["id"],
            timestamp=parsed.get("timestamp"),
            metadata={
                "thread_id": parsed["thread_id"],
                "from": parsed["from"],
                "to": parsed["to"],
                "labels": parsed["label_ids"],
                "has_attachments": parsed["has_attachments"],
            },
        )

        if is_new:
            stats["messages_indexed"] += 1

    def _extract_people(self, message: dict) -> list[dict[str, Any]]:
        """Extract people (email addresses) from a message."""
        people = []

        # Parse email addresses from From, To, CC fields
        for field, relation in [("from", "sender"), ("to", "recipient"), ("cc", "cc")]:
            value = message.get(field, "")
            if value:
                extracted = self._parse_email_addresses(value)
                for email, name in extracted:
                    people.append({
                        "email": email.lower(),
                        "name": name,
                        "relation": relation,
                    })

        return people

    def _parse_email_addresses(self, text: str) -> list[tuple[str, str | None]]:
        """Parse email addresses from a header value.

        Handles formats like:
        - email@example.com
        - Name <email@example.com>
        - "Name" <email@example.com>
        """
        results = []

        # Pattern for "Name" <email> or Name <email>
        pattern = r'(?:"?([^"<>]*)"?\s*)?<([^<>]+@[^<>]+)>'
        matches = re.findall(pattern, text)

        if matches:
            for name, email in matches:
                results.append((email.strip(), name.strip() if name else None))
        else:
            # Try plain email addresses
            email_pattern = r'[\w\.-]+@[\w\.-]+'
            emails = re.findall(email_pattern, text)
            for email in emails:
                results.append((email, None))

        return results


def _is_stale_history_error(error: Exception) -> bool:
    """Return true when Gmail history.list cannot continue from a saved token."""
    if isinstance(error, HttpError):
        status = getattr(getattr(error, "resp", None), "status", None)
        if status == 404:
            return True

    message = str(error).lower()
    return (
        "starthistoryid" in message
        and (
            "requested entity was not found" in message
            or "not found" in message
            or "history" in message
        )
    )
