"""Google Drive content indexer."""

import logging
from datetime import UTC, datetime
from typing import Any

from ..config import GOOGLE_ACCOUNTS, SYNC_BATCH_SIZE
from ..integrations.gdrive import DriveClient
from ..knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


# MIME types to index
INDEXABLE_MIME_TYPES = {
    "application/vnd.google-apps.document",
    "application/vnd.google-apps.spreadsheet",
    "application/vnd.google-apps.presentation",
    "text/plain",
    "text/html",
    "text/csv",
    "text/markdown",
    "application/json",
}


class DriveIndexer:
    """Indexer for Google Drive files across multiple accounts."""

    def __init__(self, kg: KnowledgeGraph | None = None):
        """Initialize the Drive indexer.

        Args:
            kg: Knowledge graph instance. Creates new one if not provided.
        """
        self.kg = kg or KnowledgeGraph()

    def index_all(
        self,
        account: str,
        max_files: int = 5000,
        include_content: bool = True,
    ) -> dict[str, Any]:
        """Index all files for an account.

        Args:
            account: Google account identifier.
            max_files: Maximum number of files to index.
            include_content: Whether to fetch and index file content.

        Returns:
            Statistics about the indexing operation.
        """
        if account not in GOOGLE_ACCOUNTS:
            raise ValueError(f"Unknown account: {account}")

        logger.info(f"Starting full Drive index for account '{account}'")

        client = DriveClient(account)
        stats = {
            "files_processed": 0,
            "files_indexed": 0,
            "content_indexed": 0,
            "people_extracted": 0,
            "errors": 0,
        }

        # Query for non-trashed files
        query = "trashed=false"
        page_token = None

        while stats["files_processed"] < max_files:
            try:
                response = client.list_files(
                    query=query,
                    page_size=min(SYNC_BATCH_SIZE, max_files - stats["files_processed"]),
                    page_token=page_token,
                )

                for file in response.get("files", []):
                    try:
                        self._index_file(account, client, file, include_content, stats)
                        stats["files_processed"] += 1
                    except Exception as e:
                        logger.error(f"Error indexing file {file.get('id')}: {e}")
                        stats["errors"] += 1

                page_token = response.get("nextPageToken")
                if not page_token:
                    break

                logger.info(f"Processed {stats['files_processed']} files...")

            except Exception as e:
                logger.error(f"Error listing files: {e}")
                stats["errors"] += 1
                break

        # Get page token for delta sync
        start_page_token = client.get_start_page_token()

        self.kg.set_last_sync(
            source="drive",
            account=account,
            last_sync=datetime.now(UTC),
            sync_token=start_page_token,
            metadata={"type": "full", "stats": stats},
        )

        logger.info(f"Drive indexing complete for '{account}': {stats}")
        return stats

    def index_delta(
        self,
        account: str,
        include_content: bool = True,
    ) -> dict[str, Any]:
        """Index changed files since last sync.

        Args:
            account: Google account identifier.
            include_content: Whether to fetch and index file content.

        Returns:
            Statistics about the indexing operation.
        """
        if account not in GOOGLE_ACCOUNTS:
            raise ValueError(f"Unknown account: {account}")

        logger.info(f"Starting delta Drive sync for account '{account}'")

        client = DriveClient(account)
        stats = {
            "files_added": 0,
            "files_updated": 0,
            "files_deleted": 0,
            "files_indexed": 0,
            "content_indexed": 0,
            "people_extracted": 0,
            "errors": 0,
        }

        # Get last sync state
        sync_state = self.kg.get_last_sync("drive", account)

        if not sync_state or not sync_state.get("last_sync_token"):
            logger.warning(f"No previous sync state for '{account}', doing full sync")
            return self.index_all(account, include_content=include_content)

        page_token = sync_state["last_sync_token"]

        try:
            while True:
                response = client.list_recent_changes(
                    page_token=page_token,
                    page_size=SYNC_BATCH_SIZE,
                )

                for change in response.get("changes", []):
                    file_id = change.get("fileId")

                    if change.get("removed"):
                        # File was deleted or access was revoked
                        content_id = f"drive:{account}:{file_id}"
                        if self.kg.delete_content(content_id):
                            stats["files_deleted"] += 1
                    else:
                        # File was added or modified
                        file = change.get("file")
                        if file:
                            try:
                                is_new = self._index_file(
                                    account, client, file, include_content, stats
                                )
                                if is_new:
                                    stats["files_added"] += 1
                                else:
                                    stats["files_updated"] += 1
                            except Exception as e:
                                logger.error(f"Error indexing file {file_id}: {e}")
                                stats["errors"] += 1

                # Get next page token
                page_token = response.get("nextPageToken")
                if not page_token:
                    # Save the new start page token for next sync
                    new_start_token = response.get("newStartPageToken")
                    if new_start_token:
                        self.kg.set_last_sync(
                            source="drive",
                            account=account,
                            last_sync=datetime.now(UTC),
                            sync_token=new_start_token,
                            metadata={"type": "delta", "stats": stats},
                        )
                    break

        except Exception as e:
            logger.error(f"Error in delta sync: {e}")
            stats["errors"] += 1

        logger.info(f"Drive delta sync complete for '{account}': {stats}")
        return stats

    def _index_file(
        self,
        account: str,
        client: DriveClient,
        file: dict,
        include_content: bool,
        stats: dict[str, int],
    ) -> bool:
        """Index a single file and extract entities.

        Returns:
            True if file was newly indexed, False if updated.
        """
        parsed = DriveClient.parse_file(file)
        content_id = f"drive:{account}:{parsed['id']}"

        # Extract and index owners as people
        for owner in parsed.get("owners", []):
            if owner.get("email"):
                person_id = f"person:{owner['email']}"
                is_new = self.kg.upsert_entity(
                    entity_id=person_id,
                    entity_type="person",
                    name=owner.get("name") or owner["email"],
                    email=owner["email"],
                    source="drive",
                    source_account=account,
                )
                if is_new:
                    _increment_stat(stats, "people_extracted")

                # Add relationship
                self.kg.add_relationship(
                    from_id=content_id,
                    from_type="file",
                    to_id=person_id,
                    to_type="person",
                    relation="owner",
                )

        # Get file content if applicable
        body = None
        if include_content and parsed["mime_type"] in INDEXABLE_MIME_TYPES:
            body = client.get_file_content(parsed["id"])
            if body:
                # Truncate for storage
                body = body[:50000]
                _increment_stat(stats, "content_indexed")

        # Index the file metadata
        is_new = self.kg.upsert_content(
            content_id=content_id,
            content_type="file",
            source="drive",
            source_account=account,
            title=parsed["name"],
            body=body,
            source_id=parsed["id"],
            url=parsed["web_link"],
            timestamp=parsed.get("modified_time"),
            metadata={
                "mime_type": parsed["mime_type"],
                "size": parsed["size"],
                "parents": parsed["parents"],
            },
        )

        if is_new:
            _increment_stat(stats, "files_indexed")

        return is_new


def _increment_stat(stats: dict[str, int], key: str, amount: int = 1) -> None:
    """Increment an indexer stat, tolerating older/narrow stat dictionaries."""
    stats[key] = stats.get(key, 0) + amount
