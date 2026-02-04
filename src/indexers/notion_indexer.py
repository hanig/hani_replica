"""Notion content indexer."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from ..config import NOTION_WORKSPACE
from ..integrations.notion_client import NotionClient
from ..knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


class NotionIndexer:
    """Indexer for Notion pages, databases, and content."""

    def __init__(self, kg: KnowledgeGraph | None = None):
        """Initialize the Notion indexer.

        Args:
            kg: Knowledge graph instance. Creates new one if not provided.
        """
        self.kg = kg or KnowledgeGraph()
        self.workspace = NOTION_WORKSPACE or "default"

    def index_all(
        self,
        include_databases: bool = True,
        include_pages: bool = True,
        include_users: bool = True,
        include_content: bool = True,
        max_pages: int = 1000,
    ) -> dict[str, Any]:
        """Index all Notion content.

        Args:
            include_databases: Index database schemas.
            include_pages: Index pages and database records.
            include_users: Index workspace users.
            include_content: Index full page content (blocks).
            max_pages: Maximum number of pages to index.

        Returns:
            Statistics about the indexing operation.
        """
        logger.info("Starting full Notion index")

        try:
            client = NotionClient()
        except ValueError as e:
            logger.error(f"Cannot initialize Notion client: {e}")
            return {"error": str(e)}

        stats = {
            "databases_indexed": 0,
            "pages_indexed": 0,
            "users_indexed": 0,
            "comments_indexed": 0,
            "errors": 0,
        }

        try:
            # Index users first (for relationship extraction)
            if include_users:
                users = client.list_users()
                for user in users:
                    self._index_user(user, stats)
                logger.info(f"Indexed {stats['users_indexed']} users")

            # Index databases
            if include_databases:
                databases = client.list_databases()
                for db in databases:
                    self._index_database(db, stats)

                    # Index pages in this database
                    if include_pages:
                        try:
                            pages = client.query_database(
                                db["id"],
                                max_results=min(100, max_pages - stats["pages_indexed"]),
                            )
                            for page in pages:
                                self._index_page(
                                    client, page, stats,
                                    include_content=include_content,
                                    database_title=db["title"],
                                )
                                if stats["pages_indexed"] >= max_pages:
                                    break
                        except Exception as e:
                            logger.warning(f"Error querying database {db['id']}: {e}")
                            stats["errors"] += 1

                logger.info(f"Indexed {stats['databases_indexed']} databases")

            # Search for standalone pages (not in databases)
            if include_pages and stats["pages_indexed"] < max_pages:
                # Search with empty query to get all pages
                search_results = client.search(
                    query="",
                    filter_type="page",
                    max_results=max_pages - stats["pages_indexed"],
                )
                for result in search_results:
                    # Skip pages we've already indexed (from databases)
                    page_id = result["id"]
                    content_id = f"notion:page:{page_id}"
                    if self.kg.get_content(content_id) is not None:
                        continue

                    self._index_page(
                        client, result, stats,
                        include_content=include_content,
                    )

                logger.info(f"Indexed {stats['pages_indexed']} total pages")

        except Exception as e:
            logger.error(f"Error in Notion indexing: {e}")
            stats["errors"] += 1

        self.kg.set_last_sync(
            source="notion",
            account=self.workspace,
            last_sync=datetime.now(timezone.utc),
            metadata={"type": "full", "stats": stats},
        )

        logger.info(f"Notion indexing complete: {stats}")
        return stats

    def index_delta(self, hours_back: int = 24) -> dict[str, Any]:
        """Index recently modified Notion content.

        Args:
            hours_back: Number of hours to look back.

        Returns:
            Statistics about the indexing operation.
        """
        logger.info(f"Starting delta Notion sync (last {hours_back} hours)")

        try:
            client = NotionClient()
        except ValueError as e:
            logger.error(f"Cannot initialize Notion client: {e}")
            return {"error": str(e)}

        stats = {
            "pages_updated": 0,
            "databases_updated": 0,
            "errors": 0,
        }

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        cutoff_str = cutoff.isoformat()

        try:
            # Search for recently modified pages
            # Note: Notion's search doesn't support date filtering directly,
            # so we search and filter client-side
            search_results = client.search(query="", max_results=500)

            for result in search_results:
                last_edited = result.get("last_edited_time")
                if last_edited:
                    # Parse ISO timestamp
                    edited_dt = datetime.fromisoformat(
                        last_edited.replace("Z", "+00:00")
                    )
                    if edited_dt >= cutoff:
                        if result.get("object") == "database":
                            self._index_database(result, stats)
                            stats["databases_updated"] += 1
                        else:
                            self._index_page(client, result, stats, include_content=True)
                            stats["pages_updated"] += 1

        except Exception as e:
            logger.error(f"Error in Notion delta sync: {e}")
            stats["errors"] += 1

        self.kg.set_last_sync(
            source="notion",
            account=self.workspace,
            last_sync=datetime.now(timezone.utc),
            metadata={"type": "delta", "stats": stats},
        )

        logger.info(f"Notion delta sync complete: {stats}")
        return stats

    def _index_database(self, db: dict, stats: dict[str, int]) -> None:
        """Index a database schema."""
        content_id = f"notion:database:{db['id']}"

        # Build description from schema
        schema_desc = []
        for prop_name, prop_info in db.get("schema", {}).items():
            prop_type = prop_info.get("type", "unknown")
            schema_desc.append(f"- {prop_name} ({prop_type})")

        body = db.get("description", "")
        if schema_desc:
            body += "\n\nProperties:\n" + "\n".join(schema_desc)

        self.kg.upsert_content(
            content_id=content_id,
            content_type="database",
            source="notion",
            source_account=self.workspace,
            title=db.get("title") or "Untitled Database",
            body=body,
            source_id=db["id"],
            url=db.get("url"),
            timestamp=self._parse_timestamp(db.get("last_edited_time")),
            metadata={
                "schema": db.get("schema"),
                "is_inline": db.get("is_inline", False),
            },
        )

        stats["databases_indexed"] += 1

    def _index_page(
        self,
        client: NotionClient,
        page: dict,
        stats: dict[str, int],
        include_content: bool = True,
        database_title: str | None = None,
    ) -> None:
        """Index a page and its content."""
        page_id = page["id"]
        content_id = f"notion:page:{page_id}"

        # Build the body from properties and content
        body_parts = []

        # Add properties (excluding title)
        properties = page.get("properties", {})
        for prop_name, prop_value in properties.items():
            if prop_value and prop_name.lower() != "name" and prop_name.lower() != "title":
                if isinstance(prop_value, list):
                    prop_str = ", ".join(str(v) for v in prop_value)
                elif isinstance(prop_value, dict):
                    prop_str = str(prop_value)
                else:
                    prop_str = str(prop_value)
                if prop_str and prop_str != "None":
                    body_parts.append(f"{prop_name}: {prop_str}")

        # Fetch and add page content (blocks)
        if include_content:
            try:
                blocks = client.get_page_content(page_id)
                content_text = client.blocks_to_text(blocks)
                if content_text:
                    body_parts.append("\n" + content_text)
            except Exception as e:
                logger.debug(f"Could not fetch content for page {page_id}: {e}")

        body = "\n".join(body_parts)

        # Build title
        title = page.get("title") or "Untitled"
        if database_title:
            title = f"[{database_title}] {title}"

        # Extract creator as person
        created_by = page.get("created_by")
        if created_by:
            self._extract_user_relationship(content_id, created_by, "author", stats)

        last_edited_by = page.get("last_edited_by")
        if last_edited_by and last_edited_by != created_by:
            self._extract_user_relationship(content_id, last_edited_by, "editor", stats)

        self.kg.upsert_content(
            content_id=content_id,
            content_type="page",
            source="notion",
            source_account=self.workspace,
            title=title,
            body=body,
            source_id=page_id,
            url=page.get("url"),
            timestamp=self._parse_timestamp(page.get("created_time")),
            metadata={
                "parent_type": page.get("parent_type"),
                "parent_id": page.get("parent_id"),
                "archived": page.get("archived", False),
                "last_edited_time": page.get("last_edited_time"),
            },
        )

        stats["pages_indexed"] += 1

        # Index comments on the page
        try:
            comments = client.list_comments(page_id)
            for comment in comments:
                self._index_comment(comment, page_id, title, stats)
        except Exception as e:
            logger.debug(f"Could not fetch comments for page {page_id}: {e}")

    def _index_comment(
        self,
        comment: dict,
        page_id: str,
        page_title: str,
        stats: dict[str, int],
    ) -> None:
        """Index a comment."""
        comment_id = comment["id"]
        content_id = f"notion:comment:{comment_id}"

        # Extract author
        created_by = comment.get("created_by")
        if created_by:
            self._extract_user_relationship(content_id, created_by, "author", stats)

        self.kg.upsert_content(
            content_id=content_id,
            content_type="comment",
            source="notion",
            source_account=self.workspace,
            title=f"Comment on: {page_title}",
            body=comment.get("text", ""),
            source_id=comment_id,
            url=None,  # Comments don't have direct URLs
            timestamp=self._parse_timestamp(comment.get("created_time")),
            metadata={
                "page_id": page_id,
            },
        )

        # Link comment to page
        self.kg.add_relationship(
            from_id=content_id,
            from_type="comment",
            to_id=f"notion:page:{page_id}",
            to_type="page",
            relation="comment_on",
        )

        stats["comments_indexed"] += 1

    def _index_user(self, user: dict, stats: dict[str, int]) -> None:
        """Index a Notion user as a person entity."""
        user_id = user["id"]
        person_id = f"person:notion:{user_id}"

        # Get user type and additional info
        user_type = user.get("type", "person")
        name = user.get("name") or "Unknown User"

        # For person users, try to get email
        email = None
        if user_type == "person" and user.get("person"):
            email = user["person"].get("email")

        is_new = self.kg.upsert_entity(
            entity_id=person_id,
            entity_type="person",
            name=name,
            source="notion",
            source_account=self.workspace,
            metadata={
                "notion_user_id": user_id,
                "user_type": user_type,
                "email": email,
                "avatar_url": user.get("avatar_url"),
            },
        )

        if is_new:
            stats["users_indexed"] += 1

    def _extract_user_relationship(
        self,
        content_id: str,
        user_id: str,
        relation: str,
        stats: dict[str, int],
    ) -> None:
        """Create a relationship between content and a user."""
        person_id = f"person:notion:{user_id}"

        # Ensure the user entity exists (minimal entry if not indexed)
        self.kg.upsert_entity(
            entity_id=person_id,
            entity_type="person",
            name=f"Notion User {user_id[:8]}",
            source="notion",
            source_account=self.workspace,
            metadata={"notion_user_id": user_id},
        )

        self.kg.add_relationship(
            from_id=content_id,
            from_type="page",
            to_id=person_id,
            to_type="person",
            relation=relation,
        )

    def _parse_timestamp(self, timestamp_str: str | None) -> datetime | None:
        """Parse an ISO timestamp string."""
        if not timestamp_str:
            return None
        try:
            return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None
