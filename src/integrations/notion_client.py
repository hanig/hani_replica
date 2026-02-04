"""Notion API client."""

import logging
from datetime import datetime
from typing import Any

import httpx

from ..config import NOTION_API_KEY

logger = logging.getLogger(__name__)

# Notion API version header
NOTION_VERSION = "2022-06-28"
NOTION_BASE_URL = "https://api.notion.com/v1"


class NotionClient:
    """Client for interacting with Notion API."""

    def __init__(self, api_key: str | None = None):
        """Initialize Notion client.

        Args:
            api_key: Notion internal integration token.
        """
        self.api_key = api_key or NOTION_API_KEY

        if not self.api_key:
            raise ValueError("Notion API key is required")

        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        }

    def _request(
        self,
        method: str,
        endpoint: str,
        json: dict | None = None,
        params: dict | None = None,
    ) -> dict[str, Any]:
        """Make a request to the Notion API.

        Args:
            method: HTTP method.
            endpoint: API endpoint (without base URL).
            json: JSON body for POST/PATCH requests.
            params: Query parameters.

        Returns:
            Response JSON.

        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        url = f"{NOTION_BASE_URL}{endpoint}"

        with httpx.Client() as client:
            response = client.request(
                method=method,
                url=url,
                headers=self._headers,
                json=json,
                params=params,
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()

    def test_connection(self) -> dict[str, Any]:
        """Test the connection to Notion API.

        Returns:
            User information for the integration.
        """
        try:
            result = self._request("GET", "/users/me")
            logger.info(f"Connected to Notion as: {result.get('name', 'Unknown')}")
            return {
                "success": True,
                "bot_id": result.get("id"),
                "name": result.get("name"),
                "type": result.get("type"),
            }
        except httpx.HTTPStatusError as e:
            logger.error(f"Notion connection test failed: {e}")
            return {"success": False, "error": str(e)}

    # --- Read Operations ---

    def list_databases(self, max_results: int = 100) -> list[dict[str, Any]]:
        """List all databases the integration has access to.

        Args:
            max_results: Maximum number of databases to return.

        Returns:
            List of database metadata.
        """
        databases = []
        start_cursor = None

        while len(databases) < max_results:
            body = {
                "filter": {"property": "object", "value": "database"},
                "page_size": min(100, max_results - len(databases)),
            }
            if start_cursor:
                body["start_cursor"] = start_cursor

            result = self._request("POST", "/search", json=body)

            for db in result.get("results", []):
                databases.append(self._parse_database(db))

            if not result.get("has_more"):
                break
            start_cursor = result.get("next_cursor")

        return databases[:max_results]

    def get_database(self, database_id: str) -> dict[str, Any]:
        """Get a database by ID.

        Args:
            database_id: The database ID.

        Returns:
            Database metadata including schema.
        """
        result = self._request("GET", f"/databases/{database_id}")
        return self._parse_database(result)

    def query_database(
        self,
        database_id: str,
        filter: dict | None = None,
        sorts: list[dict] | None = None,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """Query a database for pages.

        Args:
            database_id: The database ID.
            filter: Notion filter object.
            sorts: List of sort objects.
            max_results: Maximum number of results.

        Returns:
            List of pages (database rows).
        """
        pages = []
        start_cursor = None

        while len(pages) < max_results:
            body: dict[str, Any] = {
                "page_size": min(100, max_results - len(pages)),
            }
            if filter:
                body["filter"] = filter
            if sorts:
                body["sorts"] = sorts
            if start_cursor:
                body["start_cursor"] = start_cursor

            result = self._request(
                "POST", f"/databases/{database_id}/query", json=body
            )

            for page in result.get("results", []):
                pages.append(self._parse_page(page))

            if not result.get("has_more"):
                break
            start_cursor = result.get("next_cursor")

        return pages[:max_results]

    def get_page(self, page_id: str) -> dict[str, Any]:
        """Get a page by ID.

        Args:
            page_id: The page ID.

        Returns:
            Page metadata (properties, not content).
        """
        result = self._request("GET", f"/pages/{page_id}")
        return self._parse_page(result)

    def get_page_content(self, page_id: str, max_blocks: int = 200) -> list[dict[str, Any]]:
        """Get the content blocks of a page.

        Args:
            page_id: The page ID.
            max_blocks: Maximum number of blocks to return.

        Returns:
            List of content blocks.
        """
        blocks = []
        start_cursor = None

        while len(blocks) < max_blocks:
            params: dict[str, Any] = {
                "page_size": min(100, max_blocks - len(blocks)),
            }
            if start_cursor:
                params["start_cursor"] = start_cursor

            result = self._request(
                "GET", f"/blocks/{page_id}/children", params=params
            )

            for block in result.get("results", []):
                parsed = self._parse_block(block)
                blocks.append(parsed)

                # Recursively fetch children if present
                if block.get("has_children") and len(blocks) < max_blocks:
                    children = self.get_page_content(block["id"], max_blocks=20)
                    parsed["children"] = children

            if not result.get("has_more"):
                break
            start_cursor = result.get("next_cursor")

        return blocks[:max_blocks]

    def search(
        self,
        query: str,
        filter_type: str | None = None,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """Search pages and databases.

        Args:
            query: Search query text.
            filter_type: Filter by "page" or "database".
            max_results: Maximum number of results.

        Returns:
            List of search results (pages and databases).
        """
        results = []
        start_cursor = None

        while len(results) < max_results:
            body: dict[str, Any] = {
                "query": query,
                "page_size": min(100, max_results - len(results)),
            }
            if filter_type:
                body["filter"] = {"property": "object", "value": filter_type}
            if start_cursor:
                body["start_cursor"] = start_cursor

            response = self._request("POST", "/search", json=body)

            for item in response.get("results", []):
                if item["object"] == "database":
                    results.append(self._parse_database(item))
                else:
                    results.append(self._parse_page(item))

            if not response.get("has_more"):
                break
            start_cursor = response.get("next_cursor")

        return results[:max_results]

    def list_users(self, max_results: int = 100) -> list[dict[str, Any]]:
        """List all users in the workspace.

        Args:
            max_results: Maximum number of users to return.

        Returns:
            List of user metadata.
        """
        users = []
        start_cursor = None

        while len(users) < max_results:
            params: dict[str, Any] = {
                "page_size": min(100, max_results - len(users)),
            }
            if start_cursor:
                params["start_cursor"] = start_cursor

            result = self._request("GET", "/users", params=params)

            for user in result.get("results", []):
                users.append(self._parse_user(user))

            if not result.get("has_more"):
                break
            start_cursor = result.get("next_cursor")

        return users[:max_results]

    def list_comments(self, page_id: str, max_results: int = 100) -> list[dict[str, Any]]:
        """List comments on a page or block.

        Args:
            page_id: The page or block ID.
            max_results: Maximum number of comments to return.

        Returns:
            List of comments.
        """
        comments = []
        start_cursor = None

        while len(comments) < max_results:
            params: dict[str, Any] = {
                "block_id": page_id,
                "page_size": min(100, max_results - len(comments)),
            }
            if start_cursor:
                params["start_cursor"] = start_cursor

            result = self._request("GET", "/comments", params=params)

            for comment in result.get("results", []):
                comments.append(self._parse_comment(comment))

            if not result.get("has_more"):
                break
            start_cursor = result.get("next_cursor")

        return comments[:max_results]

    # --- Write Operations ---

    def create_page(
        self,
        database_id: str,
        properties: dict[str, Any],
        content: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Create a new page in a database.

        Args:
            database_id: The parent database ID.
            properties: Page properties matching the database schema.
            content: Optional list of block objects for page content.

        Returns:
            Created page metadata.
        """
        body: dict[str, Any] = {
            "parent": {"database_id": database_id},
            "properties": properties,
        }
        if content:
            body["children"] = content

        result = self._request("POST", "/pages", json=body)
        return self._parse_page(result)

    def update_page(
        self,
        page_id: str,
        properties: dict[str, Any],
    ) -> dict[str, Any]:
        """Update page properties.

        Args:
            page_id: The page ID.
            properties: Properties to update.

        Returns:
            Updated page metadata.
        """
        result = self._request(
            "PATCH", f"/pages/{page_id}", json={"properties": properties}
        )
        return self._parse_page(result)

    def add_comment(self, page_id: str, content: str) -> dict[str, Any]:
        """Add a comment to a page.

        Args:
            page_id: The page ID.
            content: Comment text.

        Returns:
            Created comment metadata.
        """
        body = {
            "parent": {"page_id": page_id},
            "rich_text": [{"type": "text", "text": {"content": content}}],
        }
        result = self._request("POST", "/comments", json=body)
        return self._parse_comment(result)

    def append_blocks(
        self,
        page_id: str,
        blocks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Append blocks to a page.

        Args:
            page_id: The page ID.
            blocks: List of block objects to append.

        Returns:
            List of created blocks.
        """
        result = self._request(
            "PATCH",
            f"/blocks/{page_id}/children",
            json={"children": blocks},
        )
        return [self._parse_block(b) for b in result.get("results", [])]

    # --- Parsing Helpers ---

    def _parse_database(self, db: dict) -> dict[str, Any]:
        """Parse a database object."""
        title = self._extract_title(db.get("title", []))

        # Parse schema (properties)
        schema = {}
        for prop_name, prop_def in db.get("properties", {}).items():
            schema[prop_name] = {
                "id": prop_def.get("id"),
                "type": prop_def.get("type"),
            }
            # Include select/multi_select options
            if prop_def.get("type") in ("select", "multi_select"):
                options = prop_def.get(prop_def["type"], {}).get("options", [])
                schema[prop_name]["options"] = [
                    {"name": o["name"], "color": o.get("color")} for o in options
                ]

        return {
            "id": db["id"],
            "object": "database",
            "title": title,
            "description": self._extract_rich_text(db.get("description", [])),
            "url": db.get("url"),
            "created_time": db.get("created_time"),
            "last_edited_time": db.get("last_edited_time"),
            "schema": schema,
            "is_inline": db.get("is_inline", False),
        }

    def _parse_page(self, page: dict) -> dict[str, Any]:
        """Parse a page object."""
        # Extract title from properties
        title = ""
        properties = {}

        for prop_name, prop_value in page.get("properties", {}).items():
            prop_type = prop_value.get("type")
            parsed_value = self._parse_property_value(prop_value)
            properties[prop_name] = parsed_value

            # Title property
            if prop_type == "title":
                title = parsed_value

        # Get parent info
        parent = page.get("parent", {})
        parent_type = parent.get("type")
        parent_id = parent.get(parent_type) if parent_type else None

        return {
            "id": page["id"],
            "object": "page",
            "title": title,
            "url": page.get("url"),
            "created_time": page.get("created_time"),
            "last_edited_time": page.get("last_edited_time"),
            "created_by": page.get("created_by", {}).get("id"),
            "last_edited_by": page.get("last_edited_by", {}).get("id"),
            "parent_type": parent_type,
            "parent_id": parent_id,
            "properties": properties,
            "archived": page.get("archived", False),
        }

    def _parse_block(self, block: dict) -> dict[str, Any]:
        """Parse a block object."""
        block_type = block.get("type", "unknown")
        block_data = block.get(block_type, {})

        # Extract text content based on block type
        text = ""
        if "rich_text" in block_data:
            text = self._extract_rich_text(block_data["rich_text"])
        elif "caption" in block_data:
            text = self._extract_rich_text(block_data["caption"])
        elif block_type == "child_page":
            text = block_data.get("title", "")
        elif block_type == "child_database":
            text = block_data.get("title", "")

        parsed = {
            "id": block["id"],
            "type": block_type,
            "text": text,
            "has_children": block.get("has_children", False),
            "created_time": block.get("created_time"),
            "last_edited_time": block.get("last_edited_time"),
        }

        # Include type-specific data
        if block_type == "to_do":
            parsed["checked"] = block_data.get("checked", False)
        elif block_type in ("bulleted_list_item", "numbered_list_item"):
            parsed["list_type"] = block_type
        elif block_type == "code":
            parsed["language"] = block_data.get("language", "plain text")
        elif block_type in ("image", "file", "pdf", "video"):
            file_data = block_data.get("file") or block_data.get("external", {})
            parsed["url"] = file_data.get("url", "")

        return parsed

    def _parse_user(self, user: dict) -> dict[str, Any]:
        """Parse a user object."""
        return {
            "id": user["id"],
            "object": "user",
            "type": user.get("type"),
            "name": user.get("name"),
            "avatar_url": user.get("avatar_url"),
            "person": user.get("person", {}),
            "bot": user.get("bot", {}),
        }

    def _parse_comment(self, comment: dict) -> dict[str, Any]:
        """Parse a comment object."""
        return {
            "id": comment["id"],
            "object": "comment",
            "text": self._extract_rich_text(comment.get("rich_text", [])),
            "created_time": comment.get("created_time"),
            "created_by": comment.get("created_by", {}).get("id"),
            "parent": comment.get("parent", {}),
        }

    def _extract_title(self, title_array: list) -> str:
        """Extract plain text from a title array."""
        return self._extract_rich_text(title_array)

    def _extract_rich_text(self, rich_text_array: list) -> str:
        """Extract plain text from a rich text array."""
        if not rich_text_array:
            return ""
        return "".join(item.get("plain_text", "") for item in rich_text_array)

    def _parse_property_value(self, prop: dict) -> Any:
        """Parse a property value based on its type."""
        prop_type = prop.get("type")

        if prop_type == "title":
            return self._extract_rich_text(prop.get("title", []))
        elif prop_type == "rich_text":
            return self._extract_rich_text(prop.get("rich_text", []))
        elif prop_type == "number":
            return prop.get("number")
        elif prop_type == "select":
            select = prop.get("select")
            return select.get("name") if select else None
        elif prop_type == "multi_select":
            return [s.get("name") for s in prop.get("multi_select", [])]
        elif prop_type == "date":
            date = prop.get("date")
            if date:
                return {
                    "start": date.get("start"),
                    "end": date.get("end"),
                    "time_zone": date.get("time_zone"),
                }
            return None
        elif prop_type == "people":
            return [p.get("id") for p in prop.get("people", [])]
        elif prop_type == "files":
            files = []
            for f in prop.get("files", []):
                file_data = f.get("file") or f.get("external", {})
                files.append({
                    "name": f.get("name"),
                    "url": file_data.get("url"),
                })
            return files
        elif prop_type == "checkbox":
            return prop.get("checkbox", False)
        elif prop_type == "url":
            return prop.get("url")
        elif prop_type == "email":
            return prop.get("email")
        elif prop_type == "phone_number":
            return prop.get("phone_number")
        elif prop_type == "formula":
            formula = prop.get("formula", {})
            formula_type = formula.get("type")
            return formula.get(formula_type)
        elif prop_type == "relation":
            return [r.get("id") for r in prop.get("relation", [])]
        elif prop_type == "rollup":
            rollup = prop.get("rollup", {})
            rollup_type = rollup.get("type")
            return rollup.get(rollup_type)
        elif prop_type == "created_time":
            return prop.get("created_time")
        elif prop_type == "created_by":
            return prop.get("created_by", {}).get("id")
        elif prop_type == "last_edited_time":
            return prop.get("last_edited_time")
        elif prop_type == "last_edited_by":
            return prop.get("last_edited_by", {}).get("id")
        elif prop_type == "status":
            status = prop.get("status")
            return status.get("name") if status else None
        else:
            return None

    def blocks_to_text(self, blocks: list[dict]) -> str:
        """Convert blocks to plain text for indexing.

        Args:
            blocks: List of parsed block objects.

        Returns:
            Plain text representation of the content.
        """
        lines = []

        for block in blocks:
            block_type = block.get("type", "")
            text = block.get("text", "")

            if text:
                # Add formatting based on block type
                if block_type.startswith("heading_"):
                    level = block_type[-1]
                    lines.append(f"{'#' * int(level)} {text}")
                elif block_type == "to_do":
                    checkbox = "[x]" if block.get("checked") else "[ ]"
                    lines.append(f"{checkbox} {text}")
                elif block_type == "bulleted_list_item":
                    lines.append(f"• {text}")
                elif block_type == "numbered_list_item":
                    lines.append(f"- {text}")
                elif block_type == "code":
                    lang = block.get("language", "")
                    lines.append(f"```{lang}\n{text}\n```")
                elif block_type == "quote":
                    lines.append(f"> {text}")
                elif block_type == "callout":
                    lines.append(f"📌 {text}")
                else:
                    lines.append(text)

            # Recursively process children
            if block.get("children"):
                child_text = self.blocks_to_text(block["children"])
                if child_text:
                    # Indent child content
                    indented = "\n".join(f"  {line}" for line in child_text.split("\n"))
                    lines.append(indented)

        return "\n".join(lines)
