"""Zotero API client."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from pyzotero import zotero

from ..config import get_env

logger = logging.getLogger(__name__)

# Configuration
ZOTERO_API_KEY = get_env("ZOTERO_API_KEY")
ZOTERO_USER_ID = get_env("ZOTERO_USER_ID")
ZOTERO_LIBRARY_TYPE = get_env("ZOTERO_LIBRARY_TYPE", "user")
ZOTERO_DEFAULT_COLLECTION = get_env("ZOTERO_DEFAULT_COLLECTION", "")


class ZoteroClient:
    """Client for interacting with Zotero API."""

    def __init__(
        self,
        api_key: str | None = None,
        user_id: str | None = None,
        library_type: str | None = None,
    ):
        """Initialize Zotero client.

        Args:
            api_key: Zotero API key.
            user_id: Zotero user ID.
            library_type: Library type ("user" or "group").
        """
        self.api_key = api_key or ZOTERO_API_KEY
        self.user_id = user_id or ZOTERO_USER_ID
        self.library_type = library_type or ZOTERO_LIBRARY_TYPE

        if not self.api_key:
            raise ValueError("Zotero API key is required (ZOTERO_API_KEY)")
        if not self.user_id:
            raise ValueError("Zotero user ID is required (ZOTERO_USER_ID)")

        self._client = zotero.Zotero(
            self.user_id,
            self.library_type,
            self.api_key,
        )

    def test_connection(self) -> dict[str, Any]:
        """Test the connection to Zotero API.

        Returns:
            Connection status and library info.
        """
        try:
            # Try to get key permissions to verify connection
            items = self._client.top(limit=1)
            item_count = self._client.count_items()
            logger.info(f"Connected to Zotero. Library has {item_count} items.")
            return {
                "success": True,
                "user_id": self.user_id,
                "library_type": self.library_type,
                "item_count": item_count,
            }
        except Exception as e:
            logger.error(f"Zotero connection test failed: {e}")
            return {"success": False, "error": str(e)}

    # --- Read Operations ---

    def list_items(
        self,
        limit: int = 100,
        item_type: str | None = None,
        sort: str = "dateModified",
        direction: str = "desc",
    ) -> list[dict[str, Any]]:
        """List items in the library.

        Args:
            limit: Maximum number of items to return.
            item_type: Filter by item type (e.g., "journalArticle", "book").
            sort: Field to sort by.
            direction: Sort direction ("asc" or "desc").

        Returns:
            List of items.
        """
        items = []
        start = 0
        batch_size = min(100, limit)

        while len(items) < limit:
            kwargs = {
                "limit": batch_size,
                "start": start,
                "sort": sort,
                "direction": direction,
            }
            if item_type:
                kwargs["itemType"] = item_type

            batch = self._client.top(**kwargs)
            if not batch:
                break

            for item in batch:
                items.append(self._parse_item(item))

            start += len(batch)
            if len(batch) < batch_size:
                break

        return items[:limit]

    def get_item(self, item_key: str) -> dict[str, Any]:
        """Get a single item by key.

        Args:
            item_key: The item's key.

        Returns:
            Item data.
        """
        item = self._client.item(item_key)
        return self._parse_item(item)

    def search_items(
        self,
        query: str,
        max_results: int = 50,
    ) -> list[dict[str, Any]]:
        """Search for items by query.

        Args:
            query: Search query text.
            max_results: Maximum number of results.

        Returns:
            List of matching items.
        """
        # Zotero quick search searches title, creator, year, and other fields
        items = self._client.top(q=query, limit=max_results)
        return [self._parse_item(item) for item in items]

    def get_recent_items(self, days: int = 7) -> list[dict[str, Any]]:
        """Get recently added items.

        Args:
            days: Look back N days.

        Returns:
            List of recently added items.
        """
        # Get items sorted by date added
        items = self._client.top(
            limit=100,
            sort="dateAdded",
            direction="desc",
        )

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        recent = []

        for item in items:
            data = item.get("data", {})
            date_added = data.get("dateAdded")
            if date_added:
                try:
                    added_dt = datetime.fromisoformat(date_added.replace("Z", "+00:00"))
                    if added_dt >= cutoff:
                        recent.append(self._parse_item(item))
                    else:
                        # Items are sorted by date, so we can stop
                        break
                except (ValueError, TypeError):
                    continue

        return recent

    # --- Collections ---

    def list_collections(self) -> list[dict[str, Any]]:
        """List all collections.

        Returns:
            List of collections.
        """
        collections = self._client.collections()
        return [self._parse_collection(c) for c in collections]

    def get_collection_by_name(self, name: str) -> dict[str, Any] | None:
        """Find a collection by name.

        Args:
            name: Collection name (case-insensitive).

        Returns:
            Collection data or None if not found.
        """
        collections = self.list_collections()
        name_lower = name.lower()

        for collection in collections:
            if collection["name"].lower() == name_lower:
                return collection

        return None

    def get_collection_items(
        self,
        collection_key: str,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """Get items in a collection.

        Args:
            collection_key: The collection's key.
            max_results: Maximum number of items.

        Returns:
            List of items in the collection.
        """
        items = self._client.collection_items(collection_key, limit=max_results)
        return [self._parse_item(item) for item in items]

    # --- Tags ---

    def list_tags(self) -> list[dict[str, Any]]:
        """List all tags in the library.

        Returns:
            List of tags with counts.
        """
        tags = self._client.tags()
        return [
            {
                "tag": tag.get("tag", ""),
                "count": tag.get("meta", {}).get("numItems", 0),
            }
            for tag in tags
        ]

    def get_items_by_tag(
        self,
        tag: str,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """Get items with a specific tag.

        Args:
            tag: The tag to search for.
            max_results: Maximum number of results.

        Returns:
            List of items with the tag.
        """
        items = self._client.top(tag=tag, limit=max_results)
        return [self._parse_item(item) for item in items]

    # --- Notes ---

    def get_item_notes(self, item_key: str) -> list[dict[str, Any]]:
        """Get notes attached to an item.

        Args:
            item_key: The parent item's key.

        Returns:
            List of notes.
        """
        children = self._client.children(item_key)
        notes = []

        for child in children:
            data = child.get("data", {})
            if data.get("itemType") == "note":
                notes.append({
                    "key": data.get("key"),
                    "note": data.get("note", ""),
                    "parent_key": item_key,
                    "date_added": data.get("dateAdded"),
                    "date_modified": data.get("dateModified"),
                })

        return notes

    def get_item_attachments(self, item_key: str) -> list[dict[str, Any]]:
        """Get attachments for an item.

        Args:
            item_key: The parent item's key.

        Returns:
            List of attachments.
        """
        children = self._client.children(item_key)
        attachments = []

        for child in children:
            data = child.get("data", {})
            if data.get("itemType") == "attachment":
                attachments.append({
                    "key": data.get("key"),
                    "title": data.get("title", ""),
                    "filename": data.get("filename"),
                    "content_type": data.get("contentType"),
                    "link_mode": data.get("linkMode"),
                    "url": data.get("url"),
                    "parent_key": item_key,
                })

        return attachments

    # --- Write Operations ---

    def _fetch_crossref_metadata(self, doi: str) -> dict[str, Any] | None:
        """Fetch metadata from CrossRef API.

        Args:
            doi: The DOI to look up.

        Returns:
            Metadata dict or None if not found.
        """
        import httpx

        url = f"https://api.crossref.org/works/{doi}"
        headers = {"User-Agent": "Engram/1.0"}

        try:
            with httpx.Client() as client:
                response = client.get(url, headers=headers, timeout=10.0)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("message", {})
        except Exception as e:
            logger.warning(f"CrossRef lookup failed for {doi}: {e}")

        return None

    def _fetch_metadata_from_page(self, url: str) -> dict[str, Any] | None:
        """Fetch metadata from page meta tags.

        Args:
            url: The page URL.

        Returns:
            Metadata dict or None if extraction fails.
        """
        import httpx
        import re

        try:
            with httpx.Client() as client:
                response = client.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0"},
                    follow_redirects=True,
                    timeout=15.0,
                )
                if response.status_code != 200:
                    return None

                html = response.text
                metadata = {}

                # Extract from meta tags
                def get_meta(name: str) -> str:
                    # Try various meta tag formats
                    patterns = [
                        rf'<meta\s+name="{name}"\s+content="([^"]+)"',
                        rf'<meta\s+content="([^"]+)"\s+name="{name}"',
                        rf'<meta\s+property="{name}"\s+content="([^"]+)"',
                        rf'<meta\s+content="([^"]+)"\s+property="{name}"',
                    ]
                    for pattern in patterns:
                        match = re.search(pattern, html, re.IGNORECASE)
                        if match:
                            return match.group(1)
                    return ""

                # Common metadata fields
                metadata["title"] = [
                    get_meta("citation_title") or
                    get_meta("dc.title") or
                    get_meta("og:title") or
                    ""
                ]

                # Authors (citation_author can appear multiple times)
                authors = re.findall(
                    r'<meta\s+name="citation_author"\s+content="([^"]+)"',
                    html,
                    re.IGNORECASE,
                )
                if authors:
                    metadata["author"] = []
                    for author in authors:
                        # Parse "Last, First" format
                        parts = author.split(",", 1)
                        if len(parts) == 2:
                            metadata["author"].append({
                                "family": parts[0].strip(),
                                "given": parts[1].strip(),
                            })
                        else:
                            metadata["author"].append({"name": author})

                # Journal
                journal = get_meta("citation_journal_title") or get_meta("dc.source")
                if journal:
                    metadata["container-title"] = [journal]

                # Date
                date = get_meta("citation_publication_date") or get_meta("citation_date")
                if date:
                    # Parse date into parts
                    parts = date.split("/") if "/" in date else date.split("-")
                    date_parts = []
                    for p in parts:
                        try:
                            date_parts.append(int(p))
                        except ValueError:
                            pass
                    if date_parts:
                        metadata["published"] = {"date-parts": [date_parts]}

                # Volume/Issue/Pages
                metadata["volume"] = get_meta("citation_volume")
                metadata["issue"] = get_meta("citation_issue")
                metadata["page"] = get_meta("citation_firstpage")

                # Abstract
                abstract = get_meta("description") or get_meta("og:description")
                if abstract:
                    metadata["abstract"] = abstract

                # Publisher
                metadata["publisher"] = get_meta("citation_publisher") or get_meta("dc.publisher")

                # Only return if we got at least a title
                if metadata.get("title", [""])[0]:
                    return metadata

        except Exception as e:
            logger.warning(f"Page metadata extraction failed for {url}: {e}")

        return None

    def add_item_by_doi(
        self,
        doi: str,
        collection: str | None = None,
    ) -> dict[str, Any]:
        """Add an item to the library by DOI.

        Args:
            doi: The DOI of the item.
            collection: Collection name to add to (uses default if not specified).

        Returns:
            Created item data.

        Raises:
            ValueError: If DOI lookup fails.
        """
        # Normalize DOI
        doi = doi.strip()
        if doi.startswith("https://doi.org/"):
            doi = doi[16:]
        elif doi.startswith("http://doi.org/"):
            doi = doi[15:]
        elif doi.startswith("doi:"):
            doi = doi[4:]

        # Fetch metadata from CrossRef, fall back to page scraping
        metadata = self._fetch_crossref_metadata(doi)
        if not metadata:
            # Try to extract from the DOI landing page
            page_url = f"https://doi.org/{doi}"
            metadata = self._fetch_metadata_from_page(page_url)

        # Create template based on item type
        item_type = "journalArticle"
        if metadata:
            cr_type = metadata.get("type", "")
            if cr_type == "book":
                item_type = "book"
            elif cr_type == "book-chapter":
                item_type = "bookSection"
            elif cr_type in ("proceedings-article", "conference-paper"):
                item_type = "conferencePaper"

        template = self._client.item_template(item_type)
        template["DOI"] = doi
        template["url"] = f"https://doi.org/{doi}"

        # Populate from CrossRef metadata
        if metadata:
            # Title
            titles = metadata.get("title", [])
            if titles:
                template["title"] = titles[0]

            # Authors
            authors = metadata.get("author", [])
            creators = []
            for author in authors:
                creator = {"creatorType": "author"}
                if author.get("family"):
                    creator["lastName"] = author.get("family", "")
                    creator["firstName"] = author.get("given", "")
                elif author.get("name"):
                    creator["name"] = author.get("name")
                if creator.get("lastName") or creator.get("name"):
                    creators.append(creator)
            if creators:
                template["creators"] = creators

            # Journal/Publication
            container = metadata.get("container-title", [])
            if container:
                if item_type == "journalArticle":
                    template["publicationTitle"] = container[0]
                elif item_type == "bookSection":
                    template["bookTitle"] = container[0]

            # Volume, Issue, Pages
            template["volume"] = metadata.get("volume", "")
            template["issue"] = metadata.get("issue", "")
            template["pages"] = metadata.get("page", "")

            # Date
            published = metadata.get("published", {})
            date_parts = published.get("date-parts", [[]])
            if date_parts and date_parts[0]:
                parts = date_parts[0]
                if len(parts) >= 1:
                    template["date"] = str(parts[0])  # Year
                    if len(parts) >= 2:
                        template["date"] = f"{parts[0]}-{parts[1]:02d}"
                    if len(parts) >= 3:
                        template["date"] = f"{parts[0]}-{parts[1]:02d}-{parts[2]:02d}"

            # Abstract
            abstract = metadata.get("abstract", "")
            if abstract:
                # Strip HTML tags from abstract
                import re
                abstract = re.sub(r"<[^>]+>", "", abstract)
                template["abstractNote"] = abstract

            # ISSN
            issn = metadata.get("ISSN", [])
            if issn:
                template["ISSN"] = issn[0]

            # Publisher
            template["publisher"] = metadata.get("publisher", "")

        # Determine collection
        collection_key = None
        collection_name = collection or ZOTERO_DEFAULT_COLLECTION
        if collection_name:
            coll = self.get_collection_by_name(collection_name)
            if coll:
                collection_key = coll["key"]
                template["collections"] = [collection_key]

        # Create the item
        result = self._client.create_items([template])

        if result.get("successful"):
            created = list(result["successful"].values())[0]
            logger.info(f"Created Zotero item with DOI: {doi}")
            return self._parse_item(created)
        else:
            error = result.get("failed", {})
            raise ValueError(f"Failed to create item: {error}")

    def _extract_doi_from_url(self, url: str) -> str | None:
        """Try to extract a DOI from a publisher URL.

        Args:
            url: The URL to extract DOI from.

        Returns:
            DOI string or None if not found.
        """
        import re

        # Direct DOI URLs
        if "doi.org/" in url:
            match = re.search(r"doi\.org/(10\.\d+/[^\s&?#]+)", url)
            if match:
                return match.group(1)

        # Nature: nature.com/articles/s41586-025-10018-w -> 10.1038/s41586-025-10018-w
        if "nature.com/articles/" in url:
            match = re.search(r"nature\.com/articles/(s\d+[\w-]+)", url)
            if match:
                return f"10.1038/{match.group(1)}"

        # Cell/Elsevier: cell.com/cell/fulltext/S0092-8674(24)00123-4
        if "cell.com/" in url:
            match = re.search(r"cell\.com/[^/]+/fulltext/(S[\d-]+\(\d+\)[\d-]+)", url)
            if match:
                # Cell DOIs are complex, try CrossRef lookup by URL
                pass

        # Science: science.org/doi/10.1126/science.xxx
        if "science.org/doi/" in url:
            match = re.search(r"science\.org/doi/(10\.\d+/[^\s&?#]+)", url)
            if match:
                return match.group(1)

        # PNAS: pnas.org/doi/10.1073/pnas.xxx
        if "pnas.org/doi/" in url:
            match = re.search(r"pnas\.org/doi/(10\.\d+/[^\s&?#]+)", url)
            if match:
                return match.group(1)

        # PubMed Central: ncbi.nlm.nih.gov/pmc/articles/PMCxxxx
        # Would need to look up DOI via PMC API

        # bioRxiv/medRxiv: biorxiv.org/content/10.1101/xxx
        if "biorxiv.org/content/" in url or "medrxiv.org/content/" in url:
            match = re.search(r"rxiv\.org/content/(10\.\d+/[^\s&?#]+)", url)
            if match:
                return match.group(1)

        # arXiv: arxiv.org/abs/2601.07372 or arxiv.org/pdf/2601.07372
        if "arxiv.org/" in url:
            match = re.search(r"arxiv\.org/(?:abs|pdf|html)/(\d+\.\d+)", url)
            if match:
                return f"10.48550/arXiv.{match.group(1)}"

        # PubMed: pubmed.ncbi.nlm.nih.gov/12345678
        if "pubmed.ncbi.nlm.nih.gov/" in url:
            match = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", url)
            if match:
                pmid = match.group(1)
                doi = self._resolve_pmid_to_doi(pmid)
                if doi:
                    return doi

        # Generic fallback: look for a DOI embedded anywhere in the URL
        # Covers Wiley, Springer, T&F, ACS, SAGE, Oxford, Frontiers, PLoS, etc.
        match = re.search(r"(?:^|[/=])(10\.\d{4,}/[^\s&#]+)", url)
        if match:
            return match.group(1).rstrip("/")

        return None

    def _resolve_pmid_to_doi(self, pmid: str) -> str | None:
        """Resolve a PubMed ID to a DOI via the NCBI API.

        Args:
            pmid: The PubMed ID (numeric string).

        Returns:
            DOI string or None if not found.
        """
        import httpx

        ncbi_url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            f"?db=pubmed&id={pmid}&retmode=json"
        )
        try:
            with httpx.Client() as client:
                response = client.get(ncbi_url, timeout=10.0)
                if response.status_code == 200:
                    data = response.json()
                    article = data.get("result", {}).get(pmid, {})
                    for id_entry in article.get("articleids", []):
                        if id_entry.get("idtype") == "doi":
                            return id_entry["value"]
        except Exception as e:
            logger.warning(f"NCBI lookup failed for PMID {pmid}: {e}")

        return None

    def add_item_by_url(
        self,
        url: str,
        collection: str | None = None,
    ) -> dict[str, Any]:
        """Add an item to the library by URL.

        Args:
            url: The URL of the item.
            collection: Collection name to add to.

        Returns:
            Created item data.
        """
        # Try to extract DOI from URL
        doi = self._extract_doi_from_url(url)
        if doi:
            logger.info(f"Extracted DOI {doi} from URL {url}")
            return self.add_item_by_doi(doi, collection)

        # Fall back to creating a webpage item
        template = self._client.item_template("webpage")
        template["url"] = url
        template["title"] = url  # Will be updated if metadata is fetched

        # Determine collection
        collection_name = collection or ZOTERO_DEFAULT_COLLECTION
        if collection_name:
            coll = self.get_collection_by_name(collection_name)
            if coll:
                template["collections"] = [coll["key"]]

        result = self._client.create_items([template])

        if result.get("successful"):
            created = list(result["successful"].values())[0]
            logger.info(f"Created Zotero item from URL: {url}")
            return self._parse_item(created)
        else:
            error = result.get("failed", {})
            raise ValueError(f"Failed to create item: {error}")

    def add_item_to_collection(
        self,
        item_key: str,
        collection_key: str,
    ) -> bool:
        """Add an existing item to a collection.

        Args:
            item_key: The item's key.
            collection_key: The collection's key.

        Returns:
            True if successful.
        """
        item = self._client.item(item_key)
        data = item.get("data", {})
        collections = data.get("collections", [])

        if collection_key not in collections:
            collections.append(collection_key)
            data["collections"] = collections
            self._client.update_item(item)
            logger.info(f"Added item {item_key} to collection {collection_key}")

        return True

    def create_note(
        self,
        parent_key: str,
        note_content: str,
    ) -> dict[str, Any]:
        """Create a note attached to an item.

        Args:
            parent_key: The parent item's key.
            note_content: The note content (HTML allowed).

        Returns:
            Created note data.
        """
        template = self._client.item_template("note")
        template["parentItem"] = parent_key
        template["note"] = note_content

        result = self._client.create_items([template])

        if result.get("successful"):
            created = list(result["successful"].values())[0]
            return {
                "key": created.get("data", {}).get("key"),
                "note": note_content,
                "parent_key": parent_key,
            }
        else:
            error = result.get("failed", {})
            raise ValueError(f"Failed to create note: {error}")

    # --- Parsing Helpers ---

    def _parse_item(self, item: dict) -> dict[str, Any]:
        """Parse a Zotero item into a clean format."""
        data = item.get("data", {})

        # Extract creators (authors, editors, etc.)
        creators = []
        for creator in data.get("creators", []):
            if creator.get("name"):
                # Single name field
                creators.append({
                    "name": creator["name"],
                    "type": creator.get("creatorType", "author"),
                })
            else:
                # First/last name fields
                first = creator.get("firstName", "")
                last = creator.get("lastName", "")
                name = f"{first} {last}".strip() if first or last else "Unknown"
                creators.append({
                    "name": name,
                    "first_name": first,
                    "last_name": last,
                    "type": creator.get("creatorType", "author"),
                })

        # Get primary authors (first 3)
        authors = [c["name"] for c in creators if c["type"] == "author"][:3]
        if len([c for c in creators if c["type"] == "author"]) > 3:
            authors.append("et al.")

        return {
            "key": data.get("key"),
            "item_type": data.get("itemType"),
            "title": data.get("title", "Untitled"),
            "abstract": data.get("abstractNote", ""),
            "authors": authors,
            "creators": creators,
            "date": data.get("date", ""),
            "year": self._extract_year(data.get("date", "")),
            "publication": data.get("publicationTitle") or data.get("bookTitle") or "",
            "journal": data.get("publicationTitle", ""),
            "volume": data.get("volume", ""),
            "issue": data.get("issue", ""),
            "pages": data.get("pages", ""),
            "doi": data.get("DOI", ""),
            "url": data.get("url", ""),
            "tags": [t.get("tag") for t in data.get("tags", []) if t.get("tag")],
            "collections": data.get("collections", []),
            "date_added": data.get("dateAdded"),
            "date_modified": data.get("dateModified"),
            "extra": data.get("extra", ""),
        }

    def _parse_collection(self, collection: dict) -> dict[str, Any]:
        """Parse a Zotero collection."""
        data = collection.get("data", {})
        meta = collection.get("meta", {})

        return {
            "key": data.get("key"),
            "name": data.get("name", ""),
            "parent_key": data.get("parentCollection") or None,
            "item_count": meta.get("numItems", 0),
        }

    def _extract_year(self, date_str: str) -> str:
        """Extract year from a date string."""
        if not date_str:
            return ""

        # Try to find a 4-digit year
        import re
        match = re.search(r"\b(19|20)\d{2}\b", date_str)
        if match:
            return match.group(0)

        return ""
