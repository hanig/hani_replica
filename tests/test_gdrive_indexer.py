"""Tests for Google Drive indexing."""

from pathlib import Path

from src.indexers.gdrive_indexer import DriveIndexer
from src.knowledge_graph import KnowledgeGraph


class _FakeDriveClient:
    """Minimal Drive client for indexer tests."""

    def get_file_content(self, _file_id: str) -> str:
        return "Indexed file body"


def test_index_file_accepts_delta_stats_shape(tmp_path: Path):
    """Delta sync stats should not fail when content is indexed."""
    kg = KnowledgeGraph(tmp_path / "kg.db")
    indexer = DriveIndexer(kg)
    stats = {
        "files_added": 0,
        "files_updated": 0,
        "files_deleted": 0,
        "errors": 0,
    }
    file = {
        "id": "file123",
        "name": "Project Notes",
        "mimeType": "text/plain",
        "modifiedTime": "2026-07-05T10:00:00Z",
        "webViewLink": "https://drive.example/file123",
        "owners": [
            {
                "emailAddress": "owner@example.com",
                "displayName": "Owner Example",
            }
        ],
    }

    is_new = indexer._index_file("work", _FakeDriveClient(), file, True, stats)

    assert is_new is True
    assert stats["files_indexed"] == 1
    assert stats["content_indexed"] == 1
    assert stats["people_extracted"] == 1
    content = kg.get_content("drive:work:file123")
    assert content is not None
    assert content["body"] == "Indexed file body"
