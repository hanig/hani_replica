"""Tests for Gmail indexing."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.indexers.gmail_indexer import GmailIndexer, _is_stale_history_error
from src.integrations.gmail import GmailClient as RealGmailClient
from src.knowledge_graph import KnowledgeGraph


def test_is_stale_history_error_detects_gmail_404_text():
    """Gmail stale history errors should be recognized from API text."""
    error = RuntimeError(
        "Requested entity was not found for startHistoryId=4942724"
    )

    assert _is_stale_history_error(error)


def test_delta_stale_history_falls_back_to_bounded_full_sync(tmp_path: Path):
    """Expired Gmail history tokens should trigger bounded full recovery."""
    kg = KnowledgeGraph(tmp_path / "kg.db")
    kg.set_last_sync(
        source="gmail",
        account="arc",
        last_sync=datetime.now(UTC),
        sync_token="4942724",
    )
    indexer = GmailIndexer(kg)
    client = MagicMock()
    client.list_history.side_effect = RuntimeError(
        "Requested entity was not found for startHistoryId=4942724"
    )

    with patch("src.indexers.gmail_indexer.GOOGLE_ACCOUNTS", ["arc"]), patch(
        "src.indexers.gmail_indexer.GMAIL_STALE_HISTORY_FULL_SYNC_LIMIT", 25
    ), patch("src.indexers.gmail_indexer.GmailClient", return_value=client), patch.object(
        indexer,
        "index_all",
        return_value={
            "messages_processed": 1,
            "messages_indexed": 1,
            "people_extracted": 0,
            "errors": 0,
        },
    ) as index_all:
        stats = indexer.index_delta("arc")

    index_all.assert_called_once_with("arc", max_messages=25)
    assert stats["recovered_from_stale_history"] is True
    assert stats["stale_history_id"] == "4942724"


def test_delta_added_message_uses_complete_stats_shape(tmp_path: Path):
    """Delta-added messages should update indexing counters without KeyError."""
    kg = KnowledgeGraph(tmp_path / "kg.db")
    kg.set_last_sync(
        source="gmail",
        account="arc",
        last_sync=datetime.now(UTC),
        sync_token="old-history",
    )
    message = {
        "id": "msg1",
        "threadId": "thread1",
        "internalDate": "1783260000000",
        "labelIds": ["INBOX"],
        "payload": {
            "headers": [
                {"name": "Subject", "value": "Hello"},
                {"name": "From", "value": "A User <a@example.com>"},
                {"name": "To", "value": "B User <b@example.com>"},
            ],
            "body": {"data": ""},
        },
    }
    client = MagicMock()
    client.list_history.return_value = {
        "historyId": "new-history",
        "history": [
            {"messagesAdded": [{"message": {"id": "msg1"}}]},
        ],
    }
    client.get_message.return_value = message
    gmail_client_cls = MagicMock(return_value=client)
    gmail_client_cls.parse_message.side_effect = RealGmailClient.parse_message

    with patch("src.indexers.gmail_indexer.GOOGLE_ACCOUNTS", ["arc"]), patch(
        "src.indexers.gmail_indexer.GmailClient",
        gmail_client_cls,
    ):
        stats = GmailIndexer(kg).index_delta("arc")

    assert stats["messages_added"] == 1
    assert stats["messages_indexed"] == 1
    assert stats["people_extracted"] == 2
    assert kg.get_last_sync("gmail", "arc")["last_sync_token"] == "new-history"
