"""Tests for GitHub client helpers."""

from unittest.mock import MagicMock

from src.integrations.github_client import GitHubClient


def test_search_code_in_repo_delegates_to_search_code():
    """search_code_in_repo should delegate to search_code with repo filter."""
    client = object.__new__(GitHubClient)
    client.search_code = MagicMock(return_value=[{"path": "src/app.py"}])

    result = GitHubClient.search_code_in_repo(
        client,
        repo="owner/repo",
        query="def run",
        max_results=7,
    )

    assert result == [{"path": "src/app.py"}]
    client.search_code.assert_called_once_with(
        query="def run",
        repo="owner/repo",
        max_results=7,
    )
