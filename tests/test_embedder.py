"""Tests for embedding generation and cache behavior."""

from unittest.mock import patch

from src.semantic import embedder as embedder_module
from src.semantic.embedder import Embedder


@patch("src.semantic.embedder.OpenAI")
def test_embedder_uses_configured_cache_dir(mock_openai, monkeypatch, tmp_path):
    """Default embedding cache location comes from config."""
    cache_dir = tmp_path / "embeddings_cache"
    monkeypatch.setattr(embedder_module, "EMBEDDING_CACHE_DIR", cache_dir)

    embedder = Embedder(api_key="test-key")

    assert embedder.cache_dir == cache_dir
    assert cache_dir.exists()
    mock_openai.assert_called_once_with(api_key="test-key")
