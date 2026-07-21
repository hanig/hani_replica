"""OpenAI embeddings generator."""

import hashlib
import json
import logging
from pathlib import Path

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_CACHE_DIR,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
)

logger = logging.getLogger(__name__)


class Embedder:
    """Generates embeddings using OpenAI's text-embedding-3-large model."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        cache_dir: Path | None = None,
    ):
        """Initialize the embedder.

        Args:
            api_key: OpenAI API key. Defaults to environment variable.
            model: Embedding model name. Defaults to config.
            cache_dir: Directory for embedding cache. Defaults to EMBEDDING_CACHE_DIR.
        """
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self.model = model or EMBEDDING_MODEL
        self.dimensions = EMBEDDING_DIMENSIONS
        self.batch_size = EMBEDDING_BATCH_SIZE

        self._client = OpenAI(api_key=self.api_key)

        # Setup cache
        self.cache_dir = cache_dir or EMBEDDING_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        self._load_cache()

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        # Check cache first
        cache_key = self._cache_key(text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Generate embedding
        embedding = self._embed_batch([text])[0]

        # Cache result
        self._cache[cache_key] = embedding
        self._save_cache_entry(cache_key, embedding)

        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        results = []
        to_embed = []
        to_embed_indices = []

        # Check cache for each text
        for i, text in enumerate(texts):
            cache_key = self._cache_key(text)
            if cache_key in self._cache:
                results.append((i, self._cache[cache_key]))
            else:
                to_embed.append(text)
                to_embed_indices.append(i)

        # Embed texts not in cache
        if to_embed:
            new_embeddings = self._embed_batch(to_embed)

            for idx, embedding in zip(to_embed_indices, new_embeddings, strict=True):
                results.append((idx, embedding))

                # Cache result
                cache_key = self._cache_key(texts[idx])
                self._cache[cache_key] = embedding
                self._save_cache_entry(cache_key, embedding)

        # Sort by original index and return just embeddings
        results.sort(key=lambda x: x[0])
        return [emb for _, emb in results]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
    )
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Internal method to generate embeddings with retry logic.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        # Process in batches to respect rate limits
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            # Truncate very long texts
            batch = [self._truncate_text(t) for t in batch]

            try:
                response = self._client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self.dimensions,
                )

                # Extract embeddings in order
                batch_embeddings = [None] * len(batch)
                for item in response.data:
                    batch_embeddings[item.index] = item.embedding

                all_embeddings.extend(batch_embeddings)

                logger.debug(f"Embedded batch {i // self.batch_size + 1}")

            except Exception as e:
                logger.error(f"Error embedding batch: {e}")
                raise

        return all_embeddings

    def _truncate_text(self, text: str, max_tokens: int = 8000) -> str:
        """Truncate text to fit within token limit.

        This is a simple character-based approximation.
        For more accurate token counting, use tiktoken.
        """
        # Rough approximation: 1 token ≈ 4 characters
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            return text[:max_chars]
        return text

    def _cache_key(self, text: str) -> str:
        """Generate a cache key for text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _load_cache(self) -> None:
        """Load embedding cache from disk."""
        loaded = 0
        for entry_file in self.cache_dir.glob("*.json"):
            if entry_file.name == "cache_index.json":
                continue
            try:
                with open(entry_file) as f:
                    embedding = json.load(f)
                if isinstance(embedding, list):
                    self._cache[entry_file.stem] = embedding
                    loaded += 1
            except Exception as e:
                logger.warning(f"Error loading embedding cache entry {entry_file.name}: {e}")

        if loaded:
            logger.info(f"Loaded {loaded} embedding cache entries")

    def _save_cache_entry(self, key: str, embedding: list[float]) -> None:
        """Save a single cache entry to disk."""
        try:
            # Save embedding to individual file
            entry_file = self.cache_dir / f"{key}.json"
            with open(entry_file, "w") as f:
                json.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Error saving cache entry: {e}")

    def _load_cache_entry(self, key: str) -> list[float] | None:
        """Load a single cache entry from disk."""
        entry_file = self.cache_dir / f"{key}.json"
        if entry_file.exists():
            try:
                with open(entry_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.dimensions

    def estimate_cost(self, texts: list[str]) -> float:
        """Estimate the cost to embed a list of texts.

        Args:
            texts: List of texts.

        Returns:
            Estimated cost in USD.
        """
        # Count approximate tokens
        total_chars = sum(len(t) for t in texts)
        estimated_tokens = total_chars / 4  # Rough approximation

        # text-embedding-3-large costs $0.00013 per 1K tokens
        cost_per_1k = 0.00013
        return (estimated_tokens / 1000) * cost_per_1k
