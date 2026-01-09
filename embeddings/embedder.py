"""Unified embedding generation using OpenAI's embedding models.

This module provides a robust embedder class that handles single and batch
embedding generation with rate limiting, retries, and error handling.
"""

import asyncio
import logging
from typing import Any

from openai import AsyncOpenAI, RateLimitError, APIError, APITimeoutError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from config import settings
from utils.cache import get_embedding_cache
from utils.circuit_breaker import call_embedding_with_breaker
from utils.rate_limiter import rate_limiter

logger = logging.getLogger(__name__)


class UnifiedEmbedder:
    """Unified embedder using OpenAI's embedding models.

    Handles embedding generation with:
    - Single text embedding
    - Batch processing with automatic splitting
    - Rate limit handling with exponential backoff
    - Retry logic for transient failures
    - Dimension validation

    Attributes:
        client: AsyncOpenAI client instance
        model: Embedding model name
        dimensions: Expected embedding dimensions
        batch_size: Maximum batch size for API calls
    """

    # OpenAI API limits
    MAX_BATCH_SIZE = 2048  # Maximum texts per API call
    MAX_INPUT_TOKENS = 8191  # Maximum tokens per input

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        """Initialize the embedder.

        Args:
            api_key: OpenAI API key (defaults to settings)
            model: Embedding model name (defaults to settings)
            dimensions: Expected embedding dimensions (defaults to settings)
            batch_size: Batch size for processing (defaults to settings)

        Raises:
            ValueError: If API key is not provided
        """
        self.api_key = api_key or settings.openai_api_key
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided via api_key parameter or "
                "OPENAI_API_KEY environment variable"
            )

        self.model = model or settings.embedding_model
        self.dimensions = dimensions or settings.embedding_dimension
        self.batch_size = min(
            batch_size or settings.embedding_batch_size,
            self.MAX_BATCH_SIZE,
        )

        # Initialize client
        self.client = AsyncOpenAI(api_key=self.api_key)

        logger.info(
            f"UnifiedEmbedder initialized with model={self.model}, "
            f"dimensions={self.dimensions}, batch_size={self.batch_size}"
        )

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text with caching.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            ValueError: If text is empty
            RuntimeError: If embedding generation fails
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        try:
            # Check cache first
            cache = get_embedding_cache()
            cache_key = cache.hash_content(text)
            cached = await cache.get(cache_key)

            if cached is not None:
                logger.debug("Embedding cache hit")
                return cached

            # Generate embedding
            embeddings = await call_embedding_with_breaker(
                self._embed_with_retry, [text]
            )
            result = embeddings[0]

            # Cache result
            await cache.set(cache_key, result)

            return result
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}") from e

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts with caching.

        Automatically splits large batches into smaller chunks to respect
        API limits. Processes chunks sequentially to avoid rate limits.
        Checks cache for each text individually to maximize cache hits.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (one per input text)

        Raises:
            ValueError: If texts list is empty or contains empty strings
            RuntimeError: If embedding generation fails
        """
        if not texts:
            raise ValueError("Cannot embed empty text list")

        # Validate all texts are non-empty
        empty_indices = [i for i, text in enumerate(texts) if not text or not text.strip()]
        if empty_indices:
            raise ValueError(
                f"Found empty texts at indices: {empty_indices[:5]}"
                f"{'...' if len(empty_indices) > 5 else ''}"
            )

        try:
            cache = get_embedding_cache()
            results: list[list[float] | None] = []
            uncached_texts: list[str] = []
            uncached_indices: list[int] = []

            # Check cache for each text
            for i, text in enumerate(texts):
                cache_key = cache.hash_content(text)
                cached = await cache.get(cache_key)
                if cached is not None:
                    results.append(cached)
                else:
                    results.append(None)  # Placeholder
                    uncached_texts.append(text)
                    uncached_indices.append(i)

            cache_hit_count = len(texts) - len(uncached_texts)
            if cache_hit_count > 0:
                logger.info(
                    f"Embedding cache: {cache_hit_count}/{len(texts)} hits "
                    f"({cache_hit_count / len(texts):.1%})"
                )

            # Generate embeddings for uncached texts
            if uncached_texts:
                # Split into batches if needed
                if len(uncached_texts) <= self.batch_size:
                    new_embeddings = await call_embedding_with_breaker(
                        self._embed_with_retry, uncached_texts
                    )
                else:
                    # Process in batches
                    logger.info(
                        f"Processing {len(uncached_texts)} uncached texts in batches of {self.batch_size}"
                    )
                    new_embeddings: list[list[float]] = []

                    for i in range(0, len(uncached_texts), self.batch_size):
                        batch = uncached_texts[i:i + self.batch_size]
                        batch_embeddings = await call_embedding_with_breaker(
                            self._embed_with_retry, batch
                        )
                        new_embeddings.extend(batch_embeddings)

                        logger.debug(
                            f"Processed batch {i // self.batch_size + 1}/"
                            f"{(len(uncached_texts) + self.batch_size - 1) // self.batch_size}"
                        )

                        # Small delay between batches to avoid rate limits
                        if i + self.batch_size < len(uncached_texts):
                            await asyncio.sleep(0.1)

                # Fill in results and cache
                for idx, embedding in zip(uncached_indices, new_embeddings):
                    results[idx] = embedding
                    cache_key = cache.hash_content(texts[idx])
                    await cache.set(cache_key, embedding)

            # Convert results to list (remove Optional type)
            return [r for r in results if r is not None]

        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise RuntimeError(f"Batch embedding generation failed: {e}") from e

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings with retry logic.

        This method is decorated with tenacity retry logic to handle:
        - Rate limit errors (429)
        - Timeout errors
        - Transient API errors

        Args:
            texts: List of texts to embed (must be <= batch_size)

        Returns:
            List of embedding vectors

        Raises:
            RateLimitError: If rate limit is hit (will retry)
            APITimeoutError: If request times out (will retry)
            APIError: For other API errors (will retry)
        """
        # Acquire rate limit tokens before making API call
        if rate_limiter.is_registered("openai"):
            await rate_limiter.acquire("openai", tokens=len(texts))
            logger.debug(f"Acquired rate limit tokens for {len(texts)} embeddings")

        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts,
                encoding_format="float",
            )

            # Extract embeddings
            embeddings = [item.embedding for item in response.data]

            # Validate dimensions
            for i, embedding in enumerate(embeddings):
                if len(embedding) != self.dimensions:
                    logger.warning(
                        f"Embedding {i} has unexpected dimensions: "
                        f"{len(embedding)} vs expected {self.dimensions}"
                    )

            logger.debug(f"Generated {len(embeddings)} embeddings")
            return embeddings

        except RateLimitError as e:
            logger.warning(f"Rate limit hit, will retry: {e}")
            raise
        except APITimeoutError as e:
            logger.warning(f"API timeout, will retry: {e}")
            raise
        except APIError as e:
            logger.warning(f"API error, will retry: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during embedding generation: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}") from e

    async def validate_connection(self) -> bool:
        """Validate that the API connection works.

        Returns:
            True if connection is valid, False otherwise
        """
        try:
            test_embedding = await self.embed_text("test")
            return len(test_embedding) == self.dimensions
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.close()

    async def close(self) -> None:
        """Close the OpenAI client and cleanup resources."""
        try:
            await self.client.close()
            logger.debug("OpenAI client closed")
        except Exception as e:
            logger.warning(f"Error closing OpenAI client: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get embedder statistics including cache performance.

        Returns:
            Dictionary with embedder configuration and cache stats
        """
        stats = {
            "model": self.model,
            "dimensions": self.dimensions,
            "batch_size": self.batch_size,
            "max_batch_size": self.MAX_BATCH_SIZE,
            "max_input_tokens": self.MAX_INPUT_TOKENS,
        }

        # Add cache statistics
        try:
            cache = get_embedding_cache()
            stats["cache"] = cache.get_stats()
        except Exception as e:
            logger.debug(f"Could not get cache stats: {e}")

        return stats

    def __repr__(self) -> str:
        """String representation of the embedder."""
        return (
            f"UnifiedEmbedder(model='{self.model}', "
            f"dimensions={self.dimensions}, "
            f"batch_size={self.batch_size})"
        )
