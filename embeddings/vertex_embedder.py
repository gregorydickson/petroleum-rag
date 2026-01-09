"""Vertex AI embedding generation using Google Cloud's text embedding models.

This module provides a Vertex AI embedder that matches the UnifiedEmbedder interface,
allowing it to be used as a drop-in replacement for OpenAI embeddings.
"""

import asyncio
import logging
from typing import Any

from google.cloud import aiplatform
from google.api_core import exceptions as google_exceptions
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


class VertexEmbedder:
    """Vertex AI embedder using Google Cloud's text embedding models.

    Provides the same interface as UnifiedEmbedder but uses Vertex AI instead of OpenAI.
    Handles embedding generation with:
    - Single text embedding
    - Batch processing with automatic splitting
    - Rate limit handling with exponential backoff
    - Retry logic for transient failures
    - Content-based caching

    Attributes:
        project: GCP project ID
        location: GCP location (us-central1, etc.)
        model: Embedding model name
        dimensions: Expected embedding dimensions
        batch_size: Maximum batch size for API calls
    """

    # Vertex AI API limits
    MAX_BATCH_SIZE = 250  # Maximum texts per API call
    MAX_INPUT_LENGTH = 20000  # Maximum characters per input

    # Available models and their dimensions
    MODELS = {
        "textembedding-gecko@003": 768,  # Latest gecko model
        "textembedding-gecko@002": 768,  # Previous gecko
        "textembedding-gecko@001": 768,  # Original gecko
        "text-embedding-preview-0409": 768,  # Preview model
        "text-multilingual-embedding-preview-0409": 768,  # Multilingual
    }

    def __init__(
        self,
        api_key: str | None = None,
        project: str | None = None,
        location: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        """Initialize the Vertex AI embedder.

        Supports two authentication methods:
        1. API Key (simpler) - Set VERTEX_API_KEY in environment
        2. Service Account (traditional) - Set GOOGLE_APPLICATION_CREDENTIALS

        Args:
            api_key: Vertex AI Studio API key (defaults to settings)
            project: GCP project ID (defaults to settings)
            location: GCP location (defaults to settings)
            model: Embedding model name (defaults to settings)
            dimensions: Expected embedding dimensions (defaults to model default)
            batch_size: Batch size for processing (defaults to settings)

        Raises:
            ValueError: If neither API key nor service account credentials provided
        """
        self.api_key = api_key or settings.vertex_api_key
        self.project = project or settings.google_cloud_project
        self.location = location or settings.vertex_docai_location or "us-central1"
        self.model = model or settings.vertex_embedding_model or "textembedding-gecko@003"

        # Get dimensions from model or settings
        if dimensions:
            self.dimensions = dimensions
        elif self.model in self.MODELS:
            self.dimensions = self.MODELS[self.model]
        else:
            # Default to gecko dimensions if unknown model
            self.dimensions = 768
            logger.warning(
                f"Unknown model '{self.model}', assuming dimensions=768"
            )

        self.batch_size = min(
            batch_size or settings.embedding_batch_size or 100,
            self.MAX_BATCH_SIZE,
        )

        # Initialize Vertex AI with API key if provided, otherwise use service account
        try:
            if self.api_key:
                # API key authentication (simpler)
                import vertexai
                vertexai.init(
                    project=self.project,
                    location=self.location,
                    api_key=self.api_key,
                )
                logger.info(
                    f"VertexEmbedder initialized with API key authentication: "
                    f"model={self.model}, location={self.location}, "
                    f"dimensions={self.dimensions}, batch_size={self.batch_size}"
                )
            else:
                # Service account authentication (traditional)
                if not self.project:
                    raise ValueError(
                        "GCP project must be provided via project parameter or "
                        "GOOGLE_CLOUD_PROJECT environment variable"
                    )
                aiplatform.init(project=self.project, location=self.location)
                logger.info(
                    f"VertexEmbedder initialized with service account: "
                    f"model={self.model}, project={self.project}, location={self.location}, "
                    f"dimensions={self.dimensions}, batch_size={self.batch_size}"
                )
        except Exception as e:
            raise ValueError(
                f"Failed to initialize Vertex AI. Make sure either:\n"
                f"1. VERTEX_API_KEY is set (Vertex AI Studio API key), OR\n"
                f"2. GOOGLE_APPLICATION_CREDENTIALS is set (service account JSON path)\n"
                f"Error: {e}"
            ) from e

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text with caching.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            ValueError: If text is empty or too long
            RuntimeError: If embedding generation fails
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        if len(text) > self.MAX_INPUT_LENGTH:
            raise ValueError(
                f"Text length ({len(text)}) exceeds maximum ({self.MAX_INPUT_LENGTH})"
            )

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

        # Validate text lengths
        too_long = [i for i, text in enumerate(texts) if len(text) > self.MAX_INPUT_LENGTH]
        if too_long:
            raise ValueError(
                f"Texts at indices {too_long[:5]} exceed maximum length {self.MAX_INPUT_LENGTH}"
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
                        batch = uncached_texts[i : i + self.batch_size]
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
                            await asyncio.sleep(0.2)

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
        retry=retry_if_exception_type(
            (
                google_exceptions.ResourceExhausted,  # Rate limit / quota
                google_exceptions.ServiceUnavailable,  # Temporary outage
                google_exceptions.DeadlineExceeded,  # Timeout
                google_exceptions.InternalServerError,  # Server error
            )
        ),
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
            google_exceptions: Various Google API exceptions (will retry)
        """
        # Acquire rate limit tokens before making API call
        if rate_limiter.is_registered("vertex"):
            await rate_limiter.acquire("vertex", tokens=len(texts))
            logger.debug(f"Acquired rate limit tokens for {len(texts)} embeddings")

        try:
            # Run synchronous Vertex AI call in thread pool
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, self._generate_embeddings_sync, texts
            )

            # Validate dimensions
            for i, embedding in enumerate(embeddings):
                if len(embedding) != self.dimensions:
                    logger.warning(
                        f"Embedding {i} has unexpected dimensions: "
                        f"{len(embedding)} vs expected {self.dimensions}"
                    )

            logger.debug(f"Generated {len(embeddings)} embeddings")
            return embeddings

        except google_exceptions.ResourceExhausted as e:
            logger.warning(f"Rate limit or quota exceeded, will retry: {e}")
            raise
        except google_exceptions.ServiceUnavailable as e:
            logger.warning(f"Service unavailable, will retry: {e}")
            raise
        except google_exceptions.DeadlineExceeded as e:
            logger.warning(f"Request timeout, will retry: {e}")
            raise
        except google_exceptions.InternalServerError as e:
            logger.warning(f"Server error, will retry: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during embedding generation: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}") from e

    def _generate_embeddings_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous wrapper for Vertex AI embedding generation.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        from vertexai.language_models import TextEmbeddingModel

        model = TextEmbeddingModel.from_pretrained(self.model)

        # Generate embeddings
        embeddings = model.get_embeddings(texts)

        # Extract values
        return [embedding.values for embedding in embeddings]

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
        """Close resources (Vertex AI doesn't require explicit cleanup)."""
        logger.debug("VertexEmbedder closed (no cleanup needed)")

    def get_stats(self) -> dict[str, Any]:
        """Get embedder statistics including cache performance.

        Returns:
            Dictionary with embedder configuration and cache stats
        """
        stats = {
            "provider": "vertex_ai",
            "model": self.model,
            "project": self.project,
            "location": self.location,
            "dimensions": self.dimensions,
            "batch_size": self.batch_size,
            "max_batch_size": self.MAX_BATCH_SIZE,
            "max_input_length": self.MAX_INPUT_LENGTH,
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
            f"VertexEmbedder(model='{self.model}', "
            f"project='{self.project}', "
            f"location='{self.location}', "
            f"dimensions={self.dimensions}, "
            f"batch_size={self.batch_size})"
        )
