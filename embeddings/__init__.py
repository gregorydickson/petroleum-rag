"""Embedding generation utilities.

This module provides unified embedding generation using multiple providers
(OpenAI, Vertex AI) with robust error handling, rate limiting, and batch
processing capabilities.
"""

from embeddings.embedder import UnifiedEmbedder
from embeddings.vertex_embedder import VertexEmbedder


def get_embedder(provider: str | None = None):
    """Factory function to get the appropriate embedder based on provider.

    Args:
        provider: Embedding provider ('openai' or 'vertex').
                 If None, uses settings.embedding_provider.

    Returns:
        UnifiedEmbedder or VertexEmbedder instance

    Raises:
        ValueError: If provider is not supported
    """
    from config import settings

    provider = provider or settings.embedding_provider

    if provider == "openai":
        return UnifiedEmbedder()
    elif provider == "vertex":
        return VertexEmbedder()
    else:
        raise ValueError(
            f"Unsupported embedding provider: {provider}. "
            f"Supported providers: 'openai', 'vertex'"
        )


__all__ = ["UnifiedEmbedder", "VertexEmbedder", "get_embedder"]
