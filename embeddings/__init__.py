"""Embedding generation utilities.

This module provides unified embedding generation using OpenAI's embedding models
with robust error handling, rate limiting, and batch processing capabilities.
"""

from embeddings.embedder import UnifiedEmbedder

__all__ = ["UnifiedEmbedder"]
