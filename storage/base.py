"""Base storage interface for vector/hybrid/graph backends.

All storage implementations must inherit from BaseStorage and implement the
abstract methods for initialization, storage, and retrieval.
"""

from abc import ABC, abstractmethod
from typing import Any

from models import DocumentChunk, RetrievalResult


class BaseStorage(ABC):
    """Abstract base class for storage backends.

    All storage implementations must inherit from this class and implement
    the abstract methods for initialization, storing chunks, and retrieval.

    Attributes:
        name: Human-readable name of the storage backend
        config: Optional configuration dictionary
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        """Initialize the storage backend.

        Args:
            name: Name of the storage backend (e.g., "Chroma", "Weaviate", "FalkorDB")
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage backend.

        This method should:
        1. Establish connection to the storage service
        2. Create necessary collections/schemas/graphs
        3. Verify connectivity and permissions
        4. Set up any required indices

        Should be idempotent - safe to call multiple times.

        Raises:
            ConnectionError: If unable to connect to storage service
            RuntimeError: If initialization fails
        """
        pass

    @abstractmethod
    async def store_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Store document chunks with their embeddings.

        This method should:
        1. Validate chunks and embeddings match in count
        2. Store chunks with their vector embeddings
        3. Index metadata for filtering
        4. Handle batch operations efficiently

        Args:
            chunks: List of DocumentChunk objects to store
            embeddings: List of embedding vectors (one per chunk)

        Raises:
            ValueError: If chunks and embeddings length mismatch
            RuntimeError: If storage operation fails
        """
        pass

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve relevant chunks for a query.

        This method should:
        1. Use query_embedding for vector similarity search
        2. Apply any additional storage-specific retrieval (BM25, graph traversal)
        3. Apply metadata filters if provided
        4. Return top_k most relevant results
        5. Include relevance scores

        Different storage backends may use different retrieval methods:
        - Chroma: Pure vector similarity
        - Weaviate: Hybrid (vector + BM25)
        - FalkorDB: Vector + graph traversal

        Args:
            query: Query text (for keyword/hybrid search)
            query_embedding: Query vector embedding
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of RetrievalResult objects, ranked by relevance

        Raises:
            RuntimeError: If retrieval fails
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all data from the storage backend.

        This method should:
        1. Remove all stored chunks and embeddings
        2. Reset indices
        3. Prepare for fresh data

        Use with caution - this is destructive!

        Raises:
            RuntimeError: If clear operation fails
        """
        pass

    async def health_check(self) -> bool:
        """Check if storage backend is healthy and responsive.

        Returns:
            True if backend is healthy, False otherwise
        """
        try:
            # Subclasses can override with specific health checks
            return self._initialized
        except Exception:
            return False

    def get_top_k(self) -> int:
        """Get configured top_k value for retrieval.

        Returns:
            Top K from config or default value
        """
        return self.config.get("top_k", 5)

    def get_min_score(self) -> float:
        """Get minimum relevance score threshold.

        Returns:
            Minimum score from config or default value
        """
        return self.config.get("min_score", 0.0)

    def validate_chunks_embeddings(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Validate that chunks and embeddings match.

        Args:
            chunks: List of chunks
            embeddings: List of embeddings

        Raises:
            ValueError: If validation fails
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks and embeddings length mismatch: "
                f"{len(chunks)} chunks vs {len(embeddings)} embeddings"
            )

        if not chunks:
            raise ValueError("Cannot store empty chunks list")

        if embeddings and not all(isinstance(e, list) for e in embeddings):
            raise ValueError("All embeddings must be lists of floats")

    def __repr__(self) -> str:
        """String representation of the storage backend."""
        return f"{self.__class__.__name__}(name='{self.name}')"
