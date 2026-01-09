"""Weaviate storage backend with hybrid search (vector + BM25).

This module implements storage using Weaviate, leveraging its hybrid search
capabilities that combine vector similarity with BM25 keyword matching.
"""

import logging
from typing import Any

import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.exceptions import WeaviateConnectionError

from config import settings
from models import DocumentChunk, RetrievalResult
from storage.base import BaseStorage

logger = logging.getLogger(__name__)


class WeaviateStore(BaseStorage):
    """Weaviate storage implementation with hybrid search.

    Features:
    - Vector similarity search using embeddings
    - BM25 keyword search on text content
    - Hybrid search combining both methods (alpha=0.7 by default)
    - Metadata filtering and indexing

    Attributes:
        client: Weaviate client instance
        class_name: Name of the Weaviate class/collection
        alpha: Hybrid search weight (0.7 = 70% vector, 30% keyword)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize Weaviate storage backend.

        Args:
            config: Optional configuration dictionary
        """
        super().__init__("Weaviate", config)
        self.client: weaviate.WeaviateClient | None = None
        self.class_name = self.config.get("class_name", settings.weaviate_class_name)
        self.alpha = self.config.get("alpha", 0.7)  # 70% vector, 30% keyword

    async def initialize(self) -> None:
        """Initialize Weaviate connection and schema.

        This method:
        1. Establishes connection to Weaviate server
        2. Creates schema with properties for full-text indexing
        3. Configures vectorizer settings
        4. Sets up hybrid search capabilities

        Raises:
            ConnectionError: If unable to connect to Weaviate
            RuntimeError: If schema creation fails
        """
        try:
            # Connect to Weaviate
            host = self.config.get("host", settings.weaviate_host)
            port = self.config.get("port", settings.weaviate_port)
            grpc_port = self.config.get("grpc_port", settings.weaviate_grpc_port)

            self.client = weaviate.connect_to_local(
                host=host,
                port=port,
                grpc_port=grpc_port,
            )

            # Check if class already exists
            if self.client.collections.exists(self.class_name):
                # Class exists, just get reference
                collection = self.client.collections.get(self.class_name)
                self._initialized = True
                return

            # Create schema with properties for full-text indexing
            # Note: Weaviate v4 uses collections instead of classes
            collection = self.client.collections.create(
                name=self.class_name,
                description="Petroleum engineering document chunks for RAG retrieval",
                vectorizer_config=Configure.Vectorizer.none(),  # We provide vectors
                properties=[
                    Property(
                        name="chunk_id",
                        data_type=DataType.TEXT,
                        description="Unique chunk identifier",
                        skip_vectorization=True,
                    ),
                    Property(
                        name="document_id",
                        data_type=DataType.TEXT,
                        description="Parent document identifier",
                        skip_vectorization=True,
                    ),
                    Property(
                        name="content",
                        data_type=DataType.TEXT,
                        description="Chunk text content",
                        skip_vectorization=False,  # Enable BM25 indexing
                    ),
                    Property(
                        name="element_ids",
                        data_type=DataType.TEXT_ARRAY,
                        description="List of element IDs in this chunk",
                        skip_vectorization=True,
                    ),
                    Property(
                        name="chunk_index",
                        data_type=DataType.INT,
                        description="Sequential index within document",
                        skip_vectorization=True,
                    ),
                    Property(
                        name="start_page",
                        data_type=DataType.INT,
                        description="Starting page number",
                        skip_vectorization=True,
                    ),
                    Property(
                        name="end_page",
                        data_type=DataType.INT,
                        description="Ending page number",
                        skip_vectorization=True,
                    ),
                    Property(
                        name="token_count",
                        data_type=DataType.INT,
                        description="Approximate token count",
                        skip_vectorization=True,
                    ),
                    Property(
                        name="parent_section",
                        data_type=DataType.TEXT,
                        description="Parent section identifier",
                        skip_vectorization=True,
                    ),
                    Property(
                        name="metadata",
                        data_type=DataType.TEXT,
                        description="JSON-encoded metadata",
                        skip_vectorization=True,
                    ),
                ],
            )

            self._initialized = True

        except WeaviateConnectionError as e:
            raise ConnectionError(f"Failed to connect to Weaviate: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Weaviate schema: {e}") from e

    async def store_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Store document chunks with embeddings in Weaviate.

        This method:
        1. Validates chunks and embeddings match
        2. Batch inserts chunks with vector embeddings
        3. Indexes metadata for filtering
        4. Enables full-text search on content

        Args:
            chunks: List of DocumentChunk objects to store
            embeddings: List of embedding vectors (one per chunk)

        Raises:
            ValueError: If chunks and embeddings length mismatch
            RuntimeError: If storage operation fails
        """
        if not self._initialized or self.client is None:
            raise RuntimeError("Weaviate storage not initialized. Call initialize() first.")

        # Validate inputs
        self.validate_chunks_embeddings(chunks, embeddings)

        try:
            # Get collection
            collection = self.client.collections.get(self.class_name)

            # Batch insert chunks
            with collection.batch.dynamic() as batch:
                for chunk, embedding in zip(chunks, embeddings):
                    # Convert metadata dict to JSON string
                    import json

                    metadata_json = json.dumps(chunk.metadata)

                    # Prepare properties
                    properties = {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "content": chunk.content,
                        "element_ids": chunk.element_ids,
                        "chunk_index": chunk.chunk_index,
                        "metadata": metadata_json,
                    }

                    # Add optional properties if they exist
                    if chunk.start_page is not None:
                        properties["start_page"] = chunk.start_page
                    if chunk.end_page is not None:
                        properties["end_page"] = chunk.end_page
                    if chunk.token_count is not None:
                        properties["token_count"] = chunk.token_count
                    if chunk.parent_section is not None:
                        properties["parent_section"] = chunk.parent_section

                    # Add object with vector
                    batch.add_object(
                        properties=properties,
                        vector=embedding,
                    )

        except Exception as e:
            raise RuntimeError(f"Failed to store chunks in Weaviate: {e}") from e

    async def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve relevant chunks using hybrid search.

        This method implements Weaviate's hybrid search combining:
        - Vector similarity search (using query_embedding)
        - BM25 keyword search (using query text)
        - Alpha parameter controls the blend (0.7 = 70% vector, 30% keyword)

        Args:
            query: Query text for keyword/BM25 search
            query_embedding: Query vector embedding for similarity search
            top_k: Number of results to return
            filters: Optional metadata filters (e.g., {"document_id": "doc123"})

        Returns:
            List of RetrievalResult objects, ranked by relevance

        Raises:
            RuntimeError: If retrieval fails
        """
        if not self._initialized or self.client is None:
            raise RuntimeError("Weaviate storage not initialized. Call initialize() first.")

        try:
            # Get collection
            collection = self.client.collections.get(self.class_name)

            # Build filter if provided
            where_filter = None
            if filters:
                # Build Weaviate filter from dict
                # For now, support simple equality filters
                if "document_id" in filters:
                    where_filter = Filter.by_property("document_id").equal(
                        filters["document_id"]
                    )

            # Perform hybrid search
            response = collection.query.hybrid(
                query=query,
                vector=query_embedding,
                alpha=self.alpha,  # 0.7 = 70% vector, 30% keyword
                limit=top_k,
                filters=where_filter,
                return_metadata=MetadataQuery(score=True, explain_score=True),
            )

            # Convert to RetrievalResult format
            results = []
            for rank, obj in enumerate(response.objects, start=1):
                # Parse metadata JSON
                import json

                metadata = json.loads(obj.properties.get("metadata", "{}"))

                result = RetrievalResult(
                    chunk_id=obj.properties["chunk_id"],
                    document_id=obj.properties["document_id"],
                    content=obj.properties["content"],
                    score=obj.metadata.score if obj.metadata.score is not None else 0.0,
                    metadata=metadata,
                    rank=rank,
                    retrieval_method="hybrid",
                )
                results.append(result)

            # Filter by minimum score if configured
            min_score = self.get_min_score()
            if min_score > 0.0:
                results = [r for r in results if r.score >= min_score]

            return results

        except Exception as e:
            raise RuntimeError(f"Failed to retrieve from Weaviate: {e}") from e

    async def clear(self) -> None:
        """Clear all data from Weaviate storage.

        This method removes all objects from the collection but keeps the schema.

        Raises:
            RuntimeError: If clear operation fails
        """
        if not self._initialized or self.client is None:
            raise RuntimeError("Weaviate storage not initialized. Call initialize() first.")

        try:
            # Get collection
            collection = self.client.collections.get(self.class_name)

            # Delete all objects in collection
            collection.data.delete_many(
                where=Filter.by_property("chunk_id").like("*")  # Match all
            )

        except Exception as e:
            raise RuntimeError(f"Failed to clear Weaviate collection: {e}") from e

    async def health_check(self) -> bool:
        """Check if Weaviate is healthy and responsive.

        Returns:
            True if Weaviate is healthy, False otherwise
        """
        try:
            if not self._initialized or self.client is None:
                return False

            # Check if client is ready
            return self.client.is_ready()

        except Exception:
            return False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.close()

    async def close(self) -> None:
        """Cleanup Weaviate client connection."""
        if self.client is not None:
            try:
                self.client.close()
                self.client = None
                self._initialized = False
                logger.info("Weaviate client closed")
            except Exception as e:
                logger.warning(f"Error closing Weaviate client: {e}")

    def __del__(self) -> None:
        """Clean up Weaviate connection on deletion."""
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
