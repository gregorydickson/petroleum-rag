"""ChromaDB storage implementation for pure vector similarity search.

This module implements the ChromaDB storage backend, which uses pure vector
similarity search (cosine distance) for retrieval without hybrid or graph features.
"""

import logging
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from models import DocumentChunk, RetrievalResult
from storage.base import BaseStorage

logger = logging.getLogger(__name__)


class ChromaStore(BaseStorage):
    """ChromaDB storage implementation using pure vector similarity search.

    ChromaDB provides efficient vector similarity search with metadata filtering.
    This implementation uses cosine distance for similarity scoring.

    Attributes:
        client: ChromaDB client instance
        collection: ChromaDB collection for document chunks
        collection_name: Name of the collection
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize ChromaDB storage.

        Args:
            config: Configuration dictionary with:
                - host: ChromaDB host (default: localhost)
                - port: ChromaDB port (default: 8000)
                - collection_name: Collection name (default: petroleum_docs)
                - top_k: Default number of results (default: 5)
                - min_score: Minimum similarity score (default: 0.0)
        """
        super().__init__("Chroma", config)

        self.host = self.config.get("host", "localhost")
        self.port = self.config.get("port", 8000)
        self.collection_name = self.config.get("collection_name", "petroleum_docs")

        self.client: chromadb.ClientAPI | None = None
        self.collection: chromadb.Collection | None = None

        logger.info(
            f"Initialized ChromaStore with host={self.host}:{self.port}, "
            f"collection={self.collection_name}"
        )

    async def initialize(self) -> None:
        """Initialize ChromaDB client and collection.

        This method:
        1. Creates a connection to the ChromaDB server
        2. Creates or retrieves the collection
        3. Configures the collection with cosine distance metric

        Raises:
            ConnectionError: If unable to connect to ChromaDB
            RuntimeError: If initialization fails
        """
        if self._initialized:
            logger.debug("ChromaStore already initialized")
            return

        try:
            # Create ChromaDB client
            self.client = chromadb.HttpClient(
                host=self.host,
                port=self.port,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )

            # Verify connection by checking heartbeat
            heartbeat = self.client.heartbeat()
            logger.debug(f"ChromaDB heartbeat: {heartbeat}")

            # Get or create collection
            # ChromaDB uses cosine distance by default which is ideal for embeddings
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},  # Use cosine similarity
            )

            self._initialized = True
            logger.info(
                f"ChromaDB initialized successfully. "
                f"Collection: {self.collection_name}, "
                f"Document count: {self.collection.count()}"
            )

        except Exception as e:
            error_msg = f"Failed to initialize ChromaDB: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e

    async def store_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Store document chunks with their embeddings in ChromaDB.

        This method:
        1. Validates chunks and embeddings match
        2. Prepares batch data for ChromaDB
        3. Stores chunks with embeddings and metadata
        4. Handles large batches efficiently

        Args:
            chunks: List of DocumentChunk objects to store
            embeddings: List of embedding vectors (one per chunk)

        Raises:
            ValueError: If chunks and embeddings length mismatch
            RuntimeError: If storage operation fails
        """
        if not self._initialized or self.collection is None:
            raise RuntimeError("ChromaStore not initialized. Call initialize() first.")

        # Validate input
        self.validate_chunks_embeddings(chunks, embeddings)

        try:
            # Prepare batch data for ChromaDB
            ids = [chunk.chunk_id for chunk in chunks]
            documents = [chunk.content for chunk in chunks]

            # Prepare metadata - ChromaDB requires all metadata values to be
            # strings, integers, floats, or booleans
            metadatas = []
            for chunk in chunks:
                metadata = {
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                }

                # Add optional fields if present
                if chunk.start_page is not None:
                    metadata["start_page"] = chunk.start_page
                if chunk.end_page is not None:
                    metadata["end_page"] = chunk.end_page
                if chunk.parent_section is not None:
                    metadata["parent_section"] = chunk.parent_section
                if chunk.token_count is not None:
                    metadata["token_count"] = chunk.token_count

                # Store element_ids as comma-separated string
                if chunk.element_ids:
                    metadata["element_ids"] = ",".join(chunk.element_ids)

                # Add any additional metadata from chunk.metadata
                for key, value in chunk.metadata.items():
                    if key not in metadata:  # Don't override existing keys
                        metadata[key] = str(value)  # Ensure string type

                metadatas.append(metadata)

            # Add to collection in batches (ChromaDB has internal batching,
            # but we log progress for large datasets)
            batch_size = 1000
            total_chunks = len(chunks)

            for i in range(0, total_chunks, batch_size):
                end_idx = min(i + batch_size, total_chunks)
                batch_ids = ids[i:end_idx]
                batch_embeddings = embeddings[i:end_idx]
                batch_documents = documents[i:end_idx]
                batch_metadatas = metadatas[i:end_idx]

                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                )

                logger.debug(
                    f"Stored batch {i // batch_size + 1}: "
                    f"chunks {i + 1}-{end_idx} of {total_chunks}"
                )

            logger.info(
                f"Successfully stored {total_chunks} chunks in ChromaDB collection "
                f"'{self.collection_name}'"
            )

        except Exception as e:
            error_msg = f"Failed to store chunks in ChromaDB: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    async def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve relevant chunks using vector similarity search.

        This method:
        1. Performs vector similarity search using query embedding
        2. Applies metadata filters if provided
        3. Returns top_k most similar chunks
        4. Maps results to RetrievalResult format with scores

        ChromaDB uses cosine similarity by default. Distances are converted
        to similarity scores (1 - distance) for consistency.

        Args:
            query: Query text (stored for reference, not used in pure vector search)
            query_embedding: Query vector embedding
            top_k: Number of results to return (default: 5)
            filters: Optional metadata filters (ChromaDB where clause format)

        Returns:
            List of RetrievalResult objects, ranked by similarity score

        Raises:
            RuntimeError: If retrieval fails
        """
        if not self._initialized or self.collection is None:
            raise RuntimeError("ChromaStore not initialized. Call initialize() first.")

        try:
            # Prepare where clause for filtering if provided
            where_clause = None
            if filters:
                where_clause = self._prepare_where_clause(filters)

            # Query ChromaDB collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )

            # Map ChromaDB results to RetrievalResult format
            retrieval_results = []

            # ChromaDB returns results as lists of lists
            if not results["ids"] or not results["ids"][0]:
                logger.info("No results found for query")
                return retrieval_results

            chunk_ids = results["ids"][0]
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            min_score = self.get_min_score()

            for rank, (chunk_id, document, metadata, distance) in enumerate(
                zip(chunk_ids, documents, metadatas, distances), start=1
            ):
                # Convert distance to similarity score
                # ChromaDB cosine distance is in range [0, 2]
                # Convert to similarity: 1 - (distance / 2) gives range [0, 1]
                similarity_score = 1.0 - (distance / 2.0)

                # Apply minimum score threshold
                if similarity_score < min_score:
                    logger.debug(
                        f"Skipping result with score {similarity_score:.3f} "
                        f"(below threshold {min_score})"
                    )
                    continue

                # Create RetrievalResult
                retrieval_result = RetrievalResult(
                    chunk_id=chunk_id,
                    document_id=metadata.get("document_id", "unknown"),
                    content=document,
                    score=similarity_score,
                    metadata=self._clean_metadata(metadata),
                    rank=rank,
                    retrieval_method="vector",
                )

                retrieval_results.append(retrieval_result)

            logger.info(
                f"Retrieved {len(retrieval_results)} results "
                f"(from {len(chunk_ids)} candidates) for query"
            )

            return retrieval_results

        except Exception as e:
            error_msg = f"Failed to retrieve from ChromaDB: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    async def clear(self) -> None:
        """Clear all data from ChromaDB collection.

        This method:
        1. Deletes the existing collection
        2. Recreates an empty collection with same configuration
        3. Resets initialization state

        Use with caution - this is destructive!

        Raises:
            RuntimeError: If clear operation fails
        """
        if not self._initialized or self.client is None:
            logger.warning("ChromaStore not initialized, nothing to clear")
            return

        try:
            # Delete the collection
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted ChromaDB collection '{self.collection_name}'")

            # Recreate empty collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            logger.info(f"Recreated empty ChromaDB collection '{self.collection_name}'")

        except Exception as e:
            error_msg = f"Failed to clear ChromaDB collection: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    async def health_check(self) -> bool:
        """Check if ChromaDB is healthy and responsive.

        Returns:
            True if ChromaDB is accessible and collection exists, False otherwise
        """
        if not self._initialized or self.client is None:
            return False

        try:
            # Check heartbeat
            self.client.heartbeat()

            # Verify collection exists
            if self.collection is not None:
                self.collection.count()
                return True

            return False

        except Exception as e:
            logger.warning(f"ChromaDB health check failed: {e}")
            return False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.close()

    async def close(self) -> None:
        """Cleanup Chroma client resources."""
        if self.client is not None:
            try:
                # ChromaDB HttpClient doesn't have an explicit close method
                # but we can clear our references
                self.collection = None
                self.client = None
                self._initialized = False
                logger.debug("ChromaDB client cleaned up")
            except Exception as e:
                logger.warning(f"Error cleaning up ChromaDB client: {e}")

    def _prepare_where_clause(self, filters: dict[str, Any]) -> dict[str, Any]:
        """Prepare ChromaDB where clause from filters.

        ChromaDB uses a specific where clause format:
        - Simple equality: {"field": "value"}
        - Multiple conditions: {"$and": [{"field1": "value1"}, {"field2": "value2"}]}
        - Operators: {"field": {"$gt": value}}

        Args:
            filters: Filter dictionary

        Returns:
            ChromaDB-compatible where clause
        """
        # For now, support simple equality filters
        # Can be extended to support more complex queries
        where_clause = {}

        for key, value in filters.items():
            if isinstance(value, (str, int, float, bool)):
                where_clause[key] = value
            elif isinstance(value, dict):
                # Pass through operator dictionaries
                where_clause[key] = value

        return where_clause if where_clause else None

    def _clean_metadata(self, metadata: dict[str, Any]) -> dict[str, str]:
        """Clean and convert metadata to string format for RetrievalResult.

        Args:
            metadata: Raw metadata from ChromaDB

        Returns:
            Cleaned metadata dictionary with string values
        """
        cleaned = {}

        for key, value in metadata.items():
            # Skip internal ChromaDB fields
            if key.startswith("_"):
                continue

            # Convert to string
            cleaned[key] = str(value)

        return cleaned

    def __repr__(self) -> str:
        """String representation of ChromaStore."""
        return (
            f"ChromaStore(host='{self.host}', port={self.port}, "
            f"collection='{self.collection_name}', initialized={self._initialized})"
        )
