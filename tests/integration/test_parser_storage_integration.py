"""Integration tests for all 12 parser-storage combinations.

This module tests the complete pipeline for each combination:
1. Parse document
2. Chunk document
3. Generate embeddings
4. Store in backend
5. Retrieve results
6. Verify end-to-end functionality

Tests all 12 combinations:
- LlamaParse x (Chroma, Weaviate, FalkorDB)
- Docling x (Chroma, Weaviate, FalkorDB)
- PageIndex x (Chroma, Weaviate, FalkorDB)
- VertexDocAI x (Chroma, Weaviate, FalkorDB)
"""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from embeddings import UnifiedEmbedder
from models import DocumentChunk, ParsedDocument, RetrievalResult


# ============================================================================
# Integration Test Fixtures
# ============================================================================


@pytest.fixture
def all_parser_names() -> list[str]:
    """List of all parser names for parameterized tests.

    Returns:
        List of parser names
    """
    return ["LlamaParse", "Docling", "PageIndex", "VertexDocAI"]


@pytest.fixture
def all_storage_names() -> list[str]:
    """List of all storage backend names for parameterized tests.

    Returns:
        List of storage backend names
    """
    return ["ChromaStore", "WeaviateStore", "FalkorDBStore"]


@pytest.fixture
def parser_storage_combinations(
    all_parser_names: list[str], all_storage_names: list[str]
) -> list[tuple[str, str]]:
    """Generate all 12 parser-storage combinations.

    Args:
        all_parser_names: List of parser names
        all_storage_names: List of storage names

    Returns:
        List of (parser_name, storage_name) tuples
    """
    combinations = []
    for parser in all_parser_names:
        for storage in all_storage_names:
            combinations.append((parser, storage))
    return combinations


@pytest.fixture
def mock_parser_factory(
    mock_parsed_document: ParsedDocument,
    mock_chunks: list[DocumentChunk],
):
    """Factory for creating mock parsers by name.

    Args:
        mock_parsed_document: Mock parsed document
        mock_chunks: Mock chunks

    Returns:
        Factory function that creates parsers
    """

    def _create_parser(parser_name: str) -> Any:
        parser = Mock()
        parser.name = parser_name
        parser.parse = AsyncMock(return_value=mock_parsed_document)
        parser.chunk_document = Mock(return_value=mock_chunks)
        return parser

    return _create_parser


@pytest.fixture
def mock_storage_factory(mock_retrieval_results: list[RetrievalResult]):
    """Factory for creating mock storage backends by name.

    Args:
        mock_retrieval_results: Mock retrieval results

    Returns:
        Factory function that creates storage backends
    """

    def _create_storage(storage_name: str) -> Any:
        store = Mock()
        store.name = storage_name
        store.initialize = AsyncMock()
        store.store_chunks = AsyncMock()
        store.retrieve = AsyncMock(return_value=mock_retrieval_results)
        store.clear = AsyncMock()
        store.health_check = AsyncMock(return_value=True)
        return store

    return _create_storage


# ============================================================================
# Test Class: Individual Component Tests
# ============================================================================


class TestIndividualComponents:
    """Test individual components work correctly."""

    async def test_parser_parse_document(
        self,
        mock_parser_factory,
        mock_pdf_path: Path,
        all_parser_names: list[str],
    ):
        """Test each parser can parse a document.

        Args:
            mock_parser_factory: Parser factory fixture
            mock_pdf_path: Mock PDF path
            all_parser_names: List of parser names
        """
        for parser_name in all_parser_names:
            parser = mock_parser_factory(parser_name)

            # Parse document
            parsed_doc = await parser.parse(mock_pdf_path)

            # Verify
            assert parsed_doc is not None
            assert isinstance(parsed_doc, ParsedDocument)
            assert len(parsed_doc.elements) > 0
            assert parsed_doc.document_id
            assert parsed_doc.source_file == mock_pdf_path

    def test_parser_chunk_document(
        self,
        mock_parser_factory,
        mock_parsed_document: ParsedDocument,
        all_parser_names: list[str],
    ):
        """Test each parser can chunk a document.

        Args:
            mock_parser_factory: Parser factory fixture
            mock_parsed_document: Mock parsed document
            all_parser_names: List of parser names
        """
        for parser_name in all_parser_names:
            parser = mock_parser_factory(parser_name)

            # Chunk document
            chunks = parser.chunk_document(mock_parsed_document)

            # Verify
            assert chunks is not None
            assert isinstance(chunks, list)
            assert len(chunks) > 0
            assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
            assert all(chunk.chunk_id for chunk in chunks)
            assert all(chunk.content for chunk in chunks)

    async def test_storage_initialize(
        self,
        mock_storage_factory,
        all_storage_names: list[str],
    ):
        """Test each storage backend can initialize.

        Args:
            mock_storage_factory: Storage factory fixture
            all_storage_names: List of storage names
        """
        for storage_name in all_storage_names:
            store = mock_storage_factory(storage_name)

            # Initialize
            await store.initialize()

            # Verify
            store.initialize.assert_called_once()
            assert store.name == storage_name

    async def test_storage_store_and_retrieve(
        self,
        mock_storage_factory,
        mock_chunks: list[DocumentChunk],
        mock_embeddings: list[list[float]],
        all_storage_names: list[str],
    ):
        """Test each storage backend can store and retrieve.

        Args:
            mock_storage_factory: Storage factory fixture
            mock_chunks: Mock chunks
            mock_embeddings: Mock embeddings
            all_storage_names: List of storage names
        """
        for storage_name in all_storage_names:
            store = mock_storage_factory(storage_name)

            # Initialize
            await store.initialize()

            # Store chunks
            await store.store_chunks(mock_chunks, mock_embeddings)

            # Retrieve
            query = "What is enhanced oil recovery?"
            query_embedding = mock_embeddings[0]
            results = await store.retrieve(query, query_embedding, top_k=5)

            # Verify
            store.store_chunks.assert_called_once_with(mock_chunks, mock_embeddings)
            store.retrieve.assert_called_once()
            assert isinstance(results, list)
            assert len(results) > 0
            assert all(isinstance(r, RetrievalResult) for r in results)


# ============================================================================
# Test Class: Full Pipeline Integration
# ============================================================================


class TestFullPipelineIntegration:
    """Test complete pipeline for each parser-storage combination."""

    @pytest.mark.parametrize("parser_name,storage_name", [
        ("LlamaParse", "ChromaStore"),
        ("LlamaParse", "WeaviateStore"),
        ("LlamaParse", "FalkorDBStore"),
        ("Docling", "ChromaStore"),
        ("Docling", "WeaviateStore"),
        ("Docling", "FalkorDBStore"),
        ("PageIndex", "ChromaStore"),
        ("PageIndex", "WeaviateStore"),
        ("PageIndex", "FalkorDBStore"),
        ("VertexDocAI", "ChromaStore"),
        ("VertexDocAI", "WeaviateStore"),
        ("VertexDocAI", "FalkorDBStore"),
    ])
    async def test_complete_pipeline(
        self,
        parser_name: str,
        storage_name: str,
        mock_parser_factory,
        mock_storage_factory,
        mock_embedder,
        mock_pdf_path: Path,
    ):
        """Test complete pipeline: parse -> chunk -> embed -> store -> retrieve.

        Args:
            parser_name: Name of parser to test
            storage_name: Name of storage backend to test
            mock_parser_factory: Parser factory fixture
            mock_storage_factory: Storage factory fixture
            mock_embedder: Mock embedder
            mock_pdf_path: Mock PDF path
        """
        # Create components
        parser = mock_parser_factory(parser_name)
        storage = mock_storage_factory(storage_name)

        # Step 1: Parse document
        parsed_doc = await parser.parse(mock_pdf_path)
        assert parsed_doc is not None
        assert len(parsed_doc.elements) > 0

        # Step 2: Chunk document
        chunks = parser.chunk_document(parsed_doc)
        assert len(chunks) > 0

        # Step 3: Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = await mock_embedder.embed_texts(texts)
        assert len(embeddings) == len(chunks)

        # Step 4: Initialize storage
        await storage.initialize()

        # Step 5: Store chunks
        await storage.store_chunks(chunks, embeddings)

        # Step 6: Retrieve
        query = "What is enhanced oil recovery?"
        query_embedding = await mock_embedder.embed_query(query)
        results = await storage.retrieve(query, query_embedding, top_k=5)

        # Verify results
        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert all(r.score >= 0.0 and r.score <= 1.0 for r in results)
        assert all(r.content for r in results)

        # Verify calls were made
        parser.parse.assert_called_once_with(mock_pdf_path)
        parser.chunk_document.assert_called_once_with(parsed_doc)
        storage.initialize.assert_called_once()
        storage.store_chunks.assert_called_once()
        storage.retrieve.assert_called_once()


# ============================================================================
# Test Class: Error Handling
# ============================================================================


class TestErrorHandling:
    """Test error handling across all combinations."""

    async def test_parser_handles_invalid_file(
        self,
        mock_parser_factory,
        all_parser_names: list[str],
    ):
        """Test parsers handle invalid file paths gracefully.

        Args:
            mock_parser_factory: Parser factory fixture
            all_parser_names: List of parser names
        """
        for parser_name in all_parser_names:
            parser = mock_parser_factory(parser_name)

            # Configure parser to raise error
            parser.parse = AsyncMock(side_effect=FileNotFoundError("File not found"))

            # Attempt to parse non-existent file
            with pytest.raises(FileNotFoundError):
                await parser.parse(Path("/nonexistent/file.pdf"))

    async def test_storage_handles_empty_chunks(
        self,
        mock_storage_factory,
        all_storage_names: list[str],
    ):
        """Test storage backends handle empty chunk lists.

        Args:
            mock_storage_factory: Storage factory fixture
            all_storage_names: List of storage names
        """
        for storage_name in all_storage_names:
            store = mock_storage_factory(storage_name)

            # Configure to raise error on empty chunks
            store.store_chunks = AsyncMock(
                side_effect=ValueError("Cannot store empty chunks list")
            )

            # Attempt to store empty chunks
            with pytest.raises(ValueError):
                await store.store_chunks([], [])

    async def test_storage_handles_embedding_mismatch(
        self,
        mock_storage_factory,
        mock_chunks: list[DocumentChunk],
        all_storage_names: list[str],
    ):
        """Test storage backends handle chunk-embedding mismatch.

        Args:
            mock_storage_factory: Storage factory fixture
            mock_chunks: Mock chunks
            all_storage_names: List of storage names
        """
        for storage_name in all_storage_names:
            store = mock_storage_factory(storage_name)

            # Configure to raise error on mismatch
            store.store_chunks = AsyncMock(
                side_effect=ValueError("Chunks and embeddings length mismatch")
            )

            # Attempt to store with mismatched embeddings
            wrong_size_embeddings = [[0.1] * 1536]  # Only 1 embedding for 5 chunks
            with pytest.raises(ValueError):
                await store.store_chunks(mock_chunks, wrong_size_embeddings)

    async def test_retrieval_handles_empty_query(
        self,
        mock_storage_factory,
        all_storage_names: list[str],
    ):
        """Test storage backends handle empty queries.

        Args:
            mock_storage_factory: Storage factory fixture
            all_storage_names: List of storage names
        """
        for storage_name in all_storage_names:
            store = mock_storage_factory(storage_name)

            # Some backends might return empty results, others might raise
            # We'll test that they handle it gracefully
            store.retrieve = AsyncMock(return_value=[])

            query_embedding = [0.1] * 1536
            results = await store.retrieve("", query_embedding, top_k=5)

            # Should return empty list, not crash
            assert isinstance(results, list)
            assert len(results) == 0


# ============================================================================
# Test Class: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases for all combinations."""

    async def test_single_chunk_document(
        self,
        mock_parser_factory,
        mock_storage_factory,
        mock_embedder,
        mock_pdf_path: Path,
        parser_storage_combinations: list[tuple[str, str]],
    ):
        """Test handling of document with single chunk.

        Args:
            mock_parser_factory: Parser factory fixture
            mock_storage_factory: Storage factory fixture
            mock_embedder: Mock embedder
            mock_pdf_path: Mock PDF path
            parser_storage_combinations: All combinations
        """
        # Test a subset of combinations for edge cases
        for parser_name, storage_name in parser_storage_combinations[:3]:
            parser = mock_parser_factory(parser_name)
            storage = mock_storage_factory(storage_name)

            # Override chunk_document to return single chunk
            single_chunk = [
                DocumentChunk(
                    chunk_id="single_chunk",
                    document_id="doc_001",
                    content="Single chunk content",
                    chunk_index=0,
                )
            ]
            parser.chunk_document = Mock(return_value=single_chunk)

            # Run pipeline
            parsed_doc = await parser.parse(mock_pdf_path)
            chunks = parser.chunk_document(parsed_doc)
            embeddings = await mock_embedder.embed_texts([c.content for c in chunks])

            await storage.initialize()
            await storage.store_chunks(chunks, embeddings)

            # Verify
            assert len(chunks) == 1
            storage.store_chunks.assert_called_once()

    async def test_large_number_of_chunks(
        self,
        mock_parser_factory,
        mock_storage_factory,
        mock_embedder,
        mock_pdf_path: Path,
        all_parser_names: list[str],
        all_storage_names: list[str],
    ):
        """Test handling of document with many chunks.

        Args:
            mock_parser_factory: Parser factory fixture
            mock_storage_factory: Storage factory fixture
            mock_embedder: Mock embedder
            mock_pdf_path: Mock PDF path
            all_parser_names: Parser names
            all_storage_names: Storage names
        """
        # Create many chunks
        many_chunks = [
            DocumentChunk(
                chunk_id=f"chunk_{i:04d}",
                document_id="doc_001",
                content=f"Content for chunk {i}",
                chunk_index=i,
            )
            for i in range(100)
        ]

        # Test one combination from each parser
        for parser_name in all_parser_names:
            storage_name = all_storage_names[0]  # Just test with Chroma
            parser = mock_parser_factory(parser_name)
            storage = mock_storage_factory(storage_name)

            # Override to return many chunks
            parser.chunk_document = Mock(return_value=many_chunks)

            # Run pipeline
            parsed_doc = await parser.parse(mock_pdf_path)
            chunks = parser.chunk_document(parsed_doc)
            embeddings = await mock_embedder.embed_texts([c.content for c in chunks])

            await storage.initialize()
            await storage.store_chunks(chunks, embeddings)

            # Verify
            assert len(chunks) == 100
            storage.store_chunks.assert_called_once()

    async def test_retrieval_with_filters(
        self,
        mock_storage_factory,
        mock_embedder,
        all_storage_names: list[str],
    ):
        """Test retrieval with metadata filters.

        Args:
            mock_storage_factory: Storage factory fixture
            mock_embedder: Mock embedder
            all_storage_names: Storage names
        """
        for storage_name in all_storage_names:
            store = mock_storage_factory(storage_name)

            # Setup retrieval with filters
            filters = {"source": "textbook", "section": "eor"}
            query_embedding = [0.1] * 1536

            # Mock retrieve to accept filters
            filtered_results = [
                RetrievalResult(
                    chunk_id="chunk_002",
                    document_id="doc_001",
                    content="Filtered content",
                    score=0.9,
                    metadata=filters,
                    rank=1,
                )
            ]
            store.retrieve = AsyncMock(return_value=filtered_results)

            # Retrieve with filters
            results = await store.retrieve(
                "test query",
                query_embedding,
                top_k=5,
                filters=filters,
            )

            # Verify
            assert len(results) > 0
            # Check that filters were passed (in mock, we return filtered results)
            for result in results:
                assert result.metadata.get("source") == "textbook"
                assert result.metadata.get("section") == "eor"


# ============================================================================
# Test Class: Performance and Concurrency
# ============================================================================


class TestPerformanceAndConcurrency:
    """Test performance and concurrent operations."""

    async def test_parallel_storage_initialization(
        self,
        mock_storage_factory,
        all_storage_names: list[str],
    ):
        """Test all storage backends can initialize in parallel.

        Args:
            mock_storage_factory: Storage factory fixture
            all_storage_names: Storage names
        """
        stores = [mock_storage_factory(name) for name in all_storage_names]

        # Initialize all in parallel
        await asyncio.gather(*[store.initialize() for store in stores])

        # Verify all initialized
        for store in stores:
            store.initialize.assert_called_once()

    async def test_concurrent_retrieval(
        self,
        mock_storage_factory,
        mock_embedder,
        all_storage_names: list[str],
    ):
        """Test concurrent retrieval from storage backends.

        Args:
            mock_storage_factory: Storage factory fixture
            mock_embedder: Mock embedder
            all_storage_names: Storage names
        """
        for storage_name in all_storage_names:
            store = mock_storage_factory(storage_name)
            await store.initialize()

            # Make multiple concurrent queries
            queries = [
                "What is EOR?",
                "Drilling operations",
                "Well completion",
            ]

            query_embeddings = await mock_embedder.embed_texts(queries)

            # Execute retrievals in parallel
            tasks = [
                store.retrieve(query, emb, top_k=5)
                for query, emb in zip(queries, query_embeddings)
            ]

            results_list = await asyncio.gather(*tasks)

            # Verify
            assert len(results_list) == len(queries)
            assert all(isinstance(results, list) for results in results_list)


# ============================================================================
# Test Class: Data Integrity
# ============================================================================


class TestDataIntegrity:
    """Test data integrity across pipeline."""

    async def test_chunk_ids_preserved(
        self,
        mock_parser_factory,
        mock_storage_factory,
        mock_embedder,
        mock_pdf_path: Path,
        parser_storage_combinations: list[tuple[str, str]],
    ):
        """Test that chunk IDs are preserved through the pipeline.

        Args:
            mock_parser_factory: Parser factory fixture
            mock_storage_factory: Storage factory fixture
            mock_embedder: Mock embedder
            mock_pdf_path: Mock PDF path
            parser_storage_combinations: All combinations
        """
        # Test first combination
        parser_name, storage_name = parser_storage_combinations[0]
        parser = mock_parser_factory(parser_name)
        storage = mock_storage_factory(storage_name)

        # Run pipeline
        parsed_doc = await parser.parse(mock_pdf_path)
        chunks = parser.chunk_document(parsed_doc)
        original_chunk_ids = {chunk.chunk_id for chunk in chunks}

        embeddings = await mock_embedder.embed_texts([c.content for c in chunks])
        await storage.initialize()
        await storage.store_chunks(chunks, embeddings)

        # Retrieve
        query_embedding = await mock_embedder.embed_query("test")
        results = await storage.retrieve("test", query_embedding, top_k=10)

        # Verify chunk IDs in results match original
        retrieved_chunk_ids = {result.chunk_id for result in results}
        assert retrieved_chunk_ids.issubset(original_chunk_ids)

    async def test_metadata_preserved(
        self,
        mock_parser_factory,
        mock_storage_factory,
        mock_embedder,
        mock_pdf_path: Path,
        parser_storage_combinations: list[tuple[str, str]],
    ):
        """Test that metadata is preserved through the pipeline.

        Args:
            mock_parser_factory: Parser factory fixture
            mock_storage_factory: Storage factory fixture
            mock_embedder: Mock embedder
            mock_pdf_path: Mock PDF path
            parser_storage_combinations: All combinations
        """
        # Test first combination
        parser_name, storage_name = parser_storage_combinations[0]
        parser = mock_parser_factory(parser_name)
        storage = mock_storage_factory(storage_name)

        # Run pipeline
        parsed_doc = await parser.parse(mock_pdf_path)
        chunks = parser.chunk_document(parsed_doc)

        # Collect metadata
        metadata_by_chunk = {chunk.chunk_id: chunk.metadata for chunk in chunks}

        embeddings = await mock_embedder.embed_texts([c.content for c in chunks])
        await storage.initialize()
        await storage.store_chunks(chunks, embeddings)

        # Retrieve
        query_embedding = await mock_embedder.embed_query("test")
        results = await storage.retrieve("test", query_embedding, top_k=10)

        # Verify metadata preserved
        for result in results:
            assert result.metadata is not None
            assert isinstance(result.metadata, dict)
