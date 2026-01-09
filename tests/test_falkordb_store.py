"""Tests for FalkorDB storage backend.

Tests basic graph operations, vector similarity, and multi-hop retrieval.
"""

import pytest

from models import DocumentChunk, RetrievalResult
from storage.falkordb_store import FalkorDBStore


@pytest.fixture
async def falkordb_store():
    """Create a FalkorDB store instance for testing."""
    config = {
        "host": "localhost",
        "port": 6379,
        "graph_name": "test_petroleum_rag",
        "top_k": 5,
        "min_score": 0.0,
    }

    # Use context manager for proper cleanup
    async with FalkorDBStore(config) as store:
        await store.clear()  # Clean slate for tests
        yield store

        # Cleanup after tests
        try:
            await store.clear()
        except Exception:
            pass


@pytest.mark.asyncio
async def test_initialization(falkordb_store):
    """Test that FalkorDB initializes correctly."""
    assert falkordb_store._initialized
    assert falkordb_store.graph is not None
    assert falkordb_store.graph_name == "test_petroleum_rag"


@pytest.mark.asyncio
async def test_health_check(falkordb_store):
    """Test health check functionality."""
    is_healthy = await falkordb_store.health_check()
    assert is_healthy is True


@pytest.mark.asyncio
async def test_store_and_retrieve_simple(falkordb_store):
    """Test storing and retrieving a simple chunk."""
    # Create test chunks
    chunks = [
        DocumentChunk(
            chunk_id="chunk_1",
            document_id="doc_1",
            content="This is about drilling operations in petroleum engineering.",
            chunk_index=0,
            start_page=1,
            end_page=1,
            token_count=50,
            parent_section="Introduction",
        ),
        DocumentChunk(
            chunk_id="chunk_2",
            document_id="doc_1",
            content="Reservoir characterization involves analyzing rock properties.",
            chunk_index=1,
            start_page=2,
            end_page=2,
            token_count=45,
            parent_section="Technical",
        ),
    ]

    # Simple embeddings (3-dimensional for testing)
    embeddings = [
        [0.1, 0.2, 0.3],  # chunk_1
        [0.4, 0.5, 0.6],  # chunk_2
    ]

    # Store chunks
    await falkordb_store.store_chunks(chunks, embeddings)

    # Retrieve with similar query embedding
    query_embedding = [0.15, 0.25, 0.35]  # Similar to chunk_1
    results = await falkordb_store.retrieve(
        query="drilling operations",
        query_embedding=query_embedding,
        top_k=2,
    )

    # Verify results
    assert len(results) >= 1
    assert any(r.chunk_id == "chunk_1" for r in results)
    assert all(isinstance(r, RetrievalResult) for r in results)
    assert all(r.retrieval_method == "graph" for r in results)


@pytest.mark.asyncio
async def test_graph_relationships(falkordb_store):
    """Test that graph relationships are created correctly."""
    chunks = [
        DocumentChunk(
            chunk_id="chunk_a",
            document_id="doc_1",
            content="First chunk in sequence.",
            chunk_index=0,
            parent_section="Section1",
        ),
        DocumentChunk(
            chunk_id="chunk_b",
            document_id="doc_1",
            content="Second chunk following first.",
            chunk_index=1,
            parent_section="Section1",
        ),
        DocumentChunk(
            chunk_id="chunk_c",
            document_id="doc_1",
            content="Third chunk in different section.",
            chunk_index=2,
            parent_section="Section2",
        ),
    ]

    embeddings = [
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [0.3, 0.3, 0.3],
    ]

    await falkordb_store.store_chunks(chunks, embeddings)

    # Query to verify FOLLOWS relationship exists
    query = """
    MATCH (c1:Chunk {chunk_id: 'chunk_a'})-[:FOLLOWS]->(c2:Chunk {chunk_id: 'chunk_b'})
    RETURN c1, c2
    """
    result = falkordb_store.graph.query(query)
    assert len(result.result_set) == 1

    # Query to verify Section CONTAINS relationship
    query = """
    MATCH (s:Section {section_name: 'Section1'})-[:CONTAINS]->(c:Chunk)
    RETURN COUNT(c) AS chunk_count
    """
    result = falkordb_store.graph.query(query)
    chunk_count = result.result_set[0][0]
    assert chunk_count == 2  # chunk_a and chunk_b


@pytest.mark.asyncio
async def test_cross_references(falkordb_store):
    """Test that REFERENCES relationships work for multi-hop retrieval."""
    chunks = [
        DocumentChunk(
            chunk_id="main_chunk",
            document_id="doc_1",
            content="Main discussion referencing other sections.",
            chunk_index=0,
            parent_section="Main",
            metadata={"references": "ref_chunk_1, ref_chunk_2"},
        ),
        DocumentChunk(
            chunk_id="ref_chunk_1",
            document_id="doc_1",
            content="Referenced content about drilling.",
            chunk_index=1,
            parent_section="Appendix",
        ),
        DocumentChunk(
            chunk_id="ref_chunk_2",
            document_id="doc_1",
            content="Referenced content about reservoir analysis.",
            chunk_index=2,
            parent_section="Appendix",
        ),
    ]

    embeddings = [
        [1.0, 0.0, 0.0],  # main_chunk
        [0.0, 1.0, 0.0],  # ref_chunk_1 (different direction)
        [0.0, 0.0, 1.0],  # ref_chunk_2 (different direction)
    ]

    await falkordb_store.store_chunks(chunks, embeddings)

    # First store all chunks, then create references
    # (references are created during storage if metadata contains them)

    # Retrieve with query similar to main_chunk
    query_embedding = [0.9, 0.1, 0.1]
    results = await falkordb_store.retrieve(
        query="main discussion",
        query_embedding=query_embedding,
        top_k=1,
    )

    # Should get main_chunk first, plus expanded referenced chunks
    chunk_ids = [r.chunk_id for r in results]
    assert "main_chunk" in chunk_ids

    # Check if expansion happened (may include referenced chunks)
    # Note: Expansion depends on graph traversal working
    if len(results) > 1:
        # Verify expanded results have lower scores
        main_score = next(r.score for r in results if r.chunk_id == "main_chunk")
        expanded_results = [r for r in results if r.chunk_id != "main_chunk"]
        if expanded_results:
            assert all(r.score <= main_score for r in expanded_results)


@pytest.mark.asyncio
async def test_filtering(falkordb_store):
    """Test metadata filtering during retrieval."""
    chunks = [
        DocumentChunk(
            chunk_id="doc1_chunk",
            document_id="doc_1",
            content="Content from document 1.",
            chunk_index=0,
        ),
        DocumentChunk(
            chunk_id="doc2_chunk",
            document_id="doc_2",
            content="Content from document 2.",
            chunk_index=0,
        ),
    ]

    embeddings = [
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],  # Same embedding
    ]

    await falkordb_store.store_chunks(chunks, embeddings)

    # Retrieve with document filter
    results = await falkordb_store.retrieve(
        query="content",
        query_embedding=[0.5, 0.5, 0.5],
        top_k=5,
        filters={"document_id": "doc_1"},
    )

    # Should only get chunks from doc_1
    assert all(r.document_id == "doc_1" for r in results)
    assert any(r.chunk_id == "doc1_chunk" for r in results)


@pytest.mark.asyncio
async def test_clear(falkordb_store):
    """Test clearing all data from the graph."""
    # Store some data
    chunks = [
        DocumentChunk(
            chunk_id="test_chunk",
            document_id="test_doc",
            content="Test content.",
            chunk_index=0,
        )
    ]
    embeddings = [[0.1, 0.2, 0.3]]

    await falkordb_store.store_chunks(chunks, embeddings)

    # Clear the graph
    await falkordb_store.clear()

    # Verify graph is empty
    query = "MATCH (n) RETURN COUNT(n) AS node_count"
    result = falkordb_store.graph.query(query)
    node_count = result.result_set[0][0]
    assert node_count == 0


@pytest.mark.asyncio
async def test_validation(falkordb_store):
    """Test that validation works for mismatched chunks and embeddings."""
    chunks = [
        DocumentChunk(
            chunk_id="chunk_1",
            document_id="doc_1",
            content="Test",
            chunk_index=0,
        )
    ]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]  # Wrong count

    with pytest.raises(ValueError, match="length mismatch"):
        await falkordb_store.store_chunks(chunks, embeddings)


@pytest.mark.asyncio
async def test_empty_chunks(falkordb_store):
    """Test that empty chunks list raises error."""
    with pytest.raises(ValueError, match="empty chunks"):
        await falkordb_store.store_chunks([], [])


def test_repr():
    """Test string representation."""
    store = FalkorDBStore({"graph_name": "test_graph", "host": "localhost", "port": 6379})
    repr_str = repr(store)
    assert "FalkorDBStore" in repr_str
    assert "test_graph" in repr_str
