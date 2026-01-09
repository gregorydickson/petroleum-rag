"""Tests for Weaviate storage backend."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models import DocumentChunk, RetrievalResult
from storage.weaviate_store import WeaviateStore


@pytest.fixture
def sample_chunks() -> list[DocumentChunk]:
    """Create sample document chunks for testing."""
    return [
        DocumentChunk(
            chunk_id="chunk_1",
            document_id="doc_1",
            content="This is about drilling operations in petroleum wells.",
            element_ids=["elem_1", "elem_2"],
            metadata={"source": "chapter1.pdf", "page": "5"},
            chunk_index=0,
            start_page=5,
            end_page=5,
            token_count=10,
            parent_section="Section 1",
        ),
        DocumentChunk(
            chunk_id="chunk_2",
            document_id="doc_1",
            content="Hydraulic fracturing techniques improve reservoir production.",
            element_ids=["elem_3"],
            metadata={"source": "chapter2.pdf", "page": "12"},
            chunk_index=1,
            start_page=12,
            end_page=12,
            token_count=8,
            parent_section="Section 2",
        ),
    ]


@pytest.fixture
def sample_embeddings() -> list[list[float]]:
    """Create sample embeddings for testing."""
    return [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
    ]


class TestWeaviateStore:
    """Tests for WeaviateStore implementation."""

    def test_initialization(self) -> None:
        """Test WeaviateStore initialization."""
        store = WeaviateStore()
        assert store.name == "Weaviate"
        assert store.class_name == "PetroleumDocument"
        assert store.alpha == 0.7
        assert not store._initialized

    def test_custom_config(self) -> None:
        """Test WeaviateStore with custom configuration."""
        config = {
            "host": "custom-host",
            "port": 9999,
            "class_name": "CustomClass",
            "alpha": 0.5,
        }
        store = WeaviateStore(config=config)
        assert store.config["host"] == "custom-host"
        assert store.config["port"] == 9999
        assert store.class_name == "CustomClass"
        assert store.alpha == 0.5

    @pytest.mark.asyncio
    async def test_initialize_creates_schema(self) -> None:
        """Test initialize creates Weaviate schema."""
        store = WeaviateStore()

        # Mock Weaviate client
        mock_client = MagicMock()
        mock_client.collections.exists.return_value = False
        mock_collection = MagicMock()
        mock_client.collections.create.return_value = mock_collection

        with patch("weaviate.connect_to_local", return_value=mock_client):
            await store.initialize()

        assert store._initialized
        assert store.client is not None
        mock_client.collections.create.assert_called_once()

        # Verify schema properties
        call_args = mock_client.collections.create.call_args
        assert call_args[1]["name"] == "PetroleumDocument"
        properties = call_args[1]["properties"]

        # Check that essential properties are present
        property_names = [prop.name for prop in properties]
        assert "chunk_id" in property_names
        assert "document_id" in property_names
        assert "content" in property_names
        assert "element_ids" in property_names
        assert "metadata" in property_names

    @pytest.mark.asyncio
    async def test_initialize_existing_schema(self) -> None:
        """Test initialize with existing schema."""
        store = WeaviateStore()

        # Mock Weaviate client with existing collection
        mock_client = MagicMock()
        mock_client.collections.exists.return_value = True
        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection

        with patch("weaviate.connect_to_local", return_value=mock_client):
            await store.initialize()

        assert store._initialized
        mock_client.collections.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_chunks_validates_input(
        self, sample_chunks: list[DocumentChunk]
    ) -> None:
        """Test store_chunks validates chunks and embeddings."""
        store = WeaviateStore()
        store._initialized = True
        store.client = MagicMock()

        # Mismatched lengths should raise ValueError
        with pytest.raises(ValueError, match="length mismatch"):
            await store.store_chunks(sample_chunks, [[0.1, 0.2]])

        # Empty chunks should raise ValueError
        with pytest.raises(ValueError, match="empty chunks"):
            await store.store_chunks([], [])

    @pytest.mark.asyncio
    async def test_store_chunks_batch_insert(
        self,
        sample_chunks: list[DocumentChunk],
        sample_embeddings: list[list[float]],
    ) -> None:
        """Test store_chunks performs batch insert."""
        store = WeaviateStore()
        store._initialized = True

        # Mock Weaviate client and batch
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_batch = MagicMock()
        mock_batch.__enter__ = MagicMock(return_value=mock_batch)
        mock_batch.__exit__ = MagicMock(return_value=False)
        mock_collection.batch.dynamic.return_value = mock_batch
        mock_client.collections.get.return_value = mock_collection
        store.client = mock_client

        await store.store_chunks(sample_chunks, sample_embeddings)

        # Verify batch operations
        assert mock_batch.add_object.call_count == 2

        # Check first chunk was added correctly
        first_call = mock_batch.add_object.call_args_list[0]
        properties = first_call[1]["properties"]
        assert properties["chunk_id"] == "chunk_1"
        assert properties["document_id"] == "doc_1"
        assert properties["content"] == "This is about drilling operations in petroleum wells."
        assert properties["chunk_index"] == 0

        # Check metadata was serialized
        metadata = json.loads(properties["metadata"])
        assert metadata["source"] == "chapter1.pdf"

        # Check vector was passed
        vector = first_call[1]["vector"]
        assert vector == [0.1, 0.2, 0.3, 0.4, 0.5]

    @pytest.mark.asyncio
    async def test_retrieve_hybrid_search(self) -> None:
        """Test retrieve performs hybrid search."""
        store = WeaviateStore()
        store._initialized = True

        # Mock Weaviate client
        mock_client = MagicMock()
        mock_collection = MagicMock()

        # Mock search response
        mock_obj = MagicMock()
        mock_obj.properties = {
            "chunk_id": "chunk_1",
            "document_id": "doc_1",
            "content": "Drilling operations content",
            "metadata": json.dumps({"page": "5"}),
        }
        mock_obj.metadata.score = 0.85

        mock_response = MagicMock()
        mock_response.objects = [mock_obj]
        mock_collection.query.hybrid.return_value = mock_response

        mock_client.collections.get.return_value = mock_collection
        store.client = mock_client

        # Perform retrieval
        query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        results = await store.retrieve(
            query="drilling operations",
            query_embedding=query_embedding,
            top_k=5,
        )

        # Verify hybrid search was called
        mock_collection.query.hybrid.assert_called_once()
        call_args = mock_collection.query.hybrid.call_args
        assert call_args[1]["query"] == "drilling operations"
        assert call_args[1]["vector"] == query_embedding
        assert call_args[1]["alpha"] == 0.7  # Default alpha
        assert call_args[1]["limit"] == 5

        # Verify results
        assert len(results) == 1
        assert isinstance(results[0], RetrievalResult)
        assert results[0].chunk_id == "chunk_1"
        assert results[0].document_id == "doc_1"
        assert results[0].score == 0.85
        assert results[0].rank == 1
        assert results[0].retrieval_method == "hybrid"

    @pytest.mark.asyncio
    async def test_retrieve_with_filters(self) -> None:
        """Test retrieve with metadata filters."""
        store = WeaviateStore()
        store._initialized = True

        # Mock Weaviate client
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_response = MagicMock()
        mock_response.objects = []
        mock_collection.query.hybrid.return_value = mock_response
        mock_client.collections.get.return_value = mock_collection
        store.client = mock_client

        # Perform retrieval with filter
        filters = {"document_id": "doc_123"}
        await store.retrieve(
            query="test query",
            query_embedding=[0.1, 0.2, 0.3],
            top_k=5,
            filters=filters,
        )

        # Verify filter was applied
        call_args = mock_collection.query.hybrid.call_args
        assert call_args[1]["filters"] is not None

    @pytest.mark.asyncio
    async def test_retrieve_applies_min_score(self) -> None:
        """Test retrieve filters results by minimum score."""
        store = WeaviateStore(config={"min_score": 0.7})
        store._initialized = True

        # Mock Weaviate client with multiple results
        mock_client = MagicMock()
        mock_collection = MagicMock()

        # Create mock objects with different scores
        mock_objs = []
        for i, score in enumerate([0.9, 0.6, 0.8], start=1):
            obj = MagicMock()
            obj.properties = {
                "chunk_id": f"chunk_{i}",
                "document_id": "doc_1",
                "content": f"Content {i}",
                "metadata": "{}",
            }
            obj.metadata.score = score
            mock_objs.append(obj)

        mock_response = MagicMock()
        mock_response.objects = mock_objs
        mock_collection.query.hybrid.return_value = mock_response
        mock_client.collections.get.return_value = mock_collection
        store.client = mock_client

        # Perform retrieval
        results = await store.retrieve(
            query="test",
            query_embedding=[0.1, 0.2, 0.3],
            top_k=5,
        )

        # Only scores >= 0.7 should be returned (0.9 and 0.8)
        assert len(results) == 2
        assert results[0].score == 0.9
        assert results[1].score == 0.8

    @pytest.mark.asyncio
    async def test_clear_deletes_all_objects(self) -> None:
        """Test clear removes all objects from collection."""
        store = WeaviateStore()
        store._initialized = True

        # Mock Weaviate client
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection
        store.client = mock_client

        await store.clear()

        # Verify delete was called
        mock_collection.data.delete_many.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_when_ready(self) -> None:
        """Test health_check returns True when Weaviate is ready."""
        store = WeaviateStore()
        store._initialized = True

        # Mock Weaviate client
        mock_client = MagicMock()
        mock_client.is_ready.return_value = True
        store.client = mock_client

        result = await store.health_check()
        assert result is True
        mock_client.is_ready.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_when_not_initialized(self) -> None:
        """Test health_check returns False when not initialized."""
        store = WeaviateStore()
        result = await store.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_on_error(self) -> None:
        """Test health_check returns False on exception."""
        store = WeaviateStore()
        store._initialized = True

        # Mock Weaviate client that raises exception
        mock_client = MagicMock()
        mock_client.is_ready.side_effect = Exception("Connection error")
        store.client = mock_client

        result = await store.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_error_when_not_initialized(self) -> None:
        """Test operations raise error when not initialized."""
        store = WeaviateStore()

        with pytest.raises(RuntimeError, match="not initialized"):
            await store.store_chunks([], [])

        with pytest.raises(RuntimeError, match="not initialized"):
            await store.retrieve("query", [0.1, 0.2], top_k=5)

        with pytest.raises(RuntimeError, match="not initialized"):
            await store.clear()
