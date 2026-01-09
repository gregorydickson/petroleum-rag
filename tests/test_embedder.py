"""Tests for the UnifiedEmbedder class.

Tests cover:
- Initialization and configuration
- Single text embedding
- Batch embedding with various sizes
- Error handling and validation
- Rate limit handling (mocked)
- Connection validation
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import RateLimitError, APIError, APITimeoutError

from embeddings.embedder import UnifiedEmbedder


def create_mock_request() -> httpx.Request:
    """Create a mock httpx.Request for error testing."""
    return httpx.Request("POST", "https://api.openai.com/v1/embeddings")


def create_mock_response(status_code: int = 429) -> httpx.Response:
    """Create a mock httpx.Response for error testing."""
    return httpx.Response(
        status_code=status_code,
        request=create_mock_request(),
        headers={},
        content=b"",
    )


class TestUnifiedEmbedderInit:
    """Tests for UnifiedEmbedder initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default settings."""
        with patch("embeddings.embedder.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"
            mock_settings.embedding_model = "text-embedding-3-small"
            mock_settings.embedding_dimension = 1536
            mock_settings.embedding_batch_size = 100

            embedder = UnifiedEmbedder()

            assert embedder.api_key == "test-key"
            assert embedder.model == "text-embedding-3-small"
            assert embedder.dimensions == 1536
            assert embedder.batch_size == 100

    def test_init_with_custom_values(self) -> None:
        """Test initialization with custom values."""
        embedder = UnifiedEmbedder(
            api_key="custom-key",
            model="text-embedding-3-large",
            dimensions=3072,
            batch_size=50,
        )

        assert embedder.api_key == "custom-key"
        assert embedder.model == "text-embedding-3-large"
        assert embedder.dimensions == 3072
        assert embedder.batch_size == 50

    def test_init_without_api_key(self) -> None:
        """Test initialization fails without API key."""
        with patch("embeddings.embedder.settings") as mock_settings:
            mock_settings.openai_api_key = ""

            with pytest.raises(ValueError, match="OpenAI API key must be provided"):
                UnifiedEmbedder()

    def test_init_respects_max_batch_size(self) -> None:
        """Test initialization caps batch size at maximum."""
        embedder = UnifiedEmbedder(
            api_key="test-key",
            batch_size=5000,  # Exceeds MAX_BATCH_SIZE
        )

        assert embedder.batch_size == UnifiedEmbedder.MAX_BATCH_SIZE


class TestUnifiedEmbedderEmbedText:
    """Tests for single text embedding."""

    @pytest.fixture
    def embedder(self) -> UnifiedEmbedder:
        """Create embedder instance for testing."""
        return UnifiedEmbedder(api_key="test-key", dimensions=1536)

    @pytest.mark.asyncio
    async def test_embed_text_success(self, embedder: UnifiedEmbedder) -> None:
        """Test successful single text embedding."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]

        with patch.object(embedder.client.embeddings, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            result = await embedder.embed_text("test text")

            assert len(result) == 1536
            assert all(isinstance(x, float) for x in result)
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_text_empty_string(self, embedder: UnifiedEmbedder) -> None:
        """Test embedding empty string raises ValueError."""
        with pytest.raises(ValueError, match="Cannot embed empty text"):
            await embedder.embed_text("")

    @pytest.mark.asyncio
    async def test_embed_text_whitespace_only(self, embedder: UnifiedEmbedder) -> None:
        """Test embedding whitespace-only string raises ValueError."""
        with pytest.raises(ValueError, match="Cannot embed empty text"):
            await embedder.embed_text("   \n\t  ")

    @pytest.mark.asyncio
    async def test_embed_text_api_error(self, embedder: UnifiedEmbedder) -> None:
        """Test handling of API errors."""
        with patch.object(embedder.client.embeddings, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = APIError(
                "API error",
                request=create_mock_request(),
                body=None,
            )

            with pytest.raises(RuntimeError, match="Embedding generation failed"):
                await embedder.embed_text("test text")


class TestUnifiedEmbedderEmbedBatch:
    """Tests for batch text embedding."""

    @pytest.fixture
    def embedder(self) -> UnifiedEmbedder:
        """Create embedder instance for testing."""
        return UnifiedEmbedder(api_key="test-key", dimensions=1536, batch_size=10)

    @pytest.mark.asyncio
    async def test_embed_batch_small(self, embedder: UnifiedEmbedder) -> None:
        """Test batch embedding with small batch (single API call)."""
        texts = ["text1", "text2", "text3"]
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536) for _ in texts]

        with patch.object(embedder.client.embeddings, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            results = await embedder.embed_batch(texts)

            assert len(results) == 3
            assert all(len(emb) == 1536 for emb in results)
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_batch_large(self, embedder: UnifiedEmbedder) -> None:
        """Test batch embedding with large batch (multiple API calls)."""
        texts = [f"text{i}" for i in range(25)]  # > batch_size of 10
        mock_response = MagicMock()

        call_count = 0

        async def mock_create_fn(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            input_texts = kwargs.get("input", [])
            return MagicMock(
                data=[MagicMock(embedding=[0.1] * 1536) for _ in input_texts]
            )

        with patch.object(embedder.client.embeddings, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = mock_create_fn

            results = await embedder.embed_batch(texts)

            assert len(results) == 25
            assert all(len(emb) == 1536 for emb in results)
            assert call_count == 3  # 25 texts / 10 batch_size = 3 calls

    @pytest.mark.asyncio
    async def test_embed_batch_empty_list(self, embedder: UnifiedEmbedder) -> None:
        """Test batch embedding with empty list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot embed empty text list"):
            await embedder.embed_batch([])

    @pytest.mark.asyncio
    async def test_embed_batch_contains_empty_strings(self, embedder: UnifiedEmbedder) -> None:
        """Test batch embedding with empty strings raises ValueError."""
        texts = ["text1", "", "text3"]

        with pytest.raises(ValueError, match="Found empty texts at indices"):
            await embedder.embed_batch(texts)

    @pytest.mark.asyncio
    async def test_embed_batch_partial_failure(self, embedder: UnifiedEmbedder) -> None:
        """Test handling of failures in batch processing."""
        texts = [f"text{i}" for i in range(5)]

        with patch.object(embedder.client.embeddings, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = APIError(
                "API error",
                request=create_mock_request(),
                body=None,
            )

            with pytest.raises(RuntimeError, match="Batch embedding generation failed"):
                await embedder.embed_batch(texts)


class TestUnifiedEmbedderRetry:
    """Tests for retry logic and rate limiting."""

    @pytest.fixture
    def embedder(self) -> UnifiedEmbedder:
        """Create embedder instance for testing."""
        return UnifiedEmbedder(api_key="test-key", dimensions=1536)

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self, embedder: UnifiedEmbedder) -> None:
        """Test retry on rate limit error."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]

        call_count = 0

        async def mock_create_fn(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RateLimitError(
                    "Rate limited",
                    response=create_mock_response(429),
                    body=None,
                )
            return mock_response

        with patch.object(embedder.client.embeddings, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = mock_create_fn

            result = await embedder._embed_with_retry(["test"])

            assert len(result) == 1
            assert call_count == 2  # First call failed, second succeeded

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self, embedder: UnifiedEmbedder) -> None:
        """Test retry on timeout error."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]

        call_count = 0

        async def mock_create_fn(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise APITimeoutError(request=None)
            return mock_response

        with patch.object(embedder.client.embeddings, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = mock_create_fn

            result = await embedder._embed_with_retry(["test"])

            assert len(result) == 1
            assert call_count == 2


class TestUnifiedEmbedderUtilities:
    """Tests for utility methods."""

    @pytest.fixture
    def embedder(self) -> UnifiedEmbedder:
        """Create embedder instance for testing."""
        return UnifiedEmbedder(api_key="test-key", dimensions=1536)

    @pytest.mark.asyncio
    async def test_validate_connection_success(self, embedder: UnifiedEmbedder) -> None:
        """Test successful connection validation."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]

        with patch.object(embedder.client.embeddings, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            result = await embedder.validate_connection()

            assert result is True

    @pytest.mark.asyncio
    async def test_validate_connection_failure(self, embedder: UnifiedEmbedder) -> None:
        """Test connection validation failure."""
        with patch.object(embedder.client.embeddings, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = APIError(
                "Connection failed",
                request=create_mock_request(),
                body=None,
            )

            result = await embedder.validate_connection()

            assert result is False

    @pytest.mark.asyncio
    async def test_close(self, embedder: UnifiedEmbedder) -> None:
        """Test closing the client."""
        with patch.object(embedder.client, "close", new_callable=AsyncMock) as mock_close:
            await embedder.close()

            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Test using embedder as async context manager."""
        embedder = UnifiedEmbedder(api_key="test-key", dimensions=1536)

        with patch.object(embedder.client, "close", new_callable=AsyncMock) as mock_close:
            async with embedder:
                # embedder should be usable within context
                assert embedder.client is not None

            # close should have been called on exit
            mock_close.assert_called_once()

    def test_get_stats(self, embedder: UnifiedEmbedder) -> None:
        """Test getting embedder statistics."""
        stats = embedder.get_stats()

        assert "model" in stats
        assert "dimensions" in stats
        assert "batch_size" in stats
        assert stats["dimensions"] == 1536

    def test_repr(self, embedder: UnifiedEmbedder) -> None:
        """Test string representation."""
        repr_str = repr(embedder)

        assert "UnifiedEmbedder" in repr_str
        assert "dimensions=1536" in repr_str
