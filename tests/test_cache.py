"""Tests for content-based caching system."""

import asyncio
import json
from pathlib import Path

import pytest

from utils.cache import ContentCache


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Create temporary cache directory."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def cache(temp_cache_dir: Path) -> ContentCache:
    """Create a ContentCache instance for testing."""
    return ContentCache(
        cache_dir=temp_cache_dir,
        max_memory_items=10,
        enabled=True,
    )


class TestContentCache:
    """Test suite for ContentCache."""

    # -------------------------------------------------------------------------
    # Basic Operations
    # -------------------------------------------------------------------------

    async def test_cache_initialization(self, temp_cache_dir: Path):
        """Test cache initialization."""
        cache = ContentCache(
            cache_dir=temp_cache_dir,
            max_memory_items=100,
            enabled=True,
        )

        assert cache.cache_dir == temp_cache_dir
        assert cache.max_memory_items == 100
        assert cache.enabled is True
        assert temp_cache_dir.exists()

    async def test_cache_disabled(self, temp_cache_dir: Path):
        """Test cache behavior when disabled."""
        cache = ContentCache(
            cache_dir=temp_cache_dir,
            enabled=False,
        )

        # Set should do nothing
        await cache.set("key1", "value1")

        # Get should return None
        result = await cache.get("key1")
        assert result is None

    async def test_hash_content(self):
        """Test content hashing."""
        text = "Hello, world!"
        hash1 = ContentCache.hash_content(text)
        hash2 = ContentCache.hash_content(text)

        # Same content should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 produces 64-char hex string

        # Different content should produce different hash
        hash3 = ContentCache.hash_content("Different text")
        assert hash1 != hash3

    # -------------------------------------------------------------------------
    # Get/Set Operations
    # -------------------------------------------------------------------------

    async def test_set_and_get(self, cache: ContentCache):
        """Test basic set and get operations."""
        key = "test_key"
        value = {"data": "test_value", "number": 42}

        # Set value
        await cache.set(key, value)

        # Get value (should be in memory)
        result = await cache.get(key)
        assert result == value

        # Check statistics
        stats = cache.get_stats()
        assert stats["sets"] == 1
        assert stats["hits"] == 1
        assert stats["memory_hits"] == 1

    async def test_get_nonexistent_key(self, cache: ContentCache):
        """Test getting a key that doesn't exist."""
        result = await cache.get("nonexistent_key")
        assert result is None

        # Check statistics
        stats = cache.get_stats()
        assert stats["misses"] == 1

    async def test_disk_persistence(self, cache: ContentCache, temp_cache_dir: Path):
        """Test that cache persists to disk."""
        key = "persist_key"
        value = [1, 2, 3, 4, 5]

        # Set value
        await cache.set(key, value)

        # Verify file exists on disk
        cache_file = temp_cache_dir / f"{key}.json"
        assert cache_file.exists()

        # Verify content
        with open(cache_file) as f:
            disk_value = json.load(f)
        assert disk_value == value

    async def test_disk_cache_hit(self, cache: ContentCache):
        """Test cache hit from disk when not in memory."""
        key = "disk_key"
        value = {"from": "disk"}

        # Set and remove from memory
        await cache.set(key, value)
        cache._memory_cache.clear()

        # Get should retrieve from disk
        result = await cache.get(key)
        assert result == value

        # Check statistics
        stats = cache.get_stats()
        assert stats["disk_hits"] == 1
        assert stats["memory_hits"] == 0

    # -------------------------------------------------------------------------
    # LRU Eviction
    # -------------------------------------------------------------------------

    async def test_lru_eviction(self, cache: ContentCache):
        """Test LRU eviction when memory cache is full."""
        # Fill cache to capacity (max_memory_items = 10)
        for i in range(10):
            await cache.set(f"key_{i}", f"value_{i}")

        assert len(cache._memory_cache) == 10

        # Add one more item - should evict oldest (key_0)
        await cache.set("key_10", "value_10")

        assert len(cache._memory_cache) == 10
        assert "key_0" not in cache._memory_cache
        assert "key_10" in cache._memory_cache

        # Check eviction statistics
        stats = cache.get_stats()
        assert stats["evictions"] == 1

    async def test_lru_access_updates_order(self, cache: ContentCache):
        """Test that accessing an item updates its LRU position."""
        # Fill cache to capacity
        for i in range(10):
            await cache.set(f"key_{i}", f"value_{i}")

        # Access key_0 (should move to end)
        await cache.get("key_0")

        # Add new item - should evict key_1 (now oldest)
        await cache.set("key_10", "value_10")

        # key_0 should still be in memory
        assert "key_0" in cache._memory_cache
        assert "key_1" not in cache._memory_cache

    # -------------------------------------------------------------------------
    # Clear Operations
    # -------------------------------------------------------------------------

    async def test_clear_cache(self, cache: ContentCache, temp_cache_dir: Path):
        """Test clearing cache (memory and disk)."""
        # Add some items
        for i in range(5):
            await cache.set(f"key_{i}", f"value_{i}")

        # Verify items exist
        assert len(cache._memory_cache) == 5
        cache_files = list(temp_cache_dir.glob("*.json"))
        assert len(cache_files) == 5

        # Clear cache
        await cache.clear()

        # Verify everything is cleared
        assert len(cache._memory_cache) == 0
        cache_files = list(temp_cache_dir.glob("*.json"))
        assert len(cache_files) == 0

        # Statistics should be reset
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["sets"] == 0

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    async def test_statistics_tracking(self, cache: ContentCache):
        """Test that statistics are tracked correctly."""
        # Perform various operations
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        await cache.get("key1")  # Hit (memory)
        await cache.get("key2")  # Hit (memory)
        await cache.get("key3")  # Miss

        # Check statistics
        stats = cache.get_stats()
        assert stats["sets"] == 2
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["memory_hits"] == 2
        assert stats["disk_hits"] == 0
        assert stats["total_requests"] == 3
        assert stats["hit_rate"] == 2 / 3

    async def test_get_size(self, cache: ContentCache):
        """Test cache size reporting."""
        # Add items
        for i in range(5):
            await cache.set(f"key_{i}", {"value": i, "data": "x" * 100})

        size_info = cache.get_size()

        assert size_info["memory_items"] == 5
        assert size_info["disk_items"] == 5
        assert size_info["disk_bytes"] > 0
        assert size_info["disk_mb"] >= 0  # Can be small, just verify it's calculated

    # -------------------------------------------------------------------------
    # Edge Cases
    # -------------------------------------------------------------------------

    async def test_empty_cache_stats(self, cache: ContentCache):
        """Test statistics for empty cache."""
        stats = cache.get_stats()

        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["total_requests"] == 0

    async def test_cache_with_complex_values(self, cache: ContentCache):
        """Test caching complex nested data structures."""
        complex_value = {
            "nested": {
                "list": [1, 2, 3],
                "dict": {"a": 1, "b": 2},
            },
            "array": [{"x": 1}, {"y": 2}],
        }

        key = "complex_key"
        await cache.set(key, complex_value)

        result = await cache.get(key)
        assert result == complex_value

    async def test_cache_with_list_values(self, cache: ContentCache):
        """Test caching list values (embeddings)."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5] * 100  # Simulate embedding

        key = "embedding_key"
        await cache.set(key, embedding)

        result = await cache.get(key)
        assert result == embedding
        assert len(result) == 500

    async def test_corrupted_cache_file(self, cache: ContentCache, temp_cache_dir: Path):
        """Test handling of corrupted cache files."""
        key = "corrupted_key"

        # Create corrupted cache file
        cache_file = temp_cache_dir / f"{key}.json"
        cache_file.write_text("not valid json {{{")

        # Get should return None and remove corrupted file
        result = await cache.get(key)
        assert result is None
        assert not cache_file.exists()

    # -------------------------------------------------------------------------
    # Concurrency
    # -------------------------------------------------------------------------

    async def test_concurrent_access(self, cache: ContentCache):
        """Test concurrent cache access."""

        async def set_value(i: int):
            await cache.set(f"key_{i}", f"value_{i}")

        async def get_value(i: int):
            return await cache.get(f"key_{i}")

        # Set values concurrently
        await asyncio.gather(*[set_value(i) for i in range(20)])

        # Get values concurrently
        results = await asyncio.gather(*[get_value(i) for i in range(20)])

        # First 10 should be evicted from memory but still on disk
        # Last 10 should be in memory
        assert len([r for r in results if r is not None]) == 20

    # -------------------------------------------------------------------------
    # Integration Tests
    # -------------------------------------------------------------------------

    async def test_real_world_usage_embeddings(self, cache: ContentCache):
        """Test real-world usage pattern for embeddings."""
        # Simulate embedding texts
        texts = [
            "The petroleum industry is important.",
            "Oil prices fluctuate daily.",
            "The petroleum industry is important.",  # Duplicate
            "Natural gas is a fossil fuel.",
            "Oil prices fluctuate daily.",  # Duplicate
        ]

        embeddings = {}

        # Simulate embedding process with cache
        for text in texts:
            key = ContentCache.hash_content(text)
            cached = await cache.get(key)

            if cached is None:
                # Simulate generating embedding
                embedding = [hash(text + str(i)) % 100 / 100 for i in range(1536)]
                await cache.set(key, embedding)
                embeddings[text] = embedding
            else:
                embeddings[text] = cached

        # Check cache statistics
        stats = cache.get_stats()
        assert stats["hits"] == 2  # Two duplicates found
        assert stats["misses"] == 3  # Three unique texts
        assert stats["sets"] == 3

    async def test_real_world_usage_llm(self, cache: ContentCache):
        """Test real-world usage pattern for LLM responses."""
        # Simulate LLM prompts
        prompts = [
            "What is petroleum?",
            "Explain oil extraction.",
            "What is petroleum?",  # Duplicate
        ]

        responses = {}

        # Simulate LLM calls with cache
        for prompt in prompts:
            key = ContentCache.hash_content(prompt)
            cached = await cache.get(key)

            if cached is None:
                # Simulate LLM response
                response = f"Response to: {prompt}"
                await cache.set(key, response)
                responses[prompt] = response
            else:
                responses[prompt] = cached

        # Check cache statistics
        stats = cache.get_stats()
        assert stats["hits"] == 1  # One duplicate found
        assert stats["misses"] == 2  # Two unique prompts
        assert stats["hit_rate"] == 1 / 3


class TestCacheIntegration:
    """Integration tests for cache with actual usage patterns."""

    async def test_cache_warmup_scenario(self, temp_cache_dir: Path):
        """Test cache warmup and subsequent usage."""
        cache = ContentCache(cache_dir=temp_cache_dir, max_memory_items=5)

        # Phase 1: Initial population (cold cache)
        for i in range(10):
            await cache.set(f"key_{i}", f"value_{i}")

        phase1_stats = cache.get_stats()
        assert phase1_stats["hits"] == 0  # All misses initially
        assert phase1_stats["sets"] == 10

        # Phase 2: Access recent items (warm cache - memory hits)
        for i in range(5, 10):  # Last 5 should be in memory
            result = await cache.get(f"key_{i}")
            assert result == f"value_{i}"

        phase2_stats = cache.get_stats()
        assert phase2_stats["memory_hits"] == 5

        # Phase 3: Access older items (warm cache - disk hits)
        for i in range(0, 5):  # First 5 evicted from memory but on disk
            result = await cache.get(f"key_{i}")
            assert result == f"value_{i}"

        phase3_stats = cache.get_stats()
        assert phase3_stats["disk_hits"] == 5

    async def test_cache_repr(self, cache: ContentCache):
        """Test string representation of cache."""
        await cache.set("key1", "value1")
        await cache.get("key1")
        await cache.get("key2")  # miss

        repr_str = repr(cache)
        assert "ContentCache" in repr_str
        assert "enabled=True" in repr_str
        assert "hit_rate=" in repr_str
