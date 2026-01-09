"""Tests for global rate limiter.

This module tests the token bucket rate limiting implementation
and global rate limiter coordination.
"""

import asyncio
import time

import pytest

from utils.rate_limiter import (
    GlobalRateLimiter,
    RateLimitConfig,
    TokenBucket,
    rate_limiter,
    setup_rate_limits,
)


class TestTokenBucket:
    """Tests for TokenBucket implementation."""

    def test_initialization(self):
        """Test token bucket initialization."""
        bucket = TokenBucket(rate=10.0, capacity=100.0)

        assert bucket.rate == 10.0
        assert bucket.capacity == 100.0
        assert bucket._tokens == 100.0

    def test_try_acquire_success(self):
        """Test successful token acquisition without waiting."""
        bucket = TokenBucket(rate=10.0, capacity=100.0)

        # Should have full capacity
        assert bucket.try_acquire(10)
        assert bucket._tokens == 90.0

    def test_try_acquire_failure(self):
        """Test failed token acquisition when insufficient tokens."""
        bucket = TokenBucket(rate=10.0, capacity=10.0)

        # Drain the bucket
        assert bucket.try_acquire(10)

        # Should fail with no tokens
        assert not bucket.try_acquire(1)

    @pytest.mark.asyncio
    async def test_acquire_with_wait(self):
        """Test token acquisition with waiting."""
        # Create bucket with low capacity
        bucket = TokenBucket(rate=10.0, capacity=5.0)

        # Drain the bucket
        assert bucket.try_acquire(5)

        # This should wait for tokens to refill
        start = time.monotonic()
        await bucket.acquire(3)
        elapsed = time.monotonic() - start

        # Should have waited approximately 0.3 seconds (3 tokens / 10 per second)
        assert elapsed >= 0.25  # Allow some tolerance
        assert elapsed < 0.5

    def test_refill(self):
        """Test token refilling over time."""
        bucket = TokenBucket(rate=10.0, capacity=100.0)

        # Drain half the bucket
        bucket.try_acquire(50)
        assert bucket._tokens == 50.0

        # Wait for refill
        time.sleep(1.0)

        # Should have refilled ~10 tokens (1 second * 10 tokens/sec)
        available = bucket.available_tokens()
        assert available >= 59.0  # Allow for timing variations
        assert available <= 61.0

    def test_capacity_limit(self):
        """Test that bucket doesn't exceed capacity."""
        bucket = TokenBucket(rate=10.0, capacity=50.0)

        # Start with full capacity
        assert bucket._tokens == 50.0

        # Wait - tokens should not exceed capacity
        time.sleep(1.0)
        available = bucket.available_tokens()
        assert available <= 50.0

    def test_available_tokens(self):
        """Test getting available tokens."""
        bucket = TokenBucket(rate=10.0, capacity=100.0)

        available = bucket.available_tokens()
        assert available == 100.0

        bucket.try_acquire(25)
        available = bucket.available_tokens()
        # Allow small floating point tolerance
        assert abs(available - 75.0) < 0.1


class TestRateLimitConfig:
    """Tests for RateLimitConfig dataclass."""

    def test_default_burst_size(self):
        """Test that burst_size defaults to requests_per_minute."""
        config = RateLimitConfig(requests_per_minute=1000)

        assert config.requests_per_minute == 1000
        assert config.burst_size == 1000

    def test_custom_burst_size(self):
        """Test custom burst_size."""
        config = RateLimitConfig(requests_per_minute=1000, burst_size=200)

        assert config.requests_per_minute == 1000
        assert config.burst_size == 200


class TestGlobalRateLimiter:
    """Tests for GlobalRateLimiter."""

    def test_register_service(self):
        """Test registering a service."""
        limiter = GlobalRateLimiter()
        config = RateLimitConfig(requests_per_minute=600, burst_size=100)

        limiter.register("test_service", config)

        assert limiter.is_registered("test_service")
        assert "test_service" in limiter._limiters

    def test_is_registered(self):
        """Test checking if service is registered."""
        limiter = GlobalRateLimiter()

        assert not limiter.is_registered("unknown_service")

        limiter.register("known_service", RateLimitConfig(requests_per_minute=100))
        assert limiter.is_registered("known_service")

    @pytest.mark.asyncio
    async def test_acquire_success(self):
        """Test successful token acquisition."""
        limiter = GlobalRateLimiter()
        limiter.register("test", RateLimitConfig(requests_per_minute=600))

        # Should not raise
        await limiter.acquire("test", tokens=10)

    @pytest.mark.asyncio
    async def test_acquire_unregistered_service(self):
        """Test acquiring tokens for unregistered service raises error."""
        limiter = GlobalRateLimiter()

        with pytest.raises(ValueError, match="not registered"):
            await limiter.acquire("unknown", tokens=1)

    def test_try_acquire_success(self):
        """Test non-blocking token acquisition."""
        limiter = GlobalRateLimiter()
        limiter.register("test", RateLimitConfig(requests_per_minute=600))

        assert limiter.try_acquire("test", tokens=10)

    def test_try_acquire_unregistered_service(self):
        """Test try_acquire for unregistered service raises error."""
        limiter = GlobalRateLimiter()

        with pytest.raises(ValueError, match="not registered"):
            limiter.try_acquire("unknown", tokens=1)

    def test_get_available(self):
        """Test getting available tokens."""
        limiter = GlobalRateLimiter()
        config = RateLimitConfig(requests_per_minute=600, burst_size=100)
        limiter.register("test", config)

        available = limiter.get_available("test")
        assert available == 100.0  # Should match burst_size

    def test_get_available_unregistered(self):
        """Test getting available tokens for unregistered service."""
        limiter = GlobalRateLimiter()

        available = limiter.get_available("unknown")
        assert available == 0.0

    @pytest.mark.asyncio
    async def test_concurrent_acquisitions(self):
        """Test concurrent token acquisitions."""
        limiter = GlobalRateLimiter()
        limiter.register("test", RateLimitConfig(requests_per_minute=600, burst_size=50))

        # Try to acquire tokens concurrently
        async def acquire_tokens():
            await limiter.acquire("test", tokens=10)

        # Run 5 concurrent acquisitions (50 tokens total)
        await asyncio.gather(*[acquire_tokens() for _ in range(5)])

        # All should succeed (allow small floating point tolerance)
        available = limiter.get_available("test")
        assert available < 0.1  # Should have used nearly all tokens

    @pytest.mark.asyncio
    async def test_rate_limiting_behavior(self):
        """Test that rate limiting actually delays requests."""
        limiter = GlobalRateLimiter()
        # Very low rate for testing: 60 req/min = 1 req/sec
        limiter.register("test", RateLimitConfig(requests_per_minute=60, burst_size=1))

        # First acquisition should be immediate
        start = time.monotonic()
        await limiter.acquire("test", tokens=1)
        first_elapsed = time.monotonic() - start
        assert first_elapsed < 0.1  # Should be nearly instant

        # Second acquisition should wait ~1 second
        start = time.monotonic()
        await limiter.acquire("test", tokens=1)
        second_elapsed = time.monotonic() - start
        assert second_elapsed >= 0.9  # Should wait ~1 second
        assert second_elapsed < 1.5  # But not too long


class TestSetupRateLimits:
    """Tests for setup_rate_limits function."""

    def test_setup_rate_limits(self):
        """Test that setup_rate_limits registers all services."""
        # Reset global rate limiter
        global rate_limiter
        from utils.rate_limiter import GlobalRateLimiter as RateLimiter

        rate_limiter._limiters.clear()

        # Call setup
        setup_rate_limits()

        # Check all services are registered
        assert rate_limiter.is_registered("openai")
        assert rate_limiter.is_registered("anthropic")
        assert rate_limiter.is_registered("llamaparse")
        assert rate_limiter.is_registered("vertex")

    def test_setup_rate_limits_uses_settings(self):
        """Test that setup_rate_limits uses configuration from settings."""
        from config import settings

        # Clear existing limiters
        rate_limiter._limiters.clear()

        # Setup with current settings
        setup_rate_limits()

        # Check that services have reasonable limits
        openai_available = rate_limiter.get_available("openai")
        assert openai_available > 0  # Should have some capacity

        anthropic_available = rate_limiter.get_available("anthropic")
        assert anthropic_available > 0


class TestIntegration:
    """Integration tests for rate limiter."""

    @pytest.mark.asyncio
    async def test_multiple_services(self):
        """Test using multiple services simultaneously."""
        limiter = GlobalRateLimiter()
        limiter.register("service1", RateLimitConfig(requests_per_minute=600))
        limiter.register("service2", RateLimitConfig(requests_per_minute=600))

        # Acquire from both services
        await limiter.acquire("service1", tokens=10)
        await limiter.acquire("service2", tokens=10)

        # Check independent token buckets
        assert limiter.get_available("service1") < 600
        assert limiter.get_available("service2") < 600

    @pytest.mark.asyncio
    async def test_burst_handling(self):
        """Test burst request handling."""
        limiter = GlobalRateLimiter()
        limiter.register(
            "burst_test", RateLimitConfig(requests_per_minute=600, burst_size=100)
        )

        # Send burst of 100 requests (should succeed immediately)
        start = time.monotonic()

        for _ in range(100):
            await limiter.acquire("burst_test", tokens=1)

        elapsed = time.monotonic() - start

        # Should complete quickly (within burst capacity)
        assert elapsed < 1.0

        # Next request should wait (burst depleted)
        start = time.monotonic()
        await limiter.acquire("burst_test", tokens=1)
        wait_time = time.monotonic() - start

        assert wait_time > 0.05  # Should have waited for refill
