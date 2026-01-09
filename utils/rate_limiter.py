"""Global rate limiting for API coordination.

This module provides coordinated rate limiting across all API-calling components
using a token bucket algorithm. Prevents API throttling under load by managing
global request quotas for different services.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for a rate-limited service.

    Attributes:
        requests_per_minute: Maximum requests allowed per minute
        burst_size: Maximum burst capacity (defaults to requests_per_minute)
    """

    requests_per_minute: int
    burst_size: int | None = None

    def __post_init__(self):
        """Set default burst size if not provided."""
        if self.burst_size is None:
            self.burst_size = self.requests_per_minute


class TokenBucket:
    """Token bucket implementation for rate limiting.

    The token bucket algorithm allows bursts of traffic while maintaining
    a steady average rate. Tokens refill at a constant rate, and requests
    consume tokens. When the bucket is empty, requests must wait.

    Attributes:
        rate: Token refill rate (tokens per second)
        capacity: Maximum token capacity (bucket size)
    """

    def __init__(self, rate: float, capacity: float):
        """Initialize token bucket.

        Args:
            rate: Token refill rate in tokens per second
            capacity: Maximum number of tokens the bucket can hold
        """
        self.rate = rate  # Tokens per second
        self.capacity = capacity  # Max tokens
        self._tokens = capacity  # Current token count
        self._last_update = time.monotonic()  # Last refill timestamp
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        """Acquire tokens, waiting if necessary.

        This method blocks until the requested number of tokens becomes
        available. It's the primary method for rate-limited API calls.

        Args:
            tokens: Number of tokens to acquire (default: 1)
        """
        while True:
            if self.try_acquire(tokens):
                return

            # Calculate wait time based on deficit
            needed = tokens - self._tokens
            wait_time = needed / self.rate

            logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s for {tokens} tokens")
            await asyncio.sleep(wait_time)

    def try_acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens without waiting.

        Attempts to acquire tokens immediately. Returns True if successful,
        False if insufficient tokens are available.

        Args:
            tokens: Number of tokens to acquire (default: 1)

        Returns:
            True if tokens were acquired, False otherwise
        """
        self._refill()

        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    def _refill(self) -> None:
        """Refill tokens based on elapsed time.

        Called automatically by acquire/try_acquire to add tokens
        based on the time elapsed since the last refill.
        """
        now = time.monotonic()
        elapsed = now - self._last_update

        # Add tokens based on rate and elapsed time
        new_tokens = elapsed * self.rate
        self._tokens = min(self.capacity, self._tokens + new_tokens)
        self._last_update = now

    def available_tokens(self) -> float:
        """Get currently available tokens.

        Returns:
            Number of tokens currently available in the bucket
        """
        self._refill()
        return self._tokens


class GlobalRateLimiter:
    """Global rate limiter with token bucket algorithm.

    Manages rate limits for multiple services (OpenAI, Anthropic, etc.)
    with independent token buckets for each service. Provides coordinated
    rate limiting across the entire application.

    Example:
        >>> limiter = GlobalRateLimiter()
        >>> limiter.register("openai", RateLimitConfig(requests_per_minute=3000))
        >>> await limiter.acquire("openai", tokens=10)  # Acquire for 10 requests
    """

    def __init__(self):
        """Initialize global rate limiter."""
        self._limiters: Dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()

    def register(self, service: str, config: RateLimitConfig) -> None:
        """Register a service with rate limit configuration.

        Args:
            service: Service name (e.g., "openai", "anthropic")
            config: Rate limit configuration for the service
        """
        # Convert requests per minute to tokens per second
        rate = config.requests_per_minute / 60.0

        self._limiters[service] = TokenBucket(
            rate=rate,
            capacity=config.burst_size,
        )

        logger.info(
            f"Registered rate limiter for {service}: "
            f"{config.requests_per_minute} req/min, "
            f"burst={config.burst_size}"
        )

    async def acquire(self, service: str, tokens: int = 1) -> None:
        """Acquire tokens for a service, blocking if necessary.

        This is the primary method for rate-limited operations. It blocks
        until the requested tokens become available.

        Args:
            service: Service name (must be registered)
            tokens: Number of tokens to acquire (default: 1)

        Raises:
            ValueError: If service is not registered
        """
        if service not in self._limiters:
            raise ValueError(
                f"Service '{service}' not registered. "
                f"Available services: {list(self._limiters.keys())}"
            )

        async with self._lock:
            await self._limiters[service].acquire(tokens)

    def try_acquire(self, service: str, tokens: int = 1) -> bool:
        """Try to acquire tokens without blocking.

        Attempts to acquire tokens immediately. Useful for non-blocking
        operations or rate limit checking.

        Args:
            service: Service name (must be registered)
            tokens: Number of tokens to acquire (default: 1)

        Returns:
            True if tokens were acquired, False otherwise

        Raises:
            ValueError: If service is not registered
        """
        if service not in self._limiters:
            raise ValueError(
                f"Service '{service}' not registered. "
                f"Available services: {list(self._limiters.keys())}"
            )

        return self._limiters[service].try_acquire(tokens)

    def get_available(self, service: str) -> float:
        """Get available tokens for a service.

        Args:
            service: Service name

        Returns:
            Number of tokens currently available

        Raises:
            ValueError: If service is not registered
        """
        if service not in self._limiters:
            return 0.0

        return self._limiters[service].available_tokens()

    def is_registered(self, service: str) -> bool:
        """Check if a service is registered.

        Args:
            service: Service name

        Returns:
            True if service is registered, False otherwise
        """
        return service in self._limiters


# Global instance
rate_limiter = GlobalRateLimiter()


def setup_rate_limits() -> None:
    """Setup rate limits for all services.

    Configures rate limiters based on API provider limits and
    application settings. Should be called during application
    initialization.

    Rate limits are based on:
    - OpenAI: Tier 3 default (adjust based on your tier)
    - Anthropic: Standard tier (adjust based on your tier)
    - LlamaParse: Free/paid tier limits
    - Vertex AI: Standard quota

    Note: Adjust these values based on your actual API tier and quotas.
    """
    from config import settings

    # OpenAI rate limits (Tier 3 default - adjust based on your tier)
    # See: https://platform.openai.com/docs/guides/rate-limits
    openai_rpm = getattr(settings, "openai_rate_limit", 3000)
    rate_limiter.register(
        "openai",
        RateLimitConfig(
            requests_per_minute=openai_rpm,
            burst_size=min(500, openai_rpm),  # Cap burst at 500
        ),
    )

    # Anthropic rate limits (adjust based on your tier)
    # See: https://docs.anthropic.com/claude/reference/rate-limits
    anthropic_rpm = getattr(settings, "anthropic_rate_limit", 1000)
    rate_limiter.register(
        "anthropic",
        RateLimitConfig(
            requests_per_minute=anthropic_rpm,
            burst_size=min(200, anthropic_rpm),  # Cap burst at 200
        ),
    )

    # LlamaParse rate limits
    # See: https://docs.llamaindex.ai/en/stable/module_guides/loading/parsing/
    llamaparse_rpm = getattr(settings, "llamaparse_rate_limit", 600)
    rate_limiter.register(
        "llamaparse",
        RateLimitConfig(
            requests_per_minute=llamaparse_rpm,
            burst_size=min(100, llamaparse_rpm),  # Cap burst at 100
        ),
    )

    # Vertex AI rate limits
    # See: https://cloud.google.com/document-ai/quotas
    vertex_rpm = getattr(settings, "vertex_rate_limit", 600)
    rate_limiter.register(
        "vertex",
        RateLimitConfig(
            requests_per_minute=vertex_rpm,
            burst_size=min(100, vertex_rpm),  # Cap burst at 100
        ),
    )

    logger.info("Rate limiters configured for all services")
