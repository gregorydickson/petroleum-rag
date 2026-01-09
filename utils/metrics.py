"""Prometheus metrics and decorators for RAG system monitoring.

This module provides metrics tracking for:
- Query processing and retrieval
- Embedding generation
- LLM API calls
- Document parsing
- Cache performance
- Active connections
- Rate limiting
"""

import asyncio
import logging
import time
from functools import wraps
from typing import Any, Callable

from prometheus_client import Counter, Gauge, Histogram, Info

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Counters - Track cumulative event counts
# -------------------------------------------------------------------------

QUERIES_TOTAL = Counter(
    "rag_queries_total",
    "Total number of queries processed",
    ["parser", "storage", "status"],
)

EMBEDDINGS_TOTAL = Counter(
    "rag_embeddings_total",
    "Total number of embeddings generated",
    ["status"],
)

LLM_CALLS_TOTAL = Counter(
    "rag_llm_calls_total",
    "Total number of LLM API calls",
    ["model", "status"],
)

PARSE_OPERATIONS_TOTAL = Counter(
    "rag_parse_operations_total",
    "Total number of parse operations",
    ["parser", "status"],
)

STORAGE_OPERATIONS_TOTAL = Counter(
    "rag_storage_operations_total",
    "Total number of storage operations",
    ["storage", "operation", "status"],
)

CACHE_OPERATIONS_TOTAL = Counter(
    "rag_cache_operations_total",
    "Total number of cache operations",
    ["cache_type", "operation", "status"],
)

# -------------------------------------------------------------------------
# Histograms - Track distribution of values (latency tracking)
# -------------------------------------------------------------------------

QUERY_DURATION = Histogram(
    "rag_query_duration_seconds",
    "Query processing duration in seconds",
    ["parser", "storage"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

EMBEDDING_DURATION = Histogram(
    "rag_embedding_duration_seconds",
    "Embedding generation duration in seconds",
    ["batch_size_range"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

LLM_DURATION = Histogram(
    "rag_llm_duration_seconds",
    "LLM API call duration in seconds",
    ["model"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

PARSE_DURATION = Histogram(
    "rag_parse_duration_seconds",
    "Document parsing duration in seconds",
    ["parser"],
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

STORAGE_OPERATION_DURATION = Histogram(
    "rag_storage_operation_duration_seconds",
    "Storage operation duration in seconds",
    ["storage", "operation"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# -------------------------------------------------------------------------
# Gauges - Track current state values
# -------------------------------------------------------------------------

ACTIVE_CONNECTIONS = Gauge(
    "rag_active_connections",
    "Number of active connections",
    ["backend"],
)

CACHE_SIZE = Gauge(
    "rag_cache_size_items",
    "Number of items in cache",
    ["cache_type"],
)

CACHE_HIT_RATE = Gauge(
    "rag_cache_hit_rate",
    "Cache hit rate (0.0-1.0)",
    ["cache_type"],
)

DOCUMENTS_IN_STORAGE = Gauge(
    "rag_documents_in_storage",
    "Number of documents/chunks in storage",
    ["storage"],
)

RATE_LIMIT_AVAILABLE = Gauge(
    "rag_rate_limit_available_tokens",
    "Available rate limit tokens",
    ["service"],
)

RATE_LIMIT_WAIT_TIME = Histogram(
    "rag_rate_limit_wait_seconds",
    "Time spent waiting for rate limit tokens",
    ["service"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
)

# -------------------------------------------------------------------------
# Info - Application metadata
# -------------------------------------------------------------------------

APP_INFO = Info("rag_app", "Application information")
APP_INFO.info(
    {
        "version": "1.0.0",
        "parsers": "4",
        "storage_backends": "3",
        "python_version": "3.11",
    }
)


# -------------------------------------------------------------------------
# Decorator: Track execution time
# -------------------------------------------------------------------------


def track_time(
    metric: Histogram, labels: dict[str, str] | None = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to track execution time of async functions.

    Args:
        metric: Prometheus Histogram metric to record duration
        labels: Optional labels to apply to the metric

    Returns:
        Decorated function that tracks execution time

    Example:
        @track_time(QUERY_DURATION, labels={'parser': 'LlamaParse', 'storage': 'Chroma'})
        async def retrieve_docs(query: str):
            ...
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)
                return result
            except Exception as e:
                # Still record duration even on error
                duration = time.time() - start
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)
                raise

        return wrapper

    return decorator


# -------------------------------------------------------------------------
# Decorator: Track invocation counts
# -------------------------------------------------------------------------


def track_count(
    metric: Counter,
    labels: dict[str, str] | None = None,
    success_label: str = "status",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to track invocation counts of async functions.

    Automatically adds success/error status based on whether the function
    raises an exception.

    Args:
        metric: Prometheus Counter metric to increment
        labels: Base labels to apply to the metric
        success_label: Name of the label to use for success/error status (default: 'status')

    Returns:
        Decorated function that tracks invocation counts

    Example:
        @track_count(EMBEDDINGS_TOTAL)
        async def embed_text(text: str):
            ...
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                result = await func(*args, **kwargs)
                # Increment with success status
                if labels:
                    metric.labels(**{**labels, success_label: "success"}).inc()
                else:
                    metric.labels(**{success_label: "success"}).inc()
                return result
            except Exception as e:
                # Increment with error status
                if labels:
                    metric.labels(**{**labels, success_label: "error"}).inc()
                else:
                    metric.labels(**{success_label: "error"}).inc()
                raise

        return wrapper

    return decorator


# -------------------------------------------------------------------------
# Helper: Get batch size range label
# -------------------------------------------------------------------------


def get_batch_size_range(batch_size: int) -> str:
    """Get a label for batch size range.

    Args:
        batch_size: Number of items in batch

    Returns:
        Label string representing the batch size range
    """
    if batch_size == 1:
        return "single"
    elif batch_size <= 10:
        return "small_1-10"
    elif batch_size <= 50:
        return "medium_11-50"
    elif batch_size <= 100:
        return "large_51-100"
    else:
        return "xlarge_100+"


# -------------------------------------------------------------------------
# Helper: Update cache metrics
# -------------------------------------------------------------------------


def update_cache_metrics(cache_type: str, size: int, hit_rate: float) -> None:
    """Update cache-related metrics.

    Args:
        cache_type: Type of cache (e.g., 'embedding', 'llm')
        size: Current number of items in cache
        hit_rate: Cache hit rate (0.0-1.0)
    """
    CACHE_SIZE.labels(cache_type=cache_type).set(size)
    CACHE_HIT_RATE.labels(cache_type=cache_type).set(hit_rate)


# -------------------------------------------------------------------------
# Helper: Update storage document count
# -------------------------------------------------------------------------


def update_storage_document_count(storage: str, count: int) -> None:
    """Update document count in storage.

    Args:
        storage: Storage backend name (e.g., 'Chroma', 'Weaviate')
        count: Number of documents/chunks in storage
    """
    DOCUMENTS_IN_STORAGE.labels(storage=storage).set(count)


# -------------------------------------------------------------------------
# Helper: Track connection lifecycle
# -------------------------------------------------------------------------


class ConnectionTracker:
    """Context manager to track active connections.

    Example:
        async with ConnectionTracker('chroma'):
            # Connection is active
            await do_work()
        # Connection count decremented automatically
    """

    def __init__(self, backend: str) -> None:
        """Initialize connection tracker.

        Args:
            backend: Backend name (e.g., 'chroma', 'weaviate')
        """
        self.backend = backend

    async def __aenter__(self) -> "ConnectionTracker":
        """Increment active connection count."""
        ACTIVE_CONNECTIONS.labels(backend=self.backend).inc()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Decrement active connection count."""
        ACTIVE_CONNECTIONS.labels(backend=self.backend).dec()


# -------------------------------------------------------------------------
# Rate Limit Monitoring
# -------------------------------------------------------------------------


async def update_rate_limit_metrics(interval: float = 5.0) -> None:
    """Update rate limit metrics periodically.

    This function continuously monitors rate limiter state and updates
    Prometheus metrics. Should be run as a background task.

    Args:
        interval: Update interval in seconds (default: 5.0)
    """
    from utils.rate_limiter import rate_limiter

    logger.info(f"Starting rate limit metrics updates (interval={interval}s)")

    services = ["openai", "anthropic", "llamaparse", "vertex"]

    while True:
        try:
            for service in services:
                if rate_limiter.is_registered(service):
                    available = rate_limiter.get_available(service)
                    RATE_LIMIT_AVAILABLE.labels(service=service).set(available)

            await asyncio.sleep(interval)

        except asyncio.CancelledError:
            logger.info("Rate limit metrics update task cancelled")
            break
        except Exception as e:
            logger.error(f"Error updating rate limit metrics: {e}", exc_info=True)
            await asyncio.sleep(interval)


def get_rate_limit_status() -> dict[str, dict[str, float]]:
    """Get current rate limit status for all services.

    Returns:
        Dictionary mapping service names to their rate limit status:
        {
            "openai": {"available_tokens": 2500.5},
            "anthropic": {"available_tokens": 800.0},
            ...
        }
    """
    from utils.rate_limiter import rate_limiter

    services = ["openai", "anthropic", "llamaparse", "vertex"]
    status = {}

    for service in services:
        if rate_limiter.is_registered(service):
            status[service] = {
                "available_tokens": rate_limiter.get_available(service),
            }
        else:
            status[service] = {
                "available_tokens": 0.0,
            }

    return status
