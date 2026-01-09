"""Circuit breaker patterns for external API calls.

This module provides circuit breaker decorators and wrapper functions to protect
against cascading failures when external services (LLM, embeddings, parsers) are down.

Circuit breakers automatically "open" after a threshold of failures, preventing
wasted retry attempts and allowing services time to recover. After a recovery timeout,
the circuit moves to "half-open" state to test if the service has recovered.

States:
    - CLOSED: Normal operation, all calls pass through
    - OPEN: Service appears down, calls fail fast without trying
    - HALF_OPEN: Testing recovery, limited calls allowed through
"""

import asyncio
import logging
from functools import wraps
from typing import Any, Callable, TypeVar

from circuitbreaker import CircuitBreaker, CircuitBreakerError

logger = logging.getLogger(__name__)

# Type variable for generic function return type
T = TypeVar("T")

# -------------------------------------------------------------------------
# Circuit Breaker Instances
# -------------------------------------------------------------------------

# LLM Circuit Breaker
# Higher failure threshold since LLM calls are expensive and critical
llm_breaker = CircuitBreaker(
    failure_threshold=5,  # Open after 5 consecutive failures
    recovery_timeout=60,  # Wait 60 seconds before testing recovery
    expected_exception=Exception,  # Catch all exceptions
    name="llm_circuit_breaker",
)

# Embedding Circuit Breaker
# Moderate settings for embedding calls (less critical than LLM)
embedding_breaker = CircuitBreaker(
    failure_threshold=5,  # Open after 5 consecutive failures
    recovery_timeout=30,  # Wait 30 seconds before testing recovery
    expected_exception=Exception,
    name="embedding_circuit_breaker",
)

# Parser Circuit Breaker
# Lower threshold since parser failures may indicate document issues
parser_breaker = CircuitBreaker(
    failure_threshold=3,  # Open after 3 consecutive failures
    recovery_timeout=120,  # Wait 120 seconds before testing recovery (parsers may need more time)
    expected_exception=Exception,
    name="parser_circuit_breaker",
)


# -------------------------------------------------------------------------
# Async Circuit Breaker Wrapper Functions
# -------------------------------------------------------------------------


async def call_llm_with_breaker(
    func: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Execute LLM call with circuit breaker protection.

    Args:
        func: Async function to call
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        Result from func

    Raises:
        CircuitBreakerError: If circuit is open (service appears down)
        Exception: Original exception from func if circuit is closed
    """
    try:
        # Wrap the function call with circuit breaker
        @llm_breaker
        async def _protected_call() -> Any:
            return await func(*args, **kwargs)

        result = await _protected_call()
        logger.debug("LLM call succeeded through circuit breaker")
        return result

    except CircuitBreakerError as e:
        logger.error(
            "LLM circuit breaker is OPEN - service appears down. "
            f"Failing fast to prevent cascading failures. Error: {e}"
        )
        raise
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise


async def call_embedding_with_breaker(
    func: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Execute embedding call with circuit breaker protection.

    Args:
        func: Async function to call
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        Result from func

    Raises:
        CircuitBreakerError: If circuit is open (service appears down)
        Exception: Original exception from func if circuit is closed
    """
    try:
        # Wrap the function call with circuit breaker
        @embedding_breaker
        async def _protected_call() -> Any:
            return await func(*args, **kwargs)

        result = await _protected_call()
        logger.debug("Embedding call succeeded through circuit breaker")
        return result

    except CircuitBreakerError as e:
        logger.error(
            "Embedding circuit breaker is OPEN - service appears down. "
            f"Failing fast to prevent cascading failures. Error: {e}"
        )
        raise
    except Exception as e:
        logger.error(f"Embedding call failed: {e}")
        raise


async def call_parser_with_breaker(
    func: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Execute parser API call with circuit breaker protection.

    Args:
        func: Async function to call (parser API call)
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        Result from func

    Raises:
        CircuitBreakerError: If circuit is open (service appears down)
        Exception: Original exception from func if circuit is closed
    """
    try:
        # Wrap the function call with circuit breaker
        @parser_breaker
        async def _protected_call() -> Any:
            return await func(*args, **kwargs)

        result = await _protected_call()
        logger.debug("Parser API call succeeded through circuit breaker")
        return result

    except CircuitBreakerError as e:
        logger.error(
            "Parser circuit breaker is OPEN - service appears down. "
            f"Failing fast to prevent cascading failures. Error: {e}"
        )
        raise
    except Exception as e:
        logger.error(f"Parser API call failed: {e}")
        raise


# -------------------------------------------------------------------------
# Synchronous Circuit Breaker Wrapper Functions
# -------------------------------------------------------------------------


def call_sync_with_breaker(
    breaker: CircuitBreaker, func: Callable[..., T], *args: Any, **kwargs: Any
) -> T:
    """Execute synchronous call with circuit breaker protection.

    Args:
        breaker: CircuitBreaker instance to use
        func: Synchronous function to call
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        Result from func

    Raises:
        CircuitBreakerError: If circuit is open (service appears down)
        Exception: Original exception from func if circuit is closed
    """
    try:
        # Wrap the function call with circuit breaker
        @breaker
        def _protected_call() -> T:
            return func(*args, **kwargs)

        result = _protected_call()
        logger.debug(f"Sync call succeeded through {breaker.name}")
        return result

    except CircuitBreakerError as e:
        logger.error(
            f"{breaker.name} is OPEN - service appears down. "
            f"Failing fast to prevent cascading failures. Error: {e}"
        )
        raise
    except Exception as e:
        logger.error(f"Sync call failed through {breaker.name}: {e}")
        raise


# -------------------------------------------------------------------------
# Circuit Breaker Status and Monitoring
# -------------------------------------------------------------------------


def get_circuit_breaker_status() -> dict[str, dict[str, Any]]:
    """Get status of all circuit breakers.

    Returns:
        Dictionary mapping breaker names to their status information:
        - state: Current state (closed, open, half_open)
        - failure_count: Number of consecutive failures
        - failure_threshold: Threshold before opening
        - recovery_timeout: Seconds to wait before testing recovery
        - last_failure_time: Timestamp of last failure (if any)
    """
    breakers = {
        "llm": llm_breaker,
        "embedding": embedding_breaker,
        "parser": parser_breaker,
    }

    status = {}
    for name, breaker in breakers.items():
        status[name] = {
            "state": breaker.state,
            "failure_count": breaker.failure_count,
            "failure_threshold": breaker._failure_threshold,
            "recovery_timeout_seconds": breaker._recovery_timeout,
            "last_failure_time": breaker.last_failure if breaker.last_failure else None,
        }

    return status


def reset_circuit_breakers() -> None:
    """Reset all circuit breakers to closed state.

    Useful for testing or manual recovery intervention.
    """
    logger.info("Resetting all circuit breakers to closed state")
    llm_breaker.__init__(
        failure_threshold=5,
        recovery_timeout=60,
        expected_exception=Exception,
        name="llm_circuit_breaker",
    )
    embedding_breaker.__init__(
        failure_threshold=5,
        recovery_timeout=30,
        expected_exception=Exception,
        name="embedding_circuit_breaker",
    )
    parser_breaker.__init__(
        failure_threshold=3,
        recovery_timeout=120,
        expected_exception=Exception,
        name="parser_circuit_breaker",
    )


def log_circuit_breaker_status() -> None:
    """Log current status of all circuit breakers."""
    status = get_circuit_breaker_status()
    logger.info("Circuit Breaker Status:")
    for name, info in status.items():
        logger.info(
            f"  {name}: state={info['state']}, "
            f"failures={info['failure_count']}/{info['failure_threshold']}, "
            f"recovery_timeout={info['recovery_timeout_seconds']}s"
        )


# -------------------------------------------------------------------------
# Decorator for automatic circuit breaker wrapping
# -------------------------------------------------------------------------


def with_circuit_breaker(breaker: CircuitBreaker) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to wrap async functions with circuit breaker protection.

    Args:
        breaker: CircuitBreaker instance to use

    Returns:
        Decorator function

    Example:
        @with_circuit_breaker(llm_breaker)
        async def call_api():
            return await api.call()
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                return await call_llm_with_breaker(func, *args, **kwargs)

            return async_wrapper  # type: ignore
        else:

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> T:
                return call_sync_with_breaker(breaker, func, *args, **kwargs)

            return sync_wrapper  # type: ignore

    return decorator
