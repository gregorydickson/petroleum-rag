"""Tests for circuit breaker functionality.

This module tests the circuit breaker pattern for external API calls,
ensuring that cascading failures are prevented and services can recover.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from circuitbreaker import CircuitBreakerError

from utils.circuit_breaker import (
    call_embedding_with_breaker,
    call_llm_with_breaker,
    call_parser_with_breaker,
    call_sync_with_breaker,
    embedding_breaker,
    get_circuit_breaker_status,
    llm_breaker,
    log_circuit_breaker_status,
    parser_breaker,
    reset_circuit_breakers,
)


@pytest.fixture(autouse=True)
def reset_breakers():
    """Reset all circuit breakers before each test."""
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


class TestCircuitBreakerBasics:
    """Test basic circuit breaker functionality."""

    async def test_llm_breaker_allows_successful_calls(self):
        """Test that LLM circuit breaker allows successful calls through."""
        mock_func = AsyncMock(return_value="success")

        result = await call_llm_with_breaker(mock_func)

        assert result == "success"
        mock_func.assert_called_once()

    async def test_embedding_breaker_allows_successful_calls(self):
        """Test that embedding circuit breaker allows successful calls through."""
        mock_func = AsyncMock(return_value=[0.1, 0.2, 0.3])

        result = await call_embedding_with_breaker(mock_func)

        assert result == [0.1, 0.2, 0.3]
        mock_func.assert_called_once()

    async def test_parser_breaker_allows_successful_calls(self):
        """Test that parser circuit breaker allows successful calls through."""
        mock_func = AsyncMock(return_value={"parsed": "data"})

        result = await call_parser_with_breaker(mock_func)

        assert result == {"parsed": "data"}
        mock_func.assert_called_once()


class TestCircuitBreakerFailureHandling:
    """Test circuit breaker behavior under failures."""

    async def test_llm_breaker_opens_after_threshold_failures(self):
        """Test that LLM circuit breaker opens after failure threshold."""
        mock_func = AsyncMock(side_effect=Exception("API Error"))

        # Cause 5 failures (threshold for llm_breaker)
        for _ in range(5):
            with pytest.raises(Exception):
                await call_llm_with_breaker(mock_func)

        # Circuit should now be open - next call should fail fast
        with pytest.raises(CircuitBreakerError):
            await call_llm_with_breaker(mock_func)

        # Should have made 5 calls (not 6, because 6th was blocked by circuit)
        assert mock_func.call_count == 5

    async def test_embedding_breaker_opens_after_threshold_failures(self):
        """Test that embedding circuit breaker opens after failure threshold."""
        mock_func = AsyncMock(side_effect=Exception("Embedding API Error"))

        # Cause 5 failures (threshold for embedding_breaker)
        for _ in range(5):
            with pytest.raises(Exception):
                await call_embedding_with_breaker(mock_func)

        # Circuit should now be open
        with pytest.raises(CircuitBreakerError):
            await call_embedding_with_breaker(mock_func)

        assert mock_func.call_count == 5

    async def test_parser_breaker_opens_after_threshold_failures(self):
        """Test that parser circuit breaker opens after failure threshold."""
        mock_func = AsyncMock(side_effect=Exception("Parser API Error"))

        # Cause 3 failures (threshold for parser_breaker)
        for _ in range(3):
            with pytest.raises(Exception):
                await call_parser_with_breaker(mock_func)

        # Circuit should now be open
        with pytest.raises(CircuitBreakerError):
            await call_parser_with_breaker(mock_func)

        assert mock_func.call_count == 3


class TestCircuitBreakerRecovery:
    """Test circuit breaker recovery behavior."""

    async def test_circuit_recovers_after_timeout(self):
        """Test that circuit breaker recovers after timeout period."""
        # Create a breaker with very short recovery timeout for testing
        from circuitbreaker import CircuitBreaker

        test_breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=1,  # 1 second recovery
            expected_exception=Exception,
            name="test_breaker",
        )

        mock_func = AsyncMock(side_effect=Exception("Error"))

        # Cause failures to open circuit
        for _ in range(2):
            with pytest.raises(Exception):

                @test_breaker
                async def protected_call():
                    return await mock_func()

                await protected_call()

        # Circuit should be open
        with pytest.raises(CircuitBreakerError):

            @test_breaker
            async def protected_call():
                return await mock_func()

            await protected_call()

        # Wait for recovery timeout
        await asyncio.sleep(1.5)

        # Circuit should allow one call through (half-open state)
        # If it succeeds, circuit closes; if it fails, circuit opens again
        mock_func.side_effect = None
        mock_func.return_value = "recovered"

        @test_breaker
        async def protected_call():
            return await mock_func()

        result = await protected_call()
        assert result == "recovered"

    async def test_circuit_stays_open_on_continued_failures(self):
        """Test that circuit stays open if service continues to fail."""
        from circuitbreaker import CircuitBreaker

        test_breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=1,
            expected_exception=Exception,
            name="test_breaker_fail",
        )

        mock_func = AsyncMock(side_effect=Exception("Continued Error"))

        # Open the circuit
        for _ in range(2):
            with pytest.raises(Exception):

                @test_breaker
                async def protected_call():
                    return await mock_func()

                await protected_call()

        # Wait for recovery timeout
        await asyncio.sleep(1.5)

        # Circuit allows one test call, which fails
        with pytest.raises(Exception):

            @test_breaker
            async def protected_call():
                return await mock_func()

            await protected_call()

        # Circuit should be open again
        with pytest.raises(CircuitBreakerError):

            @test_breaker
            async def protected_call():
                return await mock_func()

            await protected_call()


class TestCircuitBreakerWithArguments:
    """Test circuit breaker with function arguments."""

    async def test_llm_breaker_passes_args_correctly(self):
        """Test that circuit breaker correctly passes arguments to wrapped function."""
        mock_func = AsyncMock(return_value="result")

        result = await call_llm_with_breaker(mock_func, "arg1", "arg2", kwarg1="value1")

        assert result == "result"
        mock_func.assert_called_once_with("arg1", "arg2", kwarg1="value1")

    async def test_embedding_breaker_passes_args_correctly(self):
        """Test that embedding circuit breaker passes arguments correctly."""
        mock_func = AsyncMock(return_value=[0.1, 0.2])

        result = await call_embedding_with_breaker(
            mock_func, ["text1", "text2"], batch_size=10
        )

        assert result == [0.1, 0.2]
        mock_func.assert_called_once_with(["text1", "text2"], batch_size=10)


class TestSynchronousCircuitBreaker:
    """Test synchronous circuit breaker wrapper."""

    def test_sync_breaker_allows_successful_calls(self):
        """Test that sync circuit breaker allows successful calls."""
        mock_func = MagicMock(return_value="sync_result")

        result = call_sync_with_breaker(llm_breaker, mock_func, "arg1")

        assert result == "sync_result"
        mock_func.assert_called_once_with("arg1")

    def test_sync_breaker_opens_on_failures(self):
        """Test that sync circuit breaker opens on failures."""
        mock_func = MagicMock(side_effect=Exception("Sync Error"))

        # Reset breaker first
        reset_circuit_breakers()

        # Cause 5 failures
        for _ in range(5):
            with pytest.raises(Exception):
                call_sync_with_breaker(llm_breaker, mock_func)

        # Circuit should be open
        with pytest.raises(CircuitBreakerError):
            call_sync_with_breaker(llm_breaker, mock_func)

        assert mock_func.call_count == 5


class TestCircuitBreakerStatus:
    """Test circuit breaker status monitoring."""

    def test_get_circuit_breaker_status(self):
        """Test getting status of all circuit breakers."""
        status = get_circuit_breaker_status()

        assert "llm" in status
        assert "embedding" in status
        assert "parser" in status

        # Check status structure
        for breaker_name, breaker_status in status.items():
            assert "state" in breaker_status
            assert "failure_count" in breaker_status
            assert "failure_threshold" in breaker_status
            assert "recovery_timeout_seconds" in breaker_status

    def test_status_reflects_breaker_state(self):
        """Test that status reflects actual breaker state."""
        # Initially all breakers should be closed
        status = get_circuit_breaker_status()
        assert status["llm"]["state"] == "closed"
        assert status["llm"]["failure_count"] == 0

    def test_log_circuit_breaker_status(self, caplog):
        """Test logging circuit breaker status."""
        with caplog.at_level("INFO"):
            log_circuit_breaker_status()

        assert "Circuit Breaker Status" in caplog.text
        assert "llm" in caplog.text
        assert "embedding" in caplog.text
        assert "parser" in caplog.text


class TestCircuitBreakerReset:
    """Test circuit breaker reset functionality."""

    async def test_reset_closes_open_circuits(self):
        """Test that reset closes open circuits."""
        mock_func = AsyncMock(side_effect=Exception("Error"))

        # Open the LLM circuit
        for _ in range(5):
            with pytest.raises(Exception):
                await call_llm_with_breaker(mock_func)

        # Verify circuit is open
        with pytest.raises(CircuitBreakerError):
            await call_llm_with_breaker(mock_func)

        # Reset all breakers
        reset_circuit_breakers()

        # Now should be able to call again (will fail, but won't be blocked)
        with pytest.raises(Exception):
            await call_llm_with_breaker(mock_func)

    def test_reset_clears_failure_counts(self):
        """Test that reset clears failure counts."""
        mock_func = AsyncMock(side_effect=Exception("Error"))

        # Cause some failures (but not enough to open)
        for _ in range(3):
            with pytest.raises(Exception):

                async def call():
                    await call_llm_with_breaker(mock_func)

                asyncio.run(call())

        # Reset
        reset_circuit_breakers()

        # Status should show 0 failures
        status = get_circuit_breaker_status()
        assert status["llm"]["failure_count"] == 0


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breakers with actual components."""

    async def test_circuit_breaker_prevents_cascading_failures(self):
        """Test that circuit breaker prevents cascading failures in a realistic scenario."""
        call_count = 0
        max_calls = 10

        async def failing_api_call():
            nonlocal call_count
            call_count += 1
            if call_count > max_calls:
                pytest.fail("Circuit breaker failed to prevent excessive calls")
            raise Exception("Service Down")

        # Make calls until circuit opens
        failure_count = 0
        blocked_count = 0

        for _ in range(20):  # Try to make 20 calls
            try:
                await call_llm_with_breaker(failing_api_call)
            except CircuitBreakerError:
                blocked_count += 1
            except Exception:
                failure_count += 1

        # Should have 5 actual failures (threshold) and rest blocked
        assert failure_count == 5
        assert blocked_count == 15
        assert call_count == 5  # Only 5 actual API calls made

    async def test_different_breakers_are_independent(self):
        """Test that different circuit breakers operate independently."""
        llm_func = AsyncMock(side_effect=Exception("LLM Error"))
        embedding_func = AsyncMock(return_value=[0.1, 0.2])

        # Open LLM circuit
        for _ in range(5):
            with pytest.raises(Exception):
                await call_llm_with_breaker(llm_func)

        # LLM circuit should be open
        with pytest.raises(CircuitBreakerError):
            await call_llm_with_breaker(llm_func)

        # But embedding circuit should still work
        result = await call_embedding_with_breaker(embedding_func)
        assert result == [0.1, 0.2]

        # Verify status
        status = get_circuit_breaker_status()
        assert status["llm"]["state"] == "open"
        assert status["embedding"]["state"] == "closed"


class TestCircuitBreakerErrorHandling:
    """Test error handling in circuit breaker wrappers."""

    async def test_circuit_breaker_logs_errors(self, caplog):
        """Test that circuit breaker logs errors appropriately."""
        mock_func = AsyncMock(side_effect=Exception("Test Error"))

        with caplog.at_level("ERROR"):
            with pytest.raises(Exception):
                await call_llm_with_breaker(mock_func)

        assert "LLM call failed" in caplog.text

    async def test_circuit_breaker_logs_open_state(self, caplog):
        """Test that circuit breaker logs when circuit opens."""
        mock_func = AsyncMock(side_effect=Exception("Error"))

        # Open the circuit
        for _ in range(5):
            with pytest.raises(Exception):
                await call_llm_with_breaker(mock_func)

        # Try to call with open circuit
        with caplog.at_level("ERROR"):
            with pytest.raises(CircuitBreakerError):
                await call_llm_with_breaker(mock_func)

        assert "circuit breaker is OPEN" in caplog.text
        assert "service appears down" in caplog.text
