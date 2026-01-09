"""Tests for the Evaluator class.

Tests answer generation, query evaluation, and aggregate metrics calculation.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest

from evaluation.evaluator import Evaluator
from models import BenchmarkQuery, DifficultyLevel, QueryType, RetrievalResult


@pytest.fixture
def evaluator():
    """Create an Evaluator instance."""
    return Evaluator()


@pytest.fixture
def sample_query():
    """Create a sample benchmark query."""
    return BenchmarkQuery(
        query_id="query_1",
        query="What are the key considerations for drilling mud systems?",
        ground_truth_answer="Key considerations include mud density, viscosity, and filtration control.",
        relevant_element_ids=["chunk_1", "chunk_2", "chunk_3"],
        query_type=QueryType.SEMANTIC,
        difficulty=DifficultyLevel.MEDIUM,
    )


@pytest.fixture
def sample_retrieved_results():
    """Create sample retrieval results."""
    return [
        RetrievalResult(
            chunk_id="chunk_1",
            document_id="doc_1",
            content="Drilling mud density must be carefully controlled to prevent blowouts.",
            score=0.95,
            rank=1,
            metadata={"page": "12", "section": "Mud Systems"},
        ),
        RetrievalResult(
            chunk_id="chunk_2",
            document_id="doc_1",
            content="Viscosity management is critical for effective hole cleaning.",
            score=0.87,
            rank=2,
            metadata={"page": "13", "section": "Mud Systems"},
        ),
        RetrievalResult(
            chunk_id="chunk_3",
            document_id="doc_2",
            content="Filtration control prevents formation damage.",
            score=0.75,
            rank=3,
            metadata={"page": "45", "section": "Formation Protection"},
        ),
    ]


class TestAnswerGeneration:
    """Tests for answer generation."""

    @pytest.mark.asyncio
    async def test_generate_answer_with_context(self, evaluator, sample_retrieved_results):
        """Test answer generation with retrieved context."""
        query = "What are the key considerations for drilling mud systems?"

        # Mock the Anthropic API call
        with patch.object(evaluator.client.messages, "create", new_callable=AsyncMock) as mock_create:
            mock_response = Mock()
            mock_response.content = [Mock(text="Key considerations include mud density, viscosity, and filtration control.")]
            mock_create.return_value = mock_response

            answer = await evaluator.generate_answer(query, sample_retrieved_results)

            assert isinstance(answer, str)
            assert len(answer) > 0
            # Answer should not be the default "no context" message
            assert "no relevant context" not in answer.lower()

    @pytest.mark.asyncio
    async def test_generate_answer_empty_context(self, evaluator):
        """Test answer generation with no context."""
        query = "Test query"

        answer = await evaluator.generate_answer(query, [])

        assert isinstance(answer, str)
        assert "no relevant context" in answer.lower() or "cannot answer" in answer.lower()

    @pytest.mark.asyncio
    async def test_generate_answer_formats_context(self, evaluator, sample_retrieved_results):
        """Test that context is properly formatted for the LLM."""
        query = "Test query"

        # Mock the LLM call to capture the formatted prompt
        with patch.object(evaluator.client.messages, "create", new_callable=AsyncMock) as mock_create:
            mock_response = Mock()
            mock_response.content = [Mock(text="Test answer")]
            mock_create.return_value = mock_response

            await evaluator.generate_answer(query, sample_retrieved_results)

            # Verify the call was made
            assert mock_create.called
            call_kwargs = mock_create.call_args.kwargs

            # Check that messages contain formatted chunks
            messages = call_kwargs["messages"]
            user_message = messages[0]["content"]

            # Should contain chunk headers and relevance scores
            assert "[Chunk 1]" in user_message
            assert "[Chunk 2]" in user_message
            assert "Relevance:" in user_message


class TestQueryEvaluation:
    """Tests for single query evaluation."""

    @pytest.mark.asyncio
    async def test_evaluate_query_structure(
        self, evaluator, sample_query, sample_retrieved_results
    ):
        """Test that evaluate_query returns proper BenchmarkResult structure."""
        generated_answer = "Drilling mud systems require density, viscosity, and filtration control."

        result = await evaluator.evaluate_query(
            query=sample_query,
            retrieved=sample_retrieved_results,
            generated_answer=generated_answer,
            parser_name="llama_parse",
            storage_backend="chroma",
            retrieval_time=0.5,
            generation_time=2.0,
        )

        # Check result structure
        assert result.query_id == sample_query.query_id
        assert result.query == sample_query.query
        assert result.generated_answer == generated_answer
        assert result.ground_truth_answer == sample_query.ground_truth_answer
        assert result.parser_name == "llama_parse"
        assert result.storage_backend == "chroma"
        assert result.retrieval_time_seconds == 0.5
        assert result.generation_time_seconds == 2.0
        assert result.total_time_seconds == 2.5

    @pytest.mark.asyncio
    async def test_evaluate_query_calculates_ir_metrics(
        self, evaluator, sample_query, sample_retrieved_results
    ):
        """Test that IR metrics are calculated correctly."""
        generated_answer = "Test answer"

        result = await evaluator.evaluate_query(
            query=sample_query,
            retrieved=sample_retrieved_results,
            generated_answer=generated_answer,
            parser_name="test_parser",
            storage_backend="test_backend",
        )

        # Check that standard IR metrics are present
        assert "precision@1" in result.metrics
        assert "precision@3" in result.metrics
        assert "precision@5" in result.metrics
        assert "recall@1" in result.metrics
        assert "recall@3" in result.metrics
        assert "recall@5" in result.metrics
        assert "f1@1" in result.metrics
        assert "mrr" in result.metrics
        assert "map" in result.metrics
        assert "ndcg@1" in result.metrics

    @pytest.mark.asyncio
    async def test_evaluate_query_calculates_llm_metrics(
        self, evaluator, sample_query, sample_retrieved_results
    ):
        """Test that LLM-based metrics are calculated."""
        generated_answer = "Test answer"

        # Mock the metrics calculator's async methods
        with patch.object(
            evaluator.metrics_calculator, "evaluate_context_relevance", new_callable=AsyncMock
        ) as mock_context, patch.object(
            evaluator.metrics_calculator, "evaluate_answer_correctness", new_callable=AsyncMock
        ) as mock_correctness, patch.object(
            evaluator.metrics_calculator, "evaluate_faithfulness", new_callable=AsyncMock
        ) as mock_faithfulness:

            # Set up mocks
            mock_context.return_value = {
                "overall_score": 0.8,
                "chunk_scores": [0.9, 0.8, 0.7],
                "reasoning": "Test",
            }
            mock_correctness.return_value = {
                "correctness_score": 0.85,
                "semantic_similarity": 0.9,
                "factual_accuracy": 0.95,
                "completeness": 0.8,
                "reasoning": "Test",
            }
            mock_faithfulness.return_value = {
                "faithfulness_score": 0.9,
                "hallucination_count": 0,
                "supported_claims": ["claim1"],
                "unsupported_claims": [],
                "reasoning": "Test",
            }

            result = await evaluator.evaluate_query(
                query=sample_query,
                retrieved=sample_retrieved_results,
                generated_answer=generated_answer,
                parser_name="test_parser",
                storage_backend="test_backend",
            )

            # Check that LLM metrics are present
            assert "context_relevance" in result.metrics
            assert "answer_correctness" in result.metrics
            assert "semantic_similarity" in result.metrics
            assert "factual_accuracy" in result.metrics
            assert "completeness" in result.metrics
            assert "faithfulness" in result.metrics
            assert "hallucination_count" in result.metrics

            # Check values
            assert result.metrics["context_relevance"] == 0.8
            assert result.metrics["answer_correctness"] == 0.85
            assert result.metrics["faithfulness"] == 0.9

    @pytest.mark.asyncio
    async def test_evaluate_query_handles_llm_error(
        self, evaluator, sample_query, sample_retrieved_results
    ):
        """Test that evaluation continues even if LLM metrics fail."""
        generated_answer = "Test answer"

        # Mock the metrics calculator to raise an exception
        with patch.object(
            evaluator.metrics_calculator, "evaluate_context_relevance", new_callable=AsyncMock
        ) as mock_context:
            mock_context.side_effect = Exception("API error")

            result = await evaluator.evaluate_query(
                query=sample_query,
                retrieved=sample_retrieved_results,
                generated_answer=generated_answer,
                parser_name="test_parser",
                storage_backend="test_backend",
            )

            # Should still have IR metrics
            assert "precision@1" in result.metrics
            assert "mrr" in result.metrics

            # LLM metrics should be 0.0
            assert result.metrics["context_relevance"] == 0.0
            assert result.metrics["answer_correctness"] == 0.0
            assert result.metrics["faithfulness"] == 0.0


class TestBatchEvaluation:
    """Tests for batch query evaluation."""

    @pytest.mark.asyncio
    async def test_evaluate_batch(self, evaluator):
        """Test evaluating multiple queries in batch."""
        queries = [
            BenchmarkQuery(
                query_id="query_1",
                query="Query 1",
                ground_truth_answer="Answer 1",
                relevant_element_ids=["chunk_1"],
            ),
            BenchmarkQuery(
                query_id="query_2",
                query="Query 2",
                ground_truth_answer="Answer 2",
                relevant_element_ids=["chunk_2"],
            ),
        ]

        retrieval_results = {
            "query_1": [
                RetrievalResult(
                    chunk_id="chunk_1",
                    document_id="doc_1",
                    content="Content 1",
                    score=0.9,
                    rank=1,
                )
            ],
            "query_2": [
                RetrievalResult(
                    chunk_id="chunk_2",
                    document_id="doc_2",
                    content="Content 2",
                    score=0.85,
                    rank=1,
                )
            ],
        }

        # Mock the generate_answer method to avoid API calls
        with patch.object(evaluator.client.messages, "create", new_callable=AsyncMock) as mock_create:
            mock_response = Mock()
            mock_response.content = [Mock(text="Test answer")]
            mock_create.return_value = mock_response

            with patch.object(evaluator.metrics_calculator, "evaluate_context_relevance", new_callable=AsyncMock) as mock_ctx:
                mock_ctx.return_value = {
                    "overall_score": 0.8,
                    "chunk_scores": [0.8],
                    "reasoning": "Test",
                }

                with patch.object(
                    evaluator.metrics_calculator, "evaluate_answer_correctness", new_callable=AsyncMock
                ) as mock_corr:
                    mock_corr.return_value = {
                        "correctness_score": 0.85,
                        "semantic_similarity": 0.9,
                        "factual_accuracy": 0.95,
                        "completeness": 0.8,
                        "reasoning": "Test",
                    }

                    with patch.object(
                        evaluator.metrics_calculator, "evaluate_faithfulness", new_callable=AsyncMock
                    ) as mock_faith:
                        mock_faith.return_value = {
                            "faithfulness_score": 0.9,
                            "hallucination_count": 0,
                            "supported_claims": [],
                            "unsupported_claims": [],
                            "reasoning": "Test",
                        }

                        results = await evaluator.evaluate_batch(
                            queries=queries,
                            retrieval_results=retrieval_results,
                            parser_name="test_parser",
                            storage_backend="test_backend",
                        )

        assert len(results) == 2
        assert results[0].query_id == "query_1"
        assert results[1].query_id == "query_2"


class TestAggregateMetrics:
    """Tests for aggregate metrics calculation."""

    def test_calculate_aggregate_metrics(self, evaluator, sample_query):
        """Test calculation of aggregate metrics."""
        from models import BenchmarkResult

        results = [
            BenchmarkResult(
                benchmark_id="test_1",
                parser_name="parser1",
                storage_backend="backend1",
                query_id="q1",
                query="Query 1",
                retrieved_results=[],
                generated_answer="Answer 1",
                ground_truth_answer="GT 1",
                metrics={
                    "precision@1": 1.0,
                    "recall@1": 0.8,
                    "mrr": 1.0,
                },
                retrieval_time_seconds=0.5,
                generation_time_seconds=2.0,
                total_time_seconds=2.5,
            ),
            BenchmarkResult(
                benchmark_id="test_2",
                parser_name="parser1",
                storage_backend="backend1",
                query_id="q2",
                query="Query 2",
                retrieved_results=[],
                generated_answer="Answer 2",
                ground_truth_answer="GT 2",
                metrics={
                    "precision@1": 0.8,
                    "recall@1": 0.6,
                    "mrr": 0.5,
                },
                retrieval_time_seconds=0.6,
                generation_time_seconds=2.2,
                total_time_seconds=2.8,
            ),
        ]

        aggregates = evaluator.calculate_aggregate_metrics(results)

        # Check mean calculations
        assert aggregates["precision@1_mean"] == pytest.approx(0.9)
        assert aggregates["recall@1_mean"] == pytest.approx(0.7)
        assert aggregates["mrr_mean"] == pytest.approx(0.75)

        # Check timing aggregates
        assert aggregates["avg_retrieval_time"] == pytest.approx(0.55)
        assert aggregates["avg_generation_time"] == pytest.approx(2.1)
        assert aggregates["avg_total_time"] == pytest.approx(2.65)

        # Check success rate
        assert aggregates["success_rate"] == 1.0

    def test_calculate_aggregate_metrics_empty(self, evaluator):
        """Test aggregate metrics with empty results."""
        aggregates = evaluator.calculate_aggregate_metrics([])
        assert aggregates == {}

    def test_calculate_aggregate_metrics_with_failures(self, evaluator):
        """Test aggregate metrics with some failures."""
        from models import BenchmarkResult

        results = [
            BenchmarkResult(
                benchmark_id="test_1",
                parser_name="parser1",
                storage_backend="backend1",
                query_id="q1",
                query="Query 1",
                retrieved_results=[],
                generated_answer="Answer 1",
                ground_truth_answer="GT 1",
                metrics={"precision@1": 1.0},
            ),
            BenchmarkResult(
                benchmark_id="test_2",
                parser_name="parser1",
                storage_backend="backend1",
                query_id="q2",
                query="Query 2",
                retrieved_results=[],
                generated_answer="",
                ground_truth_answer="GT 2",
                metrics={"precision@1": 0.8},
                error="Failed to retrieve",
            ),
        ]

        aggregates = evaluator.calculate_aggregate_metrics(results)

        # Success rate should be 0.5 (1 success, 1 failure)
        assert aggregates["success_rate"] == 0.5


class TestHelperMethods:
    """Tests for evaluator helper methods."""

    def test_format_context(self, evaluator, sample_retrieved_results):
        """Test context formatting for LLM."""
        formatted = evaluator._format_context(sample_retrieved_results)

        assert "[Chunk 1]" in formatted
        assert "[Chunk 2]" in formatted
        assert "[Chunk 3]" in formatted
        assert "Relevance: 0.950" in formatted
        assert "Relevance: 0.870" in formatted
        assert "Mud Systems" in formatted  # Metadata
        assert sample_retrieved_results[0].content in formatted

    def test_median_odd_count(self, evaluator):
        """Test median calculation with odd number of values."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        median = evaluator._median(values)
        assert median == 3.0

    def test_median_even_count(self, evaluator):
        """Test median calculation with even number of values."""
        values = [1.0, 2.0, 3.0, 4.0]
        median = evaluator._median(values)
        assert median == 2.5

    def test_median_empty(self, evaluator):
        """Test median calculation with empty list."""
        median = evaluator._median([])
        assert median == 0.0

    def test_std_dev(self, evaluator):
        """Test standard deviation calculation."""
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        std = evaluator._std_dev(values)
        assert std == pytest.approx(2.0, rel=0.01)

    def test_std_dev_empty(self, evaluator):
        """Test standard deviation with empty list."""
        std = evaluator._std_dev([])
        assert std == 0.0
