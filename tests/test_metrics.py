"""Tests for evaluation metrics.

Tests both traditional IR metrics and LLM-based evaluation metrics.
"""

from unittest.mock import AsyncMock, patch

import pytest

from evaluation.metrics import MetricsCalculator
from models import RetrievalResult


@pytest.fixture
def metrics_calculator():
    """Create a MetricsCalculator instance."""
    return MetricsCalculator()


@pytest.fixture
def sample_retrieved_results():
    """Create sample retrieval results for testing."""
    return [
        RetrievalResult(
            chunk_id="chunk_1",
            document_id="doc_1",
            content="Content about drilling operations and mud systems.",
            score=0.95,
            rank=1,
        ),
        RetrievalResult(
            chunk_id="chunk_2",
            document_id="doc_1",
            content="Information about wellbore pressure management.",
            score=0.87,
            rank=2,
        ),
        RetrievalResult(
            chunk_id="chunk_3",
            document_id="doc_2",
            content="Data on reservoir characteristics.",
            score=0.75,
            rank=3,
        ),
        RetrievalResult(
            chunk_id="chunk_4",
            document_id="doc_2",
            content="Technical specifications for casing.",
            score=0.68,
            rank=4,
        ),
        RetrievalResult(
            chunk_id="chunk_5",
            document_id="doc_3",
            content="Irrelevant content about different topic.",
            score=0.45,
            rank=5,
        ),
    ]


class TestPrecisionAtK:
    """Tests for Precision@K metric."""

    def test_precision_at_k_all_relevant(self, metrics_calculator, sample_retrieved_results):
        """Test precision when all retrieved documents are relevant."""
        relevant_ids = ["chunk_1", "chunk_2", "chunk_3", "chunk_4", "chunk_5"]
        precision = metrics_calculator.calculate_precision_at_k(
            sample_retrieved_results, relevant_ids, k=5
        )
        assert precision == 1.0

    def test_precision_at_k_half_relevant(self, metrics_calculator, sample_retrieved_results):
        """Test precision when half of retrieved documents are relevant."""
        relevant_ids = ["chunk_1", "chunk_2", "chunk_3"]
        precision = metrics_calculator.calculate_precision_at_k(
            sample_retrieved_results, relevant_ids, k=5
        )
        assert precision == 0.6  # 3 relevant out of 5

    def test_precision_at_k_none_relevant(self, metrics_calculator, sample_retrieved_results):
        """Test precision when no retrieved documents are relevant."""
        relevant_ids = ["chunk_99", "chunk_100"]
        precision = metrics_calculator.calculate_precision_at_k(
            sample_retrieved_results, relevant_ids, k=5
        )
        assert precision == 0.0

    def test_precision_at_k_smaller_k(self, metrics_calculator, sample_retrieved_results):
        """Test precision with k smaller than result set."""
        relevant_ids = ["chunk_1", "chunk_2"]
        precision = metrics_calculator.calculate_precision_at_k(
            sample_retrieved_results, relevant_ids, k=2
        )
        assert precision == 1.0  # Both top-2 are relevant

    def test_precision_at_k_empty_results(self, metrics_calculator):
        """Test precision with empty results."""
        precision = metrics_calculator.calculate_precision_at_k([], ["chunk_1"], k=5)
        assert precision == 0.0

    def test_precision_at_k_zero_k(self, metrics_calculator, sample_retrieved_results):
        """Test precision with k=0."""
        precision = metrics_calculator.calculate_precision_at_k(
            sample_retrieved_results, ["chunk_1"], k=0
        )
        assert precision == 0.0


class TestRecallAtK:
    """Tests for Recall@K metric."""

    def test_recall_at_k_all_found(self, metrics_calculator, sample_retrieved_results):
        """Test recall when all relevant documents are in top-K."""
        relevant_ids = ["chunk_1", "chunk_2"]
        recall = metrics_calculator.calculate_recall_at_k(
            sample_retrieved_results, relevant_ids, k=5
        )
        assert recall == 1.0

    def test_recall_at_k_partial_found(self, metrics_calculator, sample_retrieved_results):
        """Test recall when some relevant documents are in top-K."""
        relevant_ids = ["chunk_1", "chunk_2", "chunk_6", "chunk_7"]
        recall = metrics_calculator.calculate_recall_at_k(
            sample_retrieved_results, relevant_ids, k=5
        )
        assert recall == 0.5  # 2 out of 4 found

    def test_recall_at_k_none_found(self, metrics_calculator, sample_retrieved_results):
        """Test recall when no relevant documents are found."""
        relevant_ids = ["chunk_99", "chunk_100"]
        recall = metrics_calculator.calculate_recall_at_k(
            sample_retrieved_results, relevant_ids, k=5
        )
        assert recall == 0.0

    def test_recall_at_k_smaller_k(self, metrics_calculator, sample_retrieved_results):
        """Test recall with k smaller than number of relevant docs."""
        relevant_ids = ["chunk_1", "chunk_2", "chunk_3"]
        recall = metrics_calculator.calculate_recall_at_k(
            sample_retrieved_results, relevant_ids, k=2
        )
        assert recall == pytest.approx(0.666, rel=0.01)  # 2 out of 3

    def test_recall_at_k_empty_relevant(self, metrics_calculator, sample_retrieved_results):
        """Test recall with empty relevant set."""
        recall = metrics_calculator.calculate_recall_at_k(sample_retrieved_results, [], k=5)
        assert recall == 0.0


class TestMRR:
    """Tests for Mean Reciprocal Rank metric."""

    def test_mrr_first_rank(self, metrics_calculator, sample_retrieved_results):
        """Test MRR when first result is relevant."""
        relevant_ids = ["chunk_1"]
        mrr = metrics_calculator.calculate_mrr(sample_retrieved_results, relevant_ids)
        assert mrr == 1.0

    def test_mrr_second_rank(self, metrics_calculator, sample_retrieved_results):
        """Test MRR when second result is relevant."""
        relevant_ids = ["chunk_2"]
        mrr = metrics_calculator.calculate_mrr(sample_retrieved_results, relevant_ids)
        assert mrr == 0.5

    def test_mrr_third_rank(self, metrics_calculator, sample_retrieved_results):
        """Test MRR when third result is relevant."""
        relevant_ids = ["chunk_3"]
        mrr = metrics_calculator.calculate_mrr(sample_retrieved_results, relevant_ids)
        assert mrr == pytest.approx(0.333, rel=0.01)

    def test_mrr_no_relevant(self, metrics_calculator, sample_retrieved_results):
        """Test MRR when no relevant documents found."""
        relevant_ids = ["chunk_99"]
        mrr = metrics_calculator.calculate_mrr(sample_retrieved_results, relevant_ids)
        assert mrr == 0.0

    def test_mrr_multiple_relevant(self, metrics_calculator, sample_retrieved_results):
        """Test MRR with multiple relevant documents (should use first)."""
        relevant_ids = ["chunk_2", "chunk_3", "chunk_4"]
        mrr = metrics_calculator.calculate_mrr(sample_retrieved_results, relevant_ids)
        assert mrr == 0.5  # First relevant is at rank 2

    def test_mrr_empty_results(self, metrics_calculator):
        """Test MRR with empty results."""
        mrr = metrics_calculator.calculate_mrr([], ["chunk_1"])
        assert mrr == 0.0


class TestNDCG:
    """Tests for Normalized Discounted Cumulative Gain metric."""

    def test_ndcg_perfect_ranking(self, metrics_calculator, sample_retrieved_results):
        """Test NDCG with perfect ranking."""
        relevance_scores = {
            "chunk_1": 1.0,
            "chunk_2": 0.8,
            "chunk_3": 0.6,
            "chunk_4": 0.4,
            "chunk_5": 0.2,
        }
        ndcg = metrics_calculator.calculate_ndcg(sample_retrieved_results, relevance_scores, k=5)
        assert ndcg == 1.0

    def test_ndcg_binary_relevance(self, metrics_calculator, sample_retrieved_results):
        """Test NDCG with binary relevance (relevant/not relevant)."""
        relevance_scores = {
            "chunk_1": 1.0,
            "chunk_2": 1.0,
            "chunk_3": 0.0,
            "chunk_4": 0.0,
            "chunk_5": 0.0,
        }
        ndcg = metrics_calculator.calculate_ndcg(sample_retrieved_results, relevance_scores, k=5)
        assert 0.0 < ndcg <= 1.0  # Should be high but not perfect

    def test_ndcg_all_irrelevant(self, metrics_calculator, sample_retrieved_results):
        """Test NDCG when all results are irrelevant."""
        relevance_scores = {
            "chunk_99": 1.0,
            "chunk_100": 1.0,
        }
        ndcg = metrics_calculator.calculate_ndcg(sample_retrieved_results, relevance_scores, k=5)
        assert ndcg == 0.0

    def test_ndcg_smaller_k(self, metrics_calculator, sample_retrieved_results):
        """Test NDCG with k smaller than result set."""
        relevance_scores = {
            "chunk_1": 1.0,
            "chunk_2": 0.5,
        }
        ndcg = metrics_calculator.calculate_ndcg(sample_retrieved_results, relevance_scores, k=2)
        assert ndcg == 1.0  # Perfect ranking for top-2

    def test_ndcg_empty_relevance(self, metrics_calculator, sample_retrieved_results):
        """Test NDCG with empty relevance scores."""
        ndcg = metrics_calculator.calculate_ndcg(sample_retrieved_results, {}, k=5)
        assert ndcg == 0.0


class TestF1AtK:
    """Tests for F1@K metric."""

    def test_f1_at_k_perfect(self, metrics_calculator, sample_retrieved_results):
        """Test F1 with perfect precision and recall."""
        relevant_ids = ["chunk_1", "chunk_2", "chunk_3"]
        f1 = metrics_calculator.calculate_f1_at_k(sample_retrieved_results, relevant_ids, k=3)
        assert f1 == 1.0

    def test_f1_at_k_balanced(self, metrics_calculator, sample_retrieved_results):
        """Test F1 with balanced precision and recall."""
        relevant_ids = ["chunk_1", "chunk_2", "chunk_6", "chunk_7"]
        f1 = metrics_calculator.calculate_f1_at_k(sample_retrieved_results, relevant_ids, k=4)
        # Precision: 2/4 = 0.5, Recall: 2/4 = 0.5
        assert f1 == 0.5

    def test_f1_at_k_zero(self, metrics_calculator, sample_retrieved_results):
        """Test F1 when no relevant documents found."""
        relevant_ids = ["chunk_99", "chunk_100"]
        f1 = metrics_calculator.calculate_f1_at_k(sample_retrieved_results, relevant_ids, k=5)
        assert f1 == 0.0


class TestAveragePrecision:
    """Tests for Average Precision metric."""

    def test_map_all_relevant(self, metrics_calculator, sample_retrieved_results):
        """Test MAP when all results are relevant."""
        relevant_ids = ["chunk_1", "chunk_2", "chunk_3", "chunk_4", "chunk_5"]
        ap = metrics_calculator.calculate_average_precision(
            sample_retrieved_results, relevant_ids
        )
        # AP = (1/1 + 2/2 + 3/3 + 4/4 + 5/5) / 5 = 1.0
        assert ap == 1.0

    def test_map_some_relevant(self, metrics_calculator, sample_retrieved_results):
        """Test MAP with some relevant documents."""
        relevant_ids = ["chunk_1", "chunk_3", "chunk_5"]
        ap = metrics_calculator.calculate_average_precision(
            sample_retrieved_results, relevant_ids
        )
        # AP = (1/1 + 2/3 + 3/5) / 3
        expected_ap = (1.0 + 2 / 3 + 3 / 5) / 3
        assert ap == pytest.approx(expected_ap, rel=0.01)

    def test_map_no_relevant(self, metrics_calculator, sample_retrieved_results):
        """Test MAP when no relevant documents found."""
        relevant_ids = ["chunk_99", "chunk_100"]
        ap = metrics_calculator.calculate_average_precision(
            sample_retrieved_results, relevant_ids
        )
        assert ap == 0.0

    def test_map_first_only(self, metrics_calculator, sample_retrieved_results):
        """Test MAP when only first result is relevant."""
        relevant_ids = ["chunk_1"]
        ap = metrics_calculator.calculate_average_precision(
            sample_retrieved_results, relevant_ids
        )
        assert ap == 1.0


class TestLLMBasedMetrics:
    """Tests for LLM-based evaluation metrics."""

    @pytest.mark.asyncio
    async def test_context_relevance_structure(self, metrics_calculator, sample_retrieved_results):
        """Test that context relevance returns expected structure."""
        query = "What are the key considerations for drilling mud systems?"

        # Mock the Claude API call
        with patch.object(metrics_calculator, "_call_claude", new_callable=AsyncMock) as mock_claude:
            mock_claude.return_value = """OVERALL_SCORE: 0.85
CHUNK_SCORES: 0.9, 0.85, 0.8
REASONING: All chunks are highly relevant to the query."""

            result = await metrics_calculator.evaluate_context_relevance(
                query, sample_retrieved_results
            )

            assert "overall_score" in result
            assert "chunk_scores" in result
            assert "reasoning" in result
            assert isinstance(result["overall_score"], float)
            assert isinstance(result["chunk_scores"], list)
            assert isinstance(result["reasoning"], str)

    @pytest.mark.asyncio
    async def test_context_relevance_empty_results(self, metrics_calculator):
        """Test context relevance with empty results."""
        query = "Test query"

        result = await metrics_calculator.evaluate_context_relevance(query, [])

        assert result["overall_score"] == 0.0
        assert result["chunk_scores"] == []
        assert "No chunks" in result["reasoning"]

    @pytest.mark.asyncio
    async def test_answer_correctness_structure(self, metrics_calculator):
        """Test that answer correctness returns expected structure."""
        question = "What is the density of drilling mud?"
        answer = "The density of drilling mud is typically 8.5 to 10 ppg."
        ground_truth = "Drilling mud density ranges from 8.5 to 10 pounds per gallon."

        # Mock the Claude API call
        with patch.object(metrics_calculator, "_call_claude", new_callable=AsyncMock) as mock_claude:
            mock_claude.return_value = """CORRECTNESS: 0.9
SEMANTIC_SIMILARITY: 0.95
FACTUAL_ACCURACY: 1.0
COMPLETENESS: 0.85
REASONING: The answer is factually correct and semantically similar."""

            result = await metrics_calculator.evaluate_answer_correctness(
                question, answer, ground_truth
            )

            assert "correctness_score" in result
            assert "semantic_similarity" in result
            assert "factual_accuracy" in result
            assert "completeness" in result
            assert "reasoning" in result
            assert all(isinstance(result[k], float) for k in result if k != "reasoning")

    @pytest.mark.asyncio
    async def test_faithfulness_structure(self, metrics_calculator, sample_retrieved_results):
        """Test that faithfulness evaluation returns expected structure."""
        answer = "Drilling operations require careful mud system management and pressure control."

        # Mock the Claude API call
        with patch.object(metrics_calculator, "_call_claude", new_callable=AsyncMock) as mock_claude:
            mock_claude.return_value = """FAITHFULNESS_SCORE: 0.9
HALLUCINATION_COUNT: 0
SUPPORTED_CLAIMS: Mud system management required; Pressure control needed
UNSUPPORTED_CLAIMS:
REASONING: All claims are well supported by the context."""

            result = await metrics_calculator.evaluate_faithfulness(
                answer, sample_retrieved_results
            )

            assert "faithfulness_score" in result
            assert "hallucination_count" in result
            assert "supported_claims" in result
            assert "unsupported_claims" in result
            assert "reasoning" in result
            assert isinstance(result["faithfulness_score"], float)
            assert isinstance(result["hallucination_count"], int)
            assert isinstance(result["supported_claims"], list)
            assert isinstance(result["unsupported_claims"], list)

    @pytest.mark.asyncio
    async def test_faithfulness_empty_context(self, metrics_calculator):
        """Test faithfulness with empty context."""
        answer = "Test answer"

        result = await metrics_calculator.evaluate_faithfulness(answer, [])

        assert result["faithfulness_score"] == 0.0
        assert result["hallucination_count"] == 0
        assert "No context" in result["reasoning"]


class TestParsingHelpers:
    """Tests for response parsing helper methods."""

    def test_parse_relevance_response(self, metrics_calculator):
        """Test parsing of relevance evaluation response."""
        response = """OVERALL_SCORE: 0.85
CHUNK_SCORES: 0.9, 0.8, 0.7
REASONING: The chunks are highly relevant to the query."""

        result = metrics_calculator._parse_relevance_response(response, num_chunks=3)

        assert result["overall_score"] == 0.85
        assert result["chunk_scores"] == [0.9, 0.8, 0.7]
        assert "highly relevant" in result["reasoning"]

    def test_parse_correctness_response(self, metrics_calculator):
        """Test parsing of correctness evaluation response."""
        response = """CORRECTNESS: 0.9
SEMANTIC_SIMILARITY: 0.85
FACTUAL_ACCURACY: 0.95
COMPLETENESS: 0.8
REASONING: The answer is accurate and complete."""

        result = metrics_calculator._parse_correctness_response(response)

        assert result["correctness_score"] == 0.9
        assert result["semantic_similarity"] == 0.85
        assert result["factual_accuracy"] == 0.95
        assert result["completeness"] == 0.8
        assert "accurate" in result["reasoning"]

    def test_parse_faithfulness_response(self, metrics_calculator):
        """Test parsing of faithfulness evaluation response."""
        response = """FAITHFULNESS_SCORE: 0.8
HALLUCINATION_COUNT: 1
SUPPORTED_CLAIMS: Claim 1; Claim 2; Claim 3
UNSUPPORTED_CLAIMS: Claim 4
REASONING: Most claims are supported by context."""

        result = metrics_calculator._parse_faithfulness_response(response)

        assert result["faithfulness_score"] == 0.8
        assert result["hallucination_count"] == 1
        assert len(result["supported_claims"]) == 3
        assert len(result["unsupported_claims"]) == 1
        assert "supported" in result["reasoning"]
