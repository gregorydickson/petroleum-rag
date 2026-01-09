"""Evaluation metrics for RAG system benchmarking.

This module provides both traditional information retrieval metrics (Precision@K, Recall@K,
MRR, NDCG) and LLM-based evaluation metrics (context relevance, answer correctness,
faithfulness).
"""

import math
from typing import Any

from anthropic import AsyncAnthropic

from config import settings
from models import RetrievalResult
from utils.cache import get_llm_cache
from utils.circuit_breaker import call_llm_with_breaker
from utils.rate_limiter import rate_limiter


class MetricsCalculator:
    """Calculate evaluation metrics for RAG systems.

    Provides both traditional IR metrics and LLM-based quality assessment metrics.
    """

    def __init__(self) -> None:
        """Initialize metrics calculator with Anthropic client."""
        self.client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.model = settings.eval_llm_model
        self.temperature = settings.eval_llm_temperature
        self.max_tokens = settings.eval_llm_max_tokens

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def close(self):
        """Close the async client."""
        await self.client.close()

    # -------------------------------------------------------------------------
    # Traditional Information Retrieval Metrics
    # -------------------------------------------------------------------------

    def calculate_precision_at_k(
        self,
        retrieved: list[RetrievalResult],
        relevant_ids: list[str],
        k: int,
    ) -> float:
        """Calculate Precision@K.

        Precision@K measures the proportion of retrieved documents that are relevant
        within the top K results.

        Args:
            retrieved: List of retrieval results
            relevant_ids: List of IDs of relevant elements/chunks
            k: Number of top results to consider

        Returns:
            Precision@K score (0.0-1.0)
        """
        if not retrieved or k <= 0:
            return 0.0

        # Take top K results
        top_k = retrieved[:k]

        # Count how many are relevant
        relevant_count = sum(
            1
            for result in top_k
            if result.chunk_id in relevant_ids or result.document_id in relevant_ids
        )

        return relevant_count / len(top_k)

    def calculate_recall_at_k(
        self,
        retrieved: list[RetrievalResult],
        relevant_ids: list[str],
        k: int,
    ) -> float:
        """Calculate Recall@K.

        Recall@K measures the proportion of relevant documents that appear
        in the top K results.

        Args:
            retrieved: List of retrieval results
            relevant_ids: List of IDs of relevant elements/chunks
            k: Number of top results to consider

        Returns:
            Recall@K score (0.0-1.0)
        """
        if not relevant_ids or not retrieved or k <= 0:
            return 0.0

        # Take top K results
        top_k = retrieved[:k]

        # Count how many relevant documents were retrieved
        retrieved_relevant = sum(
            1
            for result in top_k
            if result.chunk_id in relevant_ids or result.document_id in relevant_ids
        )

        return retrieved_relevant / len(relevant_ids)

    def calculate_mrr(
        self,
        retrieved: list[RetrievalResult],
        relevant_ids: list[str],
    ) -> float:
        """Calculate Mean Reciprocal Rank (MRR).

        MRR measures the rank of the first relevant document in the results.
        Returns the reciprocal of this rank (1/rank).

        Args:
            retrieved: List of retrieval results
            relevant_ids: List of IDs of relevant elements/chunks

        Returns:
            MRR score (0.0-1.0)
        """
        if not retrieved or not relevant_ids:
            return 0.0

        # Find rank of first relevant document (1-based indexing)
        for rank, result in enumerate(retrieved, start=1):
            if result.chunk_id in relevant_ids or result.document_id in relevant_ids:
                return 1.0 / rank

        # No relevant documents found
        return 0.0

    def calculate_ndcg(
        self,
        retrieved: list[RetrievalResult],
        relevance_scores: dict[str, float],
        k: int,
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain (NDCG@K).

        NDCG measures the quality of ranking by considering both relevance
        and position. Uses graded relevance scores (0.0-1.0).

        Args:
            retrieved: List of retrieval results
            relevance_scores: Dictionary mapping chunk/element IDs to relevance scores (0.0-1.0)
            k: Number of top results to consider

        Returns:
            NDCG@K score (0.0-1.0)
        """
        if not retrieved or not relevance_scores or k <= 0:
            return 0.0

        # Take top K results
        top_k = retrieved[:k]

        # Calculate DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for rank, result in enumerate(top_k, start=1):
            # Get relevance score (0.0 if not in relevance_scores)
            rel = relevance_scores.get(result.chunk_id, 0.0)
            if rel == 0.0:
                rel = relevance_scores.get(result.document_id, 0.0)

            # DCG formula: rel / log2(rank + 1)
            dcg += rel / math.log2(rank + 1)

        # Calculate IDCG (Ideal DCG) - DCG of perfect ranking
        sorted_relevances = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(sorted_relevances, start=1))

        # Avoid division by zero
        if idcg == 0.0:
            return 0.0

        return dcg / idcg

    def calculate_f1_at_k(
        self,
        retrieved: list[RetrievalResult],
        relevant_ids: list[str],
        k: int,
    ) -> float:
        """Calculate F1 score at K.

        F1@K is the harmonic mean of Precision@K and Recall@K.

        Args:
            retrieved: List of retrieval results
            relevant_ids: List of IDs of relevant elements/chunks
            k: Number of top results to consider

        Returns:
            F1@K score (0.0-1.0)
        """
        precision = self.calculate_precision_at_k(retrieved, relevant_ids, k)
        recall = self.calculate_recall_at_k(retrieved, relevant_ids, k)

        if precision + recall == 0.0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    def calculate_average_precision(
        self,
        retrieved: list[RetrievalResult],
        relevant_ids: list[str],
    ) -> float:
        """Calculate Average Precision (AP).

        AP is the average of precision values at each position where a relevant
        document is retrieved.

        Args:
            retrieved: List of retrieval results
            relevant_ids: List of IDs of relevant elements/chunks

        Returns:
            Average Precision score (0.0-1.0)
        """
        if not retrieved or not relevant_ids:
            return 0.0

        num_relevant = 0
        sum_precisions = 0.0

        for rank, result in enumerate(retrieved, start=1):
            is_relevant = (
                result.chunk_id in relevant_ids or result.document_id in relevant_ids
            )

            if is_relevant:
                num_relevant += 1
                precision_at_rank = num_relevant / rank
                sum_precisions += precision_at_rank

        if num_relevant == 0:
            return 0.0

        return sum_precisions / len(relevant_ids)

    # -------------------------------------------------------------------------
    # LLM-Based Evaluation Metrics
    # -------------------------------------------------------------------------

    async def evaluate_context_relevance(
        self,
        query: str,
        retrieved: list[RetrievalResult],
    ) -> dict[str, Any]:
        """Evaluate relevance of retrieved context using LLM.

        Uses Claude to assess how relevant the retrieved chunks are to answering
        the query.

        Args:
            query: User query
            retrieved: List of retrieved results

        Returns:
            Dictionary containing:
                - overall_score: Overall relevance score (0.0-1.0)
                - chunk_scores: List of per-chunk relevance scores
                - reasoning: Explanation of the evaluation
        """
        if not retrieved:
            return {
                "overall_score": 0.0,
                "chunk_scores": [],
                "reasoning": "No chunks retrieved",
            }

        # Prepare context for evaluation
        context_text = "\n\n".join(
            f"[Chunk {i+1}] (Score: {result.score:.3f})\n{result.content}"
            for i, result in enumerate(retrieved)
        )

        prompt = f"""Evaluate the relevance of the retrieved context chunks for answering the following query.

Query: {query}

Retrieved Context:
{context_text}

For each chunk, assess:
1. Does it contain information relevant to the query?
2. How directly does it address the query?
3. Is the information accurate and useful?

Provide:
1. An overall relevance score (0.0-1.0) for the entire context set
2. Individual scores (0.0-1.0) for each chunk
3. Brief reasoning for your evaluation

Format your response as:
OVERALL_SCORE: <score>
CHUNK_SCORES: <score1>, <score2>, ...
REASONING: <explanation>"""

        response = await self._call_claude(prompt)
        return self._parse_relevance_response(response, len(retrieved))

    async def evaluate_answer_correctness(
        self,
        question: str,
        answer: str,
        ground_truth: str,
    ) -> dict[str, Any]:
        """Evaluate correctness of generated answer against ground truth.

        Uses Claude to compare the generated answer with the ground truth answer
        and assess correctness.

        Args:
            question: Original question
            answer: Generated answer
            ground_truth: Expected correct answer

        Returns:
            Dictionary containing:
                - correctness_score: Correctness score (0.0-1.0)
                - semantic_similarity: Semantic similarity score (0.0-1.0)
                - factual_accuracy: Factual accuracy score (0.0-1.0)
                - reasoning: Explanation of the evaluation
        """
        prompt = f"""Evaluate the correctness of the generated answer compared to the ground truth answer.

Question: {question}

Ground Truth Answer: {ground_truth}

Generated Answer: {answer}

Assess the following:
1. Correctness: Does the generated answer match the ground truth? (0.0-1.0)
2. Semantic Similarity: How semantically similar are the answers? (0.0-1.0)
3. Factual Accuracy: Are all facts in the generated answer correct? (0.0-1.0)
4. Completeness: Does the generated answer cover all key points? (0.0-1.0)

Consider:
- The generated answer may use different wording but convey the same meaning
- Minor differences in phrasing should not heavily penalize the score
- Focus on factual correctness and semantic equivalence

Format your response as:
CORRECTNESS: <score>
SEMANTIC_SIMILARITY: <score>
FACTUAL_ACCURACY: <score>
COMPLETENESS: <score>
REASONING: <explanation>"""

        response = await self._call_claude(prompt)
        return self._parse_correctness_response(response)

    async def evaluate_faithfulness(
        self,
        answer: str,
        context: list[RetrievalResult],
    ) -> dict[str, Any]:
        """Evaluate faithfulness of answer to retrieved context.

        Uses Claude to assess whether the generated answer is grounded in
        the retrieved context (no hallucinations).

        Args:
            answer: Generated answer
            context: List of retrieved context chunks

        Returns:
            Dictionary containing:
                - faithfulness_score: Faithfulness score (0.0-1.0)
                - hallucination_count: Number of unsupported claims
                - supported_claims: List of claims supported by context
                - unsupported_claims: List of claims not supported by context
                - reasoning: Explanation of the evaluation
        """
        if not context:
            return {
                "faithfulness_score": 0.0,
                "hallucination_count": 0,
                "supported_claims": [],
                "unsupported_claims": [],
                "reasoning": "No context provided",
            }

        # Prepare context text
        context_text = "\n\n".join(
            f"[Chunk {i+1}]\n{result.content}" for i, result in enumerate(context)
        )

        prompt = f"""Evaluate whether the generated answer is faithful to the provided context.

Context:
{context_text}

Generated Answer: {answer}

Assess:
1. Identify all factual claims in the answer
2. Check if each claim is supported by the context
3. Count unsupported claims (potential hallucinations)
4. Calculate faithfulness score (% of supported claims)

A claim is supported if:
- It directly appears in the context
- It can be logically inferred from the context
- It's a reasonable generalization of context information

A claim is unsupported if:
- It introduces information not in the context
- It contradicts the context
- It makes unsupported assumptions

Format your response as:
FAITHFULNESS_SCORE: <score>
HALLUCINATION_COUNT: <count>
SUPPORTED_CLAIMS: <claim1>; <claim2>; ...
UNSUPPORTED_CLAIMS: <claim1>; <claim2>; ...
REASONING: <explanation>"""

        response = await self._call_claude(prompt)
        return self._parse_faithfulness_response(response)

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    async def _call_claude(self, prompt: str) -> str:
        """Call Claude API with the given prompt, with caching.

        Args:
            prompt: Prompt text

        Returns:
            Response text from Claude
        """
        # Check cache first
        cache = get_llm_cache()
        cache_key = cache.hash_content(f"{self.model}:{self.temperature}:{prompt}")
        cached = await cache.get(cache_key)

        if cached is not None:
            return cached

        # Acquire rate limit token before making API call
        if rate_limiter.is_registered("anthropic"):
            await rate_limiter.acquire("anthropic")

        # Wrap API call with circuit breaker
        async def _make_api_call() -> str:
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text

        result = await call_llm_with_breaker(_make_api_call)

        # Cache result
        await cache.set(cache_key, result)

        return result

    def _parse_relevance_response(self, response: str, num_chunks: int) -> dict[str, Any]:
        """Parse LLM response for context relevance evaluation.

        Args:
            response: Raw LLM response
            num_chunks: Expected number of chunks

        Returns:
            Parsed evaluation results
        """
        lines = response.strip().split("\n")
        overall_score = 0.0
        chunk_scores = []
        reasoning = ""

        for line in lines:
            if line.startswith("OVERALL_SCORE:"):
                try:
                    overall_score = float(line.split(":", 1)[1].strip())
                except (ValueError, IndexError):
                    pass
            elif line.startswith("CHUNK_SCORES:"):
                try:
                    scores_str = line.split(":", 1)[1].strip()
                    chunk_scores = [float(s.strip()) for s in scores_str.split(",")]
                except (ValueError, IndexError):
                    chunk_scores = [0.0] * num_chunks
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        # Ensure we have scores for all chunks
        if len(chunk_scores) != num_chunks:
            chunk_scores = [overall_score] * num_chunks

        return {
            "overall_score": overall_score,
            "chunk_scores": chunk_scores,
            "reasoning": reasoning,
        }

    def _parse_correctness_response(self, response: str) -> dict[str, Any]:
        """Parse LLM response for answer correctness evaluation.

        Args:
            response: Raw LLM response

        Returns:
            Parsed evaluation results
        """
        lines = response.strip().split("\n")
        result = {
            "correctness_score": 0.0,
            "semantic_similarity": 0.0,
            "factual_accuracy": 0.0,
            "completeness": 0.0,
            "reasoning": "",
        }

        for line in lines:
            if line.startswith("CORRECTNESS:"):
                try:
                    result["correctness_score"] = float(line.split(":", 1)[1].strip())
                except (ValueError, IndexError):
                    pass
            elif line.startswith("SEMANTIC_SIMILARITY:"):
                try:
                    result["semantic_similarity"] = float(line.split(":", 1)[1].strip())
                except (ValueError, IndexError):
                    pass
            elif line.startswith("FACTUAL_ACCURACY:"):
                try:
                    result["factual_accuracy"] = float(line.split(":", 1)[1].strip())
                except (ValueError, IndexError):
                    pass
            elif line.startswith("COMPLETENESS:"):
                try:
                    result["completeness"] = float(line.split(":", 1)[1].strip())
                except (ValueError, IndexError):
                    pass
            elif line.startswith("REASONING:"):
                result["reasoning"] = line.split(":", 1)[1].strip()

        return result

    def _parse_faithfulness_response(self, response: str) -> dict[str, Any]:
        """Parse LLM response for faithfulness evaluation.

        Args:
            response: Raw LLM response

        Returns:
            Parsed evaluation results
        """
        lines = response.strip().split("\n")
        result = {
            "faithfulness_score": 0.0,
            "hallucination_count": 0,
            "supported_claims": [],
            "unsupported_claims": [],
            "reasoning": "",
        }

        for line in lines:
            if line.startswith("FAITHFULNESS_SCORE:"):
                try:
                    result["faithfulness_score"] = float(line.split(":", 1)[1].strip())
                except (ValueError, IndexError):
                    pass
            elif line.startswith("HALLUCINATION_COUNT:"):
                try:
                    result["hallucination_count"] = int(line.split(":", 1)[1].strip())
                except (ValueError, IndexError):
                    pass
            elif line.startswith("SUPPORTED_CLAIMS:"):
                claims_str = line.split(":", 1)[1].strip()
                if claims_str:
                    result["supported_claims"] = [
                        c.strip() for c in claims_str.split(";") if c.strip()
                    ]
            elif line.startswith("UNSUPPORTED_CLAIMS:"):
                claims_str = line.split(":", 1)[1].strip()
                if claims_str:
                    result["unsupported_claims"] = [
                        c.strip() for c in claims_str.split(";") if c.strip()
                    ]
            elif line.startswith("REASONING:"):
                result["reasoning"] = line.split(":", 1)[1].strip()

        return result
