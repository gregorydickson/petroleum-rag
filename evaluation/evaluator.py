"""Evaluator for running comprehensive RAG system benchmarks.

This module provides the main evaluation orchestration, including answer generation
and comprehensive metric calculation.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any

from anthropic import AsyncAnthropic

from config import settings
from models import BenchmarkQuery, BenchmarkResult, RetrievalResult
from evaluation.metrics import MetricsCalculator


class Evaluator:
    """Orchestrate RAG system evaluation with answer generation and metric calculation.

    The Evaluator class:
    1. Generates answers using retrieved context
    2. Calculates comprehensive metrics
    3. Produces structured benchmark results
    """

    def __init__(self) -> None:
        """Initialize evaluator with LLM client and metrics calculator."""
        self.client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.model = settings.eval_llm_model
        self.temperature = settings.eval_llm_temperature
        self.max_tokens = settings.eval_llm_max_tokens
        self.metrics_calculator = MetricsCalculator()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def close(self):
        """Close the async client and metrics calculator."""
        await self.client.close()
        await self.metrics_calculator.close()

    async def generate_answer(
        self,
        query: str,
        context_chunks: list[RetrievalResult],
    ) -> str:
        """Generate answer to query using retrieved context.

        Uses Claude with a system prompt designed to produce grounded,
        faithful responses based on the provided context.

        Args:
            query: User query
            context_chunks: List of retrieved context chunks

        Returns:
            Generated answer text
        """
        if not context_chunks:
            return "I cannot answer this question as no relevant context was retrieved."

        # Prepare context for LLM
        context_text = self._format_context(context_chunks)

        # System prompt for grounded answering
        system_prompt = """You are a helpful assistant specializing in petroleum engineering and oil & gas operations. Your task is to answer questions based ONLY on the provided context.

Guidelines:
1. Answer based strictly on the information in the context
2. If the context doesn't contain enough information, state that clearly
3. Do not add information from your general knowledge
4. Cite specific context chunks when making claims (e.g., "According to Chunk 1...")
5. If multiple chunks provide relevant information, synthesize them coherently
6. Be concise but complete in your answer
7. Use technical terminology appropriately
8. If you're uncertain, express that uncertainty

Remember: Faithfulness to the context is more important than providing a complete answer."""

        # User prompt with context and query
        user_prompt = f"""Context:
{context_text}

Question: {query}

Please provide a comprehensive answer based on the context above. If the context doesn't contain sufficient information to answer the question, clearly state that."""

        # Generate answer
        message = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        return message.content[0].text

    async def evaluate_query(
        self,
        query: BenchmarkQuery,
        retrieved: list[RetrievalResult],
        generated_answer: str,
        parser_name: str,
        storage_backend: str,
        retrieval_time: float = 0.0,
        generation_time: float = 0.0,
    ) -> BenchmarkResult:
        """Evaluate a single query with comprehensive metrics.

        Calculates both traditional IR metrics and LLM-based quality metrics.

        Args:
            query: Benchmark query with ground truth
            retrieved: Retrieved results
            generated_answer: Generated answer from LLM
            parser_name: Name of parser used
            storage_backend: Name of storage backend used
            retrieval_time: Time taken for retrieval (seconds)
            generation_time: Time taken for answer generation (seconds)

        Returns:
            BenchmarkResult with all calculated metrics
        """
        # Calculate traditional IR metrics
        k_values = [1, 3, 5, 10]
        metrics: dict[str, float] = {}

        # Precision, Recall, F1 at various K
        for k in k_values:
            metrics[f"precision@{k}"] = self.metrics_calculator.calculate_precision_at_k(
                retrieved, query.relevant_element_ids, k
            )
            metrics[f"recall@{k}"] = self.metrics_calculator.calculate_recall_at_k(
                retrieved, query.relevant_element_ids, k
            )
            metrics[f"f1@{k}"] = self.metrics_calculator.calculate_f1_at_k(
                retrieved, query.relevant_element_ids, k
            )

        # MRR and MAP
        metrics["mrr"] = self.metrics_calculator.calculate_mrr(
            retrieved, query.relevant_element_ids
        )
        metrics["map"] = self.metrics_calculator.calculate_average_precision(
            retrieved, query.relevant_element_ids
        )

        # NDCG (use binary relevance: 1.0 for relevant, 0.0 for irrelevant)
        relevance_scores = {elem_id: 1.0 for elem_id in query.relevant_element_ids}
        for k in k_values:
            metrics[f"ndcg@{k}"] = self.metrics_calculator.calculate_ndcg(
                retrieved, relevance_scores, k
            )

        # Calculate LLM-based metrics asynchronously
        try:
            context_eval, correctness_eval, faithfulness_eval = await asyncio.gather(
                self.metrics_calculator.evaluate_context_relevance(query.query, retrieved),
                self.metrics_calculator.evaluate_answer_correctness(
                    query.query,
                    generated_answer,
                    query.ground_truth_answer,
                ),
                self.metrics_calculator.evaluate_faithfulness(generated_answer, retrieved),
            )

            # Add LLM-based metrics
            metrics["context_relevance"] = context_eval["overall_score"]
            metrics["answer_correctness"] = correctness_eval["correctness_score"]
            metrics["semantic_similarity"] = correctness_eval["semantic_similarity"]
            metrics["factual_accuracy"] = correctness_eval["factual_accuracy"]
            metrics["completeness"] = correctness_eval["completeness"]
            metrics["faithfulness"] = faithfulness_eval["faithfulness_score"]
            metrics["hallucination_count"] = float(faithfulness_eval["hallucination_count"])

        except Exception as e:
            # If LLM evaluation fails, log and continue with IR metrics only
            print(f"Warning: LLM-based evaluation failed: {e}")
            metrics["context_relevance"] = 0.0
            metrics["answer_correctness"] = 0.0
            metrics["semantic_similarity"] = 0.0
            metrics["factual_accuracy"] = 0.0
            metrics["completeness"] = 0.0
            metrics["faithfulness"] = 0.0
            metrics["hallucination_count"] = 0.0

        # Create benchmark result
        benchmark_id = f"{parser_name}_{storage_backend}_{query.query_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        return BenchmarkResult(
            benchmark_id=benchmark_id,
            parser_name=parser_name,
            storage_backend=storage_backend,
            query_id=query.query_id,
            query=query.query,
            retrieved_results=retrieved,
            generated_answer=generated_answer,
            ground_truth_answer=query.ground_truth_answer,
            metrics=metrics,
            retrieval_time_seconds=retrieval_time,
            generation_time_seconds=generation_time,
            total_time_seconds=retrieval_time + generation_time,
            timestamp=datetime.now(timezone.utc),
        )

    async def evaluate_batch(
        self,
        queries: list[BenchmarkQuery],
        retrieval_results: dict[str, list[RetrievalResult]],
        parser_name: str,
        storage_backend: str,
    ) -> list[BenchmarkResult]:
        """Evaluate a batch of queries.

        Args:
            queries: List of benchmark queries
            retrieval_results: Dictionary mapping query_id to retrieved results
            parser_name: Name of parser used
            storage_backend: Name of storage backend used

        Returns:
            List of benchmark results
        """
        results = []

        for query in queries:
            retrieved = retrieval_results.get(query.query_id, [])

            # Generate answer
            start_time = datetime.now(timezone.utc)
            generated_answer = await self.generate_answer(query.query, retrieved)
            generation_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            # Evaluate
            benchmark_result = await self.evaluate_query(
                query=query,
                retrieved=retrieved,
                generated_answer=generated_answer,
                parser_name=parser_name,
                storage_backend=storage_backend,
                retrieval_time=0.0,  # Should be provided by caller
                generation_time=generation_time,
            )

            results.append(benchmark_result)

        return results

    def calculate_aggregate_metrics(
        self,
        results: list[BenchmarkResult],
    ) -> dict[str, Any]:
        """Calculate aggregate metrics across multiple benchmark results.

        Args:
            results: List of benchmark results

        Returns:
            Dictionary of aggregate metrics including means, medians, and distributions
        """
        if not results:
            return {}

        # Collect all metric values
        metric_names = set()
        for result in results:
            metric_names.update(result.metrics.keys())

        aggregates: dict[str, Any] = {}

        for metric_name in metric_names:
            values = [r.metrics.get(metric_name, 0.0) for r in results]

            aggregates[f"{metric_name}_mean"] = sum(values) / len(values)
            aggregates[f"{metric_name}_median"] = self._median(values)
            aggregates[f"{metric_name}_min"] = min(values)
            aggregates[f"{metric_name}_max"] = max(values)
            aggregates[f"{metric_name}_std"] = self._std_dev(values)

        # Add timing aggregates
        retrieval_times = [r.retrieval_time_seconds for r in results]
        generation_times = [r.generation_time_seconds for r in results]
        total_times = [r.total_time_seconds for r in results]

        aggregates["avg_retrieval_time"] = sum(retrieval_times) / len(retrieval_times)
        aggregates["avg_generation_time"] = sum(generation_times) / len(generation_times)
        aggregates["avg_total_time"] = sum(total_times) / len(total_times)

        # Success rate
        success_count = sum(1 for r in results if r.success)
        aggregates["success_rate"] = success_count / len(results)

        return aggregates

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _format_context(self, chunks: list[RetrievalResult]) -> str:
        """Format retrieved chunks for LLM context.

        Args:
            chunks: List of retrieved chunks

        Returns:
            Formatted context string
        """
        formatted_chunks = []

        for i, chunk in enumerate(chunks, start=1):
            chunk_header = f"[Chunk {i}]"
            if chunk.metadata:
                metadata_str = ", ".join(
                    f"{k}: {v}" for k, v in chunk.metadata.items() if k != "content"
                )
                if metadata_str:
                    chunk_header += f" ({metadata_str})"

            chunk_header += f" [Relevance: {chunk.score:.3f}]"

            formatted_chunks.append(f"{chunk_header}\n{chunk.content}")

        return "\n\n" + ("-" * 80) + "\n\n".join(formatted_chunks)

    def _median(self, values: list[float]) -> float:
        """Calculate median of a list of values.

        Args:
            values: List of numeric values

        Returns:
            Median value
        """
        if not values:
            return 0.0

        sorted_values = sorted(values)
        n = len(sorted_values)

        if n % 2 == 0:
            return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
        else:
            return sorted_values[n // 2]

    def _std_dev(self, values: list[float]) -> float:
        """Calculate standard deviation of a list of values.

        Args:
            values: List of numeric values

        Returns:
            Standard deviation
        """
        if not values:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance**0.5
