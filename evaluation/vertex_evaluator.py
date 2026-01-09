"""Vertex AI evaluator using Gemini for answer generation and evaluation.

This module provides a Vertex AI evaluator that matches the Evaluator interface,
allowing it to be used as a drop-in replacement for Anthropic Claude.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any

from config import settings
from evaluation.metrics import MetricsCalculator
from models import BenchmarkQuery, BenchmarkResult, RetrievalResult


class VertexEvaluator:
    """Orchestrate RAG evaluation using Vertex AI Gemini models.

    Provides the same interface as Evaluator but uses Gemini instead of Claude.
    Handles:
    1. Answer generation using retrieved context
    2. Comprehensive metric calculation
    3. Structured benchmark results
    """

    def __init__(self) -> None:
        """Initialize evaluator with Vertex AI Gemini and metrics calculator."""
        import vertexai
        from vertexai.generative_models import GenerativeModel

        # Initialize Vertex AI
        if settings.vertex_api_key:
            vertexai.init(
                project=settings.google_cloud_project,
                location=settings.vertex_docai_location or "us-central1",
                api_key=settings.vertex_api_key,
            )
        else:
            vertexai.init(
                project=settings.google_cloud_project,
                location=settings.vertex_docai_location or "us-central1",
            )

        self.model_name = settings.vertex_llm_model or "gemini-1.5-pro"
        self.model = GenerativeModel(self.model_name)
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
        """Close the metrics calculator."""
        await self.metrics_calculator.close()

    async def generate_answer(
        self,
        query: str,
        context_chunks: list[RetrievalResult],
    ) -> str:
        """Generate answer to query using retrieved context.

        Uses Gemini with a prompt designed to produce grounded,
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

        # Prompt for grounded answering (Gemini uses single prompt, not system/user split)
        prompt = f"""You are a helpful assistant specializing in petroleum engineering and oil & gas operations. Your task is to answer questions based ONLY on the provided context.

Guidelines:
1. Answer based strictly on the information in the context
2. If the context doesn't contain enough information, state that clearly
3. Do not add information from your general knowledge
4. Cite specific context chunks when making claims (e.g., "According to Chunk 1...")
5. If multiple chunks provide relevant information, synthesize them coherently
6. Be concise but complete in your answer
7. Use technical terminology appropriately
8. If you're uncertain, express that uncertainty

Remember: Faithfulness to the context is more important than providing a complete answer.

Context:
{context_text}

Question: {query}

Please provide a comprehensive answer based on the context above. If the context doesn't contain sufficient information to answer the question, clearly state that."""

        # Generate answer using Gemini
        try:
            # Run synchronous Gemini call in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._generate_content_sync,
                prompt,
            )
            return response.text
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    def _generate_content_sync(self, prompt: str):
        """Synchronous wrapper for Gemini content generation.

        Args:
            prompt: The prompt to send to Gemini

        Returns:
            Gemini response object
        """
        generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }

        return self.model.generate_content(
            prompt,
            generation_config=generation_config,
        )

    def _format_context(self, chunks: list[RetrievalResult]) -> str:
        """Format retrieved chunks into context string for LLM.

        Args:
            chunks: Retrieved context chunks

        Returns:
            Formatted context string
        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"Chunk {i} (relevance: {chunk.score:.2f}):\n{chunk.content}\n"
            )
        return "\n".join(context_parts)

    async def evaluate_query(
        self,
        query: BenchmarkQuery,
        retrieved: list[RetrievalResult],
        generated_answer: str,
        parser_name: str,
        storage_backend: str,
        retrieval_time: float,
        generation_time: float,
    ) -> BenchmarkResult:
        """Evaluate a single query with comprehensive metrics.

        Args:
            query: Benchmark query with ground truth
            retrieved: Retrieved context chunks
            generated_answer: Generated answer from LLM
            parser_name: Name of parser used
            storage_backend: Name of storage backend used
            retrieval_time: Time taken for retrieval
            generation_time: Time taken for answer generation

        Returns:
            Complete benchmark result with metrics
        """
        # Calculate all metrics
        metrics = await self.metrics_calculator.calculate_all_metrics(
            query=query.query,
            ground_truth=query.ground_truth_answer,
            generated_answer=generated_answer,
            retrieved_chunks=retrieved,
            relevant_element_ids=query.relevant_element_ids,
        )

        # Create benchmark result
        result = BenchmarkResult(
            benchmark_id=f"{parser_name}_{storage_backend}_{query.query_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
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
            error=None,
        )

        return result
