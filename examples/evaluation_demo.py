"""Demonstration of the evaluation framework.

This script shows how to use the MetricsCalculator and Evaluator classes
to evaluate RAG system performance.
"""

import asyncio
from datetime import datetime, timezone

from evaluation import Evaluator, MetricsCalculator
from models import BenchmarkQuery, DifficultyLevel, QueryType, RetrievalResult


async def demo_metrics():
    """Demonstrate traditional IR metrics calculation."""
    print("=" * 80)
    print("DEMO: Traditional Information Retrieval Metrics")
    print("=" * 80)

    # Create sample retrieval results
    retrieved = [
        RetrievalResult(
            chunk_id="chunk_1",
            document_id="doc_1",
            content="Drilling mud density is critical for wellbore stability.",
            score=0.95,
            rank=1,
        ),
        RetrievalResult(
            chunk_id="chunk_2",
            document_id="doc_1",
            content="Viscosity control ensures effective hole cleaning.",
            score=0.87,
            rank=2,
        ),
        RetrievalResult(
            chunk_id="chunk_3",
            document_id="doc_2",
            content="Filtration control prevents formation damage.",
            score=0.75,
            rank=3,
        ),
        RetrievalResult(
            chunk_id="chunk_4",
            document_id="doc_2",
            content="Irrelevant content about different topic.",
            score=0.45,
            rank=4,
        ),
    ]

    # Define relevant chunks (ground truth)
    relevant_ids = ["chunk_1", "chunk_2", "chunk_3"]

    # Calculate metrics
    calculator = MetricsCalculator()

    print("\nRetrieval Results:")
    for result in retrieved:
        relevant_marker = "✓" if result.chunk_id in relevant_ids else "✗"
        print(f"  [{relevant_marker}] {result.chunk_id} (score: {result.score:.3f})")
        print(f"      {result.content[:60]}...")

    print("\nMetrics:")
    print(f"  Precision@3: {calculator.calculate_precision_at_k(retrieved, relevant_ids, 3):.3f}")
    print(f"  Recall@3: {calculator.calculate_recall_at_k(retrieved, relevant_ids, 3):.3f}")
    print(f"  F1@3: {calculator.calculate_f1_at_k(retrieved, relevant_ids, 3):.3f}")
    print(f"  MRR: {calculator.calculate_mrr(retrieved, relevant_ids):.3f}")
    print(f"  MAP: {calculator.calculate_average_precision(retrieved, relevant_ids):.3f}")

    # NDCG with graded relevance
    relevance_scores = {
        "chunk_1": 1.0,
        "chunk_2": 0.9,
        "chunk_3": 0.8,
    }
    print(f"  NDCG@3: {calculator.calculate_ndcg(retrieved, relevance_scores, 3):.3f}")


async def demo_llm_metrics():
    """Demonstrate LLM-based evaluation metrics."""
    print("\n" + "=" * 80)
    print("DEMO: LLM-Based Evaluation Metrics")
    print("=" * 80)
    print("\nNote: This requires valid ANTHROPIC_API_KEY in .env")
    print("Skipping LLM-based demo in example script.")
    print("See tests/test_metrics.py for usage with mocked API calls.")


async def demo_evaluator():
    """Demonstrate the Evaluator class."""
    print("\n" + "=" * 80)
    print("DEMO: Complete Query Evaluation")
    print("=" * 80)

    # Create a benchmark query
    query = BenchmarkQuery(
        query_id="demo_query_1",
        query="What are the key considerations for drilling mud systems?",
        ground_truth_answer="Key considerations include mud density for wellbore stability, viscosity for hole cleaning, and filtration control to prevent formation damage.",
        relevant_element_ids=["chunk_1", "chunk_2", "chunk_3"],
        query_type=QueryType.SEMANTIC,
        difficulty=DifficultyLevel.MEDIUM,
    )

    # Sample retrieval results
    retrieved = [
        RetrievalResult(
            chunk_id="chunk_1",
            document_id="doc_1",
            content="Drilling mud density is critical for maintaining wellbore stability and preventing blowouts. The density must be carefully controlled based on formation pressure.",
            score=0.95,
            rank=1,
        ),
        RetrievalResult(
            chunk_id="chunk_2",
            document_id="doc_1",
            content="Viscosity management is essential for effective hole cleaning. Proper viscosity ensures cuttings are transported to the surface.",
            score=0.87,
            rank=2,
        ),
        RetrievalResult(
            chunk_id="chunk_3",
            document_id="doc_2",
            content="Filtration control prevents formation damage by minimizing fluid loss into the formation. This is achieved through proper mud cake formation.",
            score=0.75,
            rank=3,
        ),
    ]

    print("\nQuery:", query.query)
    print("\nRetrieved Chunks:")
    for result in retrieved:
        print(f"  [{result.chunk_id}] (score: {result.score:.3f})")
        print(f"    {result.content[:80]}...")

    print("\nGround Truth Answer:")
    print(f"  {query.ground_truth_answer}")

    # Calculate metrics (without LLM-based metrics for demo)
    calculator = MetricsCalculator()

    print("\nInformation Retrieval Metrics:")
    k_values = [1, 3, 5]
    for k in k_values:
        precision = calculator.calculate_precision_at_k(retrieved, query.relevant_element_ids, k)
        recall = calculator.calculate_recall_at_k(retrieved, query.relevant_element_ids, k)
        f1 = calculator.calculate_f1_at_k(retrieved, query.relevant_element_ids, k)
        print(f"  Precision@{k}: {precision:.3f} | Recall@{k}: {recall:.3f} | F1@{k}: {f1:.3f}")

    mrr = calculator.calculate_mrr(retrieved, query.relevant_element_ids)
    map_score = calculator.calculate_average_precision(retrieved, query.relevant_element_ids)
    print(f"  MRR: {mrr:.3f} | MAP: {map_score:.3f}")

    print("\nNote: For complete evaluation including LLM-based metrics and answer generation,")
    print("use evaluator.evaluate_query() with valid API credentials.")


async def demo_aggregate_metrics():
    """Demonstrate aggregate metrics calculation."""
    print("\n" + "=" * 80)
    print("DEMO: Aggregate Metrics")
    print("=" * 80)

    from models import BenchmarkResult

    # Create sample benchmark results
    results = [
        BenchmarkResult(
            benchmark_id="test_1",
            parser_name="llama_parse",
            storage_backend="chroma",
            query_id="q1",
            query="Query 1",
            retrieved_results=[],
            generated_answer="Answer 1",
            ground_truth_answer="GT 1",
            metrics={
                "precision@1": 1.0,
                "recall@1": 0.8,
                "f1@1": 0.89,
                "mrr": 1.0,
                "map": 0.95,
            },
            retrieval_time_seconds=0.5,
            generation_time_seconds=2.0,
            total_time_seconds=2.5,
        ),
        BenchmarkResult(
            benchmark_id="test_2",
            parser_name="llama_parse",
            storage_backend="chroma",
            query_id="q2",
            query="Query 2",
            retrieved_results=[],
            generated_answer="Answer 2",
            ground_truth_answer="GT 2",
            metrics={
                "precision@1": 0.8,
                "recall@1": 0.6,
                "f1@1": 0.69,
                "mrr": 0.5,
                "map": 0.75,
            },
            retrieval_time_seconds=0.6,
            generation_time_seconds=2.2,
            total_time_seconds=2.8,
        ),
        BenchmarkResult(
            benchmark_id="test_3",
            parser_name="llama_parse",
            storage_backend="chroma",
            query_id="q3",
            query="Query 3",
            retrieved_results=[],
            generated_answer="Answer 3",
            ground_truth_answer="GT 3",
            metrics={
                "precision@1": 0.9,
                "recall@1": 0.9,
                "f1@1": 0.9,
                "mrr": 1.0,
                "map": 0.92,
            },
            retrieval_time_seconds=0.4,
            generation_time_seconds=1.8,
            total_time_seconds=2.2,
        ),
    ]

    evaluator = Evaluator()
    aggregates = evaluator.calculate_aggregate_metrics(results)

    print("\nAggregate Metrics across 3 queries:")
    print(f"  Precision@1 (mean): {aggregates['precision@1_mean']:.3f}")
    print(f"  Recall@1 (mean): {aggregates['recall@1_mean']:.3f}")
    print(f"  F1@1 (mean): {aggregates['f1@1_mean']:.3f}")
    print(f"  MRR (mean): {aggregates['mrr_mean']:.3f}")
    print(f"  MAP (mean): {aggregates['map_mean']:.3f}")
    print(f"\n  Average retrieval time: {aggregates['avg_retrieval_time']:.3f}s")
    print(f"  Average generation time: {aggregates['avg_generation_time']:.3f}s")
    print(f"  Average total time: {aggregates['avg_total_time']:.3f}s")
    print(f"  Success rate: {aggregates['success_rate']:.1%}")


async def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("PETROLEUM RAG EVALUATION FRAMEWORK DEMO")
    print("=" * 80)

    await demo_metrics()
    await demo_llm_metrics()
    await demo_evaluator()
    await demo_aggregate_metrics()

    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)
    print("\nFor full functionality with LLM-based metrics:")
    print("  1. Set ANTHROPIC_API_KEY in .env")
    print("  2. Use evaluator.evaluate_query() or evaluator.evaluate_batch()")
    print("\nSee tests/test_metrics.py and tests/test_evaluator.py for more examples.")


if __name__ == "__main__":
    asyncio.run(main())
