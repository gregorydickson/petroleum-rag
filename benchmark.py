"""Benchmark runner for comparing parser and storage combinations.

This module orchestrates the complete benchmark pipeline:
1. Parse documents with all 4 parsers
2. Store in all 3 storage backends
3. Run test queries against all 12 combinations
4. Evaluate and save results
"""

import asyncio
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from config import settings
from embeddings import UnifiedEmbedder
from evaluation import Evaluator
from models import BenchmarkQuery, BenchmarkResult, ParsedDocument
from parsers import (
    DoclingParser,
    LlamaParseParser,
    PageIndexParser,
    VertexDocAIParser,
)
from storage import ChromaStore, FalkorDBStore, WeaviateStore
from utils.logging import get_logger
from utils.rate_limiter import setup_rate_limits

logger = get_logger(__name__)


class BenchmarkRunner:
    """Orchestrate comprehensive RAG system benchmark.

    This class manages the complete benchmark pipeline:
    - Initialize all parsers and storage backends
    - Parse documents with all parsers
    - Store in all backends
    - Run queries against all combinations
    - Calculate metrics and save results
    """

    def __init__(self) -> None:
        """Initialize benchmark runner with all components."""
        logger.info("Initializing BenchmarkRunner")

        # Setup rate limits for all services
        setup_rate_limits()
        logger.info("Rate limiting configured")

        # Initialize parsers
        self.parsers = [
            LlamaParseParser(),
            DoclingParser(),
            PageIndexParser(),
            VertexDocAIParser(),
        ]
        logger.info(f"Initialized {len(self.parsers)} parsers")

        # Initialize storage backends
        self.storage_backends = [
            ChromaStore(),
            WeaviateStore(),
            FalkorDBStore(),
        ]
        logger.info(f"Initialized {len(self.storage_backends)} storage backends")

        # Initialize embedder and evaluator
        self.embedder = UnifiedEmbedder()
        self.evaluator = Evaluator()
        logger.info("Initialized embedder and evaluator")

        # Results storage
        self.parsed_documents: dict[str, ParsedDocument] = {}
        self.benchmark_results: list[BenchmarkResult] = []

    async def initialize_storage(self) -> None:
        """Initialize all storage backends in parallel."""
        logger.info("Initializing storage backends...")

        tasks = [backend.initialize() for backend in self.storage_backends]

        try:
            await asyncio.gather(*tasks)
            logger.info("All storage backends initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize storage backends: {e}")
            raise

    async def parse_documents(self, input_dir: Path) -> dict[str, ParsedDocument]:
        """Parse all documents with all parsers in parallel.

        Args:
            input_dir: Directory containing input PDFs

        Returns:
            Dictionary mapping parser_name to ParsedDocument
        """
        logger.info(f"Parsing documents from {input_dir}")

        pdf_files = list(input_dir.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in {input_dir}")

        logger.info(f"Found {len(pdf_files)} PDF files")

        # For now, parse just the first document with all parsers
        # In production, would iterate over all documents
        pdf_file = pdf_files[0]
        logger.info(f"Parsing {pdf_file.name} with all parsers")

        parsed_docs = {}

        # Parse with each parser
        for parser in tqdm(self.parsers, desc="Parsing with all parsers"):
            try:
                logger.info(f"Parsing with {parser.name}")
                start_time = datetime.now(timezone.utc)

                parsed_doc = await parser.parse(pdf_file)

                elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                logger.info(
                    f"{parser.name}: Parsed in {elapsed:.2f}s, "
                    f"extracted {len(parsed_doc.elements)} elements"
                )

                parsed_docs[parser.name] = parsed_doc
                self.parsed_documents[parser.name] = parsed_doc

                # Save intermediate result
                if settings.benchmark_save_intermediate_results:
                    self._save_parsed_document(parsed_doc)

            except Exception as e:
                logger.error(f"Failed to parse with {parser.name}: {e}", exc_info=True)
                # Continue with other parsers

        logger.info(f"Successfully parsed with {len(parsed_docs)}/{len(self.parsers)} parsers")
        return parsed_docs

    async def store_in_backends(
        self,
        parser_name: str,
        parsed_doc: ParsedDocument,
    ) -> None:
        """Store parsed document in all storage backends.

        Args:
            parser_name: Name of parser used
            parsed_doc: Parsed document to store
        """
        logger.info(f"Storing {parser_name} output in all backends")

        # Get parser instance for chunking
        parser = next(p for p in self.parsers if p.name == parser_name)
        chunks = parser.chunk_document(parsed_doc)

        logger.info(f"Created {len(chunks)} chunks")

        # Generate embeddings for all chunks
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = await self.embedder.embed_batch(chunk_texts)

        logger.info(f"Generated {len(embeddings)} embeddings")

        # Store in each backend
        for backend in tqdm(self.storage_backends, desc=f"Storing {parser_name} in backends"):
            try:
                logger.info(f"Storing in {backend.name}")
                await backend.clear()  # Clear previous data
                await backend.store_chunks(chunks, embeddings)
                logger.info(f"Successfully stored in {backend.name}")

            except Exception as e:
                logger.error(f"Failed to store in {backend.name}: {e}", exc_info=True)
                # Continue with other backends

    async def run_queries(
        self,
        queries: list[BenchmarkQuery],
        parser_name: str,
        storage_backend_name: str,
    ) -> list[BenchmarkResult]:
        """Run benchmark queries against a specific parser-storage combination.

        Args:
            queries: List of benchmark queries
            parser_name: Name of parser
            storage_backend_name: Name of storage backend

        Returns:
            List of benchmark results
        """
        logger.info(f"Running {len(queries)} queries for {parser_name} + {storage_backend_name}")

        # Get storage backend
        backend = next(b for b in self.storage_backends if b.name == storage_backend_name)

        results = []

        for query in tqdm(
            queries,
            desc=f"{parser_name} + {storage_backend_name}",
            leave=False,
        ):
            try:
                # Generate query embedding
                query_embedding = await self.embedder.embed_text(query.query)

                # Retrieve results
                start_time = datetime.now(timezone.utc)
                retrieved = await backend.retrieve(
                    query=query.query,
                    query_embedding=query_embedding,
                    top_k=settings.retrieval_top_k,
                )
                retrieval_time = (datetime.now(timezone.utc) - start_time).total_seconds()

                # Generate answer
                start_time = datetime.now(timezone.utc)
                generated_answer = await self.evaluator.generate_answer(
                    query.query,
                    retrieved,
                )
                generation_time = (datetime.now(timezone.utc) - start_time).total_seconds()

                # Evaluate
                benchmark_result = await self.evaluator.evaluate_query(
                    query=query,
                    retrieved=retrieved,
                    generated_answer=generated_answer,
                    parser_name=parser_name,
                    storage_backend=storage_backend_name,
                    retrieval_time=retrieval_time,
                    generation_time=generation_time,
                )

                results.append(benchmark_result)
                self.benchmark_results.append(benchmark_result)

            except Exception as e:
                logger.error(
                    f"Failed query {query.query_id} for {parser_name} + {storage_backend_name}: {e}",
                    exc_info=True,
                )
                # Continue with other queries

        logger.info(
            f"Completed {len(results)}/{len(queries)} queries for "
            f"{parser_name} + {storage_backend_name}"
        )

        return results

    async def run_full_benchmark(
        self,
        input_dir: Path,
        queries_file: Path,
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Run complete benchmark pipeline.

        Args:
            input_dir: Directory with input PDFs
            queries_file: Path to queries JSON file
            output_dir: Directory to save results (default: data/results/)

        Returns:
            Dictionary with summary statistics
        """
        if output_dir is None:
            output_dir = Path("data/results")

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("STARTING FULL BENCHMARK")
        logger.info("=" * 80)

        start_time = datetime.now(timezone.utc)

        # Use context manager for embedder to ensure cleanup
        async with self.embedder:
            # Initialize storage backends with context managers
            # Note: We still initialize manually here because we need to keep them
            # open throughout the benchmark run
            await self.initialize_storage()

            try:
                # Load queries
                queries = self._load_queries(queries_file)
                logger.info(f"Loaded {len(queries)} benchmark queries")

                # Parse documents with all parsers
                parsed_docs = await self.parse_documents(input_dir)

                if not parsed_docs:
                    raise RuntimeError("No documents were successfully parsed")

                # For each parser output
                for parser_name, parsed_doc in parsed_docs.items():
                    logger.info(f"\n{'=' * 80}")
                    logger.info(f"Processing {parser_name}")
                    logger.info(f"{'=' * 80}")

                    # Store in all backends
                    await self.store_in_backends(parser_name, parsed_doc)

                    # Run queries against each storage backend
                    for backend in self.storage_backends:
                        logger.info(f"\nTesting {parser_name} + {backend.name}")

                        await self.run_queries(
                            queries=queries,
                            parser_name=parser_name,
                            storage_backend_name=backend.name,
                        )

                # Calculate aggregate statistics
                total_time = (datetime.now(timezone.utc) - start_time).total_seconds()

                summary = {
                    "total_combinations": len(self.parsers) * len(self.storage_backends),
                    "total_queries": len(queries),
                    "total_results": len(self.benchmark_results),
                    "total_time_seconds": total_time,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "parsers": [p.name for p in self.parsers],
                    "storage_backends": [b.name for b in self.storage_backends],
                }

                logger.info("\n" + "=" * 80)
                logger.info("BENCHMARK COMPLETE")
                logger.info("=" * 80)
                logger.info(f"Total combinations: {summary['total_combinations']}")
                logger.info(f"Total queries: {summary['total_queries']}")
                logger.info(f"Total results: {summary['total_results']}")
                logger.info(f"Total time: {total_time / 60:.1f} minutes")

                # Save results
                self._save_results(output_dir, summary)

                return summary

            finally:
                # Cleanup all storage backends
                for backend in self.storage_backends:
                    try:
                        if hasattr(backend, 'close'):
                            await backend.close()
                    except Exception as e:
                        logger.warning(f"Error closing {backend.name}: {e}")

    def _load_queries(self, queries_file: Path) -> list[BenchmarkQuery]:
        """Load benchmark queries from JSON file.

        Args:
            queries_file: Path to queries JSON

        Returns:
            List of BenchmarkQuery objects
        """
        with open(queries_file) as f:
            data = json.load(f)

        queries = []
        for item in data:
            query = BenchmarkQuery(
                query_id=item["query_id"],
                query=item["query"],
                ground_truth_answer=item["ground_truth_answer"],
                relevant_element_ids=item.get("relevant_element_ids", []),
                query_type=item.get("query_type", "general"),
                difficulty=item.get("difficulty", "medium"),
                notes=item.get("note"),
            )
            queries.append(query)

        return queries

    def _save_parsed_document(self, doc: ParsedDocument) -> None:
        """Save parsed document to disk.

        Args:
            doc: Parsed document to save
        """
        output_dir = Path("data/parsed")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{doc.parser_name}_{doc.document_id}.json"

        # Convert to dict (simplified - just metadata)
        doc_data = {
            "document_id": doc.document_id,
            "parser_name": doc.parser_name,
            "source_file": str(doc.source_file),
            "total_pages": doc.total_pages,
            "element_count": len(doc.elements),
            "parse_time_seconds": doc.parse_time_seconds,
            "parsed_at": doc.parsed_at.isoformat(),
            "error": doc.error,
        }

        with open(output_file, "w") as f:
            json.dump(doc_data, f, indent=2)

        logger.debug(f"Saved parsed document to {output_file}")

    def _save_results(self, output_dir: Path, summary: dict[str, Any]) -> None:
        """Save benchmark results to disk.

        Args:
            output_dir: Directory to save results
            summary: Summary statistics
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save raw results
        raw_results_file = output_dir / "raw_results.json"
        results_data = {
            "summary": summary,
            "results": [self._result_to_dict(r) for r in self.benchmark_results],
        }

        with open(raw_results_file, "w") as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"Saved raw results to {raw_results_file}")

        # Save comparison CSV
        comparison_file = output_dir / "comparison.csv"
        self._save_comparison_csv(comparison_file)
        logger.info(f"Saved comparison CSV to {comparison_file}")

    def _result_to_dict(self, result: BenchmarkResult) -> dict[str, Any]:
        """Convert BenchmarkResult to serializable dictionary.

        Args:
            result: Benchmark result

        Returns:
            Dictionary representation
        """
        return {
            "benchmark_id": result.benchmark_id,
            "parser_name": result.parser_name,
            "storage_backend": result.storage_backend,
            "combination": result.combination_name,
            "query_id": result.query_id,
            "query": result.query,
            "generated_answer": result.generated_answer,
            "ground_truth_answer": result.ground_truth_answer,
            "metrics": result.metrics,
            "retrieval_time_seconds": result.retrieval_time_seconds,
            "generation_time_seconds": result.generation_time_seconds,
            "total_time_seconds": result.total_time_seconds,
            "timestamp": result.timestamp.isoformat(),
            "success": result.success,
            "error": result.error,
            "retrieved_count": len(result.retrieved_results),
        }

    def _save_comparison_csv(self, output_file: Path) -> None:
        """Save results as CSV for easy comparison.

        Args:
            output_file: Path to CSV file
        """
        import csv

        if not self.benchmark_results:
            logger.warning("No results to save to CSV")
            return

        # Calculate aggregate metrics per combination
        from collections import defaultdict

        combination_metrics: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for result in self.benchmark_results:
            combo = result.combination_name
            for metric_name, metric_value in result.metrics.items():
                combination_metrics[combo][metric_name].append(metric_value)

            # Add timing metrics
            combination_metrics[combo]["retrieval_time"].append(result.retrieval_time_seconds)
            combination_metrics[combo]["generation_time"].append(result.generation_time_seconds)
            combination_metrics[combo]["total_time"].append(result.total_time_seconds)

        # Calculate averages
        rows = []
        for combo, metrics in combination_metrics.items():
            row = {"combination": combo}

            for metric_name, values in metrics.items():
                if values:
                    row[f"{metric_name}_mean"] = sum(values) / len(values)

            rows.append(row)

        # Write CSV
        if rows:
            with open(output_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)


async def main():
    """Main entry point for benchmark."""
    import argparse

    parser = argparse.ArgumentParser(description="Run RAG system benchmark")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/input"),
        help="Directory with input PDFs",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("evaluation/queries.json"),
        help="Path to queries JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/results"),
        help="Directory to save results",
    )

    args = parser.parse_args()

    # Validate required API keys
    missing_keys = settings.validate_required_keys()
    if missing_keys:
        logger.error(f"Missing required API keys: {missing_keys}")
        logger.error("Please set them in .env file")
        return

    # Run benchmark
    runner = BenchmarkRunner()

    try:
        summary = await runner.run_full_benchmark(
            input_dir=args.input_dir,
            queries_file=args.queries,
            output_dir=args.output_dir,
        )

        logger.info("\n" + "=" * 80)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 80)
        for key, value in summary.items():
            logger.info(f"{key}: {value}")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
