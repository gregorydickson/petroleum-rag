"""Test runner that orchestrates test execution across all combinations."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from benchmark import BenchmarkRunner
from config import settings
from models import BenchmarkQuery
from petroleum_rag.testing.models import TestCase, TestResult
from petroleum_rag.testing.report_generator import ReportGenerator
from petroleum_rag.testing.validator import ValidationEngine
from utils.logging import get_logger

logger = get_logger(__name__)


class TestRunner(BenchmarkRunner):
    """Test runner that extends BenchmarkRunner with test case validation.

    Inherits parser/storage initialization and orchestration from BenchmarkRunner,
    adds test-specific validation and reporting.
    """

    def __init__(self) -> None:
        """Initialize test runner."""
        super().__init__()
        self.validator = ValidationEngine()
        self.report_generator = ReportGenerator()
        self.test_results: list[TestResult] = []

    async def run_test_suite(
        self,
        input_dir: Path,
        test_cases_file: Path,
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Run complete test suite against all combinations.

        Args:
            input_dir: Directory with input PDFs
            test_cases_file: Path to test_cases.json
            output_dir: Directory to save results (default: data/test_results/)

        Returns:
            Dictionary with summary statistics
        """
        if output_dir is None:
            output_dir = Path("data/test_results")

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("STARTING TEST SUITE EXECUTION")
        logger.info("=" * 80)

        start_time = datetime.now(timezone.utc)

        # Load test cases
        test_cases = self._load_test_cases(test_cases_file)
        logger.info(f"Loaded {len(test_cases)} test cases")

        # Use context manager for embedder
        async with self.embedder:
            # Initialize storage backends
            await self.initialize_storage()

            try:
                # Parse documents with all parsers
                parsed_docs = await self.parse_documents(input_dir)

                if not parsed_docs:
                    raise RuntimeError("No documents were successfully parsed")

                # For each parser output
                for parser_name, parsed_doc in parsed_docs.items():
                    logger.info(f"\n{'=' * 80}")
                    logger.info(f"Testing {parser_name}")
                    logger.info(f"{'=' * 80}")

                    # Store in all backends
                    await self.store_in_backends(parser_name, parsed_doc)

                    # Test against each storage backend
                    for backend in self.storage_backends:
                        logger.info(f"\nTesting {parser_name} + {backend.name}")

                        await self._run_test_cases(
                            test_cases=test_cases,
                            parser_name=parser_name,
                            storage_backend_name=backend.name,
                        )

                # Calculate aggregate statistics
                total_time = (datetime.now(timezone.utc) - start_time).total_seconds()

                # Generate summary
                combo_summaries = self.validator.aggregate_by_combination(self.test_results)

                summary = {
                    "total_combinations": len(self.parsers) * len(self.storage_backends),
                    "total_test_cases": len(test_cases),
                    "total_tests_run": len(self.test_results),
                    "total_time_seconds": total_time,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "combination_summaries": combo_summaries,
                }

                logger.info("\n" + "=" * 80)
                logger.info("TEST SUITE COMPLETE")
                logger.info("=" * 80)
                logger.info(f"Total combinations: {summary['total_combinations']}")
                logger.info(f"Total test cases: {summary['total_test_cases']}")
                logger.info(f"Total tests run: {summary['total_tests_run']}")
                logger.info(f"Total time: {total_time / 60:.1f} minutes")

                # Generate report
                report_path = self._generate_report(output_dir, test_cases, summary)
                logger.info(f"\nReport generated: {report_path}")

                return summary

            finally:
                # Cleanup storage backends
                for backend in self.storage_backends:
                    try:
                        if hasattr(backend, "close"):
                            await backend.close()
                    except Exception as e:
                        logger.warning(f"Error closing {backend.name}: {e}")

    async def _run_test_cases(
        self,
        test_cases: list[TestCase],
        parser_name: str,
        storage_backend_name: str,
    ) -> None:
        """Run all test cases against a specific combination.

        Args:
            test_cases: List of test cases to run
            parser_name: Parser name
            storage_backend_name: Storage backend name
        """
        # Get storage backend
        backend = next(b for b in self.storage_backends if b.name == storage_backend_name)

        for test_case in tqdm(
            test_cases,
            desc=f"{parser_name} + {storage_backend_name}",
            leave=False,
        ):
            try:
                # Convert TestCase to BenchmarkQuery
                query = BenchmarkQuery(
                    query_id=test_case.test_id,
                    query=test_case.question,
                    ground_truth_answer=test_case.ground_truth,
                    relevant_element_ids=[],  # Not used in test validation
                    query_type="general",
                    difficulty=test_case.difficulty,
                )

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

                # Get standard benchmark metrics (optional, for additional analysis)
                benchmark_result = await self.evaluator.evaluate_query(
                    query=query,
                    retrieved=retrieved,
                    generated_answer=generated_answer,
                    parser_name=parser_name,
                    storage_backend=storage_backend_name,
                    retrieval_time=retrieval_time,
                    generation_time=generation_time,
                )

                # Validate with test-specific logic
                test_result = self.validator.validate(
                    test_case=test_case,
                    generated_answer=generated_answer,
                    parser_name=parser_name,
                    storage_backend=storage_backend_name,
                    benchmark_metrics=benchmark_result.metrics,
                    retrieval_time=retrieval_time,
                    generation_time=generation_time,
                )

                self.test_results.append(test_result)

            except Exception as e:
                logger.error(
                    f"Failed test {test_case.test_id} for {parser_name} + {storage_backend_name}: {e}",
                    exc_info=True,
                )
                # Continue with other tests

    def _load_test_cases(self, test_cases_file: Path) -> list[TestCase]:
        """Load test cases from JSON file.

        Args:
            test_cases_file: Path to test_cases.json

        Returns:
            List of TestCase objects
        """
        with open(test_cases_file) as f:
            data = json.load(f)

        test_cases = []
        for item in data:
            test_case = TestCase(
                test_id=item["test_id"],
                question=item["question"],
                ground_truth=item["ground_truth"],
                required_terms=item.get("required_terms", []),
                forbidden_terms=item.get("forbidden_terms", []),
                source_location=item.get("source_location", ""),
                difficulty=item.get("difficulty", "medium"),
                scoring_type=item.get("scoring_type", "semantic"),
                failure_mode_tested=item.get("failure_mode_tested", ""),
            )
            test_cases.append(test_case)

        return test_cases

    def _generate_report(
        self,
        output_dir: Path,
        test_cases: list[TestCase],
        summary: dict[str, Any],
    ) -> Path:
        """Generate markdown report.

        Args:
            output_dir: Output directory
            test_cases: List of test cases
            summary: Summary statistics

        Returns:
            Path to generated report
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"test_results_{timestamp}.md"

        markdown_content = self.report_generator.generate_report(
            test_results=self.test_results,
            test_cases=test_cases,
            combination_summaries=summary["combination_summaries"],
            summary_stats=summary,
        )

        with open(report_path, "w") as f:
            f.write(markdown_content)

        return report_path
