"""Validation engine for test case evaluation."""

import re
from collections import defaultdict

from petroleum_rag.testing.models import CombinationSummary, TestCase, TestResult
from utils.logging import get_logger

logger = get_logger(__name__)


class ValidationEngine:
    """Validates RAG responses against test case requirements.

    Implements term matching, hallucination detection, and scoring logic
    aligned with the test case workplan requirements.
    """

    # Difficulty weights for scoring
    DIFFICULTY_WEIGHTS = {
        "easy": 1.0,
        "medium": 1.5,
        "hard": 2.0,
        "expert": 3.0,
    }

    # Pass threshold
    PASS_THRESHOLD = 0.70

    def __init__(self) -> None:
        """Initialize validation engine."""
        self.logger = logger

    def validate(
        self,
        test_case: TestCase,
        generated_answer: str,
        parser_name: str,
        storage_backend: str,
        benchmark_metrics: dict[str, float],
        retrieval_time: float,
        generation_time: float,
    ) -> TestResult:
        """Validate a generated answer against test case requirements.

        Args:
            test_case: Test case definition
            generated_answer: Answer from RAG system
            parser_name: Parser used
            storage_backend: Storage backend used
            benchmark_metrics: Standard benchmark metrics
            retrieval_time: Time for retrieval
            generation_time: Time for answer generation

        Returns:
            TestResult with validation scores
        """
        # Normalize answer for comparison
        answer_lower = generated_answer.lower()

        # Check required terms
        required_found = []
        required_missing = []
        for term in test_case.required_terms:
            if self._term_match(term, answer_lower):
                required_found.append(term)
            else:
                required_missing.append(term)

        # Calculate required term coverage
        if test_case.required_terms:
            required_coverage = len(required_found) / len(test_case.required_terms)
        else:
            required_coverage = 1.0

        # Check forbidden terms (hallucinations)
        forbidden_found = []
        for term in test_case.forbidden_terms:
            if self._term_match(term, answer_lower):
                forbidden_found.append(term)

        # Calculate hallucination penalty (-20% per forbidden term)
        hallucination_penalty = len(forbidden_found) * 0.20

        # Calculate base score
        base_score = max(0.0, required_coverage - hallucination_penalty)

        # Apply difficulty weighting
        difficulty_weight = self.DIFFICULTY_WEIGHTS[test_case.difficulty]
        weighted_score = base_score * difficulty_weight

        # Normalize to 0-1 range (expert difficulty has 3x weight)
        final_score = min(1.0, weighted_score / difficulty_weight)

        # Determine pass/fail
        passed = (final_score >= self.PASS_THRESHOLD) and (len(forbidden_found) == 0)

        # Log validation details
        self.logger.info(
            f"Validated {test_case.test_id} for {parser_name}_{storage_backend}: "
            f"Score={final_score:.2f}, Passed={passed}"
        )

        if not passed:
            self.logger.warning(
                f"Test {test_case.test_id} FAILED: "
                f"Missing terms: {required_missing}, "
                f"Hallucinations: {forbidden_found}"
            )

        # Create result object
        return TestResult(
            test_id=test_case.test_id,
            combination=f"{parser_name}_{storage_backend}",
            parser_name=parser_name,
            storage_backend=storage_backend,
            question=test_case.question,
            generated_answer=generated_answer,
            ground_truth=test_case.ground_truth,
            passed=passed,
            score=final_score,
            required_term_coverage=required_coverage,
            required_terms_found=required_found,
            required_terms_missing=required_missing,
            forbidden_terms_found=forbidden_found,
            hallucination_penalty=hallucination_penalty,
            benchmark_metrics=benchmark_metrics,
            retrieval_time_seconds=retrieval_time,
            generation_time_seconds=generation_time,
        )

    def _term_match(self, term: str, text: str) -> bool:
        """Check if term appears in text with flexible matching.

        Handles:
        - Case-insensitive matching
        - Partial number matching ($2,500 matches $2500)
        - Word boundary matching for non-numeric terms

        Args:
            term: Term to search for
            text: Text to search in

        Returns:
            True if term found
        """
        term_lower = term.lower()

        # Numeric term handling - strip formatting
        if any(c.isdigit() for c in term):
            # Remove common formatting characters
            term_normalized = re.sub(r"[$,\s]", "", term_lower)
            text_normalized = re.sub(r"[$,\s]", "", text)
            return term_normalized in text_normalized

        # Regular text matching with word boundaries
        pattern = r"\b" + re.escape(term_lower) + r"\b"
        return bool(re.search(pattern, text))

    def aggregate_by_combination(
        self,
        test_results: list[TestResult],
    ) -> dict[str, CombinationSummary]:
        """Aggregate test results by parser-storage combination.

        Args:
            test_results: List of all test results

        Returns:
            Dictionary mapping combination name to summary
        """
        # Group results by combination
        combo_results: dict[str, list[TestResult]] = defaultdict(list)
        for result in test_results:
            combo_results[result.combination].append(result)

        # Create summaries
        summaries = {}
        for combo, results in combo_results.items():
            passed = sum(1 for r in results if r.passed)
            failed = len(results) - passed
            pass_rate = passed / len(results) if results else 0.0
            avg_score = sum(r.score for r in results) / len(results) if results else 0.0
            avg_retrieval = (
                sum(r.retrieval_time_seconds for r in results) / len(results)
                if results
                else 0.0
            )
            avg_generation = (
                sum(r.generation_time_seconds for r in results) / len(results)
                if results
                else 0.0
            )
            total_time = sum(
                r.retrieval_time_seconds + r.generation_time_seconds for r in results
            )

            # Extract parser and storage names
            parser_name = results[0].parser_name if results else ""
            storage_backend = results[0].storage_backend if results else ""

            summaries[combo] = CombinationSummary(
                combination=combo,
                parser_name=parser_name,
                storage_backend=storage_backend,
                total_tests=len(results),
                passed_tests=passed,
                failed_tests=failed,
                pass_rate=pass_rate,
                avg_score=avg_score,
                avg_retrieval_time=avg_retrieval,
                avg_generation_time=avg_generation,
                total_time=total_time,
                test_results=results,
            )

        return summaries
