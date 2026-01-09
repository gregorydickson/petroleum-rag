"""Data models for test case execution and validation."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal


@dataclass
class TestCase:
    """Test case definition with validation criteria.

    Attributes:
        test_id: Unique identifier (e.g., TC001_ALKYLATION_ECONOMICS)
        question: Query text
        ground_truth: Expected answer
        required_terms: List of terms that must appear in answer
        forbidden_terms: List of terms indicating hallucination
        source_location: Source section in document
        difficulty: Difficulty level (easy, medium, hard, expert)
        scoring_type: Type of scoring (exact, numerical, semantic, multi-part)
        failure_mode_tested: What failure mode this tests
    """

    test_id: str
    question: str
    ground_truth: str
    required_terms: list[str] = field(default_factory=list)
    forbidden_terms: list[str] = field(default_factory=list)
    source_location: str = ""
    difficulty: Literal["easy", "medium", "hard", "expert"] = "medium"
    scoring_type: Literal["exact", "numerical", "semantic", "multi-part"] = "semantic"
    failure_mode_tested: str = ""


@dataclass
class TestResult:
    """Result of executing a single test case against one combination.

    Attributes:
        test_id: Test case identifier
        combination: Parser_Storage name
        parser_name: Parser used
        storage_backend: Storage used
        question: Original question
        generated_answer: Answer from RAG system
        ground_truth: Expected answer
        passed: Whether test passed
        score: Final weighted score (0.0-1.0)
        required_term_coverage: Percentage of required terms found
        required_terms_found: List of found terms
        required_terms_missing: List of missing terms
        forbidden_terms_found: List of hallucinated terms
        hallucination_penalty: Penalty applied for hallucinations
        benchmark_metrics: Dictionary of standard benchmark metrics
        retrieval_time_seconds: Time for retrieval
        generation_time_seconds: Time for answer generation
        timestamp: When test was run
    """

    test_id: str
    combination: str
    parser_name: str
    storage_backend: str
    question: str
    generated_answer: str
    ground_truth: str
    passed: bool
    score: float
    required_term_coverage: float
    required_terms_found: list[str]
    required_terms_missing: list[str]
    forbidden_terms_found: list[str]
    hallucination_penalty: float
    benchmark_metrics: dict[str, float] = field(default_factory=dict)
    retrieval_time_seconds: float = 0.0
    generation_time_seconds: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CombinationSummary:
    """Summary statistics for one parser-storage combination.

    Attributes:
        combination: Parser_Storage name
        parser_name: Parser used
        storage_backend: Storage used
        total_tests: Total test cases run
        passed_tests: Number of tests passed
        failed_tests: Number of tests failed
        pass_rate: Pass rate (0.0-1.0)
        avg_score: Average score across all tests
        avg_retrieval_time: Average retrieval time
        avg_generation_time: Average generation time
        total_time: Total time for all tests
        test_results: List of individual test results
    """

    combination: str
    parser_name: str
    storage_backend: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    pass_rate: float
    avg_score: float
    avg_retrieval_time: float
    avg_generation_time: float
    total_time: float
    test_results: list[TestResult] = field(default_factory=list)
