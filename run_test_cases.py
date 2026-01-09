"""Command-line interface for running RAG test cases.

Usage:
    python run_test_cases.py
    python run_test_cases.py --parsers LlamaParse Docling
    python run_test_cases.py --storage ChromaDB Weaviate
    python run_test_cases.py --test-cases TC001 TC002 TC003
"""

import argparse
import asyncio
import sys
from pathlib import Path

from config import settings
from petroleum_rag.testing.test_runner import TestRunner
from utils.logging import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run petroleum RAG test cases across all parser-storage combinations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests (4 parsers × 3 storage × 10 test cases = 120 tests)
  python run_test_cases.py

  # Test only specific parsers
  python run_test_cases.py --parsers LlamaParse Docling

  # Test only specific storage backends
  python run_test_cases.py --storage Weaviate

  # Combine filters
  python run_test_cases.py --parsers LlamaParse --storage Weaviate ChromaDB
        """,
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/input"),
        help="Directory with input PDFs (default: data/input)",
    )

    parser.add_argument(
        "--test-cases-file",
        type=Path,
        default=Path("test_cases.json"),
        help="Path to test cases JSON file (default: test_cases.json)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/test_results"),
        help="Directory to save results (default: data/test_results)",
    )

    parser.add_argument(
        "--parsers",
        nargs="+",
        choices=["LlamaParse", "Docling", "PageIndex", "VertexDocAI"],
        help="Run only specified parsers (default: all)",
    )

    parser.add_argument(
        "--storage",
        nargs="+",
        choices=["ChromaDB", "Weaviate", "FalkorDB"],
        help="Run only specified storage backends (default: all)",
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    # Validate required API keys
    missing_keys = settings.validate_required_keys()
    if missing_keys:
        logger.error(f"Missing required API keys: {missing_keys}")
        logger.error("Please set them in .env file")
        sys.exit(1)

    # Validate input files exist
    if not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)

    if not args.test_cases_file.exists():
        logger.error(f"Test cases file not found: {args.test_cases_file}")
        sys.exit(1)

    # Initialize test runner
    runner = TestRunner()

    # Apply filters if specified
    if args.parsers:
        runner.parsers = [p for p in runner.parsers if p.name in args.parsers]
        logger.info(f"Filtered parsers: {[p.name for p in runner.parsers]}")

    if args.storage:
        runner.storage_backends = [
            s for s in runner.storage_backends if s.name in args.storage
        ]
        logger.info(f"Filtered storage: {[s.name for s in runner.storage_backends]}")

    try:
        # Run test suite
        summary = await runner.run_test_suite(
            input_dir=args.input_dir,
            test_cases_file=args.test_cases_file,
            output_dir=args.output_dir,
        )

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("TEST SUITE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total combinations tested: {summary['total_combinations']}")
        logger.info(f"Total test cases: {summary['total_test_cases']}")
        logger.info(f"Total tests run: {summary['total_tests_run']}")
        logger.info(f"Execution time: {summary['total_time_seconds'] / 60:.1f} minutes")

        # Find winner
        combo_summaries = summary["combination_summaries"]
        if combo_summaries:
            winner = max(combo_summaries.values(), key=lambda s: (s.pass_rate, s.avg_score))
            logger.info(f"\n🏆 WINNER: {winner.combination}")
            logger.info(f"   Pass rate: {winner.pass_rate * 100:.1f}%")
            logger.info(f"   Avg score: {winner.avg_score:.3f}")

    except Exception as e:
        logger.error(f"Test suite failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
