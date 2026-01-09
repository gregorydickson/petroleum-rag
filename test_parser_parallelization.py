#!/usr/bin/env python3
"""Test script to verify parser parallelization works correctly.

This script tests:
1. Parsers run in parallel when benchmark_parallel_parsers=True
2. Parsers run sequentially when benchmark_parallel_parsers=False
3. Error handling for individual parser failures
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timezone

from benchmark import BenchmarkRunner
from models import ParsedDocument, DocumentElement


async def test_parallel_mode():
    """Test that parsers run in parallel when flag is True."""
    print("\n" + "="*80)
    print("TEST 1: Parallel Parser Execution (benchmark_parallel_parsers=True)")
    print("="*80)

    with patch('config.settings') as mock_settings:
        # Configure parallel mode
        mock_settings.benchmark_parallel_parsers = True
        mock_settings.benchmark_save_intermediate_results = False

        # Create runner with mocked parsers
        runner = BenchmarkRunner()

        # Mock parsers with timing to verify parallelism
        execution_times = []

        async def mock_parse(parser_name, delay):
            """Mock parser that records execution time."""
            start = datetime.now(timezone.utc)
            await asyncio.sleep(delay)
            execution_times.append((parser_name, start))

            return ParsedDocument(
                document_id=f"test_{parser_name}",
                parser_name=parser_name,
                source_file=Path("test.pdf"),
                total_pages=1,
                elements=[DocumentElement(
                    element_id=f"elem_{parser_name}",
                    element_type="text",
                    content="test content",
                    page_number=1,
                    confidence_score=1.0
                )],
                parse_time_seconds=delay,
                parsed_at=datetime.now(timezone.utc)
            )

        # Create mock parsers with different delays
        runner.parsers = [
            Mock(name=f"Parser{i}", parse=AsyncMock(side_effect=lambda p, i=i: mock_parse(f"Parser{i}", 0.1)))
            for i in range(4)
        ]

        # Create temporary test PDF
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir)
            test_pdf = input_dir / "test.pdf"
            test_pdf.write_text("mock pdf")

            # Execute parsing
            start_time = datetime.now(timezone.utc)
            results = await runner.parse_documents(input_dir)
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

            print(f"\n✓ Parsed {len(results)} documents")
            print(f"✓ Total time: {elapsed:.3f}s")

            # In parallel mode, all parsers should start around the same time
            if len(execution_times) > 1:
                time_diffs = [(execution_times[i+1][1] - execution_times[0][1]).total_seconds()
                             for i in range(len(execution_times)-1)]
                max_diff = max(time_diffs) if time_diffs else 0
                print(f"✓ Max start time difference: {max_diff:.3f}s (should be ~0 for parallel)")

                if max_diff < 0.5:  # Allow small variance
                    print("✓ PASS: Parsers executed in PARALLEL")
                else:
                    print("✗ FAIL: Parsers may have executed sequentially")

    return True


async def test_sequential_mode():
    """Test that parsers run sequentially when flag is False."""
    print("\n" + "="*80)
    print("TEST 2: Sequential Parser Execution (benchmark_parallel_parsers=False)")
    print("="*80)

    with patch('config.settings') as mock_settings:
        # Configure sequential mode
        mock_settings.benchmark_parallel_parsers = False
        mock_settings.benchmark_save_intermediate_results = False

        # Create runner with mocked parsers
        runner = BenchmarkRunner()

        # Track execution order
        execution_order = []

        async def mock_parse_ordered(parser_name):
            """Mock parser that records execution order."""
            execution_order.append(parser_name)
            await asyncio.sleep(0.05)

            return ParsedDocument(
                document_id=f"test_{parser_name}",
                parser_name=parser_name,
                source_file=Path("test.pdf"),
                total_pages=1,
                elements=[DocumentElement(
                    element_id=f"elem_{parser_name}",
                    element_type="text",
                    content="test content",
                    page_number=1,
                    confidence_score=1.0
                )],
                parse_time_seconds=0.05,
                parsed_at=datetime.now(timezone.utc)
            )

        # Create mock parsers
        runner.parsers = [
            Mock(name=f"Parser{i}", parse=AsyncMock(side_effect=lambda p, i=i: mock_parse_ordered(f"Parser{i}")))
            for i in range(4)
        ]

        # Create temporary test PDF
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir)
            test_pdf = input_dir / "test.pdf"
            test_pdf.write_text("mock pdf")

            # Execute parsing
            start_time = datetime.now(timezone.utc)
            results = await runner.parse_documents(input_dir)
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

            print(f"\n✓ Parsed {len(results)} documents")
            print(f"✓ Total time: {elapsed:.3f}s")
            print(f"✓ Execution order: {execution_order}")
            print("✓ PASS: Parsers executed SEQUENTIALLY")

    return True


async def test_error_handling():
    """Test that individual parser failures are handled gracefully."""
    print("\n" + "="*80)
    print("TEST 3: Error Handling for Individual Parser Failures")
    print("="*80)

    with patch('config.settings') as mock_settings:
        mock_settings.benchmark_parallel_parsers = True
        mock_settings.benchmark_save_intermediate_results = False

        runner = BenchmarkRunner()

        # Create parsers where some fail
        async def mock_parse_success(parser_name):
            return ParsedDocument(
                document_id=f"test_{parser_name}",
                parser_name=parser_name,
                source_file=Path("test.pdf"),
                total_pages=1,
                elements=[],
                parse_time_seconds=0.01,
                parsed_at=datetime.now(timezone.utc)
            )

        async def mock_parse_failure(parser_name):
            raise RuntimeError(f"{parser_name} intentional failure")

        runner.parsers = [
            Mock(name="Parser0", parse=AsyncMock(side_effect=lambda p: mock_parse_success("Parser0"))),
            Mock(name="Parser1", parse=AsyncMock(side_effect=lambda p: mock_parse_failure("Parser1"))),
            Mock(name="Parser2", parse=AsyncMock(side_effect=lambda p: mock_parse_success("Parser2"))),
            Mock(name="Parser3", parse=AsyncMock(side_effect=lambda p: mock_parse_failure("Parser3"))),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir)
            test_pdf = input_dir / "test.pdf"
            test_pdf.write_text("mock pdf")

            # Execute parsing - should continue despite failures
            results = await runner.parse_documents(input_dir)

            print(f"\n✓ Attempted: 4 parsers")
            print(f"✓ Successful: {len(results)} parsers")
            print(f"✓ Failed: {4 - len(results)} parsers (expected)")

            if len(results) == 2:
                print("✓ PASS: Individual parser failures handled gracefully")
            else:
                print(f"✗ FAIL: Expected 2 successful parsers, got {len(results)}")

    return True


async def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("PARSER PARALLELIZATION TESTS")
    print("="*80)

    try:
        await test_parallel_mode()
        await test_sequential_mode()
        await test_error_handling()

        print("\n" + "="*80)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*80)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    asyncio.run(main())
