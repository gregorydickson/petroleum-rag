#!/usr/bin/env python
"""Verification script for Wave 2 integration layer.

This script checks that all Wave 2 components are properly installed and configured.
"""

import sys
from pathlib import Path


def check_imports():
    """Verify all required imports work."""
    print("Checking imports...")
    errors = []

    try:
        from benchmark import BenchmarkRunner
        print("  ✓ benchmark.BenchmarkRunner")
    except Exception as e:
        errors.append(f"  ✗ benchmark.BenchmarkRunner: {e}")

    try:
        from analyze_results import ResultsAnalyzer
        print("  ✓ analyze_results.ResultsAnalyzer")
    except Exception as e:
        errors.append(f"  ✗ analyze_results.ResultsAnalyzer: {e}")

    try:
        import streamlit
        print("  ✓ streamlit")
    except Exception as e:
        errors.append(f"  ✗ streamlit: {e}")

    try:
        from tqdm import tqdm
        print("  ✓ tqdm")
    except Exception as e:
        errors.append(f"  ✗ tqdm: {e}")

    try:
        from PIL import Image
        print("  ✓ PIL (pillow)")
    except Exception as e:
        errors.append(f"  ✗ PIL (pillow): {e}")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        print("  ✓ matplotlib, seaborn")
    except Exception as e:
        errors.append(f"  ✗ matplotlib, seaborn: {e}")

    try:
        import pandas as pd
        print("  ✓ pandas")
    except Exception as e:
        errors.append(f"  ✗ pandas: {e}")

    # Check Wave 1 imports
    try:
        from parsers import LlamaParseParser, DoclingParser, PageIndexParser, VertexDocAIParser
        print("  ✓ All parsers")
    except Exception as e:
        errors.append(f"  ✗ Parsers: {e}")

    try:
        from storage import ChromaStore, WeaviateStore, FalkorDBStore
        print("  ✓ All storage backends")
    except Exception as e:
        errors.append(f"  ✗ Storage backends: {e}")

    try:
        from embeddings import UnifiedEmbedder
        print("  ✓ UnifiedEmbedder")
    except Exception as e:
        errors.append(f"  ✗ UnifiedEmbedder: {e}")

    try:
        from evaluation import Evaluator, MetricsCalculator
        print("  ✓ Evaluator, MetricsCalculator")
    except Exception as e:
        errors.append(f"  ✗ Evaluator, MetricsCalculator: {e}")

    return errors


def check_files():
    """Verify all required files exist."""
    print("\nChecking files...")
    errors = []

    required_files = [
        "benchmark.py",
        "analyze_results.py",
        "demo_app.py",
        "WAVE_2_INTEGRATION_GUIDE.md",
        "QUICKSTART_WAVE2.md",
        "WAVE_2_COMPLETE.md",
        "evaluation/queries.json",
        "pyproject.toml",
    ]

    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✓ {file_path}")
        else:
            errors.append(f"  ✗ {file_path} not found")

    return errors


def check_directories():
    """Verify required directories exist."""
    print("\nChecking directories...")
    errors = []

    required_dirs = [
        "data/input",
        "data/parsed",
        "data/results",
        "parsers",
        "storage",
        "embeddings",
        "evaluation",
        "tests",
    ]

    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            print(f"  ✓ {dir_path}/")
        else:
            errors.append(f"  ✗ {dir_path}/ not found")

    return errors


def check_config():
    """Verify configuration."""
    print("\nChecking configuration...")
    errors = []

    try:
        from config import settings

        # Check required API keys
        missing_keys = settings.validate_required_keys()

        if missing_keys:
            for key in missing_keys:
                errors.append(f"  ✗ Missing API key: {key}")
        else:
            print("  ✓ All required API keys configured")

        # Check optional configurations
        print(f"  ✓ Chunk size: {settings.chunk_size}")
        print(f"  ✓ Chunk overlap: {settings.chunk_overlap}")
        print(f"  ✓ Retrieval top-k: {settings.retrieval_top_k}")
        print(f"  ✓ Embedding model: {settings.embedding_model}")
        print(f"  ✓ Eval LLM model: {settings.eval_llm_model}")

    except Exception as e:
        errors.append(f"  ✗ Config error: {e}")

    return errors


def check_docker_services():
    """Check if Docker services are running."""
    print("\nChecking Docker services...")
    import subprocess

    services = [
        ("ChromaDB", "8000"),
        ("Weaviate", "8080"),
        ("FalkorDB", "6379"),
    ]

    for service_name, port in services:
        try:
            result = subprocess.run(
                ["docker", "ps"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if port in result.stdout:
                print(f"  ✓ {service_name} (port {port})")
            else:
                print(f"  ⚠ {service_name} (port {port}) - not running")
        except Exception as e:
            print(f"  ⚠ Could not check Docker services: {e}")
            break


def main():
    """Run all verification checks."""
    print("=" * 80)
    print("Wave 2 Integration Verification")
    print("=" * 80)

    all_errors = []

    # Run checks
    all_errors.extend(check_imports())
    all_errors.extend(check_files())
    all_errors.extend(check_directories())
    all_errors.extend(check_config())

    # Docker services (warnings only)
    check_docker_services()

    # Summary
    print("\n" + "=" * 80)
    if all_errors:
        print(f"FAILED: {len(all_errors)} error(s) found")
        print("=" * 80)
        for error in all_errors:
            print(error)
        sys.exit(1)
    else:
        print("SUCCESS: All checks passed!")
        print("=" * 80)
        print("\nWave 2 integration is ready to use.")
        print("\nNext steps:")
        print("  1. Ensure Docker services are running: docker-compose up -d")
        print("  2. Add PDF files to data/input/")
        print("  3. Run benchmark: python benchmark.py")
        print("  4. Analyze results: python analyze_results.py")
        print("  5. Launch demo: streamlit run demo_app.py")
        sys.exit(0)


if __name__ == "__main__":
    main()
