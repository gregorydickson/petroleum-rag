#!/usr/bin/env python3
"""Verification script to ensure Wave 0 foundation is complete and correct.

This script validates:
1. All required files exist
2. All modules can be imported
3. Base classes have required methods
4. Data models are properly defined
5. Configuration loads successfully
"""

import sys
from pathlib import Path
from typing import Any


def check_file_exists(file_path: Path) -> bool:
    """Check if a file exists."""
    if file_path.exists():
        print(f"✅ {file_path}")
        return True
    else:
        print(f"❌ {file_path} - NOT FOUND")
        return False


def check_directory_exists(dir_path: Path) -> bool:
    """Check if a directory exists."""
    if dir_path.is_dir():
        print(f"✅ {dir_path}/")
        return True
    else:
        print(f"❌ {dir_path}/ - NOT FOUND")
        return False


def main() -> int:
    """Run all verification checks."""
    project_root = Path(__file__).parent
    all_checks_passed = True

    print("=" * 70)
    print("WAVE 0 FOUNDATION VERIFICATION")
    print("=" * 70)

    # Check project structure
    print("\n📁 Project Structure:")
    print("-" * 70)

    required_dirs = [
        project_root / "parsers",
        project_root / "storage",
        project_root / "embeddings",
        project_root / "evaluation",
        project_root / "data",
        project_root / "data" / "input",
        project_root / "data" / "parsed",
        project_root / "data" / "results",
        project_root / "tests",
    ]

    for dir_path in required_dirs:
        if not check_directory_exists(dir_path):
            all_checks_passed = False

    # Check required files
    print("\n📄 Required Files:")
    print("-" * 70)

    required_files = [
        project_root / "pyproject.toml",
        project_root / ".env.example",
        project_root / ".gitignore",
        project_root / "README.md",
        project_root / "models.py",
        project_root / "config.py",
        project_root / "parsers" / "__init__.py",
        project_root / "parsers" / "base.py",
        project_root / "storage" / "__init__.py",
        project_root / "storage" / "base.py",
        project_root / "embeddings" / "__init__.py",
        project_root / "evaluation" / "__init__.py",
        project_root / "tests" / "__init__.py",
    ]

    for file_path in required_files:
        if not check_file_exists(file_path):
            all_checks_passed = False

    # Check imports
    print("\n📦 Module Imports:")
    print("-" * 70)

    try:
        import models

        print("✅ models imported successfully")

        # Check data models
        required_models = [
            "ParsedElement",
            "ParsedDocument",
            "DocumentChunk",
            "RetrievalResult",
            "BenchmarkQuery",
            "BenchmarkResult",
            "ElementType",
            "QueryType",
            "DifficultyLevel",
        ]

        for model_name in required_models:
            if hasattr(models, model_name):
                print(f"  ✅ {model_name}")
            else:
                print(f"  ❌ {model_name} - NOT FOUND")
                all_checks_passed = False

    except ImportError as e:
        print(f"❌ Failed to import models: {e}")
        all_checks_passed = False

    try:
        from parsers.base import BaseParser

        print("✅ parsers.base.BaseParser imported successfully")

        # Check required methods
        required_methods = ["parse", "chunk_document"]
        for method_name in required_methods:
            if hasattr(BaseParser, method_name):
                print(f"  ✅ BaseParser.{method_name}()")
            else:
                print(f"  ❌ BaseParser.{method_name}() - NOT FOUND")
                all_checks_passed = False

    except ImportError as e:
        print(f"❌ Failed to import BaseParser: {e}")
        all_checks_passed = False

    try:
        from storage.base import BaseStorage

        print("✅ storage.base.BaseStorage imported successfully")

        # Check required methods
        required_methods = ["initialize", "store_chunks", "retrieve", "clear"]
        for method_name in required_methods:
            if hasattr(BaseStorage, method_name):
                print(f"  ✅ BaseStorage.{method_name}()")
            else:
                print(f"  ❌ BaseStorage.{method_name}() - NOT FOUND")
                all_checks_passed = False

    except ImportError as e:
        print(f"❌ Failed to import BaseStorage: {e}")
        all_checks_passed = False

    try:
        from config import Settings, settings

        print("✅ config.Settings imported successfully")
        print(f"  ✅ Global settings instance created")

        # Check key configuration attributes
        required_attrs = [
            "chunk_size",
            "chunk_overlap",
            "embedding_model",
            "eval_llm_model",
            "chroma_host",
            "weaviate_host",
            "falkordb_host",
        ]

        for attr_name in required_attrs:
            if hasattr(settings, attr_name):
                print(f"  ✅ settings.{attr_name}")
            else:
                print(f"  ❌ settings.{attr_name} - NOT FOUND")
                all_checks_passed = False

    except ImportError as e:
        print(f"❌ Failed to import config: {e}")
        all_checks_passed = False

    # Final summary
    print("\n" + "=" * 70)
    if all_checks_passed:
        print("✅ ALL CHECKS PASSED - Wave 0 Foundation Complete!")
        print("=" * 70)
        print("\n🚀 Ready for Wave 1: Core Implementation")
        print("\nNext steps:")
        print("  1. Start storage backends: docker-compose up -d")
        print("  2. Configure .env with API keys")
        print("  3. Launch Wave 1 parallel agents")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - Please review errors above")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
