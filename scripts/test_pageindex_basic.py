"""Basic validation test for PageIndex parser implementation."""

import asyncio
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import directly to avoid other parser dependencies
import importlib.util
spec = importlib.util.spec_from_file_location(
    "pageindex_parser",
    Path(__file__).parent / "parsers" / "pageindex_parser.py"
)
pageindex_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pageindex_module)
PageIndexParser = pageindex_module.PageIndexParser

from models import ElementType


async def test_basic_functionality():
    """Test that PageIndexParser can be instantiated and has required methods."""
    print("Testing PageIndexParser basic functionality...\n")

    # Test 1: Instantiation
    print("✓ Test 1: Instantiation")
    parser = PageIndexParser()
    assert parser.name == "PageIndex"
    assert hasattr(parser, "use_fallback")
    print(f"  - Parser mode: {'fallback' if parser.use_fallback else 'native'}")

    # Test 2: Configuration
    print("\n✓ Test 2: Configuration")
    config = {
        "chunk_size": 1500,
        "chunk_overlap": 300,
        "semantic_threshold": 0.8,
        "preserve_structure": True,
    }
    parser_with_config = PageIndexParser(config=config)
    assert parser_with_config.get_chunk_size() == 1500
    assert parser_with_config.get_chunk_overlap() == 300
    assert parser_with_config.semantic_threshold == 0.8
    assert parser_with_config.preserve_structure is True
    print("  - Custom configuration applied correctly")

    # Test 3: Required methods exist
    print("\n✓ Test 3: Required methods")
    assert hasattr(parser, "parse")
    assert hasattr(parser, "chunk_document")
    assert callable(parser.parse)
    assert callable(parser.chunk_document)
    print("  - parse() method exists")
    print("  - chunk_document() method exists")

    # Test 4: Helper methods
    print("\n✓ Test 4: Helper methods")
    assert hasattr(parser, "_create_semantic_units")
    assert hasattr(parser, "_chunk_semantic_unit")
    assert hasattr(parser, "_infer_element_type")
    print("  - Semantic chunking methods present")

    # Test 5: Element type detection heuristics
    print("\n✓ Test 5: Element type detection")

    # Test heading detection
    heading_text = "1. Introduction"
    block = {"lines": []}
    lines = [heading_text]
    assert parser._is_heading(heading_text, block) is True
    print("  - Heading detection works")

    # Test list detection
    list_lines = ["• First item", "• Second item", "• Third item"]
    assert parser._is_list(list_lines) is True
    print("  - List detection works")

    # Test table detection
    table_text = "Column1\tColumn2\tColumn3\nValue1\tValue2\tValue3"
    assert parser._looks_like_table(table_text) is True
    print("  - Table detection works")

    # Test 6: Token estimation
    print("\n✓ Test 6: Token estimation")
    test_text = "This is a test sentence with multiple words."
    tokens = parser.estimate_tokens(test_text)
    assert tokens > 0
    assert tokens == len(test_text) // 4
    print(f"  - Token estimation: {tokens} tokens for {len(test_text)} chars")

    print("\n" + "=" * 60)
    print("All basic tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_basic_functionality())
