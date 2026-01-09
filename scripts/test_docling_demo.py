"""Demo script to test Docling parser with a sample document.

This script demonstrates the Docling parser's capabilities:
- Document parsing with table extraction
- Structure preservation (HTML and Markdown)
- Semantic chunking
"""

import asyncio
import tempfile
from pathlib import Path

from parsers.docling_parser import DoclingParser


async def demo_docling_parser():
    """Demonstrate Docling parser functionality."""
    print("=" * 80)
    print("Docling Parser Demo for Petroleum RAG Benchmark")
    print("=" * 80)
    print()

    # Initialize parser
    parser = DoclingParser(config={"chunk_size": 500, "chunk_overlap": 100})
    print(f"✓ Initialized {parser.name} parser")
    print(f"  - Chunk size: {parser.get_chunk_size()} chars")
    print(f"  - Chunk overlap: {parser.get_chunk_overlap()} chars")
    print(f"  - OCR enabled: {parser.pipeline_options.do_ocr}")
    print(f"  - Table structure extraction: {parser.pipeline_options.do_table_structure}")
    print()

    # Create a sample document with text and tables
    sample_content = """
# Petroleum Engineering Data

## Introduction

This document provides key petroleum engineering data and measurements
for reservoir characterization and production optimization.

## Fluid Properties Table

| Property | Value | Units |
|----------|-------|-------|
| Oil Density | 850 | kg/m³ |
| Gas Density | 0.8 | kg/m³ |
| Viscosity | 10 | cP |
| API Gravity | 35 | °API |

## Production Data

The reservoir produces at the following rates:

- Oil rate: 500 bbl/day
- Gas rate: 1000 Mcf/day
- Water cut: 20%

## Reservoir Parameters

| Parameter | Value | Units |
|-----------|-------|-------|
| Porosity | 0.25 | fraction |
| Permeability | 100 | mD |
| Formation Volume Factor | 1.25 | bbl/STB |
| Solution GOR | 500 | scf/STB |

## Conclusion

These parameters are essential for reservoir simulation and production
forecasting in petroleum engineering applications.
"""

    # Create temporary PDF (in practice, would use a real PDF)
    # For this demo, we'll create a simple text file as Docling can handle multiple formats
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(sample_content)
        temp_file = Path(f.name)

    try:
        print(f"📄 Created sample document: {temp_file.name}")
        print()

        # Parse document
        print("⚙️  Parsing document with Docling...")
        parsed_doc = await parser.parse(temp_file)

        print(f"✓ Parsing complete!")
        print(f"  - Document ID: {parsed_doc.document_id}")
        print(f"  - Parser: {parsed_doc.parser_name}")
        print(f"  - Total elements: {len(parsed_doc.elements)}")
        print()

        # Show element breakdown
        from collections import Counter

        element_types = Counter(e.element_type.value for e in parsed_doc.elements)
        print("📊 Element breakdown:")
        for elem_type, count in element_types.items():
            print(f"  - {elem_type}: {count}")
        print()

        # Show sample elements
        print("📝 Sample elements:")
        for i, element in enumerate(parsed_doc.elements[:5], 1):
            print(f"\n{i}. {element.element_type.value.upper()}: {element.element_id}")
            content_preview = element.content[:100].replace("\n", " ")
            if len(element.content) > 100:
                content_preview += "..."
            print(f"   Content: {content_preview}")

            if element.element_type.value == "table" and element.formatted_content:
                print(f"   ✓ Has formatted content (HTML)")
                if element.metadata.get("markdown"):
                    print(f"   ✓ Has Markdown representation")

        # Chunk document
        print("\n" + "=" * 80)
        print("⚙️  Chunking document...")
        chunks = parser.chunk_document(parsed_doc)

        print(f"✓ Chunking complete!")
        print(f"  - Total chunks: {len(chunks)}")
        print(f"  - Average chunk size: {sum(len(c.content) for c in chunks) // len(chunks)} chars")
        print()

        # Show chunk statistics
        print("📊 Chunk statistics:")
        table_chunks = [c for c in chunks if any("table" in eid for eid in c.element_ids)]
        text_chunks = [c for c in chunks if not any("table" in eid for eid in c.element_ids)]
        print(f"  - Text chunks: {len(text_chunks)}")
        print(f"  - Table chunks: {len(table_chunks)}")
        print()

        # Show sample chunks
        print("📝 Sample chunks:")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\n{i}. Chunk {chunk.chunk_index}: {chunk.chunk_id}")
            print(f"   Elements: {', '.join(chunk.element_ids)}")
            print(f"   Size: {len(chunk.content)} chars ({chunk.token_count} tokens)")
            if chunk.parent_section:
                print(f"   Section: {chunk.parent_section}")

            content_preview = chunk.content[:150].replace("\n", " ")
            if len(chunk.content) > 150:
                content_preview += "..."
            print(f"   Content: {content_preview}")

        # Show table chunk details if available
        if table_chunks:
            print("\n" + "=" * 80)
            print("📊 Table Chunk Details:")
            table_chunk = table_chunks[0]
            print(f"\nChunk ID: {table_chunk.chunk_id}")
            print(f"Elements: {table_chunk.element_ids}")
            print(f"Size: {len(table_chunk.content)} chars")
            print("\nContent (first 300 chars):")
            print("-" * 80)
            print(table_chunk.content[:300])
            if len(table_chunk.content) > 300:
                print("...")
            print("-" * 80)

        print("\n" + "=" * 80)
        print("✓ Demo complete!")
        print("=" * 80)

    finally:
        # Clean up
        temp_file.unlink()


if __name__ == "__main__":
    asyncio.run(demo_docling_parser())
