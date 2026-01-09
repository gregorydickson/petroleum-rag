"""Demo script for LlamaParse parser.

This script demonstrates how to use the LlamaParse parser to parse
petroleum engineering documents and extract structured content.

Usage:
    python demo_llamaparse.py path/to/document.pdf

Requirements:
    - LLAMA_CLOUD_API_KEY must be set in .env file
    - PDF file to parse
"""

import asyncio
import sys
from pathlib import Path

from config import settings
from parsers.llamaparse_parser import LlamaParseParser
from models import ElementType


async def main():
    """Demonstrate LlamaParse parser capabilities."""

    # Check if PDF path provided
    if len(sys.argv) < 2:
        print("Usage: python demo_llamaparse.py path/to/document.pdf")
        print("\nThis demo will:")
        print("  1. Parse the PDF using LlamaParse API")
        print("  2. Extract structured elements (headings, tables, text, figures)")
        print("  3. Display parsing statistics and sample elements")
        print("  4. Generate intelligent chunks for RAG")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])

    # Validate API key
    if not settings.llama_cloud_api_key:
        print("ERROR: LLAMA_CLOUD_API_KEY not set.")
        print("Please add it to your .env file:")
        print("  LLAMA_CLOUD_API_KEY=llx-your-api-key-here")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"LlamaParse Parser Demo")
    print(f"{'='*70}")
    print(f"File: {pdf_path.name}")
    print(f"{'='*70}\n")

    # Initialize parser
    print("Initializing LlamaParse parser...")
    parser = LlamaParseParser()
    print(f"✓ Parser initialized: {parser.name}\n")

    # Parse document
    print("Parsing document (this may take a moment)...")
    parsed_doc = await parser.parse(pdf_path)

    if parsed_doc.error:
        print(f"✗ Parsing failed: {parsed_doc.error}")
        sys.exit(1)

    print(f"✓ Parsing completed in {parsed_doc.parse_time_seconds:.2f} seconds\n")

    # Display parsing statistics
    print(f"{'='*70}")
    print(f"Parsing Statistics")
    print(f"{'='*70}")
    print(f"Document ID:      {parsed_doc.document_id}")
    print(f"Total Pages:      {parsed_doc.total_pages}")
    print(f"Total Elements:   {len(parsed_doc.elements)}")
    print(f"Parse Time:       {parsed_doc.parse_time_seconds:.2f}s")

    # Count element types
    element_counts = {}
    for element in parsed_doc.elements:
        element_type = element.element_type.value
        element_counts[element_type] = element_counts.get(element_type, 0) + 1

    print(f"\nElement Type Breakdown:")
    for element_type, count in sorted(element_counts.items()):
        print(f"  {element_type.capitalize():12} {count:4d}")

    # Display sample elements
    print(f"\n{'='*70}")
    print(f"Sample Elements")
    print(f"{'='*70}\n")

    # Show first 3 headings
    headings = [e for e in parsed_doc.elements if e.element_type == ElementType.HEADING]
    if headings:
        print("Headings (first 3):")
        for i, heading in enumerate(headings[:3], 1):
            level = heading.metadata.get('level', '?')
            print(f"  {i}. [H{level}] {heading.content[:60]}...")
        print()

    # Show first table
    tables = [e for e in parsed_doc.elements if e.element_type == ElementType.TABLE]
    if tables:
        print(f"Tables found: {len(tables)}")
        print("First table preview:")
        table_lines = tables[0].content.split('\n')
        for line in table_lines[:5]:
            print(f"  {line}")
        if len(table_lines) > 5:
            print(f"  ... ({len(table_lines) - 5} more lines)")
        print()

    # Show figures
    figures = [e for e in parsed_doc.elements if e.element_type == ElementType.FIGURE]
    if figures:
        print(f"Figures found: {len(figures)}")
        for i, fig in enumerate(figures[:3], 1):
            print(f"  {i}. {fig.content} (page {fig.page_number})")
        print()

    # Chunking demonstration
    print(f"{'='*70}")
    print(f"Chunking Demonstration")
    print(f"{'='*70}\n")

    print("Generating chunks with intelligent boundaries...")
    chunks = parser.chunk_document(parsed_doc)

    print(f"✓ Generated {len(chunks)} chunks\n")

    print("Chunk Statistics:")
    total_chars = sum(len(chunk.content) for chunk in chunks)
    avg_chunk_size = total_chars / len(chunks) if chunks else 0
    print(f"  Total characters:  {total_chars:,}")
    print(f"  Average chunk size: {avg_chunk_size:.0f} chars")
    print(f"  Configured size:    {parser.get_chunk_size()} chars")
    print(f"  Configured overlap: {parser.get_chunk_overlap()} chars")

    # Show sample chunk
    if chunks:
        print(f"\nSample Chunk (chunk_0):")
        print(f"  ID:            {chunks[0].chunk_id}")
        print(f"  Pages:         {chunks[0].start_page}-{chunks[0].end_page}")
        print(f"  Elements:      {len(chunks[0].element_ids)}")
        print(f"  Tokens:        ~{chunks[0].token_count}")
        print(f"  Section:       {chunks[0].parent_section or 'N/A'}")
        print(f"  Content preview:")
        preview = chunks[0].content[:200].replace('\n', ' ')
        print(f"    {preview}...")

    print(f"\n{'='*70}")
    print(f"Demo Complete!")
    print(f"{'='*70}\n")

    print("Key Features Demonstrated:")
    print("  ✓ Async parsing with LlamaParse API")
    print("  ✓ Structured element extraction")
    print("  ✓ Table preservation")
    print("  ✓ Section hierarchy tracking")
    print("  ✓ Intelligent chunking")
    print("  ✓ Performance metrics\n")


if __name__ == "__main__":
    asyncio.run(main())
