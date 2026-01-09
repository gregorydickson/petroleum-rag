"""Example usage of Vertex Document AI parser.

This example demonstrates how to use the VertexDocAIParser to parse
petroleum engineering documents with enterprise-grade OCR and layout analysis.

Prerequisites:
1. Install dependencies: pip install google-cloud-documentai
2. Set up Google Cloud credentials:
   - Create a service account in GCP
   - Download the JSON key file
   - Set GOOGLE_APPLICATION_CREDENTIALS environment variable
3. Create a Document AI processor in GCP Console
4. Configure environment variables in .env:
   - GOOGLE_CLOUD_PROJECT
   - VERTEX_DOCAI_PROCESSOR_ID
   - VERTEX_DOCAI_LOCATION (optional, defaults to "us")

Usage:
    python examples/vertex_parser_example.py path/to/document.pdf
"""

import asyncio
import sys
from pathlib import Path

from parsers.vertex_parser import VertexDocAIParser


async def main():
    """Demonstrate Vertex Document AI parser usage."""
    if len(sys.argv) < 2:
        print("Usage: python vertex_parser_example.py <path_to_document>")
        sys.exit(1)

    file_path = Path(sys.argv[1])

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    print(f"Parsing document: {file_path}")
    print("-" * 80)

    # Initialize parser
    try:
        parser = VertexDocAIParser(config={"chunk_size": 1000, "chunk_overlap": 200})
        print(f"✓ Initialized {parser.name} parser")
        print(f"  Project: {parser.project_id}")
        print(f"  Location: {parser.location}")
        print(f"  Processor: {parser.processor_id}")
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        print("\nPlease ensure you have set the following environment variables:")
        print("  - GOOGLE_CLOUD_PROJECT")
        print("  - VERTEX_DOCAI_PROCESSOR_ID")
        print("  - GOOGLE_APPLICATION_CREDENTIALS (path to service account key)")
        sys.exit(1)

    # Parse document
    print(f"\nParsing {file_path.name}...")
    parsed_doc = await parser.parse(file_path)

    if parsed_doc.error:
        print(f"✗ Parsing failed: {parsed_doc.error}")
        sys.exit(1)

    print(f"✓ Successfully parsed document in {parsed_doc.parse_time_seconds:.2f}s")
    print(f"\nDocument Statistics:")
    print(f"  Document ID: {parsed_doc.document_id}")
    print(f"  Total Pages: {parsed_doc.total_pages}")
    print(f"  Elements Extracted: {len(parsed_doc.elements)}")

    # Show element types breakdown
    element_types = {}
    for element in parsed_doc.elements:
        element_types[element.element_type.value] = (
            element_types.get(element.element_type.value, 0) + 1
        )

    print(f"\nElement Types:")
    for elem_type, count in sorted(element_types.items()):
        print(f"  {elem_type}: {count}")

    # Show sample elements
    print(f"\nSample Elements (first 3):")
    for i, element in enumerate(parsed_doc.elements[:3], 1):
        print(f"\n  Element {i}: {element.element_type.value}")
        print(f"    ID: {element.element_id}")
        print(f"    Page: {element.page_number}")
        content_preview = element.content[:100].replace("\n", " ")
        if len(element.content) > 100:
            content_preview += "..."
        print(f"    Content: {content_preview}")
        if element.bbox:
            print(f"    Bounding Box: {element.bbox}")

    # Chunk document
    print(f"\nChunking document...")
    chunks = parser.chunk_document(parsed_doc)
    print(f"✓ Created {len(chunks)} chunks")

    print(f"\nChunk Statistics:")
    total_tokens = sum(c.token_count or 0 for c in chunks)
    avg_chunk_size = sum(len(c.content) for c in chunks) / len(chunks) if chunks else 0
    print(f"  Total Tokens: {total_tokens}")
    print(f"  Average Chunk Size: {avg_chunk_size:.0f} characters")

    # Show sample chunks
    print(f"\nSample Chunks (first 2):")
    for i, chunk in enumerate(chunks[:2], 1):
        print(f"\n  Chunk {i}:")
        print(f"    ID: {chunk.chunk_id}")
        print(f"    Index: {chunk.chunk_index}")
        print(f"    Pages: {chunk.start_page}-{chunk.end_page}")
        print(f"    Elements: {len(chunk.element_ids)}")
        print(f"    Tokens: {chunk.token_count}")
        print(f"    Size: {len(chunk.content)} characters")
        content_preview = chunk.content[:150].replace("\n", " ")
        if len(chunk.content) > 150:
            content_preview += "..."
        print(f"    Content: {content_preview}")

    print("\n" + "-" * 80)
    print("✓ Parsing complete!")


if __name__ == "__main__":
    asyncio.run(main())
