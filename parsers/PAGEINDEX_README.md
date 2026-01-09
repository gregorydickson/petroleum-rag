# PageIndex Parser

## Overview

The PageIndex parser implements novel semantic chunking capabilities for document parsing. It demonstrates an advanced approach to document understanding that goes beyond simple fixed-size chunking.

## Key Features

### 1. Semantic Boundary Detection
- Identifies natural semantic boundaries (sections, paragraphs, topics)
- Respects document structure and hierarchy
- Avoids splitting mid-thought or mid-concept

### 2. Structure-Aware Chunking
- Maintains document hierarchy (headings, sections, subsections)
- Preserves relationships between elements
- Groups related content together

### 3. Intelligent Overlap
- Creates overlap based on semantic similarity, not just character count
- Maintains context across chunk boundaries
- Includes relevant elements from previous chunks

### 4. Context Preservation
- Tracks parent sections and structural relationships
- Includes metadata about element types in chunks
- Preserves page numbers and bounding boxes

## Implementation Modes

### Native Mode (when PageIndex is available)
```python
parser = PageIndexParser(config={"api_key": "your_key"})
doc = await parser.parse(file_path)
chunks = parser.chunk_document(doc)
```

The native mode leverages the actual PageIndex library for:
- Advanced semantic analysis
- Machine learning-based element detection
- Superior table and figure extraction

### Fallback Mode (PyMuPDF-based)
```python
# Automatically activates when PageIndex is not available
parser = PageIndexParser()
doc = await parser.parse(file_path)
chunks = parser.chunk_document(doc)
```

The fallback mode provides:
- PyMuPDF-based PDF parsing
- Heuristic-based element type detection
- Structure-aware semantic chunking
- Similar API to native mode

## Configuration Options

```python
config = {
    # Chunking parameters
    "chunk_size": 1000,           # Target chunk size in characters
    "chunk_overlap": 200,         # Overlap size in characters

    # Semantic chunking parameters
    "semantic_threshold": 0.7,    # Similarity threshold for boundaries
    "preserve_structure": True,   # Maintain document hierarchy
}

parser = PageIndexParser(config=config)
```

## Semantic Chunking Algorithm

### Step 1: Create Semantic Units
The parser groups document elements into semantic units based on:
- **Headings**: Each heading starts a new semantic unit
- **Structural markers**: Page breaks, section boundaries
- **Content continuity**: Related paragraphs stay together

### Step 2: Chunk Semantic Units
Each semantic unit is divided into chunks while:
- Respecting the target chunk size
- Avoiding mid-sentence splits
- Maintaining semantic coherence
- Preserving important context

### Step 3: Intelligent Overlap
Overlap is created by:
1. Taking the last N characters from the previous chunk
2. Finding which elements contribute to that overlap
3. Including complete elements (not partial text)
4. Maintaining semantic meaning across boundaries

## Element Type Detection

The fallback mode uses heuristics to detect element types:

### Headings
- Short text (< 100 chars, < 15 words)
- Common patterns: numbered sections, ALL CAPS, "Chapter X"
- Often in larger font sizes

### Lists
- Bullet point markers (•, *, -, +)
- Numbered lists (1., 2., 3.)
- Lettered lists (a., b., c.)
- At least 50% of lines have markers

### Tables
- Consistent column separators (tabs, pipes)
- Regular structure across rows
- Multiple columns detected

### Code Blocks
- Contains programming keywords (def, class, import)
- Has code punctuation (brackets, semicolons)
- At least 30% of lines look like code

### Figures
- Image blocks from PDF
- Caption text near images

## Usage Examples

### Basic Parsing
```python
from pathlib import Path
from parsers import PageIndexParser

parser = PageIndexParser()

# Parse a document
doc = await parser.parse(Path("document.pdf"))

print(f"Parsed {len(doc.elements)} elements")
print(f"Total pages: {doc.total_pages}")
print(f"Parse time: {doc.parse_time_seconds:.2f}s")
```

### Chunking
```python
# Chunk the parsed document
chunks = parser.chunk_document(doc)

for chunk in chunks:
    print(f"Chunk {chunk.chunk_index}:")
    print(f"  Pages: {chunk.start_page}-{chunk.end_page}")
    print(f"  Elements: {len(chunk.element_ids)}")
    print(f"  Tokens: {chunk.token_count}")
    print(f"  Section: {chunk.parent_section[:50] if chunk.parent_section else 'N/A'}")
    print()
```

### Custom Configuration
```python
# Fine-tune chunking behavior
config = {
    "chunk_size": 1500,
    "chunk_overlap": 300,
    "preserve_structure": True,
}

parser = PageIndexParser(config=config)
doc = await parser.parse(Path("technical_report.pdf"))
chunks = parser.chunk_document(doc)
```

## Advantages Over Simple Chunking

### Traditional Fixed-Size Chunking
```
"...reservoir pressure at 3000 psi. The production rate was measured|
at 500 barrels per day. Temperature readings showed..."
```
❌ Splits mid-sentence
❌ Breaks semantic units
❌ Loses context

### PageIndex Semantic Chunking
```
"...reservoir pressure at 3000 psi. The production rate was measured
at 500 barrels per day."

[New Chunk]
"The production rate was measured at 500 barrels per day. Temperature
readings showed consistent performance across the well..."
```
✅ Respects sentence boundaries
✅ Maintains semantic coherence
✅ Provides contextual overlap

## Performance Characteristics

### Parsing Speed
- **Fallback mode**: ~1-3 seconds per page (PDF complexity dependent)
- **Native mode**: Expected to be faster with optimized parsing

### Memory Usage
- Scales linearly with document size
- Elements are processed incrementally
- Chunks are created on-demand

### Chunk Quality
- Higher semantic coherence than fixed-size chunking
- Better preservation of tables and structured content
- Improved context for RAG retrieval

## Installation

### With PageIndex (when available)
```bash
pip install pageindex
```

### Fallback Mode (always works)
```bash
pip install pymupdf
```

Both dependencies are included in the project's `pyproject.toml`.

## Troubleshooting

### "PageIndex fallback requires PyMuPDF"
Install PyMuPDF:
```bash
pip install pymupdf
```

### Parser uses fallback mode unexpectedly
Check if PageIndex is properly installed:
```python
import importlib.util
spec = importlib.util.find_spec("pageindex")
print("PageIndex available:", spec is not None)
```

### Poor element type detection
The fallback mode uses heuristics. For best results:
- Use well-formatted PDFs
- Ensure text is extractable (not scanned images)
- Consider using native PageIndex for complex documents

## Future Enhancements

1. **Machine Learning Element Detection**: Replace heuristics with ML models
2. **Advanced Table Extraction**: Better structure preservation for complex tables
3. **Figure Caption Association**: Link figures with their captions automatically
4. **Cross-Reference Resolution**: Maintain references between sections
5. **Adaptive Chunk Sizing**: Dynamically adjust chunk size based on content type

## Comparison with Other Parsers

| Feature | PageIndex | Traditional | LlamaParse | Docling |
|---------|-----------|-------------|------------|---------|
| Semantic Chunking | ✅ Yes | ❌ No | ⚠️ Basic | ⚠️ Basic |
| Structure Preservation | ✅ Strong | ❌ Weak | ✅ Strong | ✅ Strong |
| Element Detection | ✅ Advanced | ⚠️ Basic | ✅ Advanced | ✅ Advanced |
| Intelligent Overlap | ✅ Yes | ❌ No | ❌ No | ❌ No |
| Fallback Mode | ✅ Yes | N/A | ❌ No | ❌ No |

## References

- PageIndex documentation (when available)
- PyMuPDF documentation: https://pymupdf.readthedocs.io/
- Semantic chunking research papers
- RAG best practices for document processing
