# Vertex Document AI Parser

## Overview

The Vertex Document AI parser leverages Google Cloud's enterprise-grade Document AI service to provide high-quality OCR, layout analysis, and structured content extraction from petroleum engineering documents.

## Key Features

- **Enterprise-Grade OCR**: Handles both digital and scanned PDFs with high accuracy
- **Advanced Layout Detection**: Automatically identifies headers, paragraphs, tables, and document structure
- **Table Extraction**: Extracts tables with cell-level structure and converts to markdown format
- **Bounding Box Coordinates**: Provides spatial information for all detected elements
- **Multi-Page Processing**: Efficiently processes documents of any length
- **Error Handling**: Robust error handling with detailed error messages
- **Layout-Aware Chunking**: Chunks documents based on natural semantic boundaries detected by Document AI

## Configuration

### Prerequisites

1. **Google Cloud Project**: Active GCP project with billing enabled
2. **Document AI API**: Enable the Document AI API in your project
3. **Service Account**: Create a service account with Document AI User role
4. **Processor**: Create a Document AI processor (use "Form Parser" or "OCR Processor")

### Environment Variables

Required configuration in `.env` file:

```bash
# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
VERTEX_DOCAI_PROCESSOR_ID=your-processor-id
VERTEX_DOCAI_LOCATION=us  # Options: us, eu, asia-northeast1

# Parser Settings (optional)
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Setup Instructions

#### 1. Install Dependencies

```bash
pip install google-cloud-documentai>=2.20.0
```

#### 2. Set Up Google Cloud

```bash
# Install gcloud CLI
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID

# Enable Document AI API
gcloud services enable documentai.googleapis.com

# Create service account
gcloud iam service-accounts create docai-parser \
    --display-name="Document AI Parser"

# Grant permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:docai-parser@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/documentai.apiUser"

# Create and download key
gcloud iam service-accounts keys create ~/docai-key.json \
    --iam-account=docai-parser@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

#### 3. Create Document AI Processor

Via Console:
1. Go to [Document AI Console](https://console.cloud.google.com/ai/document-ai)
2. Click "Create Processor"
3. Select "Form Parser" or "OCR Processor"
4. Choose location (us, eu, or asia-northeast1)
5. Copy the Processor ID

Via CLI:
```bash
# List available processor types
gcloud documentai processor-types list --location=us

# Create processor
gcloud documentai processors create \
    --location=us \
    --type=FORM_PARSER_PROCESSOR \
    --display-name="Petroleum Document Parser"
```

#### 4. Configure Environment

```bash
# Export environment variables
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/docai-key.json
export VERTEX_DOCAI_PROCESSOR_ID=your-processor-id
export VERTEX_DOCAI_LOCATION=us
```

Or add to `.env` file:
```
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/docai-key.json
VERTEX_DOCAI_PROCESSOR_ID=your-processor-id
VERTEX_DOCAI_LOCATION=us
```

## Usage

### Basic Usage

```python
import asyncio
from pathlib import Path
from parsers.vertex_parser import VertexDocAIParser

async def main():
    # Initialize parser
    parser = VertexDocAIParser()

    # Parse document
    parsed_doc = await parser.parse(Path("document.pdf"))

    # Check for errors
    if parsed_doc.error:
        print(f"Error: {parsed_doc.error}")
        return

    # Access parsed content
    print(f"Extracted {len(parsed_doc.elements)} elements")
    print(f"Pages: {parsed_doc.total_pages}")

    # Chunk document
    chunks = parser.chunk_document(parsed_doc)
    print(f"Created {len(chunks)} chunks")

asyncio.run(main())
```

### Custom Configuration

```python
# Custom chunk sizes
parser = VertexDocAIParser(config={
    "chunk_size": 1500,
    "chunk_overlap": 300,
    "min_chunk_size": 200,
    "max_chunk_size": 3000,
})
```

### Accessing Elements

```python
parsed_doc = await parser.parse(file_path)

# Filter by element type
from models import ElementType

tables = [e for e in parsed_doc.elements if e.element_type == ElementType.TABLE]
paragraphs = [e for e in parsed_doc.elements if e.element_type == ElementType.TEXT]

# Access element properties
for element in parsed_doc.elements[:5]:
    print(f"Type: {element.element_type}")
    print(f"Page: {element.page_number}")
    print(f"Content: {element.content[:100]}...")
    print(f"BBox: {element.bbox}")
    print(f"Formatted: {element.formatted_content}")
```

### Working with Tables

```python
# Extract all tables
tables = [e for e in parsed_doc.elements if e.element_type == ElementType.TABLE]

for table in tables:
    # Plain text content
    print(table.content)

    # Markdown formatted table
    print(table.formatted_content)

    # Metadata
    print(f"Rows: {table.metadata.get('rows')}")
    print(f"Columns: {table.metadata.get('columns')}")
```

### Error Handling

```python
try:
    parser = VertexDocAIParser()
    parsed_doc = await parser.parse(file_path)

    if parsed_doc.error:
        print(f"Parsing error: {parsed_doc.error}")
    else:
        print(f"Success: {len(parsed_doc.elements)} elements")

except ValueError as e:
    print(f"Configuration error: {e}")
except FileNotFoundError as e:
    print(f"File error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Architecture

### Parsing Pipeline

1. **File Validation**: Checks file existence and supported format
2. **API Request**: Sends document to Vertex Document AI
3. **Response Processing**: Extracts elements from Document AI response
4. **Element Mapping**: Maps Document AI structures to ParsedElement objects
5. **Metadata Extraction**: Captures document-level metadata

### Element Extraction

The parser extracts the following element types:

- **Paragraphs**: Detected text blocks with layout information
- **Tables**: Structured tables with cell-level content
- **Blocks**: Generic text blocks from layout analysis
- **Bounding Boxes**: Spatial coordinates for all elements

### Chunking Strategy

The chunking algorithm:

1. Iterates through elements in document order
2. Accumulates elements until chunk size limit
3. Respects semantic boundaries (paragraphs, sections)
4. Implements configurable overlap for context preservation
5. Maintains page number ranges and element references

## Performance Characteristics

### Processing Speed

- **Digital PDFs**: ~1-3 seconds per page
- **Scanned PDFs**: ~2-5 seconds per page (depends on image quality)
- **Network Latency**: Adds ~200-500ms per request

### API Quotas

Default quotas (per project):
- **Requests per minute**: 300
- **Requests per day**: 1,000,000
- **Pages per request**: 15 (synchronous), 500 (async)

For production use, consider:
- Request quota increases via GCP console
- Batch processing for large document sets
- Async processing API for large documents

### Cost

Pricing (as of 2024):
- **OCR Processor**: $1.50 per 1000 pages
- **Form Parser**: $2.50 per 1000 pages
- **Specialized Parsers**: $5-10 per 1000 pages

First 1000 pages per month are free.

## Comparison with Other Parsers

| Feature | Vertex DocAI | LlamaParse | Docling | PageIndex |
|---------|--------------|------------|---------|-----------|
| **OCR Quality** | Excellent | Good | Good | Very Good |
| **Table Extraction** | Excellent | Very Good | Good | Good |
| **Layout Detection** | Excellent | Good | Very Good | Good |
| **Scanned PDF Support** | Excellent | Good | Limited | Good |
| **Speed** | Fast | Very Fast | Fast | Fast |
| **Cost** | Pay-per-use | Subscription | Open Source | TBD |
| **Enterprise Ready** | Yes | Yes | No | TBD |
| **Bounding Boxes** | Yes | Limited | Yes | Yes |

## Best Practices

### 1. Authentication

```python
# Use service account for production
# Set GOOGLE_APPLICATION_CREDENTIALS to key file path

# For development, use application default credentials
# gcloud auth application-default login
```

### 2. Error Handling

```python
# Always check for parsing errors
if parsed_doc.error:
    logger.error(f"Parse failed: {parsed_doc.error}")
    # Implement fallback or retry logic
```

### 3. Batch Processing

```python
# Process multiple documents efficiently
async def process_documents(file_paths):
    parser = VertexDocAIParser()

    tasks = [parser.parse(fp) for fp in file_paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle results
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Failed: {result}")
        else:
            print(f"Success: {result.document_id}")
```

### 4. Quota Management

```python
# Implement rate limiting
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def parse_with_retry(parser, file_path):
    return await parser.parse(file_path)
```

### 5. Cost Optimization

```python
# Cache parsed results to avoid re-parsing
import json

# Save parsed document
with open(f"{doc_id}.json", "w") as f:
    json.dump(parsed_doc.to_dict(), f)

# Reuse cached results when possible
```

## Troubleshooting

### Common Issues

#### 1. Authentication Error

```
Error: GOOGLE_CLOUD_PROJECT environment variable is required
```

**Solution**: Set environment variables in `.env` file:
```bash
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
```

#### 2. API Not Enabled

```
Error: Document AI API has not been used in project
```

**Solution**: Enable the API:
```bash
gcloud services enable documentai.googleapis.com
```

#### 3. Invalid Processor ID

```
Error: Processor not found
```

**Solution**: Verify processor exists and ID is correct:
```bash
gcloud documentai processors list --location=us
```

#### 4. Quota Exceeded

```
Error: Resource has been exhausted
```

**Solution**:
- Wait for quota to reset
- Request quota increase in GCP Console
- Implement rate limiting

#### 5. File Size Limit

```
Error: Request payload size exceeds the limit
```

**Solution**:
- Use async processing API for large files
- Split large documents into smaller files
- Compress images before processing

## Testing

### Unit Tests

Run the test suite:
```bash
pytest tests/test_vertex_parser.py -v
```

### Integration Tests

Create a test document and parse:
```bash
python examples/vertex_parser_example.py test_document.pdf
```

### Validation

```python
# Verify parser is configured correctly
from parsers.vertex_parser import VertexDocAIParser

try:
    parser = VertexDocAIParser()
    print("✓ Parser initialized successfully")
    print(f"  Project: {parser.project_id}")
    print(f"  Processor: {parser.processor_id}")
except ValueError as e:
    print(f"✗ Configuration error: {e}")
```

## References

- [Google Cloud Document AI Documentation](https://cloud.google.com/document-ai/docs)
- [Document AI Python Client Library](https://cloud.google.com/python/docs/reference/documentai/latest)
- [Document AI Pricing](https://cloud.google.com/document-ai/pricing)
- [Document AI Quotas](https://cloud.google.com/document-ai/quotas)

## Support

For issues or questions:
1. Check the [troubleshooting section](#troubleshooting)
2. Review [GCP Document AI documentation](https://cloud.google.com/document-ai/docs)
3. Open an issue in the project repository
