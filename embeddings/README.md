# Embeddings Module

Unified embedding generation for the petroleum RAG benchmark system using OpenAI's embedding models.

## Overview

The `UnifiedEmbedder` class provides a robust, production-ready interface for generating text embeddings with:

- Single and batch text embedding
- Automatic rate limiting and retry logic
- Batch splitting for large inputs
- Comprehensive error handling
- Dimension validation

## Quick Start

```python
from embeddings import UnifiedEmbedder

# Initialize embedder (uses settings from config.py)
embedder = UnifiedEmbedder()

# Single text embedding
embedding = await embedder.embed_text("Petroleum reservoir analysis")
# Returns: list[float] with 1536 dimensions

# Batch embedding
texts = ["Oil well drilling", "Gas production", "Reservoir simulation"]
embeddings = await embedder.embed_batch(texts)
# Returns: list[list[float]]

# Cleanup
await embedder.close()
```

## Configuration

Settings are pulled from `config.py` via environment variables:

```bash
# .env
OPENAI_API_KEY=your-api-key-here
```

Default settings:
- Model: `text-embedding-3-small`
- Dimensions: 1536
- Batch size: 100

### Custom Configuration

```python
embedder = UnifiedEmbedder(
    api_key="custom-key",
    model="text-embedding-3-large",
    dimensions=3072,
    batch_size=50,
)
```

## Features

### Automatic Batch Splitting

Large batches are automatically split to respect OpenAI's limits:

```python
# Automatically splits into multiple API calls
large_texts = [f"Text {i}" for i in range(500)]
embeddings = await embedder.embed_batch(large_texts)  # 5 API calls (100 per batch)
```

### Rate Limiting & Retry

Built-in retry logic with exponential backoff:

- Max 5 retry attempts
- Exponential backoff: 2s → 4s → 8s → 16s → 32s (max 60s)
- Handles: `RateLimitError`, `APITimeoutError`, `APIError`

```python
# Automatically retries on rate limits
embeddings = await embedder.embed_batch(texts)
```

### Error Handling

Comprehensive validation and error messages:

```python
# Validates empty texts
try:
    await embedder.embed_text("")
except ValueError as e:
    print(e)  # "Cannot embed empty text"

# Validates batch content
try:
    await embedder.embed_batch(["text1", "", "text3"])
except ValueError as e:
    print(e)  # "Found empty texts at indices: [1]"
```

### Connection Validation

Test API connectivity before processing:

```python
if await embedder.validate_connection():
    print("Ready to generate embeddings")
else:
    print("Connection issue - check API key")
```

## API Reference

### `UnifiedEmbedder`

#### `__init__(api_key=None, model=None, dimensions=None, batch_size=None)`

Initialize the embedder.

**Parameters:**
- `api_key` (str, optional): OpenAI API key. Defaults to `settings.openai_api_key`.
- `model` (str, optional): Embedding model name. Defaults to `settings.embedding_model`.
- `dimensions` (int, optional): Expected embedding dimensions. Defaults to `settings.embedding_dimension`.
- `batch_size` (int, optional): Batch size for processing. Defaults to `settings.embedding_batch_size`.

**Raises:**
- `ValueError`: If API key is not provided.

#### `async embed_text(text: str) -> list[float]`

Generate embedding for a single text.

**Parameters:**
- `text` (str): Text to embed.

**Returns:**
- `list[float]`: Embedding vector.

**Raises:**
- `ValueError`: If text is empty.
- `RuntimeError`: If embedding generation fails.

#### `async embed_batch(texts: list[str]) -> list[list[float]]`

Generate embeddings for a batch of texts.

**Parameters:**
- `texts` (list[str]): List of texts to embed.

**Returns:**
- `list[list[float]]`: List of embedding vectors.

**Raises:**
- `ValueError`: If texts list is empty or contains empty strings.
- `RuntimeError`: If embedding generation fails.

#### `async validate_connection() -> bool`

Validate API connection.

**Returns:**
- `bool`: True if connection is valid, False otherwise.

#### `async close() -> None`

Close the OpenAI client and cleanup resources.

#### `get_stats() -> dict[str, Any]`

Get embedder statistics.

**Returns:**
- `dict`: Configuration dictionary with model, dimensions, batch_size, etc.

## Testing

### Unit Tests

Run the comprehensive test suite:

```bash
python3 -m pytest tests/test_embedder.py -v
```

20 tests covering:
- Initialization and configuration
- Single and batch embedding
- Error handling and validation
- Rate limit handling
- Retry logic
- Utility functions

### Integration Tests

Test with real API calls:

```bash
# Requires OPENAI_API_KEY in .env
python scripts/test_embeddings_integration.py
```

### Verification

Quick module verification (no API key required):

```bash
python scripts/verify_embeddings.py
```

## Performance

Typical performance characteristics:

- **Single embedding**: ~100-200ms per text
- **Batch of 100**: ~2-3 seconds
- **Large batch (1000)**: ~20-30 seconds (10 API calls + delays)
- **Memory**: Minimal - embeddings are lists of floats

## Integration with Storage Backends

The embedder integrates seamlessly with storage backends:

```python
from embeddings import UnifiedEmbedder
from storage.chroma_store import ChromaStore

# Initialize
embedder = UnifiedEmbedder()
storage = ChromaStore(config)

# Generate embeddings
embeddings = await embedder.embed_batch([chunk.content for chunk in chunks])

# Store in backend
await storage.store_chunks(chunks, embeddings)
```

## Error Messages

Clear, actionable error messages:

- `"OpenAI API key must be provided via api_key parameter or OPENAI_API_KEY environment variable"`
- `"Cannot embed empty text"`
- `"Found empty texts at indices: [0, 5, 10]"`
- `"Embedding generation failed: Rate limited"`

## Logging

Structured logging with timestamps:

```python
# Configure logging level
from utils.logging import setup_logging
setup_logging(log_level="DEBUG")

# Embedder logs at appropriate levels
embedder = UnifiedEmbedder()
# INFO: UnifiedEmbedder initialized with model=text-embedding-3-small...
# DEBUG: Generated 5 embeddings
# WARNING: Rate limit hit, will retry: ...
```

## Dependencies

- `openai>=1.12.0` - Official OpenAI Python client
- `tenacity>=8.2.0` - Retry logic with exponential backoff
- `httpx` - HTTP client (OpenAI dependency)

## Files

```
embeddings/
├── __init__.py         # Module exports
├── embedder.py         # UnifiedEmbedder class (238 lines)
└── README.md          # This file
```

## Best Practices

1. **Reuse embedder instances**: Create once, use many times
2. **Use batch processing**: More efficient than individual calls
3. **Handle errors gracefully**: Catch `ValueError` and `RuntimeError`
4. **Close when done**: Call `await embedder.close()` for cleanup
5. **Monitor rate limits**: Built-in handling, but be aware of quotas

## Example: Full Pipeline

```python
from embeddings import UnifiedEmbedder
from models import DocumentChunk

async def process_chunks(chunks: list[DocumentChunk]) -> None:
    """Process document chunks through embedding pipeline."""

    # Initialize embedder
    embedder = UnifiedEmbedder()

    try:
        # Validate connection
        if not await embedder.validate_connection():
            raise RuntimeError("Cannot connect to OpenAI API")

        # Extract text content
        texts = [chunk.content for chunk in chunks]

        # Generate embeddings
        embeddings = await embedder.embed_batch(texts)

        # Store or process embeddings
        for chunk, embedding in zip(chunks, embeddings):
            print(f"Chunk {chunk.chunk_id}: {len(embedding)} dimensions")
            # Store in vector database...

    finally:
        # Cleanup
        await embedder.close()
```

## Support

For issues or questions:
1. Check test suite: `tests/test_embedder.py`
2. Run verification: `python scripts/verify_embeddings.py`
3. Review documentation: `AGENT_8_COMPLETE.md`

## License

Part of the petroleum RAG benchmark system.
