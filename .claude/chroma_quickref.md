# ChromaStore Quick Reference

## Files
- **Implementation**: `/Users/gregorydickson/petroleum-rag/storage/chroma_store.py`
- **Tests**: `/Users/gregorydickson/petroleum-rag/tests/test_chroma_store.py`
- **Verification**: `/Users/gregorydickson/petroleum-rag/verify_chroma_implementation.py`

## Quick Start

```python
from config import settings
from storage.chroma_store import ChromaStore

# Initialize
config = settings.get_storage_config("chroma")
store = ChromaStore(config)
await store.initialize()

# Store
await store.store_chunks(chunks, embeddings)

# Retrieve
results = await store.retrieve(
    query="test",
    query_embedding=embedding,
    top_k=5
)
```

## Commands

```bash
# Start ChromaDB
docker-compose up -d chroma

# Run tests
pytest tests/test_chroma_store.py -v

# Verify implementation
python3 verify_chroma_implementation.py
```

## Key Methods

| Method | Purpose |
|--------|---------|
| `initialize()` | Connect to ChromaDB |
| `store_chunks()` | Batch store with embeddings |
| `retrieve()` | Vector similarity search |
| `clear()` | Delete all data |
| `health_check()` | Verify connection |

## Configuration

```python
{
    "host": "localhost",
    "port": 8000,
    "collection_name": "petroleum_docs",
    "top_k": 5,
    "min_score": 0.0
}
```

## Score Conversion

ChromaDB distance → Similarity:
```python
similarity = 1.0 - (distance / 2.0)
```

## Status: ✅ COMPLETE
