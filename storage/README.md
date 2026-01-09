# Storage Backends

This directory contains storage backend implementations for the petroleum RAG benchmark system.

## Available Backends

| Backend | Type | Key Feature | Best For |
|---------|------|-------------|----------|
| **ChromaDB** | Vector | Pure similarity | Semantic search |
| **Weaviate** | Hybrid | Vector + BM25 | Technical documents |
| **FalkorDB** | Graph | Vector + relationships | Complex queries |

## Weaviate Implementation

### Quick Start

```python
from storage.weaviate_store import WeaviateStore
import asyncio

async def main():
    # Initialize
    store = WeaviateStore()
    await store.initialize()

    # Store chunks with embeddings
    await store.store_chunks(chunks, embeddings)

    # Hybrid search
    results = await store.retrieve(
        query="drilling mud weight specifications",
        query_embedding=query_vector,
        top_k=5
    )

asyncio.run(main())
```

### Key Features

1. **Hybrid Search**: Combines vector similarity (70%) with BM25 keyword matching (30%)
2. **Tunable Alpha**: Adjust semantic vs keyword balance
3. **Full-text Indexing**: BM25 on content for exact term matching
4. **Metadata Filtering**: Filter by document_id and other fields
5. **Batch Operations**: Efficient storage with dynamic batching

### Configuration

```python
# Default configuration
store = WeaviateStore()

# Custom configuration
store = WeaviateStore(config={
    "host": "localhost",
    "port": 8080,
    "grpc_port": 50051,
    "class_name": "PetroleumDocument",
    "alpha": 0.7,  # 70% vector, 30% keyword
    "min_score": 0.5
})
```

### Alpha Parameter

Controls the balance between vector and keyword search:

- **1.0**: 100% vector (pure semantic)
- **0.7**: 70% vector, 30% keyword (recommended)
- **0.5**: Equal balance
- **0.3**: 30% vector, 70% keyword
- **0.0**: 100% keyword (pure BM25)

**Recommendation**: Start with 0.7 for petroleum engineering documents.

### Running Weaviate

```bash
# Docker
docker run -p 8080:8080 -p 50051:50051 \
  cr.weaviate.io/semitechnologies/weaviate:latest

# Verify
curl http://localhost:8080/v1/meta
```

### Schema

Properties stored in Weaviate:

- `chunk_id` (TEXT): Unique identifier
- `document_id` (TEXT): Parent document
- `content` (TEXT): Chunk text (BM25 indexed)
- `element_ids` (TEXT_ARRAY): Element references
- `chunk_index` (INT): Sequential position
- `start_page`, `end_page` (INT): Page range
- `token_count` (INT): Approximate tokens
- `parent_section` (TEXT): Section reference
- `metadata` (TEXT): JSON-encoded metadata

## Base Storage Interface

All storage backends implement the `BaseStorage` abstract class:

```python
class BaseStorage(ABC):
    @abstractmethod
    async def initialize(self) -> None

    @abstractmethod
    async def store_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]]
    ) -> None

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None
    ) -> list[RetrievalResult]

    @abstractmethod
    async def clear(self) -> None

    async def health_check(self) -> bool
```

## Testing

### Unit Tests
```bash
pytest tests/test_weaviate_store.py -v
```

### Integration Test
```bash
python examples/test_weaviate_hybrid_search.py
```

## Documentation

- **Implementation Guide**: `/docs/weaviate_implementation.md`
- **Completion Report**: `/AGENT_6_COMPLETE.md`
- **Base Interface**: `storage/base.py`

## Files

```
storage/
├── __init__.py              # Exports all storage backends
├── base.py                  # BaseStorage abstract class
├── weaviate_store.py        # Weaviate implementation ✅
├── chroma_store.py          # ChromaDB implementation
├── falkordb_store.py        # FalkorDB implementation
└── README.md                # This file
```

## Usage in Benchmark

```python
from storage import WeaviateStore
from embeddings.embedder import Embedder

# Initialize storage
store = WeaviateStore()
await store.initialize()

# Generate embeddings
embedder = Embedder()
embeddings = await embedder.embed_texts([chunk.content for chunk in chunks])

# Store
await store.store_chunks(chunks, embeddings)

# Retrieve
query_embedding = await embedder.embed_text(query)
results = await store.retrieve(
    query=query,
    query_embedding=query_embedding,
    top_k=5
)
```

## Performance

### Strengths
- Fast hybrid search (HNSW + BM25 indices)
- Efficient gRPC for vector operations
- Dynamic batching for storage
- Metadata filters applied efficiently

### Scalability
- Horizontal scaling supported
- Sharding for large datasets
- Replication for availability
- Kubernetes-ready

## Comparison

### When to Use Weaviate

✅ Technical documents with specific terminology
✅ Need both semantic and exact term matching
✅ Equipment names and specifications
✅ Numerical values in context
✅ Tunable search strategy needed

### When to Use Others

- **ChromaDB**: Pure semantic search, no technical terms
- **FalkorDB**: Relationship queries, graph traversal needed

## References

- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [Hybrid Search Guide](https://weaviate.io/developers/weaviate/search/hybrid)
- [Python Client](https://weaviate.io/developers/weaviate/client-libraries/python)

## Support

For issues or questions:
1. Check `/docs/weaviate_implementation.md`
2. Run integration test: `python examples/test_weaviate_hybrid_search.py`
3. Verify Weaviate is running: `curl http://localhost:8080/v1/meta`
