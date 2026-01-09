# Weaviate Storage Implementation

## Overview

The `WeaviateStore` class implements the `BaseStorage` interface for the petroleum RAG benchmark, providing **hybrid search** capabilities that combine vector similarity with BM25 keyword matching.

**Key Feature:** Unlike pure vector stores (like ChromaDB), Weaviate's hybrid search allows balancing between semantic similarity and exact keyword matching, which is particularly valuable for petroleum engineering documents containing technical terminology, equipment names, and numerical specifications.

## Implementation Details

### File Location
`/Users/gregorydickson/petroleum-rag/storage/weaviate_store.py`

### Dependencies
- `weaviate-client>=4.4.0` (Weaviate Python client v4)
- Weaviate server running on `localhost:8080` (HTTP) and `localhost:50051` (gRPC)

### Class: `WeaviateStore`

Inherits from `BaseStorage` and implements all required abstract methods.

#### Key Attributes

- **`client`**: Weaviate client instance
- **`class_name`**: Name of the Weaviate collection (default: "PetroleumDocument")
- **`alpha`**: Hybrid search weight parameter (default: 0.7)
  - `1.0` = 100% vector similarity, 0% keyword
  - `0.0` = 0% vector similarity, 100% keyword (BM25)
  - `0.7` = 70% vector, 30% keyword (recommended default)

## Schema Design

The Weaviate schema includes the following properties:

| Property | Type | Indexed | Description |
|----------|------|---------|-------------|
| `chunk_id` | TEXT | No | Unique chunk identifier |
| `document_id` | TEXT | No | Parent document ID |
| `content` | TEXT | **Yes** | Chunk text (BM25 indexed) |
| `element_ids` | TEXT_ARRAY | No | List of element IDs |
| `chunk_index` | INT | No | Sequential index |
| `start_page` | INT | No | Starting page number |
| `end_page` | INT | No | Ending page number |
| `token_count` | INT | No | Approximate tokens |
| `parent_section` | TEXT | No | Parent section ID |
| `metadata` | TEXT | No | JSON-encoded metadata |

**Key Design Decision:** Only the `content` field has `skip_vectorization=False`, enabling BM25 full-text indexing while preventing ID fields from polluting keyword search results.

## Core Methods

### `__init__(config: dict | None = None)`

Initializes the store with optional configuration.

**Config Options:**
- `host`: Weaviate host (default: from settings)
- `port`: HTTP port (default: 8080)
- `grpc_port`: gRPC port (default: 50051)
- `class_name`: Collection name (default: "PetroleumDocument")
- `alpha`: Hybrid search weight (default: 0.7)
- `top_k`: Default retrieval limit (default: 5)
- `min_score`: Minimum relevance threshold (default: 0.5)

### `async initialize()`

Connects to Weaviate and creates the schema if it doesn't exist.

**Behavior:**
- Idempotent: Safe to call multiple times
- Checks if collection exists before creating
- Uses `Configure.Vectorizer.none()` (we provide embeddings)
- Configures full-text indexing on `content` field

**Raises:**
- `ConnectionError`: Cannot connect to Weaviate
- `RuntimeError`: Schema creation fails

### `async store_chunks(chunks, embeddings)`

Batch inserts document chunks with embeddings.

**Process:**
1. Validates chunks/embeddings match
2. Serializes metadata to JSON
3. Uses Weaviate's dynamic batching for optimal performance
4. Stores both properties and vector embeddings

**Implementation Note:** Uses `collection.batch.dynamic()` which automatically handles batch sizing and retries.

### `async retrieve(query, query_embedding, top_k=5, filters=None)`

Performs hybrid search combining vector and keyword matching.

**Parameters:**
- `query`: Text query for BM25 keyword search
- `query_embedding`: Vector for similarity search
- `top_k`: Number of results to return
- `filters`: Optional metadata filters (e.g., `{"document_id": "doc123"}`)

**Returns:** List of `RetrievalResult` objects with:
- `chunk_id`, `document_id`, `content`
- `score`: Hybrid relevance score (0.0-1.0)
- `rank`: Position in results (1-based)
- `retrieval_method`: Always "hybrid"

**Hybrid Search Details:**
```python
response = collection.query.hybrid(
    query=query,              # Text for BM25
    vector=query_embedding,   # Vector for similarity
    alpha=self.alpha,         # 0.7 = 70% vector, 30% BM25
    limit=top_k,
    filters=where_filter,
    return_metadata=MetadataQuery(score=True, explain_score=True)
)
```

**Score Filtering:** Results below `min_score` threshold are automatically filtered out.

### `async clear()`

Deletes all objects from the collection while preserving the schema.

Uses `delete_many()` with a wildcard filter to remove all chunks.

### `async health_check()`

Checks if Weaviate is ready and responsive.

Returns `True` if `client.is_ready()` succeeds, `False` otherwise.

## Hybrid Search Use Cases

The hybrid search capability is particularly valuable for petroleum engineering documents:

### 1. Technical Terminology
**Query:** "BOP testing procedures"
- **Keyword (BM25):** Finds exact mentions of "BOP"
- **Vector:** Understands "blowout preventer" is semantically similar
- **Result:** Retrieves both explicit mentions and related concepts

### 2. Numerical Specifications
**Query:** "mud weight 10 ppg"
- **Keyword:** Finds exact "10 ppg" values
- **Vector:** Finds discussions about drilling fluid density
- **Result:** Precise numerical matches + conceptual context

### 3. Equipment Names
**Query:** "Christmas tree valve configuration"
- **Keyword:** Exact term "Christmas tree" (petroleum wellhead equipment)
- **Vector:** Related wellhead and production equipment
- **Result:** Specific equipment + related systems

### 4. Semantic + Keyword Combination
**Query:** "How to prevent formation damage during drilling?"
- **Keyword:** Key terms like "formation damage", "drilling"
- **Vector:** Understands the question semantics
- **Result:** Balanced retrieval of relevant procedural and technical content

## Alpha Parameter Tuning

The `alpha` parameter controls the vector/keyword balance:

| Alpha | Vector % | Keyword % | Best For |
|-------|----------|-----------|----------|
| 1.0 | 100% | 0% | Pure semantic search, conceptual queries |
| 0.8 | 80% | 20% | Mostly semantic with some keyword boost |
| **0.7** | **70%** | **30%** | **Balanced (default recommendation)** |
| 0.5 | 50% | 50% | Equal weighting |
| 0.3 | 30% | 70% | Keyword-dominant with semantic fallback |
| 0.0 | 0% | 100% | Pure BM25 keyword search |

**Recommended Starting Point:** 0.7 (70% vector, 30% keyword)

**Tuning Approach:**
1. Start with default 0.7
2. If missing exact technical terms, decrease alpha (more keyword weight)
3. If getting too literal matches, increase alpha (more semantic weight)
4. Benchmark with evaluation queries to find optimal value

## Usage Example

```python
import asyncio
from storage.weaviate_store import WeaviateStore
from models import DocumentChunk

async def example():
    # Initialize store
    store = WeaviateStore(config={"alpha": 0.7})
    await store.initialize()

    # Store chunks
    chunks = [
        DocumentChunk(
            chunk_id="chunk_1",
            document_id="drilling_guide",
            content="Drilling mud density must be maintained at 10.5 ppg...",
            metadata={"section": "drilling_operations"},
            chunk_index=0
        )
    ]
    embeddings = [[0.1, 0.2, 0.3, ...]]  # From OpenAI embeddings
    await store.store_chunks(chunks, embeddings)

    # Hybrid search
    query_embedding = [0.15, 0.22, 0.31, ...]  # Embed query
    results = await store.retrieve(
        query="What is the recommended mud weight for drilling?",
        query_embedding=query_embedding,
        top_k=5
    )

    for result in results:
        print(f"[{result.score:.3f}] {result.content[:100]}...")

asyncio.run(example())
```

## Testing

### Unit Tests
Location: `/Users/gregorydickson/petroleum-rag/tests/test_weaviate_store.py`

Coverage includes:
- Initialization and schema creation
- Batch storage with validation
- Hybrid search with various parameters
- Metadata filtering
- Score threshold filtering
- Health checks
- Error handling

Run tests:
```bash
pytest tests/test_weaviate_store.py -v
```

### Integration Test
Location: `/Users/gregorydickson/petroleum-rag/examples/test_weaviate_hybrid_search.py`

Demonstrates:
- Real Weaviate connection
- Storing petroleum engineering content
- Semantic vs keyword search comparison
- Alpha parameter effects
- Filtered search

Prerequisites:
```bash
# Start Weaviate with Docker
docker run -p 8080:8080 -p 50051:50051 \
  cr.weaviate.io/semitechnologies/weaviate:latest
```

Run test:
```bash
python examples/test_weaviate_hybrid_search.py
```

## Differences from Other Storage Backends

| Feature | ChromaDB | **Weaviate** | FalkorDB |
|---------|----------|--------------|----------|
| Search Type | Pure vector | **Hybrid (vector + BM25)** | Graph + vector |
| Best For | Semantic similarity | **Technical documents** | Relationship queries |
| Keyword Match | No | **Yes (BM25)** | Limited |
| Exact Term Search | Poor | **Excellent** | N/A |
| Semantic Search | Excellent | **Excellent** | Good |
| Tunable Balance | No | **Yes (alpha)** | No |

**Why Weaviate for Petroleum Documents:**
- Technical terminology requires exact matches (BM25)
- Equipment names and specifications need keyword precision
- Semantic understanding for conceptual queries
- Tunable balance adapts to different query types

## Configuration

### Environment Variables (.env)
```bash
# Weaviate Connection
WEAVIATE_HOST=localhost
WEAVIATE_PORT=8080
WEAVIATE_GRPC_PORT=50051
WEAVIATE_CLASS_NAME=PetroleumDocument

# Retrieval Settings
RETRIEVAL_TOP_K=5
RETRIEVAL_MIN_SCORE=0.5
```

### Docker Compose Setup
```yaml
services:
  weaviate:
    image: cr.weaviate.io/semitechnologies/weaviate:latest
    ports:
      - "8080:8080"    # HTTP API
      - "50051:50051"  # gRPC
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'
      ENABLE_MODULES: ''
      CLUSTER_HOSTNAME: 'node1'
    volumes:
      - weaviate_data:/var/lib/weaviate

volumes:
  weaviate_data:
```

## Performance Considerations

### Batch Insertion
- Uses Weaviate's dynamic batching
- Automatically handles optimal batch sizes
- Built-in retry logic for failures
- Efficient for large document sets

### Query Performance
- Hybrid search is fast due to HNSW + BM25 indices
- gRPC used for vector operations (faster than HTTP)
- Metadata filters applied efficiently
- Top-k limiting reduces data transfer

### Scaling
- Weaviate supports horizontal scaling
- Sharding available for large datasets
- Replication for high availability
- Compatible with Kubernetes deployment

## Troubleshooting

### "Failed to connect to Weaviate"
**Solution:** Ensure Weaviate is running:
```bash
docker ps | grep weaviate
curl http://localhost:8080/v1/meta
```

### "Weaviate storage not initialized"
**Solution:** Call `await store.initialize()` before operations

### Low retrieval scores
**Solution:**
1. Check if embeddings are normalized
2. Adjust alpha parameter
3. Verify content is properly indexed

### Missing expected results
**Solution:**
1. Decrease alpha for more keyword weight
2. Check if content contains exact terms
3. Verify chunks were stored successfully

## Future Enhancements

Possible improvements:
1. **Multi-tenancy**: Separate collections per project
2. **Advanced filtering**: Support complex filter queries
3. **Geo-spatial**: Add location-based filtering for field operations
4. **Aggregations**: Statistics and analytics queries
5. **Named vectors**: Multiple vector spaces per object
6. **Hybrid fusion**: Custom scoring algorithms

## References

- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [Hybrid Search Guide](https://weaviate.io/developers/weaviate/search/hybrid)
- [Python Client v4](https://weaviate.io/developers/weaviate/client-libraries/python)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)

## Summary

The `WeaviateStore` implementation provides production-ready hybrid search for the petroleum RAG benchmark:

✅ Complete `BaseStorage` interface implementation
✅ Hybrid search (vector + BM25) with tunable alpha
✅ Full-text indexing on content
✅ Metadata filtering support
✅ Batch operations with automatic optimization
✅ Score threshold filtering
✅ Health checks and error handling
✅ Comprehensive test coverage
✅ Integration examples

**Status:** Ready for Wave 1 benchmarking
