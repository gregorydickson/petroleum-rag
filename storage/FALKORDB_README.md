# FalkorDB Storage Implementation

## Overview

FalkorDB is a graph database built on Redis that combines vector similarity search with graph traversal capabilities. This implementation leverages both features to provide enhanced retrieval for the petroleum RAG benchmark.

## Key Features

### 1. Graph Structure

The implementation creates a hierarchical graph structure:

```
Document (node)
    ↓ CONTAINS
Section (node)
    ↓ CONTAINS
Chunk (node with embedding vector)
```

### 2. Relationship Types

- **CONTAINS**: Hierarchical relationships (Document → Section → Chunk)
- **FOLLOWS**: Sequential relationships between chunks in the same section
- **REFERENCES**: Cross-reference relationships based on metadata

### 3. Multi-Hop Retrieval

Unlike pure vector search, FalkorDB retrieval:
1. Performs initial vector similarity search to find top-k chunks
2. Follows REFERENCES edges to find related context
3. Returns expanded results with discounted scores for graph-traversed chunks

## Architecture

### Graph Schema

```cypher
# Nodes
(:Document {document_id: str})
(:Section {section_name: str, document_id: str})
(:Chunk {
    chunk_id: str,
    document_id: str,
    content: str,
    chunk_index: int,
    start_page: int,
    end_page: int,
    token_count: int,
    parent_section: str,
    metadata: str,
    embedding: vector
})

# Relationships
(Document)-[:CONTAINS]->(Section)
(Section)-[:CONTAINS]->(Chunk)
(Chunk)-[:FOLLOWS]->(Chunk)
(Chunk)-[:REFERENCES]->(Chunk)

# Indices
CREATE INDEX ON :Chunk(chunk_id)
CREATE INDEX ON :Document(document_id)
CREATE INDEX ON :Section(section_name)
CALL db.idx.vector.create('Chunk', 'embedding', 'FLAT', {}, 'COSINE')
```

### Retrieval Strategy

```python
# Step 1: Vector similarity search
MATCH (c:Chunk)
WHERE vecf.euclideandistance(c.embedding, $query_embedding) IS NOT NULL
ORDER BY distance ASC
LIMIT top_k

# Step 2: Graph expansion
MATCH (initial:Chunk)-[:REFERENCES]->(related:Chunk)
WHERE NOT related.chunk_id IN $already_retrieved
RETURN related
LIMIT 2 per initial chunk

# Step 3: Score combination
initial_score = 1.0 / (1.0 + distance)
expanded_score = initial_score * 0.7  # Discount factor
```

## Usage

### Basic Setup

```python
from storage.falkordb_store import FalkorDBStore

# Initialize store
config = {
    "host": "localhost",
    "port": 6379,
    "graph_name": "petroleum_rag",
    "top_k": 5,
    "min_score": 0.5,
}
store = FalkorDBStore(config)

# Initialize connection and schema
await store.initialize()
```

### Storing Chunks

```python
from models import DocumentChunk

chunks = [
    DocumentChunk(
        chunk_id="chunk_1",
        document_id="doc_1",
        content="Drilling operations require...",
        chunk_index=0,
        parent_section="Operations",
        metadata={"references": "chunk_5, chunk_8"}  # For cross-refs
    ),
    # ... more chunks
]

embeddings = [
    [0.1, 0.2, 0.3, ...],  # Vector for chunk_1
    # ... more embeddings
]

await store.store_chunks(chunks, embeddings)
```

### Retrieving with Graph Expansion

```python
# Query embedding from your embedding model
query_embedding = [0.15, 0.25, 0.35, ...]

# Retrieve with multi-hop expansion
results = await store.retrieve(
    query="drilling mud weight control",
    query_embedding=query_embedding,
    top_k=5,
    filters={"document_id": "doc_1"}  # Optional filtering
)

# Results include both vector matches and graph-expanded context
for result in results:
    print(f"{result.chunk_id}: {result.score:.3f}")
    if "expanded_from" in result.metadata:
        print(f"  ↳ Expanded from: {result.metadata['expanded_from']}")
```

## Cross-References

To enable multi-hop retrieval, include references in chunk metadata:

```python
chunk = DocumentChunk(
    chunk_id="main_discussion",
    document_id="doc_1",
    content="...",
    metadata={
        "references": "appendix_a, figure_5, table_2"  # Comma-separated chunk IDs
    }
)
```

The storage backend automatically creates REFERENCES edges during `store_chunks()`.

## Prerequisites

### Docker

```bash
# Run FalkorDB with vector support
docker run -d \
    --name falkordb \
    -p 6379:6379 \
    falkordb/falkordb:latest

# Verify it's running
docker ps | grep falkordb
```

### Python Dependencies

```bash
pip install falkordb>=1.0.0
```

## Testing

### Unit Tests

```bash
# Run FalkorDB-specific tests
pytest tests/test_falkordb_store.py -v

# Requires FalkorDB running on localhost:6379
```

### Manual Testing

```bash
# Run manual test script
python test_falkordb_manual.py

# This script:
# - Creates test graph
# - Stores chunks with relationships
# - Tests retrieval and graph traversal
# - Verifies multi-hop expansion
# - Cleans up after
```

## Performance Considerations

### Vector Index

FalkorDB uses Redis Vector Similarity for vector search:
- **Index Type**: FLAT (brute force, suitable for small-medium datasets)
- **Distance Metric**: COSINE (standard for embeddings)
- **Performance**: O(n) for queries but fast for datasets < 100k vectors

For larger datasets, consider:
- Using HNSW index (if supported)
- Partitioning by document or section
- Pre-filtering before vector search

### Graph Traversal

Multi-hop expansion adds latency:
- Each initial result triggers 1 additional graph query
- Limited to 2 expanded chunks per initial result
- Total overhead: ~10-50ms depending on graph size

To optimize:
- Reduce expansion depth or breadth
- Use graph indices on chunk_id
- Batch expansion queries

## Comparison with Other Backends

| Feature | ChromaDB | Weaviate | FalkorDB |
|---------|----------|----------|----------|
| Vector Search | ✓ Pure | ✓ Hybrid | ✓ Pure |
| Keyword Search | ✗ | ✓ BM25 | ✗ |
| Graph Traversal | ✗ | ✗ | ✓ Multi-hop |
| Relationships | ✗ | ✓ Limited | ✓ Full graph |
| Best For | Simple RAG | Hybrid search | Structured docs |

### When to Use FalkorDB

**Good fit:**
- Documents with internal cross-references
- Multi-section technical documents
- When related context improves answers
- Structured knowledge (standards, regulations)

**Not ideal:**
- Simple Q&A over unstructured text
- When pure semantic search is sufficient
- Very large datasets (>1M chunks)

## Troubleshooting

### Connection Failed

```
ConnectionError: Could not connect to FalkorDB
```

**Solution:** Verify FalkorDB is running:
```bash
docker ps | grep falkordb
redis-cli -h localhost -p 6379 PING
```

### Vector Index Not Created

```
WARNING: Vector index creation skipped
```

**Cause:** Vector similarity extension not enabled in Redis/FalkorDB

**Solution:** Use FalkorDB >= 3.0 or Redis Stack with vector support

### No Multi-Hop Results

If retrieval doesn't expand beyond initial results:

1. Check REFERENCES relationships exist:
```python
query = "MATCH ()-[r:REFERENCES]->() RETURN COUNT(r)"
result = store.graph.query(query)
```

2. Verify chunk metadata includes references:
```python
chunk.metadata = {"references": "other_chunk_id"}
```

3. Check expansion query logs for errors

## Future Enhancements

Potential improvements:

1. **Bidirectional expansion**: Follow both incoming and outgoing REFERENCES
2. **Relevance propagation**: Adjust scores based on graph structure
3. **Temporal relationships**: Add PRECEDES/FOLLOWS for temporal context
4. **Entity linking**: Extract and link entities across chunks
5. **Hybrid search**: Combine with BM25 keyword search

## References

- [FalkorDB Documentation](https://docs.falkordb.com/)
- [Redis Vector Similarity](https://redis.io/docs/stack/search/reference/vectors/)
- [Cypher Query Language](https://neo4j.com/docs/cypher-manual/current/)

## License

MIT License - See repository root for details
