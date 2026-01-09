# System Architecture

This document provides a comprehensive overview of the Petroleum RAG Benchmark system architecture, design decisions, and component interactions.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Core Components](#core-components)
- [Data Flow](#data-flow)
- [Design Decisions](#design-decisions)
- [Component Comparison](#component-comparison)
- [Extension Points](#extension-points)
- [Performance Characteristics](#performance-characteristics)

## Overview

The Petroleum RAG Benchmark is a modular, production-ready system designed to evaluate different RAG pipeline configurations on technical documents. The architecture follows SOLID principles and emphasizes:

- **Modularity**: Components are independent and interchangeable
- **Extensibility**: Easy to add new parsers, storage backends, or metrics
- **Testability**: Each component can be tested in isolation
- **Production-Ready**: Robust error handling, logging, and monitoring

### High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                    PETROLEUM RAG BENCHMARK                          │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                    APPLICATION LAYER                          │ │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐             │ │
│  │  │benchmark.py│  │analyze     │  │demo_app.py │             │ │
│  │  │            │  │_results.py │  │            │             │ │
│  │  │Orchestrate │  │Visualize   │  │Interactive │             │ │
│  │  └────────────┘  └────────────┘  └────────────┘             │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                            │                                        │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                    INTEGRATION LAYER                          │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │ │
│  │  │Embedder  │  │Evaluator │  │Metrics   │  │Config    │    │ │
│  │  │          │  │          │  │Calculator│  │          │    │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                            │                                        │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                  ABSTRACTION LAYER                            │ │
│  │  ┌───────────────────┐        ┌───────────────────┐          │ │
│  │  │  BaseParser       │        │  BaseStorage      │          │ │
│  │  │  (Abstract)       │        │  (Abstract)       │          │ │
│  │  └───────────────────┘        └───────────────────┘          │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                            │                                        │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │              IMPLEMENTATION LAYER                             │ │
│  │  ┌──────┐ ┌──────┐ ┌────────┐ ┌──────┐                      │ │
│  │  │Llama │ │Docling│ │Page    │ │Vertex│                      │ │
│  │  │Parse │ │       │ │Index   │ │DocAI │                      │ │
│  │  └──────┘ └──────┘ └────────┘ └──────┘                      │ │
│  │                                                                │ │
│  │  ┌──────┐ ┌──────┐ ┌──────┐                                  │ │
│  │  │Chroma│ │Weaviate│ │Falkor│                                │ │
│  │  │  DB  │ │       │ │  DB  │                                 │ │
│  │  └──────┘ └──────┘ └──────┘                                  │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                            │                                        │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                   DATA MODEL LAYER                            │ │
│  │  ParsedDocument, DocumentChunk, RetrievalResult,             │ │
│  │  BenchmarkQuery, BenchmarkResult                             │ │
│  └──────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
```

## System Architecture

### Layer Descriptions

#### 1. Data Model Layer

**Purpose**: Define all core data structures used throughout the system.

**Components**:
- `ParsedElement`: Single extracted element (text, table, figure)
- `ParsedDocument`: Complete parsed document with all elements
- `DocumentChunk`: Chunk for RAG storage
- `RetrievalResult`: Single retrieval result with score
- `BenchmarkQuery`: Test query with ground truth
- `BenchmarkResult`: Complete benchmark run results

**Design Pattern**: Data Transfer Objects (DTOs) using Python dataclasses

**Key Features**:
- Immutable by design (frozen dataclasses where appropriate)
- Type-safe with full type hints
- Self-documenting with comprehensive docstrings
- Validation via property methods

#### 2. Abstraction Layer

**Purpose**: Define contracts for parsers and storage backends.

**Components**:
- `BaseParser`: Abstract base class for all parsers
- `BaseStorage`: Abstract base class for all storage backends

**Design Pattern**: Abstract Base Class (ABC) pattern

**Key Methods**:

**BaseParser**:
```python
@abstractmethod
async def parse(self, file_path: Path) -> ParsedDocument
    """Parse document and extract structured content"""

@abstractmethod
def chunk_document(self, doc: ParsedDocument) -> list[DocumentChunk]
    """Chunk document for RAG storage"""
```

**BaseStorage**:
```python
@abstractmethod
async def initialize(self) -> None
    """Initialize storage backend"""

@abstractmethod
async def store_chunks(self, chunks, embeddings) -> None
    """Store chunks with embeddings"""

@abstractmethod
async def retrieve(self, query, query_embedding, top_k) -> list[RetrievalResult]
    """Retrieve relevant chunks"""

@abstractmethod
async def clear(self) -> None
    """Clear all stored data"""
```

#### 3. Implementation Layer

**Purpose**: Concrete implementations of parsers and storage backends.

**Parsers**:
- `LlamaParseParser`: Cloud-based parsing with table extraction
- `DoclingParser`: Local document understanding
- `PageIndexParser`: Semantic chunking with page-level indexing
- `VertexDocAIParser`: Google Cloud OCR and layout analysis

**Storage Backends**:
- `ChromaStore`: Pure vector similarity search
- `WeaviateStore`: Hybrid search (vector + BM25)
- `FalkorDBStore`: Graph-based retrieval with vectors

Each implementation:
- Inherits from appropriate base class
- Implements all abstract methods
- Adds backend-specific features
- Handles errors gracefully

#### 4. Integration Layer

**Purpose**: Coordinate interactions between components.

**Components**:
- `UnifiedEmbedder`: Generate embeddings using OpenAI
- `Evaluator`: LLM-based answer evaluation
- `MetricsCalculator`: Calculate IR and quality metrics
- `Settings`: Centralized configuration

**Design Pattern**: Facade pattern

**Key Features**:
- Unified interface to complex subsystems
- Rate limiting and retry logic
- Async/await for concurrent operations
- Comprehensive error handling

#### 5. Application Layer

**Purpose**: Orchestrate end-to-end workflows.

**Components**:
- `benchmark.py`: Main benchmark orchestration
- `analyze_results.py`: Results analysis and visualization
- `demo_app.py`: Streamlit interactive demo

**Design Pattern**: Controller/Orchestrator pattern

## Core Components

### 1. Benchmark Runner (benchmark.py)

**Responsibilities**:
- Load input documents and queries
- Initialize all parsers and storage backends
- Execute benchmark across all combinations
- Save results and intermediate outputs

**Key Methods**:

```python
class BenchmarkRunner:
    async def run_full_benchmark(self) -> list[BenchmarkResult]:
        """Run complete benchmark: parse, store, query, evaluate"""

    async def parse_documents(self) -> dict[str, ParsedDocument]:
        """Parse with all 4 parsers in parallel"""

    async def store_in_all_backends(self, parsed_docs) -> None:
        """Store in all 3 backends in parallel"""

    async def run_queries(self) -> list[BenchmarkResult]:
        """Run all queries on all 12 combinations"""
```

**Execution Flow**:
```
1. Load Configuration & Queries
   ↓
2. Initialize Components (parsers, storage, embedder, evaluator)
   ↓
3. Parse Documents (parallel across 4 parsers)
   ↓
4. Generate Embeddings
   ↓
5. Store in Backends (parallel across 3 storage × 4 parsed docs)
   ↓
6. Run Queries (sequential across 12 combinations)
   ↓
7. Calculate Metrics (IR + LLM-based)
   ↓
8. Save Results
```

### 2. Parsers

#### LlamaParse Parser

**Technology**: Cloud API (LlamaIndex)

**Strengths**:
- Advanced table extraction and structure recognition
- Multi-column layout handling
- Schema extraction from tables

**Chunking Strategy**: Semantic chunking by sections and tables

**Best For**: Documents with complex tables and multi-column layouts

#### Docling Parser

**Technology**: IBM Research library (local processing)

**Strengths**:
- Document structure analysis
- Layout understanding
- Fast local processing

**Chunking Strategy**: Structure-aware semantic chunking

**Best For**: Documents requiring structure preservation

#### PageIndex Parser

**Technology**: Novel semantic approach

**Strengths**:
- Page-level context preservation
- Semantic boundary detection
- Relationship maintenance

**Chunking Strategy**: Page-aware semantic chunking

**Best For**: Documents where page context matters

#### Vertex Document AI Parser

**Technology**: Google Cloud Platform

**Strengths**:
- Enterprise-grade OCR
- Form and entity extraction
- Production scalability

**Chunking Strategy**: Layout-aware chunking

**Best For**: Production workloads, scanned documents

### 3. Storage Backends

#### ChromaDB

**Type**: Pure vector database

**Retrieval Method**: Cosine similarity on embeddings

**Strengths**:
- Fast semantic search
- Simple setup (embedded or client-server)
- Efficient vector operations

**Use Case**: Applications prioritizing semantic similarity

**Connection**:
```python
import chromadb
client = chromadb.HttpClient(host="localhost", port=8000)
collection = client.get_or_create_collection("petroleum_docs")
```

#### Weaviate

**Type**: Hybrid search engine

**Retrieval Method**: Vector + BM25 keyword search

**Strengths**:
- Combines semantic and lexical matching
- GraphQL API
- Production-ready scalability

**Use Case**: Applications needing both semantic and keyword search

**Connection**:
```python
import weaviate
client = weaviate.Client("http://localhost:8080")
```

**Hybrid Query**:
```python
results = client.query.get("PetroleumDocument", ["content"]) \
    .with_hybrid(query=query_text, alpha=0.75) \
    .with_limit(top_k) \
    .do()
```

#### FalkorDB

**Type**: Graph database with vector support

**Retrieval Method**: Vector similarity + graph traversal

**Strengths**:
- Multi-hop relationship queries
- Cypher query language
- Redis-based performance

**Use Case**: Applications requiring relationship traversal

**Connection**:
```python
from falkordb import FalkorDB
db = FalkorDB(host="localhost", port=6379)
graph = db.select_graph("petroleum_rag")
```

**Graph Query**:
```python
query = """
MATCH (c:Chunk)
CALL db.idx.vector.queryNodes('chunk_embeddings', $k, $embedding)
YIELD node, score
RETURN node.content, score
ORDER BY score DESC
"""
```

### 4. Embedder

**Purpose**: Generate consistent embeddings across all components

**Model**: OpenAI `text-embedding-3-small` (1536 dimensions)

**Features**:
- Batch processing for efficiency
- Rate limiting to avoid API throttling
- Retry logic with exponential backoff
- Caching for repeated texts

**Usage**:
```python
embedder = UnifiedEmbedder()
await embedder.initialize()

# Single embedding
embedding = await embedder.embed_text("query text")

# Batch embeddings
embeddings = await embedder.embed_batch(["text1", "text2", ...])
```

### 5. Evaluator

**Purpose**: Evaluate answer quality using LLM

**Model**: Claude Sonnet 4 (Anthropic)

**Metrics Calculated**:
- Context Relevance (0.0-1.0)
- Answer Correctness (0.0-1.0)
- Semantic Similarity (0.0-1.0)
- Factual Accuracy (0.0-1.0)
- Completeness (0.0-1.0)
- Faithfulness (0.0-1.0)
- Hallucination Count (integer)

**Evaluation Prompt**:
```python
prompt = f"""
You are evaluating a RAG system's performance.

Query: {query}
Ground Truth: {ground_truth}
Generated Answer: {answer}
Retrieved Context: {context}

Rate the following on a scale of 0.0 to 1.0:
1. Context Relevance: How relevant is the context to the query?
2. Answer Correctness: How well does the answer match ground truth?
3. Semantic Similarity: Semantic overlap between answer and ground truth?
4. Factual Accuracy: Are facts in the answer correct?
5. Completeness: Does the answer cover all aspects?
6. Faithfulness: Is the answer supported by the context?
7. Hallucinations: Count of unsupported claims (integer)

Return JSON: {{"context_relevance": 0.8, ...}}
"""
```

### 6. Metrics Calculator

**Purpose**: Calculate traditional IR metrics

**Metrics**:
- **Precision@K**: Fraction of top K results that are relevant
- **Recall@K**: Fraction of relevant results in top K
- **F1@K**: Harmonic mean of Precision and Recall
- **MRR**: Mean Reciprocal Rank (1/rank of first relevant)
- **MAP**: Mean Average Precision (average of Precision@K for all K)
- **NDCG@K**: Normalized Discounted Cumulative Gain (ranking quality)

**Composite Score**:
```python
composite_score = (
    0.4 * retrieval_accuracy +  # IR metrics
    0.4 * answer_quality +       # LLM metrics
    0.2 * speed_score            # Efficiency
)
```

## Data Flow

### End-to-End Flow

```
┌─────────────┐
│  Input PDF  │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Parse (4 parsers in parallel)      │
│  - LlamaParse                        │
│  - Docling                           │
│  - PageIndex                         │
│  - Vertex Document AI                │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  4 × ParsedDocument objects          │
│  (elements, metadata, structure)     │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Chunk Documents                     │
│  - Parser-specific chunking          │
│  - Respect semantic boundaries       │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  4 × List[DocumentChunk]             │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Generate Embeddings                 │
│  - OpenAI text-embedding-3-small     │
│  - Batch processing                  │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Store (3 backends × 4 parsers)      │
│  - ChromaDB                          │
│  - Weaviate                          │
│  - FalkorDB                          │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  12 Populated Storage Backends       │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  For Each Query:                     │
│  - Generate query embedding          │
│  - Retrieve from each backend        │
│  - Generate answer with LLM          │
│  - Calculate metrics                 │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  BenchmarkResult objects              │
│  (12 per query)                      │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Aggregate & Analyze                 │
│  - Calculate composite scores        │
│  - Identify winner                   │
│  - Generate visualizations           │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Output:                             │
│  - raw_results.json                  │
│  - comparison.csv                    │
│  - charts/*.png                      │
│  - REPORT.md                         │
└─────────────────────────────────────┘
```

## Design Decisions

### 1. Async/Await Architecture

**Decision**: Use async/await for all I/O operations

**Rationale**:
- Document parsing involves network calls (LlamaParse, Vertex)
- Storage operations are I/O bound
- Can parallelize independent operations
- Better resource utilization

**Implementation**:
- All parsers implement `async def parse()`
- All storage implements `async def retrieve()`
- Benchmark runner uses `asyncio.gather()` for parallelization

### 2. Abstract Base Classes

**Decision**: Use ABC pattern for parsers and storage

**Rationale**:
- Enforces consistent interface across implementations
- Makes it easy to add new parsers/storage
- Enables polymorphic usage in benchmark runner
- Facilitates testing with mocks

**Trade-off**: Slightly more boilerplate, but worth it for extensibility

### 3. Pydantic Settings for Configuration

**Decision**: Use Pydantic Settings for configuration management

**Rationale**:
- Type-safe configuration
- Automatic validation
- Environment variable loading
- IDE autocomplete support
- Self-documenting

**Alternative Considered**: Plain environment variables → rejected due to lack of validation

### 4. Dataclasses for Models

**Decision**: Use Python dataclasses for data models

**Rationale**:
- Clean syntax with minimal boilerplate
- Built-in serialization support
- Type hints for IDE support
- Immutability option (frozen)

**Alternative Considered**: Pydantic models → rejected as overkill for internal models

### 5. Composite Scoring

**Decision**: Combine IR metrics, LLM metrics, and speed into composite score

**Rationale**:
- Single metric easier to compare
- Balances multiple objectives
- Configurable weights for different priorities

**Weights**:
- Retrieval accuracy: 40%
- Answer quality: 40%
- Speed: 20%

### 6. Parallel Execution

**Decision**: Parse and store in parallel where possible

**Rationale**:
- Parsers are independent → parallelize
- Storage operations independent → parallelize
- Queries must be sequential (fair comparison)

**Performance Impact**: 4x speedup for parsing, 3x for storage

### 7. Intermediate Results

**Decision**: Save parsed documents and storage states

**Rationale**:
- Debugging failed benchmarks
- Resume from checkpoints
- Analyze parser outputs independently
- Cost savings (avoid re-parsing)

**Trade-off**: More disk space (~500 MB per document)

## Component Comparison

### Parser Comparison

| Feature | LlamaParse | Docling | PageIndex | Vertex AI |
|---------|-----------|---------|-----------|-----------|
| **Processing** | Cloud | Local | Local | Cloud |
| **Table Extraction** | Excellent | Good | Fair | Excellent |
| **Structure Preservation** | Good | Excellent | Good | Good |
| **Speed** | Medium | Fast | Fast | Medium |
| **Cost** | API fees | Free | Free | API fees |
| **OCR** | No | No | No | Yes |
| **Best For** | Complex tables | Structure analysis | Semantic context | Scanned docs |
| **Chunking** | Semantic | Structure-aware | Page-aware | Layout-aware |

### Storage Comparison

| Feature | ChromaDB | Weaviate | FalkorDB |
|---------|----------|----------|----------|
| **Search Type** | Vector | Hybrid | Graph + Vector |
| **Setup** | Simple | Medium | Medium |
| **Scalability** | Good | Excellent | Good |
| **Query Types** | Semantic | Semantic + Keyword | Multi-hop |
| **Speed** | Fast | Fast | Medium |
| **Best For** | Pure similarity | Balanced search | Relationship queries |
| **API** | Python/REST | GraphQL/REST | Cypher/Redis |

### Metric Comparison

| Metric Type | Metrics | Pros | Cons |
|-------------|---------|------|------|
| **IR Metrics** | Precision, Recall, NDCG | Objective, Fast, Established | Requires ground truth |
| **LLM Metrics** | Correctness, Faithfulness | Nuanced, Contextual | Subjective, Slow, Costly |
| **Composite** | Weighted combination | Balanced, Single value | Weight selection |

## Extension Points

### Adding a New Parser

1. Create new file in `parsers/` directory
2. Inherit from `BaseParser`
3. Implement `parse()` and `chunk_document()` methods
4. Add to `parsers/__init__.py`
5. Update benchmark runner to include new parser

Example:
```python
from parsers.base import BaseParser
from models import ParsedDocument, DocumentChunk

class CustomParser(BaseParser):
    def __init__(self):
        super().__init__(name="CustomParser")

    async def parse(self, file_path: Path) -> ParsedDocument:
        # Your parsing logic
        ...
        return ParsedDocument(...)

    def chunk_document(self, doc: ParsedDocument) -> list[DocumentChunk]:
        # Your chunking logic
        ...
        return chunks
```

### Adding a New Storage Backend

1. Create new file in `storage/` directory
2. Inherit from `BaseStorage`
3. Implement all abstract methods
4. Add to `storage/__init__.py`
5. Update benchmark runner and docker-compose.yml

Example:
```python
from storage.base import BaseStorage
from models import DocumentChunk, RetrievalResult

class CustomStorage(BaseStorage):
    def __init__(self):
        super().__init__(name="CustomStorage")

    async def initialize(self) -> None:
        # Setup connection
        ...

    async def store_chunks(self, chunks, embeddings) -> None:
        # Store data
        ...

    async def retrieve(self, query, query_embedding, top_k) -> list[RetrievalResult]:
        # Retrieve relevant chunks
        ...
        return results

    async def clear(self) -> None:
        # Clear all data
        ...
```

### Adding Custom Metrics

1. Add metric calculation to `evaluation/metrics.py`
2. Update `MetricsCalculator` class
3. Include in composite score if desired

Example:
```python
class MetricsCalculator:
    def calculate_custom_metric(
        self,
        retrieved: list[RetrievalResult],
        ground_truth: list[str]
    ) -> float:
        # Your metric logic
        ...
        return score
```

## Performance Characteristics

### Parsing Performance

| Parser | Speed | Memory | API Cost |
|--------|-------|--------|----------|
| LlamaParse | 30-60s | Low | $0.003/page |
| Docling | 10-20s | Medium | Free |
| PageIndex | 10-20s | Medium | Free |
| Vertex AI | 20-40s | Low | $1.50/1000 pages |

### Storage Performance

| Operation | ChromaDB | Weaviate | FalkorDB |
|-----------|----------|----------|----------|
| **Insert (1000 chunks)** | 2-3s | 3-4s | 4-5s |
| **Retrieve (top-5)** | 30-50ms | 40-60ms | 60-80ms |
| **Memory (1000 chunks)** | 50 MB | 80 MB | 100 MB |

### Benchmark Runtime

For **1 document, 15 queries**:
- Parsing (parallel): 2-5 minutes
- Storage (parallel): 1-2 minutes
- Queries (sequential): 45-90 minutes
- **Total**: 60-100 minutes

**Scaling**:
- N documents: N × base time (linear)
- M queries: M × base time (linear)

### Resource Requirements

**Minimum**:
- CPU: 2 cores
- Memory: 4 GB
- Storage: 10 GB
- Network: Broadband

**Recommended**:
- CPU: 4+ cores
- Memory: 8 GB
- Storage: 50 GB SSD
- Network: High-speed broadband

## Conclusion

The Petroleum RAG Benchmark architecture is designed for:
- **Modularity**: Easy to extend and modify
- **Production-Readiness**: Robust error handling and logging
- **Performance**: Parallel execution where possible
- **Testability**: Clear interfaces and separation of concerns
- **Maintainability**: Clean code with comprehensive documentation

This architecture supports the project's goal of providing comprehensive, fair, and actionable benchmarking of RAG pipeline configurations for technical documents.
