# API Reference

Complete API documentation for the Petroleum RAG Benchmark system.

## Table of Contents

- [Core Interfaces](#core-interfaces)
  - [BaseParser](#baseparser)
  - [BaseStorage](#basestorage)
- [Parser Implementations](#parser-implementations)
- [Storage Implementations](#storage-implementations)
- [Integration Components](#integration-components)
  - [UnifiedEmbedder](#unifiedembedder)
  - [Evaluator](#evaluator)
  - [MetricsCalculator](#metricscalculator)
  - [BenchmarkRunner](#benchmarkrunner)
- [Data Models](#data-models)
- [Configuration](#configuration)

## Core Interfaces

### BaseParser

Abstract base class for all document parsers.

**Location**: `parsers/base.py`

#### Class Definition

```python
class BaseParser(ABC):
    """Abstract base class for document parsers."""

    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        """Initialize the parser.

        Args:
            name: Name of the parser (e.g., "LlamaParse", "Docling")
            config: Optional configuration dictionary
        """
```

#### Abstract Methods

##### parse()

```python
@abstractmethod
async def parse(self, file_path: Path) -> ParsedDocument:
    """Parse a document file and extract structured content.

    Args:
        file_path: Path to the document file to parse

    Returns:
        ParsedDocument containing all extracted elements and metadata

    Raises:
        FileNotFoundError: If file_path does not exist
        ValueError: If file format is not supported
        RuntimeError: If parsing fails
    """
```

**Example**:
```python
from pathlib import Path
from parsers import DoclingParser

parser = DoclingParser()
doc = await parser.parse(Path("data/input/report.pdf"))

print(f"Parsed {len(doc.elements)} elements from {doc.total_pages} pages")
```

##### chunk_document()

```python
@abstractmethod
def chunk_document(self, doc: ParsedDocument) -> list[DocumentChunk]:
    """Chunk a parsed document into smaller units for RAG.

    Args:
        doc: ParsedDocument to chunk

    Returns:
        List of DocumentChunk objects

    Raises:
        ValueError: If document is empty or invalid
    """
```

**Example**:
```python
chunks = parser.chunk_document(doc)
print(f"Created {len(chunks)} chunks")

for chunk in chunks[:3]:
    print(f"Chunk {chunk.chunk_id}: {len(chunk.content)} chars")
```

#### Helper Methods

##### validate_file()

```python
def validate_file(self, file_path: Path) -> None:
    """Validate that the file exists and has a supported extension.

    Args:
        file_path: Path to validate

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file extension is not supported
    """
```

##### get_chunk_size()

```python
def get_chunk_size(self) -> int:
    """Get configured chunk size in characters.

    Returns:
        Chunk size from config or default value (1000)
    """
```

##### get_chunk_overlap()

```python
def get_chunk_overlap(self) -> int:
    """Get configured chunk overlap in characters.

    Returns:
        Chunk overlap from config or default value (200)
    """
```

##### estimate_tokens()

```python
def estimate_tokens(self, text: str) -> int:
    """Estimate token count for text.

    Uses heuristic: ~4 characters per token.

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
```

### BaseStorage

Abstract base class for all storage backends.

**Location**: `storage/base.py`

#### Class Definition

```python
class BaseStorage(ABC):
    """Abstract base class for storage backends."""

    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        """Initialize the storage backend.

        Args:
            name: Name of the storage backend (e.g., "Chroma", "Weaviate")
            config: Optional configuration dictionary
        """
```

#### Abstract Methods

##### initialize()

```python
@abstractmethod
async def initialize(self) -> None:
    """Initialize the storage backend.

    Should:
    - Establish connection to storage service
    - Create necessary collections/schemas/graphs
    - Verify connectivity and permissions
    - Set up indices

    Raises:
        ConnectionError: If unable to connect to storage service
        RuntimeError: If initialization fails
    """
```

**Example**:
```python
from storage import ChromaStore

store = ChromaStore()
await store.initialize()
print("Storage initialized and ready")
```

##### store_chunks()

```python
@abstractmethod
async def store_chunks(
    self,
    chunks: list[DocumentChunk],
    embeddings: list[list[float]],
) -> None:
    """Store document chunks with their embeddings.

    Args:
        chunks: List of DocumentChunk objects to store
        embeddings: List of embedding vectors (one per chunk)

    Raises:
        ValueError: If chunks and embeddings length mismatch
        RuntimeError: If storage operation fails
    """
```

**Example**:
```python
from embeddings import UnifiedEmbedder

embedder = UnifiedEmbedder()
await embedder.initialize()

# Generate embeddings for chunks
texts = [chunk.content for chunk in chunks]
embeddings = await embedder.embed_batch(texts)

# Store in backend
await store.store_chunks(chunks, embeddings)
print(f"Stored {len(chunks)} chunks")
```

##### retrieve()

```python
@abstractmethod
async def retrieve(
    self,
    query: str,
    query_embedding: list[float],
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
) -> list[RetrievalResult]:
    """Retrieve relevant chunks for a query.

    Args:
        query: Query text (for keyword/hybrid search)
        query_embedding: Query vector embedding
        top_k: Number of results to return
        filters: Optional metadata filters

    Returns:
        List of RetrievalResult objects, ranked by relevance

    Raises:
        RuntimeError: If retrieval fails
    """
```

**Example**:
```python
# Generate query embedding
query = "What is the optimal drilling fluid density?"
query_embedding = await embedder.embed_text(query)

# Retrieve relevant chunks
results = await store.retrieve(
    query=query,
    query_embedding=query_embedding,
    top_k=5
)

for result in results:
    print(f"Score: {result.score:.3f} | {result.content[:100]}...")
```

##### clear()

```python
@abstractmethod
async def clear(self) -> None:
    """Clear all data from the storage backend.

    WARNING: This is destructive and cannot be undone!

    Raises:
        RuntimeError: If clear operation fails
    """
```

#### Helper Methods

##### health_check()

```python
async def health_check(self) -> bool:
    """Check if storage backend is healthy and responsive.

    Returns:
        True if backend is healthy, False otherwise
    """
```

##### validate_chunks_embeddings()

```python
def validate_chunks_embeddings(
    self,
    chunks: list[DocumentChunk],
    embeddings: list[list[float]],
) -> None:
    """Validate that chunks and embeddings match.

    Args:
        chunks: List of chunks
        embeddings: List of embeddings

    Raises:
        ValueError: If validation fails
    """
```

## Parser Implementations

### LlamaParseParser

Cloud-based parser with advanced table extraction.

**Location**: `parsers/llamaparse_parser.py`

#### Initialization

```python
from parsers import LlamaParseParser

parser = LlamaParseParser(
    api_key="llx_...",  # Optional, uses settings if not provided
)
```

#### Configuration

```python
parser = LlamaParseParser(
    api_key="llx_...",
    config={
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "result_type": "markdown",  # or "text"
        "language": "en",
    }
)
```

#### Features

- Cloud-based processing via LlamaIndex API
- Advanced table extraction and structure recognition
- Multi-column layout handling
- Schema extraction from tables
- Semantic chunking by sections

### DoclingParser

IBM's local document understanding library.

**Location**: `parsers/docling_parser.py`

#### Initialization

```python
from parsers import DoclingParser

parser = DoclingParser()  # No API key required (local processing)
```

#### Configuration

```python
parser = DoclingParser(
    config={
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "preserve_structure": True,
        "extract_tables": True,
        "extract_figures": True,
    }
)
```

#### Features

- Local processing (no API calls)
- Document structure analysis
- Layout understanding
- Structure-aware semantic chunking
- Fast processing

### PageIndexParser

Novel semantic chunking with page-level indexing.

**Location**: `parsers/pageindex_parser.py`

#### Initialization

```python
from parsers import PageIndexParser

parser = PageIndexParser()
```

#### Configuration

```python
parser = PageIndexParser(
    config={
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "page_aware": True,
        "semantic_boundaries": True,
    }
)
```

#### Features

- Page-level context preservation
- Semantic boundary detection
- Relationship maintenance across pages
- Page-aware semantic chunking

### VertexDocAIParser

Google Cloud's enterprise OCR and layout analysis.

**Location**: `parsers/vertex_parser.py`

#### Initialization

```python
from parsers import VertexDocAIParser

parser = VertexDocAIParser(
    project_id="your-gcp-project",
    location="us",
    processor_id="your-processor-id",
)
```

#### Configuration

```python
parser = VertexDocAIParser(
    project_id="your-project",
    location="us",
    processor_id="processor-123",
    config={
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "credentials_path": "/path/to/key.json",
    }
)
```

#### Features

- Enterprise-grade OCR
- Form and entity extraction
- Production scalability
- GCP integration
- Layout-aware chunking

## Storage Implementations

### ChromaStore

Pure vector similarity search.

**Location**: `storage/chroma_store.py`

#### Initialization

```python
from storage import ChromaStore

store = ChromaStore(
    host="localhost",
    port=8000,
    collection_name="petroleum_docs",
)
await store.initialize()
```

#### Configuration

```python
store = ChromaStore(
    config={
        "host": "localhost",
        "port": 8000,
        "collection_name": "petroleum_docs",
        "top_k": 5,
        "min_score": 0.5,
    }
)
```

#### Retrieval Method

Pure vector similarity using cosine distance.

### WeaviateStore

Hybrid search (vector + BM25 keyword).

**Location**: `storage/weaviate_store.py`

#### Initialization

```python
from storage import WeaviateStore

store = WeaviateStore(
    host="localhost",
    port=8080,
    class_name="PetroleumDocument",
)
await store.initialize()
```

#### Configuration

```python
store = WeaviateStore(
    config={
        "host": "localhost",
        "port": 8080,
        "grpc_port": 50051,
        "class_name": "PetroleumDocument",
        "top_k": 5,
        "alpha": 0.75,  # Hybrid search balance (0=keyword, 1=vector)
    }
)
```

#### Retrieval Method

Hybrid search combining vector similarity and BM25 keyword matching.

### FalkorDBStore

Graph-based retrieval with vector embeddings.

**Location**: `storage/falkordb_store.py`

#### Initialization

```python
from storage import FalkorDBStore

store = FalkorDBStore(
    host="localhost",
    port=6379,
    graph_name="petroleum_rag",
)
await store.initialize()
```

#### Configuration

```python
store = FalkorDBStore(
    config={
        "host": "localhost",
        "port": 6379,
        "graph_name": "petroleum_rag",
        "top_k": 5,
        "min_score": 0.5,
        "enable_multi_hop": True,
    }
)
```

#### Retrieval Method

Vector similarity search with optional graph traversal for multi-hop queries.

## Integration Components

### UnifiedEmbedder

Generate embeddings using OpenAI's embedding models.

**Location**: `embeddings/embedder.py`

#### Class Definition

```python
class UnifiedEmbedder:
    """Unified embedder using OpenAI's embedding models."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        """Initialize the embedder.

        Args:
            api_key: OpenAI API key (defaults to settings)
            model: Embedding model name (default: text-embedding-3-small)
            dimensions: Embedding dimensions (default: 1536)
            batch_size: Batch size for processing (default: 100)
        """
```

#### Methods

##### initialize()

```python
async def initialize(self) -> None:
    """Initialize the embedder (placeholder for future setup)."""
```

##### embed_text()

```python
async def embed_text(self, text: str) -> list[float]:
    """Generate embedding for a single text.

    Args:
        text: Text to embed

    Returns:
        Embedding vector as list of floats

    Raises:
        ValueError: If text is empty
        RuntimeError: If embedding generation fails
    """
```

**Example**:
```python
embedder = UnifiedEmbedder()
await embedder.initialize()

embedding = await embedder.embed_text("Sample query text")
print(f"Embedding dimension: {len(embedding)}")
```

##### embed_batch()

```python
async def embed_batch(self, texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of texts.

    Automatically splits large batches and handles rate limiting.

    Args:
        texts: List of texts to embed

    Returns:
        List of embedding vectors (one per input text)

    Raises:
        ValueError: If texts list is empty or contains empty strings
        RuntimeError: If embedding generation fails
    """
```

**Example**:
```python
texts = [chunk.content for chunk in chunks]
embeddings = await embedder.embed_batch(texts)
print(f"Generated {len(embeddings)} embeddings")
```

#### Features

- Automatic batch processing
- Rate limit handling with exponential backoff
- Retry logic for transient failures
- Dimension validation

### Evaluator

LLM-based answer generation and evaluation.

**Location**: `evaluation/evaluator.py`

#### Class Definition

```python
class Evaluator:
    """Orchestrate RAG system evaluation with answer generation."""

    def __init__(self) -> None:
        """Initialize evaluator with LLM client and metrics calculator."""
```

#### Methods

##### generate_answer()

```python
async def generate_answer(
    self,
    query: str,
    context_chunks: list[RetrievalResult],
) -> str:
    """Generate answer to query using retrieved context.

    Args:
        query: User query
        context_chunks: List of retrieved context chunks

    Returns:
        Generated answer text
    """
```

**Example**:
```python
from evaluation import Evaluator

evaluator = Evaluator()

answer = await evaluator.generate_answer(
    query="What is the optimal drilling fluid density?",
    context_chunks=retrieved_results
)
print(f"Answer: {answer}")
```

##### evaluate_query()

```python
async def evaluate_query(
    self,
    query: BenchmarkQuery,
    retrieved: list[RetrievalResult],
    generated_answer: str,
    parser_name: str,
    storage_backend: str,
    retrieval_time: float = 0.0,
    generation_time: float = 0.0,
) -> BenchmarkResult:
    """Evaluate a single query with comprehensive metrics.

    Args:
        query: Benchmark query with ground truth
        retrieved: Retrieved results
        generated_answer: Generated answer from LLM
        parser_name: Name of parser used
        storage_backend: Name of storage backend used
        retrieval_time: Time taken for retrieval (seconds)
        generation_time: Time taken for answer generation (seconds)

    Returns:
        BenchmarkResult with all calculated metrics
    """
```

**Example**:
```python
result = await evaluator.evaluate_query(
    query=benchmark_query,
    retrieved=retrieved_chunks,
    generated_answer=answer,
    parser_name="docling",
    storage_backend="weaviate",
    retrieval_time=0.045,
    generation_time=2.3
)

print(f"Precision@5: {result.metrics['precision@5']:.3f}")
print(f"Answer Correctness: {result.metrics['answer_correctness']:.3f}")
```

### MetricsCalculator

Calculate traditional IR metrics.

**Location**: `evaluation/metrics.py`

#### Class Definition

```python
class MetricsCalculator:
    """Calculate comprehensive retrieval and quality metrics."""
```

#### Methods

##### calculate_precision_at_k()

```python
def calculate_precision_at_k(
    self,
    retrieved: list[RetrievalResult],
    relevant_ids: list[str],
    k: int,
) -> float:
    """Calculate Precision@K metric.

    Args:
        retrieved: List of retrieved results
        relevant_ids: List of relevant document/chunk IDs
        k: Number of top results to consider

    Returns:
        Precision@K score (0.0 to 1.0)
    """
```

##### calculate_recall_at_k()

```python
def calculate_recall_at_k(
    self,
    retrieved: list[RetrievalResult],
    relevant_ids: list[str],
    k: int,
) -> float:
    """Calculate Recall@K metric.

    Args:
        retrieved: List of retrieved results
        relevant_ids: List of relevant document/chunk IDs
        k: Number of top results to consider

    Returns:
        Recall@K score (0.0 to 1.0)
    """
```

##### calculate_mrr()

```python
def calculate_mrr(
    self,
    retrieved: list[RetrievalResult],
    relevant_ids: list[str],
) -> float:
    """Calculate Mean Reciprocal Rank.

    Args:
        retrieved: List of retrieved results
        relevant_ids: List of relevant document/chunk IDs

    Returns:
        MRR score (0.0 to 1.0)
    """
```

##### calculate_ndcg()

```python
def calculate_ndcg(
    self,
    retrieved: list[RetrievalResult],
    relevance_scores: dict[str, float],
    k: int,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain.

    Args:
        retrieved: List of retrieved results
        relevance_scores: Map of ID to relevance score
        k: Number of top results to consider

    Returns:
        NDCG@K score (0.0 to 1.0)
    """
```

**Example**:
```python
from evaluation.metrics import MetricsCalculator

calc = MetricsCalculator()

precision = calc.calculate_precision_at_k(retrieved, relevant_ids, k=5)
recall = calc.calculate_recall_at_k(retrieved, relevant_ids, k=5)
mrr = calc.calculate_mrr(retrieved, relevant_ids)

print(f"Precision@5: {precision:.3f}")
print(f"Recall@5: {recall:.3f}")
print(f"MRR: {mrr:.3f}")
```

### BenchmarkRunner

Main benchmark orchestration.

**Location**: `benchmark.py`

#### Class Definition

```python
class BenchmarkRunner:
    """Orchestrate complete RAG benchmark across all combinations."""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        queries_file: Path,
    ) -> None:
        """Initialize benchmark runner.

        Args:
            input_dir: Directory containing input PDFs
            output_dir: Directory for results
            queries_file: Path to queries JSON file
        """
```

#### Methods

##### run_full_benchmark()

```python
async def run_full_benchmark(self) -> list[BenchmarkResult]:
    """Run complete benchmark: parse, store, query, evaluate.

    Returns:
        List of BenchmarkResult objects (one per query per combination)
    """
```

**Example**:
```python
from pathlib import Path
from benchmark import BenchmarkRunner

runner = BenchmarkRunner(
    input_dir=Path("data/input"),
    output_dir=Path("data/results"),
    queries_file=Path("evaluation/queries.json"),
)

results = await runner.run_full_benchmark()
print(f"Completed {len(results)} benchmark tests")
```

## Data Models

### ParsedElement

Single extracted element from a document.

**Location**: `models.py`

```python
@dataclass
class ParsedElement:
    """A single element extracted from a document."""

    element_id: str
    element_type: ElementType
    content: str
    formatted_content: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    page_number: int | None = None
    bbox: list[float] | None = None
    parent_section: str | None = None
```

### ParsedDocument

Complete parsed document with all elements.

```python
@dataclass
class ParsedDocument:
    """A complete parsed document with all extracted elements."""

    document_id: str
    source_file: Path
    parser_name: str
    elements: list[ParsedElement]
    metadata: dict[str, str] = field(default_factory=dict)
    parse_time_seconds: float = 0.0
    parsed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_pages: int | None = None
    error: str | None = None
```

### DocumentChunk

Chunk of document content for RAG storage.

```python
@dataclass
class DocumentChunk:
    """A chunk of document content for RAG storage and retrieval."""

    chunk_id: str
    document_id: str
    content: str
    element_ids: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    chunk_index: int = 0
    start_page: int | None = None
    end_page: int | None = None
    token_count: int | None = None
    parent_section: str | None = None
```

### RetrievalResult

Single retrieval result from a storage backend.

```python
@dataclass
class RetrievalResult:
    """A single retrieval result from a storage backend."""

    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: dict[str, str] = field(default_factory=dict)
    rank: int = 1
    retrieval_method: Literal["vector", "hybrid", "graph"] | None = None
```

### BenchmarkQuery

Benchmark query with ground truth annotations.

```python
@dataclass
class BenchmarkQuery:
    """A benchmark query with ground truth annotations."""

    query_id: str
    query: str
    ground_truth_answer: str
    relevant_element_ids: list[str] = field(default_factory=list)
    query_type: QueryType = QueryType.GENERAL
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    notes: str | None = None
    expected_chunks: list[str] | None = None
```

### BenchmarkResult

Results from a single benchmark run.

```python
@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    benchmark_id: str
    parser_name: str
    storage_backend: str
    query_id: str
    query: str
    retrieved_results: list[RetrievalResult]
    generated_answer: str
    ground_truth_answer: str
    metrics: dict[str, float] = field(default_factory=dict)
    retrieval_time_seconds: float = 0.0
    generation_time_seconds: float = 0.0
    total_time_seconds: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error: str | None = None

    @property
    def combination_name(self) -> str:
        """Get the parser-storage combination name."""
        return f"{self.parser_name}_{self.storage_backend}"

    @property
    def success(self) -> bool:
        """Check if benchmark completed without errors."""
        return self.error is None
```

## Configuration

### Settings

Centralized configuration using Pydantic Settings.

**Location**: `config.py`

#### Class Definition

```python
class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
```

#### Key Settings

```python
# API Keys
anthropic_api_key: str
openai_api_key: str
llama_cloud_api_key: str

# Chunking
chunk_size: int = 1000
chunk_overlap: int = 200
min_chunk_size: int = 100
max_chunk_size: int = 2000

# Embeddings
embedding_model: str = "text-embedding-3-small"
embedding_dimension: int = 1536
embedding_batch_size: int = 100

# Evaluation
eval_llm_model: str = "claude-sonnet-4-20250514"
eval_llm_temperature: float = 0.0
eval_llm_max_tokens: int = 4096

# Retrieval
retrieval_top_k: int = 5
retrieval_min_score: float = 0.5

# Storage
chroma_host: str = "localhost"
chroma_port: int = 8000
weaviate_host: str = "localhost"
weaviate_port: int = 8080
falkordb_host: str = "localhost"
falkordb_port: int = 6379
```

#### Usage

```python
from config import settings

# Access settings
chunk_size = settings.chunk_size
api_key = settings.openai_api_key

# Get parser config
parser_config = settings.get_parser_config()

# Get storage config
chroma_config = settings.get_storage_config("chroma")
```

---

## Complete Example

Here's a complete example showing how all components work together:

```python
import asyncio
from pathlib import Path

from parsers import DoclingParser
from storage import WeaviateStore
from embeddings import UnifiedEmbedder
from evaluation import Evaluator
from models import BenchmarkQuery
from config import settings


async def run_example():
    """Complete RAG pipeline example."""

    # 1. Parse document
    parser = DoclingParser()
    doc = await parser.parse(Path("data/input/report.pdf"))
    print(f"Parsed {len(doc.elements)} elements")

    # 2. Chunk document
    chunks = parser.chunk_document(doc)
    print(f"Created {len(chunks)} chunks")

    # 3. Generate embeddings
    embedder = UnifiedEmbedder()
    await embedder.initialize()
    texts = [chunk.content for chunk in chunks]
    embeddings = await embedder.embed_batch(texts)
    print(f"Generated {len(embeddings)} embeddings")

    # 4. Store in backend
    store = WeaviateStore()
    await store.initialize()
    await store.store_chunks(chunks, embeddings)
    print("Stored chunks in Weaviate")

    # 5. Query
    query = "What is the optimal drilling fluid density?"
    query_embedding = await embedder.embed_text(query)
    results = await store.retrieve(
        query=query,
        query_embedding=query_embedding,
        top_k=5
    )
    print(f"Retrieved {len(results)} results")

    # 6. Generate answer
    evaluator = Evaluator()
    answer = await evaluator.generate_answer(query, results)
    print(f"Answer: {answer}")

    # 7. Evaluate (if ground truth available)
    benchmark_query = BenchmarkQuery(
        query_id="example_1",
        query=query,
        ground_truth_answer="Expected answer...",
        relevant_element_ids=["doc1_para_10", "doc1_table_2"]
    )

    result = await evaluator.evaluate_query(
        query=benchmark_query,
        retrieved=results,
        generated_answer=answer,
        parser_name="docling",
        storage_backend="weaviate",
    )

    print(f"Precision@5: {result.metrics['precision@5']:.3f}")
    print(f"Answer Correctness: {result.metrics['answer_correctness']:.3f}")


if __name__ == "__main__":
    asyncio.run(run_example())
```

---

For more details, see:
- [User Guide](USER_GUIDE.md) for usage instructions
- [Architecture](ARCHITECTURE.md) for system design
- [Deployment Guide](DEPLOYMENT.md) for production deployment
