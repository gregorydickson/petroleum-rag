# Parallel Agent Execution Plan
## Optimized Workplan for Claude Code Agents

**Goal:** Build the benchmark system in **4-6 hours** using parallel agents, versus 15+ hours sequential.

**Strategy:** Decompose work into atomic tasks that can run simultaneously, minimizing dependencies.

---

## 🧠 Ultrathink: Dependency Analysis

### Critical Path Identification

```
┌─────────────────────────────────────────────────────────────┐
│ WAVE 0: Foundation (SEQUENTIAL - 1 hour)                    │
│ ✓ Project structure                                         │
│ ✓ Base classes (BaseParser, BaseStorage)                   │
│ ✓ Data models (ParsedDocument, DocumentChunk, etc.)        │
│ ✓ Config (Settings with Pydantic)                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ WAVE 1: Core Implementation (PARALLEL - 3 hours)            │
│                                                             │
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│ │ Agent 1     │  │ Agent 2     │  │ Agent 3     │         │
│ │ LlamaParse  │  │ Docling     │  │ PageIndex   │         │
│ │ Parser      │  │ Parser      │  │ Parser      │         │
│ └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│ │ Agent 4     │  │ Agent 5     │  │ Agent 6     │         │
│ │ Vertex      │  │ Chroma      │  │ Weaviate    │         │
│ │ Parser      │  │ Storage     │  │ Storage     │         │
│ └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│ │ Agent 7     │  │ Agent 8     │  │ Agent 9     │         │
│ │ FalkorDB    │  │ Embedder +  │  │ Evaluation  │         │
│ │ Storage     │  │ Utilities   │  │ Framework   │         │
│ └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
│ ┌─────────────┐  ┌─────────────┐                          │
│ │ Agent 10    │  │ Agent 11    │                          │
│ │ Test        │  │ Docker +    │                          │
│ │ Queries     │  │ Deployment  │                          │
│ └─────────────┘  └─────────────┘                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ WAVE 2: Integration (SEQUENTIAL - 1.5 hours)                │
│ ✓ Benchmark runner (connects all components)               │
│ ✓ Results analysis & visualization                         │
│ ✓ Demo app with winner selection                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ WAVE 3: Testing & Validation (PARALLEL - 1 hour)            │
│ ✓ Integration tests                                        │
│ ✓ End-to-end test with sample PDF                          │
│ ✓ Verify all 12 combinations work                          │
└─────────────────────────────────────────────────────────────┘
```

### Time Comparison

| Approach | Time | Notes |
|----------|------|-------|
| **Sequential** | 15-16 hours | One developer, one task at a time |
| **Parallel (4 devs)** | 8-10 hours | Still sequential per person |
| **Parallel Agents** | **4-6 hours** | 11 agents working simultaneously |

### Key Insight: Loose Coupling

Because we use **abstract base classes**, each implementation is independent:
- Parsers only need to implement `BaseParser` interface
- Storage only needs to implement `BaseStorage` interface
- No cross-dependencies between parsers
- No cross-dependencies between storage backends

This enables **massive parallelization**.

---

## 📋 Wave 0: Foundation (Sequential)

**Duration:** 1 hour
**Agent:** 1 (or manual setup)
**Must complete before Wave 1**

### Task 0.1: Project Structure & Dependencies

```bash
# Create structure
mkdir -p petroleum-rag-benchmark/{parsers,storage,embeddings,evaluation,data/{input,parsed,results},tests}

# Install dependencies
pip install llama-parse docling pageindex google-cloud-documentai \
    chromadb weaviate-client falkordb \
    anthropic openai sentence-transformers \
    pandas matplotlib seaborn streamlit \
    python-dotenv pydantic pydantic-settings pytest pytest-asyncio
```

**Output:**
- `pyproject.toml`
- `.env.example`
- `README.md`
- Directory structure

### Task 0.2: Base Classes & Models

**File: `models.py`** (shared data models)
```python
@dataclass
class ParsedElement: ...
@dataclass
class ParsedDocument: ...
@dataclass
class DocumentChunk: ...
@dataclass
class RetrievalResult: ...
@dataclass
class BenchmarkQuery: ...
@dataclass
class BenchmarkResult: ...
```

**File: `parsers/base.py`** (abstract parser interface)
```python
class BaseParser(ABC):
    @abstractmethod
    async def parse(self, file_path: Path) -> ParsedDocument: ...

    @abstractmethod
    def chunk_document(self, doc: ParsedDocument) -> list[DocumentChunk]: ...
```

**File: `storage/base.py`** (abstract storage interface)
```python
class BaseStorage(ABC):
    @abstractmethod
    async def initialize(self): ...

    @abstractmethod
    async def store_chunks(self, chunks, embeddings): ...

    @abstractmethod
    async def retrieve(self, query, query_embedding, top_k) -> list[RetrievalResult]: ...
```

**File: `config.py`** (configuration)
```python
class Settings(BaseSettings):
    # API keys
    anthropic_api_key: str
    llama_cloud_api_key: str
    openai_api_key: str

    # Parser settings
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Storage settings
    chroma_host: str = "localhost"
    weaviate_host: str = "localhost"
    falkordb_host: str = "localhost"

    # Evaluation
    eval_llm_model: str = "claude-sonnet-4-20250514"
```

**Acceptance Criteria:**
- ✅ All base classes defined with complete interfaces
- ✅ All data models defined with proper types
- ✅ Configuration loads from environment
- ✅ Project structure matches specification
- ✅ Dependencies installed

---

## 🚀 Wave 1: Core Implementation (Parallel)

**Duration:** 3 hours (parallel execution)
**Agents:** 11 simultaneously
**Dependencies:** Requires Wave 0 complete

### Agent 1: LlamaParse Parser

**Task:** Implement `parsers/llamaparse_parser.py`

**Requirements:**
```python
class LlamaParseParser(BaseParser):
    def __init__(self):
        super().__init__("LlamaParse")
        self.parser = LlamaParse(
            api_key=settings.llama_cloud_api_key,
            result_type="markdown",
            ...
        )

    async def parse(self, file_path: Path) -> ParsedDocument:
        # Use LlamaParse API
        # Convert to ParsedDocument format
        # Extract tables, figures, sections
        ...

    def chunk_document(self, doc: ParsedDocument) -> list[DocumentChunk]:
        # Intelligent chunking preserving structure
        ...
```

**Test Data:** Use mock PDF or sample 1-page PDF
**Acceptance Criteria:**
- ✅ Parses PDF successfully
- ✅ Extracts tables correctly
- ✅ Returns ParsedDocument with all fields
- ✅ Chunks intelligently (respects section boundaries)
- ✅ Unit tests pass

### Agent 2: Docling Parser

**Task:** Implement `parsers/docling_parser.py`

**Requirements:**
```python
from docling.document_converter import DocumentConverter

class DoclingParser(BaseParser):
    def __init__(self):
        super().__init__("Docling")
        self.converter = DocumentConverter(...)

    async def parse(self, file_path: Path) -> ParsedDocument:
        # Use Docling converter
        # Map Docling blocks to ParsedElements
        # Preserve table structure (HTML + Markdown)
        ...

    def chunk_document(self, doc: ParsedDocument) -> list[DocumentChunk]:
        # Leverage Docling's semantic structure
        ...
```

**Key Focus:** Table extraction quality
**Acceptance Criteria:**
- ✅ Handles complex multi-column layouts
- ✅ Extracts tables with structure preserved
- ✅ Returns ParsedDocument format
- ✅ Chunks by semantic units
- ✅ Unit tests pass

### Agent 3: PageIndex Parser

**Task:** Implement `parsers/pageindex_parser.py`

**Requirements:**
```python
from pageindex import PageIndex

class PageIndexParser(BaseParser):
    def __init__(self):
        super().__init__("PageIndex")
        self.client = PageIndex(...)

    async def parse(self, file_path: Path) -> ParsedDocument:
        # Use PageIndex API
        # Leverage their novel chunking
        ...

    def chunk_document(self, doc: ParsedDocument) -> list[DocumentChunk]:
        # PageIndex has built-in intelligent chunking
        # Use their semantic units directly
        ...
```

**Key Focus:** Novel chunking approach
**Acceptance Criteria:**
- ✅ Integrates PageIndex API/library
- ✅ Preserves their chunking strategy
- ✅ Returns ParsedDocument format
- ✅ Unit tests pass

### Agent 4: Vertex DocAI Parser

**Task:** Implement `parsers/vertex_parser.py`

**Requirements:**
```python
from google.cloud import documentai_v1

class VertexDocAIParser(BaseParser):
    def __init__(self):
        super().__init__("VertexDocAI")
        self.client = documentai_v1.DocumentProcessorServiceClient(...)

    async def parse(self, file_path: Path) -> ParsedDocument:
        # Call Vertex Document AI
        # Extract layout, tables, text
        ...

    def chunk_document(self, doc: ParsedDocument) -> list[DocumentChunk]:
        # Chunk based on Vertex layout analysis
        ...
```

**Key Focus:** Enterprise OCR quality
**Acceptance Criteria:**
- ✅ Connects to Vertex DocAI
- ✅ Handles scanned documents
- ✅ Returns ParsedDocument format
- ✅ Unit tests pass

### Agent 5: Chroma Storage

**Task:** Implement `storage/chroma_store.py`

**Requirements:**
```python
import chromadb

class ChromaStore(BaseStorage):
    def __init__(self):
        super().__init__("Chroma")
        self.client = chromadb.HttpClient(...)

    async def initialize(self):
        # Create collection
        ...

    async def store_chunks(self, chunks, embeddings):
        # Batch insert to Chroma
        ...

    async def retrieve(self, query, query_embedding, top_k) -> list[RetrievalResult]:
        # Vector similarity search
        # Map to RetrievalResult format
        ...
```

**Key Focus:** Pure vector search baseline
**Acceptance Criteria:**
- ✅ Connects to Chroma (localhost:8000)
- ✅ Stores chunks with embeddings
- ✅ Retrieves by similarity
- ✅ Returns RetrievalResult format
- ✅ Unit tests with mock data

### Agent 6: Weaviate Storage

**Task:** Implement `storage/weaviate_store.py`

**Requirements:**
```python
import weaviate

class WeaviateStore(BaseStorage):
    def __init__(self):
        super().__init__("Weaviate")
        self.client = weaviate.connect_to_local(...)

    async def initialize(self):
        # Create schema with properties
        ...

    async def store_chunks(self, chunks, embeddings):
        # Batch insert with metadata
        ...

    async def retrieve(self, query, query_embedding, top_k) -> list[RetrievalResult]:
        # HYBRID search: vector + keyword (BM25)
        # alpha=0.7 (70% vector, 30% keyword)
        ...
```

**Key Focus:** Hybrid search (different from pure vector)
**Acceptance Criteria:**
- ✅ Connects to Weaviate (localhost:8080)
- ✅ Stores chunks with full-text indexing
- ✅ Retrieves with hybrid search
- ✅ Returns RetrievalResult format
- ✅ Unit tests with mock data

### Agent 7: FalkorDB Storage

**Task:** Implement `storage/falkordb_store.py`

**Requirements:**
```python
from falkordb import FalkorDB

class FalkorDBStore(BaseStorage):
    def __init__(self):
        super().__init__("FalkorDB")
        self.db = FalkorDB(...)

    async def initialize(self):
        # Create graph schema
        # Nodes: Document, Section, Chunk
        # Edges: CONTAINS, FOLLOWS, REFERENCES
        ...

    async def store_chunks(self, chunks, embeddings):
        # Store as graph nodes with embeddings
        # Create relationships based on structure
        ...

    async def retrieve(self, query, query_embedding, top_k) -> list[RetrievalResult]:
        # Vector similarity + graph traversal
        # Follow REFERENCES edges for multi-hop
        ...
```

**Key Focus:** Graph relationships for multi-hop queries
**Acceptance Criteria:**
- ✅ Connects to FalkorDB (localhost:6379)
- ✅ Creates graph structure
- ✅ Stores chunks with relationships
- ✅ Retrieves with graph traversal
- ✅ Returns RetrievalResult format
- ✅ Unit tests with mock data

### Agent 8: Embedder & Utilities

**Task:** Implement `embeddings/embedder.py` and `utils/`

**Requirements:**
```python
class UnifiedEmbedder:
    def __init__(self):
        self.client = AsyncOpenAI(...)

    async def embed_text(self, text: str) -> list[float]:
        # Generate single embedding
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Batch processing for efficiency
        ...
```

**Also:** GCP utilities, logging setup, helper functions

**Acceptance Criteria:**
- ✅ Generates embeddings via OpenAI API
- ✅ Handles batching efficiently
- ✅ Error handling and retries
- ✅ Unit tests pass

### Agent 9: Evaluation Framework

**Task:** Implement `evaluation/metrics.py` and `evaluation/evaluator.py`

**Requirements:**
```python
class MetricsCalculator:
    def calculate_precision_at_k(self, retrieved, relevant_ids, k): ...
    def calculate_recall_at_k(self, retrieved, relevant_ids, k): ...
    def calculate_mrr(self, retrieved, relevant_ids): ...
    def calculate_ndcg(self, retrieved, relevance_scores, k): ...

    async def evaluate_context_relevance(self, query, retrieved): ...
    async def evaluate_answer_correctness(self, question, answer, ground_truth): ...
    async def evaluate_faithfulness(self, answer, context): ...

class Evaluator:
    async def generate_answer(self, query, context_chunks) -> str:
        # Call Claude with context
        ...

    async def evaluate_query(self, query, retrieved, generated_answer):
        # Calculate all metrics
        ...
```

**Acceptance Criteria:**
- ✅ All metrics implemented
- ✅ LLM-as-judge for quality metrics
- ✅ Returns structured results
- ✅ Unit tests pass

### Agent 10: Test Queries

**Task:** Create `evaluation/queries.json` with 10-15 test queries

**Requirements:**
```json
[
  {
    "query_id": "q1_table",
    "query": "What is the maximum operating pressure for 2-inch valve?",
    "ground_truth_answer": "10,000 PSI",
    "relevant_element_ids": ["doc1_table_3_2"],
    "query_type": "table",
    "difficulty": "easy"
  },
  {
    "query_id": "q2_keyword",
    "query": "API 6A valve pressure rating 1500 PSI",
    "ground_truth_answer": "...",
    "query_type": "general",
    "difficulty": "easy",
    "note": "Tests Weaviate keyword matching"
  },
  {
    "query_id": "q3_multihop",
    "query": "What safety procedures are required before maintenance on high-pressure valves?",
    "ground_truth_answer": "...",
    "query_type": "multi_hop",
    "difficulty": "hard",
    "note": "Tests FalkorDB graph traversal"
  },
  ...
]
```

**Query Types:**
- Table queries (specific data lookup)
- Keyword queries (exact term matching - tests Weaviate)
- Semantic queries (conceptual understanding)
- Multi-hop queries (cross-references - tests FalkorDB)
- Numerical queries (calculations)

**Acceptance Criteria:**
- ✅ 10-15 diverse queries
- ✅ Mix of difficulty levels
- ✅ Mix of query types
- ✅ Ground truth answers provided
- ✅ Tests all 3 storage approaches

### Agent 11: Docker & Deployment

**Task:** Create `docker-compose.yml` and deployment scripts

**Requirements:**
```yaml
version: '3.8'
services:
  chroma:
    image: chromadb/chroma:latest
    ports: ["8000:8000"]
    ...

  weaviate:
    image: semitechnologies/weaviate:1.24.4
    ports: ["8080:8080"]
    ...

  falkordb:
    image: falkordb/falkordb:latest
    ports: ["6379:6379"]
    ...
```

**Also:** Setup scripts, README instructions

**Acceptance Criteria:**
- ✅ All 3 services start successfully
- ✅ Services accessible on correct ports
- ✅ Data persists across restarts
- ✅ Clear documentation

---

## 🔗 Wave 2: Integration (Sequential)

**Duration:** 1.5 hours
**Agent:** 1-2
**Dependencies:** Requires all Wave 1 agents complete

### Task 2.1: Benchmark Runner

**File: `benchmark.py`**

```python
class BenchmarkRunner:
    def __init__(self):
        self.parsers = [
            LlamaParseParser(),
            DoclingParser(),
            PageIndexParser(),
            VertexDocAIParser(),
        ]

        self.storage_backends = [
            ChromaStore(),
            WeaviateStore(),
            FalkorDBStore(),
        ]

        self.embedder = UnifiedEmbedder()
        self.evaluator = Evaluator()

    async def run_full_benchmark(self, input_dir, queries_file):
        # For each document:
        #   1. Parse with all 4 parsers (parallel)
        #   2. Store in all 3 backends (parallel)
        #   3. Run queries against 12 combinations (parallel)
        #   4. Evaluate results
        # Save results to JSON/CSV
        ...
```

**Key:** This ties everything together

**Acceptance Criteria:**
- ✅ Orchestrates all 12 combinations
- ✅ Runs in parallel where possible
- ✅ Handles errors gracefully
- ✅ Saves results in structured format

### Task 2.2: Analysis & Visualization

**File: `analyze_results.py`**

```python
def load_results(): ...
def create_comparison_charts(results):
    # Heatmap: Parser × Storage → Accuracy
    # Bar charts: Parse time, retrieval time
    # Radar chart: Top 3 combinations
    ...
def print_winner(results): ...
def generate_report(results): ...
```

**Acceptance Criteria:**
- ✅ Loads results from benchmark
- ✅ Generates comparison charts
- ✅ Identifies winner
- ✅ Creates markdown report

### Task 2.3: Demo App

**File: `demo_app.py`**

```python
# Streamlit app that:
# 1. Shows benchmark results
# 2. Lets user query using winning combination
# 3. Displays sources
```

**Acceptance Criteria:**
- ✅ Displays winner with metrics
- ✅ Chat interface works
- ✅ Shows source attribution

---

## ✅ Wave 3: Testing & Validation (Parallel)

**Duration:** 1 hour
**Agents:** 2-3
**Dependencies:** Requires Wave 2 complete

### Agent Test-1: Integration Tests

**Task:** Write integration tests for each combination

```python
@pytest.mark.asyncio
async def test_llamaparse_chroma():
    # Parse with LlamaParse
    # Store in Chroma
    # Query and verify retrieval
    ...

# Repeat for all 12 combinations
```

**Acceptance Criteria:**
- ✅ All 12 combinations tested
- ✅ Tests pass

### Agent Test-2: End-to-End Test

**Task:** Run full benchmark with 1 sample PDF

```bash
# Add sample petroleum PDF to data/input/
python benchmark.py
python analyze_results.py
# Verify results look reasonable
```

**Acceptance Criteria:**
- ✅ Benchmark completes without errors
- ✅ All parsers successfully parse
- ✅ All storage backends work
- ✅ Results saved correctly
- ✅ Charts generated
- ✅ Winner identified

### Agent Test-3: Documentation

**Task:** Update README with:
- Setup instructions
- How to run benchmark
- How to interpret results
- Troubleshooting

**Acceptance Criteria:**
- ✅ Complete setup guide
- ✅ Clear execution instructions
- ✅ Examples included

---

## 📊 Execution Timeline

### Option A: Sequential (Traditional)
```
Foundation:    |████| 1h
Parser 1:      |████████| 2h
Parser 2:      |████████| 2h
Parser 3:      |████████| 2h
Parser 4:      |████████| 2h
Storage 1:     |██████████| 2.5h
Storage 2:     |██████████| 2.5h
Storage 3:     |██████████| 2.5h
Embeddings:    |████| 1h
Evaluation:    |████| 1h
Queries:       |██| 0.5h
Docker:        |██| 0.5h
Integration:   |██████| 1.5h
Testing:       |████| 1h
─────────────────────────────────
Total: ~22 hours (sequential)
```

### Option B: Parallel Agents (This Plan)
```
Wave 0 (Sequential):     |████| 1h
Wave 1 (11 Agents):      |████████████| 3h (parallel)
Wave 2 (Sequential):     |██████| 1.5h
Wave 3 (3 Agents):       |████| 1h (parallel)
─────────────────────────────────
Total: ~6.5 hours (mostly parallel)
Speedup: 3.4x
```

### Option C: Parallel Agents + Background Execution
```
Hour 0:   Launch Wave 0 agent → go do other work
Hour 1:   Launch 11 Wave 1 agents → go do other work
Hour 4:   Check agents, launch Wave 2 → go do other work
Hour 5.5: Launch Wave 3 agents → go do other work
Hour 6.5: Review results, done!

Human active time: ~1 hour
Wall clock time: ~6.5 hours
```

---

## 🎯 Parallel Execution Instructions

### Step 1: Launch Foundation (Sequential)

```bash
# Start 1 agent for Wave 0
claude --resume # or start new session

"Please complete Wave 0 from PARALLEL-EXECUTION-PLAN.md:
1. Create project structure
2. Implement base classes (parsers/base.py, storage/base.py)
3. Implement data models (models.py)
4. Implement config (config.py)

Work autonomously and let me know when complete."
```

**Wait for completion (~1 hour)**

### Step 2: Launch Wave 1 Agents (Parallel)

**In 11 separate Claude sessions:**

```bash
# Session 1
"Implement parsers/llamaparse_parser.py per Wave 1, Agent 1 spec in PARALLEL-EXECUTION-PLAN.md"

# Session 2
"Implement parsers/docling_parser.py per Wave 1, Agent 2 spec in PARALLEL-EXECUTION-PLAN.md"

# Session 3
"Implement parsers/pageindex_parser.py per Wave 1, Agent 3 spec in PARALLEL-EXECUTION-PLAN.md"

# Session 4
"Implement parsers/vertex_parser.py per Wave 1, Agent 4 spec in PARALLEL-EXECUTION-PLAN.md"

# Session 5
"Implement storage/chroma_store.py per Wave 1, Agent 5 spec in PARALLEL-EXECUTION-PLAN.md"

# Session 6
"Implement storage/weaviate_store.py per Wave 1, Agent 6 spec in PARALLEL-EXECUTION-PLAN.md"

# Session 7
"Implement storage/falkordb_store.py per Wave 1, Agent 7 spec in PARALLEL-EXECUTION-PLAN.md"

# Session 8
"Implement embeddings/embedder.py per Wave 1, Agent 8 spec in PARALLEL-EXECUTION-PLAN.md"

# Session 9
"Implement evaluation framework per Wave 1, Agent 9 spec in PARALLEL-EXECUTION-PLAN.md"

# Session 10
"Create evaluation/queries.json per Wave 1, Agent 10 spec in PARALLEL-EXECUTION-PLAN.md"

# Session 11
"Create docker-compose.yml per Wave 1, Agent 11 spec in PARALLEL-EXECUTION-PLAN.md"
```

**Wait for all 11 to complete (~3 hours)**

### Step 3: Launch Wave 2 (Sequential)

```bash
# Single session
"Complete Wave 2 from PARALLEL-EXECUTION-PLAN.md:
1. Implement benchmark.py
2. Implement analyze_results.py
3. Implement demo_app.py"
```

**Wait for completion (~1.5 hours)**

### Step 4: Launch Wave 3 (Parallel)

```bash
# 3 separate sessions
"Write integration tests per Wave 3, Agent Test-1"
"Run end-to-end test per Wave 3, Agent Test-2"
"Update documentation per Wave 3, Agent Test-3"
```

**Wait for completion (~1 hour)**

---

## 🚨 Key Success Factors

### 1. Clear Specifications
Each agent must have:
- ✅ Exact file to create
- ✅ Complete interface specification
- ✅ Example data structures
- ✅ Acceptance criteria

### 2. Minimal Dependencies
- Wave 0 defines ALL interfaces
- Wave 1 agents are independent
- Each can test with mock data

### 3. Communication Protocol
- Each agent creates a single module
- No cross-talk between agents
- Integration happens in Wave 2

### 4. Error Handling
- If an agent fails, restart just that agent
- Other agents continue
- No cascading failures

### 5. Testing Strategy
- Each agent writes unit tests
- Wave 3 adds integration tests
- Mock data for independent testing

---

## 📈 Optimization Opportunities

### Further Parallelization

**Wave 2 could be split:**
- Agent 2A: Benchmark runner
- Agent 2B: Analysis scripts
- Agent 2C: Demo app

**Saves another 30 minutes**

### Background Execution

All agents can run in background:
```bash
# Launch all 11 Wave 1 agents in background
for i in {1..11}; do
    claude --resume agent-$i &
done

# Come back in 3 hours, check results
```

### Caching & Reuse

- Base classes rarely change → cache
- Parsers are reusable across projects
- Storage implementations are reusable

---

## 🎁 Final Output

After 6.5 hours of parallel execution:

```
petroleum-rag-benchmark/
├── parsers/
│   ├── base.py ✅
│   ├── llamaparse_parser.py ✅
│   ├── docling_parser.py ✅
│   ├── pageindex_parser.py ✅
│   └── vertex_parser.py ✅
├── storage/
│   ├── base.py ✅
│   ├── chroma_store.py ✅
│   ├── weaviate_store.py ✅
│   └── falkordb_store.py ✅
├── embeddings/
│   └── embedder.py ✅
├── evaluation/
│   ├── metrics.py ✅
│   ├── evaluator.py ✅
│   └── queries.json ✅
├── benchmark.py ✅
├── analyze_results.py ✅
├── demo_app.py ✅
├── docker-compose.yml ✅
├── tests/ ✅
└── README.md ✅

Ready to run benchmark with:
python benchmark.py
```

---

## 💡 Next Steps

Would you like me to:
1. **Start Wave 0** - Set up foundation (1 hour)
2. **Launch Wave 1 agents** - All 11 in parallel (3 hours)
3. **Create agent coordination script** - Automate the launches

Or would you prefer to review the plan first and adjust the agent tasks?
