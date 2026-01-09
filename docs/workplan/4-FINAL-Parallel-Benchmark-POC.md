# Technical Document RAG - FINAL Parallel Benchmark POC
## One-Shot Evaluation: 4 Parsers × 3 Storage = 12 Combinations

**Goal:** Test ALL major approaches in parallel to identify the best parser+storage combination for petroleum technical documents.

**Success Criteria:**
1. Parse same documents with 4 parsers simultaneously
2. Store in 3 different storage backends simultaneously
3. Run identical test queries against all 12 combinations
4. Quantitatively compare accuracy, performance, cost
5. Pick winning combination
6. Build simple demo UI with winner

**Timeline:** 1 day for benchmark + analysis

---

## Final Technology Stack

### Parsers (4) - Diverse Approaches

1. **LlamaParse**
   - Commercial, cloud API
   - Specialized for technical documents with tables/diagrams
   - Proven track record

2. **Docling** (IBM) ⭐ NEW
   - Open source, designed for scientific/technical documents
   - Excellent table extraction and layout preservation
   - Handles equations/formulas
   - Fast, local processing

3. **PageIndex** (VectifyAI)
   - Novel semantic chunking approach
   - Preserves document structure and relationships
   - Community-recommended for technical docs

4. **Vertex Document AI** (Google)
   - Enterprise-grade OCR and layout analysis
   - GCP native (good for deployment)
   - Baseline for comparison

**Why these 4:**
- Different underlying technologies (ML models, rule-based, hybrid)
- Mix of commercial and open source
- Mix of cloud API and local processing
- All proven for technical/scientific documents

### Storage Backends (3) - Different Retrieval Approaches

1. **Chroma**
   - Pure vector similarity search (baseline)
   - Simple, lightweight, easy to run locally
   - Good for semantic search

2. **Weaviate**
   - Hybrid search: Vector + Keyword (BM25)
   - Better for queries with specific technical terms/numbers
   - Different retrieval approach from pure vector

3. **FalkorDB**
   - Graph database (multi-hop reasoning)
   - Redis-compatible, lightweight
   - Good for cross-reference queries and relationships
   - Better for "why" and "how" questions that span sections

**Why these 3:**
- Represent 3 fundamentally different retrieval approaches
- All can run locally via Docker
- Cover different query patterns (semantic, exact-match, relational)

### Total: **12 Combinations**

```
LlamaParse + Chroma
LlamaParse + Weaviate
LlamaParse + FalkorDB
Docling + Chroma
Docling + Weaviate
Docling + FalkorDB
PageIndex + Chroma
PageIndex + Weaviate
PageIndex + FalkorDB
Vertex + Chroma
Vertex + Weaviate
Vertex + FalkorDB
```

---

## Architecture

```
                    ┌─────────────────────────────────┐
                    │   Test PDFs (3-5 documents)     │
                    └──────────┬──────────────────────┘
                               │
                ┌──────────────┼──────────────┬──────────────┐
                │              │              │              │
                ▼              ▼              ▼              ▼
        ┌──────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
        │  LlamaParse  │ │ Docling  │ │PageIndex │ │  Vertex AI   │
        │  (Cloud)     │ │  (Local) │ │ (Cloud?) │ │   (Cloud)    │
        └──────┬───────┘ └────┬─────┘ └────┬─────┘ └──────┬───────┘
               │              │            │              │
               └──────────────┼────────────┼──────────────┘
                              │
                    ┌─────────▼────────────┐
                    │ Normalized Format    │
                    │ (ParsedDocument)     │
                    └─────────┬────────────┘
                              │
                ┌─────────────┼─────────────┬─────────────┐
                │             │             │             │
                ▼             ▼             ▼             ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
        │  Chroma  │  │ Weaviate │  │ FalkorDB │  │          │
        │ (Vector) │  │ (Hybrid) │  │  (Graph) │  │  (Drop   │
        │          │  │          │  │          │  │  Qdrant) │
        └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────────┘
             │             │             │
             └─────────────┼─────────────┘
                           │
                  ┌────────▼─────────────┐
                  │ Unified Query API    │
                  │ (Test all 12 combos) │
                  └────────┬─────────────┘
                           │
                           ▼
                  ┌────────────────────────────┐
                  │ Evaluation & Comparison    │
                  │ - Accuracy metrics         │
                  │ - Performance metrics      │
                  │ - Cost analysis            │
                  └────────┬───────────────────┘
                           │
                           ▼
                  ┌────────────────────────────┐
                  │ Winner Selection + Demo UI │
                  └────────────────────────────┘
```

---

## Project Structure

```
petroleum-rag-benchmark/
├── README.md
├── pyproject.toml
├── .env.example
├── docker-compose.yml           # Chroma, Weaviate, FalkorDB
├── config.py                    # Single config
├── models.py                    # Shared data models
├── parsers/
│   ├── __init__.py
│   ├── base.py
│   ├── llamaparse_parser.py
│   ├── docling_parser.py       # ⭐ NEW
│   ├── pageindex_parser.py
│   └── vertex_parser.py
├── storage/
│   ├── __init__.py
│   ├── base.py
│   ├── chroma_store.py
│   ├── weaviate_store.py       # ⭐ NEW
│   └── falkordb_store.py
├── embeddings/
│   ├── __init__.py
│   └── embedder.py
├── evaluation/
│   ├── __init__.py
│   ├── queries.json             # Test queries
│   ├── metrics.py
│   └── evaluator.py
├── benchmark.py                 # MAIN: Run all 12 combinations
├── analyze_results.py           # Generate charts and report
├── demo_app.py                  # Streamlit demo with winner
├── data/
│   ├── input/                   # Test PDFs
│   ├── parsed/                  # Parser outputs
│   │   ├── llamaparse/
│   │   ├── docling/
│   │   ├── pageindex/
│   │   └── vertex/
│   └── results/                 # Benchmark results
│       ├── raw_results.json
│       ├── comparison.csv
│       └── charts/
└── tests/
    ├── test_parsers.py
    └── test_storage.py
```

---

## Dependencies (pyproject.toml)

```toml
[project]
name = "petroleum-rag-benchmark"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    # Parsers
    "llama-parse>=0.5.0",
    "docling>=1.0.0",                    # ⭐ NEW
    "pageindex>=0.1.0",
    "google-cloud-documentai>=2.35.0",

    # Storage
    "chromadb>=0.5.0",
    "weaviate-client>=4.4.0",            # ⭐ NEW
    "falkordb>=1.0.0",

    # Embeddings & LLM
    "anthropic>=0.39.0",
    "openai>=1.55.0",
    "sentence-transformers>=3.3.0",

    # Utilities
    "pydantic>=2.10.0",
    "pydantic-settings>=2.6.0",
    "python-dotenv>=1.0.0",
    "httpx>=0.28.0",
    "tenacity>=9.0.0",

    # Analysis
    "pandas>=2.2.0",
    "matplotlib>=3.9.0",
    "seaborn>=0.13.0",

    # Demo UI
    "streamlit>=1.40.0",

    # Testing
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
]
```

---

## Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  # Chroma - Vector Database
  chroma:
    image: chromadb/chroma:latest
    ports:
      - "8000:8000"
    volumes:
      - chroma_data:/chroma/chroma
    environment:
      - IS_PERSISTENT=TRUE
      - ANONYMIZED_TELEMETRY=FALSE

  # Weaviate - Hybrid Search (Vector + Keyword)
  weaviate:
    image: semitechnologies/weaviate:1.24.4
    ports:
      - "8080:8080"
    environment:
      - AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true
      - PERSISTENCE_DATA_PATH=/var/lib/weaviate
      - DEFAULT_VECTORIZER_MODULE=none
      - ENABLE_MODULES=text2vec-openai
      - QUERY_DEFAULTS_LIMIT=25
    volumes:
      - weaviate_data:/var/lib/weaviate

  # FalkorDB - Graph Database
  falkordb:
    image: falkordb/falkordb:latest
    ports:
      - "6379:6379"
    volumes:
      - falkordb_data:/data
    command: ["redis-server", "--loadmodule", "/usr/lib/redis/modules/falkordb.so"]

volumes:
  chroma_data:
  weaviate_data:
  falkordb_data:
```

---

## Key Implementation: Docling Parser

```python
# parsers/docling_parser.py
import time
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

from models import ParsedDocument, ParsedElement, DocumentChunk
from parsers.base import BaseParser
from config import settings


class DoclingParser(BaseParser):
    """
    Docling parser - IBM's open-source technical document parser.

    Optimized for:
    - Scientific/technical documents
    - Complex tables
    - Multi-column layouts
    - Equations and formulas
    - Hierarchical structure
    """

    def __init__(self):
        super().__init__("Docling")

        # Configure pipeline for technical documents
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = True  # Extract table structure
        pipeline_options.do_ocr = True              # OCR for scanned docs
        pipeline_options.generate_page_images = False  # Save memory

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: pipeline_options
            }
        )

    async def parse(self, file_path: Path) -> ParsedDocument:
        """Parse PDF using Docling."""
        start_time = time.time()

        # Convert document
        result = self.converter.convert(str(file_path))
        document = result.document

        # Extract elements
        elements = []
        table_count = 0
        figure_count = 0

        # Process each page
        for page_idx, page in enumerate(document.pages, 1):
            # Extract text blocks
            for block in page.blocks:
                element_type = self._map_block_type(block.block_type)

                element = ParsedElement(
                    element_id=f"{file_path.stem}_docling_{page_idx}_{block.id}",
                    element_type=element_type,
                    content=block.text,
                    page_number=page_idx,
                    section_hierarchy=self._extract_hierarchy(block),
                    metadata={
                        "block_type": block.block_type,
                        "confidence": getattr(block, 'confidence', None),
                        "bbox": self._extract_bbox(block),
                    }
                )

                # Handle tables specially
                if element_type == "table":
                    table_count += 1
                    # Docling preserves table structure excellently
                    element.table_markdown = block.to_markdown() if hasattr(block, 'to_markdown') else block.text
                    element.table_html = block.to_html() if hasattr(block, 'to_html') else None

                # Handle figures
                elif element_type == "figure":
                    figure_count += 1
                    element.figure_caption = self._extract_caption(block)

                elements.append(element)

        parse_time = time.time() - start_time

        return ParsedDocument(
            document_id=file_path.stem,
            source_file=file_path,
            parser_name=self.name,
            elements=elements,
            total_pages=len(document.pages),
            table_count=table_count,
            figure_count=figure_count,
            parse_time_seconds=parse_time,
            raw_output=document.to_dict() if hasattr(document, 'to_dict') else None
        )

    def chunk_document(self, doc: ParsedDocument) -> list[DocumentChunk]:
        """
        Chunk Docling output.

        Docling maintains semantic structure, so we can chunk
        intelligently based on sections and blocks.
        """
        chunks = []
        current_chunk = []
        current_section = []
        chunk_index = 0

        for element in doc.elements:
            # Start new chunk on headings or when chunk gets large
            if element.element_type == "heading" or len('\n'.join(current_chunk)) > settings.chunk_size:
                # Save current chunk if it has content
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk)
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{doc.document_id}_{self.name}_chunk_{chunk_index}",
                            document_id=doc.document_id,
                            parser_name=self.name,
                            content=chunk_text,
                            chunk_index=chunk_index,
                            page_number=element.page_number,
                            chunk_type="text",
                            section_hierarchy=current_section.copy(),
                            metadata={"source_parser": "docling"}
                        )
                    )
                    chunk_index += 1
                    current_chunk = []

                # Update section hierarchy on headings
                if element.element_type == "heading":
                    current_section = element.section_hierarchy

            # Tables get their own chunks (don't split)
            if element.element_type == "table":
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc.document_id}_{self.name}_chunk_{chunk_index}",
                        document_id=doc.document_id,
                        parser_name=self.name,
                        content=element.content,
                        chunk_index=chunk_index,
                        page_number=element.page_number,
                        chunk_type="table",
                        section_hierarchy=element.section_hierarchy,
                        metadata={
                            "source_parser": "docling",
                            "table_markdown": element.table_markdown,
                            "table_html": element.table_html,
                        }
                    )
                )
                chunk_index += 1
            else:
                current_chunk.append(element.content)

        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc.document_id}_{self.name}_chunk_{chunk_index}",
                    document_id=doc.document_id,
                    parser_name=self.name,
                    content=chunk_text,
                    chunk_index=chunk_index,
                    section_hierarchy=current_section,
                    chunk_type="text",
                    metadata={"source_parser": "docling"}
                )
            )

        return chunks

    def _map_block_type(self, block_type: str) -> str:
        """Map Docling block types to our element types."""
        mapping = {
            "text": "text",
            "title": "heading",
            "section_header": "heading",
            "paragraph": "text",
            "table": "table",
            "figure": "figure",
            "list": "text",
            "equation": "code",  # Treat equations as code blocks
            "caption": "text",
        }
        return mapping.get(block_type.lower(), "text")

    def _extract_hierarchy(self, block) -> list[str]:
        """Extract section hierarchy from Docling block."""
        hierarchy = []
        if hasattr(block, 'section_path'):
            hierarchy = block.section_path
        elif hasattr(block, 'parent_sections'):
            hierarchy = block.parent_sections
        return hierarchy

    def _extract_bbox(self, block) -> dict | None:
        """Extract bounding box if available."""
        if hasattr(block, 'bbox'):
            bbox = block.bbox
            return {
                "x1": bbox.x1,
                "y1": bbox.y1,
                "x2": bbox.x2,
                "y2": bbox.y2,
            }
        return None

    def _extract_caption(self, block) -> str | None:
        """Extract figure caption."""
        if hasattr(block, 'caption'):
            return block.caption
        # Docling often puts captions in adjacent blocks
        return None
```

---

## Key Implementation: Weaviate Store

```python
# storage/weaviate_store.py
import logging
import weaviate
import weaviate.classes as wvc
from weaviate.classes.query import Filter

from models import DocumentChunk, RetrievalResult
from storage.base import BaseStorage
from config import settings

logger = logging.getLogger(__name__)


class WeaviateStore(BaseStorage):
    """
    Weaviate storage - Hybrid search (Vector + Keyword).

    Key advantage: Combines semantic similarity (vector) with
    exact keyword matching (BM25), weighted by alpha parameter.

    Good for technical queries where specific terms matter:
    - "valve pressure rating 1500 PSI"
    - "corrosion resistance Inconel 625"
    - "API 6A section 3.2"
    """

    def __init__(self):
        super().__init__("Weaviate")

        # Connect to Weaviate
        self.client = weaviate.connect_to_local(
            host=settings.weaviate_host,
            port=settings.weaviate_port,
        )

        self.collection_name = "PetroleumDocument"
        self.collection = None

    async def initialize(self):
        """Initialize Weaviate collection."""
        try:
            # Check if collection exists
            if self.client.collections.exists(self.collection_name):
                self.collection = self.client.collections.get(self.collection_name)
                logger.info(f"Using existing collection: {self.collection_name}")
            else:
                # Create collection
                self.collection = self.client.collections.create(
                    name=self.collection_name,
                    vectorizer_config=wvc.config.Configure.Vectorizer.none(),  # We provide embeddings
                    vector_index_config=wvc.config.Configure.VectorIndex.hnsw(),
                    properties=[
                        wvc.config.Property(
                            name="content",
                            data_type=wvc.config.DataType.TEXT,
                            skip_vectorization=False,  # Include in BM25 search
                        ),
                        wvc.config.Property(
                            name="chunk_id",
                            data_type=wvc.config.DataType.TEXT,
                        ),
                        wvc.config.Property(
                            name="document_id",
                            data_type=wvc.config.DataType.TEXT,
                        ),
                        wvc.config.Property(
                            name="parser_name",
                            data_type=wvc.config.DataType.TEXT,
                        ),
                        wvc.config.Property(
                            name="chunk_index",
                            data_type=wvc.config.DataType.INT,
                        ),
                        wvc.config.Property(
                            name="page_number",
                            data_type=wvc.config.DataType.INT,
                        ),
                        wvc.config.Property(
                            name="chunk_type",
                            data_type=wvc.config.DataType.TEXT,
                        ),
                        wvc.config.Property(
                            name="section_hierarchy",
                            data_type=wvc.config.DataType.TEXT,
                        ),
                    ]
                )
                logger.info(f"Created collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Error initializing Weaviate: {e}")
            raise

    async def store_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]):
        """Store chunks with embeddings in Weaviate."""
        logger.info(f"Storing {len(chunks)} chunks in Weaviate")

        try:
            # Batch insert
            with self.collection.batch.dynamic() as batch:
                for chunk, embedding in zip(chunks, embeddings):
                    batch.add_object(
                        properties={
                            "content": chunk.content,
                            "chunk_id": chunk.chunk_id,
                            "document_id": chunk.document_id,
                            "parser_name": chunk.parser_name,
                            "chunk_index": chunk.chunk_index,
                            "page_number": chunk.page_number or 0,
                            "chunk_type": chunk.chunk_type,
                            "section_hierarchy": " > ".join(chunk.section_hierarchy),
                        },
                        vector=embedding
                    )

            logger.info(f"Successfully stored {len(chunks)} chunks")

        except Exception as e:
            logger.error(f"Error storing chunks: {e}")
            raise

    async def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 5
    ) -> list[RetrievalResult]:
        """
        Retrieve using hybrid search (vector + keyword).

        Alpha parameter controls weighting:
        - alpha=1.0: Pure vector search
        - alpha=0.0: Pure keyword (BM25) search
        - alpha=0.7: Balanced (70% vector, 30% keyword) - RECOMMENDED
        """
        try:
            # Hybrid search
            response = self.collection.query.hybrid(
                query=query,  # For BM25 keyword matching
                vector=query_embedding,  # For vector similarity
                alpha=0.7,  # 70% vector, 30% keyword (tunable)
                limit=top_k,
                return_metadata=wvc.query.MetadataQuery(score=True, distance=True)
            )

            # Convert to RetrievalResult
            results = []
            for idx, obj in enumerate(response.objects):
                chunk = DocumentChunk(
                    chunk_id=obj.properties["chunk_id"],
                    document_id=obj.properties["document_id"],
                    parser_name=obj.properties["parser_name"],
                    content=obj.properties["content"],
                    chunk_index=obj.properties["chunk_index"],
                    page_number=obj.properties["page_number"],
                    chunk_type=obj.properties["chunk_type"],
                    section_hierarchy=obj.properties["section_hierarchy"].split(" > "),
                )

                results.append(
                    RetrievalResult(
                        chunk=chunk,
                        score=obj.metadata.score if obj.metadata else 0.0,
                        rank=idx,
                        storage_backend=self.name,
                        metadata={
                            "distance": obj.metadata.distance if obj.metadata else None,
                            "search_type": "hybrid",
                        }
                    )
                )

            logger.info(f"Retrieved {len(results)} chunks with hybrid search")
            return results

        except Exception as e:
            logger.error(f"Error retrieving: {e}")
            raise

    async def clear(self):
        """Clear all data from collection."""
        try:
            self.client.collections.delete(self.collection_name)
            await self.initialize()  # Recreate
            logger.info("Cleared Weaviate collection")
        except Exception as e:
            logger.error(f"Error clearing: {e}")
            raise

    def __del__(self):
        """Close connection on cleanup."""
        if hasattr(self, 'client'):
            self.client.close()
```

---

## Updated Benchmark Runner

```python
# benchmark.py (key section)

class BenchmarkRunner:
    """Runs all 12 parser+storage combinations."""

    def __init__(self):
        self.parsers = [
            LlamaParseParser(),
            DoclingParser(),        # ⭐ NEW
            PageIndexParser(),
            VertexDocAIParser(),
        ]

        self.storage_backends = [
            ChromaStore(),
            WeaviateStore(),        # ⭐ NEW (hybrid search)
            FalkorDBStore(),
        ]

        self.embedder = UnifiedEmbedder()
        self.evaluator = Evaluator()
        self.metrics_calculator = MetricsCalculator()

        self.results = []

    async def run_full_benchmark(self, input_dir: Path, queries_file: Path):
        """Run complete benchmark - all 12 combinations."""
        logger.info("="*80)
        logger.info("TESTING 12 COMBINATIONS:")
        logger.info("  Parsers: LlamaParse, Docling, PageIndex, Vertex")
        logger.info("  Storage: Chroma, Weaviate (hybrid), FalkorDB (graph)")
        logger.info("="*80)

        # ... rest of implementation from previous workplan ...
```

---

## Test Queries for Petroleum Documents

```json
// evaluation/queries.json
[
  {
    "query_id": "q1_table",
    "query": "What is the maximum operating pressure for 2-inch valve in section 3.2?",
    "ground_truth_answer": "10,000 PSI",
    "relevant_element_ids": ["doc1_table_3_2"],
    "query_type": "table",
    "difficulty": "easy"
  },
  {
    "query_id": "q2_keyword",
    "query": "API 6A valve pressure rating 1500 PSI",
    "ground_truth_answer": "2-inch valve rated for 1500 PSI per API 6A standard",
    "relevant_element_ids": ["doc1_table_3_2", "doc1_section_3_1"],
    "query_type": "general",
    "difficulty": "easy",
    "note": "Tests Weaviate's keyword matching - should find exact 'API 6A' and '1500 PSI'"
  },
  {
    "query_id": "q3_figure",
    "query": "According to the corrosion resistance chart, which material is best for H2S?",
    "ground_truth_answer": "Inconel 625",
    "relevant_element_ids": ["doc1_figure_5"],
    "query_type": "figure",
    "difficulty": "medium"
  },
  {
    "query_id": "q4_multihop",
    "query": "What safety procedures must be followed before maintenance on valves rated above 5000 PSI?",
    "ground_truth_answer": "1) Depressurize system, 2) Lock out energy sources, 3) Verify zero pressure, 4) Wear PPE per section 8.3",
    "relevant_element_ids": ["doc1_section_7_3", "doc1_section_8_3", "doc1_table_7_1"],
    "query_type": "multi_hop",
    "difficulty": "hard",
    "note": "Tests FalkorDB graph traversal - requires following references across sections"
  },
  {
    "query_id": "q5_numerical",
    "query": "Calculate flow rate in barrels per day given pressure drop 50 PSI across 2-inch pipe",
    "ground_truth_answer": "1,234 BPD using Darcy-Weisbach equation from section 4.2",
    "relevant_element_ids": ["doc1_section_4_2", "doc1_formula_4_1"],
    "query_type": "numerical",
    "difficulty": "hard"
  },
  {
    "query_id": "q6_semantic",
    "query": "What should I do if I see rust on the valve body?",
    "ground_truth_answer": "Inspect for corrosion per section 6, may require replacement if pitting exceeds 10% wall thickness",
    "relevant_element_ids": ["doc1_section_6_2", "doc1_section_6_3"],
    "query_type": "general",
    "difficulty": "medium",
    "note": "Tests pure semantic search - 'rust' not mentioned, needs to understand means 'corrosion'"
  },
  {
    "query_id": "q7_cross_reference",
    "query": "Section 3.2 references a torque specification - what is it?",
    "ground_truth_answer": "75 ft-lbs per Table 7.1",
    "relevant_element_ids": ["doc1_section_3_2", "doc1_table_7_1"],
    "query_type": "multi_hop",
    "difficulty": "medium",
    "note": "Tests graph traversal - follows cross-reference from section to table"
  }
]
```

---

## Execution Instructions

### 1. Setup (30 minutes)

```bash
# Clone/create project
mkdir petroleum-rag-benchmark && cd petroleum-rag-benchmark

# Python environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install llama-parse docling pageindex \
    google-cloud-documentai \
    chromadb weaviate-client falkordb \
    anthropic openai sentence-transformers \
    pandas matplotlib seaborn streamlit \
    python-dotenv pydantic pydantic-settings

# Start Docker services
docker-compose up -d

# Wait for services to be ready
sleep 10

# Verify services
curl http://localhost:8000/api/v1/heartbeat  # Chroma
curl http://localhost:8080/v1/.well-known/ready  # Weaviate
redis-cli -p 6379 ping  # FalkorDB

# Configure environment
cp .env.example .env
# Edit .env with your API keys:
#   ANTHROPIC_API_KEY=...
#   LLAMA_CLOUD_API_KEY=...
#   OPENAI_API_KEY=...
#   GOOGLE_APPLICATION_CREDENTIALS=...
```

### 2. Prepare Test Data (30 minutes)

```bash
# Add 3-5 petroleum technical PDFs
mkdir -p data/input
# Copy PDFs: API standards, valve specs, technical manuals, etc.

# Create test queries
# Edit evaluation/queries.json with 10-15 questions
# Include:
#   - Table queries (specific data lookup)
#   - Keyword queries (exact term matching)
#   - Semantic queries (conceptual understanding)
#   - Multi-hop queries (cross-reference following)
#   - Numerical queries (calculations)
```

### 3. Run Benchmark (2-4 hours, automated)

```bash
# Run full benchmark
python benchmark.py

# This will:
# 1. Parse all PDFs with 4 parsers (parallel)      ~15-30 min
# 2. Store in 3 storage backends (parallel)        ~10-20 min
# 3. Run queries against 12 combinations (parallel) ~1-3 hours
# 4. Calculate metrics for each                     ~5 min
# 5. Save results

# Monitor progress
tail -f logs/benchmark.log

# Expected output:
# ✓ LlamaParse: 245 elements, 12 tables, 3.2s
# ✓ Docling: 312 elements, 14 tables, 2.1s
# ✓ PageIndex: 198 chunks, 1.8s
# ✓ Vertex: 287 elements, 11 tables, 2.7s
#
# Stored in 3 backends × 4 parsers = 12 combinations
#
# Running 15 queries × 12 combinations = 180 total queries
# Progress: [████████████████] 180/180
#
# Results saved to data/results/
```

### 4. Analyze Results (5 minutes)

```bash
# Generate comparison charts and report
python analyze_results.py

# Output:
# ✓ Charts saved to data/results/charts/
# ✓ Report saved to data/results/REPORT.md
#
# 🏆 WINNER: Docling + Weaviate
#    Overall Score: 0.847
#    Answer Correctness: 0.892
#    Precision@5: 0.823

# View results
cat data/results/REPORT.md
open data/results/charts/heatmap_correctness.png
```

### 5. Run Demo (immediate)

```bash
# Launch demo with winning combination
streamlit run demo_app.py

# Opens browser at http://localhost:8501
# Demo uses winning combination (e.g., Docling + Weaviate)
```

---

## Expected Results Analysis

### Likely Winners by Category

**Best for Tables:**
- Parser: Docling or AWS Textract (if added)
- Storage: Weaviate (exact term matching helps find table content)

**Best for Semantic Search:**
- Parser: LlamaParse or Docling
- Storage: Chroma (pure vector similarity)

**Best for Multi-hop Questions:**
- Parser: Any (depends on chunking)
- Storage: FalkorDB (graph traversal)

**Best for Technical Terms:**
- Parser: Docling (preserves structure)
- Storage: Weaviate (keyword matching finds exact terms)

**Best Overall:**
- Likely: **Docling + Weaviate**
- Why: Excellent parsing + hybrid search

---

## Cost Estimate

### Per Document (100 pages)

**Parsing:**
- LlamaParse: ~$0.50/doc (cloud API)
- Docling: Free (local)
- PageIndex: ~$0.30/doc (if cloud API)
- Vertex DocAI: ~$1.50/doc (GCP pricing)

**Embeddings (OpenAI text-embedding-3-small):**
- ~500 chunks × $0.00002/1K tokens = ~$0.05/doc

**Claude queries (15 queries × 4K tokens avg):**
- ~$0.45 total

**Total for 5 documents:**
- ~$15-25 for complete benchmark

---

## Next Steps After Benchmark

Based on results:

1. **Clear Winner (>0.85 score):**
   - Build production app with winner
   - Deploy to GCP Cloud Run
   - Add features (multi-doc, chat history)

2. **Close Race (multiple >0.80):**
   - Consider hybrid: best parser + best storage
   - Or: offer both options to users

3. **No Clear Winner (<0.75 all):**
   - Review test queries (are they too hard?)
   - Try different chunking strategies
   - Consider adding more parsers (AWS Textract, etc.)

---

## Summary

**12 Combinations:**
- 4 Parsers: LlamaParse, Docling, PageIndex, Vertex
- 3 Storage: Chroma, Weaviate, FalkorDB

**1 Day Timeline:**
- Setup: 1 hour
- Benchmark: 3-4 hours (automated)
- Analysis: 30 minutes
- Demo: Built automatically

**Output:**
- Quantitative comparison with charts
- Clear winner identified
- Working demo with best combination
- Foundation for production app

Ready to start implementing?
