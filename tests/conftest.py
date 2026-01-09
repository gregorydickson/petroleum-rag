"""Shared pytest fixtures for all test modules.

This module provides common fixtures for:
- Mock parsers and storage backends
- Test data generation
- Docker service management
- Cleanup utilities
"""

import asyncio
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock, Mock, patch

import pytest
from PIL import Image, ImageDraw

from config import settings
from embeddings import UnifiedEmbedder
from models import (
    BenchmarkQuery,
    BenchmarkResult,
    DifficultyLevel,
    DocumentChunk,
    ElementType,
    ParsedDocument,
    ParsedElement,
    QueryType,
    RetrievalResult,
)


# ============================================================================
# Mock PDF Generation
# ============================================================================


@pytest.fixture
def mock_pdf_path(tmp_path: Path) -> Path:
    """Create a mock PDF file for testing.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Path to mock PDF file
    """
    pdf_path = tmp_path / "test_petroleum_doc.pdf"

    # Create a simple mock PDF-like file
    # Real parsers would need actual PDFs, but for mocking we just need a file
    pdf_path.write_text("Mock PDF content for testing")

    return pdf_path


@pytest.fixture
def mock_pdf_bytes() -> bytes:
    """Create mock PDF bytes for testing.

    Returns:
        Mock PDF file content as bytes
    """
    # Create a simple image that could represent a PDF page
    img = Image.new("RGB", (800, 1000), color="white")
    draw = ImageDraw.Draw(img)

    # Add some text-like content
    draw.text((50, 50), "Petroleum Engineering Document", fill="black")
    draw.text((50, 100), "Chapter 1: Reservoir Engineering", fill="black")
    draw.text((50, 150), "This is a test document about petroleum engineering.", fill="black")

    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    return img_bytes.getvalue()


# ============================================================================
# Mock Data Fixtures
# ============================================================================


@pytest.fixture
def mock_parsed_elements() -> list[ParsedElement]:
    """Create mock parsed elements for testing.

    Returns:
        List of ParsedElement objects
    """
    return [
        ParsedElement(
            element_id="elem_001",
            element_type=ElementType.HEADING,
            content="Introduction to Petroleum Engineering",
            page_number=1,
            metadata={"level": "1"},
        ),
        ParsedElement(
            element_id="elem_002",
            element_type=ElementType.TEXT,
            content="Petroleum reservoir engineering involves the study of fluid flow in porous media. This field combines principles from geology, physics, and engineering to optimize hydrocarbon recovery.",
            page_number=1,
            metadata={"section": "introduction"},
        ),
        ParsedElement(
            element_id="elem_003",
            element_type=ElementType.HEADING,
            content="Enhanced Oil Recovery",
            page_number=2,
            metadata={"level": "2"},
        ),
        ParsedElement(
            element_id="elem_004",
            element_type=ElementType.TEXT,
            content="Enhanced oil recovery (EOR) techniques include thermal recovery, gas injection, and chemical flooding. These methods are used to extract additional oil after primary and secondary recovery methods.",
            page_number=2,
            metadata={"section": "eor"},
        ),
        ParsedElement(
            element_id="elem_005",
            element_type=ElementType.TABLE,
            content="| Method | Recovery Factor | Cost |\n|--------|----------------|------|\n| Thermal | 40-50% | High |\n| Gas Injection | 30-40% | Medium |\n| Chemical | 35-45% | High |",
            formatted_content="<table>...</table>",
            page_number=2,
            metadata={"rows": "3", "cols": "3"},
        ),
    ]


@pytest.fixture
def mock_parsed_document(mock_pdf_path: Path, mock_parsed_elements: list[ParsedElement]) -> ParsedDocument:
    """Create mock parsed document for testing.

    Args:
        mock_pdf_path: Path to mock PDF file
        mock_parsed_elements: Mock parsed elements

    Returns:
        ParsedDocument object
    """
    return ParsedDocument(
        document_id="test_doc_001",
        source_file=mock_pdf_path,
        parser_name="MockParser",
        elements=mock_parsed_elements,
        metadata={"title": "Test Petroleum Document", "author": "Test Author"},
        parse_time_seconds=1.5,
        parsed_at=datetime.now(timezone.utc),
        total_pages=2,
    )


@pytest.fixture
def mock_chunks() -> list[DocumentChunk]:
    """Create mock document chunks for testing.

    Returns:
        List of DocumentChunk objects
    """
    return [
        DocumentChunk(
            chunk_id="chunk_001",
            document_id="doc_001",
            content="Petroleum reservoir engineering involves the study of fluid flow in porous media. This field combines principles from geology, physics, and engineering.",
            element_ids=["elem_001", "elem_002"],
            metadata={"source": "textbook", "section": "introduction"},
            chunk_index=0,
            start_page=1,
            end_page=1,
            token_count=30,
            parent_section="Introduction",
        ),
        DocumentChunk(
            chunk_id="chunk_002",
            document_id="doc_001",
            content="Enhanced oil recovery (EOR) techniques include thermal recovery, gas injection, and chemical flooding. These methods are used to extract additional oil.",
            element_ids=["elem_003", "elem_004"],
            metadata={"source": "textbook", "section": "eor"},
            chunk_index=1,
            start_page=2,
            end_page=2,
            token_count=28,
            parent_section="EOR",
        ),
        DocumentChunk(
            chunk_id="chunk_003",
            document_id="doc_001",
            content="EOR methods: Thermal (40-50% recovery, high cost), Gas Injection (30-40% recovery, medium cost), Chemical (35-45% recovery, high cost).",
            element_ids=["elem_005"],
            metadata={"source": "textbook", "section": "eor", "type": "table"},
            chunk_index=2,
            start_page=2,
            end_page=2,
            token_count=25,
            parent_section="EOR",
        ),
        DocumentChunk(
            chunk_id="chunk_004",
            document_id="doc_002",
            content="Drilling operations require careful planning and execution. Mud weight must be carefully controlled to prevent formation damage and wellbore instability.",
            element_ids=["elem_006"],
            metadata={"source": "manual", "section": "drilling"},
            chunk_index=0,
            start_page=1,
            end_page=1,
            token_count=26,
            parent_section="Drilling",
        ),
        DocumentChunk(
            chunk_id="chunk_005",
            document_id="doc_002",
            content="Well completion techniques vary based on reservoir characteristics. Common methods include cased-hole completions, open-hole completions, and gravel packs.",
            element_ids=["elem_007"],
            metadata={"source": "manual", "section": "completion"},
            chunk_index=1,
            start_page=2,
            end_page=2,
            token_count=24,
            parent_section="Completion",
        ),
    ]


@pytest.fixture
def mock_embeddings(mock_chunks: list[DocumentChunk]) -> list[list[float]]:
    """Create mock embeddings for chunks.

    Args:
        mock_chunks: Mock document chunks

    Returns:
        List of embedding vectors
    """
    # Create deterministic but varied embeddings for each chunk
    embeddings = []
    for i, chunk in enumerate(mock_chunks):
        # Create a 1536-dimensional vector (OpenAI embedding size)
        # Use different patterns for different chunks
        base = [0.1 * (i + 1)] * 1536
        # Add some variation based on chunk content length
        variation = len(chunk.content) % 10 / 100
        embedding = [val + variation for val in base]
        embeddings.append(embedding)

    return embeddings


@pytest.fixture
def mock_benchmark_queries() -> list[BenchmarkQuery]:
    """Create mock benchmark queries for testing.

    Returns:
        List of BenchmarkQuery objects
    """
    return [
        BenchmarkQuery(
            query_id="query_001",
            query="What is enhanced oil recovery?",
            ground_truth_answer="Enhanced oil recovery (EOR) techniques include thermal recovery, gas injection, and chemical flooding used to extract additional oil.",
            relevant_element_ids=["elem_003", "elem_004", "elem_005"],
            query_type=QueryType.SEMANTIC,
            difficulty=DifficultyLevel.EASY,
            expected_chunks=["chunk_002", "chunk_003"],
        ),
        BenchmarkQuery(
            query_id="query_002",
            query="What is the recovery factor for thermal EOR methods?",
            ground_truth_answer="Thermal EOR methods have a recovery factor of 40-50%.",
            relevant_element_ids=["elem_005"],
            query_type=QueryType.TABLE,
            difficulty=DifficultyLevel.MEDIUM,
            expected_chunks=["chunk_003"],
        ),
        BenchmarkQuery(
            query_id="query_003",
            query="Describe drilling operations and mud weight control.",
            ground_truth_answer="Drilling operations require careful planning. Mud weight must be controlled to prevent formation damage and wellbore instability.",
            relevant_element_ids=["elem_006"],
            query_type=QueryType.SEMANTIC,
            difficulty=DifficultyLevel.MEDIUM,
            expected_chunks=["chunk_004"],
        ),
    ]


@pytest.fixture
def mock_retrieval_results() -> list[RetrievalResult]:
    """Create mock retrieval results for testing.

    Returns:
        List of RetrievalResult objects
    """
    return [
        RetrievalResult(
            chunk_id="chunk_002",
            document_id="doc_001",
            content="Enhanced oil recovery (EOR) techniques include thermal recovery, gas injection, and chemical flooding.",
            score=0.92,
            metadata={"source": "textbook", "section": "eor"},
            rank=1,
            retrieval_method="vector",
        ),
        RetrievalResult(
            chunk_id="chunk_003",
            document_id="doc_001",
            content="EOR methods: Thermal (40-50% recovery, high cost), Gas Injection (30-40% recovery, medium cost).",
            score=0.85,
            metadata={"source": "textbook", "type": "table"},
            rank=2,
            retrieval_method="vector",
        ),
        RetrievalResult(
            chunk_id="chunk_001",
            document_id="doc_001",
            content="Petroleum reservoir engineering involves the study of fluid flow in porous media.",
            score=0.68,
            metadata={"source": "textbook", "section": "introduction"},
            rank=3,
            retrieval_method="vector",
        ),
    ]


@pytest.fixture
def mock_benchmark_result(
    mock_benchmark_queries: list[BenchmarkQuery],
    mock_retrieval_results: list[RetrievalResult],
) -> BenchmarkResult:
    """Create mock benchmark result for testing.

    Args:
        mock_benchmark_queries: Mock queries
        mock_retrieval_results: Mock retrieval results

    Returns:
        BenchmarkResult object
    """
    query = mock_benchmark_queries[0]

    return BenchmarkResult(
        benchmark_id="bench_001",
        parser_name="LlamaParse",
        storage_backend="ChromaStore",
        query_id=query.query_id,
        query=query.query,
        retrieved_results=mock_retrieval_results,
        generated_answer="Enhanced oil recovery (EOR) includes thermal, gas injection, and chemical methods.",
        ground_truth_answer=query.ground_truth_answer,
        metrics={
            "precision@5": 0.85,
            "recall@5": 0.90,
            "f1@5": 0.87,
            "mrr": 0.92,
            "ndcg@5": 0.88,
            "context_relevance": 0.80,
            "answer_correctness": 0.85,
            "faithfulness": 0.90,
        },
        retrieval_time_seconds=0.15,
        generation_time_seconds=0.45,
        total_time_seconds=0.60,
        timestamp=datetime.now(timezone.utc),
    )


# ============================================================================
# Mock Parser Fixtures
# ============================================================================


@pytest.fixture
def mock_llamaparse_parser(mock_parsed_document: ParsedDocument, mock_chunks: list[DocumentChunk]):
    """Create mock LlamaParse parser.

    Args:
        mock_parsed_document: Mock parsed document
        mock_chunks: Mock chunks

    Returns:
        Mock LlamaParseParser
    """
    with patch("parsers.llamaparse_parser.settings") as mock_settings:
        mock_settings.llama_cloud_api_key = "test-key"
        mock_settings.debug = False

        # Create mock parser
        parser = Mock()
        parser.name = "LlamaParse"
        parser.parse = AsyncMock(return_value=mock_parsed_document)
        parser.chunk_document = Mock(return_value=mock_chunks)

        return parser


@pytest.fixture
def mock_docling_parser(mock_parsed_document: ParsedDocument, mock_chunks: list[DocumentChunk]):
    """Create mock Docling parser.

    Args:
        mock_parsed_document: Mock parsed document
        mock_chunks: Mock chunks

    Returns:
        Mock DoclingParser
    """
    parser = Mock()
    parser.name = "Docling"
    parser.parse = AsyncMock(return_value=mock_parsed_document)
    parser.chunk_document = Mock(return_value=mock_chunks)

    return parser


@pytest.fixture
def mock_pageindex_parser(mock_parsed_document: ParsedDocument, mock_chunks: list[DocumentChunk]):
    """Create mock PageIndex parser.

    Args:
        mock_parsed_document: Mock parsed document
        mock_chunks: Mock chunks

    Returns:
        Mock PageIndexParser
    """
    parser = Mock()
    parser.name = "PageIndex"
    parser.parse = AsyncMock(return_value=mock_parsed_document)
    parser.chunk_document = Mock(return_value=mock_chunks)

    return parser


@pytest.fixture
def mock_vertex_parser(mock_parsed_document: ParsedDocument, mock_chunks: list[DocumentChunk]):
    """Create mock Vertex DocAI parser.

    Args:
        mock_parsed_document: Mock parsed document
        mock_chunks: Mock chunks

    Returns:
        Mock VertexDocAIParser
    """
    parser = Mock()
    parser.name = "VertexDocAI"
    parser.parse = AsyncMock(return_value=mock_parsed_document)
    parser.chunk_document = Mock(return_value=mock_chunks)

    return parser


# ============================================================================
# Mock Storage Fixtures
# ============================================================================


@pytest.fixture
async def mock_chroma_store(mock_retrieval_results: list[RetrievalResult]) -> AsyncGenerator:
    """Create mock ChromaStore.

    Args:
        mock_retrieval_results: Mock retrieval results

    Yields:
        Mock ChromaStore
    """
    store = Mock()
    store.name = "ChromaStore"
    store.initialize = AsyncMock()
    store.store_chunks = AsyncMock()
    store.retrieve = AsyncMock(return_value=mock_retrieval_results)
    store.clear = AsyncMock()
    store.health_check = AsyncMock(return_value=True)

    yield store


@pytest.fixture
async def mock_weaviate_store(mock_retrieval_results: list[RetrievalResult]) -> AsyncGenerator:
    """Create mock WeaviateStore.

    Args:
        mock_retrieval_results: Mock retrieval results

    Yields:
        Mock WeaviateStore
    """
    store = Mock()
    store.name = "WeaviateStore"
    store.initialize = AsyncMock()
    store.store_chunks = AsyncMock()
    store.retrieve = AsyncMock(return_value=mock_retrieval_results)
    store.clear = AsyncMock()
    store.health_check = AsyncMock(return_value=True)

    yield store


@pytest.fixture
async def mock_falkordb_store(mock_retrieval_results: list[RetrievalResult]) -> AsyncGenerator:
    """Create mock FalkorDBStore.

    Args:
        mock_retrieval_results: Mock retrieval results

    Yields:
        Mock FalkorDBStore
    """
    store = Mock()
    store.name = "FalkorDBStore"
    store.initialize = AsyncMock()
    store.store_chunks = AsyncMock()
    store.retrieve = AsyncMock(return_value=mock_retrieval_results)
    store.clear = AsyncMock()
    store.health_check = AsyncMock(return_value=True)

    yield store


# ============================================================================
# Mock Embedder Fixtures
# ============================================================================


@pytest.fixture
def mock_embedder(mock_embeddings: list[list[float]]):
    """Create mock UnifiedEmbedder.

    Args:
        mock_embeddings: Mock embeddings

    Returns:
        Mock UnifiedEmbedder
    """
    embedder = Mock()
    embedder.embed_texts = AsyncMock(return_value=mock_embeddings)
    embedder.embed_query = AsyncMock(return_value=mock_embeddings[0])

    return embedder


# ============================================================================
# Cleanup Fixtures
# ============================================================================


@pytest.fixture
def cleanup_test_files(tmp_path: Path):
    """Cleanup fixture that removes test files after tests.

    Args:
        tmp_path: Pytest temporary directory

    Yields:
        Temporary path
    """
    yield tmp_path

    # Cleanup logic runs after test
    for file in tmp_path.glob("*"):
        if file.is_file():
            file.unlink()


# ============================================================================
# Docker Service Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def docker_services_available() -> bool:
    """Check if Docker services are available.

    Returns:
        True if Docker services can be used, False otherwise
    """
    try:
        import subprocess

        result = subprocess.run(
            ["docker", "ps"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


@pytest.fixture(scope="session")
def skip_if_no_docker(docker_services_available: bool):
    """Skip test if Docker is not available.

    Args:
        docker_services_available: Docker availability check
    """
    if not docker_services_available:
        pytest.skip("Docker services not available")


# ============================================================================
# Settings Override Fixtures
# ============================================================================


@pytest.fixture
def test_settings():
    """Override settings for testing.

    Returns:
        Test settings configuration
    """
    return {
        "chunk_size": 500,
        "chunk_overlap": 100,
        "retrieval_top_k": 3,
        "retrieval_min_score": 0.5,
        "embedding_batch_size": 10,
        "log_level": "DEBUG",
    }
