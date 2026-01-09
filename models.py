"""Data models for the petroleum RAG benchmark system.

This module defines all core data structures used across parsers, storage backends,
and evaluation components.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Literal


class ElementType(str, Enum):
    """Types of elements extracted from documents."""

    TEXT = "text"
    HEADING = "heading"
    TABLE = "table"
    FIGURE = "figure"
    LIST = "list"
    CODE = "code"
    EQUATION = "equation"
    CAPTION = "caption"
    FOOTER = "footer"
    HEADER = "header"


class QueryType(str, Enum):
    """Types of benchmark queries."""

    TABLE = "table"  # Queries targeting table data
    KEYWORD = "keyword"  # Exact keyword matching
    SEMANTIC = "semantic"  # Conceptual understanding
    MULTI_HOP = "multi_hop"  # Requires multiple document sections
    NUMERICAL = "numerical"  # Calculations or numerical comparisons
    GENERAL = "general"  # General information retrieval


class DifficultyLevel(str, Enum):
    """Difficulty levels for benchmark queries."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class ParsedElement:
    """A single element extracted from a document.

    Attributes:
        element_id: Unique identifier (e.g., "doc1_table_3_2")
        element_type: Type of element (text, table, figure, etc.)
        content: Raw content as string
        formatted_content: Optional formatted version (HTML, Markdown)
        metadata: Additional element-specific metadata
        page_number: Page number where element appears
        bbox: Optional bounding box coordinates [x1, y1, x2, y2]
        parent_section: Optional parent section identifier
    """

    element_id: str
    element_type: ElementType
    content: str
    formatted_content: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    page_number: int | None = None
    bbox: list[float] | None = None
    parent_section: str | None = None


@dataclass
class ParsedDocument:
    """A complete parsed document with all extracted elements.

    Attributes:
        document_id: Unique document identifier
        source_file: Path to original file
        parser_name: Name of parser used
        elements: List of all extracted elements
        metadata: Document-level metadata
        parse_time_seconds: Time taken to parse
        parsed_at: Timestamp of parsing
        total_pages: Total number of pages
        error: Optional error message if parsing failed
    """

    document_id: str
    source_file: Path
    parser_name: str
    elements: list[ParsedElement]
    metadata: dict[str, str] = field(default_factory=dict)
    parse_time_seconds: float = 0.0
    parsed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_pages: int | None = None
    error: str | None = None


@dataclass
class DocumentChunk:
    """A chunk of document content for RAG storage and retrieval.

    Attributes:
        chunk_id: Unique chunk identifier
        document_id: Parent document identifier
        content: Text content of the chunk
        element_ids: List of element IDs included in this chunk
        metadata: Chunk-level metadata
        chunk_index: Sequential index within document
        start_page: Starting page number
        end_page: Ending page number
        token_count: Approximate token count
        parent_section: Section this chunk belongs to
    """

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


@dataclass
class RetrievalResult:
    """A single retrieval result from a storage backend.

    Attributes:
        chunk_id: ID of retrieved chunk
        document_id: Parent document ID
        content: Chunk content
        score: Relevance score (0.0-1.0)
        metadata: Result metadata
        rank: Rank in result set (1-based)
        retrieval_method: Method used (vector, hybrid, graph)
    """

    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: dict[str, str] = field(default_factory=dict)
    rank: int = 1
    retrieval_method: Literal["vector", "hybrid", "graph"] | None = None


@dataclass
class BenchmarkQuery:
    """A benchmark query with ground truth annotations.

    Attributes:
        query_id: Unique query identifier
        query: Query text
        ground_truth_answer: Expected answer
        relevant_element_ids: IDs of relevant elements
        query_type: Type of query
        difficulty: Difficulty level
        notes: Optional notes about the query
        expected_chunks: Optional list of expected chunk IDs
    """

    query_id: str
    query: str
    ground_truth_answer: str
    relevant_element_ids: list[str] = field(default_factory=list)
    query_type: QueryType = QueryType.GENERAL
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    notes: str | None = None
    expected_chunks: list[str] | None = None


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run.

    Attributes:
        benchmark_id: Unique benchmark run identifier
        parser_name: Parser used
        storage_backend: Storage backend used
        query_id: Query identifier
        query: Original query text
        retrieved_results: List of retrieved results
        generated_answer: Answer generated by LLM
        ground_truth_answer: Expected answer
        metrics: Dictionary of metric scores
        retrieval_time_seconds: Time for retrieval
        generation_time_seconds: Time for answer generation
        total_time_seconds: Total time
        timestamp: When benchmark was run
        error: Optional error message
    """

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
