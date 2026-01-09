"""Base parser interface for document parsing.

All parser implementations must inherit from BaseParser and implement the
abstract methods for parsing documents and chunking content.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from models import DocumentChunk, ParsedDocument


class BaseParser(ABC):
    """Abstract base class for document parsers.

    All parser implementations must inherit from this class and implement
    the parse() and chunk_document() methods.

    Attributes:
        name: Human-readable name of the parser
        config: Optional configuration dictionary
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        """Initialize the parser.

        Args:
            name: Name of the parser (e.g., "LlamaParse", "Docling")
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}

    @abstractmethod
    async def parse(self, file_path: Path) -> ParsedDocument:
        """Parse a document file and extract structured content.

        This method should:
        1. Read the document from file_path
        2. Extract all elements (text, tables, figures, etc.)
        3. Preserve document structure and metadata
        4. Return a ParsedDocument with all extracted elements

        Args:
            file_path: Path to the document file to parse

        Returns:
            ParsedDocument containing all extracted elements and metadata

        Raises:
            FileNotFoundError: If file_path does not exist
            ValueError: If file format is not supported
            RuntimeError: If parsing fails
        """
        pass

    @abstractmethod
    def chunk_document(self, doc: ParsedDocument) -> list[DocumentChunk]:
        """Chunk a parsed document into smaller units for RAG.

        This method should:
        1. Take a ParsedDocument and split it into chunks
        2. Respect semantic boundaries (sections, paragraphs)
        3. Maintain context by including metadata
        4. Keep chunks within configured size limits

        Different parsers may use different chunking strategies:
        - Fixed-size chunking with overlap
        - Semantic chunking (by section/paragraph)
        - Hybrid approaches

        Args:
            doc: ParsedDocument to chunk

        Returns:
            List of DocumentChunk objects

        Raises:
            ValueError: If document is empty or invalid
        """
        pass

    def validate_file(self, file_path: Path) -> None:
        """Validate that the file exists and has a supported extension.

        Args:
            file_path: Path to validate

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file extension is not supported
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        supported_extensions = {".pdf", ".docx", ".doc", ".txt", ".html", ".md"}
        if file_path.suffix.lower() not in supported_extensions:
            raise ValueError(
                f"Unsupported file extension: {file_path.suffix}. "
                f"Supported: {supported_extensions}"
            )

    def get_chunk_size(self) -> int:
        """Get configured chunk size in characters.

        Returns:
            Chunk size from config or default value
        """
        return self.config.get("chunk_size", 1000)

    def get_chunk_overlap(self) -> int:
        """Get configured chunk overlap in characters.

        Returns:
            Chunk overlap from config or default value
        """
        return self.config.get("chunk_overlap", 200)

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses a simple heuristic: ~4 characters per token.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        return len(text) // 4

    def __repr__(self) -> str:
        """String representation of the parser."""
        return f"{self.__class__.__name__}(name='{self.name}')"
