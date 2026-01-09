"""LlamaParse parser implementation for petroleum documents.

This parser uses LlamaParse API to extract structured content from PDFs,
with special handling for tables, figures, and technical diagrams common
in petroleum engineering documents.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from llama_parse import LlamaParse

from config import settings
from models import DocumentChunk, ElementType, ParsedDocument, ParsedElement
from parsers.base import BaseParser
from utils.circuit_breaker import call_parser_with_breaker
from utils.rate_limiter import rate_limiter

logger = logging.getLogger(__name__)


class LlamaParseParser(BaseParser):
    """Parser implementation using LlamaParse API.

    LlamaParse excels at extracting tables and structured content from
    technical documents. It preserves layout information and provides
    multiple output formats (markdown, text, JSON).

    Features:
    - High-accuracy table extraction
    - Figure and diagram detection
    - Section hierarchy preservation
    - Async parsing support
    - Configurable result types
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize LlamaParse parser.

        Args:
            config: Optional configuration dict. Uses settings from config.py if not provided.
        """
        super().__init__("LlamaParse", config)

        # Validate API key
        api_key = settings.llama_cloud_api_key
        if not api_key:
            raise ValueError(
                "LLAMA_CLOUD_API_KEY not set. Please set it in .env file or environment."
            )

        # Initialize LlamaParse client
        self.parser = LlamaParse(
            api_key=api_key,
            result_type="markdown",  # Markdown preserves structure better than plain text
            verbose=settings.debug,
            num_workers=4,  # Parallelize multi-file processing
            language="en",
        )

        logger.info("LlamaParse parser initialized successfully")

    async def parse(self, file_path: Path) -> ParsedDocument:
        """Parse a PDF document using LlamaParse API.

        Args:
            file_path: Path to the PDF file to parse

        Returns:
            ParsedDocument with all extracted elements

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported
            RuntimeError: If parsing fails
        """
        # Validate file exists and is supported
        self.validate_file(file_path)

        logger.info(f"Starting LlamaParse parsing of {file_path.name}")
        start_time = time.time()

        try:
            # Acquire rate limit token before making API call
            if rate_limiter.is_registered("llamaparse"):
                await rate_limiter.acquire("llamaparse")

            # Parse document using LlamaParse async API with circuit breaker
            async def _do_parse() -> Any:
                return await self.parser.aparse(str(file_path))

            result = await call_parser_with_breaker(_do_parse)

            # LlamaParse returns a JobResult object with parsed content
            # Extract elements from the result
            elements = await self._extract_elements(result, file_path)

            parse_time = time.time() - start_time

            # Get total pages from result
            total_pages = len(result.pages) if hasattr(result, "pages") else None

            # Create document metadata
            metadata = {
                "parser": "LlamaParse",
                "filename": file_path.name,
                "file_size_bytes": str(file_path.stat().st_size),
                "result_type": "markdown",
            }

            parsed_doc = ParsedDocument(
                document_id=file_path.stem,
                source_file=file_path,
                parser_name=self.name,
                elements=elements,
                metadata=metadata,
                parse_time_seconds=parse_time,
                total_pages=total_pages,
            )

            logger.info(
                f"Successfully parsed {file_path.name}: "
                f"{len(elements)} elements in {parse_time:.2f}s"
            )

            return parsed_doc

        except Exception as e:
            parse_time = time.time() - start_time
            error_msg = f"LlamaParse parsing failed: {str(e)}"
            logger.error(error_msg)

            # Return document with error
            return ParsedDocument(
                document_id=file_path.stem,
                source_file=file_path,
                parser_name=self.name,
                elements=[],
                parse_time_seconds=parse_time,
                error=error_msg,
            )

    async def _extract_elements(
        self, result: Any, file_path: Path
    ) -> list[ParsedElement]:
        """Extract structured elements from LlamaParse result.

        Args:
            result: JobResult object from LlamaParse
            file_path: Source file path for element ID generation

        Returns:
            List of ParsedElement objects
        """
        elements: list[ParsedElement] = []
        doc_id = file_path.stem

        # LlamaParse result has pages with text, markdown, images, etc.
        if not hasattr(result, "pages") or not result.pages:
            logger.warning("No pages found in LlamaParse result")
            return elements

        for page_idx, page in enumerate(result.pages, start=1):
            # Extract markdown content (preserves structure)
            markdown_content = page.md if hasattr(page, "md") else ""
            text_content = page.text if hasattr(page, "text") else ""

            if not markdown_content and not text_content:
                continue

            # Parse markdown to identify different element types
            page_elements = self._parse_markdown_content(
                markdown_content or text_content,
                doc_id,
                page_idx,
            )

            elements.extend(page_elements)

            # Extract images if available
            if hasattr(page, "images") and page.images:
                for img_idx, image in enumerate(page.images, start=1):
                    element_id = f"{doc_id}_fig_{page_idx}_{img_idx}"
                    elements.append(
                        ParsedElement(
                            element_id=element_id,
                            element_type=ElementType.FIGURE,
                            content=f"[Figure {img_idx} on page {page_idx}]",
                            metadata={
                                "image_index": str(img_idx),
                                "has_caption": "false",
                            },
                            page_number=page_idx,
                        )
                    )

        logger.info(f"Extracted {len(elements)} elements from {len(result.pages)} pages")
        return elements

    def _parse_markdown_content(
        self, markdown: str, doc_id: str, page_num: int
    ) -> list[ParsedElement]:
        """Parse markdown content into structured elements.

        Identifies headings, tables, code blocks, and text paragraphs.

        Args:
            markdown: Markdown content string
            doc_id: Document identifier
            page_num: Page number

        Returns:
            List of ParsedElement objects
        """
        elements: list[ParsedElement] = []
        lines = markdown.split("\n")

        current_section = None
        current_table_lines: list[str] = []
        current_code_lines: list[str] = []
        current_text_lines: list[str] = []
        element_counter = 0

        in_table = False
        in_code = False

        for line in lines:
            stripped = line.strip()

            # Detect code blocks
            if stripped.startswith("```"):
                if in_code:
                    # End of code block
                    if current_code_lines:
                        element_counter += 1
                        content = "\n".join(current_code_lines)
                        elements.append(
                            ParsedElement(
                                element_id=f"{doc_id}_code_{page_num}_{element_counter}",
                                element_type=ElementType.CODE,
                                content=content,
                                formatted_content=f"```\n{content}\n```",
                                page_number=page_num,
                                parent_section=current_section,
                            )
                        )
                        current_code_lines = []
                    in_code = False
                else:
                    # Start of code block
                    # Flush any accumulated text
                    if current_text_lines:
                        elements.append(self._create_text_element(
                            current_text_lines, doc_id, page_num,
                            element_counter, current_section
                        ))
                        element_counter += 1
                        current_text_lines = []
                    in_code = True
                continue

            if in_code:
                current_code_lines.append(line)
                continue

            # Detect headings
            if stripped.startswith("#"):
                # Flush accumulated content
                if current_text_lines:
                    element_counter += 1
                    elements.append(self._create_text_element(
                        current_text_lines, doc_id, page_num,
                        element_counter, current_section
                    ))
                    current_text_lines = []

                # Extract heading
                heading_level = len(stripped) - len(stripped.lstrip("#"))
                heading_text = stripped.lstrip("#").strip()
                element_counter += 1

                current_section = heading_text  # Update current section

                elements.append(
                    ParsedElement(
                        element_id=f"{doc_id}_heading_{page_num}_{element_counter}",
                        element_type=ElementType.HEADING,
                        content=heading_text,
                        formatted_content=stripped,
                        metadata={"level": str(heading_level)},
                        page_number=page_num,
                        parent_section=current_section,
                    )
                )
                continue

            # Detect tables (markdown tables have | characters)
            if "|" in stripped and stripped.count("|") >= 2:
                if not in_table:
                    # Start of table - flush text
                    if current_text_lines:
                        element_counter += 1
                        elements.append(self._create_text_element(
                            current_text_lines, doc_id, page_num,
                            element_counter, current_section
                        ))
                        current_text_lines = []
                    in_table = True

                current_table_lines.append(line)
                continue
            elif in_table:
                # End of table
                if current_table_lines:
                    element_counter += 1
                    table_content = "\n".join(current_table_lines)
                    elements.append(
                        ParsedElement(
                            element_id=f"{doc_id}_table_{page_num}_{element_counter}",
                            element_type=ElementType.TABLE,
                            content=table_content,
                            formatted_content=table_content,
                            metadata={"format": "markdown"},
                            page_number=page_num,
                            parent_section=current_section,
                        )
                    )
                    current_table_lines = []
                in_table = False

            # Regular text
            if stripped:
                current_text_lines.append(line)
            elif current_text_lines:
                # Empty line - might be paragraph break
                # Flush accumulated text
                element_counter += 1
                elements.append(self._create_text_element(
                    current_text_lines, doc_id, page_num,
                    element_counter, current_section
                ))
                current_text_lines = []

        # Flush remaining content
        if current_text_lines:
            element_counter += 1
            elements.append(self._create_text_element(
                current_text_lines, doc_id, page_num,
                element_counter, current_section
            ))

        if current_table_lines:
            element_counter += 1
            table_content = "\n".join(current_table_lines)
            elements.append(
                ParsedElement(
                    element_id=f"{doc_id}_table_{page_num}_{element_counter}",
                    element_type=ElementType.TABLE,
                    content=table_content,
                    formatted_content=table_content,
                    page_number=page_num,
                    parent_section=current_section,
                )
            )

        return elements

    def _create_text_element(
        self,
        lines: list[str],
        doc_id: str,
        page_num: int,
        element_num: int,
        parent_section: str | None,
    ) -> ParsedElement:
        """Create a text element from accumulated lines.

        Args:
            lines: Text lines
            doc_id: Document ID
            page_num: Page number
            element_num: Element counter
            parent_section: Parent section name

        Returns:
            ParsedElement for text
        """
        content = "\n".join(lines).strip()
        return ParsedElement(
            element_id=f"{doc_id}_text_{page_num}_{element_num}",
            element_type=ElementType.TEXT,
            content=content,
            page_number=page_num,
            parent_section=parent_section,
        )

    def chunk_document(self, doc: ParsedDocument) -> list[DocumentChunk]:
        """Chunk a parsed document intelligently.

        Strategy:
        - Respect section boundaries (don't split sections if possible)
        - Keep tables intact (never split a table)
        - Use configured chunk_size and chunk_overlap
        - Maintain context through metadata

        Args:
            doc: ParsedDocument to chunk

        Returns:
            List of DocumentChunk objects

        Raises:
            ValueError: If document is empty or invalid
        """
        if not doc.elements:
            raise ValueError(f"Document {doc.document_id} has no elements to chunk")

        chunks: list[DocumentChunk] = []
        chunk_size = self.get_chunk_size()
        chunk_overlap = self.get_chunk_overlap()

        # Group elements by section for better chunking
        current_chunk_elements: list[ParsedElement] = []
        current_chunk_text: list[str] = []
        current_chunk_size = 0
        chunk_index = 0

        for element in doc.elements:
            element_size = len(element.content)

            # Tables should never be split - if a table is too large,
            # it gets its own chunk
            if element.element_type == ElementType.TABLE:
                # Flush current chunk if it has content
                if current_chunk_elements:
                    chunks.append(
                        self._create_chunk(
                            doc.document_id,
                            current_chunk_elements,
                            current_chunk_text,
                            chunk_index,
                        )
                    )
                    chunk_index += 1
                    current_chunk_elements = []
                    current_chunk_text = []
                    current_chunk_size = 0

                # Create dedicated chunk for table (even if oversized)
                chunks.append(
                    self._create_chunk(
                        doc.document_id,
                        [element],
                        [element.content],
                        chunk_index,
                    )
                )
                chunk_index += 1
                continue

            # Check if adding this element would exceed chunk size
            if current_chunk_size + element_size > chunk_size and current_chunk_elements:
                # Flush current chunk
                chunks.append(
                    self._create_chunk(
                        doc.document_id,
                        current_chunk_elements,
                        current_chunk_text,
                        chunk_index,
                    )
                )
                chunk_index += 1

                # Handle overlap by keeping last element if it fits in overlap
                if chunk_overlap > 0 and current_chunk_elements:
                    last_element = current_chunk_elements[-1]
                    if len(last_element.content) <= chunk_overlap:
                        current_chunk_elements = [last_element]
                        current_chunk_text = [last_element.content]
                        current_chunk_size = len(last_element.content)
                    else:
                        current_chunk_elements = []
                        current_chunk_text = []
                        current_chunk_size = 0
                else:
                    current_chunk_elements = []
                    current_chunk_text = []
                    current_chunk_size = 0

            # Add element to current chunk
            current_chunk_elements.append(element)
            current_chunk_text.append(element.content)
            current_chunk_size += element_size

        # Flush remaining elements
        if current_chunk_elements:
            chunks.append(
                self._create_chunk(
                    doc.document_id,
                    current_chunk_elements,
                    current_chunk_text,
                    chunk_index,
                )
            )

        logger.info(
            f"Chunked document {doc.document_id}: "
            f"{len(doc.elements)} elements → {len(chunks)} chunks"
        )

        return chunks

    def _create_chunk(
        self,
        document_id: str,
        elements: list[ParsedElement],
        text_parts: list[str],
        chunk_index: int,
    ) -> DocumentChunk:
        """Create a DocumentChunk from elements.

        Args:
            document_id: Parent document ID
            elements: Elements included in this chunk
            text_parts: Text content parts
            chunk_index: Sequential chunk index

        Returns:
            DocumentChunk object
        """
        content = "\n\n".join(text_parts)
        element_ids = [e.element_id for e in elements]

        # Determine page range
        pages = [e.page_number for e in elements if e.page_number is not None]
        start_page = min(pages) if pages else None
        end_page = max(pages) if pages else None

        # Get parent section from first element
        parent_section = elements[0].parent_section if elements else None

        # Count element types
        element_type_counts = {}
        for element in elements:
            type_name = element.element_type.value
            element_type_counts[type_name] = element_type_counts.get(type_name, 0) + 1

        metadata = {
            "element_count": str(len(elements)),
            "parent_section": parent_section or "unknown",
            **{f"count_{k}": str(v) for k, v in element_type_counts.items()},
        }

        return DocumentChunk(
            chunk_id=f"{document_id}_chunk_{chunk_index}",
            document_id=document_id,
            content=content,
            element_ids=element_ids,
            metadata=metadata,
            chunk_index=chunk_index,
            start_page=start_page,
            end_page=end_page,
            token_count=self.estimate_tokens(content),
            parent_section=parent_section,
        )
