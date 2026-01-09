"""Vertex Document AI parser implementation.

This parser leverages Google Cloud's Vertex Document AI for enterprise-grade
OCR and layout analysis, with support for both digital and scanned PDFs.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1 as documentai

from config import settings
from models import DocumentChunk, ElementType, ParsedDocument, ParsedElement
from parsers.base import BaseParser
from utils.circuit_breaker import call_parser_with_breaker
from utils.rate_limiter import rate_limiter

logger = logging.getLogger(__name__)


class VertexDocAIParser(BaseParser):
    """Parser using Google Vertex Document AI.

    Provides enterprise-grade OCR and layout analysis with support for:
    - High-quality OCR for scanned documents
    - Layout detection (headers, paragraphs, tables, etc.)
    - Table extraction with cell-level structure
    - Bounding box coordinates for all elements
    - Multi-page document processing

    Configuration:
        Requires environment variables:
        - GOOGLE_APPLICATION_CREDENTIALS: Path to service account key
        - GOOGLE_CLOUD_PROJECT: GCP project ID
        - VERTEX_DOCAI_PROCESSOR_ID: Document AI processor ID
        - VERTEX_DOCAI_LOCATION: Location (default: "us")
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize Vertex Document AI parser.

        Args:
            config: Optional configuration dictionary

        Raises:
            ValueError: If required Google Cloud credentials are not configured
        """
        super().__init__("VertexDocAI", config)

        # Validate required configuration
        if not settings.google_cloud_project:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT environment variable is required. "
                "Set it in .env file or environment."
            )
        if not settings.vertex_docai_processor_id:
            raise ValueError(
                "VERTEX_DOCAI_PROCESSOR_ID environment variable is required. "
                "Set it in .env file or environment."
            )
        if settings.google_application_credentials:
            credentials_path = Path(settings.google_application_credentials)
            if not credentials_path.exists():
                raise ValueError(
                    f"Google credentials file not found: {credentials_path}. "
                    f"Update GOOGLE_APPLICATION_CREDENTIALS in .env"
                )

        # Initialize Document AI client
        self.project_id = settings.google_cloud_project
        self.location = settings.vertex_docai_location
        self.processor_id = settings.vertex_docai_processor_id

        # Build processor name
        self.processor_name = (
            f"projects/{self.project_id}/"
            f"locations/{self.location}/"
            f"processors/{self.processor_id}"
        )

        # Initialize client with location
        opts = ClientOptions(api_endpoint=f"{self.location}-documentai.googleapis.com")
        self.client = documentai.DocumentProcessorServiceClient(client_options=opts)

        logger.info(
            f"Initialized VertexDocAI parser with processor: {self.processor_name}"
        )

    async def parse(self, file_path: Path) -> ParsedDocument:
        """Parse a document using Vertex Document AI.

        This method:
        1. Validates the input file
        2. Sends the document to Vertex Document AI for processing
        3. Extracts all detected elements (text, tables, layout)
        4. Maps Document AI output to ParsedDocument format

        Args:
            file_path: Path to the document file to parse

        Returns:
            ParsedDocument containing all extracted elements and metadata

        Raises:
            FileNotFoundError: If file_path does not exist
            ValueError: If file format is not supported
            RuntimeError: If Document AI processing fails
        """
        start_time = time.time()

        # Validate file
        self.validate_file(file_path)

        try:
            # Read file content
            with open(file_path, "rb") as f:
                file_content = f.read()

            logger.info(f"Processing {file_path.name} with Vertex Document AI...")

            # Acquire rate limit token before making API call
            if rate_limiter.is_registered("vertex"):
                await rate_limiter.acquire("vertex")

            # Create process request
            raw_document = documentai.RawDocument(
                content=file_content,
                mime_type=self._get_mime_type(file_path),
            )

            request = documentai.ProcessRequest(
                name=self.processor_name,
                raw_document=raw_document,
            )

            # Process document with circuit breaker
            # Note: process_document is synchronous, wrap in executor
            async def _do_parse() -> Any:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, self.client.process_document, request
                )

            result = await call_parser_with_breaker(_do_parse)
            document = result.document

            # Extract elements from Document AI response
            elements = self._extract_elements(document, file_path)

            # Calculate processing time
            parse_time = time.time() - start_time

            # Build metadata
            metadata = {
                "file_name": file_path.name,
                "file_size": str(file_path.stat().st_size),
                "mime_type": self._get_mime_type(file_path),
                "processor_id": self.processor_id,
                "text_length": str(len(document.text)),
            }

            parsed_doc = ParsedDocument(
                document_id=file_path.stem,
                source_file=file_path,
                parser_name=self.name,
                elements=elements,
                metadata=metadata,
                parse_time_seconds=parse_time,
                total_pages=len(document.pages),
            )

            logger.info(
                f"Successfully parsed {file_path.name}: "
                f"{len(elements)} elements, {len(document.pages)} pages, "
                f"{parse_time:.2f}s"
            )

            return parsed_doc

        except Exception as e:
            parse_time = time.time() - start_time
            error_msg = f"Failed to parse {file_path.name}: {str(e)}"
            logger.error(error_msg, exc_info=True)

            # Return document with error
            return ParsedDocument(
                document_id=file_path.stem,
                source_file=file_path,
                parser_name=self.name,
                elements=[],
                metadata={"file_name": file_path.name},
                parse_time_seconds=parse_time,
                error=error_msg,
            )

    def chunk_document(self, doc: ParsedDocument) -> list[DocumentChunk]:
        """Chunk document based on Vertex Document AI layout analysis.

        This method leverages the layout information from Document AI to create
        semantically meaningful chunks:
        1. Groups elements by detected sections/paragraphs
        2. Respects natural document boundaries
        3. Maintains context with metadata
        4. Ensures chunks stay within size limits

        Args:
            doc: ParsedDocument to chunk

        Returns:
            List of DocumentChunk objects

        Raises:
            ValueError: If document is empty or invalid
        """
        if not doc.elements:
            if doc.error:
                raise ValueError(f"Cannot chunk document with error: {doc.error}")
            raise ValueError("Cannot chunk empty document")

        chunks: list[DocumentChunk] = []
        current_chunk_text: list[str] = []
        current_element_ids: list[str] = []
        current_page = doc.elements[0].page_number
        chunk_index = 0

        chunk_size = self.get_chunk_size()
        chunk_overlap = self.get_chunk_overlap()

        for element in doc.elements:
            # Skip empty elements
            if not element.content.strip():
                continue

            # Check if adding this element would exceed chunk size
            combined_text = "\n\n".join(current_chunk_text + [element.content])

            if len(combined_text) > chunk_size and current_chunk_text:
                # Create chunk from accumulated content
                chunk = self._create_chunk(
                    doc=doc,
                    chunk_text=current_chunk_text,
                    element_ids=current_element_ids,
                    chunk_index=chunk_index,
                    start_page=current_page,
                    end_page=element.page_number or current_page,
                )
                chunks.append(chunk)
                chunk_index += 1

                # Handle overlap: keep last portion of text
                if chunk_overlap > 0 and current_chunk_text:
                    overlap_text = combined_text[-chunk_overlap:]
                    current_chunk_text = [overlap_text]
                    current_element_ids = [element.element_id]
                else:
                    current_chunk_text = []
                    current_element_ids = []

                current_page = element.page_number or current_page

            # Add element to current chunk
            current_chunk_text.append(element.content)
            current_element_ids.append(element.element_id)

        # Create final chunk if there's remaining content
        if current_chunk_text:
            last_page = doc.elements[-1].page_number or current_page
            chunk = self._create_chunk(
                doc=doc,
                chunk_text=current_chunk_text,
                element_ids=current_element_ids,
                chunk_index=chunk_index,
                start_page=current_page,
                end_page=last_page,
            )
            chunks.append(chunk)

        logger.info(
            f"Created {len(chunks)} chunks from document {doc.document_id} "
            f"using layout-aware chunking"
        )

        return chunks

    def _extract_elements(
        self, document: documentai.Document, file_path: Path
    ) -> list[ParsedElement]:
        """Extract all elements from Document AI response.

        Args:
            document: Document AI document object
            file_path: Source file path for ID generation

        Returns:
            List of ParsedElement objects
        """
        elements: list[ParsedElement] = []
        doc_id = file_path.stem

        # Process each page
        for page_idx, page in enumerate(document.pages):
            page_number = page_idx + 1

            # Extract paragraphs
            for para_idx, paragraph in enumerate(page.paragraphs):
                text = self._get_layout_text(paragraph.layout, document.text)
                if text.strip():
                    elements.append(
                        ParsedElement(
                            element_id=f"{doc_id}_para_{page_number}_{para_idx}",
                            element_type=ElementType.TEXT,
                            content=text,
                            page_number=page_number,
                            bbox=self._get_bbox(paragraph.layout),
                        )
                    )

            # Extract tables
            for table_idx, table in enumerate(page.tables):
                table_content = self._extract_table_content(table, document.text)
                if table_content.strip():
                    elements.append(
                        ParsedElement(
                            element_id=f"{doc_id}_table_{page_number}_{table_idx}",
                            element_type=ElementType.TABLE,
                            content=table_content,
                            formatted_content=self._format_table_markdown(
                                table, document.text
                            ),
                            page_number=page_number,
                            metadata={
                                "rows": str(len(table.body_rows)),
                                "columns": str(
                                    len(table.header_rows[0].cells)
                                    if table.header_rows
                                    else "0"
                                ),
                            },
                        )
                    )

            # Extract detected blocks (if any)
            for block_idx, block in enumerate(page.blocks):
                text = self._get_layout_text(block.layout, document.text)
                if text.strip():
                    elements.append(
                        ParsedElement(
                            element_id=f"{doc_id}_block_{page_number}_{block_idx}",
                            element_type=ElementType.TEXT,
                            content=text,
                            page_number=page_number,
                            bbox=self._get_bbox(block.layout),
                        )
                    )

        return elements

    def _extract_table_content(
        self, table: documentai.Document.Page.Table, full_text: str
    ) -> str:
        """Extract plain text content from a table.

        Args:
            table: Document AI table object
            full_text: Full document text

        Returns:
            Plain text representation of table
        """
        rows: list[str] = []

        # Extract header rows
        for row in table.header_rows:
            cells = [
                self._get_layout_text(cell.layout, full_text) for cell in row.cells
            ]
            rows.append(" | ".join(cells))

        # Extract body rows
        for row in table.body_rows:
            cells = [
                self._get_layout_text(cell.layout, full_text) for cell in row.cells
            ]
            rows.append(" | ".join(cells))

        return "\n".join(rows)

    def _format_table_markdown(
        self, table: documentai.Document.Page.Table, full_text: str
    ) -> str:
        """Format table as Markdown.

        Args:
            table: Document AI table object
            full_text: Full document text

        Returns:
            Markdown formatted table
        """
        rows: list[str] = []

        # Extract header rows
        if table.header_rows:
            for row in table.header_rows:
                cells = [
                    self._get_layout_text(cell.layout, full_text) for cell in row.cells
                ]
                rows.append("| " + " | ".join(cells) + " |")

            # Add separator
            num_cols = len(table.header_rows[0].cells)
            rows.append("| " + " | ".join(["---"] * num_cols) + " |")

        # Extract body rows
        for row in table.body_rows:
            cells = [
                self._get_layout_text(cell.layout, full_text) for cell in row.cells
            ]
            rows.append("| " + " | ".join(cells) + " |")

        return "\n".join(rows)

    def _get_layout_text(
        self, layout: documentai.Document.Page.Layout, full_text: str
    ) -> str:
        """Extract text from a layout element.

        Args:
            layout: Document AI layout object
            full_text: Full document text

        Returns:
            Text content of the layout element
        """
        text = ""
        for segment in layout.text_anchor.text_segments:
            start_index = int(segment.start_index) if segment.start_index else 0
            end_index = int(segment.end_index) if segment.end_index else len(full_text)
            text += full_text[start_index:end_index]
        return text

    def _get_bbox(self, layout: documentai.Document.Page.Layout) -> list[float] | None:
        """Extract bounding box coordinates from layout.

        Args:
            layout: Document AI layout object

        Returns:
            Bounding box as [x1, y1, x2, y2] or None if not available
        """
        if not layout.bounding_poly or not layout.bounding_poly.vertices:
            return None

        vertices = layout.bounding_poly.vertices
        if len(vertices) < 2:
            return None

        # Get min/max coordinates
        x_coords = [v.x for v in vertices]
        y_coords = [v.y for v in vertices]

        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

    def _get_mime_type(self, file_path: Path) -> str:
        """Get MIME type for file.

        Args:
            file_path: Path to file

        Returns:
            MIME type string

        Raises:
            ValueError: If file extension is not supported
        """
        mime_types = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".doc": "application/msword",
            ".txt": "text/plain",
            ".html": "text/html",
        }

        ext = file_path.suffix.lower()
        if ext not in mime_types:
            raise ValueError(f"Unsupported file type: {ext}")

        return mime_types[ext]

    def _create_chunk(
        self,
        doc: ParsedDocument,
        chunk_text: list[str],
        element_ids: list[str],
        chunk_index: int,
        start_page: int | None,
        end_page: int | None,
    ) -> DocumentChunk:
        """Create a DocumentChunk from accumulated content.

        Args:
            doc: Source ParsedDocument
            chunk_text: List of text strings to combine
            element_ids: List of element IDs in this chunk
            chunk_index: Sequential index of this chunk
            start_page: Starting page number
            end_page: Ending page number

        Returns:
            DocumentChunk object
        """
        content = "\n\n".join(chunk_text)

        return DocumentChunk(
            chunk_id=f"{doc.document_id}_chunk_{chunk_index}",
            document_id=doc.document_id,
            content=content,
            element_ids=element_ids,
            metadata={
                "parser": self.name,
                "source_file": str(doc.source_file.name),
            },
            chunk_index=chunk_index,
            start_page=start_page,
            end_page=end_page,
            token_count=self.estimate_tokens(content),
        )
