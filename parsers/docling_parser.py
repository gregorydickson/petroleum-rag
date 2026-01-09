"""Docling parser implementation for the petroleum RAG benchmark.

Docling is IBM's document understanding library that excels at table extraction
and structure preservation. It uses advanced layout analysis models to extract
hierarchical document structure with high accuracy.
"""

import logging
from pathlib import Path
from typing import Any

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption

from models import DocumentChunk, ElementType, ParsedDocument, ParsedElement
from parsers.base import BaseParser

logger = logging.getLogger(__name__)


class DoclingParser(BaseParser):
    """Parser implementation using IBM Docling for document processing.

    Docling provides excellent table extraction with structure preservation,
    semantic document understanding, and hierarchical content extraction.
    It's particularly strong at handling technical documents with complex tables.

    Attributes:
        converter: Docling DocumentConverter instance
        pipeline_options: Configuration for PDF processing pipeline
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize Docling parser with optimized settings.

        Args:
            config: Optional configuration dictionary
        """
        super().__init__("Docling", config)

        # Configure pipeline for technical documents with tables
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_ocr = True
        self.pipeline_options.do_table_structure = True
        self.pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

        # Initialize converter with PDF options
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)
            }
        )

        logger.info("Initialized Docling parser with table extraction enabled")

    async def parse(self, file_path: Path) -> ParsedDocument:
        """Parse a document using Docling converter.

        This method:
        1. Validates the file
        2. Converts the document using Docling
        3. Extracts all elements (text, tables, figures)
        4. Preserves table structure in both HTML and Markdown
        5. Maintains document hierarchy

        Args:
            file_path: Path to document file

        Returns:
            ParsedDocument with all extracted elements

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format not supported
            RuntimeError: If parsing fails
        """
        self.validate_file(file_path)

        logger.info(f"Parsing document with Docling: {file_path}")

        try:
            # Convert document
            result = self.converter.convert(str(file_path))
            doc = result.document

            # Extract all elements
            elements: list[ParsedElement] = []
            element_counter = {"text": 0, "table": 0, "figure": 0, "heading": 0}

            # Process document items
            # Note: iterate_items() returns tuples of (item, level)
            for item_tuple in doc.iterate_items():
                # Unpack tuple - item is the DocItem, level is the hierarchy level
                item = item_tuple[0] if isinstance(item_tuple, tuple) else item_tuple

                element_type = self._map_docling_type(item.label)
                element_counter[element_type.value] = (
                    element_counter.get(element_type.value, 0) + 1
                )

                # Generate unique element ID
                element_id = (
                    f"{file_path.stem}_{element_type.value}_{element_counter[element_type.value]}"
                )

                # Extract content
                if hasattr(item, "text"):
                    content = item.text
                elif hasattr(item, "data"):
                    # For tables and other structured items, build text from cells
                    content = self._extract_table_text(item.data)
                else:
                    content = str(item)

                # For tables, extract both HTML and Markdown representations
                formatted_content = None
                metadata: dict[str, str] = {}

                if element_type == ElementType.TABLE:
                    try:
                        # Get table data and export to different formats
                        if hasattr(item, "data"):
                            table_data = item.data

                            # Export table using document-level export
                            # Since individual items don't have export methods,
                            # we'll use the document's export and cache it
                            formatted_content = doc.export_to_html()
                            metadata["format"] = "html"
                            metadata["markdown"] = doc.export_to_markdown()

                            # Store table dimensions
                            metadata["num_rows"] = str(table_data.num_rows)
                            metadata["num_cols"] = str(table_data.num_cols)
                    except Exception as e:
                        logger.warning(f"Failed to extract table formats: {e}")

                # Get page number if available
                page_number = None
                if hasattr(item, "page_no"):
                    page_number = item.page_no

                # Get bounding box if available
                bbox = None
                if hasattr(item, "bbox"):
                    bbox_obj = item.bbox
                    if bbox_obj:
                        bbox = [
                            bbox_obj.l if hasattr(bbox_obj, "l") else 0,
                            bbox_obj.t if hasattr(bbox_obj, "t") else 0,
                            bbox_obj.r if hasattr(bbox_obj, "r") else 0,
                            bbox_obj.b if hasattr(bbox_obj, "b") else 0,
                        ]

                # Get parent section if available
                parent_section = None
                if hasattr(item, "parent"):
                    parent = item.parent
                    if parent and hasattr(parent, "text"):
                        parent_section = parent.text[:100]  # Truncate if too long

                # Store element type in metadata
                if hasattr(item, "label"):
                    metadata["docling_label"] = item.label

                element = ParsedElement(
                    element_id=element_id,
                    element_type=element_type,
                    content=content,
                    formatted_content=formatted_content,
                    metadata=metadata,
                    page_number=page_number,
                    bbox=bbox,
                    parent_section=parent_section,
                )

                elements.append(element)

            # Get document metadata
            doc_metadata: dict[str, str] = {}
            if hasattr(doc, "name"):
                doc_metadata["document_name"] = doc.name
            if hasattr(doc, "origin"):
                doc_metadata["origin"] = str(doc.origin)

            # Count pages
            total_pages = None
            if hasattr(doc, "pages"):
                total_pages = len(doc.pages)

            logger.info(
                f"Successfully parsed {len(elements)} elements from {file_path.name} "
                f"(Tables: {element_counter.get('table', 0)}, "
                f"Headings: {element_counter.get('heading', 0)})"
            )

            return ParsedDocument(
                document_id=file_path.stem,
                source_file=file_path,
                parser_name=self.name,
                elements=elements,
                metadata=doc_metadata,
                parse_time_seconds=0.0,  # Will be set by benchmarking code
                total_pages=total_pages,
            )

        except Exception as e:
            error_msg = f"Docling parsing failed for {file_path}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def chunk_document(self, doc: ParsedDocument) -> list[DocumentChunk]:
        """Chunk parsed document by semantic units.

        Docling's strength is semantic structure understanding, so we chunk by:
        1. Each table becomes its own chunk (preserve table integrity)
        2. Text between tables is chunked by size
        3. Headings are kept with their content
        4. Maintain hierarchical context

        Args:
            doc: ParsedDocument to chunk

        Returns:
            List of DocumentChunk objects

        Raises:
            ValueError: If document is empty
        """
        if not doc.elements:
            raise ValueError(f"Cannot chunk empty document: {doc.document_id}")

        chunks: list[DocumentChunk] = []
        chunk_index = 0

        # Get chunking parameters
        chunk_size = self.get_chunk_size()
        chunk_overlap = self.get_chunk_overlap()

        logger.debug(
            f"Chunking document {doc.document_id} with chunk_size={chunk_size}, "
            f"overlap={chunk_overlap}"
        )

        current_text = ""
        current_elements: list[str] = []
        current_start_page = None
        current_end_page = None
        current_section = None

        for element in doc.elements:
            # Tables always get their own chunk
            if element.element_type == ElementType.TABLE:
                # First, flush any accumulated text
                if current_text.strip():
                    chunk = self._create_chunk(
                        doc_id=doc.document_id,
                        chunk_index=chunk_index,
                        content=current_text,
                        element_ids=current_elements,
                        start_page=current_start_page,
                        end_page=current_end_page,
                        parent_section=current_section,
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                    # Reset accumulators
                    current_text = ""
                    current_elements = []
                    current_start_page = None
                    current_end_page = None

                # Create table chunk with both HTML and Markdown
                table_content = element.content
                if element.formatted_content and isinstance(element.formatted_content, str):
                    # Prefer HTML formatted content for tables
                    table_content = element.formatted_content

                # Add markdown representation if available
                markdown = element.metadata.get("markdown")
                if markdown and isinstance(markdown, str):
                    # Only append if table_content is a string
                    if isinstance(table_content, str):
                        table_content += f"\n\n{markdown}"

                chunk = self._create_chunk(
                    doc_id=doc.document_id,
                    chunk_index=chunk_index,
                    content=table_content,
                    element_ids=[element.element_id],
                    start_page=element.page_number,
                    end_page=element.page_number,
                    parent_section=element.parent_section or current_section,
                )
                chunks.append(chunk)
                chunk_index += 1

            else:
                # For non-table elements, accumulate until size threshold
                element_text = element.content

                # Track section headers
                if element.element_type == ElementType.HEADING:
                    current_section = element.content[:100]

                # Check if adding this element would exceed chunk size
                if current_text and len(current_text) + len(element_text) > chunk_size:
                    # Create chunk with accumulated content
                    chunk = self._create_chunk(
                        doc_id=doc.document_id,
                        chunk_index=chunk_index,
                        content=current_text,
                        element_ids=current_elements,
                        start_page=current_start_page,
                        end_page=current_end_page,
                        parent_section=current_section,
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                    # Start new chunk with overlap
                    if chunk_overlap > 0:
                        overlap_text = current_text[-chunk_overlap:]
                        current_text = overlap_text + "\n\n" + element_text
                    else:
                        current_text = element_text

                    current_elements = [element.element_id]
                    current_start_page = element.page_number
                    current_end_page = element.page_number
                else:
                    # Add to current chunk
                    if current_text:
                        current_text += "\n\n" + element_text
                    else:
                        current_text = element_text

                    current_elements.append(element.element_id)

                    if current_start_page is None:
                        current_start_page = element.page_number
                    current_end_page = element.page_number

        # Don't forget the last accumulated chunk
        if current_text.strip():
            chunk = self._create_chunk(
                doc_id=doc.document_id,
                chunk_index=chunk_index,
                content=current_text,
                element_ids=current_elements,
                start_page=current_start_page,
                end_page=current_end_page,
                parent_section=current_section,
            )
            chunks.append(chunk)

        logger.info(f"Created {len(chunks)} chunks from document {doc.document_id}")

        return chunks

    def _extract_table_text(self, table_data: Any) -> str:
        """Extract plain text from table data.

        Args:
            table_data: TableData object from Docling

        Returns:
            Plain text representation of the table
        """
        if not hasattr(table_data, "table_cells"):
            return str(table_data)

        try:
            # Build a grid from cells
            num_rows = table_data.num_rows
            num_cols = table_data.num_cols
            grid = [[" " * 10 for _ in range(num_cols)] for _ in range(num_rows)]

            # Fill in cell values
            for cell in table_data.table_cells:
                if hasattr(cell, "text") and hasattr(cell, "start_row_offset_idx"):
                    row = cell.start_row_offset_idx
                    col = cell.start_col_offset_idx
                    if 0 <= row < num_rows and 0 <= col < num_cols:
                        grid[row][col] = cell.text

            # Convert to plain text table
            lines = []
            for row in grid:
                lines.append(" | ".join(cell.ljust(15) for cell in row))

            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"Failed to build table text: {e}")
            return str(table_data)

    def _map_docling_type(self, docling_label: str) -> ElementType:
        """Map Docling element label to our ElementType.

        Args:
            docling_label: Label from Docling

        Returns:
            Corresponding ElementType
        """
        # Docling uses labels like: title, section_header, text, table, figure, etc.
        label_lower = docling_label.lower()

        if "title" in label_lower or "heading" in label_lower or "header" in label_lower:
            return ElementType.HEADING
        elif "table" in label_lower:
            return ElementType.TABLE
        elif "figure" in label_lower or "picture" in label_lower or "image" in label_lower:
            return ElementType.FIGURE
        elif "list" in label_lower:
            return ElementType.LIST
        elif "code" in label_lower:
            return ElementType.CODE
        elif "caption" in label_lower:
            return ElementType.CAPTION
        elif "footer" in label_lower:
            return ElementType.FOOTER
        else:
            return ElementType.TEXT

    def _create_chunk(
        self,
        doc_id: str,
        chunk_index: int,
        content: str,
        element_ids: list[str],
        start_page: int | None,
        end_page: int | None,
        parent_section: str | None,
    ) -> DocumentChunk:
        """Create a DocumentChunk with proper metadata.

        Args:
            doc_id: Document identifier
            chunk_index: Sequential chunk index
            content: Chunk content text
            element_ids: List of element IDs in this chunk
            start_page: Starting page number
            end_page: Ending page number
            parent_section: Parent section title

        Returns:
            DocumentChunk instance
        """
        chunk_id = f"{doc_id}_chunk_{chunk_index}"

        metadata: dict[str, str] = {
            "parser": self.name,
            "chunk_method": "semantic",
        }

        if parent_section:
            metadata["section"] = parent_section

        return DocumentChunk(
            chunk_id=chunk_id,
            document_id=doc_id,
            content=content,
            element_ids=element_ids,
            metadata=metadata,
            chunk_index=chunk_index,
            start_page=start_page,
            end_page=end_page,
            token_count=self.estimate_tokens(content),
            parent_section=parent_section,
        )
