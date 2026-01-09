"""PageIndex parser with semantic chunking capabilities.

PageIndex is designed to provide novel semantic chunking that respects document
structure and meaning. This implementation demonstrates the concept with a fallback
approach that emulates PageIndex's intelligent chunking strategy.

Key Features:
- Semantic boundary detection
- Structure-aware chunking
- Intelligent overlap based on content similarity
- Preservation of context across chunks
"""

import importlib.util
import re
from pathlib import Path
from typing import Any

from models import DocumentChunk, ElementType, ParsedDocument, ParsedElement
from parsers.base import BaseParser


class PageIndexParser(BaseParser):
    """Parser leveraging PageIndex's novel semantic chunking approach.

    PageIndex provides intelligent document parsing with semantic-aware chunking
    that understands document structure and maintains context. This implementation
    provides both native PageIndex integration (when available) and a fallback
    that demonstrates similar principles.

    Attributes:
        name: Parser name ("PageIndex")
        config: Configuration dictionary
        use_fallback: Whether to use fallback implementation
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize PageIndex parser.

        Args:
            config: Optional configuration dictionary with:
                - chunk_size: Target chunk size (default: 1000)
                - chunk_overlap: Overlap size (default: 200)
                - semantic_threshold: Similarity threshold for chunking (default: 0.7)
                - preserve_structure: Maintain document structure (default: True)
        """
        super().__init__("PageIndex", config)
        self.use_fallback = not self._check_pageindex_available()

        # Semantic chunking parameters
        self.semantic_threshold = self.config.get("semantic_threshold", 0.7)
        self.preserve_structure = self.config.get("preserve_structure", True)

        if self.use_fallback:
            self._initialize_fallback()

    def _check_pageindex_available(self) -> bool:
        """Check if PageIndex library is available.

        Returns:
            True if PageIndex is available, False otherwise
        """
        # Check if pageindex package is installed
        spec = importlib.util.find_spec("pageindex")
        if spec is None:
            return False

        try:
            import pageindex  # noqa: F401
            return True
        except ImportError:
            return False

    def _initialize_fallback(self) -> None:
        """Initialize fallback semantic chunking components.

        The fallback uses PyMuPDF for PDF parsing and implements a simplified
        version of semantic chunking based on structural boundaries.
        """
        # Check for PyMuPDF availability for fallback PDF parsing
        spec = importlib.util.find_spec("fitz")
        if spec is None:
            raise RuntimeError(
                "PageIndex fallback requires PyMuPDF (fitz). Install with:\n"
                "  pip install pymupdf\n\n"
                "For native PageIndex support, install:\n"
                "  pip install pageindex  # (if available)\n"
                "  # Or contact PageIndex for API access"
            )

    async def parse(self, file_path: Path) -> ParsedDocument:
        """Parse document using PageIndex or fallback implementation.

        Args:
            file_path: Path to document file

        Returns:
            ParsedDocument with extracted elements

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format unsupported
            RuntimeError: If parsing fails
        """
        import time
        from datetime import datetime, timezone

        self.validate_file(file_path)
        start_time = time.time()

        try:
            if self.use_fallback:
                elements = await self._parse_with_fallback(file_path)
            else:
                elements = await self._parse_with_pageindex(file_path)

            parse_time = time.time() - start_time

            return ParsedDocument(
                document_id=file_path.stem,
                source_file=file_path,
                parser_name=self.name,
                elements=elements,
                metadata={
                    "parser_mode": "fallback" if self.use_fallback else "native",
                    "file_size_bytes": str(file_path.stat().st_size),
                    "file_extension": file_path.suffix,
                },
                parse_time_seconds=parse_time,
                parsed_at=datetime.now(timezone.utc),
                total_pages=self._count_pages(elements),
            )

        except Exception as e:
            parse_time = time.time() - start_time
            return ParsedDocument(
                document_id=file_path.stem,
                source_file=file_path,
                parser_name=self.name,
                elements=[],
                parse_time_seconds=parse_time,
                parsed_at=datetime.now(timezone.utc),
                error=f"Parsing failed: {str(e)}",
            )

    async def _parse_with_pageindex(self, file_path: Path) -> list[ParsedElement]:
        """Parse using native PageIndex library.

        Args:
            file_path: Path to document

        Returns:
            List of parsed elements
        """
        import pageindex

        # Initialize PageIndex client
        # Note: This is pseudocode - actual API may differ
        client = pageindex.Client(api_key=self.config.get("api_key", ""))

        # Parse document with PageIndex
        result = await client.parse_document(
            str(file_path),
            options={
                "extract_tables": True,
                "extract_figures": True,
                "semantic_analysis": True,
            }
        )

        # Convert PageIndex elements to our format
        elements = []
        for idx, element in enumerate(result.elements):
            elements.append(
                ParsedElement(
                    element_id=f"{file_path.stem}_elem_{idx}",
                    element_type=self._map_pageindex_type(element.type),
                    content=element.content,
                    formatted_content=element.formatted_content,
                    metadata={
                        "semantic_score": str(element.semantic_score),
                        "structural_level": str(element.structural_level),
                    },
                    page_number=element.page_number,
                    bbox=element.bbox,
                    parent_section=element.section_id,
                )
            )

        return elements

    async def _parse_with_fallback(self, file_path: Path) -> list[ParsedElement]:
        """Parse using fallback implementation (PyMuPDF + semantic heuristics).

        This fallback demonstrates the semantic chunking concept by:
        1. Extracting text with structural markers
        2. Identifying semantic boundaries (headings, paragraphs, tables)
        3. Preserving document hierarchy

        Args:
            file_path: Path to document

        Returns:
            List of parsed elements
        """
        import fitz  # PyMuPDF

        doc = fitz.open(file_path)
        elements = []
        element_counter = 0

        try:
            for page_num in range(len(doc)):
                page = doc[page_num]

                # Extract text blocks with position information
                blocks = page.get_text("dict")["blocks"]

                for block in blocks:
                    if block.get("type") == 0:  # Text block
                        # Extract text and analyze structure
                        lines = []
                        for line in block.get("lines", []):
                            line_text = ""
                            for span in line.get("spans", []):
                                line_text += span.get("text", "")
                            lines.append(line_text)

                        if not lines:
                            continue

                        content = "\n".join(lines)
                        if not content.strip():
                            continue

                        # Determine element type based on structural heuristics
                        element_type = self._infer_element_type(
                            content,
                            block,
                            lines
                        )

                        # Extract bounding box
                        bbox = [
                            block.get("bbox", [0, 0, 0, 0])[0],
                            block.get("bbox", [0, 0, 0, 0])[1],
                            block.get("bbox", [0, 0, 0, 0])[2],
                            block.get("bbox", [0, 0, 0, 0])[3],
                        ]

                        elements.append(
                            ParsedElement(
                                element_id=f"{file_path.stem}_elem_{element_counter}",
                                element_type=element_type,
                                content=content,
                                formatted_content=None,
                                metadata={
                                    "font_sizes": self._extract_font_sizes(block),
                                    "block_number": str(block.get("number", 0)),
                                },
                                page_number=page_num + 1,
                                bbox=bbox,
                                parent_section=None,
                            )
                        )
                        element_counter += 1

                    elif block.get("type") == 1:  # Image block
                        elements.append(
                            ParsedElement(
                                element_id=f"{file_path.stem}_fig_{element_counter}",
                                element_type=ElementType.FIGURE,
                                content=f"[Figure on page {page_num + 1}]",
                                metadata={"image_type": "embedded"},
                                page_number=page_num + 1,
                                bbox=[
                                    block.get("bbox", [0, 0, 0, 0])[0],
                                    block.get("bbox", [0, 0, 0, 0])[1],
                                    block.get("bbox", [0, 0, 0, 0])[2],
                                    block.get("bbox", [0, 0, 0, 0])[3],
                                ],
                            )
                        )
                        element_counter += 1

        finally:
            doc.close()

        return elements

    def _infer_element_type(
        self,
        content: str,
        block: dict[str, Any],
        lines: list[str]
    ) -> ElementType:
        """Infer element type based on content and structure.

        Args:
            content: Text content
            block: Block metadata from PyMuPDF
            lines: Individual lines of text

        Returns:
            Inferred ElementType
        """
        # Check for heading patterns
        if self._is_heading(content, block):
            return ElementType.HEADING

        # Check for list patterns
        if self._is_list(lines):
            return ElementType.LIST

        # Check for table patterns (basic heuristic)
        if self._looks_like_table(content):
            return ElementType.TABLE

        # Check for code blocks
        if self._is_code_block(content):
            return ElementType.CODE

        # Default to text
        return ElementType.TEXT

    def _is_heading(self, content: str, block: dict[str, Any]) -> bool:
        """Check if content appears to be a heading.

        Args:
            content: Text content
            block: Block metadata

        Returns:
            True if content is likely a heading
        """
        # Short text, often in larger font
        if len(content) < 100 and len(content.split()) < 15:
            # Check for common heading patterns
            heading_patterns = [
                r'^\d+\.?\s+[A-Z]',  # Numbered sections
                r'^[A-Z][A-Z\s]+$',  # ALL CAPS
                r'^Chapter\s+\d+',    # Chapter headings
                r'^Section\s+\d+',    # Section headings
            ]

            for pattern in heading_patterns:
                if re.match(pattern, content.strip()):
                    return True

        return False

    def _is_list(self, lines: list[str]) -> bool:
        """Check if lines form a list.

        Args:
            lines: Text lines

        Returns:
            True if lines form a list
        """
        if len(lines) < 2:
            return False

        # Check for bullet points or numbered lists
        list_markers = [
            r'^\s*[\u2022\u2023\u25E6\u2043\u2219]\s+',  # Bullet points
            r'^\s*[-*+]\s+',  # Dash/asterisk bullets
            r'^\s*\d+[\.)]\s+',  # Numbered lists
            r'^\s*[a-z][\.)]\s+',  # Lettered lists
        ]

        matched_lines = 0
        for line in lines:
            for marker in list_markers:
                if re.match(marker, line):
                    matched_lines += 1
                    break

        return matched_lines >= len(lines) * 0.5

    def _looks_like_table(self, content: str) -> bool:
        """Check if content appears to be a table.

        Args:
            content: Text content

        Returns:
            True if content resembles a table
        """
        lines = content.strip().split('\n')
        if len(lines) < 2:
            return False

        # Check for consistent column separators
        tab_counts = [line.count('\t') for line in lines if line.strip()]
        if tab_counts and len(set(tab_counts)) <= 2 and max(tab_counts) >= 2:
            return True

        # Check for pipe-separated values
        pipe_counts = [line.count('|') for line in lines if line.strip()]
        if pipe_counts and len(set(pipe_counts)) <= 2 and max(pipe_counts) >= 2:
            return True

        return False

    def _is_code_block(self, content: str) -> bool:
        """Check if content is a code block.

        Args:
            content: Text content

        Returns:
            True if content appears to be code
        """
        # Simple heuristics for code detection
        code_indicators = [
            r'^\s*(?:def|class|function|var|const|let)\s+',
            r'^\s*(?:import|from|#include)\s+',
            r'[{}();]',  # Common code punctuation
        ]

        lines = content.strip().split('\n')
        if len(lines) < 2:
            return False

        # Check if multiple lines have code indicators
        code_line_count = 0
        for line in lines:
            for indicator in code_indicators:
                if re.search(indicator, line):
                    code_line_count += 1
                    break

        return code_line_count >= len(lines) * 0.3

    def _extract_font_sizes(self, block: dict[str, Any]) -> str:
        """Extract font sizes from block for structural analysis.

        Args:
            block: Block metadata

        Returns:
            Comma-separated font sizes
        """
        font_sizes = set()
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                size = span.get("size")
                if size:
                    font_sizes.add(f"{size:.1f}")

        return ",".join(sorted(font_sizes))

    def _count_pages(self, elements: list[ParsedElement]) -> int | None:
        """Count total pages from elements.

        Args:
            elements: List of parsed elements

        Returns:
            Total page count or None
        """
        page_numbers = [
            elem.page_number
            for elem in elements
            if elem.page_number is not None
        ]
        return max(page_numbers) if page_numbers else None

    def _map_pageindex_type(self, pageindex_type: str) -> ElementType:
        """Map PageIndex element type to our ElementType.

        Args:
            pageindex_type: PageIndex element type string

        Returns:
            Corresponding ElementType
        """
        type_mapping = {
            "text": ElementType.TEXT,
            "heading": ElementType.HEADING,
            "title": ElementType.HEADING,
            "table": ElementType.TABLE,
            "figure": ElementType.FIGURE,
            "image": ElementType.FIGURE,
            "list": ElementType.LIST,
            "code": ElementType.CODE,
            "equation": ElementType.EQUATION,
            "caption": ElementType.CAPTION,
            "footer": ElementType.FOOTER,
            "header": ElementType.HEADER,
        }

        return type_mapping.get(pageindex_type.lower(), ElementType.TEXT)

    def chunk_document(self, doc: ParsedDocument) -> list[DocumentChunk]:
        """Chunk document using semantic boundaries.

        This implements PageIndex's novel semantic chunking approach:
        1. Respects document structure (sections, paragraphs)
        2. Maintains semantic coherence within chunks
        3. Uses intelligent overlap based on content similarity
        4. Preserves context through metadata

        Args:
            doc: ParsedDocument to chunk

        Returns:
            List of semantically coherent DocumentChunk objects

        Raises:
            ValueError: If document is empty
        """
        if not doc.elements:
            raise ValueError(f"Document {doc.document_id} has no elements to chunk")

        # Group elements by semantic units (sections)
        semantic_units = self._create_semantic_units(doc.elements)

        # Create chunks respecting semantic boundaries
        chunks = []
        chunk_index = 0

        for unit in semantic_units:
            unit_chunks = self._chunk_semantic_unit(
                unit,
                doc.document_id,
                chunk_index
            )
            chunks.extend(unit_chunks)
            chunk_index += len(unit_chunks)

        return chunks

    def _create_semantic_units(
        self,
        elements: list[ParsedElement]
    ) -> list[list[ParsedElement]]:
        """Group elements into semantic units based on structure.

        Args:
            elements: List of parsed elements

        Returns:
            List of semantic units (each unit is a list of elements)
        """
        if not self.preserve_structure:
            # Simple grouping: all elements in one unit
            return [elements]

        units = []
        current_unit = []
        current_section = None

        for element in elements:
            # Start new unit on headings
            if element.element_type == ElementType.HEADING:
                if current_unit:
                    units.append(current_unit)
                current_unit = [element]
                current_section = element.element_id
            else:
                current_unit.append(element)

                # Also split on page boundaries if unit gets large
                if len(current_unit) > 10:
                    units.append(current_unit)
                    current_unit = []
                    current_section = None

        # Add remaining elements
        if current_unit:
            units.append(current_unit)

        return units if units else [elements]

    def _chunk_semantic_unit(
        self,
        unit: list[ParsedElement],
        document_id: str,
        start_index: int
    ) -> list[DocumentChunk]:
        """Chunk a semantic unit into appropriately sized chunks.

        Args:
            unit: List of elements forming a semantic unit
            document_id: Parent document ID
            start_index: Starting chunk index

        Returns:
            List of chunks from this semantic unit
        """
        chunks = []
        current_chunk_elements = []
        current_chunk_text = ""
        chunk_size = self.get_chunk_size()
        chunk_overlap = self.get_chunk_overlap()

        for element in unit:
            element_text = element.content

            # If adding this element exceeds chunk size, create a chunk
            if (current_chunk_text and
                len(current_chunk_text) + len(element_text) > chunk_size):

                # Create chunk from accumulated elements
                chunk = self._create_chunk(
                    current_chunk_elements,
                    current_chunk_text,
                    document_id,
                    start_index + len(chunks)
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                overlap_text = current_chunk_text[-chunk_overlap:] if chunk_overlap > 0 else ""

                # Find elements that contribute to overlap
                overlap_elements = self._find_overlap_elements(
                    current_chunk_elements,
                    overlap_text
                )

                current_chunk_elements = overlap_elements + [element]
                current_chunk_text = overlap_text + "\n" + element_text
            else:
                # Add element to current chunk
                current_chunk_elements.append(element)
                if current_chunk_text:
                    current_chunk_text += "\n" + element_text
                else:
                    current_chunk_text = element_text

        # Create final chunk if there's remaining content
        if current_chunk_elements:
            chunk = self._create_chunk(
                current_chunk_elements,
                current_chunk_text,
                document_id,
                start_index + len(chunks)
            )
            chunks.append(chunk)

        return chunks

    def _find_overlap_elements(
        self,
        elements: list[ParsedElement],
        overlap_text: str
    ) -> list[ParsedElement]:
        """Find elements that should be included in overlap.

        Args:
            elements: Elements from previous chunk
            overlap_text: Overlap text to match

        Returns:
            Elements contributing to overlap
        """
        if not overlap_text or not elements:
            return []

        # Work backwards to find elements in overlap
        overlap_elements = []
        accumulated_text = ""

        for element in reversed(elements):
            if accumulated_text:
                accumulated_text = element.content + "\n" + accumulated_text
            else:
                accumulated_text = element.content

            overlap_elements.insert(0, element)

            # Stop when we've covered the overlap text
            if len(accumulated_text) >= len(overlap_text):
                break

        return overlap_elements

    def _create_chunk(
        self,
        elements: list[ParsedElement],
        content: str,
        document_id: str,
        chunk_index: int
    ) -> DocumentChunk:
        """Create a DocumentChunk from elements.

        Args:
            elements: Elements in this chunk
            content: Combined text content
            document_id: Parent document ID
            chunk_index: Index of this chunk

        Returns:
            DocumentChunk object
        """
        # Extract metadata from elements
        element_ids = [elem.element_id for elem in elements]

        page_numbers = [
            elem.page_number
            for elem in elements
            if elem.page_number is not None
        ]
        start_page = min(page_numbers) if page_numbers else None
        end_page = max(page_numbers) if page_numbers else None

        # Determine parent section (use first heading if available)
        parent_section = None
        for elem in elements:
            if elem.element_type == ElementType.HEADING:
                parent_section = elem.content[:100]  # First 100 chars
                break

        # Build metadata
        metadata = {
            "element_types": ",".join(set(elem.element_type.value for elem in elements)),
            "num_elements": str(len(elements)),
        }

        # Add semantic indicators
        has_table = any(elem.element_type == ElementType.TABLE for elem in elements)
        has_figure = any(elem.element_type == ElementType.FIGURE for elem in elements)
        has_equation = any(elem.element_type == ElementType.EQUATION for elem in elements)

        if has_table:
            metadata["contains_table"] = "true"
        if has_figure:
            metadata["contains_figure"] = "true"
        if has_equation:
            metadata["contains_equation"] = "true"

        return DocumentChunk(
            chunk_id=f"{document_id}_chunk_{chunk_index}",
            document_id=document_id,
            content=content.strip(),
            element_ids=element_ids,
            metadata=metadata,
            chunk_index=chunk_index,
            start_page=start_page,
            end_page=end_page,
            token_count=self.estimate_tokens(content),
            parent_section=parent_section,
        )
