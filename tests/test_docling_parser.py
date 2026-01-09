"""Tests for Docling parser.

Tests cover:
- Parser initialization and configuration
- Document parsing with tables
- Table structure preservation (HTML and Markdown)
- Element extraction (text, headings, tables)
- Semantic chunking strategies
- Error handling and edge cases
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from models import DocumentChunk, ElementType, ParsedDocument, ParsedElement
from parsers.docling_parser import DoclingParser


class TestDoclingParserInitialization:
    """Tests for parser initialization and configuration."""

    def test_initialization_success(self):
        """Test successful parser initialization."""
        parser = DoclingParser()

        assert parser.name == "Docling"
        assert parser.converter is not None
        assert parser.pipeline_options is not None
        assert parser.pipeline_options.do_ocr is True
        assert parser.pipeline_options.do_table_structure is True

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = {"chunk_size": 500, "chunk_overlap": 100}
        parser = DoclingParser(config=config)

        assert parser.name == "Docling"
        assert parser.config == config
        assert parser.get_chunk_size() == 500
        assert parser.get_chunk_overlap() == 100

    def test_initialization_default_config(self):
        """Test initialization uses default config values."""
        parser = DoclingParser()

        # Should use base class defaults
        assert parser.get_chunk_size() == 1000
        assert parser.get_chunk_overlap() == 200


class TestDoclingParsing:
    """Tests for document parsing functionality."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return DoclingParser()

    @pytest.fixture
    def mock_docling_result(self):
        """Create mock Docling conversion result."""
        # Create mock document
        mock_doc = Mock()
        mock_doc.name = "test.pdf"
        mock_doc.origin = "/path/to/test.pdf"

        # Create mock pages
        mock_page = Mock()
        mock_doc.pages = [mock_page]

        # Mock export methods
        mock_doc.export_to_html = Mock(
            return_value="<html><body><table><tr><th>Property</th><th>Value</th></tr>"
            "<tr><td>Density</td><td>850 kg/m³</td></tr></table></body></html>"
        )
        mock_doc.export_to_markdown = Mock(
            return_value="| Property | Value |\n|----------|-------|\n| Density | 850 kg/m³ |"
        )

        # Create mock items (text, table, heading)
        mock_items = []

        # Heading item
        heading_item = Mock()
        heading_item.label = "title"
        heading_item.text = "Introduction to Petroleum Engineering"
        heading_item.page_no = 1
        heading_item.bbox = None
        heading_item.parent = None
        mock_items.append(heading_item)

        # Text item
        text_item = Mock()
        text_item.label = "text"
        text_item.text = "This document covers petroleum engineering fundamentals."
        text_item.page_no = 1
        text_item.bbox = Mock()
        text_item.bbox.l = 10
        text_item.bbox.t = 20
        text_item.bbox.r = 100
        text_item.bbox.b = 50
        text_item.parent = None
        mock_items.append(text_item)

        # Table item
        table_item = Mock()
        table_item.label = "table"
        table_item.text = "Property | Value\nDensity | 850 kg/m³"
        table_item.page_no = 1
        table_item.bbox = None

        # Mock table data with export methods
        mock_table_data = Mock()
        mock_table_data.export_to_html = Mock(
            return_value="<table><tr><th>Property</th><th>Value</th></tr>"
            "<tr><td>Density</td><td>850 kg/m³</td></tr></table>"
        )
        mock_table_data.export_to_markdown = Mock(
            return_value="| Property | Value |\n|----------|-------|\n| Density | 850 kg/m³ |"
        )
        table_item.data = mock_table_data
        table_item.parent = None
        mock_items.append(table_item)

        # Mock iterate_items
        mock_doc.iterate_items = Mock(return_value=iter(mock_items))

        # Create mock result
        mock_result = Mock()
        mock_result.document = mock_doc

        return mock_result

    @pytest.mark.asyncio
    async def test_parse_pdf_success(self, parser, mock_docling_result, tmp_path):
        """Test successful PDF parsing with Docling."""
        # Create test PDF file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 test content")

        # Mock the converter
        with patch.object(parser.converter, "convert", return_value=mock_docling_result):
            result = await parser.parse(test_file)

        # Verify result
        assert isinstance(result, ParsedDocument)
        assert result.document_id == "test"
        assert result.parser_name == "Docling"
        assert result.source_file == test_file
        assert len(result.elements) == 3  # heading, text, table
        assert result.error is None

        # Verify elements
        heading_elements = [e for e in result.elements if e.element_type == ElementType.HEADING]
        text_elements = [e for e in result.elements if e.element_type == ElementType.TEXT]
        table_elements = [e for e in result.elements if e.element_type == ElementType.TABLE]

        assert len(heading_elements) == 1
        assert len(text_elements) == 1
        assert len(table_elements) == 1

        # Verify table has both HTML and markdown
        table = table_elements[0]
        assert table.formatted_content is not None
        assert "<table>" in table.formatted_content
        assert table.metadata.get("format") == "html"
        assert "markdown" in table.metadata
        assert "Property" in table.metadata["markdown"]

    @pytest.mark.asyncio
    async def test_parse_unsupported_file_type(self, parser, tmp_path):
        """Test parsing fails for unsupported file type."""
        test_file = tmp_path / "test.xyz"
        test_file.write_text("content")

        with pytest.raises(ValueError, match="Unsupported file extension"):
            await parser.parse(test_file)

    @pytest.mark.asyncio
    async def test_parse_nonexistent_file(self, parser):
        """Test parsing fails for nonexistent file."""
        test_file = Path("/nonexistent/file.pdf")

        with pytest.raises(FileNotFoundError):
            await parser.parse(test_file)

    @pytest.mark.asyncio
    async def test_parse_docling_error(self, parser, tmp_path):
        """Test handling of Docling conversion errors."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 test content")

        # Mock converter to raise error
        with patch.object(
            parser.converter, "convert", side_effect=Exception("Docling conversion failed")
        ):
            with pytest.raises(RuntimeError, match="Docling parsing failed"):
                await parser.parse(test_file)

    @pytest.mark.asyncio
    async def test_parse_with_bounding_boxes(self, parser, tmp_path):
        """Test parsing extracts bounding boxes correctly."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 test content")

        # Create mock with bbox
        mock_item = Mock()
        mock_item.label = "text"
        mock_item.text = "Text with bbox"
        mock_item.page_no = 1
        mock_item.bbox = Mock()
        mock_item.bbox.l = 10
        mock_item.bbox.t = 20
        mock_item.bbox.r = 100
        mock_item.bbox.b = 50
        mock_item.parent = None

        mock_doc = Mock()
        mock_doc.name = "test.pdf"
        mock_doc.origin = "/path/to/test.pdf"
        mock_doc.pages = [Mock()]
        mock_doc.iterate_items = Mock(return_value=iter([mock_item]))

        mock_result = Mock()
        mock_result.document = mock_doc

        with patch.object(parser.converter, "convert", return_value=mock_result):
            result = await parser.parse(test_file)

        # Verify bbox
        assert len(result.elements) == 1
        element = result.elements[0]
        assert element.bbox == [10, 20, 100, 50]

    @pytest.mark.asyncio
    async def test_parse_table_without_export_methods(self, parser, tmp_path):
        """Test parsing handles tables without export methods gracefully."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 test content")

        # Create table item without data attribute
        table_item = Mock()
        table_item.label = "table"
        table_item.text = "Simple table text"
        table_item.page_no = 1
        table_item.bbox = None
        table_item.parent = None
        # No data attribute

        mock_doc = Mock()
        mock_doc.name = "test.pdf"
        mock_doc.origin = "/path/to/test.pdf"
        mock_doc.pages = [Mock()]
        mock_doc.iterate_items = Mock(return_value=iter([table_item]))

        mock_result = Mock()
        mock_result.document = mock_doc

        with patch.object(parser.converter, "convert", return_value=mock_result):
            result = await parser.parse(test_file)

        # Should still create table element
        table_elements = [e for e in result.elements if e.element_type == ElementType.TABLE]
        assert len(table_elements) == 1
        assert table_elements[0].content == "Simple table text"


class TestDoclingChunking:
    """Tests for document chunking functionality."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return DoclingParser(config={"chunk_size": 200, "chunk_overlap": 50})

    @pytest.fixture
    def sample_document(self, tmp_path):
        """Create sample parsed document with text and tables."""
        elements = [
            ParsedElement(
                element_id="doc1_heading_1",
                element_type=ElementType.HEADING,
                content="Chapter 1: Introduction",
                page_number=1,
            ),
            ParsedElement(
                element_id="doc1_text_1",
                element_type=ElementType.TEXT,
                content="This is a paragraph about petroleum engineering. " * 5,
                page_number=1,
            ),
            ParsedElement(
                element_id="doc1_table_1",
                element_type=ElementType.TABLE,
                content="Property | Value\nDensity | 850 kg/m³\nViscosity | 10 cP",
                formatted_content="<table><tr><th>Property</th><th>Value</th></tr></table>",
                metadata={
                    "format": "html",
                    "markdown": "| Property | Value |\n|----------|-------|",
                },
                page_number=1,
            ),
            ParsedElement(
                element_id="doc1_text_2",
                element_type=ElementType.TEXT,
                content="This is another paragraph with more content. " * 5,
                page_number=2,
            ),
        ]

        return ParsedDocument(
            document_id="test_doc",
            source_file=tmp_path / "test.pdf",
            parser_name="Docling",
            elements=elements,
            total_pages=2,
        )

    def test_chunk_document_success(self, parser, sample_document):
        """Test successful document chunking."""
        chunks = parser.chunk_document(sample_document)

        assert len(chunks) > 0
        for i, chunk in enumerate(chunks):
            assert chunk.document_id == "test_doc"
            assert chunk.chunk_index == i
            assert chunk.content
            assert len(chunk.element_ids) > 0
            assert chunk.token_count is not None
            assert chunk.metadata["parser"] == "Docling"
            assert chunk.metadata["chunk_method"] == "semantic"

    def test_chunk_table_gets_own_chunk(self, parser, sample_document):
        """Test that tables always get their own chunk."""
        chunks = parser.chunk_document(sample_document)

        # Find the table chunk
        table_chunks = [c for c in chunks if "doc1_table_1" in c.element_ids]
        assert len(table_chunks) == 1

        table_chunk = table_chunks[0]
        # Table chunk should only contain the table
        assert len(table_chunk.element_ids) == 1
        assert table_chunk.element_ids[0] == "doc1_table_1"

        # Should include HTML content
        assert "<table>" in table_chunk.content

    def test_chunk_empty_document(self, parser, tmp_path):
        """Test chunking fails for empty document."""
        empty_doc = ParsedDocument(
            document_id="empty",
            source_file=tmp_path / "empty.pdf",
            parser_name="Docling",
            elements=[],
        )

        with pytest.raises(ValueError, match="Cannot chunk empty document"):
            parser.chunk_document(empty_doc)

    def test_chunk_respects_size_limits(self, parser, sample_document):
        """Test chunks respect configured size limits."""
        chunk_size = parser.get_chunk_size()
        chunks = parser.chunk_document(sample_document)

        for chunk in chunks:
            # Tables can exceed size, but text chunks should be reasonable
            if "table" not in chunk.element_ids[0]:
                # Allow some overflow for semantic boundaries
                assert len(chunk.content) <= chunk_size * 2

    def test_chunk_with_overlap(self, parser):
        """Test chunking with overlap for text elements."""
        # Create document with long text elements
        elements = [
            ParsedElement(
                element_id="doc1_text_1",
                element_type=ElementType.TEXT,
                content="A" * 150,
                page_number=1,
            ),
            ParsedElement(
                element_id="doc1_text_2",
                element_type=ElementType.TEXT,
                content="B" * 150,
                page_number=1,
            ),
        ]

        doc = ParsedDocument(
            document_id="test",
            source_file=Path("/tmp/test.pdf"),
            parser_name="Docling",
            elements=elements,
        )

        chunks = parser.chunk_document(doc)

        # Should create multiple chunks
        assert len(chunks) > 1

    def test_chunk_preserves_section_context(self, parser, sample_document):
        """Test that chunks preserve parent section information."""
        chunks = parser.chunk_document(sample_document)

        # Check that chunks track their parent section
        for chunk in chunks:
            # Some chunks should have section context
            if chunk.parent_section:
                assert chunk.metadata.get("section") == chunk.parent_section

    def test_chunk_sequential_indexing(self, parser, sample_document):
        """Test that chunk indices are sequential."""
        chunks = parser.chunk_document(sample_document)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunk_includes_page_numbers(self, parser, sample_document):
        """Test that chunks include page number information."""
        chunks = parser.chunk_document(sample_document)

        for chunk in chunks:
            # Should have at least start page
            assert chunk.start_page is not None or len(chunk.element_ids) == 0


class TestDoclingHelperMethods:
    """Tests for helper methods."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return DoclingParser()

    def test_map_docling_type_heading(self, parser):
        """Test mapping Docling heading labels."""
        assert parser._map_docling_type("title") == ElementType.HEADING
        assert parser._map_docling_type("heading") == ElementType.HEADING
        assert parser._map_docling_type("section_header") == ElementType.HEADING

    def test_map_docling_type_table(self, parser):
        """Test mapping Docling table labels."""
        assert parser._map_docling_type("table") == ElementType.TABLE
        assert parser._map_docling_type("TABLE") == ElementType.TABLE

    def test_map_docling_type_figure(self, parser):
        """Test mapping Docling figure labels."""
        assert parser._map_docling_type("figure") == ElementType.FIGURE
        assert parser._map_docling_type("picture") == ElementType.FIGURE
        assert parser._map_docling_type("image") == ElementType.FIGURE

    def test_map_docling_type_list(self, parser):
        """Test mapping Docling list labels."""
        assert parser._map_docling_type("list") == ElementType.LIST
        assert parser._map_docling_type("list_item") == ElementType.LIST

    def test_map_docling_type_text(self, parser):
        """Test mapping Docling text labels."""
        assert parser._map_docling_type("text") == ElementType.TEXT
        assert parser._map_docling_type("paragraph") == ElementType.TEXT
        assert parser._map_docling_type("unknown") == ElementType.TEXT

    def test_create_chunk(self, parser):
        """Test chunk creation with metadata."""
        chunk = parser._create_chunk(
            doc_id="test_doc",
            chunk_index=0,
            content="Test content",
            element_ids=["elem1", "elem2"],
            start_page=1,
            end_page=2,
            parent_section="Section 1",
        )

        assert isinstance(chunk, DocumentChunk)
        assert chunk.chunk_id == "test_doc_chunk_0"
        assert chunk.document_id == "test_doc"
        assert chunk.content == "Test content"
        assert chunk.element_ids == ["elem1", "elem2"]
        assert chunk.chunk_index == 0
        assert chunk.start_page == 1
        assert chunk.end_page == 2
        assert chunk.parent_section == "Section 1"
        assert chunk.metadata["parser"] == "Docling"
        assert chunk.metadata["chunk_method"] == "semantic"
        assert chunk.metadata["section"] == "Section 1"
        assert chunk.token_count > 0

    def test_create_chunk_without_section(self, parser):
        """Test chunk creation without parent section."""
        chunk = parser._create_chunk(
            doc_id="test_doc",
            chunk_index=1,
            content="Test content",
            element_ids=["elem1"],
            start_page=1,
            end_page=1,
            parent_section=None,
        )

        assert "section" not in chunk.metadata
        assert chunk.parent_section is None


class TestDoclingIntegration:
    """Integration tests for full parsing workflow."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return DoclingParser(config={"chunk_size": 500, "chunk_overlap": 100})

    @pytest.mark.asyncio
    async def test_full_workflow_with_tables(self, parser, tmp_path):
        """Test complete workflow: parse and chunk document with tables."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 test content")

        # Create comprehensive mock document
        heading = Mock()
        heading.label = "title"
        heading.text = "Petroleum Properties"
        heading.page_no = 1
        heading.bbox = None
        heading.parent = None

        text = Mock()
        text.label = "text"
        text.text = "This document describes key petroleum properties. " * 10
        text.page_no = 1
        text.bbox = None
        text.parent = None

        table = Mock()
        table.label = "table"
        table.text = "Property | Value\nDensity | 850 kg/m³"
        table.page_no = 1
        table.bbox = None
        table.parent = None
        table.data = Mock()
        table.data.export_to_html = Mock(return_value="<table>HTML</table>")
        table.data.export_to_markdown = Mock(return_value="| Property | Value |")

        mock_doc = Mock()
        mock_doc.name = "test.pdf"
        mock_doc.origin = str(test_file)
        mock_doc.pages = [Mock()]
        mock_doc.iterate_items = Mock(return_value=iter([heading, text, table]))

        mock_result = Mock()
        mock_result.document = mock_doc

        with patch.object(parser.converter, "convert", return_value=mock_result):
            # Parse
            parsed_doc = await parser.parse(test_file)

            # Verify parsing
            assert parsed_doc.document_id == "test"
            assert len(parsed_doc.elements) == 3

            # Chunk
            chunks = parser.chunk_document(parsed_doc)

            # Verify chunking
            assert len(chunks) > 0

            # Verify table chunk exists and contains table content
            table_chunks = [
                c for c in chunks if any("table" in eid for eid in c.element_ids)
            ]
            assert len(table_chunks) > 0
            # Table content should contain either HTML or structured text
            assert (
                "<table>" in table_chunks[0].content
                or "Property" in table_chunks[0].content
            )
