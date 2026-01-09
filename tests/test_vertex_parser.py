"""Tests for Vertex Document AI parser.

Tests cover:
- Parser initialization and configuration validation
- Document parsing with both digital and scanned PDFs
- Element extraction (text, tables, layout)
- Chunking strategies based on layout analysis
- Error handling and edge cases
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from google.cloud import documentai_v1 as documentai

from models import ElementType, ParsedDocument
from parsers.vertex_parser import VertexDocAIParser


class TestVertexDocAIParserInitialization:
    """Tests for parser initialization and configuration."""

    @patch.dict(
        os.environ,
        {
            "GOOGLE_CLOUD_PROJECT": "test-project",
            "VERTEX_DOCAI_PROCESSOR_ID": "test-processor-id",
            "VERTEX_DOCAI_LOCATION": "us",
        },
    )
    @patch("parsers.vertex_parser.documentai.DocumentProcessorServiceClient")
    def test_initialization_success(self, mock_client_class):
        """Test successful parser initialization with valid config."""
        parser = VertexDocAIParser()

        assert parser.name == "VertexDocAI"
        assert parser.project_id == "test-project"
        assert parser.processor_id == "test-processor-id"
        assert parser.location == "us"
        assert "test-project" in parser.processor_name
        assert "test-processor-id" in parser.processor_name

    @patch.dict(os.environ, {}, clear=True)
    def test_initialization_missing_project(self):
        """Test initialization fails without project ID."""
        with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT"):
            VertexDocAIParser()

    @patch.dict(
        os.environ,
        {
            "GOOGLE_CLOUD_PROJECT": "test-project",
        },
        clear=True,
    )
    def test_initialization_missing_processor_id(self):
        """Test initialization fails without processor ID."""
        with pytest.raises(ValueError, match="VERTEX_DOCAI_PROCESSOR_ID"):
            VertexDocAIParser()

    @patch.dict(
        os.environ,
        {
            "GOOGLE_CLOUD_PROJECT": "test-project",
            "VERTEX_DOCAI_PROCESSOR_ID": "test-processor-id",
            "GOOGLE_APPLICATION_CREDENTIALS": "/nonexistent/path/key.json",
        },
    )
    def test_initialization_invalid_credentials_path(self):
        """Test initialization fails with invalid credentials path."""
        with pytest.raises(ValueError, match="Google credentials file not found"):
            VertexDocAIParser()


class TestVertexDocAIParsing:
    """Tests for document parsing functionality."""

    @pytest.fixture
    def parser(self):
        """Create parser instance with mocked client."""
        with patch.dict(
            os.environ,
            {
                "GOOGLE_CLOUD_PROJECT": "test-project",
                "VERTEX_DOCAI_PROCESSOR_ID": "test-processor-id",
                "VERTEX_DOCAI_LOCATION": "us",
            },
        ):
            with patch("parsers.vertex_parser.documentai.DocumentProcessorServiceClient"):
                return VertexDocAIParser()

    @pytest.fixture
    def mock_document_ai_response(self):
        """Create mock Document AI response."""
        # Create mock document
        mock_doc = Mock(spec=documentai.Document)
        mock_doc.text = "Sample document text with multiple paragraphs.\n\nThis is a test."

        # Create mock page with paragraphs
        mock_page = Mock()
        mock_page.paragraphs = []
        mock_page.tables = []
        mock_page.blocks = []

        # Create mock paragraph
        mock_para = Mock()
        mock_layout = Mock()
        mock_segment = Mock()
        mock_segment.start_index = 0
        mock_segment.end_index = 50
        mock_layout.text_anchor.text_segments = [mock_segment]
        mock_layout.bounding_poly.vertices = [
            Mock(x=10, y=20),
            Mock(x=100, y=20),
            Mock(x=100, y=50),
            Mock(x=10, y=50),
        ]
        mock_para.layout = mock_layout
        mock_page.paragraphs.append(mock_para)

        mock_doc.pages = [mock_page]

        # Create mock result
        mock_result = Mock()
        mock_result.document = mock_doc

        return mock_result

    @pytest.mark.asyncio
    async def test_parse_pdf_success(self, parser, mock_document_ai_response, tmp_path):
        """Test successful PDF parsing."""
        # Create test PDF file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 test content")

        # Mock the client's process_document method
        parser.client.process_document = Mock(return_value=mock_document_ai_response)

        # Parse document
        result = await parser.parse(test_file)

        # Verify result
        assert isinstance(result, ParsedDocument)
        assert result.document_id == "test"
        assert result.parser_name == "VertexDocAI"
        assert result.source_file == test_file
        assert len(result.elements) > 0
        assert result.error is None
        assert result.parse_time_seconds > 0

        # Verify client was called
        parser.client.process_document.assert_called_once()

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
    async def test_parse_api_error(self, parser, tmp_path):
        """Test handling of Document AI API errors."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 test content")

        # Mock API error
        parser.client.process_document = Mock(
            side_effect=Exception("API quota exceeded")
        )

        # Parse should return document with error
        result = await parser.parse(test_file)

        assert isinstance(result, ParsedDocument)
        assert result.error is not None
        assert "API quota exceeded" in result.error
        assert len(result.elements) == 0

    @pytest.mark.asyncio
    async def test_extract_tables(self, parser, tmp_path):
        """Test table extraction from Document AI response."""
        # Create test file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 test content")

        # Create mock with table
        mock_doc = Mock(spec=documentai.Document)
        mock_doc.text = "Header1 | Header2\nValue1 | Value2"

        mock_page = Mock()
        mock_page.paragraphs = []
        mock_page.blocks = []

        # Create mock table
        mock_table = Mock()

        # Header row
        mock_header_row = Mock()
        mock_header_cell1 = Mock()
        mock_header_layout1 = Mock()
        mock_segment1 = Mock()
        mock_segment1.start_index = 0
        mock_segment1.end_index = 7
        mock_header_layout1.text_anchor.text_segments = [mock_segment1]
        mock_header_cell1.layout = mock_header_layout1

        mock_header_cell2 = Mock()
        mock_header_layout2 = Mock()
        mock_segment2 = Mock()
        mock_segment2.start_index = 10
        mock_segment2.end_index = 17
        mock_header_layout2.text_anchor.text_segments = [mock_segment2]
        mock_header_cell2.layout = mock_header_layout2

        mock_header_row.cells = [mock_header_cell1, mock_header_cell2]
        mock_table.header_rows = [mock_header_row]

        # Body row
        mock_body_row = Mock()
        mock_body_cell1 = Mock()
        mock_body_layout1 = Mock()
        mock_segment3 = Mock()
        mock_segment3.start_index = 18
        mock_segment3.end_index = 24
        mock_body_layout1.text_anchor.text_segments = [mock_segment3]
        mock_body_cell1.layout = mock_body_layout1

        mock_body_cell2 = Mock()
        mock_body_layout2 = Mock()
        mock_segment4 = Mock()
        mock_segment4.start_index = 27
        mock_segment4.end_index = 33
        mock_body_layout2.text_anchor.text_segments = [mock_segment4]
        mock_body_cell2.layout = mock_body_layout2

        mock_body_row.cells = [mock_body_cell1, mock_body_cell2]
        mock_table.body_rows = [mock_body_row]

        mock_page.tables = [mock_table]
        mock_doc.pages = [mock_page]

        mock_result = Mock()
        mock_result.document = mock_doc

        parser.client.process_document = Mock(return_value=mock_result)

        # Parse document
        result = await parser.parse(test_file)

        # Verify table was extracted
        table_elements = [e for e in result.elements if e.element_type == ElementType.TABLE]
        assert len(table_elements) > 0

        # Check table content
        table = table_elements[0]
        assert "Header1" in table.content
        assert "Header2" in table.content
        assert "Value1" in table.content
        assert "Value2" in table.content


class TestVertexDocAIChunking:
    """Tests for document chunking functionality."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        with patch.dict(
            os.environ,
            {
                "GOOGLE_CLOUD_PROJECT": "test-project",
                "VERTEX_DOCAI_PROCESSOR_ID": "test-processor-id",
                "VERTEX_DOCAI_LOCATION": "us",
            },
        ):
            with patch("parsers.vertex_parser.documentai.DocumentProcessorServiceClient"):
                return VertexDocAIParser(config={"chunk_size": 200, "chunk_overlap": 50})

    @pytest.fixture
    def sample_document(self, tmp_path):
        """Create sample parsed document."""
        from models import ParsedElement

        elements = [
            ParsedElement(
                element_id="doc1_para_1_0",
                element_type=ElementType.TEXT,
                content="This is the first paragraph. " * 10,
                page_number=1,
            ),
            ParsedElement(
                element_id="doc1_para_1_1",
                element_type=ElementType.TEXT,
                content="This is the second paragraph. " * 10,
                page_number=1,
            ),
            ParsedElement(
                element_id="doc1_table_1_0",
                element_type=ElementType.TABLE,
                content="Header1 | Header2\nValue1 | Value2",
                page_number=1,
            ),
        ]

        return ParsedDocument(
            document_id="test_doc",
            source_file=tmp_path / "test.pdf",
            parser_name="VertexDocAI",
            elements=elements,
            total_pages=1,
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

    def test_chunk_empty_document(self, parser, tmp_path):
        """Test chunking fails for empty document."""
        empty_doc = ParsedDocument(
            document_id="empty",
            source_file=tmp_path / "empty.pdf",
            parser_name="VertexDocAI",
            elements=[],
        )

        with pytest.raises(ValueError, match="Cannot chunk empty document"):
            parser.chunk_document(empty_doc)

    def test_chunk_document_with_error(self, parser, tmp_path):
        """Test chunking fails for document with error."""
        error_doc = ParsedDocument(
            document_id="error",
            source_file=tmp_path / "error.pdf",
            parser_name="VertexDocAI",
            elements=[],
            error="Parse error",
        )

        with pytest.raises(ValueError, match="Cannot chunk document with error"):
            parser.chunk_document(error_doc)

    def test_chunk_respects_size_limits(self, parser, sample_document):
        """Test chunks respect configured size limits."""
        chunk_size = parser.get_chunk_size()
        chunks = parser.chunk_document(sample_document)

        for chunk in chunks:
            # Allow small overflow for semantic boundaries
            assert len(chunk.content) <= chunk_size * 1.5

    def test_chunk_with_overlap(self, parser, sample_document):
        """Test chunking with overlap."""
        chunks = parser.chunk_document(sample_document)

        if len(chunks) > 1:
            # Check that consecutive chunks may have overlapping content
            # (this is expected behavior)
            assert chunks[0].content != chunks[1].content


class TestVertexDocAIHelperMethods:
    """Tests for helper methods."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        with patch.dict(
            os.environ,
            {
                "GOOGLE_CLOUD_PROJECT": "test-project",
                "VERTEX_DOCAI_PROCESSOR_ID": "test-processor-id",
            },
        ):
            with patch("parsers.vertex_parser.documentai.DocumentProcessorServiceClient"):
                return VertexDocAIParser()

    def test_get_mime_type_pdf(self, parser):
        """Test MIME type detection for PDF."""
        assert parser._get_mime_type(Path("test.pdf")) == "application/pdf"

    def test_get_mime_type_docx(self, parser):
        """Test MIME type detection for DOCX."""
        mime_type = parser._get_mime_type(Path("test.docx"))
        assert "wordprocessing" in mime_type

    def test_get_mime_type_unsupported(self, parser):
        """Test MIME type detection fails for unsupported type."""
        with pytest.raises(ValueError, match="Unsupported file type"):
            parser._get_mime_type(Path("test.xyz"))

    def test_get_bbox(self, parser):
        """Test bounding box extraction."""
        mock_layout = Mock()
        mock_layout.bounding_poly.vertices = [
            Mock(x=10, y=20),
            Mock(x=100, y=20),
            Mock(x=100, y=50),
            Mock(x=10, y=50),
        ]

        bbox = parser._get_bbox(mock_layout)

        assert bbox == [10, 20, 100, 50]

    def test_get_bbox_none(self, parser):
        """Test bounding box extraction returns None if not available."""
        mock_layout = Mock()
        mock_layout.bounding_poly = None

        bbox = parser._get_bbox(mock_layout)

        assert bbox is None

    def test_get_layout_text(self, parser):
        """Test text extraction from layout."""
        full_text = "This is a sample document text."
        mock_layout = Mock()
        mock_segment = Mock()
        mock_segment.start_index = 0
        mock_segment.end_index = 20
        mock_layout.text_anchor.text_segments = [mock_segment]

        text = parser._get_layout_text(mock_layout, full_text)

        assert text == "This is a sample doc"
