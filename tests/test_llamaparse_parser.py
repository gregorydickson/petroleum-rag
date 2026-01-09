"""Tests for LlamaParse parser.

Tests cover:
- Parser initialization and configuration
- Document parsing with mock LlamaParse API
- Table extraction and preservation
- Element extraction (text, headings, tables, figures)
- Intelligent chunking that respects boundaries
- Error handling and edge cases
"""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from models import DocumentChunk, ElementType, ParsedDocument, ParsedElement
from parsers.llamaparse_parser import LlamaParseParser


class TestLlamaParseInitialization:
    """Tests for parser initialization and configuration."""

    @patch("parsers.llamaparse_parser.settings")
    def test_initialization_success(self, mock_settings):
        """Test successful parser initialization."""
        mock_settings.llama_cloud_api_key = "llx-test-key"
        mock_settings.debug = False

        parser = LlamaParseParser()

        assert parser.name == "LlamaParse"
        assert parser.parser is not None

    @patch("parsers.llamaparse_parser.settings")
    def test_initialization_missing_api_key(self, mock_settings):
        """Test initialization fails without API key."""
        mock_settings.llama_cloud_api_key = ""

        with pytest.raises(ValueError, match="LLAMA_CLOUD_API_KEY not set"):
            LlamaParseParser()

    @patch("parsers.llamaparse_parser.settings")
    def test_initialization_with_config(self, mock_settings):
        """Test initialization with custom config."""
        mock_settings.llama_cloud_api_key = "llx-test-key"
        mock_settings.debug = False

        config = {"chunk_size": 500, "chunk_overlap": 100}
        parser = LlamaParseParser(config=config)

        assert parser.name == "LlamaParse"
        assert parser.config == config
        assert parser.get_chunk_size() == 500
        assert parser.get_chunk_overlap() == 100

    @patch("parsers.llamaparse_parser.settings")
    def test_initialization_default_config(self, mock_settings):
        """Test initialization uses default config values."""
        mock_settings.llama_cloud_api_key = "llx-test-key"
        mock_settings.debug = False

        parser = LlamaParseParser()

        # Should use base class defaults
        assert parser.get_chunk_size() == 1000
        assert parser.get_chunk_overlap() == 200


class TestLlamaParseParsing:
    """Tests for document parsing functionality."""

    @pytest.fixture
    @patch("parsers.llamaparse_parser.settings")
    def parser(self, mock_settings):
        """Create parser instance."""
        mock_settings.llama_cloud_api_key = "llx-test-key"
        mock_settings.debug = False
        return LlamaParseParser()

    @pytest.fixture
    def mock_llamaparse_result(self):
        """Create mock LlamaParse JobResult."""
        # Create mock page with markdown content
        mock_page1 = Mock()
        mock_page1.md = """# Introduction to Petroleum Engineering

This document covers petroleum engineering fundamentals including drilling operations and reservoir management.

## Drilling Operations

Key aspects of drilling include:
- Well planning
- Drilling fluids
- Formation evaluation

### Common Drilling Fluids

| Fluid Type | Density (ppg) | Viscosity (cP) | Application |
|-----------|---------------|----------------|-------------|
| Water-based | 8.5-12.0 | 10-50 | Standard drilling |
| Oil-based | 10.0-16.0 | 20-100 | High-pressure zones |
| Synthetic | 9.0-15.0 | 15-80 | Environmentally sensitive |

## Reservoir Management

Reservoir management involves monitoring and optimizing production rates."""

        mock_page1.text = "Plain text version..."
        mock_page1.images = []

        # Create mock page 2 with images
        mock_page2 = Mock()
        mock_page2.md = "## Well Completion\n\nFigure 1 shows the completion design."
        mock_page2.text = "Plain text..."
        mock_page2.images = [Mock(), Mock()]  # Two images

        # Create mock result
        mock_result = Mock()
        mock_result.pages = [mock_page1, mock_page2]

        return mock_result

    @pytest.fixture
    def temp_pdf(self, tmp_path):
        """Create temporary test PDF file."""
        pdf_file = tmp_path / "test_petroleum.pdf"
        pdf_file.write_text("Mock PDF content")
        return pdf_file

    @pytest.mark.asyncio
    async def test_parse_success(self, parser, temp_pdf, mock_llamaparse_result):
        """Test successful document parsing."""
        # Mock the LlamaParse.aparse at class level
        with patch("parsers.llamaparse_parser.LlamaParse.aparse", new_callable=AsyncMock) as mock_aparse:
            mock_aparse.return_value = mock_llamaparse_result

            result = await parser.parse(temp_pdf)

            assert isinstance(result, ParsedDocument)
            assert result.document_id == "test_petroleum"
            assert result.source_file == temp_pdf
            assert result.parser_name == "LlamaParse"
            assert result.error is None
            assert result.total_pages == 2
            assert result.parse_time_seconds > 0
            assert len(result.elements) > 0

    @pytest.mark.asyncio
    async def test_parse_extracts_headings(self, parser, temp_pdf, mock_llamaparse_result):
        """Test that parser correctly identifies headings."""
        with patch("parsers.llamaparse_parser.LlamaParse.aparse", new_callable=AsyncMock) as mock_aparse:
            mock_aparse.return_value = mock_llamaparse_result

            result = await parser.parse(temp_pdf)

            # Find heading elements
            headings = [e for e in result.elements if e.element_type == ElementType.HEADING]

            assert len(headings) >= 3  # We have 3 headings in mock
            assert any("Introduction to Petroleum Engineering" in h.content for h in headings)
            assert any("Drilling Operations" in h.content for h in headings)
            assert any("Reservoir Management" in h.content for h in headings)

            # Check heading metadata
            h1_headings = [h for h in headings if h.metadata.get("level") == "1"]
            assert len(h1_headings) >= 1

    @pytest.mark.asyncio
    async def test_parse_extracts_tables(self, parser, temp_pdf, mock_llamaparse_result):
        """Test that parser correctly identifies and preserves tables."""
        with patch("parsers.llamaparse_parser.LlamaParse.aparse", new_callable=AsyncMock) as mock_aparse:
            mock_aparse.return_value = mock_llamaparse_result

            result = await parser.parse(temp_pdf)

            # Find table elements
            tables = [e for e in result.elements if e.element_type == ElementType.TABLE]

            assert len(tables) >= 1
            table = tables[0]

            # Verify table content
            assert "Fluid Type" in table.content
            assert "Density (ppg)" in table.content
            assert "Water-based" in table.content
            assert "Oil-based" in table.content

            # Verify table is markdown formatted
            assert "|" in table.content
            assert table.metadata.get("format") == "markdown"

    @pytest.mark.asyncio
    async def test_parse_extracts_text(self, parser, temp_pdf, mock_llamaparse_result):
        """Test that parser extracts text elements."""
        with patch("parsers.llamaparse_parser.LlamaParse.aparse", new_callable=AsyncMock) as mock_aparse:
            mock_aparse.return_value = mock_llamaparse_result

            result = await parser.parse(temp_pdf)

            # Find text elements
            text_elements = [e for e in result.elements if e.element_type == ElementType.TEXT]

            assert len(text_elements) > 0
            assert any("petroleum engineering fundamentals" in t.content.lower() for t in text_elements)

    @pytest.mark.asyncio
    async def test_parse_extracts_figures(self, parser, temp_pdf, mock_llamaparse_result):
        """Test that parser identifies figures from images."""
        with patch("parsers.llamaparse_parser.LlamaParse.aparse", new_callable=AsyncMock) as mock_aparse:
            mock_aparse.return_value = mock_llamaparse_result

            result = await parser.parse(temp_pdf)

            # Find figure elements
            figures = [e for e in result.elements if e.element_type == ElementType.FIGURE]

            assert len(figures) == 2  # We have 2 images in page 2
            assert all(f.page_number == 2 for f in figures)

    @pytest.mark.asyncio
    async def test_parse_preserves_section_hierarchy(self, parser, temp_pdf, mock_llamaparse_result):
        """Test that parser maintains parent section relationships."""
        with patch("parsers.llamaparse_parser.LlamaParse.aparse", new_callable=AsyncMock) as mock_aparse:
            mock_aparse.return_value = mock_llamaparse_result

            result = await parser.parse(temp_pdf)

            # Find elements under "Drilling Operations" section
            drilling_elements = [
                e for e in result.elements
                if e.parent_section and "Drilling Operations" in e.parent_section
            ]

            # Should have heading, text, and table under this section
            assert len(drilling_elements) > 0

    @pytest.mark.asyncio
    async def test_parse_handles_file_not_found(self, parser):
        """Test parsing handles missing files gracefully."""
        fake_path = Path("/nonexistent/file.pdf")

        with pytest.raises(FileNotFoundError):
            await parser.parse(fake_path)

    @pytest.mark.asyncio
    async def test_parse_handles_unsupported_file(self, parser, tmp_path):
        """Test parsing rejects unsupported file types."""
        txt_file = tmp_path / "test.xyz"
        txt_file.write_text("content")

        with pytest.raises(ValueError, match="Unsupported file extension"):
            await parser.parse(txt_file)

    @pytest.mark.asyncio
    async def test_parse_handles_api_error(self, parser, temp_pdf):
        """Test parsing handles LlamaParse API errors gracefully."""
        # Mock API failure
        with patch("parsers.llamaparse_parser.LlamaParse.aparse", new_callable=AsyncMock) as mock_aparse:
            mock_aparse.side_effect = Exception("API Error")

            result = await parser.parse(temp_pdf)

            assert isinstance(result, ParsedDocument)
            assert result.error is not None
            assert "API Error" in result.error
            assert len(result.elements) == 0

    @pytest.mark.asyncio
    async def test_parse_handles_empty_result(self, parser, temp_pdf):
        """Test parsing handles empty results from API."""
        # Mock empty result
        mock_empty_result = Mock()
        mock_empty_result.pages = []

        with patch("parsers.llamaparse_parser.LlamaParse.aparse", new_callable=AsyncMock) as mock_aparse:
            mock_aparse.return_value = mock_empty_result

            result = await parser.parse(temp_pdf)

            assert isinstance(result, ParsedDocument)
            assert len(result.elements) == 0
            assert result.error is None  # Not an error, just empty


class TestLlamaParseChunking:
    """Tests for document chunking functionality."""

    @pytest.fixture
    @patch("parsers.llamaparse_parser.settings")
    def parser(self, mock_settings):
        """Create parser instance with small chunks for testing."""
        mock_settings.llama_cloud_api_key = "llx-test-key"
        mock_settings.debug = False
        return LlamaParseParser(config={"chunk_size": 200, "chunk_overlap": 50})

    @pytest.fixture
    def parsed_doc_with_elements(self, tmp_path):
        """Create a parsed document with various elements."""
        elements = [
            ParsedElement(
                element_id="doc1_heading_1_1",
                element_type=ElementType.HEADING,
                content="Introduction",
                page_number=1,
                parent_section="Introduction",
            ),
            ParsedElement(
                element_id="doc1_text_1_2",
                element_type=ElementType.TEXT,
                content="This is a paragraph about petroleum engineering. " * 3,
                page_number=1,
                parent_section="Introduction",
            ),
            ParsedElement(
                element_id="doc1_table_1_3",
                element_type=ElementType.TABLE,
                content="| Col1 | Col2 |\n|------|------|\n| A | B |\n| C | D |",
                page_number=1,
                parent_section="Introduction",
            ),
            ParsedElement(
                element_id="doc1_text_1_4",
                element_type=ElementType.TEXT,
                content="Another paragraph with more details. " * 4,
                page_number=1,
                parent_section="Introduction",
            ),
        ]

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_text("mock")

        return ParsedDocument(
            document_id="doc1",
            source_file=pdf_file,
            parser_name="LlamaParse",
            elements=elements,
            total_pages=1,
        )

    def test_chunk_document_success(self, parser, parsed_doc_with_elements):
        """Test successful document chunking."""
        chunks = parser.chunk_document(parsed_doc_with_elements)

        assert len(chunks) > 0
        assert all(isinstance(c, DocumentChunk) for c in chunks)
        assert all(c.document_id == "doc1" for c in chunks)

    def test_chunk_document_sequential_indices(self, parser, parsed_doc_with_elements):
        """Test chunks have sequential indices."""
        chunks = parser.chunk_document(parsed_doc_with_elements)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.chunk_id == f"doc1_chunk_{i}"

    def test_chunk_document_tables_not_split(self, parser, parsed_doc_with_elements):
        """Test that tables are never split across chunks."""
        chunks = parser.chunk_document(parsed_doc_with_elements)

        # Find chunk containing table
        table_chunks = [c for c in chunks if "doc1_table_1_3" in c.element_ids]

        assert len(table_chunks) == 1  # Table appears in exactly one chunk
        table_chunk = table_chunks[0]

        # Table should be complete in this chunk
        assert "| Col1 | Col2 |" in table_chunk.content
        assert "| A | B |" in table_chunk.content

    def test_chunk_document_preserves_metadata(self, parser, parsed_doc_with_elements):
        """Test chunks preserve important metadata."""
        chunks = parser.chunk_document(parsed_doc_with_elements)

        for chunk in chunks:
            assert "element_count" in chunk.metadata
            assert chunk.parent_section == "Introduction"
            assert chunk.start_page == 1
            assert chunk.end_page == 1
            assert chunk.token_count is not None
            assert chunk.token_count > 0

    def test_chunk_document_respects_chunk_size(self, parser, parsed_doc_with_elements):
        """Test chunks respect configured size limits (where possible)."""
        chunks = parser.chunk_document(parsed_doc_with_elements)

        # Most chunks should be under chunk_size (except tables which can't be split)
        text_chunks = [c for c in chunks if "table" not in c.metadata.get("count_table", "0")]

        for chunk in text_chunks:
            # Allow some flexibility for overlap
            assert len(chunk.content) <= parser.get_chunk_size() + parser.get_chunk_overlap()

    def test_chunk_document_empty_document_raises_error(self, parser, tmp_path):
        """Test chunking empty document raises ValueError."""
        pdf_file = tmp_path / "empty.pdf"
        pdf_file.write_text("mock")

        empty_doc = ParsedDocument(
            document_id="empty",
            source_file=pdf_file,
            parser_name="LlamaParse",
            elements=[],
        )

        with pytest.raises(ValueError, match="has no elements to chunk"):
            parser.chunk_document(empty_doc)

    def test_chunk_document_large_table_gets_own_chunk(self, parser, tmp_path):
        """Test that large tables get dedicated chunks even if oversized."""
        # Create a large table that exceeds chunk_size
        large_table_content = "| Col1 | Col2 | Col3 |\n" + ("|------|------|------|\n" + "| A | B | C |\n") * 20

        elements = [
            ParsedElement(
                element_id="doc1_table_1_1",
                element_type=ElementType.TABLE,
                content=large_table_content,
                page_number=1,
            ),
        ]

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_text("mock")

        doc = ParsedDocument(
            document_id="doc1",
            source_file=pdf_file,
            parser_name="LlamaParse",
            elements=elements,
        )

        chunks = parser.chunk_document(doc)

        # Should create one chunk for the table
        assert len(chunks) == 1
        assert "doc1_table_1_1" in chunks[0].element_ids
        assert len(chunks[0].content) > parser.get_chunk_size()  # Oversized is OK for tables

    def test_chunk_document_tracks_element_types(self, parser, parsed_doc_with_elements):
        """Test chunk metadata tracks element type counts."""
        chunks = parser.chunk_document(parsed_doc_with_elements)

        # Find chunk with table
        table_chunks = [c for c in chunks if "count_table" in c.metadata]

        assert len(table_chunks) > 0
        table_chunk = table_chunks[0]
        assert int(table_chunk.metadata["count_table"]) == 1


class TestLlamaParseMarkdownParsing:
    """Tests for markdown content parsing."""

    @pytest.fixture
    @patch("parsers.llamaparse_parser.settings")
    def parser(self, mock_settings):
        """Create parser instance."""
        mock_settings.llama_cloud_api_key = "llx-test-key"
        mock_settings.debug = False
        return LlamaParseParser()

    def test_parse_markdown_identifies_headings(self, parser):
        """Test markdown parser identifies different heading levels."""
        markdown = """# Level 1 Heading
## Level 2 Heading
### Level 3 Heading"""

        elements = parser._parse_markdown_content(markdown, "doc1", 1)

        headings = [e for e in elements if e.element_type == ElementType.HEADING]
        assert len(headings) == 3
        assert headings[0].metadata["level"] == "1"
        assert headings[1].metadata["level"] == "2"
        assert headings[2].metadata["level"] == "3"

    def test_parse_markdown_identifies_code_blocks(self, parser):
        """Test markdown parser identifies code blocks."""
        markdown = """Some text

```python
def hello():
    print("world")
```

More text"""

        elements = parser._parse_markdown_content(markdown, "doc1", 1)

        code_elements = [e for e in elements if e.element_type == ElementType.CODE]
        assert len(code_elements) == 1
        assert "def hello():" in code_elements[0].content

    def test_parse_markdown_identifies_tables(self, parser):
        """Test markdown parser identifies tables."""
        markdown = """Some text

| Header 1 | Header 2 |
|----------|----------|
| Cell 1   | Cell 2   |

More text"""

        elements = parser._parse_markdown_content(markdown, "doc1", 1)

        tables = [e for e in elements if e.element_type == ElementType.TABLE]
        assert len(tables) == 1
        assert "Header 1" in tables[0].content
        assert "Cell 1" in tables[0].content

    def test_parse_markdown_separates_paragraphs(self, parser):
        """Test markdown parser separates paragraphs."""
        markdown = """First paragraph.

Second paragraph.

Third paragraph."""

        elements = parser._parse_markdown_content(markdown, "doc1", 1)

        text_elements = [e for e in elements if e.element_type == ElementType.TEXT]
        assert len(text_elements) == 3

    def test_parse_markdown_preserves_section_context(self, parser):
        """Test markdown parser tracks parent sections."""
        markdown = """# Main Section

Some content under main.

## Subsection

Content under subsection."""

        elements = parser._parse_markdown_content(markdown, "doc1", 1)

        # Find elements after "Main Section"
        main_section_idx = next(
            i for i, e in enumerate(elements)
            if e.element_type == ElementType.HEADING and "Main Section" in e.content
        )

        # Elements after main section should have it as parent
        subsection_idx = next(
            i for i, e in enumerate(elements)
            if e.element_type == ElementType.HEADING and "Subsection" in e.content
        )

        # Text between main and subsection should have main as parent
        for i in range(main_section_idx + 1, subsection_idx):
            if elements[i].parent_section:
                assert "Main Section" in elements[i].parent_section
