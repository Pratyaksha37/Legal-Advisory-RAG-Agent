import pytest

from src.chunking import ChunkResult, DocumentChunk, HierarchicalChunker
from src.config.constants import LegalDocumentType
from src.schema.legal_document import LegalDocument


class TestDocumentChunk:
    def test_minimal_chunk(self):
        chunk = DocumentChunk(chunk_id="chk/001", document_id="doc/001", text="Content")
        assert chunk.hierarchy_level == 0
        assert chunk.chunk_type == "fallback"
        assert chunk.parent_id is None

    def test_full_chunk(self):
        chunk = DocumentChunk(
            chunk_id="chk/001",
            document_id="doc/001",
            parent_id="chk/000",
            text="Section text",
            hierarchy_level=2,
            heading="Section 1",
            chunk_type="section",
            metadata={"source": "IPC"},
        )
        assert chunk.parent_id == "chk/000"
        assert chunk.heading == "Section 1"


class TestChunkResult:
    def test_empty_result(self):
        result = ChunkResult(chunks=[], document_id="doc/001", total_chunks=0)
        assert result.total_chunks == 0

    def test_with_chunks(self):
        chunks = [DocumentChunk(chunk_id="1", document_id="doc/001", text="A")]
        result = ChunkResult(chunks=chunks, document_id="doc/001", total_chunks=1)
        assert result.total_chunks == 1


class TestHierarchicalChunker:
    def test_empty_text(self):
        doc = LegalDocument(id="doc/001", doc_type=LegalDocumentType.CUSTOM, title="Empty", text="")
        chunker = HierarchicalChunker()
        result = chunker.chunk(doc)
        assert result.total_chunks == 0

    def test_section_chunking(self, sample_legal_text):
        doc = LegalDocument(id="doc/001", doc_type=LegalDocumentType.ACT, title="Test Act", text=sample_legal_text)
        chunker = HierarchicalChunker()
        result = chunker.chunk(doc)
        assert result.total_chunks > 0
        sections = [c for c in result.chunks if c.chunk_type == "section"]
        assert len(sections) >= 1

    def test_parent_child_relationships(self, sample_legal_text):
        doc = LegalDocument(id="doc/001", doc_type=LegalDocumentType.ACT, title="Test", text=sample_legal_text)
        chunker = HierarchicalChunker()
        result = chunker.chunk(doc)
        for c in result.chunks:
            if c.parent_id:
                parent = next((p for p in result.chunks if p.chunk_id == c.parent_id), None)
                if parent:
                    assert c.hierarchy_level > parent.hierarchy_level

    def test_fallback_paragraph_chunking(self):
        text = "First paragraph about law.\n\nSecond paragraph about crime.\n\nThird paragraph about punishment."
        doc = LegalDocument(id="doc/001", doc_type=LegalDocumentType.CUSTOM, title="Plain", text=text)
        chunker = HierarchicalChunker()
        result = chunker.chunk(doc)
        assert result.total_chunks > 0
        for c in result.chunks:
            assert c.chunk_type == "fallback"
            assert c.parent_id is None

    def test_hierarchy_levels_strictly_increasing(self, sample_legal_text):
        doc = LegalDocument(id="doc/001", doc_type=LegalDocumentType.ACT, title="Test", text=sample_legal_text)
        chunker = HierarchicalChunker()
        result = chunker.chunk(doc)
        for c in result.chunks:
            if c.parent_id:
                parent = next((p for p in result.chunks if p.chunk_id == c.parent_id), None)
                if parent:
                    assert c.hierarchy_level > parent.hierarchy_level

    def test_chunk_ids_unique(self, sample_legal_text):
        doc = LegalDocument(id="doc/001", doc_type=LegalDocumentType.ACT, title="Test", text=sample_legal_text)
        chunker = HierarchicalChunker()
        result = chunker.chunk(doc)
        ids = [c.chunk_id for c in result.chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_text_no_short_chunks(self):
        text = "A\n\nB\n\nC\n\nD"
        doc = LegalDocument(id="doc/001", doc_type=LegalDocumentType.CUSTOM, title="Short", text=text)
        chunker = HierarchicalChunker()
        result = chunker.chunk(doc)
        for c in result.chunks:
            assert len(c.text) >= 20

    def test_constitution_article_chunking(self):
        text = """PART III

Article 14. Equality before law.
The State shall not deny to any person equality before the law.

Article 15. Prohibition of discrimination.
The State shall not discriminate against any citizen."""
        doc = LegalDocument(id="doc/001", doc_type=LegalDocumentType.CONSTITUTION_ARTICLE, title="FR", text=text)
        chunker = HierarchicalChunker()
        result = chunker.chunk(doc)
        articles = [c for c in result.chunks if c.chunk_type == "article"]
        assert len(articles) >= 1
