import json
import tempfile
from pathlib import Path

import pytest

from src.chunking.chunk_models import DocumentChunk
from src.config.constants import LegalDocumentType
from src.ingestion.base import DocumentData
from src.retrieval.retrieval_config import RetrievalConfig
from src.retrieval.retrieval_result import RetrievalResult
from src.schema.legal_document import LegalDocument


@pytest.fixture
def sample_pdf_text() -> str:
    return "data/raw/Fundamental Rights.pdf"


@pytest.fixture
def sample_legal_text() -> str:
    return """CHAPTER I - PRELIMINARY

Section 1. Short title and commencement.

Section 2. Definitions.
(1) In this Act, unless the context otherwise requires.
(a) "person" includes any individual.
(b) "document" includes any matter written or recorded.

Section 3. General Explanations.
Throughout this Act, every definition shall be understood.

CHAPTER II - GENERAL EXCEPTIONS

Section 76. Act done by a person bound by law.
Nothing is an offence which is done by a person who is bound by law to do it.

Section 77. Act of Judge when acting judicially.
Nothing is an offence which is done by a Judge when acting judicially.
"""


@pytest.fixture
def sample_document_data() -> DocumentData:
    return DocumentData(
        text="""CHAPTER 3 FUNDAMENTAL RIGHTS

Article 14. Equality before law.
The State shall not deny to any person equality before the law.

Article 15. Prohibition of discrimination.
The State shall not discriminate against any citizen.

Article 19. Protection of certain rights.
(1) All citizens shall have the right to freedom of speech.""",
        metadata={
            "source_filename": "Fundamental Rights",
            "document_type_hint": "article",
            "page_count": 10,
        },
        file_path="/tmp/test.pdf",
        file_type="pdf",
    )


@pytest.fixture
def sample_legal_document() -> LegalDocument:
    return LegalDocument(
        id="section/abc123",
        doc_type=LegalDocumentType.SECTION,
        title="Test Act",
        text="""Section 1. Short title.
This Act may be called the Test Act.

Section 2. Definitions.
In this Act, unless the context otherwise requires.
(a) "person" means any individual.
(b) "document" includes any written matter.

Section 3. Punishment.
Whoever violates this Act shall be punished.""",
        act_name="Test Act",
        section_number="1",
    )


@pytest.fixture
def sample_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "chk/001",
            "document_id": "doc/001",
            "parent_id": None,
            "text": "Section 1. This is the first section of the IPC.",
            "hierarchy_level": 2,
            "heading": "Section 1",
            "chunk_type": "section",
            "metadata": {"source_doc_title": "IPC", "act_name": "Indian Penal Code"},
        },
        {
            "chunk_id": "chk/002",
            "document_id": "doc/001",
            "parent_id": "chk/001",
            "text": "Section 2. This is the second section.",
            "hierarchy_level": 2,
            "heading": "Section 2",
            "chunk_type": "section",
            "metadata": {"source_doc_title": "IPC", "act_name": "Indian Penal Code"},
        },
        {
            "chunk_id": "chk/003",
            "document_id": "doc/002",
            "parent_id": None,
            "text": "Article 14. Equality before law.",
            "hierarchy_level": 2,
            "heading": "Article 14",
            "chunk_type": "article",
            "metadata": {"source_doc_title": "Constitution", "act_name": "Constitution"},
        },
    ]


@pytest.fixture
def sample_retrieval_results() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id="chk/001",
            text="Section 302: Punishment for murder.",
            heading="Section 302",
            chunk_type="section",
            hierarchy_level=2,
            vector_score=0.75,
            bm25_score=15.0,
            combined_score=0.85,
            document_title="Indian Penal Code",
            act_name="Indian Penal Code",
        ),
        RetrievalResult(
            chunk_id="chk/002",
            text="Section 300: Murder defined.",
            heading="Section 300",
            chunk_type="section",
            hierarchy_level=2,
            vector_score=0.65,
            bm25_score=12.0,
            combined_score=0.75,
            document_title="Indian Penal Code",
            act_name="Indian Penal Code",
        ),
    ]


@pytest.fixture
def temp_dir() -> Path:
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)
