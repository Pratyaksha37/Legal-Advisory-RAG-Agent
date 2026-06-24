from datetime import date

import pytest
from pydantic import ValidationError

from src.ingestion.base import DocumentData
from src.schema import CorpusBuilder, LegalCorpus, LegalDocument, LegalDocumentType, SchemaValidator


class TestLegalDocument:
    def test_minimal_document(self):
        doc = LegalDocument(
            id="test/001",
            doc_type=LegalDocumentType.SECTION,
            title="Test",
            text="Content",
        )
        assert doc.id == "test/001"
        assert doc.act_name is None

    def test_all_optional_fields_none_by_default(self):
        doc = LegalDocument(
            id="test/001",
            doc_type=LegalDocumentType.SECTION,
            title="Test",
            text="Content",
        )
        assert doc.act_name is None
        assert doc.parent_id is None
        assert doc.chapter is None
        assert doc.section_number is None
        assert doc.date is None
        assert doc.metadata == {}

    def test_full_document(self):
        doc = LegalDocument(
            id="section/302",
            doc_type=LegalDocumentType.SECTION,
            title="Indian Penal Code",
            text="Section 302: Punishment for murder.",
            act_name="Indian Penal Code",
            section_number="302",
            date=date(1860, 10, 6),
            metadata={"source": "IPC"},
        )
        assert doc.act_name == "Indian Penal Code"
        assert doc.date == date(1860, 10, 6)

    def test_extra_fields_allowed(self):
        doc = LegalDocument(
            id="test/001",
            doc_type=LegalDocumentType.CUSTOM,
            title="Test",
            text="Content",
            future_field="future_value",
        )
        assert doc.model_extra.get("future_field") == "future_value"

    def test_string_stripping(self):
        doc = LegalDocument(
            id="test/001",
            doc_type=LegalDocumentType.SECTION,
            title="  Test Title  ",
            text="  Content  ",
        )
        assert doc.title == "Test Title"
        assert doc.text == "Content"

    def test_missing_required_fields_raises(self):
        with pytest.raises(ValidationError):
            LegalDocument()

    def test_invalid_doc_type_raises(self):
        with pytest.raises(ValidationError):
            LegalDocument(id="test/001", doc_type="invalid", title="T", text="C")


class TestLegalCorpus:
    def test_empty_corpus(self):
        corpus = LegalCorpus(documents=[])
        assert len(corpus.documents) == 0

    def test_corpus_with_documents(self):
        docs = [
            LegalDocument(id="1", doc_type=LegalDocumentType.SECTION, title="T1", text="C1"),
            LegalDocument(id="2", doc_type=LegalDocumentType.SECTION, title="T2", text="C2"),
        ]
        corpus = LegalCorpus(documents=docs)
        assert len(corpus.documents) == 2

    def test_serialization_roundtrip(self):
        doc = LegalDocument(
            id="section/302",
            doc_type=LegalDocumentType.SECTION,
            title="IPC",
            text="Section 302",
            act_name="IPC",
            date=date(1860, 10, 6),
        )
        corpus = LegalCorpus(documents=[doc])
        json_str = corpus.model_dump_json()
        reloaded = LegalCorpus.model_validate_json(json_str)
        assert reloaded.documents[0].title == "IPC"
        assert reloaded.documents[0].date == date(1860, 10, 6)


class TestSchemaValidator:
    def test_normalize_doc_type(self):
        assert SchemaValidator.normalize_doc_type("section") == LegalDocumentType.SECTION
        assert SchemaValidator.normalize_doc_type("Section") == LegalDocumentType.SECTION
        assert SchemaValidator.normalize_doc_type("constitution article") == LegalDocumentType.CONSTITUTION_ARTICLE
        assert SchemaValidator.normalize_doc_type("unknown") == LegalDocumentType.CUSTOM

    def test_extract_section_number(self):
        assert SchemaValidator.extract_section_number("Section 302 IPC") == "302"
        assert SchemaValidator.extract_section_number("Sec. 14") == "14"
        assert SchemaValidator.extract_section_number("No section here") is None

    def test_extract_article_number(self):
        assert SchemaValidator.extract_article_number("Article 14 of Constitution") == "14"
        assert SchemaValidator.extract_article_number("Art. 21") == "21"
        assert SchemaValidator.extract_article_number("No article") is None

    def test_extract_rule_number(self):
        assert SchemaValidator.extract_rule_number("Rule 5 of the Rules") == "5"
        assert SchemaValidator.extract_rule_number("No rule") is None

    def test_coerce_date(self):
        assert SchemaValidator.coerce_date("1950-01-26") == date(1950, 1, 26)
        assert SchemaValidator.coerce_date("26/01/1950") == date(1950, 1, 26)
        assert SchemaValidator.coerce_date(None) is None
        assert SchemaValidator.coerce_date("invalid") is None

    def test_coerce_date_from_int(self):
        assert SchemaValidator.coerce_date("1860") is None  # years not parseable as dates


class TestCorpusBuilder:
    def test_build_document_from_data(self, sample_document_data):
        builder = CorpusBuilder()
        doc = builder.build_document(sample_document_data)
        assert doc.title == "test"
        assert doc.doc_type == LegalDocumentType.CONSTITUTION_ARTICLE
        assert len(doc.text) > 100

    def test_build_document_with_missing_metadata(self):
        data = DocumentData(text="Some text", file_path="/tmp/test.pdf", file_type="pdf")
        builder = CorpusBuilder()
        doc = builder.build_document(data)
        assert doc.title == "test"
        assert doc.doc_type == LegalDocumentType.CUSTOM
        assert doc.act_name is None

    def test_build_corpus(self, sample_document_data):
        builder = CorpusBuilder()
        corpus = builder.build_corpus([sample_document_data])
        assert len(corpus.documents) == 1

    def test_save_and_load_corpus(self, sample_document_data, temp_dir):
        builder = CorpusBuilder(corpus_path=str(temp_dir))
        corpus = builder.build_corpus([sample_document_data])
        saved = builder.save_corpus(corpus)
        assert saved.exists()
        loaded = builder.load_corpus()
        assert loaded is not None
        assert len(loaded.documents) == 1

    def test_load_nonexistent_corpus(self, temp_dir):
        builder = CorpusBuilder(corpus_path=str(temp_dir))
        loaded = builder.load_corpus()
        assert loaded is None
