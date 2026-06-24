import tempfile
from pathlib import Path

import pytest

from src.ingestion import (
    DOCXLoader,
    JSONLoader,
    PDFLoader,
    TXTLoader,
    DocumentData,
    LoaderFactory,
    MetadataExtractor,
    TextCleaner,
    load_document,
)
from src.ingestion.base import DocumentLoader


class TestDocumentLoader:
    def test_pdf_loader(self):
        loader = PDFLoader()
        assert ".pdf" in loader.supported_extensions()
        assert not loader.supports("test.txt")

    def test_txt_loader(self):
        loader = TXTLoader()
        assert ".txt" in loader.supported_extensions()

    def test_docx_loader(self):
        loader = DOCXLoader()
        assert ".docx" in loader.supported_extensions()

    def test_json_loader(self):
        loader = JSONLoader()
        assert ".json" in loader.supported_extensions()

    def test_txt_loader_loads_content(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello world")
            path = f.name
        result = load_document(path)
        assert result.text == "Hello world"
        assert result.file_type == "txt"
        Path(path).unlink()

    def test_json_loader_dict(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"text": "Article 14", "act": "Constitution"}')
            path = f.name
        result = load_document(path)
        assert "Article 14" in result.text
        assert result.metadata.get("act") == "Constitution"
        Path(path).unlink()

    def test_json_loader_array(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('[{"text": "Article 14"}, {"text": "Article 15"}]')
            path = f.name
        result = load_document(path)
        assert "Article 14" in result.text
        assert "Article 15" in result.text
        Path(path).unlink()

    def test_unsupported_extension_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            path = f.name
        with pytest.raises(ValueError, match="Unsupported"):
            load_document(path)
        Path(path).unlink()

    def test_loader_factory_custom_loader(self):
        class CustomLoader(DocumentLoader):
            def load(self, path: str) -> DocumentData:
                return DocumentData(text="custom", file_path=path, file_type="custom")
            @classmethod
            def supported_extensions(cls) -> set[str]:
                return {".custom"}
        factory = LoaderFactory()
        factory.register_loader(CustomLoader())
        with tempfile.NamedTemporaryFile(suffix=".custom", delete=False) as f:
            path = f.name
        result = factory.get_loader(path).load(path)
        assert result.text == "custom"
        Path(path).unlink()


class TestTextCleaner:
    def test_normalize_whitespace(self):
        cleaner = TextCleaner()
        result = cleaner.normalize_whitespace("  Hello   world\n\n\nNext")
        assert "  " not in result
        assert "\n\n\n" not in result

    def test_normalize_unicode(self):
        cleaner = TextCleaner()
        result = cleaner.normalize_unicode("\u201cquoted\u201d")
        assert result  # NFKC normalization applied

    def test_strip_headers_footers(self):
        cleaner = TextCleaner()
        text = "Page 1\nThis is content.\n- 5 -\nMore content."
        result = cleaner.strip_common_headers_footers(text)
        assert "Page 1" not in result
        assert "- 5 -" not in result
        assert "This is content" in result

    def test_legal_abbreviations(self):
        cleaner = TextCleaner()
        result = cleaner.normalize_legal_abbreviations("IPC and CrPC are important.")
        assert "Indian Penal Code" in result
        assert "Code of Criminal Procedure" in result

    def test_full_clean(self):
        cleaner = TextCleaner()
        result = cleaner.clean("  IPC  \n\n\nSection\n\n")
        assert "Indian Penal Code" in result
        assert "  " not in result


class TestMetadataExtractor:
    def test_from_filename_year(self):
        meta = MetadataExtractor().from_filename("IPC_1860_Section_302.txt")
        assert meta.get("year") == 1860

    def test_from_filename_act(self):
        meta = MetadataExtractor().from_filename("Constitution_Article_14.txt")
        assert meta.get("act_name") == "Constitution"

    def test_from_filename_doc_type(self):
        meta = MetadataExtractor().from_filename("Notification_2024.txt")
        assert meta.get("document_type_hint") == "notification"

    def test_from_filename_no_year(self):
        meta = MetadataExtractor().from_filename("Notes.txt")
        assert meta.get("year") is None

    def test_from_content_preamble(self):
        meta = MetadataExtractor().from_content("Preamble of the Constitution")
        assert meta.get("has_preamble") is True

    def test_from_content_date(self):
        meta = MetadataExtractor().from_content("Dated: 15/03/1950")
        assert meta.get("date_found") == "15/03/1950"

    def test_full_extract(self):
        ext = MetadataExtractor()
        meta = ext.extract("/tmp/Fundamental_Rights.pdf", "Preamble of the Constitution")
        assert "source_filename" in meta
