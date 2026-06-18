import json
import uuid
from pathlib import Path

from src.ingestion.base import DocumentData
from src.ingestion.metadata_extractor import MetadataExtractor
from src.ingestion.text_cleaner import TextCleaner
from src.schema.legal_document import LegalCorpus, LegalDocument, LegalDocumentType
from src.schema.validators import SchemaValidator


class CorpusBuilder:
    def __init__(self, corpus_path: str = "data/corpus"):
        self.corpus_path = Path(corpus_path)
        self.corpus_path.mkdir(parents=True, exist_ok=True)
        self._cleaner = TextCleaner()
        self._metadata_extractor = MetadataExtractor()
        self._validator = SchemaValidator()

    def build_document(self, data: DocumentData) -> LegalDocument:
        cleaned_text = self._cleaner.clean(data.text)
        extracted_meta = self._metadata_extractor.extract(data.file_path, cleaned_text)

        enriched_meta = {**data.metadata, **extracted_meta}
        doc_type = self._infer_doc_type(enriched_meta)
        id_str = f"{doc_type.value}/{uuid.uuid4().hex[:12]}"
        text_preview = cleaned_text[:500]

        return LegalDocument(
            id=id_str,
            doc_type=doc_type,
            title=enriched_meta.get("source_filename", "Untitled"),
            text=cleaned_text,
            act_name=enriched_meta.get("act_name"),
            chapter=enriched_meta.get("chapter"),
            part=enriched_meta.get("part"),
            section_number=self._validator.extract_section_number(text_preview)
            or enriched_meta.get("section_number"),
            article_number=self._validator.extract_article_number(text_preview)
            or enriched_meta.get("article_number"),
            rule_number=self._validator.extract_rule_number(text_preview)
            or enriched_meta.get("rule_number"),
            date=self._validator.coerce_date(enriched_meta.get("year")),
            metadata=enriched_meta,
        )

    def build_corpus(self, documents: list[DocumentData]) -> LegalCorpus:
        legal_docs = [self.build_document(doc) for doc in documents]
        return LegalCorpus(documents=legal_docs)

    def save_corpus(self, corpus: LegalCorpus, filename: str = "legal_corpus.json") -> Path:
        filepath = self.corpus_path / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(corpus.model_dump_json(indent=2))
        return filepath

    def load_corpus(self, filename: str = "legal_corpus.json") -> LegalCorpus | None:
        filepath = self.corpus_path / filename
        if not filepath.exists():
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            return LegalCorpus.model_validate(json.load(f))

    @staticmethod
    def _infer_doc_type(metadata: dict) -> LegalDocumentType:
        hint = metadata.get("document_type_hint", "").lower()
        type_map = {
            "article": LegalDocumentType.CONSTITUTION_ARTICLE,
            "section": LegalDocumentType.SECTION,
            "rule": LegalDocumentType.RULE,
            "notification": LegalDocumentType.NOTIFICATION,
            "judgment": LegalDocumentType.JUDGMENT,
            "act": LegalDocumentType.ACT,
        }
        return type_map.get(hint, LegalDocumentType.CUSTOM)
