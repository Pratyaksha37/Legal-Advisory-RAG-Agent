from pathlib import Path

from docx import Document

from src.ingestion.base import DocumentData, DocumentLoader


class DOCXLoader(DocumentLoader):
    def load(self, path: str) -> DocumentData:
        doc = Document(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

        return DocumentData(
            text="\n\n".join(paragraphs),
            metadata={"paragraph_count": len(paragraphs)},
            file_path=str(Path(path).resolve()),
            file_type="docx",
        )

    @classmethod
    def supported_extensions(cls) -> set[str]:
        return {".docx"}
