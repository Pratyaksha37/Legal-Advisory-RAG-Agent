from pathlib import Path

import fitz

from src.ingestion.base import DocumentData, DocumentLoader


class PDFLoader(DocumentLoader):
    def load(self, path: str) -> DocumentData:
        doc = fitz.open(path)
        pages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            pages.append(text)
        doc.close()

        return DocumentData(
            text="\n\n".join(pages),
            metadata={"page_count": len(pages)},
            file_path=str(Path(path).resolve()),
            file_type="pdf",
        )

    @classmethod
    def supported_extensions(cls) -> set[str]:
        return {".pdf"}
