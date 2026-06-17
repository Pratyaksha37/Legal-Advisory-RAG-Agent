from pathlib import Path

from src.ingestion.base import DocumentData, DocumentLoader


class TXTLoader(DocumentLoader):
    def load(self, path: str) -> DocumentData:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        return DocumentData(
            text=text,
            file_path=str(Path(path).resolve()),
            file_type="txt",
        )

    @classmethod
    def supported_extensions(cls) -> set[str]:
        return {".txt"}
