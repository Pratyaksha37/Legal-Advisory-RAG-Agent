from pathlib import Path

from src.ingestion.base import DocumentLoader
from src.ingestion.docx_loader import DOCXLoader
from src.ingestion.json_loader import JSONLoader
from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.txt_loader import TXTLoader


class LoaderFactory:
    _loaders: list[DocumentLoader] = [
        PDFLoader(),
        DOCXLoader(),
        TXTLoader(),
        JSONLoader(),
    ]

    def get_loader(self, path: str) -> DocumentLoader:
        ext = Path(path).suffix.lower()
        for loader in self._loaders:
            if ext in loader.supported_extensions():
                return loader
        raise ValueError(f"Unsupported file extension: {ext}")

    def register_loader(self, loader: DocumentLoader) -> None:
        self._loaders.append(loader)


def load_document(path: str) -> "DocumentData":
    from src.ingestion.base import DocumentData

    factory = LoaderFactory()
    loader = factory.get_loader(path)
    return loader.load(path)
