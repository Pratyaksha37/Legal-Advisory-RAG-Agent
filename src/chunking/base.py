from abc import ABC, abstractmethod

from src.chunking.chunk_models import ChunkResult
from src.schema.legal_document import LegalDocument


class Chunker(ABC):
    @abstractmethod
    def chunk(self, document: LegalDocument) -> ChunkResult:
        pass
