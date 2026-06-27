import json
import uuid
from pathlib import Path

from src.chunking import HierarchicalChunker
from src.config.constants import CHUNKS_FILE
from src.config.settings import settings
from src.core.logging import logger
from src.embeddings import EmbeddingPipeline
from src.ingestion import load_document
from src.retrieval import FAISSIndexManager
from src.schema import CorpusBuilder


class IngestionOrchestrator:
    def __init__(self, upload_dir: str = ""):
        self.upload_dir = Path(upload_dir or "data/uploads")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.corpus_builder = CorpusBuilder()
        self.chunker = HierarchicalChunker()
        self.embedding_pipeline = EmbeddingPipeline()
        self._faiss: FAISSIndexManager | None = None

    def set_faiss(self, faiss: FAISSIndexManager) -> None:
        self._faiss = faiss

    def ingest(self, file_path: str) -> dict:
        filepath = Path(file_path)
        logger.info("Ingesting document", path=str(filepath))

        data = load_document(str(filepath))
        doc = self.corpus_builder.build_document(data)
        chunk_result = self.chunker.chunk(doc)
        vector_store = self.embedding_pipeline.embed_chunks(chunk_result)

        if self._faiss is not None:
            vec_dict = {
                vid: vector_store.get_vector(vid)
                for vid in vector_store.vector_ids
            }
            chunk_map = {
                vid: vector_store.get_chunk_id(vid)
                for vid in vector_store.vector_ids
            }
            self._faiss.add_vectors(vec_dict, chunk_map)
            self._faiss.save()

        chunks_path = Path(settings.corpus_path) / CHUNKS_FILE
        if chunks_path.exists():
            with open(chunks_path) as f:
                existing = json.load(f)
        else:
            existing = []
        existing.extend(c.model_dump() for c in chunk_result.chunks)
        with open(chunks_path, "w") as f:
            json.dump(existing, f)

        result = {
            "document_id": doc.id,
            "title": doc.title,
            "chunks": len(chunk_result.chunks),
            "vectors": len(vector_store),
            "status": "ingested",
        }
        logger.info("Document ingested", **result)
        return result

    def save_upload(self, file_bytes: bytes, filename: str) -> str:
        ext = Path(filename).suffix.lower()
        if ext not in (".pdf", ".docx", ".txt", ".json"):
            raise ValueError(f"Unsupported file type: {ext}")
        unique_name = f"{uuid.uuid4().hex[:8]}_{filename}"
        dest = self.upload_dir / unique_name
        dest.write_bytes(file_bytes)
        return str(dest)
