import json
import uuid
from pathlib import Path

from src.chunking.chunk_models import ChunkResult
from src.config.settings import settings
from src.core.logging import logger
from src.embeddings.embedding_model import EmbeddingModel
from src.embeddings.vector_store import VectorStore


class EmbeddingPipeline:
    def __init__(
        self,
        model: EmbeddingModel | None = None,
        embeddings_path: str = "",
    ):
        self.model = model or EmbeddingModel.get_instance()
        self.embeddings_path = Path(embeddings_path or settings.embeddings_path)
        self.embeddings_path.mkdir(parents=True, exist_ok=True)

    def embed_chunks(
        self, chunk_result: ChunkResult, batch_size: int = 32
    ) -> VectorStore:
        vector_store = VectorStore()
        texts: list[str] = []
        chunk_ids: list[str] = []

        for chunk in chunk_result.chunks:
            texts.append(chunk.text)
            chunk_ids.append(chunk.chunk_id)

        if not texts:
            logger.warning("No chunks to embed", document_id=chunk_result.document_id)
            return vector_store

        embeddings = self.model.encode(texts, batch_size=batch_size)

        for chunk_id, embedding in zip(chunk_ids, embeddings):
            vector_id = f"vec/{uuid.uuid4().hex[:12]}"
            vector_store.add(vector_id, chunk_id, embedding)

        logger.info(
            "Embedded chunks",
            document_id=chunk_result.document_id,
            count=len(texts),
        )

        return vector_store

    def save(self, vector_store: VectorStore, filename: str = "vectors.json") -> Path:
        filepath = self.embeddings_path / filename
        data = {
            "embeddings": vector_store.to_dict(),
            "dimension": len(next(iter(vector_store.vectors.values())))
            if vector_store.vectors
            else 0,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)
        logger.info("Saved embeddings", path=str(filepath), vectors=len(vector_store))
        return filepath

    def load(self, filename: str = "vectors.json") -> VectorStore:
        filepath = self.embeddings_path / filename
        if not filepath.exists():
            logger.warning("Embeddings file not found", path=str(filepath))
            return VectorStore()
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        vector_store = VectorStore()
        for item in data["embeddings"]:
            vector_store.add(item["vector_id"], item["chunk_id"], item["vector"])
        logger.info("Loaded embeddings", path=str(filepath), vectors=len(vector_store))
        return vector_store
