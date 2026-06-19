import json
import uuid
from pathlib import Path

import numpy as np

from src.config.constants import FAISS_INDEX_FILE, FAISS_MAPPING_FILE
from src.config.settings import settings
from src.core.exceptions import RetrievalError
from src.core.logging import logger

try:
    import faiss
except ImportError:
    faiss = None


class FAISSIndexManager:
    def __init__(self, index_path: str = ""):
        self.index_path = Path(index_path or settings.faiss_index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.index: faiss.Index | None = None
        self.id_to_chunk: dict[int, str] = {}
        self.chunk_to_id: dict[str, int] = {}

    def build(self, vectors: dict[str, list[float]], chunk_map: dict[str, str]) -> None:
        if faiss is None:
            raise RetrievalError("faiss package is not installed")

        if not vectors:
            raise RetrievalError("No vectors provided to build index")

        dim = len(next(iter(vectors.values())))
        n = len(vectors)

        ids = np.arange(n, dtype=np.int64)
        matrix = np.zeros((n, dim), dtype=np.float32)

        for i, (vector_id, vec) in enumerate(vectors.items()):
            matrix[i] = np.array(vec, dtype=np.float32)
            chunk_id = chunk_map.get(vector_id, "")
            self.id_to_chunk[i] = chunk_id
            self.chunk_to_id[chunk_id] = i

        base_index = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIDMap(base_index)
        self.index.add_with_ids(matrix, ids)

        logger.info(
            "Built FAISS index",
            dimension=dim,
            vectors=n,
            path=str(self.index_path),
        )

    def save(self) -> None:
        if self.index is None:
            raise RetrievalError("No index to save")

        index_file = self.index_path / FAISS_INDEX_FILE
        faiss.write_index(self.index, str(index_file))

        mapping_file = self.index_path / FAISS_MAPPING_FILE
        mapping = {
            "id_to_chunk": {str(k): v for k, v in self.id_to_chunk.items()},
            "chunk_to_id": self.chunk_to_id,
        }
        with open(mapping_file, "w", encoding="utf-8") as f:
            json.dump(mapping, f)

        logger.info("Saved FAISS index", path=str(index_file))

    def load(self) -> None:
        if faiss is None:
            raise RetrievalError("faiss package is not installed")

        index_file = self.index_path / FAISS_INDEX_FILE
        mapping_file = self.index_path / FAISS_MAPPING_FILE

        if not index_file.exists():
            raise RetrievalError(f"FAISS index not found: {index_file}")

        self.index = faiss.read_index(str(index_file))

        if mapping_file.exists():
            with open(mapping_file, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            self.id_to_chunk = {int(k): v for k, v in mapping["id_to_chunk"].items()}
            self.chunk_to_id = mapping["chunk_to_id"]

        logger.info("Loaded FAISS index", path=str(index_file), vectors=self.index.ntotal)

    def search(
        self, query_vector: list[float], top_k: int = 10
    ) -> list[tuple[str, float]]:
        if self.index is None:
            raise RetrievalError("Index not loaded. Call load() or build() first.")

        query = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, top_k)

        results: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk_id = self.id_to_chunk.get(int(idx), "")
            if chunk_id:
                results.append((chunk_id, float(score)))

        return results

    def add_vectors(
        self, vectors: dict[str, list[float]], chunk_map: dict[str, str]
    ) -> None:
        if self.index is None:
            self.build(vectors, chunk_map)
            return

        if not vectors:
            return

        dim = len(next(iter(vectors.values())))
        n = len(vectors)
        start_id = max(self.id_to_chunk.keys(), default=-1) + 1

        ids = np.arange(start_id, start_id + n, dtype=np.int64)
        matrix = np.zeros((n, dim), dtype=np.float32)

        for i, (vector_id, vec) in enumerate(vectors.items()):
            matrix[i] = np.array(vec, dtype=np.float32)
            faiss_id = start_id + i
            chunk_id = chunk_map.get(vector_id, "")
            self.id_to_chunk[faiss_id] = chunk_id
            self.chunk_to_id[chunk_id] = faiss_id

        self.index.add_with_ids(matrix, ids)

        logger.info("Added vectors to FAISS index", count=n)

    @property
    def is_loaded(self) -> bool:
        return self.index is not None
