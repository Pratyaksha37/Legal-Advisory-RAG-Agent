import json
import time
from pathlib import Path
from typing import Optional

from src.config.constants import CHUNKS_FILE
from src.config.settings import settings
from src.core.exceptions import RetrievalError
from src.core.logging import logger
from src.embeddings import EmbeddingModel
from src.retrieval.bm25_engine import BM25Engine
from src.retrieval.faiss_index import FAISSIndexManager
from src.retrieval.reranker import Reranker
from src.retrieval.retrieval_config import RetrievalConfig
from src.retrieval.retrieval_result import HybridResult, RetrievalResult


class HybridRetriever:
    def __init__(
        self,
        faiss_manager: FAISSIndexManager | None = None,
        bm25_engine: BM25Engine | None = None,
        embedding_model: EmbeddingModel | None = None,
        config: RetrievalConfig | None = None,
        reranker: Optional["Reranker"] = None,
    ):
        self.faiss = faiss_manager or FAISSIndexManager()
        self.bm25 = bm25_engine or BM25Engine()
        self.embeddings = embedding_model or EmbeddingModel.get_instance()
        self.config = config or RetrievalConfig(
            vector_weight=settings.vector_weight,
            bm25_weight=settings.bm25_weight,
            top_k=settings.retrieval_top_k,
            fusion_method=settings.fusion_method,
        )
        self.chunks_by_id: dict[str, dict] = {}
        self.reranker = reranker

    def load(self) -> None:
        self.faiss.load()
        self.bm25.load()
        self._load_chunks()

    def _load_chunks(self) -> None:
        chunks_path = Path(settings.corpus_path) / CHUNKS_FILE
        if chunks_path.exists():
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            self.chunks_by_id = {c["chunk_id"]: c for c in chunks}

    def retrieve(self, query: str, top_k: int | None = None) -> HybridResult:
        if not self.faiss.is_loaded:
            raise RetrievalError("FAISS index not loaded. Call load() first.")

        k = top_k or self.config.top_k
        results: dict[str, RetrievalResult] = {}

        vector_start = time.monotonic()
        query_vec = self.embeddings.encode_single(query)
        vector_results = self.faiss.search(query_vec, top_k=k)
        vector_elapsed = (time.monotonic() - vector_start) * 1000

        for rank, (chunk_id, score) in enumerate(vector_results, start=1):
            if score < self.config.min_vector_score:
                continue
            chunk = self.chunks_by_id.get(chunk_id, {})
            results[chunk_id] = RetrievalResult(
                chunk_id=chunk_id,
                text=chunk.get("text", ""),
                heading=chunk.get("heading"),
                chunk_type=chunk.get("chunk_type", ""),
                hierarchy_level=chunk.get("hierarchy_level", 0),
                vector_score=score,
                vector_rank=rank,
                document_title=chunk.get("metadata", {}).get("source_doc_title", ""),
                act_name=chunk.get("metadata", {}).get("act_name"),
                metadata=chunk.get("metadata", {}),
            )

        bm25_start = time.monotonic()
        bm25_results = self.bm25.search(query, top_k=k)
        bm25_elapsed = (time.monotonic() - bm25_start) * 1000

        for rank, (chunk_id, score) in enumerate(bm25_results, start=1):
            if score < self.config.min_bm25_score:
                continue
            if chunk_id in results:
                results[chunk_id].bm25_score = score
                results[chunk_id].bm25_rank = rank
            else:
                chunk = self.chunks_by_id.get(chunk_id, {})
                results[chunk_id] = RetrievalResult(
                    chunk_id=chunk_id,
                    text=chunk.get("text", ""),
                    heading=chunk.get("heading"),
                    chunk_type=chunk.get("chunk_type", ""),
                    hierarchy_level=chunk.get("hierarchy_level", 0),
                    bm25_score=score,
                    bm25_rank=rank,
                    document_title=chunk.get("metadata", {}).get("source_doc_title", ""),
                    act_name=chunk.get("metadata", {}).get("act_name"),
                    metadata=chunk.get("metadata", {}),
                )

        self._compute_combined_scores(results)
        sorted_results = sorted(
            results.values(), key=lambda r: r.combined_score, reverse=True
        )[:k]

        reranker_elapsed = 0.0
        if self.reranker is not None and sorted_results:
            reranker_start = time.monotonic()
            sorted_results = self.reranker.rerank(query, sorted_results, top_k=k)
            reranker_elapsed = (time.monotonic() - reranker_start) * 1000

        return HybridResult(
            results=sorted_results,
            query=query,
            total_results=len(sorted_results),
            vector_latency_ms=round(vector_elapsed, 2),
            bm25_latency_ms=round(bm25_elapsed, 2),
            reranker_latency_ms=round(reranker_elapsed, 2),
        )

    def _compute_combined_scores(
        self, results: dict[str, RetrievalResult]
    ) -> None:
        max_vec = max(
            (r.vector_score for r in results.values() if r.vector_score is not None),
            default=1.0,
        )
        max_bm25 = max(
            (r.bm25_score for r in results.values() if r.bm25_score is not None),
            default=1.0,
        )

        for result in results.values():
            vec_norm = (result.vector_score or 0.0) / max_vec if max_vec > 0 else 0.0
            bm25_norm = (result.bm25_score or 0.0) / max_bm25 if max_bm25 > 0 else 0.0

            if self.config.fusion_method == "rrf":
                vec_rank = getattr(result, "vector_rank", 0) or len(results)
                bm25_rank = getattr(result, "bm25_rank", 0) or len(results)
                vec_score_rrf = 1.0 / (self.config.rrf_k + vec_rank) if result.vector_score is not None else 0.0
                bm25_score_rrf = 1.0 / (self.config.rrf_k + bm25_rank) if result.bm25_score is not None else 0.0
                result.combined_score = (
                    self.config.vector_weight * vec_score_rrf
                    + self.config.bm25_weight * bm25_score_rrf
                )
            else:
                result.combined_score = (
                    self.config.vector_weight * vec_norm
                    + self.config.bm25_weight * bm25_norm
                )
