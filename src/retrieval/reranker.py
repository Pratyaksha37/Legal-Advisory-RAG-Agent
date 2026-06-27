from abc import ABC, abstractmethod
from typing import Optional

from src.core.logging import logger
from src.retrieval.retrieval_result import RetrievalResult


class Reranker(ABC):
    @abstractmethod
    def rerank(
        self, query: str, documents: list[RetrievalResult], top_k: Optional[int] = None
    ) -> list[RetrievalResult]:
        pass


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            logger.info("Loaded cross-encoder model", model=self.model_name)

    def rerank(
        self, query: str, documents: list[RetrievalResult], top_k: Optional[int] = None
    ) -> list[RetrievalResult]:
        if not documents:
            return documents

        self._load_model()
        pairs = [(query, doc.text) for doc in documents]
        scores = self._model.predict(pairs)

        for doc, score in zip(documents, scores):
            doc.reranker_score = float(score)

        documents.sort(key=lambda d: d.reranker_score or 0.0, reverse=True)
        k = top_k or len(documents)
        logger.info(
            "Reranked documents",
            count=len(documents),
            top_score=documents[0].reranker_score if documents else None,
        )
        return documents[:k]


class NoopReranker(Reranker):
    def rerank(
        self, query: str, documents: list[RetrievalResult], top_k: Optional[int] = None
    ) -> list[RetrievalResult]:
        k = top_k or len(documents)
        return documents[:k]
