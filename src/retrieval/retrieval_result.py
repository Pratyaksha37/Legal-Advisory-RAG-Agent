from typing import Any, Optional

from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    chunk_id: str
    text: str
    heading: Optional[str] = None
    chunk_type: str = ""
    hierarchy_level: int = 0
    vector_score: Optional[float] = None
    vector_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    combined_score: float = 0.0
    reranker_score: Optional[float] = None
    document_title: str = ""
    act_name: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HybridResult(BaseModel):
    results: list[RetrievalResult]
    query: str
    total_results: int
    vector_latency_ms: float = 0.0
    bm25_latency_ms: float = 0.0
    reranker_latency_ms: float = 0.0
