from typing import Any, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)


class RetrievedDocInfo(BaseModel):
    chunk_id: str
    heading: Optional[str] = None
    document_title: str = ""
    act_name: Optional[str] = None
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    combined_score: float = 0.0
    reranker_score: Optional[float] = None


class AskResponse(BaseModel):
    answer: str
    confidence: float = 0.0
    confidence_details: dict[str, float] = Field(default_factory=dict)
    retrieved_documents: list[RetrievedDocInfo] = Field(default_factory=list)
    guardrail_results: dict[str, Any] = Field(default_factory=dict)
    disclaimer: str = ""
    latency_ms: float = 0.0


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "0.1.0"
    indexes_loaded: bool = False
    model_loaded: bool = False
