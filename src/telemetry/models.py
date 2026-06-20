from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class TelemetryEvent(BaseModel):
    timestamp: str = ""
    query: str = ""
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    vector_scores: dict[str, float] = Field(default_factory=dict)
    bm25_scores: dict[str, float] = Field(default_factory=dict)
    retrieval_latency_ms: float = 0.0
    llm_model: str = ""
    llm_tokens: dict[str, int] = Field(default_factory=dict)
    llm_latency_ms: float = 0.0
    guardrail_results: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    confidence_details: dict[str, float] = Field(default_factory=dict)
    response: str = ""
    error: Optional[str] = None
    span_id: str = ""
