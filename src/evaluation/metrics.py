from typing import Any, Optional

from pydantic import BaseModel, Field


class EvaluationSample(BaseModel):
    query: str
    ground_truth: str
    relevant_doc_ids: list[str] = Field(default_factory=list)


class MetricResult(BaseModel):
    metric_name: str
    score: float
    details: Optional[dict[str, Any]] = None


class EvaluationReport(BaseModel):
    samples: int = 0
    metrics: list[MetricResult] = Field(default_factory=list)
    per_sample_scores: list[dict[str, Any]] = Field(default_factory=list)


class RetrievalPrecision:
    @staticmethod
    def compute(retrieved_ids: list[str], relevant_ids: list[str], k: int = 10) -> float:
        if not relevant_ids:
            return 0.0
        retrieved_at_k = retrieved_ids[:k]
        relevant_retrieved = sum(1 for rid in retrieved_at_k if rid in relevant_ids)
        return relevant_retrieved / min(k, len(retrieved_at_k)) if retrieved_at_k else 0.0


class CitationCorrectness:
    @staticmethod
    def compute(response_text: str, expected_citations: list[str]) -> float:
        if not expected_citations:
            return 1.0
        found = sum(1 for c in expected_citations if c.lower() in response_text.lower())
        return found / len(expected_citations)
