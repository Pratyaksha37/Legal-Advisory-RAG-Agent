import uuid
from datetime import datetime
from typing import Any, Optional

from src.telemetry.models import TelemetryEvent
from src.telemetry.store import TelemetryStore


class TelemetryLogger:
    def __init__(self, store: TelemetryStore | None = None):
        self.store = store or TelemetryStore()
        self._events: dict[str, TelemetryEvent] = {}

    def start_query(self, query: str) -> str:
        span_id = uuid.uuid4().hex[:16]
        event = TelemetryEvent(
            span_id=span_id,
            query=query,
            timestamp=datetime.utcnow().isoformat(),
        )
        self._events[span_id] = event
        return span_id

    def log_retrieval(
        self,
        span_id: str,
        doc_ids: list[str],
        vector_scores: dict[str, float],
        bm25_scores: dict[str, float],
        latency_ms: float,
    ) -> None:
        event = self._events.get(span_id)
        if event:
            event.retrieved_doc_ids = doc_ids
            event.vector_scores = vector_scores
            event.bm25_scores = bm25_scores
            event.retrieval_latency_ms = latency_ms

    def log_generation(
        self,
        span_id: str,
        model: str,
        tokens: dict[str, int],
        latency_ms: float,
    ) -> None:
        event = self._events.get(span_id)
        if event:
            event.llm_model = model
            event.llm_tokens = tokens
            event.llm_latency_ms = latency_ms

    def log_guardrails(
        self, span_id: str, results: dict[str, Any]
    ) -> None:
        event = self._events.get(span_id)
        if event:
            event.guardrail_results = results

    def log_confidence(
        self,
        span_id: str,
        score: float,
        details: dict[str, float],
    ) -> None:
        event = self._events.get(span_id)
        if event:
            event.confidence = score
            event.confidence_details = details

    def complete(
        self,
        span_id: str,
        response: str = "",
        error: Optional[str] = None,
    ) -> None:
        event = self._events.get(span_id)
        if event:
            event.response = response
            event.error = error
            self.store.append(event)
            del self._events[span_id]
