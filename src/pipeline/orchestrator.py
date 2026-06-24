import time
from typing import Optional

from src.api.request_models import AskResponse, RetrievedDocInfo
from src.confidence import ConfidenceEngine
from src.core.logging import logger
from src.generation import InferencePipeline, LLMClient, PromptBuilder
from src.guardrails import (
    CitationEnforcer,
    ConfidenceThreshold,
    Disclaimer,
    InjectionDetector,
    ScopeChecker,
)
from src.retrieval import BM25Engine, FAISSIndexManager, HybridRetriever
from src.retrieval.retrieval_config import RetrievalConfig
from src.telemetry import TelemetryLogger


class PipelineOrchestrator:
    def __init__(self):
        self.injection_detector = InjectionDetector()
        self.scope_checker = ScopeChecker()
        self.citation_enforcer = CitationEnforcer()
        self.confidence_threshold = ConfidenceThreshold()
        self.disclaimer = Disclaimer()
        self.confidence_engine = ConfidenceEngine()
        self.prompt_builder = PromptBuilder()
        self.llm = None
        self.retriever = None
        self.telemetry = TelemetryLogger()

    def initialize(self) -> None:
        retriever = HybridRetriever()
        retriever.load()
        self.retriever = retriever
        try:
            self.llm = LLMClient()
        except Exception as e:
            logger.warning("LLM not available", error=str(e))
        logger.info("Pipeline initialized")

    async def ask(self, query: str, top_k: Optional[int] = None) -> AskResponse:
        span_id = self.telemetry.start_query(query)
        start_time = time.monotonic()
        guardrail_results: dict[str, bool] = {}
        error: Optional[str] = None
        response_text = ""
        confidence_score = 0.0
        confidence_details: dict[str, float] = {}

        try:
            passed, reason = self.injection_detector.check(query)
            guardrail_results["injection"] = passed
            if not passed:
                response_text = self.prompt_builder.build_refusal_prompt(reason)
                return self._build_response(query, response_text, guardrail_results, confidence_score, confidence_details, start_time, span_id, top_k=top_k)

            passed, reason = self.scope_checker.check(query)
            guardrail_results["scope"] = passed
            if not passed:
                response_text = self.prompt_builder.build_refusal_prompt(reason)
                return self._build_response(query, response_text, guardrail_results, confidence_score, confidence_details, start_time, span_id, top_k=top_k)

            hybrid_result = self.retriever.retrieve(query, top_k=top_k)
            self.telemetry.log_retrieval(
                span_id,
                [r.chunk_id for r in hybrid_result.results],
                {r.chunk_id: r.vector_score for r in hybrid_result.results if r.vector_score is not None},
                {r.chunk_id: r.bm25_score for r in hybrid_result.results if r.bm25_score is not None},
                hybrid_result.vector_latency_ms + hybrid_result.bm25_latency_ms,
            )

            guardrail_results["retrieval"] = len(hybrid_result.results) > 0
            if not hybrid_result.results:
                response_text = self.prompt_builder.build_refusal_prompt("No relevant legal documents found.")
                return self._build_response(query, response_text, guardrail_results, confidence_score, confidence_details, start_time, span_id, top_k=top_k)

            if self.llm:
                gen_result = self.llm.generate(
                    self.prompt_builder.build_grounded_prompt(query, hybrid_result.results)
                )
                response_text = gen_result.text
                self.telemetry.log_generation(span_id, gen_result.model, gen_result.usage, gen_result.latency_ms)
            else:
                first = hybrid_result.results[0]
                response_text = f"Based on {first.document_title} ({first.heading or ''}):\n\n{first.text[:500]}"
                self.telemetry.log_generation(span_id, "fallback", {"total_tokens": 0}, 0.0)

            retrieved_texts = [r.text for r in hybrid_result.results]
            citation_passed, _ = self.citation_enforcer.check(response_text, retrieved_texts)
            guardrail_results["citation"] = citation_passed

            cr = self.confidence_engine.calculate(
                hybrid_result, citation_passed, guardrail_results, response_text
            )
            confidence_score = cr.overall_score
            confidence_details = {
                "retrieval": cr.retrieval_score,
                "citation": cr.citation_score,
                "guardrail": cr.guardrail_score,
            }

            guardrail_results["confidence"] = cr.threshold_met
            if not cr.threshold_met:
                response_text = self.prompt_builder.build_refusal_prompt(
                    f"Insufficient confidence ({cr.overall_score:.2f}) to provide a reliable answer."
                )
                confidence_score = cr.overall_score

            response_text = self.disclaimer.append(response_text)

        except Exception as e:
            error = str(e)
            logger.error("Pipeline error", error=error)
            response_text = self.prompt_builder.build_refusal_prompt(f"An error occurred: {error}")
        finally:
            self.telemetry.log_guardrails(span_id, guardrail_results)
            self.telemetry.log_confidence(span_id, confidence_score, confidence_details)
            self.telemetry.complete(span_id, response=response_text, error=error)

        return self._build_response(query, response_text, guardrail_results, confidence_score, confidence_details, start_time, span_id, top_k=top_k)

    def _build_response(
        self, query: str, response_text: str, guardrail_results: dict[str, bool],
        confidence_score: float, confidence_details: dict[str, float],
        start_time: float, span_id: str, top_k: int | None = None
    ) -> AskResponse:
        latency = (time.monotonic() - start_time) * 1000
        retrieved_docs: list[RetrievedDocInfo] = []
        if self.retriever and guardrail_results.get("retrieval", False):
            hybrid_result = self.retriever.retrieve(query, top_k=top_k)
            for r in hybrid_result.results:
                retrieved_docs.append(
                    RetrievedDocInfo(
                        chunk_id=r.chunk_id,
                        heading=r.heading,
                        document_title=r.document_title,
                        act_name=r.act_name,
                        vector_score=r.vector_score,
                        bm25_score=r.bm25_score,
                        combined_score=r.combined_score,
                    )
                )

        return AskResponse(
            answer=response_text,
            confidence=round(confidence_score, 3),
            confidence_details=confidence_details,
            retrieved_documents=retrieved_docs,
            guardrail_results=guardrail_results,
            disclaimer=self.disclaimer.text,
            latency_ms=round(latency, 2),
        )
