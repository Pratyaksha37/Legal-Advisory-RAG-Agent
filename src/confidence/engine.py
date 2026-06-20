from src.config.constants import CONFIDENCE_WEIGHTS
from src.retrieval.retrieval_result import HybridResult, RetrievalResult


class ConfidenceResult:
    def __init__(
        self,
        overall_score: float,
        retrieval_score: float,
        citation_score: float,
        guardrail_score: float,
        supporting_evidence_count: int,
        threshold_met: bool,
    ):
        self.overall_score = overall_score
        self.retrieval_score = retrieval_score
        self.citation_score = citation_score
        self.guardrail_score = guardrail_score
        self.supporting_evidence_count = supporting_evidence_count
        self.threshold_met = threshold_met


class ConfidenceEngine:
    def __init__(self, threshold: float | None = None):
        from src.config.settings import settings
        self.threshold = threshold if threshold is not None else settings.confidence_threshold

    def calculate(
        self,
        retrieval_result: HybridResult,
        citation_ok: bool,
        guardrail_results: dict[str, bool],
        response_text: str | None = None,
    ) -> ConfidenceResult:
        retrieval_score = self._compute_retrieval_score(retrieval_result)
        citation_score = 1.0 if citation_ok else 0.3
        guardrail_score = self._compute_guardrail_score(guardrail_results)
        supporting_count = len(retrieval_result.results) if retrieval_result.results else 0

        overall_score = (
            CONFIDENCE_WEIGHTS["retrieval_score"] * retrieval_score
            + CONFIDENCE_WEIGHTS["citation_score"] * citation_score
            + CONFIDENCE_WEIGHTS["guardrail_score"] * guardrail_score
        )

        return ConfidenceResult(
            overall_score=min(overall_score, 1.0),
            retrieval_score=retrieval_score,
            citation_score=citation_score,
            guardrail_score=guardrail_score,
            supporting_evidence_count=supporting_count,
            threshold_met=overall_score >= self.threshold,
        )

    def _compute_retrieval_score(self, result: HybridResult) -> float:
        if not result.results:
            return 0.0
        vec_scores = [
            r.vector_score for r in result.results if r.vector_score is not None
        ]
        bm25_scores = [
            r.bm25_score for r in result.results if r.bm25_score is not None
        ]
        avg_vec = sum(vec_scores) / len(vec_scores) if vec_scores else 0.0
        bm25_norm = 0.0
        if bm25_scores:
            max_bm25 = max(bm25_scores)
            bm25_norm = sum(bm25_scores) / len(bm25_scores) / max_bm25 if max_bm25 > 0 else 0.0
        coverage = min(len(result.results) / 5.0, 1.0)
        return (avg_vec * 0.5 + bm25_norm * 0.3 + coverage * 0.2)

    @staticmethod
    def _compute_guardrail_score(results: dict[str, bool]) -> float:
        if not results:
            return 1.0
        passed = sum(1 for v in results.values() if v)
        return passed / len(results)
