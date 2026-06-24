import json
from pathlib import Path
from typing import Optional

from src.config.settings import settings
from src.core.logging import logger
from src.evaluation.metrics import (
    CitationCorrectness,
    EvaluationReport,
    EvaluationSample,
    MetricResult,
    RetrievalPrecision,
)
from src.retrieval import BM25Engine, FAISSIndexManager, HybridRetriever
from src.retrieval.retrieval_config import RetrievalConfig


class RAGASEvaluator:
    def __init__(self, retriever: Optional[HybridRetriever] = None):
        self.retriever = retriever
        self.retrieval_precision = RetrievalPrecision()
        self.citation_correctness = CitationCorrectness()

    def evaluate(
        self,
        samples: list[EvaluationSample],
        top_k: int = 10,
    ) -> EvaluationReport:
        if self.retriever is None:
            self.retriever = HybridRetriever()
            self.retriever.load()

        report = EvaluationReport(samples=len(samples))

        retrieval_precisions = []
        citation_scores = []

        for sample in samples:
            result = self.retriever.retrieve(sample.query, top_k=top_k)
            retrieved_ids = [r.chunk_id for r in result.results]

            rp = self.retrieval_precision.compute(retrieved_ids, sample.relevant_doc_ids, k=top_k)
            retrieval_precisions.append(rp)

            cs = self.citation_correctness.compute(sample.ground_truth, sample.relevant_doc_ids)
            citation_scores.append(cs)

            report.per_sample_scores.append({
                "query": sample.query,
                "retrieval_precision": rp,
                "citation_correctness": cs,
                "retrieved_count": len(retrieved_ids),
            })

        report.metrics = [
            MetricResult(
                metric_name="retrieval_precision",
                score=sum(retrieval_precisions) / len(retrieval_precisions) if retrieval_precisions else 0.0,
            ),
            MetricResult(
                metric_name="citation_correctness",
                score=sum(citation_scores) / len(citation_scores) if citation_scores else 0.0,
            ),
        ]

        logger.info("Evaluation complete", metrics={m.metric_name: round(m.score, 3) for m in report.metrics})
        return report

    @staticmethod
    def load_samples(path: str) -> list[EvaluationSample]:
        filepath = Path(path)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [EvaluationSample(**item) for item in data]

    @staticmethod
    def save_report(report: EvaluationReport, path: str = "") -> Path:
        filepath = Path(path or settings.telemetry_path) / "evaluation_report.json"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))
        logger.info("Saved evaluation report", path=str(filepath))
        return filepath
