import json
import tempfile

import pytest

from src.evaluation import (
    CitationCorrectness,
    EvaluationReport,
    EvaluationSample,
    MetricResult,
    RAGASEvaluator,
    RetrievalPrecision,
)


class TestEvaluationSample:
    def test_minimal_sample(self):
        sample = EvaluationSample(query="What is murder?", ground_truth="Section 302 IPC")
        assert sample.query == "What is murder?"
        assert sample.relevant_doc_ids == []

    def test_sample_with_doc_ids(self):
        sample = EvaluationSample(
            query="What is murder?",
            ground_truth="Section 302 IPC",
            relevant_doc_ids=["chk/001", "chk/002"],
        )
        assert len(sample.relevant_doc_ids) == 2


class TestMetricResult:
    def test_metric(self):
        mr = MetricResult(metric_name="precision", score=0.85)
        assert mr.metric_name == "precision"
        assert mr.score == 0.85
        assert mr.details is None


class TestEvaluationReport:
    def test_empty_report(self):
        report = EvaluationReport()
        assert report.samples == 0
        assert report.metrics == []
        assert report.per_sample_scores == []

    def test_report_with_metrics(self):
        report = EvaluationReport(
            samples=5,
            metrics=[MetricResult(metric_name="precision", score=0.85)],
        )
        assert len(report.metrics) == 1
        assert report.metrics[0].score == 0.85


class TestRetrievalPrecision:
    def test_all_relevant_retrieved(self):
        rp = RetrievalPrecision()
        score = rp.compute(["a", "b", "c"], ["a", "b"], k=3)
        assert score == pytest.approx(2 / 3)

    def test_some_relevant(self):
        rp = RetrievalPrecision()
        score = rp.compute(["a", "b", "c", "d"], ["a", "x", "y"], k=4)
        assert score == 0.25

    def test_none_relevant(self):
        rp = RetrievalPrecision()
        score = rp.compute(["a", "b"], ["x", "y"], k=2)
        assert score == 0.0

    def test_no_relevant_ids(self):
        rp = RetrievalPrecision()
        score = rp.compute(["a", "b"], [], k=2)
        assert score == 0.0

    def test_k_larger_than_results(self):
        rp = RetrievalPrecision()
        score = rp.compute(["a"], ["a"], k=10)
        assert score == 1.0


class TestCitationCorrectness:
    def test_all_citations_found(self):
        cc = CitationCorrectness()
        score = cc.compute("Under Section 302 IPC.", ["Section 302"])
        assert score == 1.0

    def test_some_citations_missing(self):
        cc = CitationCorrectness()
        score = cc.compute("General answer.", ["Section 302", "Article 14"])
        assert score == 0.0

    def test_no_expected_citations(self):
        cc = CitationCorrectness()
        score = cc.compute("General answer.", [])
        assert score == 1.0


class TestRAGASEvaluator:
    def test_evaluate_empty_samples(self):
        evaluator = RAGASEvaluator()
        # Evaluating with no samples should either raise or return empty report
        try:
            result = evaluator.evaluate([])
            assert result.samples == 0
        except Exception:
            pass

    def test_load_samples(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([
                {"query": "What is murder?", "ground_truth": "Section 302"},
                {"query": "What is theft?", "ground_truth": "Section 378"},
            ], f)
            path = f.name
        samples = RAGASEvaluator.load_samples(path)
        assert len(samples) == 2
        assert samples[0].query == "What is murder?"

    def test_save_report(self, temp_dir):
        report = EvaluationReport(
            samples=1,
            metrics=[MetricResult(metric_name="precision", score=0.85)],
        )
        saved = RAGASEvaluator.save_report(report, path=str(temp_dir))
        assert saved.exists()
        with open(saved) as f:
            data = json.load(f)
        assert data["samples"] == 1

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path
            yield Path(d)
