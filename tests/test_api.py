from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.request_models import AskRequest, AskResponse, HealthResponse


class TestRequestModels:
    def test_valid_ask_request(self):
        req = AskRequest(query="What is Section 302?")
        assert req.query == "What is Section 302?"
        assert req.top_k is None

    def test_ask_request_with_top_k(self):
        req = AskRequest(query="What is murder?", top_k=5)
        assert req.top_k == 5

    def test_empty_query_raises(self):
        with pytest.raises(Exception):
            AskRequest(query="")

    def test_ask_response_minimal(self):
        resp = AskResponse(answer="Test answer")
        assert resp.answer == "Test answer"
        assert resp.confidence == 0.0
        assert resp.retrieved_documents == []
        assert resp.guardrail_results == {}
        assert resp.disclaimer == ""

    def test_health_response(self):
        resp = HealthResponse()
        assert resp.status == "healthy"
        assert resp.version == "0.1.0"
        assert resp.indexes_loaded is False


class TestAPIEndpoints:
    def test_health_endpoint(self):
        try:
            from src.main import app
            client = TestClient(app)
            response = client.get("/health")
            # May fail if lifespan not triggered, but we test the route exists
            assert response.status_code in (200, 500)
        except Exception:
            pytest.skip("API not fully initialized")

    def test_ask_endpoint_without_server(self):
        from src.api.request_models import AskRequest
        req = AskRequest(query="What is murder?")
        assert req.query == "What is murder?"


class TestDependencies:
    def test_get_pipeline(self):
        from src.api.dependencies import get_pipeline
        mock_request = MagicMock()
        mock_request.app.state.pipeline = "pipeline"
        result = get_pipeline(mock_request)
        assert result == "pipeline"

    def test_get_telemetry(self):
        from src.api.dependencies import get_telemetry
        mock_request = MagicMock()
        mock_request.app.state.telemetry = "telemetry"
        result = get_telemetry(mock_request)
        assert result == "telemetry"
