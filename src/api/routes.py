from fastapi import APIRouter, Depends

from src.api.dependencies import get_pipeline, get_telemetry
from src.api.request_models import AskRequest, AskResponse, HealthResponse
from src.pipeline.orchestrator import PipelineOrchestrator
from src.telemetry import TelemetryLogger

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask(
    request: AskRequest,
    pipeline: PipelineOrchestrator = Depends(get_pipeline),
) -> AskResponse:
    return await pipeline.ask(request.query, top_k=request.top_k)


@router.get("/health", response_model=HealthResponse)
async def health(
    pipeline: PipelineOrchestrator = Depends(get_pipeline),
) -> HealthResponse:
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        indexes_loaded=pipeline.retriever is not None,
        model_loaded=pipeline.llm is not None,
    )
