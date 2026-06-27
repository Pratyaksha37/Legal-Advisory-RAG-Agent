from fastapi import APIRouter, Depends, File, UploadFile

from src.api.dependencies import get_pipeline, get_telemetry
from src.api.request_models import AskRequest, AskResponse, HealthResponse
from src.pipeline.ingestion_orchestrator import IngestionOrchestrator
from src.pipeline.orchestrator import PipelineOrchestrator
from src.telemetry import TelemetryLogger

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask(
    request: AskRequest,
    pipeline: PipelineOrchestrator = Depends(get_pipeline),
) -> AskResponse:
    return await pipeline.ask(request.query, top_k=request.top_k)


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    pipeline: PipelineOrchestrator = Depends(get_pipeline),
):
    if pipeline.retriever is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Indexes not loaded")

    orchestrator = IngestionOrchestrator()
    orchestrator.set_faiss(pipeline.retriever.faiss)

    contents = await file.read()
    try:
        saved_path = orchestrator.save_upload(contents, file.filename)
        result = orchestrator.ingest(saved_path)
        return result
    except ValueError as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(e))


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
