from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api import router
from src.config.settings import settings
from src.core.logging import setup_logging, logger
from src.pipeline.orchestrator import PipelineOrchestrator
from src.telemetry import TelemetryLogger


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.log_level)
    pipeline = PipelineOrchestrator()
    pipeline.initialize()
    app.state.pipeline = pipeline
    app.state.telemetry = TelemetryLogger()
    logger.info("Application started", app_name=settings.app_name)
    yield
    logger.info("Application shutting down", app_name=settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Legal Advisory RAG Platform for Indian Law",
    lifespan=lifespan,
)

app.include_router(router)
