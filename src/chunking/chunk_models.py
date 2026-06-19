from typing import Any, Optional

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    parent_id: Optional[str] = None
    text: str
    hierarchy_level: int = 0
    heading: Optional[str] = None
    chunk_type: str = "fallback"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkResult(BaseModel):
    chunks: list[DocumentChunk]
    document_id: str
    total_chunks: int
