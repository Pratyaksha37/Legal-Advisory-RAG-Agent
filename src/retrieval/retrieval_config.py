from pydantic import BaseModel, Field


class RetrievalConfig(BaseModel):
    vector_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    bm25_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    top_k: int = Field(default=10, ge=1, le=100)
    fusion_method: str = Field(default="rrf", pattern="^(rrf|weighted)$")
    rrf_k: int = Field(default=60, ge=1)
    min_vector_score: float = Field(default=0.0, ge=0.0)
    min_bm25_score: float = Field(default=0.0, ge=0.0)
