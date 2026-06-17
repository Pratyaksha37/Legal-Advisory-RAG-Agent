from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    groq_api_key: str = ""
    llm_model_name: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048

    embedding_model_name: str = "BAAI/bge-large-en-v1.5"

    corpus_path: str = "data/corpus"
    faiss_index_path: str = "data/indexes/faiss"
    bm25_index_path: str = "data/indexes/bm25"
    embeddings_path: str = "data/embeddings"
    telemetry_path: str = "data/telemetry"

    retrieval_top_k: int = 10
    vector_weight: float = 0.5
    bm25_weight: float = 0.5
    fusion_method: str = "rrf"

    confidence_threshold: float = 0.6

    log_level: str = "INFO"
    app_name: str = "legal-rag"


settings = Settings()
