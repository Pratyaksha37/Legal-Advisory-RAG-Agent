class LegalRAGError(Exception):
    """Base exception for all Legal RAG platform errors."""


class ConfigurationError(LegalRAGError):
    """Raised when system configuration is invalid or missing."""


class IngestionError(LegalRAGError):
    """Raised when document ingestion fails."""


class SchemaValidationError(LegalRAGError):
    """Raised when document schema validation fails."""


class ChunkingError(LegalRAGError):
    """Raised when document chunking fails."""


class EmbeddingError(LegalRAGError):
    """Raised when embedding generation fails."""


class RetrievalError(LegalRAGError):
    """Raised when document retrieval fails."""


class GenerationError(LegalRAGError):
    """Raised when LLM generation fails."""


class GuardrailError(LegalRAGError):
    """Raised when a guardrail check fails."""


class PipelineError(LegalRAGError):
    """Raised when the orchestration pipeline fails."""
