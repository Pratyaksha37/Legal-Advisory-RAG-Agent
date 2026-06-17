from enum import Enum


class LegalDocumentType(str, Enum):
    CONSTITUTION_ARTICLE = "constitution_article"
    ACT = "act"
    SECTION = "section"
    RULE = "rule"
    NOTIFICATION = "notification"
    JUDGMENT = "judgment"
    CUSTOM = "custom"


LEGAL_DOCUMENT_TYPES: list[str] = [t.value for t in LegalDocumentType]


SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".docx", ".txt", ".json"}


DEFAULT_CHUNK_SIZE: int = 1024
DEFAULT_CHUNK_OVERLAP: int = 50
MAX_CHUNK_SIZE: int = 2048


HIERARCHY_PATTERNS: dict[str, str] = {
    "article": r"(?:Article|Art\.)\s+\d+[A-Z]*(?:\s*[-–]\s*\d+[A-Z]*)?",
    "section": r"(?:Section|Sec\.)\s+\d+[A-Z]*(?:\s*[-–]\s*\d+[A-Z]*)?",
    "rule": r"(?:Rule)\s+\d+[A-Z]*(?:\s*[-–]\s*\d+[A-Z]*)?",
    "clause": r"\([a-z]\)\s*",
    "paragraph": r"(?:Paragraph|Para\.)\s+\d+",
}


EMBEDDING_DIMENSION: int = 1024
EMBEDDING_BATCH_SIZE: int = 32
EMBEDDING_NORMALIZE: bool = True


FAISS_INDEX_FILE: str = "index.faiss"
FAISS_MAPPING_FILE: str = "index_mapping.json"


BM25_INDEX_FILE: str = "bm25_index.pkl"
BM25_CORPUS_FILE: str = "bm25_corpus.pkl"


CORPUS_FILE: str = "legal_corpus.json"
CHUNKS_FILE: str = "chunks.json"
CHECKPOINT_FILE: str = ".embedding_checkpoint"


CONFIDENCE_WEIGHTS: dict[str, float] = {
    "retrieval_score": 0.4,
    "citation_score": 0.3,
    "guardrail_score": 0.3,
}


DEFAULT_DISCLAIMER: str = (
    "This information is for educational purposes only "
    "and does not constitute legal advice. "
    "Please consult a qualified legal professional."
)
