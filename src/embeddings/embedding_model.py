from sentence_transformers import SentenceTransformer

from src.config.settings import settings


class EmbeddingModel:
    _instance: "EmbeddingModel | None" = None

    def __init__(self, model_name: str = ""):
        self.model_name = model_name or settings.embedding_model_name
        self._model: SentenceTransformer | None = None

    def load(self) -> None:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

    @property
    def model(self) -> SentenceTransformer:     
        if self._model is None:
            self.load()
        return self._model

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        self.load()
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def encode_single(self, text: str) -> list[float]:
        return self.encode([text])[0]

    @classmethod
    def get_instance(cls, model_name: str = "") -> "EmbeddingModel":
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance
