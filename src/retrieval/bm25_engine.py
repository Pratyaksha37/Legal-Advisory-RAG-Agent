import json
import pickle
import re
import ssl
import string
from pathlib import Path

import nltk
from rank_bm25 import BM25Okapi

from src.config.constants import BM25_CORPUS_FILE, BM25_INDEX_FILE
from src.config.settings import settings
from src.core.exceptions import RetrievalError
from src.core.logging import logger


def _ensure_nltk_data() -> None:
    try:
        _create_unverified = ssl._create_unverified_context
    except AttributeError:
        _create_unverified = None

    resources = ["tokenizers/punkt_tab", "tokenizers/punkt", "corpora/stopwords"]
    for resource in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            try:
                nltk.download(resource.replace("tokenizers/", "").replace("corpora/", ""), quiet=True)
            except Exception:
                if _create_unverified:
                    ssl._create_default_https_context = _create_unverified
                    nltk.download(resource.replace("tokenizers/", "").replace("corpora/", ""), quiet=True)


_ensure_nltk_data()
STOPWORDS = set(nltk.corpus.stopwords.words("english"))

LEGAL_ABBREVIATIONS: dict[str, str] = {
    "ipc": "indian penal code",
    "crpc": "code of criminal procedure",
    "cpc": "code of civil procedure",
    "bnss": "bharatiya nagarik suraksha sanhita",
    "bns": "bharatiya nyaya sanhita",
    "sc": "supreme court",
    "hc": "high court",
    "uoi": "union of india",
    "art": "article",
    "sec": "section",
    "ch": "chapter",
    "sch": "schedule",
}


class BM25Engine:
    def __init__(self, index_path: str = ""):
        self.index_path = Path(index_path or settings.bm25_index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.bm25: BM25Okapi | None = None
        self.chunk_ids: list[str] = []
        self.tokenized_corpus: list[list[str]] = []

    def build(self, chunks: list[dict]) -> None:
        self.chunk_ids = []
        self.tokenized_corpus = []

        for chunk in chunks:
            text = chunk.get("text", "")
            tokens = self._tokenize(text)
            if not tokens:
                continue
            self.tokenized_corpus.append(tokens)
            self.chunk_ids.append(chunk["chunk_id"])

        if not self.tokenized_corpus:
            raise RetrievalError("No tokenizable content to build BM25 index")

        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=1.5, b=0.75)

        logger.info(
            "Built BM25 index",
            documents=len(self.chunk_ids),
            path=str(self.index_path),
        )

    def save(self) -> None:
        if self.bm25 is None:
            raise RetrievalError("No index to save")

        index_file = self.index_path / BM25_INDEX_FILE
        with open(index_file, "wb") as f:
            pickle.dump(self.bm25, f)

        corpus_file = self.index_path / BM25_CORPUS_FILE
        with open(corpus_file, "wb") as f:
            pickle.dump(
                {"chunk_ids": self.chunk_ids, "tokenized_corpus": self.tokenized_corpus},
                f,
            )

        logger.info("Saved BM25 index", path=str(index_file))

    def load(self) -> None:
        index_file = self.index_path / BM25_INDEX_FILE
        corpus_file = self.index_path / BM25_CORPUS_FILE

        if not index_file.exists():
            raise RetrievalError(f"BM25 index not found: {index_file}")

        with open(index_file, "rb") as f:
            self.bm25 = pickle.load(f)
        with open(corpus_file, "rb") as f:
            data = pickle.load(f)
            self.chunk_ids = data["chunk_ids"]
            self.tokenized_corpus = data["tokenized_corpus"]

        logger.info(
            "Loaded BM25 index",
            path=str(index_file),
            documents=len(self.chunk_ids),
        )

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if self.bm25 is None:
            raise RetrievalError("BM25 index not loaded. Call load() or build() first.")

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)

        results: list[tuple[str, float]] = []
        for idx, score in indexed[:top_k]:
            if score <= 0:
                continue
            results.append((self.chunk_ids[idx], float(score)))

        return results

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        text = self._expand_abbreviations(text)
        text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        tokens = nltk.word_tokenize(text)
        return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    @staticmethod
    def _expand_abbreviations(text: str) -> str:
        pattern = re.compile(
            r"\b(" + "|".join(sorted(LEGAL_ABBREVIATIONS, key=len, reverse=True)) + r")\b",
            re.IGNORECASE,
        )
        return pattern.sub(lambda m: LEGAL_ABBREVIATIONS[m.group(1).lower()], text)
