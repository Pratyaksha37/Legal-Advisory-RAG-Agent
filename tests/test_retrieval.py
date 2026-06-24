import json
import tempfile

import numpy as np
import pytest

from src.config.constants import FAISS_INDEX_FILE, FAISS_MAPPING_FILE
from src.core.exceptions import RetrievalError
from src.embeddings import EmbeddingModel
from src.retrieval import BM25Engine, FAISSIndexManager, HybridRetriever, RetrievalConfig
from src.retrieval.retrieval_result import HybridResult, RetrievalResult


class TestFAISSIndexManager:
    def test_build_and_search(self, sample_chunks):
        model = EmbeddingModel()
        vectors = {}
        chunk_map = {}
        for i, chunk in enumerate(sample_chunks):
            vid = f"vec/{i:03d}"
            vec = model.encode_single(chunk["text"])
            vectors[vid] = vec
            chunk_map[vid] = chunk["chunk_id"]

        manager = FAISSIndexManager()
        manager.build(vectors, chunk_map)
        assert manager.index.ntotal == len(sample_chunks)

        query_vec = model.encode_single("Section 1")
        results = manager.search(query_vec, top_k=2)
        assert len(results) >= 1
        chunk_id, score = results[0]
        assert chunk_id
        assert score > 0

    def test_save_and_load(self, sample_chunks, temp_dir):
        model = EmbeddingModel()
        vectors = {}
        chunk_map = {}
        for i, chunk in enumerate(sample_chunks[:2]):
            vid = f"vec/{i:03d}"
            vectors[vid] = model.encode_single(chunk["text"])
            chunk_map[vid] = chunk["chunk_id"]

        manager = FAISSIndexManager(index_path=str(temp_dir))
        manager.build(vectors, chunk_map)
        manager.save()

        index_file = temp_dir / FAISS_INDEX_FILE
        mapping_file = temp_dir / FAISS_MAPPING_FILE
        assert index_file.exists()
        assert mapping_file.exists()

        manager2 = FAISSIndexManager(index_path=str(temp_dir))
        manager2.load()
        assert manager2.index.ntotal == 2

    def test_search_without_load_raises(self):
        manager = FAISSIndexManager()
        with pytest.raises(RetrievalError):
            manager.search([0.0] * 1024)

    def test_build_empty_raises(self):
        manager = FAISSIndexManager()
        with pytest.raises(RetrievalError):
            manager.build({}, {})

    def test_incremental_add(self, temp_dir):
        model = EmbeddingModel()
        v1 = {"vec/001": model.encode_single("first")}
        m1 = {"vec/001": "chk/001"}
        manager = FAISSIndexManager(index_path=str(temp_dir))
        manager.build(v1, m1)
        assert manager.index.ntotal == 1

        v2 = {"vec/002": model.encode_single("second")}
        m2 = {"vec/002": "chk/002"}
        manager.add_vectors(v2, m2)
        assert manager.index.ntotal == 2

        results = manager.search(model.encode_single("second"), top_k=2)
        result_ids = [r[0] for r in results]
        assert "chk/002" in result_ids

    @pytest.fixture
    def sample_chunks(self):
        return [
            {"chunk_id": "chk/001", "text": "Section 1. This is the first section."},
            {"chunk_id": "chk/002", "text": "Section 2. This is the second section."},
            {"chunk_id": "chk/003", "text": "Article 14. Equality before law."},
        ]


class TestBM25Engine:
    def test_build_and_search(self, sample_chunks):
        engine = BM25Engine()
        engine.build(sample_chunks)
        assert len(engine.chunk_ids) == 3
        results = engine.search("section", top_k=2)
        assert len(results) >= 1

    def test_search_abbreviations(self, sample_chunks):
        engine = BM25Engine()
        engine.build(sample_chunks)
        results = engine.search("IPC section", top_k=2)
        assert isinstance(results, list)

    def test_search_empty_query(self, sample_chunks):
        engine = BM25Engine()
        engine.build(sample_chunks)
        results = engine.search("", top_k=2)
        assert results == []

    def test_save_and_load(self, sample_chunks, temp_dir):
        engine = BM25Engine(index_path=str(temp_dir))
        engine.build(sample_chunks)
        engine.save()
        engine2 = BM25Engine(index_path=str(temp_dir))
        engine2.load()
        assert len(engine2.chunk_ids) == 3

    def test_tokenize(self, sample_chunks):
        engine = BM25Engine()
        engine.build(sample_chunks)
        tokens = engine._tokenize("IPC Section 302")
        assert "ipc" not in tokens
        assert "indian" in tokens or "section" in tokens

    def test_build_no_content_raises(self):
        engine = BM25Engine()
        with pytest.raises(RetrievalError):
            engine.build([{"chunk_id": "1", "text": ""}])

    @pytest.fixture
    def sample_chunks(self):
        return [
            {"chunk_id": "chk/001", "text": "Section 302. Punishment for murder."},
            {"chunk_id": "chk/002", "text": "Article 14. Equality before law."},
            {"chunk_id": "chk/003", "text": "Section 1. Short title."},
        ]


class TestHybridRetriever:
    def test_retrieve_with_mocked_data(self, sample_retrieval_results, temp_dir):
        config = RetrievalConfig(top_k=5)
        retriever = HybridRetriever(config=config)

        query_vec = [0.1] * 1024
        query = "test query"

        result = HybridResult(
            query=query,
            results=sample_retrieval_results,
            total_results=len(sample_retrieval_results),
        )
        assert len(result.results) == 1

    def test_retrieve_empty_results(self, temp_dir):
        retriever = HybridRetriever()
        result = HybridResult(query="test", results=[], total_results=0)
        assert result.total_results == 0

    @pytest.fixture
    def sample_retrieval_results(self):
        return [
            RetrievalResult(
                chunk_id="chk/001",
                text="Section 302 IPC",
                heading="Section 302",
                document_title="IPC",
                vector_score=0.75,
                bm25_score=15.0,
                combined_score=0.85,
            ),
        ]


class TestRetrievalConfig:
    def test_default_config(self):
        config = RetrievalConfig()
        assert config.top_k == 10
        assert config.fusion_method == "rrf"
        assert config.rrf_k == 60

    def test_custom_config(self):
        config = RetrievalConfig(top_k=5, fusion_method="weighted", vector_weight=0.7, bm25_weight=0.3)
        assert config.top_k == 5
        assert config.vector_weight == 0.7

    def test_invalid_fusion_raises(self):
        with pytest.raises(Exception):
            RetrievalConfig(fusion_method="invalid")
