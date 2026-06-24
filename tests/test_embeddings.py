import math
import tempfile

import pytest

from src.chunking.chunk_models import ChunkResult, DocumentChunk
from src.embeddings import EmbeddingModel, EmbeddingPipeline, VectorStore


class TestVectorStore:
    def test_empty(self):
        vs = VectorStore()
        assert len(vs) == 0

    def test_add_and_retrieve(self):
        vs = VectorStore()
        vs.add("vec/001", "chk/001", [0.1, 0.2, 0.3])
        assert len(vs) == 1
        assert vs.get_chunk_id("vec/001") == "chk/001"
        assert vs.get_vector_id("chk/001") == "vec/001"
        assert vs.get_vector("vec/001") == [0.1, 0.2, 0.3]

    def test_to_dict(self):
        vs = VectorStore()
        vs.add("vec/001", "chk/001", [0.1, 0.2])
        data = vs.to_dict()
        assert len(data) == 1
        assert data[0]["vector_id"] == "vec/001"
        assert data[0]["chunk_id"] == "chk/001"

    def test_multiple_entries(self):
        vs = VectorStore()
        vs.add("vec/001", "chk/001", [0.1])
        vs.add("vec/002", "chk/002", [0.2])
        assert len(vs) == 2
        assert vs.vector_ids == ["vec/001", "vec/002"]
        assert vs.chunk_ids == ["chk/001", "chk/002"]


class TestEmbeddingModel:
    def test_embed_single(self):
        model = EmbeddingModel()
        vec = model.encode_single("test sentence")
        assert len(vec) == 1024
        norm = math.sqrt(sum(v * v for v in vec))
        assert abs(norm - 1.0) < 0.001

    def test_embed_batch(self):
        model = EmbeddingModel()
        vecs = model.encode(["test one", "test two"])
        assert len(vecs) == 2
        assert len(vecs[0]) == 1024

    def test_singleton(self):
        m1 = EmbeddingModel.get_instance()
        m2 = EmbeddingModel.get_instance()
        assert m1 is m2


class TestEmbeddingPipeline:
    def test_embed_chunks(self):
        chunks = [
            DocumentChunk(chunk_id="chk/001", document_id="doc/001", text="Section 302 IPC"),
            DocumentChunk(chunk_id="chk/002", document_id="doc/001", text="Article 14 Constitution"),
        ]
        result = ChunkResult(chunks=chunks, document_id="doc/001", total_chunks=2)
        pipeline = EmbeddingPipeline()
        vs = pipeline.embed_chunks(result)
        assert len(vs) == 2

    def test_empty_chunks(self):
        result = ChunkResult(chunks=[], document_id="doc/001", total_chunks=0)
        pipeline = EmbeddingPipeline()
        vs = pipeline.embed_chunks(result)
        assert len(vs) == 0

    def test_save_and_load(self):
        chunks = [DocumentChunk(chunk_id="chk/001", document_id="doc/001", text="Test section")]
        result = ChunkResult(chunks=chunks, document_id="doc/001", total_chunks=1)
        with tempfile.TemporaryDirectory() as d:
            pipeline = EmbeddingPipeline(embeddings_path=d)
            vs = pipeline.embed_chunks(result)
            pipeline.save(vs, "test.json")
            loaded = pipeline.load("test.json")
            assert len(loaded) == 1
            assert loaded.get_chunk_id(list(loaded.vectors.keys())[0]) == "chk/001"
