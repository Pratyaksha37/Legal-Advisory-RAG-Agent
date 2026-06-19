class VectorStore:
    def __init__(self):
        self.vectors: dict[str, list[float]] = {}
        self.chunk_map: dict[str, str] = {}
        self.vector_map: dict[str, str] = {}

    def add(self, vector_id: str, chunk_id: str, vector: list[float]) -> None:
        self.vectors[vector_id] = vector
        self.chunk_map[vector_id] = chunk_id
        self.vector_map[chunk_id] = vector_id

    def get_vector(self, vector_id: str) -> list[float] | None:
        return self.vectors.get(vector_id)

    def get_chunk_id(self, vector_id: str) -> str | None:
        return self.chunk_map.get(vector_id)

    def get_vector_id(self, chunk_id: str) -> str | None:
        return self.vector_map.get(chunk_id)

    def to_dict(self) -> list[dict]:
        return [
            {"vector_id": vid, "chunk_id": cid, "vector": vec}
            for vid, cid in self.chunk_map.items()
            if (vec := self.vectors.get(vid)) is not None
        ]

    def __len__(self) -> int:
        return len(self.vectors)

    @property
    def vector_ids(self) -> list[str]:
        return list(self.vectors.keys())

    @property
    def chunk_ids(self) -> list[str]:
        return list(self.vector_map.keys())
