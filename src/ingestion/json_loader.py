import json
from pathlib import Path

from src.ingestion.base import DocumentData, DocumentLoader


class JSONLoader(DocumentLoader):
    def load(self, path: str) -> DocumentData:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            text = data.get("text", data.get("content", json.dumps(data)))
            metadata = {k: v for k, v in data.items() if k not in ("text", "content")}
        elif isinstance(data, list):
            text = "\n\n".join(
                item.get("text", item.get("content", json.dumps(item)))
                for item in data
            )
            metadata = {"document_count": len(data)}
        else:
            text = str(data)
            metadata = {}

        return DocumentData(
            text=text,
            metadata=metadata,
            file_path=str(Path(path).resolve()),
            file_type="json",
        )

    @classmethod
    def supported_extensions(cls) -> set[str]:
        return {".json"}
