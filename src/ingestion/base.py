from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DocumentData:
    text: str
    metadata: dict = field(default_factory=dict)
    file_path: str = ""
    file_type: str = ""


class DocumentLoader(ABC):
    @abstractmethod
    def load(self, path: str) -> DocumentData:
        pass

    def supports(self, path: str) -> bool:
        ext = Path(path).suffix.lower()
        return ext in self.supported_extensions()

    @classmethod
    @abstractmethod
    def supported_extensions(cls) -> set[str]:
        pass
