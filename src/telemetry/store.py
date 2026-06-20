import json
from datetime import datetime
from pathlib import Path

from src.config.settings import settings
from src.core.logging import logger
from src.telemetry.models import TelemetryEvent


class TelemetryStore:
    def __init__(self, path: str = ""):
        self.path = Path(path or settings.telemetry_path)
        self.path.mkdir(parents=True, exist_ok=True)

    def append(self, event: TelemetryEvent) -> None:
        if not event.timestamp:
            event.timestamp = datetime.utcnow().isoformat()
        filepath = self._get_filepath()
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")

    def _get_filepath(self) -> Path:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        return self.path / f"{today}.jsonl"
