import re
from pathlib import Path


class MetadataExtractor:
    @staticmethod
    def from_filename(path: str) -> dict:
        stem = Path(path).stem
        metadata = {"source_filename": stem}

        year_match = re.search(r"(?<!\d)(?:18|19|20)\d{2}(?!\d)", stem)
        if year_match:
            metadata["year"] = int(year_match.group(0))

        act_match = re.search(
            r"(Constitution|IPC|CrPC|CPC|Indian\s+Penal\s+Code|Evidence\s+Act|Contract\s+Act)",
            stem,
            re.IGNORECASE,
        )
        if act_match:
            metadata["act_name"] = act_match.group(1)

        doc_type_match = re.search(
            r"(Article|Section|Rule|Notification|Judgment|Act|Amendment|Schedule|Order)",
            stem,
            re.IGNORECASE,
        )
        if doc_type_match:
            metadata["document_type_hint"] = doc_type_match.group(1).lower()

        return metadata

    @staticmethod
    def from_content(text: str) -> dict:
        metadata = {}

        act_match = re.search(
            r"(?:THE\s+)?([A-Z][A-Z\s]+(?:ACT|CODE|RULES|REGULATIONS))\b",
            text[:2000],
        )
        if act_match:
            metadata["act_name"] = act_match.group(1).strip()

        preamble_match = re.search(
            r"(?:Preamble|An Act to|In exercise of the powers)",
            text[:3000],
            re.IGNORECASE,
        )
        if preamble_match:
            metadata["has_preamble"] = True

        date_match = re.search(
            r"\b(\d{1,2}[-/]\d{1,2}[-/](?:19|20)\d{2})\b", text[:2000]
        )
        if date_match:
            metadata["date_found"] = date_match.group(1)

        return metadata

    def extract(self, path: str, text: str) -> dict:
        metadata = {}
        metadata.update(self.from_filename(path))
        metadata.update(self.from_content(text))
        return metadata
