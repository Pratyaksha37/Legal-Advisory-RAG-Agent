import re
from datetime import date, datetime

from src.schema.legal_document import LegalDocumentType


class SchemaValidator:
    @staticmethod
    def normalize_doc_type(raw: str) -> LegalDocumentType:
        lookup = raw.strip().lower().replace(" ", "_")
        for dt in LegalDocumentType:
            if dt.value == lookup:
                return dt
        return LegalDocumentType.CUSTOM

    @staticmethod
    def extract_section_number(text: str) -> str | None:
        match = re.search(
            r"(?:Section|Sec\.|S\.)\s*(\d+[A-Za-z]*(?:\s*[-–]\s*\d+[A-Za-z]*)?)",
            text[:500],
            re.IGNORECASE,
        )
        return match.group(1) if match else None

    @staticmethod
    def extract_article_number(text: str) -> str | None:
        match = re.search(
            r"(?:Article|Art\.)\s*(\d+[A-Za-z]*(?:\s*[-–]\s*\d+[A-Za-z]*)?)",
            text[:500],
            re.IGNORECASE,
        )
        return match.group(1) if match else None

    @staticmethod
    def extract_rule_number(text: str) -> str | None:
        match = re.search(
            r"Rule\s*(\d+[A-Za-z]*(?:\s*[-–]\s*\d+[A-Za-z]*)?)",
            text[:500],
            re.IGNORECASE,
        )
        return match.group(1) if match else None

    @staticmethod
    def try_parse_date(raw: str) -> date | None:
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%B %d, %Y"):
            try:
                return datetime.strptime(raw.strip(), fmt).date()
            except (ValueError, AttributeError):
                continue
        return None

    @staticmethod
    def coerce_date(value: object) -> date | None:
        if value is None:
            return None
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            return SchemaValidator.try_parse_date(value)
        return None
