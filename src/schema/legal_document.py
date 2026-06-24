import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class LegalDocumentType(str, Enum):
    CONSTITUTION_ARTICLE = "constitution_article"
    ACT = "act"
    SECTION = "section"
    RULE = "rule"
    NOTIFICATION = "notification"
    JUDGMENT = "judgment"
    CUSTOM = "custom"


class LegalDocument(BaseModel):
    model_config = {"extra": "allow", "str_strip_whitespace": True}

    id: str
    doc_type: LegalDocumentType
    title: str
    text: str

    act_name: Optional[str] = None
    parent_id: Optional[str] = None
    chapter: Optional[str] = None
    part: Optional[str] = None

    article_number: Optional[str] = None
    section_number: Optional[str] = None
    rule_number: Optional[str] = None
    notification_number: Optional[str] = None

    case_citation: Optional[str] = None
    court: Optional[str] = None
    date: Optional[datetime.date] = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class LegalCorpus(BaseModel):
    documents: list[LegalDocument]

    model_config = {"extra": "ignore"}
