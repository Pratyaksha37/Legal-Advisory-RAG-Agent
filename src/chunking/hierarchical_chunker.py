import re
import uuid
from typing import Optional

from src.chunking.base import Chunker
from src.chunking.chunk_models import ChunkResult, DocumentChunk
from src.config.constants import HIERARCHY_PATTERNS
from src.schema.legal_document import LegalDocument


class BoundaryMatch:
    def __init__(self, start: int, end: int, level: int, label: str, heading: str):
        self.start = start
        self.end = end
        self.level = level
        self.label = label
        self.heading = heading


class HierarchicalChunker(Chunker):
    def __init__(self):
        self._patterns: list[tuple[int, str, str, re.Pattern]] = []
        for level, (label, pattern) in enumerate(
            [
                ("part", r"(?:PART|Part)\s+[IVXLCDM]+\b"),
                ("chapter", r"(?:CHAPTER|Chapter|Ch\.)\s+[\dIVXLCDM]+\b"),
                ("article", HIERARCHY_PATTERNS["article"]),
                ("section", HIERARCHY_PATTERNS["section"]),
                ("rule", HIERARCHY_PATTERNS["rule"]),
                ("clause", HIERARCHY_PATTERNS["clause"]),
            ],
            start=1,
        ):
            self._patterns.append((level, label, f"^{pattern}", re.compile(f"^{pattern}", re.MULTILINE)))
            self._patterns.append((level, label, pattern, re.compile(pattern, re.MULTILINE)))

    def chunk(self, document: LegalDocument) -> ChunkResult:
        text = document.text
        boundaries = self._find_boundaries(text)

        if not boundaries:
            chunks = self._fallback_chunk(document)
        else:
            chunks = self._build_chunks(document, text, boundaries)

        return ChunkResult(
            chunks=chunks,
            document_id=document.id,
            total_chunks=len(chunks),
        )

    def chunk_text(self, text: str, document_id: str = "") -> ChunkResult:
        fake_doc = LegalDocument(id=document_id, doc_type="custom", title="", text=text)
        return self.chunk(fake_doc)

    def _find_boundaries(self, text: str) -> list[BoundaryMatch]:
        boundaries: list[BoundaryMatch] = []

        for level, label, _description, pattern in self._patterns:
            for match in pattern.finditer(text):
                heading = match.group(0).strip()
                boundaries.append(
                    BoundaryMatch(
                        start=match.start(),
                        end=match.end(),
                        level=level,
                        label=label,
                        heading=heading,
                    )
                )

        boundaries.sort(key=lambda b: (b.start, b.level))
        boundaries = self._deduplicate_overlapping(boundaries)
        return boundaries

    @staticmethod
    def _deduplicate_overlapping(boundaries: list[BoundaryMatch]) -> list[BoundaryMatch]:
        if not boundaries:
            return []
        selected = [boundaries[0]]
        for b in boundaries[1:]:
            if b.start >= selected[-1].start:
                selected.append(b)
        return selected

    def _build_chunks(
        self, document: LegalDocument, text: str, boundaries: list[BoundaryMatch]
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        stack: list[DocumentChunk] = []

        for i, boundary in enumerate(boundaries):
            chunk_start = boundary.start
            chunk_end = boundaries[i + 1].start if i + 1 < len(boundaries) else len(text)

            chunk_text = text[chunk_start:chunk_end].strip()
            if not chunk_text:
                continue

            while stack and stack[-1].hierarchy_level >= boundary.level:
                stack.pop()

            parent_id = stack[-1].chunk_id if stack else None

            chunk = DocumentChunk(
                chunk_id=f"chk/{uuid.uuid4().hex[:12]}",
                document_id=document.id,
                parent_id=parent_id,
                text=chunk_text,
                hierarchy_level=boundary.level,
                heading=boundary.heading,
                chunk_type=boundary.label,
                metadata={
                    "source_doc_title": document.title,
                    "act_name": document.act_name,
                },
            )
            chunks.append(chunk)
            stack.append(chunk)

        return chunks

    def _fallback_chunk(self, document: LegalDocument) -> list[DocumentChunk]:
        paragraphs = re.split(r"\n\n+", document.text.strip())
        chunks: list[DocumentChunk] = []

        for para in paragraphs:
            para = para.strip()
            if not para or len(para) < 20:
                continue

            chunk = DocumentChunk(
                chunk_id=f"chk/{uuid.uuid4().hex[:12]}",
                document_id=document.id,
                parent_id=None,
                text=para,
                hierarchy_level=0,
                heading=None,
                chunk_type="fallback",
                metadata={
                    "source_doc_title": document.title,
                    "act_name": document.act_name,
                },
            )
            chunks.append(chunk)

        return chunks
