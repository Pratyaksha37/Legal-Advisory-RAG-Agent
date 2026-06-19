import re

class CitationEnforcer:
    def check(self, response: str, retrieved_texts: list[str]) -> tuple[bool, list[str]]:
        response_lower = response.lower()
        missing: list[str] = []

        for i, text in enumerate(retrieved_texts):
            citations = self._extract_citations(text)
            for citation in citations:
                if citation.lower() not in response_lower:
                    missing.append(citation)

        if missing:
            return False, missing
        return True, []

    @staticmethod
    def _extract_citations(text: str) -> list[str]:
        citations: list[str] = []
        patterns = [
            r"(?:Section|Sec\.|S\.)\s+\d+[A-Za-z]*(?:\s*[-–]\s*\d+[A-Za-z]*)?",
            r"(?:Article|Art\.)\s+\d+[A-Za-z]*(?:\s*[-–]\s*\d+[A-Za-z]*)?",
            r"(?:Rule)\s+\d+[A-Za-z]*(?:\s*[-–]\s*\d+[A-Za-z]*)?",
            r"(?:Schedule|Sch\.)\s+\d+",
            r"(?:Section|Sec\.)\s+\d+[A-Za-z]*\s+of\s+(?:the\s+)?(?:IPC|CrPC|CPC|BNS|BNSS|IEA)",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            citations.extend(matches)
        return list(set(citations))
