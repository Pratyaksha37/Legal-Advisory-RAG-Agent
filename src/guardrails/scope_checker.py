import re


IN_SCOPE_TERMS: list[re.Pattern] = [
    re.compile(r"\b(indian|bharat|bharatiya)\s+(law|legal|constitution|penal|criminal|code|act|section|article|rule|judgment)\b", re.IGNORECASE),
    re.compile(r"\b(supreme\s+court|high\s+court|district\s+court|tribunal)\b", re.IGNORECASE),
    re.compile(r"\b(IPC|CrPC|CPC|BNS|BNSS|IEA)\b", re.IGNORECASE),
    re.compile(r"\b(article|section|schedule|clause)\s+\d+", re.IGNORECASE),
    re.compile(r"\b(fundamental\s+rights|directive\s+principles|constitutional\s+remedies)\b", re.IGNORECASE),
    re.compile(r"\b(murder|theft|robbery|assault|fraud|cheating|defamation|dowry|rape|kidnapping)\b", re.IGNORECASE),
    re.compile(r"\b(contract|property|evidence|civil|writ|petition|appeal)\b", re.IGNORECASE),
    re.compile(r"\b(legal|law|statute|legislation|ordinance|regulation)\b", re.IGNORECASE),
]


OUT_OF_SCOPE_TERMS: list[re.Pattern] = [
    re.compile(r"\b(american|us\s+law|uk\s+law|copyright|patent|trademark)\b", re.IGNORECASE),
    re.compile(r"\b(medicine|prescription|treatment|diagnosis|surgery)\b", re.IGNORECASE),
    re.compile(r"\b(recipe|cooking|baking)\b", re.IGNORECASE),
    re.compile(r"\b(stock|investment|trading|cryptocurrency)\b", re.IGNORECASE),
    re.compile(r"\b(religion|religious\s+advice|theology)\b", re.IGNORECASE),
]


class ScopeChecker:
    def check(self, query: str) -> tuple[bool, str]:
        for pattern in OUT_OF_SCOPE_TERMS:
            if pattern.search(query):
                return False, f"Query outside Indian law scope: matched '{pattern.pattern}'"
        for pattern in IN_SCOPE_TERMS:
            if pattern.search(query):
                return True, ""
        has_question = "?" in query or query.strip().endswith("?")
        return has_question, ""
