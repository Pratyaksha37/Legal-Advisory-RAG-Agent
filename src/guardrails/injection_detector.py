import re


INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|directions)", re.IGNORECASE),
    re.compile(r"forget\s+(all\s+)?(previous|prior|above)", re.IGNORECASE),
    re.compile(r"system\s+prompt", re.IGNORECASE),
    re.compile(r"you\s+are\s+(now|not)\s+(a\s+)?", re.IGNORECASE),
    re.compile(r"act\s+as\s+(if\s+)?", re.IGNORECASE),
    re.compile(r"do\s+not\s+(follow|obey|adhere\s+to)", re.IGNORECASE),
    re.compile(r"role[- ]?play", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"DAN|do\s+anything\s+now", re.IGNORECASE),
    re.compile(r"bypass\s+(restrictions|safeguards|filters)", re.IGNORECASE),
]


class InjectionDetector:
    def check(self, query: str) -> tuple[bool, str]:
        for pattern in INJECTION_PATTERNS:
            match = pattern.search(query)
            if match:
                return False, f"Prompt injection detected: matched pattern '{match.group(0)}'"
        return True, ""
