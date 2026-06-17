import re
import unicodedata


class TextCleaner:
    @staticmethod
    def normalize_unicode(text: str) -> str:
        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        return text

    @staticmethod
    def strip_non_ascii(text: str) -> str:
        return re.sub(r"[^\x20-\x7E\n]", "", text)

    @staticmethod
    def strip_common_headers_footers(text: str) -> str:
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if re.match(r"^(Page\s+\d+|-\s*\d+\s*-|\d+\s*/\s*\d+)$", stripped, re.IGNORECASE):
                continue
            if re.match(r"^[\s\-_=]{10,}$", stripped):
                continue
            cleaned.append(line)
        return "\n".join(cleaned)

    @staticmethod
    def normalize_legal_abbreviations(text: str) -> str:
        replacements = {
            r"\bIPC\b": "Indian Penal Code",
            r"\bCrPC\b": "Code of Criminal Procedure",
            r"\bCPC\b": "Code of Civil Procedure",
            r"\bSC\b": "Supreme Court",
            r"\bHC\b": "High Court",
            r"\bUOI\b": "Union of India",
        }
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        return text

    def clean(self, text: str) -> str:
        text = self.normalize_unicode(text)
        text = self.strip_common_headers_footers(text)
        text = self.normalize_whitespace(text)
        text = self.normalize_legal_abbreviations(text)
        return text
