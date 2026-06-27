from abc import ABC, abstractmethod

from src.core.logging import logger


class QueryRewriter(ABC):
    @abstractmethod
    def rewrite(self, query: str) -> str:
        pass


class LLMQueryRewriter(QueryRewriter):
    def __init__(self, llm_client=None):
        self.llm = llm_client

    def rewrite(self, query: str) -> str:
        if self.llm is None:
            return query

        prompt = (
            "Rewrite the following legal query to improve retrieval accuracy. "
            "Expand abbreviations, add relevant legal terms, and remove ambiguity. "
            "Return ONLY the rewritten query, no explanation.\n\n"
            f"Original query: {query}\n\n"
            "Rewritten query:"
        )
        try:
            result = self.llm.generate(prompt)
            rewritten = result.text.strip().strip('"').strip("'")
            if rewritten:
                logger.info("Query rewritten", original=query, rewritten=rewritten)
                return rewritten
        except Exception as e:
            logger.warning("Query rewriting failed", error=str(e))

        return query


class LegalQueryExpander(QueryRewriter):
    ABBREVIATIONS: dict[str, str] = {
        "ipc": "Indian Penal Code",
        "crpc": "Code of Criminal Procedure",
        "cpc": "Code of Civil Procedure",
        "bns": "Bharatiya Nyaya Sanhita",
        "bnss": "Bharatiya Nagarik Suraksha Sanhita",
        "sc": "Supreme Court",
        "hc": "High Court",
        "art": "Article",
        "sec": "Section",
    }

    def rewrite(self, query: str) -> str:
        import re
        rewritten = query
        for abbr, full in self.ABBREVIATIONS.items():
            rewritten = re.sub(
                rf"\b{abbr}\b", full, rewritten, flags=re.IGNORECASE
            )
        if rewritten != query:
            logger.info("Query expanded", original=query, rewritten=rewritten)
        return rewritten


class NoopQueryRewriter(QueryRewriter):
    def rewrite(self, query: str) -> str:
        return query
