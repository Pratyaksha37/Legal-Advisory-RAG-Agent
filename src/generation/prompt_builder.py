from pathlib import Path

from src.retrieval.retrieval_result import RetrievalResult


class PromptBuilder:
    def __init__(self, template_dir: str = ""):
        if not template_dir:
            template_dir = str(Path(__file__).parent / "templates")
        self.template_dir = Path(template_dir)

    def build_grounded_prompt(self, query: str, documents: list[RetrievalResult]) -> str:
        template = self._load_template("grounding.txt")
        docs_section = ""
        for doc in documents:
            heading = f" ({doc.heading})" if doc.heading else ""
            act = f"\nAct: {doc.act_name}" if doc.act_name else ""
            docs_section += (
                f"--- Source: {doc.document_title}{heading}\n"
                f"{act}\n"
                f"{doc.text}\n\n"
            )
        prompt = template.replace("{{ query }}", query).replace("{{ documents }}", docs_section)
        return prompt

    def build_refusal_prompt(self, reason: str) -> str:
        template = self._load_template("refusal.txt")
        return template.replace("{{ reason }}", reason)

    def _load_template(self, filename: str) -> str:
        path = self.template_dir / filename
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
