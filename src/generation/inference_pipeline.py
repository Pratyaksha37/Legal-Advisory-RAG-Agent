from src.core.exceptions import GenerationError
from src.core.logging import logger
from src.generation.llm_client import GenerationResult, LLMClient
from src.generation.prompt_builder import PromptBuilder
from src.retrieval.retrieval_result import HybridResult


class InferencePipeline:
    def __init__(
        self,
        llm: LLMClient | None = None,
        prompt_builder: PromptBuilder | None = None,
    ):
        self.llm = llm or LLMClient()
        self.prompt_builder = prompt_builder or PromptBuilder()

    def generate(
        self, query: str, retrieval_result: HybridResult
    ) -> GenerationResult:
        if not retrieval_result.results:
            return GenerationResult(
                text=self.prompt_builder.build_refusal_prompt(
                    "No relevant legal documents found for this query."
                ),
                model=self.llm.model,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                latency_ms=0.0,
            )

        prompt = self.prompt_builder.build_grounded_prompt(
            query, retrieval_result.results
        )
        return self.llm.generate(prompt)
