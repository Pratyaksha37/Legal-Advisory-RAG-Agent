from groq import Groq

from src.config.settings import settings
from src.core.exceptions import GenerationError
from src.core.logging import logger


class GenerationResult:
    def __init__(self, text: str, model: str, usage: dict, latency_ms: float):
        self.text = text
        self.model = model
        self.usage = usage
        self.latency_ms = latency_ms


class LLMClient:
    def __init__(self):
        api_key = settings.groq_api_key
        if not api_key:
            raise GenerationError("GROQ_API_KEY is not set")
        self.client = Groq(api_key=api_key)
        self.model = settings.llm_model_name
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens

    def generate(self, prompt: str) -> GenerationResult:
        import time
        start = time.monotonic()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            raise GenerationError(f"LLM call failed: {e}") from e

        elapsed = (time.monotonic() - start) * 1000
        choice = response.choices[0]
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        logger.info(
            "LLM generation complete",
            model=self.model,
            latency_ms=round(elapsed, 2),
            total_tokens=usage["total_tokens"],
        )

        return GenerationResult(
            text=choice.message.content or "",
            model=self.model,
            usage=usage,
            latency_ms=round(elapsed, 2),
        )
