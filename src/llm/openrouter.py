from src.llm.openai import OpenAICompatibleProvider
from src.config import settings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(OpenAICompatibleProvider):
    """LLM provider backed by OpenRouter using the OpenAI-compatible API."""

    def __init__(self) -> None:
        super().__init__(
            api_key=settings.openrouter.api_key,
            base_url=OPENROUTER_BASE_URL,
            model=settings.openrouter.llm_model,
        )
