from functools import lru_cache

from src.llm.base import LLMProvider
from src.config import settings


@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider:
    """Return the singleton LLM provider configured by LLM_PROVIDER env var.

    Imports are deferred so unused provider deps (boto3 / httpx) are only
    imported when actually needed.
    """
    if settings.llm_provider == "openrouter":
        from src.llm.openrouter import OpenRouterProvider
        return OpenRouterProvider()
    elif settings.llm_provider == "bedrock":
        from src.llm.bedrock import BedrockProvider
        return BedrockProvider()
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider!r}")
