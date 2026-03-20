from openai import AsyncOpenAI

from src.llm.base import LLMProvider, LLMResponse


class OpenAICompatibleProvider(LLMProvider):
    """Generic provider for any OpenAI-SDK-compatible endpoint (OpenRouter, OpenAI, Azure, etc.).

    All configuration is passed via constructor so the same class can be re-used
    for different backends by supplying different base_url / api_key / model values.
    """

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def complete(self, messages: list[dict]) -> LLMResponse:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1024,
            temperature=0.1,
        )
        return LLMResponse(
            text=response.choices[0].message.content,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )