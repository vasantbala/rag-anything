from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int


class LLMProvider(ABC):
    @abstractmethod
    async def complete(self, messages: list[dict]) -> LLMResponse:
        """
        messages: list of {"role": "system"|"user"|"assistant", "content": "..."}
        Returns LLMResponse with text and token counts.
        """
        pass
